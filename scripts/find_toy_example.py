#!/usr/bin/env python3
"""
Find and visualize a teacher/cache-level toy example for DPies:
"low-mass but decision-critical evidence is missed by mass-based retention".

This script does NOT need a trained model. It uses labels already written by
`dpies.data.preprocess_nuplan`:
  - oracle_action_index
  - teacher_cost
  - rival_label
  - signed_evidence_label        # s_i(a,b)=g_i(b)-g_i(a), positive favors a over b
  - signed_evidence_mask
  - evidence_features[:, 10]     # confidence/relevance proxy p_i, used here as mass proxy

Recommended usage:
python find_low_mass_critical_toy.py \
  --cache-dir /data0/senzeyu2/dataset/nuplan/data/cache/train_v4 \
  --out-dir ./toy_low_mass_critical \
  --workers 16 \
  --boundary-keep 8000 \
  --mass-budget 12

For a very quick smoke test, add: --max-files 20000 --boundary-keep 1000
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Headless plotting on servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EVIDENCE_TYPE = {
    0: "dynamic_agent",
    1: "conflict_point",
    2: "gap",
    3: "map_rule",
    4: "low_ttc_risk",
    5: "padding",
}


def _json_default(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def list_npz_files(cache_dir: Path, max_files: int | None = None) -> list[Path]:
    """Use manifest.jsonl when available; fall back to recursive glob."""
    manifest = cache_dir / "manifest.jsonl"
    files: list[Path] = []
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    rel = row.get("file")
                    if rel:
                        p = cache_dir / rel
                        if p.exists():
                            files.append(p)
                except Exception:
                    continue
                if max_files is not None and len(files) >= max_files:
                    return files
    if not files:
        it = cache_dir.rglob("*.npz")
        for p in it:
            files.append(p)
            if max_files is not None and len(files) >= max_files:
                break
    return files


def _load_mask(z: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key in z.files:
        return np.asarray(z[key]).astype(bool)
    return np.ones((n,), dtype=bool)


def cheap_boundary_record(path_str: str) -> dict[str, Any] | None:
    """Read only small arrays and return the closest oracle-vs-rival boundary."""
    path = Path(path_str)
    try:
        with np.load(path, allow_pickle=False) as z:
            if "teacher_cost" not in z.files or "oracle_action_index" not in z.files or "action_mask" not in z.files:
                return None
            cost = np.asarray(z["teacher_cost"], dtype=np.float32)
            action_mask = np.asarray(z["action_mask"]).astype(bool)
            if cost.ndim != 1 or not action_mask.any():
                return None
            oracle = int(np.asarray(z["oracle_action_index"]).item())
            if oracle < 0 or oracle >= len(cost) or not action_mask[oracle] or not np.isfinite(cost[oracle]):
                return None

            valid = action_mask & np.isfinite(cost)
            valid[oracle] = False
            if not valid.any():
                return None

            if "rival_label" in z.files:
                rival_label = np.asarray(z["rival_label"]).astype(bool)
                rivals = valid & rival_label[oracle]
                if not rivals.any():
                    rivals = valid
            else:
                rivals = valid

            rival_ids = np.flatnonzero(rivals)
            gaps = cost[rival_ids] - cost[oracle]  # positive means teacher oracle is better
            # Prefer positive, small margins. If numerical labels produce no positive margin, keep nearest abs margin.
            pos = gaps > 0
            if pos.any():
                j = int(np.argmin(gaps[pos]))
                r = int(rival_ids[pos][j])
                gap = float(gaps[pos][j])
            else:
                j = int(np.argmin(np.abs(gaps)))
                r = int(rival_ids[j])
                gap = float(gaps[j])

            return {"path": str(path), "teacher_gap": gap, "oracle": oracle, "rival": r}
    except Exception as e:
        return {"path": str(path), "error": repr(e)}


def keep_smallest_boundary(records: Iterable[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Keep records with smallest positive teacher gaps; robust to negative/zero gaps."""
    heap: list[tuple[float, int, dict[str, Any]]] = []
    counter = 0
    for rec in records:
        if rec is None or rec.get("error"):
            continue
        gap = float(rec.get("teacher_gap", math.inf))
        if not np.isfinite(gap):
            continue
        # Sort key: positive small gaps first; then absolute small gaps.
        key = gap if gap > 0 else abs(gap) + 1e6
        item = (-key, counter, rec)
        counter += 1
        if len(heap) < keep:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    out = [x[2] for x in heap]
    out.sort(key=lambda r: (float(r["teacher_gap"]) <= 0, abs(float(r["teacher_gap"]))))
    return out


def analyze_boundary_candidate(
    rec: dict[str, Any],
    mass_budget: int,
    low_mass_quantile: float,
    min_signed_contrib: float,
    prefer_budget_flip: bool = True,
) -> dict[str, Any] | None:
    """Find a low-mass evidence atom that a mass-top-B selector misses but pairwise signed labels value."""
    path = Path(rec["path"])
    try:
        with np.load(path, allow_pickle=False) as z:
            needed = ["signed_evidence_label", "evidence_features", "evidence_mask", "action_mask", "teacher_cost", "oracle_action_index"]
            if any(k not in z.files for k in needed):
                return None
            signed = np.asarray(z["signed_evidence_label"], dtype=np.float32)  # [N,K,K]
            ev = np.asarray(z["evidence_features"], dtype=np.float32)
            e_mask = np.asarray(z["evidence_mask"]).astype(bool)
            a_mask = np.asarray(z["action_mask"]).astype(bool)
            cost = np.asarray(z["teacher_cost"], dtype=np.float32)
            oracle = int(np.asarray(z["oracle_action_index"]).item())
            if signed.ndim != 3 or oracle < 0 or oracle >= signed.shape[1] or not a_mask[oracle]:
                return None
            n = min(signed.shape[0], ev.shape[0], e_mask.shape[0])
            if n == 0:
                return None
            e_mask = e_mask[:n]
            ev = ev[:n]
            signed = signed[:n]
            mass = np.clip(ev[:, 10].astype(np.float32), 0.0, np.inf) if ev.shape[1] > 10 else np.ones(n, dtype=np.float32)
            valid_e = e_mask & np.isfinite(mass)
            if valid_e.sum() < max(3, mass_budget + 1):
                return None

            # Mass-only baseline: retain top-B by p_i / cost_i if evidence_cost exists, otherwise p_i.
            if "evidence_cost" in z.files:
                ecost = np.asarray(z["evidence_cost"], dtype=np.float32)[:n]
                mass_score = mass / np.maximum(ecost, 1e-6)
            else:
                mass_score = mass
            mass_order = np.argsort(-np.where(valid_e, mass_score, -np.inf))
            top_mass = np.zeros(n, dtype=bool)
            top_mass[mass_order[: min(mass_budget, int(valid_e.sum()))]] = True

            if "rival_label" in z.files:
                rival_label = np.asarray(z["rival_label"]).astype(bool)
                rivals = a_mask.copy()
                rivals[oracle] = False
                if rival_label.ndim == 2 and rival_label.shape[0] > oracle:
                    rivals &= rival_label[oracle]
                if not rivals.any():
                    rivals = a_mask.copy(); rivals[oracle] = False
            else:
                rivals = a_mask.copy(); rivals[oracle] = False
            rival_ids = np.flatnonzero(rivals)
            if rival_ids.size == 0:
                return None

            if "signed_evidence_mask" in z.files:
                active3 = np.asarray(z["signed_evidence_mask"]).astype(bool)[:n]
            else:
                active3 = np.ones_like(signed, dtype=bool)

            low_thr = float(np.quantile(mass[valid_e], low_mass_quantile))
            best: dict[str, Any] | None = None

            for r in rival_ids:
                if r >= signed.shape[2] or not np.isfinite(cost[r]):
                    continue
                s = signed[:, oracle, r].astype(np.float32)
                active = valid_e & active3[:, oracle, r] & np.isfinite(s)
                if not active.any():
                    continue
                full_margin = float(np.sum(np.where(active, s, 0.0)))
                mass_margin = float(np.sum(np.where(active & top_mass, s, 0.0)))
                # Candidate must favor oracle, be low-mass, and be missed by mass-top-B.
                cand = active & (s >= min_signed_contrib) & (mass <= low_thr) & (~top_mass)
                if not cand.any():
                    continue
                # Criticality score emphasizes cases where adding this atom repairs a mass-budget failure.
                ids = np.flatnonzero(cand)
                for i in ids:
                    si = float(s[i])
                    without_i_full = full_margin - si
                    mass_plus_i = mass_margin + si
                    omission_flip = (full_margin > 0.0 and without_i_full <= 0.0)
                    budget_flip = (mass_margin <= 0.0 and mass_plus_i > 0.0)
                    # Larger is better. Tuple fields are intentionally interpretable.
                    score_tuple = (
                        1 if (budget_flip and prefer_budget_flip) else 0,
                        1 if omission_flip else 0,
                        si,
                        float(low_thr - mass[i]),
                        -abs(float(cost[r] - cost[oracle])),
                    )
                    typ = int(z["evidence_type"][i]) if "evidence_type" in z.files and i < len(z["evidence_type"]) else -1
                    cur = {
                        "path": str(path),
                        "oracle": int(oracle),
                        "rival": int(r),
                        "critical_evidence": int(i),
                        "evidence_type": typ,
                        "evidence_type_name": EVIDENCE_TYPE.get(typ, str(typ)),
                        "evidence_xy": [float(ev[i, 1]), float(ev[i, 2])] if ev.shape[1] > 2 else [0.0, 0.0],
                        "evidence_mass_proxy": float(mass[i]),
                        "low_mass_threshold": low_thr,
                        "signed_contribution_oracle_vs_rival": si,
                        "full_local_pair_margin": full_margin,
                        "full_margin_without_critical": without_i_full,
                        "mass_top_b_pair_margin": mass_margin,
                        "mass_top_b_plus_critical_margin": mass_plus_i,
                        "budget_flip": bool(budget_flip),
                        "omission_flip": bool(omission_flip),
                        "teacher_cost_oracle": float(cost[oracle]),
                        "teacher_cost_rival": float(cost[r]),
                        "teacher_gap_rival_minus_oracle": float(cost[r] - cost[oracle]),
                        "mass_budget": int(mass_budget),
                        "low_mass_quantile": float(low_mass_quantile),
                        "score_tuple": score_tuple,
                    }
                    if best is None or cur["score_tuple"] > best["score_tuple"]:
                        best = cur
            return best
    except Exception as e:
        return {"path": str(path), "error": repr(e)}


def _plot_traj(ax, traj: np.ndarray, *, lw: float, alpha: float, label: str | None = None, marker: str | None = None):
    arr = np.asarray(traj, dtype=float)
    if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
        ax.plot(arr[:, 0], arr[:, 1], linewidth=lw, alpha=alpha, label=label, marker=marker, markersize=3)


def visualize_toy(best: dict[str, Any], out_png: Path, top_evidence: int = 20) -> None:
    path = Path(best["path"])
    with np.load(path, allow_pickle=False) as z:
        actions = np.asarray(z["actions"], dtype=float)
        a_mask = np.asarray(z["action_mask"]).astype(bool)
        ev = np.asarray(z["evidence_features"], dtype=float)
        e_mask = np.asarray(z["evidence_mask"]).astype(bool)
        signed = np.asarray(z["signed_evidence_label"], dtype=float)
        oracle = int(best["oracle"])
        rival = int(best["rival"])
        crit = int(best["critical_evidence"])
        mass = np.clip(ev[:, 10], 0.0, np.inf) if ev.shape[1] > 10 else np.ones(ev.shape[0])
        s_pair = signed[:, oracle, rival] if signed.ndim == 3 else np.zeros(ev.shape[0])
        action_mask_ids = np.flatnonzero(a_mask)

        fig, ax = plt.subplots(figsize=(13, 9))

        if "map_polylines" in z.files and "map_masks" in z.files:
            maps = np.asarray(z["map_polylines"], dtype=float)
            mm = np.asarray(z["map_masks"]).astype(bool)
            for pts, m in zip(maps, mm):
                pts = pts[m]
                if pts.shape[0] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], linewidth=0.6, alpha=0.25, zorder=0)

        if "agent_history" in z.files:
            ah = np.asarray(z["agent_history"], dtype=float)
            if "agent_history_mask" in z.files:
                hm = np.asarray(z["agent_history_mask"]).astype(bool)
            else:
                hm = np.ones(ah.shape[:2], dtype=bool)
            for h, m in zip(ah, hm):
                pts = h[m]
                if pts.shape[0] >= 1:
                    ax.plot(pts[:, 0], pts[:, 1], linewidth=0.8, alpha=0.35, zorder=1)
                    ax.scatter(pts[-1, 0], pts[-1, 1], s=15, alpha=0.45, zorder=2)

        if "logged_ego_future" in z.files:
            _plot_traj(ax, np.asarray(z["logged_ego_future"], dtype=float), lw=3.0, alpha=0.85, label="logged ego future", marker="o")

        for a in action_mask_ids:
            if a == oracle or a == rival:
                continue
            _plot_traj(ax, actions[a], lw=1.0, alpha=0.16)
        if 0 <= rival < len(actions):
            _plot_traj(ax, actions[rival], lw=3.0, alpha=0.95, label=f"rival action a{rival}", marker="x")
        if 0 <= oracle < len(actions):
            _plot_traj(ax, actions[oracle], lw=3.5, alpha=0.98, label=f"oracle action a{oracle}", marker="o")

        valid = e_mask & np.isfinite(s_pair)
        top_ids = np.argsort(-np.where(valid, np.abs(s_pair), -np.inf))[:top_evidence]
        if valid.any():
            ax.scatter(ev[valid, 1], ev[valid, 2], s=18, marker=".", alpha=0.22, label="all evidence", zorder=3)
        top_ids = [int(i) for i in top_ids if i < len(ev) and valid[i] and i != crit]
        if top_ids:
            sizes = 40 + 18 * np.sqrt(np.maximum(np.abs(s_pair[top_ids]), 0.0))
            ax.scatter(ev[top_ids, 1], ev[top_ids, 2], s=sizes, marker="s", alpha=0.55, label="high |signed evidence|", zorder=4)
        if 0 <= crit < len(ev):
            ax.scatter([ev[crit, 1]], [ev[crit, 2]], s=260, marker="*", edgecolors="black", linewidths=1.0,
                       label="LOW-MASS CRITICAL evidence", zorder=6)
            ax.annotate(
                f"critical e{crit}\n{best['evidence_type_name']}\np={best['evidence_mass_proxy']:.3g}, s={best['signed_contribution_oracle_vs_rival']:.2f}",
                xy=(ev[crit, 1], ev[crit, 2]), xytext=(ev[crit, 1] + 4, ev[crit, 2] + 7),
                arrowprops=dict(arrowstyle="->", lw=1.2), fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.85), zorder=7,
            )

        title = (
            "Low-mass but decision-critical evidence toy example\n"
            f"{path.name} | oracle a{oracle} vs rival a{rival} | "
            f"teacher gap={best['teacher_gap_rival_minus_oracle']:.3f}"
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("ego-frame x [m]")
        ax.set_ylabel("ego-frame y [m]")
        ax.axis("equal")
        ax.grid(True, linewidth=0.4, alpha=0.25)

        # Center around important content.
        xy_parts = []
        if actions.ndim == 3 and action_mask_ids.size:
            xy_parts.append(actions[action_mask_ids, :, :2].reshape(-1, 2))
        if e_mask.any() and ev.shape[1] > 2:
            xy_parts.append(ev[e_mask, 1:3])
        if xy_parts:
            pts = np.concatenate(xy_parts, axis=0)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts):
                lo = np.percentile(pts, 2, axis=0) - 8
                hi = np.percentile(pts, 98, axis=0) + 8
                ax.set_xlim(max(lo[0], -70), min(hi[0], 120))
                ax.set_ylim(max(lo[1], -70), min(hi[1], 70))

        info = (
            f"Mass-only top-{best['mass_budget']} pair margin: {best['mass_top_b_pair_margin']:.2f}\n"
            f"After adding critical evidence: {best['mass_top_b_plus_critical_margin']:.2f}\n"
            f"Full local pair margin: {best['full_local_pair_margin']:.2f}\n"
            f"Without critical evidence: {best['full_margin_without_critical']:.2f}\n"
            f"low-mass threshold q={best['low_mass_quantile']:.2f}: p <= {best['low_mass_threshold']:.3g}\n"
            f"budget_flip={best['budget_flip']}  omission_flip={best['omission_flip']}"
        )
        ax.text(0.012, 0.988, info, transform=ax.transAxes, va="top", ha="left", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.4", alpha=0.9))
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=180)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--max-files", type=int, default=None, help="optional cap for smoke tests")
    ap.add_argument("--boundary-keep", type=int, default=8000, help="number of near-boundary scenes to inspect deeply")
    ap.add_argument("--mass-budget", type=int, default=12, help="simulated mass-only evidence budget B")
    ap.add_argument("--low-mass-quantile", type=float, default=0.35)
    ap.add_argument("--min-signed-contrib", type=float, default=0.5)
    ap.add_argument("--top-evidence", type=int, default=20)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = list_npz_files(args.cache_dir, args.max_files)
    if not files:
        raise FileNotFoundError(f"No npz files found under {args.cache_dir}")
    print(f"[1/4] discovered {len(files)} npz files")

    # Stage 1: cheap boundary scan over all files.
    records: list[dict[str, Any]] = []
    if args.workers <= 1:
        for idx, p in enumerate(files, 1):
            r = cheap_boundary_record(str(p))
            if r and not r.get("error"):
                records.append(r)
            if idx % 10000 == 0:
                print(f"  scanned {idx}/{len(files)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(cheap_boundary_record, str(p)) for p in files]
            for idx, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r and not r.get("error"):
                    records.append(r)
                if idx % 10000 == 0:
                    print(f"  scanned {idx}/{len(files)}")
    boundary = keep_smallest_boundary(records, args.boundary_keep)
    (args.out_dir / "boundary_candidates.json").write_text(json.dumps(boundary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"[2/4] kept {len(boundary)} near-boundary candidates")

    # Stage 2: deep inspect only boundary scenes, loading the large signed_evidence_label array.
    best: dict[str, Any] | None = None
    if args.workers <= 1:
        iterator = (analyze_boundary_candidate(r, args.mass_budget, args.low_mass_quantile, args.min_signed_contrib) for r in boundary)
        for idx, cand in enumerate(iterator, 1):
            if cand and not cand.get("error") and (best is None or cand["score_tuple"] > best["score_tuple"]):
                best = cand
            if idx % 500 == 0:
                print(f"  deep-inspected {idx}/{len(boundary)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(analyze_boundary_candidate, r, args.mass_budget, args.low_mass_quantile, args.min_signed_contrib) for r in boundary]
            for idx, fut in enumerate(as_completed(futs), 1):
                cand = fut.result()
                if cand and not cand.get("error") and (best is None or cand["score_tuple"] > best["score_tuple"]):
                    best = cand
                if idx % 500 == 0:
                    print(f"  deep-inspected {idx}/{len(boundary)}")
    if best is None:
        raise RuntimeError(
            "No low-mass critical example found. Try increasing --boundary-keep, lowering --min-signed-contrib, "
            "or increasing --mass-budget/--max-files."
        )

    # Tuples are not JSON-friendly and not needed for reports.
    best = dict(best)
    best.pop("score_tuple", None)
    best_json = args.out_dir / "best_low_mass_critical.json"
    best_json.write_text(json.dumps(best, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(f"[3/4] best candidate written to {best_json}")
    print(json.dumps(best, indent=2, ensure_ascii=False, default=_json_default))

    out_png = args.out_dir / "low_mass_critical_toy.png"
    visualize_toy(best, out_png, top_evidence=args.top_evidence)
    print(f"[4/4] visualization written to {out_png}")


if __name__ == "__main__":
    main()