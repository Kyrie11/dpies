from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dpies.teacher.teacher_evaluator import TeacherWeights

W = TeacherWeights()
COMPONENT_WEIGHTS = {
    "ade": W.imitation_ade,
    "fde": W.imitation_fde,
    "progress_norm": W.route_progress,
    "speed_over": W.speed_limit,
    "accel_cost": W.comfort_accel,
    "jerk_cost": W.comfort_jerk,
    "curv_cost": W.comfort_curvature,
    "future_collision": W.future_collision,
    "future_proximity": W.future_proximity,
    "local_sum": W.local_evidence,
    "rel_low_progress": W.low_progress,
    "abs_low_progress": W.absolute_progress_weight,
    "speed_floor": W.speed_floor_weight,
    "target_speed": W.target_speed_weight,
    "excessive_progress": W.excessive_progress,
}
COMPONENT_NAMES = [
    "ade", "fde", "progress_norm", "speed_over",
    "accel_cost", "jerk_cost", "curv_cost",
    "future_collision", "future_proximity", "local_sum",
    "rel_low_progress", "abs_low_progress", "speed_floor", "target_speed", "excessive_progress",
    "move_gate", "global_without_local", "total",
]


def files_from_manifest(root: Path) -> list[Path]:
    manifest = root / "manifest.jsonl"
    out: list[Path] = []
    if not manifest.exists():
        return out
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rel = row.get("file", "")
                if rel:
                    p = root / rel
                    if p.exists():
                        out.append(p)
            except Exception:
                continue
    return out


def list_npz_files(root: Path, use_manifest: bool) -> list[Path]:
    files = files_from_manifest(root) if use_manifest else []
    if not files:
        files = list(root.rglob("*.npz"))
    return [p for p in files if p.suffix == ".npz" and ".tmp." not in p.name and not p.name.startswith(".")]


def select_files(files: list[Path], sample: int, max_files: int, stride: int, seed: int) -> list[Path]:
    if stride > 1:
        files = files[::stride]
    if sample and sample > 0 and sample < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, sample)
    if max_files and max_files > 0:
        files = files[:max_files]
    return files


def analyze_one(arg: tuple[str, bool]) -> tuple[dict[str, Any] | None, tuple[str, str] | None]:
    path_str, read_components = arg
    p = Path(path_str)
    try:
        with np.load(p, allow_pickle=False) as d:
            required = ["actions", "action_mask", "teacher_cost"]
            missing = [k for k in required if k not in d]
            if missing:
                return None, (path_str, f"missing {missing}")

            actions = d["actions"]
            action_mask = d["action_mask"].astype(bool)
            teacher_cost = d["teacher_cost"].astype(np.float32)

            if actions.ndim != 3 or actions.shape[-1] < 4:
                return None, (path_str, f"bad actions shape {actions.shape}")
            valid = np.where(action_mask)[0]
            if len(valid) == 0:
                return None, (path_str, "no valid action")

            if "oracle_action_index" in d:
                oracle = int(np.asarray(d["oracle_action_index"]).reshape(-1)[0])
            else:
                tmp = teacher_cost.copy()
                tmp[~action_mask] = np.inf
                oracle = int(np.argmin(tmp))
            if oracle < 0 or oracle >= len(action_mask) or not action_mask[oracle]:
                tmp = teacher_cost.copy()
                tmp[~action_mask] = np.inf
                oracle = int(np.argmin(tmp))

            progress = actions[:, -1, 0].astype(np.float32)
            final_speed = actions[:, -1, 3].astype(np.float32)
            max_progress_idx = int(valid[np.argmax(progress[valid])])
            min_cost_idx = int(valid[np.argmin(teacher_cost[valid])])

            row: dict[str, Any] = {
                "file": path_str,
                "valid_action_count": int(action_mask.sum()),
                "oracle": oracle,
                "min_cost_idx": min_cost_idx,
                "oracle_progress": float(progress[oracle]),
                "oracle_final_speed": float(final_speed[oracle]),
                "oracle_teacher_cost": float(teacher_cost[oracle]),
                "max_progress": float(progress[max_progress_idx]),
                "max_progress_final_speed": float(final_speed[max_progress_idx]),
                "max_progress_teacher_cost": float(teacher_cost[max_progress_idx]),
                "teacher_cost_min": float(np.min(teacher_cost[valid])),
                "teacher_cost_p50": float(np.percentile(teacher_cost[valid], 50)),
                "teacher_cost_max": float(np.max(teacher_cost[valid])),
                "oracle_is_near_stop": float(final_speed[oracle] < 0.5),
                "oracle_low_progress_8m": float(progress[oracle] < 8.0),
                "oracle_low_progress_20m": float(progress[oracle] < 20.0),
                "max_progress_minus_oracle": float(progress[max_progress_idx] - progress[oracle]),
                "cost_max_progress_minus_oracle": float(teacher_cost[max_progress_idx] - teacher_cost[oracle]),
            }

            if "action_meta" in d:
                meta = d["action_meta"]
                if meta.ndim == 2 and meta.shape[0] > oracle:
                    row["oracle_mode"] = int(meta[oracle, 0])
                    row["max_progress_mode"] = int(meta[max_progress_idx, 0])

            if "rival_label" in d:
                rival = d["rival_label"].astype(bool)
                row["rival_label_count"] = int(rival.sum())
                row["rival_label_density"] = float(rival.sum() / max(1, action_mask.sum() * (action_mask.sum() - 1)))
                row["oracle_rival_out"] = int(rival[oracle].sum())
                row["oracle_rival_in"] = int(rival[:, oracle].sum())

            if "signed_evidence_mask" in d:
                sem = d["signed_evidence_mask"].astype(bool)
                row["signed_evidence_active_count"] = int(sem.sum())
                row["signed_evidence_active_density"] = float(sem.mean())

            if "evidence_mask" in d:
                em = d["evidence_mask"].astype(bool)
                row["evidence_count"] = int(em.sum())

            if read_components and "teacher_components" in d:
                comp = d["teacher_components"]
                if comp.ndim == 2 and comp.shape[0] > oracle and comp.shape[0] > max_progress_idx:
                    for j in range(min(comp.shape[1], len(COMPONENT_NAMES))):
                        name = COMPONENT_NAMES[j]
                        row[f"oracle_comp_{name}"] = float(comp[oracle, j])
                        row[f"maxprog_comp_{name}"] = float(comp[max_progress_idx, j])
                        if name in COMPONENT_WEIGHTS:
                            wt = float(COMPONENT_WEIGHTS[name])
                            row[f"oracle_wcomp_{name}"] = wt * float(comp[oracle, j])
                            row[f"maxprog_wcomp_{name}"] = wt * float(comp[max_progress_idx, j])
                            row[f"delta_wcomp_{name}"] = wt * (float(comp[max_progress_idx, j]) - float(comp[oracle, j]))

            return row, None
    except Exception as e:
        return None, (path_str, repr(e))


def describe(df: pd.DataFrame, cols: list[str]) -> None:
    cols = [c for c in cols if c in df.columns]
    if cols:
        print(df[cols].describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99]).T.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast dataset oracle diagnostics")
    ap.add_argument("--cache-dir", "--CACHE_DIR", dest="cache_dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--sample", type=int, default=100000, help="Random sample size. Use 0 for all files.")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--components", action="store_true", help="Read teacher_components and weighted component columns")
    ap.add_argument("--no-manifest", action="store_true")
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--out", type=str, default="runs/dataset_oracle_diagnostics.csv")
    ap.add_argument("--bad-out", type=str, default="runs/dataset_oracle_diagnostics_bad.txt")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    if not cache.exists():
        raise FileNotFoundError(f"CACHE_DIR not found: {cache}")

    all_files = list_npz_files(cache, use_manifest=not args.no_manifest)
    files = select_files(all_files, args.sample, args.max_files, args.stride, args.seed)
    print("CACHE_DIR:", cache)
    print("num npz total:", len(all_files))
    print("num npz selected:", len(files))
    print("workers:", args.workers, "components:", args.components)

    rows: list[dict[str, Any]] = []
    bad: list[tuple[str, str]] = []
    start = time.time()
    work = ((str(p), bool(args.components)) for p in files)
    if args.workers <= 1:
        iterator = map(analyze_one, work)
    else:
        pool = Pool(processes=args.workers)
        iterator = pool.imap_unordered(analyze_one, work, chunksize=32)

    try:
        for i, (row, err) in enumerate(iterator, 1):
            if row is not None:
                rows.append(row)
            if err is not None:
                bad.append(err)
            if i % max(1, args.progress_every) == 0 or i == len(files):
                elapsed = max(time.time() - start, 1e-6)
                rate = i / elapsed
                eta = (len(files) - i) / max(rate, 1e-6)
                print(f"progress {i}/{len(files)} good={len(rows)} bad={len(bad)} rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)
    finally:
        if args.workers > 1:
            pool.close()
            pool.join()

    df = pd.DataFrame(rows)
    print("\nGOOD FILES:", len(df))
    print("BAD FILES:", len(bad))
    if bad[:10]:
        print("BAD EXAMPLES:")
        for x in bad[:10]:
            print(x)

    bad_out = Path(args.bad_out)
    bad_out.parent.mkdir(parents=True, exist_ok=True)
    bad_out.write_text("\n".join(f"{p}\t{e}" for p, e in bad), encoding="utf-8")

    if len(df) == 0:
        print("No valid rows. Likely wrong CACHE_DIR or incompatible npz schema.")
        return

    print("\nORACLE / ACTION COVERAGE")
    describe(df, [
        "valid_action_count", "oracle_progress", "oracle_final_speed",
        "max_progress", "max_progress_final_speed", "max_progress_minus_oracle",
        "cost_max_progress_minus_oracle", "oracle_teacher_cost", "max_progress_teacher_cost",
        "teacher_cost_min", "teacher_cost_p50", "teacher_cost_max",
    ])

    print("\nORACLE STOP RATIOS")
    for c in ["oracle_is_near_stop", "oracle_low_progress_8m", "oracle_low_progress_20m"]:
        if c in df.columns:
            print(c, "mean=", df[c].mean(), "sum=", int(df[c].sum()), "/", len(df))

    if "oracle_mode" in df.columns:
        print("\nORACLE MODE COUNTS")
        print(df["oracle_mode"].value_counts(dropna=False).sort_index())
    if "max_progress_mode" in df.columns:
        print("\nMAX PROGRESS MODE COUNTS")
        print(df["max_progress_mode"].value_counts(dropna=False).sort_index())

    print("\nRIVAL / EVIDENCE")
    describe(df, [
        "rival_label_count", "rival_label_density", "oracle_rival_out", "oracle_rival_in",
        "signed_evidence_active_count", "signed_evidence_active_density", "evidence_count",
    ])

    if args.components:
        comp_cols = [c for c in df.columns if c.startswith("oracle_comp_")]
        if comp_cols:
            print("\nORACLE TEACHER COMPONENTS")
            describe(df, comp_cols)
        wcomp_cols = [c for c in df.columns if c.startswith("oracle_wcomp_")]
        if wcomp_cols:
            print("\nORACLE WEIGHTED TEACHER COMPONENT CONTRIBUTIONS")
            describe(df, wcomp_cols)
        delta_cols = [c for c in df.columns if c.startswith("delta_wcomp_")]
        if delta_cols:
            print("\nMAX-PROGRESS MINUS ORACLE WEIGHTED CONTRIBUTION DELTAS")
            describe(df, delta_cols)

    sus = df[(df["oracle_final_speed"] < 0.5) & (df["max_progress"] >= 20.0)].copy()
    print("\nSUSPECT: oracle near stop while max_progress>=20:", len(sus), "/", len(df))
    if len(sus):
        show_cols = [
            "file", "oracle", "oracle_mode", "oracle_progress", "oracle_final_speed",
            "max_progress", "max_progress_final_speed", "max_progress_minus_oracle",
            "cost_max_progress_minus_oracle", "oracle_teacher_cost", "max_progress_teacher_cost",
        ]
        show_cols = [c for c in show_cols if c in sus.columns]
        print(sus.sort_values("max_progress_minus_oracle", ascending=False)[show_cols].head(30).to_string(index=False))

    print("\nTEACHER SANITY TARGETS")
    targets = {
        "oracle_progress_mean_gt_30": float(df["oracle_progress"].mean() > 30.0),
        "oracle_progress_median_gt_20": float(df["oracle_progress"].median() > 20.0),
        "oracle_low_progress_20m_lt_0.35": float(df["oracle_low_progress_20m"].mean() < 0.35),
        "oracle_is_near_stop_lt_0.10": float(df["oracle_is_near_stop"].mean() < 0.10),
        "max_progress_minus_oracle_mean_lt_30": float(df["max_progress_minus_oracle"].mean() < 30.0),
    }
    if "oracle_rival_out" in df.columns:
        targets["oracle_rival_out_mean_gt_20"] = float(df["oracle_rival_out"].mean() > 20.0)
    for k, v in targets.items():
        print(f"{k}: {'PASS' if v else 'FAIL'}")
    print("overall:", "PASS" if all(bool(v) for v in targets.values()) else "FAIL")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\nsaved:", out.resolve())
    print("bad saved:", bad_out.resolve())


if __name__ == "__main__":
    main()