#!/usr/bin/env python3
"""
Rich debug visualizer for DPIES nuPlan NPZ cache files.

It checks and visualizes the pieces that matter for training:
  - ego-centric coordinate consistency
  - HD map polylines and exact map-rule geometries from evidence metadata
  - route/drivable/crosswalk/stop-line/traffic-light/speed-limit evidence
  - agent current/future tracks and masks
  - candidate actions, oracle action, logged ego future, teacher costs
  - evidence type distribution, signed-evidence activity, geometry-query sanity

Usage:
  python debug_visualize_npz.py /path/to/sample_000000123.npz --out debug.png --summary debug.json
  python debug_visualize_npz.py /cache/dir --out-dir debug_viz --limit 20
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

EVIDENCE_TYPE = {
    0: "dynamic_agent",
    1: "conflict_point",
    2: "gap",
    3: "map_rule",
    4: "low_ttc_risk",
    5: "padding",
}
ACTION_MODE = {
    0: "keep",
    1: "stop",
    2: "proceed",
    3: "lane_change_left",
    4: "lane_change_right",
    5: "merge",
    6: "nudge_left",
    7: "nudge_right",
    8: "creep",
}
RULE_CODE = {
    0: "none",
    1: "stop_line",
    2: "crosswalk",
    3: "lane_boundary",
    4: "traffic_light_red",
    5: "drivable_area",
    6: "speed_limit",
    7: "route_deviation",
    8: "intersection",
    9: "lane_connector",
}


def as_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        value = str(value)
    try:
        return json.loads(value)
    except Exception:
        return default


def get_mask(z: np.lib.npyio.NpzFile, key: str, fallback_key: str | None = None, length: int | None = None) -> np.ndarray:
    if key in z.files:
        return np.asarray(z[key]).astype(bool)
    if fallback_key and fallback_key in z.files:
        return np.asarray(z[fallback_key]).astype(bool)
    if length is None:
        return np.zeros((0,), dtype=bool)
    return np.ones((length,), dtype=bool)


def draw_box(ax, x: float, y: float, yaw: float, length: float, width: float, *, label: str = "", alpha: float = 0.35, lw: float = 1.0):
    # local box corners centered at (x, y), yaw in ego frame
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = length / 2.0, width / 2.0
    pts = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=float)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = pts @ rot.T + np.array([x, y])
    patch = Polygon(pts, closed=True, fill=False, alpha=alpha, linewidth=lw)
    ax.add_patch(patch)
    if label:
        ax.text(x, y, label, fontsize=5, alpha=0.75)


def plot_poly(ax, pts: Any, *, closed: bool = False, lw: float = 1.0, alpha: float = 0.7, linestyle: str = "-", marker: str | None = None):
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 2:
        return
    ax.plot(arr[:, 0], arr[:, 1], linewidth=lw, alpha=alpha, linestyle=linestyle, marker=marker, markersize=2)
    if closed and arr.shape[0] >= 3:
        patch = Polygon(arr[:, :2], closed=True, fill=False, alpha=alpha * 0.5, linewidth=max(0.5, lw * 0.8))
        ax.add_patch(patch)


def evidence_importance(z: np.lib.npyio.NpzFile) -> np.ndarray:
    if "signed_evidence_label" not in z.files:
        return np.zeros((0,), dtype=float)
    signed = np.asarray(z["signed_evidence_label"], dtype=float)
    if signed.ndim != 3:
        return np.zeros((signed.shape[0],), dtype=float)
    if "signed_evidence_mask" in z.files:
        mask = np.asarray(z["signed_evidence_mask"]).astype(bool)
        vals = np.where(mask, np.abs(signed), 0.0)
        return vals.reshape(vals.shape[0], -1).max(axis=1)
    return np.abs(signed).reshape(signed.shape[0], -1).max(axis=1)


def summarize_sample(path: Path, z: np.lib.npyio.NpzFile) -> dict[str, Any]:
    meta = as_json(z["metadata_json"], {}) if "metadata_json" in z.files else {}
    ev_meta = as_json(z["evidence_metadata_json"], []) if "evidence_metadata_json" in z.files else []
    route_info = as_json(z["route_info_json"], {}) if "route_info_json" in z.files else {}

    action_mask = get_mask(z, "action_mask", "action_valid_mask", z["actions"].shape[0] if "actions" in z.files else 0)
    evidence_mask = get_mask(z, "evidence_mask", length=z["evidence_features"].shape[0] if "evidence_features" in z.files else 0)
    e_types = np.asarray(z["evidence_type"]) if "evidence_type" in z.files else np.zeros((0,), dtype=int)
    valid_types = [EVIDENCE_TYPE.get(int(t), str(int(t))) for t in e_types[evidence_mask]] if len(e_types) else []
    layer_counts = Counter(str(m.get("layer", "")) for m in ev_meta if isinstance(m, dict))
    rule_counts = Counter(RULE_CODE.get(int(m.get("rule_code", -1)), str(m.get("rule_code", ""))) for m in ev_meta if isinstance(m, dict) and "rule_code" in m)
    teacher = np.asarray(z["teacher_cost"], dtype=float) if "teacher_cost" in z.files else np.zeros((0,), dtype=float)
    signed = np.asarray(z["signed_evidence_label"], dtype=float) if "signed_evidence_label" in z.files else np.zeros((0, 0, 0), dtype=float)
    q = np.asarray(z["geometry_query"], dtype=float) if "geometry_query" in z.files else np.zeros((0, 0, 0), dtype=float)
    rival = np.asarray(z["rival_label"]) if "rival_label" in z.files else np.zeros((0, 0), dtype=bool)

    return {
        "file": str(path),
        "scenario_id": meta.get("scenario_id", ""),
        "map_name": meta.get("map_name", ""),
        "timestamp_us": meta.get("timestamp_us", None),
        "map_success": bool(meta.get("map_success", False)),
        "map_error": meta.get("map_error", ""),
        "valid_action_count": int(action_mask.sum()),
        "valid_evidence_count": int(evidence_mask.sum()),
        "evidence_type_counts": dict(Counter(valid_types)),
        "map_layer_counts": dict(layer_counts.most_common(30)),
        "rule_code_counts": dict(rule_counts),
        "route_roadblock_count": int(len(route_info.get("route_roadblock_ids", []))),
        "route_polygons": int(len(route_info.get("route_polygons", []))),
        "route_polylines": int(len(route_info.get("route_polylines", []))),
        "teacher_cost_finite": bool(np.isfinite(teacher).all()) if teacher.size else False,
        "teacher_cost_min": float(np.nanmin(teacher[action_mask])) if teacher.size and action_mask.any() else None,
        "teacher_cost_max": float(np.nanmax(teacher[action_mask])) if teacher.size and action_mask.any() else None,
        "geometry_query_finite": bool(np.isfinite(q).all()) if q.size else False,
        "geometry_query_valid_ratio": float(np.mean(q[..., 23] > 0.5)) if q.ndim == 3 and q.shape[-1] > 23 else None,
        "signed_evidence_finite": bool(np.isfinite(signed).all()) if signed.size else False,
        "signed_nonzero_ratio": float(np.mean(np.abs(signed) > 1e-6)) if signed.size else None,
        "rival_positive_ratio": float(np.mean(rival)) if rival.size else None,
        "oracle_action_index": int(np.asarray(z["oracle_action_index"]).item()) if "oracle_action_index" in z.files else None,
        "min_ade": meta.get("min_ade", None),
        "min_fde": meta.get("min_fde", None),
    }


def visualize_one(
    path: Path,
    out: Path | None = None,
    summary_out: Path | None = None,
    xlim=(-40, 90),
    ylim=(-55, 55),
    top_evidence: int = 25,
    skip_drivable_union: bool = True,
    max_exact_polygons: int = 20,
    max_polygon_points: int = 80,
    save_dpi: int = 120,
):
    with np.load(path, allow_pickle=True) as z:
        meta = as_json(z["metadata_json"], {}) if "metadata_json" in z.files else {}
        ev_meta = as_json(z["evidence_metadata_json"], []) if "evidence_metadata_json" in z.files else []
        route_info = as_json(z["route_info_json"], {}) if "route_info_json" in z.files else {}
        summary = summarize_sample(path, z)

        actions = np.asarray(z["actions"], dtype=float)
        action_mask = get_mask(z, "action_mask", "action_valid_mask", actions.shape[0])
        action_meta = np.asarray(z["action_meta"], dtype=float) if "action_meta" in z.files else np.zeros((actions.shape[0], 8))
        oracle = int(np.asarray(z["oracle_action_index"]).item()) if "oracle_action_index" in z.files else -1
        teacher = np.asarray(z["teacher_cost"], dtype=float) if "teacher_cost" in z.files else np.zeros((actions.shape[0],), dtype=float)
        logged = np.asarray(z["logged_ego_future"], dtype=float) if "logged_ego_future" in z.files else np.zeros((0, 2), dtype=float)

        map_polys = np.asarray(z["map_polylines"], dtype=float) if "map_polylines" in z.files else np.zeros((0, 0, 4), dtype=float)
        map_masks = np.asarray(z["map_masks"]).astype(bool) if "map_masks" in z.files else np.zeros(map_polys.shape[:2], dtype=bool)
        agent_hist = np.asarray(z["agent_history"], dtype=float) if "agent_history" in z.files else np.zeros((0, 0, 8), dtype=float)
        agent_hist_mask = np.asarray(z["agent_history_mask"]).astype(bool) if "agent_history_mask" in z.files else np.ones(agent_hist.shape[:2], dtype=bool)
        agent_mask = get_mask(z, "agent_mask", length=agent_hist.shape[0])
        agent_future = np.asarray(z["agent_future"], dtype=float) if "agent_future" in z.files else np.zeros((agent_hist.shape[0], 0, 8), dtype=float)
        agent_future_mask = np.asarray(z["agent_future_mask"]).astype(bool) if "agent_future_mask" in z.files else np.zeros(agent_future.shape[:2], dtype=bool)
        evidence = np.asarray(z["evidence_features"], dtype=float) if "evidence_features" in z.files else np.zeros((0, 32), dtype=float)
        e_type = np.asarray(z["evidence_type"], dtype=int) if "evidence_type" in z.files else np.zeros((evidence.shape[0],), dtype=int)
        e_mask = get_mask(z, "evidence_mask", length=evidence.shape[0])
        importance = evidence_importance(z)
        if importance.shape[0] != evidence.shape[0]:
            importance = np.zeros((evidence.shape[0],), dtype=float)

        fig = plt.figure(figsize=(18, 11))
        gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1.0, 1.0], height_ratios=[2.0, 1.0])
        ax = fig.add_subplot(gs[:, 0])
        ax_cost = fig.add_subplot(gs[0, 1])
        ax_ev = fig.add_subplot(gs[0, 2])
        ax_q = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[1, 2])

        # Compact encoder map polylines.
        for i in range(map_polys.shape[0]):
            m = map_masks[i]
            if not m.any():
                continue
            pts = map_polys[i, m, :2]
            lane_flag = float(np.nanmax(map_polys[i, m, 2])) if map_polys.shape[-1] > 2 else 0.0
            rule_flag = float(np.nanmax(map_polys[i, m, 3])) if map_polys.shape[-1] > 3 else 0.0
            lw = 0.8 + 0.4 * rule_flag
            alpha = 0.25 + 0.25 * lane_flag
            plot_poly(ax, pts, lw=lw, alpha=alpha)

        # Exact geometries from evidence metadata and route_info. These are what GeometryQuery uses.
        for poly in route_info.get("route_polygons", [])[:64]:
            plot_poly(ax, poly, closed=True, lw=1.2, alpha=0.35, linestyle="--")
        for line in route_info.get("route_polylines", [])[:64]:
            plot_poly(ax, line, closed=False, lw=1.2, alpha=0.45, linestyle="--")

        for m in ev_meta:
            if not isinstance(m, dict) or m.get("type") != "map_rule":
                continue
            layer = str(m.get("layer", ""))
            code = int(m.get("rule_code", 0))

            if skip_drivable_union and layer == "DRIVABLE_AREA_UNION":
                xy = np.asarray(m.get("xy", []), dtype=float)
                if xy.shape == (2,):
                    ax.scatter([xy[0]], [xy[1]], marker=".", s=20, alpha=0.4, label="drivable union centroid")
                continue

            polygons = m.get("polygons", []) or []
            for poly in polygons[:max_exact_polygons]:
                arr = np.asarray(poly, dtype=float)
                if arr.ndim == 2 and arr.shape[0] > max_polygon_points:
                    step = max(1, int(np.ceil(arr.shape[0] / max_polygon_points)))
                    arr = arr[::step]
                plot_poly(ax, arr, closed=True, lw=lw, alpha=alpha)
        # Agents: current boxes and future tracks.
        for i in np.where(agent_mask)[0]:
            if i >= agent_hist.shape[0]:
                continue
            valid_h = np.where(agent_hist_mask[i])[0] if i < agent_hist_mask.shape[0] else np.arange(agent_hist.shape[1])
            if len(valid_h) == 0:
                continue
            st = agent_hist[i, valid_h[-1]]
            x, y, yaw, vx, vy, length, width = st[:7]
            draw_box(ax, x, y, yaw, max(length, 1.0), max(width, 0.5), label=str(i), alpha=0.35, lw=0.8)
            if agent_future.shape[1] > 0 and i < agent_future.shape[0] and agent_future_mask[i].any():
                pts = agent_future[i, agent_future_mask[i], :2]
                if len(pts):
                    ax.plot(pts[:, 0], pts[:, 1], linewidth=0.8, alpha=0.45)

        # Actions and oracle.
        valid_actions = np.where(action_mask)[0]
        order = valid_actions
        if teacher.size == actions.shape[0] and action_mask.any():
            order = valid_actions[np.argsort(teacher[valid_actions])]
        for k in valid_actions:
            mode = int(action_meta[k, 0]) if action_meta.shape[1] else -1
            lw = 3.0 if k == oracle else (1.8 if k in order[:3] else 0.75)
            alpha = 0.95 if k == oracle else (0.75 if k in order[:3] else 0.35)
            ax.plot(actions[k, :, 0], actions[k, :, 1], linewidth=lw, alpha=alpha)
            if k == oracle or k in order[:3]:
                ax.text(actions[k, -1, 0], actions[k, -1, 1], f"a{k}:{ACTION_MODE.get(mode, mode)}", fontsize=6)
        if logged.size and logged.ndim == 2 and logged.shape[1] >= 2:
            ax.plot(logged[:, 0], logged[:, 1], "--", linewidth=2.2, label="logged ego future")

        # Ego box / axes.
        draw_box(ax, 0, 0, 0, 4.8, 2.0, label="ego", alpha=0.9, lw=1.8)
        ax.arrow(0, 0, 7, 0, head_width=1.0, length_includes_head=True, alpha=0.8)
        ax.arrow(0, 0, 0, 5, head_width=1.0, length_includes_head=True, alpha=0.8)
        ax.text(7.5, 0, "+x forward", fontsize=8)
        ax.text(0, 5.5, "+y left", fontsize=8)

        # Evidence points by type, annotate the most action-relevant labels.
        marker_by_type = {0: "o", 1: "X", 2: "P", 3: "^", 4: "v"}
        for t in sorted(set(e_type[e_mask].tolist())) if e_mask.any() else []:
            ids = np.where(e_mask & (e_type == t))[0]
            if len(ids) == 0 or int(t) == 5:
                continue
            ax.scatter(evidence[ids, 1], evidence[ids, 2], s=18, marker=marker_by_type.get(int(t), "x"), alpha=0.8, label=EVIDENCE_TYPE.get(int(t), str(t)))
        top_ids = np.argsort(-importance)[:top_evidence]
        for i in top_ids:
            if i >= len(evidence) or not e_mask[i] or importance[i] <= 0:
                continue
            ax.text(evidence[i, 1], evidence[i, 2], f"e{i}:{importance[i]:.1f}", fontsize=5, alpha=0.8)

        ax.set_title(f"{path.name}\nmap={meta.get('map_name','')} success={meta.get('map_success')} oracle=a{oracle} valid_actions={int(action_mask.sum())} evidence={int(e_mask.sum())}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right", ncol=2)

        # Teacher cost panel.
        if teacher.size and action_mask.any():
            sorted_ids = valid_actions[np.argsort(teacher[valid_actions])]
            vals = teacher[sorted_ids]
            labels = [f"a{i}\n{ACTION_MODE.get(int(action_meta[i,0]), '?')}" for i in sorted_ids]
            ax_cost.bar(np.arange(len(sorted_ids)), vals)
            ax_cost.set_xticks(np.arange(len(sorted_ids)))
            ax_cost.set_xticklabels(labels, rotation=90, fontsize=6)
            ax_cost.set_title("Teacher cost by valid action (lower is better)")
            ax_cost.grid(True, axis="y", alpha=0.25)

        # Evidence histogram / map layers.
        type_counts = Counter(EVIDENCE_TYPE.get(int(t), str(int(t))) for t in e_type[e_mask]) if len(e_type) else Counter()
        names = list(type_counts.keys())
        vals = [type_counts[n] for n in names]
        ax_ev.bar(np.arange(len(names)), vals)
        ax_ev.set_xticks(np.arange(len(names)))
        ax_ev.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax_ev.set_title("Evidence type counts")
        ax_ev.grid(True, axis="y", alpha=0.25)

        # Geometry/signed evidence quick checks.
        q = np.asarray(z["geometry_query"], dtype=float) if "geometry_query" in z.files else np.zeros((0, 0, 0))
        if q.ndim == 3 and q.shape[-1] >= 24 and e_mask.any() and action_mask.any():
            qv = q[np.ix_(e_mask, action_mask, np.arange(q.shape[-1]))]
            min_dist = qv[..., 0].reshape(-1)
            min_dist = min_dist[np.isfinite(min_dist)]
            ax_q.hist(min_dist, bins=40)
            ax_q.set_title("GeometryQuery min_distance distribution")
            ax_q.set_xlabel("meters")
            ax_q.grid(True, alpha=0.25)
        else:
            ax_q.text(0.1, 0.5, "No geometry_query", transform=ax_q.transAxes)
            ax_q.axis("off")

        ax_text.axis("off")
        layer_counts = Counter(str(m.get("layer", "")) for m in ev_meta if isinstance(m, dict))
        text_lines = [
            "Sample summary",
            f"scenario: {summary['scenario_id']}",
            f"map_success: {summary['map_success']}  error: {summary['map_error']}",
            f"route roadblocks: {summary['route_roadblock_count']}  route polys: {summary['route_polygons']}  route lines: {summary['route_polylines']}",
            f"valid actions: {summary['valid_action_count']}  valid evidence: {summary['valid_evidence_count']}",
            f"minADE/minFDE: {summary['min_ade']} / {summary['min_fde']}",
            f"finite teacher/q/signed: {summary['teacher_cost_finite']} / {summary['geometry_query_finite']} / {summary['signed_evidence_finite']}",
            f"q valid ratio: {summary['geometry_query_valid_ratio']}",
            f"signed nonzero ratio: {summary['signed_nonzero_ratio']}",
            f"rival positive ratio: {summary['rival_positive_ratio']}",
            "",
            "Top evidence layers:",
        ]
        for k, v in layer_counts.most_common(12):
            text_lines.append(f"  {repr(k)}: {v}")
        ax_text.text(0.0, 1.0, "\n".join(text_lines), va="top", fontsize=8, family="monospace")

        fig.tight_layout()
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=save_dpi, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        if summary_out:
            summary_out.parent.mkdir(parents=True, exist_ok=True)
            summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary


def iter_samples(path: Path, limit: int) -> list[Path]:
    if path.is_dir():
        files = sorted(path.rglob("*.npz"))
    else:
        files = [path]

    return files[:limit] if limit > 0 else files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="sample_*.npz file or cache directory")
    p.add_argument("--out", type=Path, default=None, help="output PNG for a single sample")
    p.add_argument("--out-dir", type=Path, default=None, help="output directory when input is a cache directory")
    p.add_argument("--summary", type=Path, default=None, help="write one JSON summary for a single sample")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--xlim", nargs=2, type=float, default=(-40.0, 90.0))
    p.add_argument("--ylim", nargs=2, type=float, default=(-55.0, 55.0))
    p.add_argument("--top-evidence", type=int, default=25)
    p.add_argument("--skip-drivable-union", action="store_true", default=True)
    p.add_argument("--draw-drivable-union", action="store_true")
    p.add_argument("--max-exact-polygons", type=int, default=20)
    p.add_argument("--max-polygon-points", type=int, default=80)
    p.add_argument("--save-dpi", type=int, default=120)
    args = p.parse_args()
    if args.draw_drivable_union:
        args.skip_drivable_union = False

    samples = iter_samples(args.input, args.limit)
    if not samples:
        raise SystemExit(f"No sample_*.npz found in {args.input}")

    all_summaries = []
    for sample in samples:
        if args.input.is_file():
            out = args.out
            summary_out = args.summary
        else:
            out_dir = args.out_dir or args.input / "debug_viz"
            out = out_dir / f"{sample.stem}_debug.png"
            summary_out = out_dir / f"{sample.stem}_summary.json"
        all_summaries.append(visualize_one(
    sample,
    out=out,
    summary_out=summary_out,
    xlim=tuple(args.xlim),
    ylim=tuple(args.ylim),
    top_evidence=args.top_evidence,
    skip_drivable_union=args.skip_drivable_union,
    max_exact_polygons=args.max_exact_polygons,
    max_polygon_points=args.max_polygon_points,
    save_dpi=args.save_dpi,
))
        if out:
            print(f"wrote {out}")

    if len(all_summaries) > 1:
        out_dir = args.out_dir or args.input / "debug_viz"
        (out_dir / "summary_all.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out_dir / 'summary_all.json'}")


if __name__ == "__main__":
    main()