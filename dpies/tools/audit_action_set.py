from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

from dpies.actions.trajectory_quality import batch_action_quality, coverage_summary
from dpies.common.types import ActionMode, QueryIndex


def _json_npz(x: Any) -> Any:
    try:
        if isinstance(x, np.ndarray):
            if x.shape == ():
                x = x.item()
            else:
                x = x.tobytes()
        if isinstance(x, bytes):
            return json.loads(x.decode("utf-8"))
        if isinstance(x, str):
            return json.loads(x)
    except Exception:
        return None
    return None


def _mode_counts(action_meta: np.ndarray, action_mask: np.ndarray) -> dict[str, int]:
    modes = action_meta[:, 0].astype(int)
    out = {}
    for m in modes[action_mask.astype(bool)]:
        try:
            name = ActionMode(int(m)).name
        except Exception:
            name = str(int(m))
        out[name] = out.get(name, 0) + 1
    return out


def _lane_topology_summary(npz: Any, q: dict[str, np.ndarray]) -> dict[str, float]:
    # Uses geometry_query produced by preprocessing. For map-rule evidence,
    # QueryIndex.ROUTE_DEVIATION, LANE_BOUNDARY, and drivable flags are route/topology proxies.
    if "geometry_query" not in npz or "evidence_type" not in npz or "evidence_mask" not in npz:
        return {}
    gq = npz["geometry_query"]
    et = npz["evidence_type"].astype(int)
    em = npz["evidence_mask"].astype(bool)
    am = npz["action_mask"].astype(bool)
    modes = npz["action_meta"][:, 0].astype(int)
    lane = am & np.isin(modes, [int(ActionMode.LATERAL_LEFT), int(ActionMode.LATERAL_RIGHT)])
    if not lane.any():
        return {"lane_action_count": 0, "lane_route_bad_frac": 0.0, "lane_boundary_bad_frac": 0.0, "lane_drivable_bad_frac": 0.0}
    # Query fields are zero if no relevant evidence. Aggregate max over evidence units.
    route_dev = np.max(gq[em, :, int(QueryIndex.ROUTE_DEVIATION)], axis=0) if em.any() else np.zeros_like(am, dtype=np.float32)
    lane_boundary = np.max(gq[em, :, int(QueryIndex.GAP_APPLIES_OR_LANE_BOUNDARY)], axis=0) if em.any() else np.zeros_like(am, dtype=np.float32)
    drivable = np.max(gq[em, :, int(QueryIndex.MERGE_TIME_OR_DRIVABLE)], axis=0) if em.any() else np.zeros_like(am, dtype=np.float32)
    return {
        "lane_action_count": int(lane.sum()),
        "lane_route_bad_frac": float(np.mean(route_dev[lane] > 0.5)),
        "lane_boundary_bad_frac": float(np.mean(lane_boundary[lane] > 0.5)),
        "lane_drivable_bad_frac": float(np.mean(drivable[lane] > 0.5)),
        "lane_smooth_frac": float(np.mean(q["comfort_violation"][lane] < 0.5)),
        "lane_median_progress": float(np.median(q["progress"][lane])),
    }


def audit_one(path: Path, dt_default: float) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        meta = _json_npz(z["metadata_json"]) if "metadata_json" in z else {}
        dt = float(meta.get("dt", dt_default)) if isinstance(meta, dict) else dt_default
        actions = z["actions"]
        action_mask = z["action_mask"].astype(bool)
        action_meta = z["action_meta"]
        q = batch_action_quality(actions, action_mask, dt)
        cov = coverage_summary(actions, action_mask, z["logged_ego_future"], dt)
        modes = action_meta[:, 0].astype(int)
        stop_creep = action_mask & np.isin(modes, [int(ActionMode.STOP), int(ActionMode.CREEP)])
        valid = action_mask.astype(bool)
        oracle = int(np.asarray(z["oracle_action_index"]).item()) if "oracle_action_index" in z else -1
        teacher_cost = z["teacher_cost"] if "teacher_cost" in z else np.full((len(valid),), np.nan)
        oracle_mode = int(modes[oracle]) if 0 <= oracle < len(modes) else -1
        current_speed = float(z["ego_history"][-1, 8]) if "ego_history" in z and z["ego_history"].shape[-1] > 8 else 0.0
        row: dict[str, Any] = {
            "file": str(path),
            "scenario_id": meta.get("scenario_id", path.stem) if isinstance(meta, dict) else path.stem,
            "current_speed": current_speed,
            "high_speed": float(current_speed >= 10.0),
            "valid_action_count": int(valid.sum()),
            "oracle_action_index": oracle,
            "oracle_mode": str(ActionMode(oracle_mode).name) if oracle_mode in [int(x) for x in ActionMode] else str(oracle_mode),
            "oracle_is_stop_creep": float(oracle_mode in [int(ActionMode.STOP), int(ActionMode.CREEP)]),
            "stop_creep_frac": float(stop_creep.sum() / max(valid.sum(), 1)),
            "stop_creep_best_cost_rank": -1,
            "mode_counts_json": json.dumps(_mode_counts(action_meta, valid), ensure_ascii=False),
        }
        if np.isfinite(teacher_cost[valid]).any() and stop_creep.any():
            order = np.argsort(teacher_cost[valid])
            valid_idx = np.where(valid)[0]
            ranks = {int(idx): int(r) for r, idx in enumerate(valid_idx[order])}
            row["stop_creep_best_cost_rank"] = min(ranks[int(i)] for i in np.where(stop_creep)[0])
        row.update(cov)
        row.update(_lane_topology_summary(z, q))
        return row


def main() -> None:
    p = argparse.ArgumentParser("Audit candidate action set coverage/smoothness/topology/bias from DPIES npz cache.")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", default="action_set_audit.csv")
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--max-files", type=int, default=None)
    args = p.parse_args()
    files = sorted(Path(args.cache_dir).rglob("*.npz"))
    if args.max_files:
        files = files[:args.max_files]
    rows = [audit_one(f, args.dt) for f in tqdm(files, desc="audit")]
    if not rows:
        raise SystemExit("No .npz files found")
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # Compact aggregate printed to terminal.
    def mean(name: str) -> float:
        vals = [float(r[name]) for r in rows if name in r and r[name] not in (None, "")]
        return float(np.mean(vals)) if vals else float("nan")
    print(json.dumps({
        "num_samples": len(rows),
        "mean_max_progress_ratio": mean("max_progress_ratio"),
        "frac_cover_80pct_expert": mean("has_action_cover_80pct_expert"),
        "frac_smooth_cover_80pct_expert": mean("has_smooth_action_cover_80pct_expert"),
        "mean_smooth_action_frac": mean("smooth_action_frac"),
        "mean_stop_creep_frac": mean("stop_creep_frac"),
        "mean_oracle_is_stop_creep": mean("oracle_is_stop_creep"),
        "mean_lane_route_bad_frac": mean("lane_route_bad_frac"),
        "mean_lane_boundary_bad_frac": mean("lane_boundary_bad_frac"),
        "mean_lane_drivable_bad_frac": mean("lane_drivable_bad_frac"),
    }, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
