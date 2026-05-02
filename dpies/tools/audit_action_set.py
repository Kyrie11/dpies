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

from dpies.actions.trajectory_quality import batch_action_quality
from dpies.common.types import ActionMode, EvidenceType, MapRuleCode, QueryIndex


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


def _mode_name(m: int) -> str:
    try:
        return ActionMode(int(m)).name
    except Exception:
        return str(int(m))


def _mode_counts(action_meta: np.ndarray, action_mask: np.ndarray) -> dict[str, int]:
    modes = action_meta[:, 0].astype(int)
    out = {}
    for m in modes[action_mask.astype(bool)]:
        name = _mode_name(int(m))
        out[name] = out.get(name, 0) + 1
    return out


def _expert_progress(logged_future: np.ndarray) -> tuple[float, float, float]:
    fut = np.asarray(logged_future, dtype=np.float32)
    if fut.ndim != 2 or len(fut) == 0:
        return 0.0, 0.0, 0.0
    final_x = float(max(fut[-1, 0], 0.0))
    disp = float(np.linalg.norm(fut[-1, :2]))
    path = float(np.linalg.norm(np.diff(fut[:, :2], axis=0), axis=1).sum()) if len(fut) > 1 else disp
    return final_x, disp, path


def _coverage(actions: np.ndarray, action_mask: np.ndarray, logged_future: np.ndarray, dt: float, min_expert_progress: float) -> dict[str, float]:
    q = batch_action_quality(actions, action_mask, dt)
    valid = q["valid"].astype(bool)
    if not valid.any():
        return {}
    expert_x, expert_disp, expert_path = _expert_progress(logged_future)
    progress = q["progress"]
    smooth = valid & (q["comfort_violation"] < 0.5)
    denom = max(expert_x, 1e-6)
    ratios = progress[valid] / denom
    smooth_ratios = progress[smooth] / denom if smooth.any() else np.asarray([], dtype=np.float32)
    moving = float(expert_x >= min_expert_progress)
    if moving:
        max_ratio = float(np.max(ratios))
        p90_ratio = float(np.percentile(ratios, 90))
        capped_max_ratio = float(min(max_ratio, 2.0))
        cover80 = float(np.any(ratios >= 0.80))
        smooth_cover80 = float(smooth_ratios.size > 0 and np.any(smooth_ratios >= 0.80))
    else:
        max_ratio = float("nan")
        p90_ratio = float("nan")
        capped_max_ratio = float("nan")
        cover80 = float("nan")
        smooth_cover80 = float("nan")
    return {
        "expert_progress_x": float(expert_x),
        "expert_displacement": float(expert_disp),
        "expert_path_length": float(expert_path),
        "is_moving_expert": moving,
        "max_action_progress": float(np.max(progress[valid])),
        "max_progress_ratio_moving_only": max_ratio,
        "max_progress_ratio_capped_moving_only": capped_max_ratio,
        "p90_progress_ratio_moving_only": p90_ratio,
        "has_action_cover_80pct_expert_moving_only": cover80,
        "has_smooth_action_cover_80pct_expert_moving_only": smooth_cover80,
        "smooth_action_frac": float(smooth.sum() / max(valid.sum(), 1)),
        "min_max_abs_accel": float(np.min(q["max_abs_accel"][valid])),
        "p50_max_abs_accel": float(np.percentile(q["max_abs_accel"][valid], 50)),
        "p90_max_abs_accel": float(np.percentile(q["max_abs_accel"][valid], 90)),
        "p90_max_abs_jerk": float(np.percentile(q["max_abs_jerk"][valid], 90)),
    }


def _map_rule_action_bad_frac(npz: Any, lane_mask: np.ndarray, rule: MapRuleCode, qidx: QueryIndex, threshold: float) -> float:
    if not lane_mask.any() or "geometry_query" not in npz or "evidence_type" not in npz or "evidence_features" not in npz or "evidence_mask" not in npz:
        return float("nan")
    gq = npz["geometry_query"]
    et = npz["evidence_type"].astype(int)
    ef = npz["evidence_features"]
    em = npz["evidence_mask"].astype(bool)
    rule_code = ef[:, 12].round().astype(int) if ef.shape[-1] > 12 else np.zeros((len(et),), dtype=int)
    ev = em & (et == int(EvidenceType.MAP_RULE)) & (rule_code == int(rule))
    if not ev.any():
        return float("nan")
    vals = np.max(gq[ev, :, int(qidx)], axis=0)
    return float(np.mean(vals[lane_mask] > threshold))


def _lane_topology_summary(npz: Any, q: dict[str, np.ndarray]) -> dict[str, float]:
    am = npz["action_mask"].astype(bool)
    modes = npz["action_meta"][:, 0].astype(int)
    lane = am & np.isin(modes, [int(ActionMode.LATERAL_LEFT), int(ActionMode.LATERAL_RIGHT)])
    if not lane.any():
        return {"lane_action_count": 0, "lane_route_bad_frac": 0.0, "lane_boundary_bad_frac": 0.0, "lane_drivable_bad_frac": 0.0}
    return {
        "lane_action_count": int(lane.sum()),
        # IMPORTANT: compute per actual MAP_RULE type only. Do not aggregate QueryIndex 19/20/22 over GAP evidence.
        "lane_route_bad_frac": _map_rule_action_bad_frac(npz, lane, MapRuleCode.ROUTE_DEVIATION, QueryIndex.ROUTE_DEVIATION, 0.5),
        "lane_boundary_bad_frac": _map_rule_action_bad_frac(npz, lane, MapRuleCode.LANE_BOUNDARY, QueryIndex.GAP_APPLIES_OR_LANE_BOUNDARY, 0.5),
        "lane_drivable_bad_frac": _map_rule_action_bad_frac(npz, lane, MapRuleCode.DRIVABLE_AREA, QueryIndex.MERGE_TIME_OR_DRIVABLE, 0.5),
        "lane_smooth_frac": float(np.mean(q["comfort_violation"][lane] < 0.5)),
        "lane_median_progress": float(np.median(q["progress"][lane])),
    }


def audit_one(path: Path, dt_default: float, min_expert_progress: float) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        meta = _json_npz(z["metadata_json"]) if "metadata_json" in z else {}
        dt = float(meta.get("dt", dt_default)) if isinstance(meta, dict) else dt_default
        actions = z["actions"]
        action_mask = z["action_mask"].astype(bool)
        action_meta = z["action_meta"]
        q = batch_action_quality(actions, action_mask, dt)
        modes = action_meta[:, 0].astype(int)
        valid = action_mask.astype(bool)
        stop_creep = valid & np.isin(modes, [int(ActionMode.STOP), int(ActionMode.CREEP)])
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
            "oracle_mode": _mode_name(oracle_mode),
            "oracle_is_stop_creep": float(oracle_mode in [int(ActionMode.STOP), int(ActionMode.CREEP)]),
            "oracle_progress": float(q["progress"][oracle]) if 0 <= oracle < len(valid) else float("nan"),
            "oracle_comfort_violation": float(q["comfort_violation"][oracle]) if 0 <= oracle < len(valid) else float("nan"),
            "stop_creep_frac": float(stop_creep.sum() / max(valid.sum(), 1)),
            "stop_creep_best_cost_rank": -1,
            "mode_counts_json": json.dumps(_mode_counts(action_meta, valid), ensure_ascii=False),
            "action_filter_pre_count": int(meta.get("action_filter_pre_count", -1)) if isinstance(meta, dict) else -1,
            "action_filter_post_count": int(meta.get("action_filter_post_count", -1)) if isinstance(meta, dict) else -1,
            "action_filter_dropped_count": int(meta.get("action_filter_dropped_count", -1)) if isinstance(meta, dict) else -1,
        }
        if np.isfinite(teacher_cost[valid]).any() and stop_creep.any():
            order = np.argsort(teacher_cost[valid])
            valid_idx = np.where(valid)[0]
            ranks = {int(idx): int(r) for r, idx in enumerate(valid_idx[order])}
            row["stop_creep_best_cost_rank"] = min(ranks[int(i)] for i in np.where(stop_creep)[0])
        row.update(_coverage(actions, valid, z["logged_ego_future"], dt, min_expert_progress))
        row.update(_lane_topology_summary(z, q))
        return row


def main() -> None:
    p = argparse.ArgumentParser("Audit DPIES action-set coverage/smoothness/topology/bias from npz cache; v2 fixes near-zero expert ratios and GAP-vs-MAP_RULE topology mixing.")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--out", default="action_set_audit_v2.csv")
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--min-expert-progress", type=float, default=5.0)
    args = p.parse_args()
    files = sorted(Path(args.cache_dir).rglob("*.npz"))
    if args.max_files:
        files = files[:args.max_files]
    rows = [audit_one(f, args.dt, args.min_expert_progress) for f in tqdm(files, desc="audit-v2")]
    if not rows:
        raise SystemExit("No .npz files found")
    keys = sorted({k for r in rows for k in r.keys()})
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def mean(name: str) -> float:
        vals = []
        for r in rows:
            if name not in r or r[name] in (None, ""):
                continue
            try:
                v = float(r[name])
            except Exception:
                continue
            if np.isfinite(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    print(json.dumps({
        "num_samples": len(rows),
        "moving_expert_frac": mean("is_moving_expert"),
        "mean_max_progress_ratio_capped_moving_only": mean("max_progress_ratio_capped_moving_only"),
        "frac_cover_80pct_expert_moving_only": mean("has_action_cover_80pct_expert_moving_only"),
        "frac_smooth_cover_80pct_expert_moving_only": mean("has_smooth_action_cover_80pct_expert_moving_only"),
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
