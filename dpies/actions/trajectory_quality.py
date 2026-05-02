from __future__ import annotations

import numpy as np


def _safe_dt(dt: float) -> float:
    return max(float(dt), 1e-3)


def trajectory_kinematics(action: np.ndarray, dt: float = 0.5) -> dict[str, float]:
    """Compute lightweight smoothness/progress statistics for one ego-frame action.

    Expected action schema: [x, y, yaw, speed, accel, ...]. If speed/accel are
    missing or unreliable, speed/accel are also estimated from xy finite differences.
    """
    a = np.asarray(action, dtype=np.float32)
    dt = _safe_dt(dt)
    if a.ndim != 2 or len(a) < 2:
        return {k: 0.0 for k in (
            "progress", "lateral_abs", "path_length", "final_speed", "max_abs_accel",
            "max_abs_jerk", "max_abs_yaw_rate", "max_abs_yaw_accel", "comfort_violation")}

    xy = a[:, :2]
    dxy = np.diff(xy, axis=0)
    step_dist = np.linalg.norm(dxy, axis=-1)
    path_length = float(step_dist.sum())
    progress = float(a[-1, 0] - a[0, 0])
    lateral_abs = float(np.max(np.abs(a[:, 1])))

    if a.shape[1] > 3:
        speed = np.asarray(a[:, 3], dtype=np.float32)
    else:
        speed = np.r_[step_dist[:1] / dt, step_dist / dt].astype(np.float32)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

    if a.shape[1] > 4:
        accel = np.asarray(a[:, 4], dtype=np.float32)
    else:
        accel = np.gradient(speed, dt).astype(np.float32)
    accel_fd = np.gradient(speed, dt).astype(np.float32)
    # Use the worse of stored accel and finite-difference accel. This catches
    # rollouts whose metadata says smooth but xy/speed imply a jump.
    max_abs_accel = float(max(np.max(np.abs(accel)), np.max(np.abs(accel_fd))))
    jerk = np.gradient(accel, dt).astype(np.float32)
    jerk_fd = np.gradient(accel_fd, dt).astype(np.float32)
    max_abs_jerk = float(max(np.max(np.abs(jerk)), np.max(np.abs(jerk_fd))))

    yaw = np.unwrap(a[:, 2].astype(np.float32)) if a.shape[1] > 2 else np.zeros((len(a),), dtype=np.float32)
    yaw_rate = np.gradient(yaw, dt).astype(np.float32)
    yaw_accel = np.gradient(yaw_rate, dt).astype(np.float32)
    max_abs_yaw_rate = float(np.max(np.abs(yaw_rate)))
    max_abs_yaw_accel = float(np.max(np.abs(yaw_accel)))

    # nuPlan-like conservative bounds. Adjust if your benchmark uses different bounds.
    comfort_violation = float(
        max_abs_accel > 4.0 or
        max_abs_jerk > 8.0 or
        max_abs_yaw_rate > 0.95 or
        max_abs_yaw_accel > 1.93
    )
    return {
        "progress": progress,
        "lateral_abs": lateral_abs,
        "path_length": path_length,
        "final_speed": float(speed[-1]),
        "max_abs_accel": max_abs_accel,
        "max_abs_jerk": max_abs_jerk,
        "max_abs_yaw_rate": max_abs_yaw_rate,
        "max_abs_yaw_accel": max_abs_yaw_accel,
        "comfort_violation": comfort_violation,
    }


def batch_action_quality(actions: np.ndarray, action_mask: np.ndarray, dt: float = 0.5) -> dict[str, np.ndarray]:
    valid = np.asarray(action_mask).astype(bool)
    rows = [trajectory_kinematics(actions[i], dt) for i in range(actions.shape[0])]
    keys = list(rows[0].keys()) if rows else []
    out = {k: np.asarray([r[k] for r in rows], dtype=np.float32) for k in keys}
    out["valid"] = valid.astype(bool)
    return out


def expert_route_progress(logged_future: np.ndarray) -> float:
    fut = np.asarray(logged_future, dtype=np.float32)
    if fut.ndim != 2 or len(fut) == 0:
        return 0.0
    # In the ego frame, x is forward route-progress proxy. Clamp because reverse
    # motion or noisy labels should not explode ratios.
    return float(max(fut[-1, 0], 0.0))


def coverage_summary(actions: np.ndarray, action_mask: np.ndarray, logged_future: np.ndarray, dt: float = 0.5) -> dict[str, float]:
    q = batch_action_quality(actions, action_mask, dt)
    valid = q["valid"]
    if not valid.any():
        return {}
    expert_p = expert_route_progress(logged_future)
    progress = q["progress"]
    ratios = progress[valid] / max(expert_p, 1e-3)
    smooth = valid & (q["comfort_violation"] < 0.5)
    smooth_ratios = progress[smooth] / max(expert_p, 1e-3) if smooth.any() else np.asarray([], dtype=np.float32)
    return {
        "expert_progress": float(expert_p),
        "max_action_progress": float(np.max(progress[valid])),
        "max_progress_ratio": float(np.max(ratios)),
        "p90_progress_ratio": float(np.percentile(ratios, 90)),
        "has_action_cover_80pct_expert": float(np.any(ratios >= 0.80)),
        "has_smooth_action_cover_80pct_expert": float(smooth_ratios.size > 0 and np.any(smooth_ratios >= 0.80)),
        "smooth_action_frac": float(smooth.sum() / max(valid.sum(), 1)),
        "min_max_abs_accel": float(np.min(q["max_abs_accel"][valid])),
        "p50_max_abs_accel": float(np.percentile(q["max_abs_accel"][valid], 50)),
        "p90_max_abs_accel": float(np.percentile(q["max_abs_accel"][valid], 90)),
        "p90_max_abs_jerk": float(np.percentile(q["max_abs_jerk"][valid], 90)),
    }
