from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from dpies.common.geometry import smoothstep, wrap_angle
from dpies.common.types import ACTION_STATE_DIM


def kinematic_rollout(
    current_speed: float,
    target_speed: float,
    terminal_lateral: float,
    horizon_s: float,
    dt: float,
    terminal_progress: float | None = None,
    max_accel: float = 3.0,
    min_accel: float = -4.0,
) -> np.ndarray:
    """Generate ego-centric nominal trajectory [x,y,yaw,v,accel,curvature]."""
    steps = int(round(horizon_s / dt))
    times = np.arange(1, steps + 1, dtype=np.float32) * dt
    accel = np.clip((target_speed - current_speed) / max(horizon_s, 1e-3), min_accel, max_accel)
    v = np.maximum(current_speed + accel * times, 0.0)
    x = current_speed * times + 0.5 * accel * times * times
    if terminal_progress is not None and terminal_progress > 0:
        final_x = max(float(x[-1]), 1e-3)
        scale = min(1.25, terminal_progress / final_x)
        x = x * scale
        v = v * scale
    u = times / max(horizon_s, 1e-3)
    y = terminal_lateral * smoothstep(u)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    yaw = np.arctan2(dy, np.maximum(dx, 1e-3))
    speed = np.sqrt(dx * dx + dy * dy)
    acc = np.gradient(speed, dt)
    dyaw = np.gradient(yaw, dt)
    curvature = dyaw / np.maximum(speed, 1e-2)
    out = np.stack([x, y, yaw, speed, acc, curvature], axis=-1).astype(np.float32)
    return out


def stop_rollout(current_speed: float, stop_distance: float, horizon_s: float, dt: float) -> np.ndarray:
    steps = int(round(horizon_s / dt))
    times = np.arange(1, steps + 1, dtype=np.float32) * dt
    stop_distance = max(stop_distance, 1.0)
    # Smoothly approach the stop point and keep zero velocity after arrival.
    u = np.clip(times / max(horizon_s * 0.6, 1e-3), 0.0, 1.0)
    x = stop_distance * smoothstep(u)
    y = np.zeros_like(x)
    dx = np.gradient(x, dt)
    speed = np.maximum(dx, 0.0)
    acc = np.gradient(speed, dt)
    yaw = np.zeros_like(x)
    curvature = np.zeros_like(x)
    return np.stack([x, y, yaw, speed, acc, curvature], axis=-1).astype(np.float32)


def action_feasible(traj: np.ndarray, max_accel: float = 3.5, min_accel: float = -5.0, max_curvature: float = 0.35) -> bool:
    if not np.isfinite(traj).all():
        return False
    if np.nanmax(traj[:, 4]) > max_accel + 1e-3 or np.nanmin(traj[:, 4]) < min_accel - 1e-3:
        return False
    if np.nanmax(np.abs(traj[:, 5])) > max_curvature + 1e-3:
        return False
    if traj[-1, 0] < 0.0:
        return False
    return True
