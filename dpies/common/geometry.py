from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

try:
    from shapely.geometry import LineString, Point, Polygon
except Exception:  # pragma: no cover
    LineString = None
    Point = None
    Polygon = None


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def quaternion_yaw(qw: float, qx: float, qy: float, qz: float) -> float:
    # z-y-x convention; sufficient for nuPlan ego poses on a flat map.
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def global_to_ego_points(points_xy: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape((-1, 2)).astype(np.float32)
    shifted = pts.reshape(-1, 2) - np.asarray(ego_xy, dtype=np.float32).reshape(1, 2)
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return (shifted @ rot.T).reshape(pts.shape).astype(np.float32)


def ego_to_global_points(points_xy: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return (pts.reshape(-1, 2) @ rot.T + np.asarray(ego_xy, dtype=np.float32).reshape(1, 2)).reshape(pts.shape)


def transform_state_global_to_ego(state: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    """Transform [x,y,yaw,vx,vy,length,width,type] to ego coordinates."""
    s = np.asarray(state, dtype=np.float32).copy()
    xy = global_to_ego_points(s[..., 0:2], ego_xy, ego_yaw)
    s[..., 0:2] = xy
    s[..., 2] = wrap_angle(s[..., 2] - ego_yaw)
    vel = s[..., 3:5]
    c, sn = math.cos(-ego_yaw), math.sin(-ego_yaw)
    rot = np.asarray([[c, -sn], [sn, c]], dtype=np.float32)
    s[..., 3:5] = (vel.reshape(-1, 2) @ rot.T).reshape(vel.shape)
    return s


def interp_rows_by_time(times_us: np.ndarray, values: np.ndarray, target_times_us: np.ndarray) -> np.ndarray:
    """Nearest-neighbor resampling for small DB snippets."""
    times_us = np.asarray(times_us, dtype=np.int64)
    values = np.asarray(values)
    target_times_us = np.asarray(target_times_us, dtype=np.int64)
    if len(times_us) == 0:
        raise ValueError("Cannot interpolate empty time series")
    order = np.argsort(times_us)
    t = times_us[order]
    v = values[order]
    idx = np.searchsorted(t, target_times_us, side="left")
    idx = np.clip(idx, 0, len(t) - 1)
    left = np.clip(idx - 1, 0, len(t) - 1)
    choose_left = np.abs(t[left] - target_times_us) < np.abs(t[idx] - target_times_us)
    nearest = np.where(choose_left, left, idx)
    return v[nearest]


def pairwise_min_distance(traj_xy: np.ndarray, points_xy: np.ndarray) -> Tuple[float, int, int]:
    traj = np.asarray(traj_xy, dtype=np.float32)
    pts = np.asarray(points_xy, dtype=np.float32)
    if traj.size == 0 or pts.size == 0:
        return float("inf"), -1, -1
    diff = traj[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    flat = int(np.argmin(dist))
    i, j = np.unravel_index(flat, dist.shape)
    return float(dist[i, j]), int(i), int(j)


def box_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    hl, hw = max(length, 0.1) / 2.0, max(width, 0.1) / 2.0
    local = np.asarray([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]], dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return local @ rot.T + np.asarray([x, y], dtype=np.float32)


def polygon_overlap_area(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    if Polygon is None:
        # Fallback: no exact overlap without shapely.
        return 0.0
    try:
        pa = Polygon(poly_a)
        pb = Polygon(poly_b)
        if not pa.is_valid or not pb.is_valid:
            return 0.0
        return float(pa.intersection(pb).area)
    except Exception:
        return 0.0


def polyline_length(polyline_xy: np.ndarray) -> float:
    pts = np.asarray(polyline_xy, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def time_to_collision_1d(rel_pos: np.ndarray, rel_vel: np.ndarray, radius: float = 4.0) -> float:
    """Closest-approach TTC proxy. Returns inf if not closing."""
    rel_pos = np.asarray(rel_pos, dtype=np.float32)
    rel_vel = np.asarray(rel_vel, dtype=np.float32)
    denom = float(np.dot(rel_vel, rel_vel))
    if denom < 1e-6:
        return float("inf")
    t_star = -float(np.dot(rel_pos, rel_vel)) / denom
    if t_star < 0.0:
        return float("inf")
    closest = rel_pos + t_star * rel_vel
    if float(np.linalg.norm(closest)) > radius:
        return float("inf")
    return t_star


def smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def pad_or_trim(arr: np.ndarray, shape: Sequence[int], value: float = 0.0) -> np.ndarray:
    out = np.full(shape, value, dtype=arr.dtype if hasattr(arr, "dtype") else np.float32)
    slices = tuple(slice(0, min(arr.shape[i], shape[i])) for i in range(len(shape)))
    out[slices] = arr[slices]
    return out
