from __future__ import annotations

import math
from typing import Sequence, Tuple

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
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def rotation_matrix(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _rot_to_ego(ego_yaw: float) -> np.ndarray:
    c, s = math.cos(-ego_yaw), math.sin(-ego_yaw)
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def global_to_ego_points(points_xy: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape((-1, 2)).astype(np.float32)
    shifted = pts.reshape(-1, 2) - np.asarray(ego_xy, dtype=np.float32).reshape(1, 2)
    rot = _rot_to_ego(ego_yaw)
    return (shifted @ rot.T).reshape(pts.shape).astype(np.float32)


def ego_to_global_points(points_xy: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float32)
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return (pts.reshape(-1, 2) @ rot.T + np.asarray(ego_xy, dtype=np.float32).reshape(1, 2)).reshape(pts.shape)


def transform_agent_state_global_to_ego(state: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    """Transform agent box state [x,y,yaw,vx,vy,length,width,type_id] to ego coordinates."""
    s = np.asarray(state, dtype=np.float32).copy()
    s[..., 0:2] = global_to_ego_points(s[..., 0:2], ego_xy, ego_yaw)
    s[..., 2] = wrap_angle(s[..., 2] - ego_yaw)
    vel = s[..., 3:5]
    s[..., 3:5] = (vel.reshape(-1, 2) @ _rot_to_ego(ego_yaw).T).reshape(vel.shape)
    return s


def transform_ego_state_global_to_ego(state: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    """Transform ego state [x,y,yaw,vx,vy,ax,ay,yaw_rate,speed] to ego coordinates.

    This fixes the old mixed speed/yaw-rate schema by rotating acceleration
    separately and preserving both yaw_rate and speed when present.
    """
    s = np.asarray(state, dtype=np.float32).copy()
    if s.shape[-1] == 8:
        # Backward-compatible input: [x,y,yaw,vx,vy,ax,ay,yaw_rate]. Add speed.
        speed = np.linalg.norm(s[..., 3:5], axis=-1, keepdims=True)
        s = np.concatenate([s, speed.astype(np.float32)], axis=-1)
    s[..., 0:2] = global_to_ego_points(s[..., 0:2], ego_xy, ego_yaw)
    s[..., 2] = wrap_angle(s[..., 2] - ego_yaw)
    rot = _rot_to_ego(ego_yaw)
    vel = s[..., 3:5]
    acc = s[..., 5:7]
    s[..., 3:5] = (vel.reshape(-1, 2) @ rot.T).reshape(vel.shape)
    s[..., 5:7] = (acc.reshape(-1, 2) @ rot.T).reshape(acc.shape)
    # Preserve yaw_rate in column 7; recompute speed in column 8 in the ego frame.
    s[..., 8] = np.linalg.norm(s[..., 3:5], axis=-1)
    return s.astype(np.float32)


def transform_state_global_to_ego(state: np.ndarray, ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    """Backward-compatible dispatcher for old callers.

    Agent states are 8-D [x,y,yaw,vx,vy,length,width,type_id]. Ego states should
    call transform_ego_state_global_to_ego directly because they are 9-D.
    """
    return transform_agent_state_global_to_ego(state, ego_xy, ego_yaw)



def ego_pose_to_transform(ego_xy: Sequence[float], ego_yaw: float) -> np.ndarray:
    """Return a 3x3 ego-to-global SE(2) matrix for cache metadata."""
    c, ss = math.cos(ego_yaw), math.sin(ego_yaw)
    mat = np.eye(3, dtype=np.float32)
    mat[:2, :2] = np.asarray([[c, -ss], [ss, c]], dtype=np.float32)
    mat[:2, 2] = np.asarray(ego_xy, dtype=np.float32)
    return mat

def interp_rows_by_time(times_us: np.ndarray, values: np.ndarray, target_times_us: np.ndarray) -> np.ndarray:
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
    valid = np.isfinite(traj).all(axis=-1)[:, None] & np.isfinite(pts).all(axis=-1)[None, :]
    diff = traj[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    dist = np.where(valid, dist, np.inf)
    if not np.isfinite(dist).any():
        return float("inf"), -1, -1
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


def _make_polygon(points_xy: np.ndarray):
    if Polygon is None:
        return None
    try:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or len(pts) < 3:
            return None
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0.0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def _make_linestring(points_xy: np.ndarray):
    if LineString is None:
        return None
    try:
        pts = np.asarray(points_xy, dtype=np.float32)
        if pts.ndim != 2 or len(pts) < 2:
            return None
        line = LineString(pts)
        return line if not line.is_empty else None
    except Exception:
        return None


def line_intersects_polyline(line_xy: np.ndarray, polyline_xy: np.ndarray, buffer_m: float = 0.0) -> bool:
    """Return True when a trajectory line intersects a map polyline or its small buffer."""
    if LineString is None:
        return False
    line = _make_linestring(line_xy)
    other = _make_linestring(polyline_xy)
    if line is None or other is None:
        return False
    try:
        geom = other.buffer(buffer_m) if buffer_m > 0.0 else other
        return bool(line.intersects(geom))
    except Exception:
        return False


def line_intersects_polygon(line_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    if LineString is None or Polygon is None:
        return False
    line = _make_linestring(line_xy)
    poly = _make_polygon(polygon_xy)
    if line is None or poly is None:
        return False
    try:
        return bool(line.intersects(poly))
    except Exception:
        return False


def footprint_outside_area_sum(action: np.ndarray, area_polygons: list[np.ndarray], length: float = 4.8, width: float = 2.1) -> tuple[float, float]:
    """Return sum/max ego footprint area outside the union of allowed polygons.

    The input action is ego-frame [T, action_state_dim]. Polygons are also ego-frame.
    If shapely is unavailable or no polygons are supplied, returns zeros so old
    map-free smoke tests remain usable.
    """
    if Polygon is None or not area_polygons:
        return 0.0, 0.0
    try:
        from shapely.ops import unary_union  # type: ignore
        allowed = []
        for p in area_polygons:
            poly = _make_polygon(np.asarray(p, dtype=np.float32))
            if poly is not None:
                allowed.append(poly)
        if not allowed:
            return 0.0, 0.0
        union = unary_union(allowed)
        outside_sum = 0.0
        outside_max = 0.0
        for st in np.asarray(action, dtype=np.float32):
            fp = _make_polygon(box_corners(float(st[0]), float(st[1]), float(st[2]), length, width))
            if fp is None:
                continue
            outside = float(fp.difference(union).area)
            outside_sum += outside
            outside_max = max(outside_max, outside)
        return outside_sum, outside_max
    except Exception:
        return 0.0, 0.0


def footprint_intersects_polylines(action: np.ndarray, polylines: list[np.ndarray], length: float = 4.8, width: float = 2.1, buffer_m: float = 0.05) -> bool:
    """Check if any ego footprint intersects any supplied boundary polyline."""
    if Polygon is None or LineString is None or not polylines:
        return False
    try:
        boundary_geoms = []
        for line_xy in polylines:
            line = _make_linestring(np.asarray(line_xy, dtype=np.float32))
            if line is not None:
                boundary_geoms.append(line.buffer(buffer_m) if buffer_m > 0.0 else line)
        if not boundary_geoms:
            return False
        for st in np.asarray(action, dtype=np.float32):
            fp = _make_polygon(box_corners(float(st[0]), float(st[1]), float(st[2]), length, width))
            if fp is None:
                continue
            if any(fp.intersects(g) for g in boundary_geoms):
                return True
        return False
    except Exception:
        return False
