from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from dpies.common.geometry import box_corners, pairwise_min_distance, polygon_overlap_area, time_to_collision_1d, wrap_angle
from dpies.common.types import EvidenceType, MapRuleCode, QUERY_DIM

try:  # optional but recommended for exact map-rule queries
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    LineString = None
    Point = None
    Polygon = None
    unary_union = None


def _valid_type(type_id: int) -> EvidenceType:
    try:
        return EvidenceType(int(type_id))
    except Exception:
        return EvidenceType.PADDING


def _dynamic_agent_sequence(feature: np.ndarray, action_steps: int, dt: float,
                            future_agents: np.ndarray | None = None,
                            future_agent_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return positions [T,2], valid mask [T], and yaw [T] for an evidence agent."""
    idx = int(round(float(feature[11])))
    if future_agents is not None and idx >= 0 and idx < future_agents.shape[0]:
        fut = future_agents[idx, :action_steps]
        if future_agent_mask is not None and idx < future_agent_mask.shape[0]:
            valid = future_agent_mask[idx, :action_steps].astype(bool)
        else:
            valid = np.any(np.abs(fut[:, :2]) > 1e-4, axis=-1)
        if valid.any():
            xy = fut[:, :2].astype(np.float32)
            yaw = fut[:, 2].astype(np.float32) if fut.shape[-1] > 2 else np.zeros((len(fut),), dtype=np.float32)
            return xy, valid.astype(bool), yaw
    x, y, vx, vy = float(feature[1]), float(feature[2]), float(feature[3]), float(feature[4])
    t = np.arange(1, action_steps + 1, dtype=np.float32) * dt
    xy = np.stack([x + vx * t, y + vy * t], axis=-1).astype(np.float32)
    yaw0 = float(feature[13]) if len(feature) > 13 and abs(float(feature[13])) < np.pi * 2 else (np.arctan2(vy, vx) if abs(vx) + abs(vy) > 1e-3 else 0.0)
    yaw = np.full((action_steps,), yaw0, dtype=np.float32)
    return xy, np.ones((action_steps,), dtype=bool), yaw


def _footprint_distance_and_overlap(action: np.ndarray, agent_xy: np.ndarray, agent_valid: np.ndarray,
                                    agent_yaw: np.ndarray, length: float, width: float) -> tuple[float, int, int, float, float]:
    steps = min(len(action), len(agent_xy))
    if steps <= 0 or not agent_valid[:steps].any():
        return 99.0, -1, -1, 0.0, 0.0
    ego_radius = 2.4
    agent_radius = max(1.0, 0.25 * (length + width))
    best_signed = 99.0
    best_i = -1
    overlap_sum = 0.0
    overlap_max = 0.0
    for t in range(steps):
        if not bool(agent_valid[t]):
            continue
        center_d = float(np.linalg.norm(action[t, :2] - agent_xy[t]))
        signed = center_d - ego_radius - agent_radius
        if signed < best_signed:
            best_signed = signed
            best_i = t
        ego_poly = box_corners(float(action[t, 0]), float(action[t, 1]), float(action[t, 2]), 4.8, 2.1)
        ag_poly = box_corners(float(agent_xy[t, 0]), float(agent_xy[t, 1]), float(agent_yaw[t]), length, width)
        area = polygon_overlap_area(ego_poly, ag_poly)
        overlap_sum += area
        overlap_max = max(overlap_max, area)
    return float(best_signed), best_i, best_i, float(overlap_sum), float(overlap_max)


def _points(value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape((-1, 2))
        return arr if len(arr) else np.zeros((0, 2), dtype=np.float32)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)


def _polygons_from_meta(meta: dict[str, Any] | None) -> list[np.ndarray]:
    if not meta:
        return []
    out: list[np.ndarray] = []
    for key in ("polygons",):
        vals = meta.get(key, [])
        if vals:
            for poly in vals:
                pts = _points(poly)
                if len(pts) >= 3:
                    out.append(pts)
    poly = _points(meta.get("polygon", []))
    if len(poly) >= 3:
        out.append(poly)
    return out


def _polylines_from_meta(meta: dict[str, Any] | None) -> list[np.ndarray]:
    if not meta:
        return []
    out: list[np.ndarray] = []
    vals = meta.get("polylines", [])
    if vals:
        for line in vals:
            pts = _points(line)
            if len(pts) >= 2:
                out.append(pts)
    line = _points(meta.get("polyline", []))
    if len(line) >= 2:
        out.append(line)
    return out


def _make_polygon(pts: np.ndarray):
    if Polygon is None or len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def _make_line(pts: np.ndarray):
    if LineString is None or len(pts) < 2:
        return None
    try:
        return LineString(pts)
    except Exception:
        return None


def _action_centerline(action: np.ndarray):
    if LineString is None or len(action) < 2:
        return None
    try:
        return LineString(action[:, :2])
    except Exception:
        return None


def _ego_polygons(action: np.ndarray) -> list[Any]:
    if Polygon is None:
        return []
    polys = []
    for st in action:
        try:
            p = Polygon(box_corners(float(st[0]), float(st[1]), float(st[2]), 4.8, 2.1))
            if p.is_valid and not p.is_empty:
                polys.append(p)
        except Exception:
            continue
    return polys


def _union_polygons(polys: Sequence[np.ndarray]) -> Any | None:
    if Polygon is None or unary_union is None:
        return None
    geoms = [_make_polygon(p) for p in polys]
    geoms = [g for g in geoms if g is not None]
    if not geoms:
        return None
    try:
        return unary_union(geoms)
    except Exception:
        return None


def _distance_to_meta_geometry(action: np.ndarray, meta: dict[str, Any] | None, fallback_point: np.ndarray) -> tuple[float, int]:
    # Fallback to cheap center distance when shapely geometry is unavailable.
    if LineString is None or Point is None:
        d, idx, _ = pairwise_min_distance(action[:, :2], fallback_point.reshape(1, 2))
        return float(d), int(idx)
    line = _action_centerline(action)
    geoms = []
    for pts in _polylines_from_meta(meta):
        g = _make_line(pts)
        if g is not None:
            geoms.append(g)
    for pts in _polygons_from_meta(meta):
        g = _make_polygon(pts)
        if g is not None:
            geoms.append(g)
    if not geoms or line is None:
        d, idx, _ = pairwise_min_distance(action[:, :2], fallback_point.reshape(1, 2))
        return float(d), int(idx)
    try:
        dist = min(float(line.distance(g)) for g in geoms)
        # Nearest discrete step for timing.
        step_d = []
        for st in action:
            pt = Point(float(st[0]), float(st[1]))
            step_d.append(min(float(pt.distance(g)) for g in geoms))
        return dist, int(np.argmin(step_d))
    except Exception:
        d, idx, _ = pairwise_min_distance(action[:, :2], fallback_point.reshape(1, 2))
        return float(d), int(idx)


def _first_interaction_time(action: np.ndarray, meta: dict[str, Any] | None, dt: float) -> float | None:
    if Polygon is None or LineString is None:
        return None
    line_geoms = [_make_line(x) for x in _polylines_from_meta(meta)]
    poly_geoms = [_make_polygon(x) for x in _polygons_from_meta(meta)]
    geoms = [g for g in (line_geoms + poly_geoms) if g is not None]
    if not geoms:
        return None
    ego_polys = _ego_polygons(action)
    for idx, ep in enumerate(ego_polys):
        try:
            if any(ep.intersects(g) or ep.distance(g) < 0.25 for g in geoms):
                return float(idx + 1) * dt
        except Exception:
            continue
    center = _action_centerline(action)
    if center is not None:
        try:
            if any(center.intersects(g) or center.distance(g) < 0.25 for g in geoms):
                return dt
        except Exception:
            pass
    return None


def _map_rule_exact(q: np.ndarray, feature: np.ndarray, action: np.ndarray, dt: float,
                    meta: dict[str, Any] | None = None, route_info: dict[str, Any] | None = None,
                    use_future_traffic: bool = False) -> None:
    rule_code = int(round(float(feature[12]))) if len(feature) > 12 else int(meta.get("rule_code", 0)) if meta else 0
    ex, ey = float(feature[1]), float(feature[2])
    point = np.asarray([ex, ey], dtype=np.float32)
    min_dist, idx = _distance_to_meta_geometry(action, meta, point)
    q[0] = min(float(min_dist), 99.0)
    q[1] = float(idx + 1) * dt if idx >= 0 else 99.0
    q[2] = 1.0 if min_dist < 0.25 else q[2]
    q[16] = 1.0 if min_dist < 2.5 else 0.0
    speed_limit = float(meta.get("speed_limit_mps", feature[15] if len(feature) > 15 and feature[15] > 0 else 13.4)) if meta else 13.4
    q[21] = max(0.0, float(np.max(action[:, 3])) - speed_limit)

    if Polygon is None or LineString is None:
        return
    ego_polys = _ego_polygons(action)
    centerline = _action_centerline(action)
    polygons = _polygons_from_meta(meta)
    polylines = _polylines_from_meta(meta)

    if rule_code == int(MapRuleCode.STOP_LINE):
        touched = _first_interaction_time(action, meta, dt)
        q[17] = 1.0 if touched is not None else q[17]
    elif rule_code == int(MapRuleCode.TRAFFIC_LIGHT_RED):
        touched_t = _first_interaction_time(action, meta, dt)
        if touched_t is not None:
            status_now = str((meta or {}).get("traffic_light_status", "UNKNOWN")).upper()
            current_red = "RED" in status_now or status_now.endswith("STOP")
            if use_future_traffic:
                red_times = [float(x) for x in (meta or {}).get("red_times_s", [])]
                if current_red or any(abs(touched_t - rt) <= max(1.5 * dt, 0.75) or rt <= touched_t for rt in red_times):
                    q[17] = 1.0
            else:
                # Online/model-input GeometryQuery may only use the current signal state.
                # Future traffic-light statuses are reserved for teacher labels.
                q[17] = 1.0 if current_red else q[17]
    elif rule_code == int(MapRuleCode.CROSSWALK):
        max_area = 0.0
        if polygons:
            cross = _union_polygons(polygons)
            if cross is not None:
                for ep in ego_polys:
                    try:
                        max_area = max(max_area, float(ep.intersection(cross).area))
                    except Exception:
                        pass
        elif centerline is not None:
            for pts in polylines:
                g = _make_line(pts)
                if g is not None and centerline.distance(g) < 2.0:
                    max_area = max(max_area, 1.0)
        q[18] = max(q[18], max_area)
    elif rule_code == int(MapRuleCode.DRIVABLE_AREA):
        drv = _union_polygons(polygons)
        if drv is not None and ego_polys:
            outside = 0.0
            for ep in ego_polys:
                try:
                    outside = max(outside, float(ep.difference(drv).area))
                except Exception:
                    pass
            q[19] = max(q[19], outside)
    elif rule_code == int(MapRuleCode.LANE_BOUNDARY):
        violated = False
        for pts in polylines:
            ln = _make_line(pts)
            if ln is None:
                continue
            if centerline is not None and (centerline.crosses(ln) or centerline.intersects(ln)):
                violated = True
                break
            for ep in ego_polys:
                try:
                    if ep.crosses(ln) or ep.intersects(ln.buffer(0.05)):
                        violated = True
                        break
                except Exception:
                    continue
        q[20] = max(q[20], 1.0 if violated else 0.0)
    elif rule_code == int(MapRuleCode.SPEED_LIMIT):
        q[21] = max(0.0, float(np.max(action[:, 3])) - speed_limit)
    elif rule_code == int(MapRuleCode.ROUTE_DEVIATION):
        route_polys = polygons
        if route_info is not None and not route_polys:
            try:
                route_polys = [np.asarray(p, dtype=np.float32) for p in route_info.get("route_polygons", [])]
            except Exception:
                route_polys = []
        route_union = _union_polygons(route_polys)
        if route_union is not None and ego_polys:
            outside = 0.0
            for ep in ego_polys:
                try:
                    outside = max(outside, float(ep.difference(route_union).area))
                except Exception:
                    pass
            q[22] = max(q[22], outside)


def query_one(feature: np.ndarray, type_id: int, action: np.ndarray, dt: float = 0.5,
              future_agents: np.ndarray | None = None,
              future_agent_mask: np.ndarray | None = None,
              metadata: dict[str, Any] | None = None,
              route_info: dict[str, Any] | None = None,
              use_future_traffic: bool = False) -> np.ndarray:
    q = np.zeros((QUERY_DIM,), dtype=np.float32)
    xy = action[:, :2]
    speed = action[:, 3]
    typ = _valid_type(int(type_id))
    if typ == EvidenceType.PADDING:
        return q
    ex, ey = float(feature[1]), float(feature[2])
    point = np.asarray([[ex, ey]], dtype=np.float32)
    d_point, t_idx, _ = pairwise_min_distance(xy, point)
    arrival_time = float(max(t_idx, 0) + 1) * dt if t_idx >= 0 else 99.0
    q[0] = min(float(d_point), 99.0)
    q[1] = arrival_time
    q[2] = 1.0 if d_point < 2.5 else 0.0
    q[5] = arrival_time
    q[6] = float(wrap_angle(action[t_idx, 2] if t_idx >= 0 else 0.0))
    q[7] = ex
    q[8] = 1.0 if ex >= 0.0 else -1.0

    if typ in (EvidenceType.DYNAMIC_AGENT, EvidenceType.CONFLICT_POINT, EvidenceType.LOW_TTC_RISK):
        agent_xy, agent_valid, agent_yaw = _dynamic_agent_sequence(feature, action.shape[0], dt, future_agents, future_agent_mask)
        length = max(float(feature[5]), 4.5)
        width = max(float(feature[6]), 2.0)
        signed_dist, ai, aj, overlap_sum, overlap_max = _footprint_distance_and_overlap(action, agent_xy, agent_valid, agent_yaw, length, width)
        q[0] = min(signed_dist, 99.0)
        q[1] = float(ai + 1) * dt if ai >= 0 else 99.0
        q[2] = 1.0 if signed_dist < 0.0 or overlap_max > 0.0 else 0.0
        q[3] = overlap_sum
        q[4] = overlap_max
        q[5] = float(ai + 1) * dt if ai >= 0 else 99.0
        q[9] = 0.0 if ai >= 0 and aj >= 0 else 99.0
        vx, vy = float(feature[3]), float(feature[4])
        safe_i = max(ai, 0)
        safe_j = max(aj, 0)
        ego_v = np.asarray([np.cos(action[safe_i, 2]) * action[safe_i, 3], np.sin(action[safe_i, 2]) * action[safe_i, 3]], dtype=np.float32)
        rel_v = ego_v - np.asarray([vx, vy], dtype=np.float32)
        rel_p = xy[safe_i] - agent_xy[safe_j]
        q[10] = float(np.linalg.norm(rel_v))
        q[11] = float(wrap_angle(action[safe_i, 2] - agent_yaw[safe_j])) if safe_j < len(agent_yaw) else 0.0
        q[12] = 1.0 if ai >= 0 and aj >= 0 and ai < aj else 0.0
        q[13] = 1.0 if ai >= 0 and aj >= 0 and aj < ai else 0.0
        ttc = time_to_collision_1d(rel_p, rel_v, radius=4.0)
        q[14] = min(float(ttc), 99.0)
        q[15] = overlap_max
    elif typ == EvidenceType.GAP:
        side = float(feature[12]) if len(feature) > 12 else 0.0
        front_gap = float(feature[13]) if len(feature) > 13 else 80.0
        rear_gap = float(feature[14]) if len(feature) > 14 else 80.0
        rear_relv = float(feature[15]) if len(feature) > 15 else 0.0
        front_relv = float(feature[16]) if len(feature) > 16 else 0.0
        terminal_y = float(action[-1, 1])
        threshold = 0.5 * abs(side) * 3.5 if abs(side) > 0 else 0.0
        indices = np.where(np.abs(action[:, 1]) > threshold)[0]
        merge_time_idx = int(indices[0]) if len(indices) else 0
        q[16] = front_gap
        q[17] = rear_gap
        q[18] = rear_relv
        q[19] = float(merge_time_idx + 1) * dt
        q[20] = 1.0 if side * terminal_y > 1.0 else 0.0
        q[21] = front_relv
    elif typ == EvidenceType.MAP_RULE:
        rule_code = int(round(float(feature[12]))) if len(feature) > 12 else 0
        lane_side = float(feature[13]) if len(feature) > 13 else 0.0
        max_abs_y = float(np.max(np.abs(action[:, 1])))
        max_speed = float(np.max(speed))
        min_dist = float(np.min(np.linalg.norm(xy - point, axis=-1)))
        q[16] = 1.0 if min_dist < 2.5 else 0.0
        if rule_code == int(MapRuleCode.STOP_LINE):
            q[17] = 1.0 if action[-1, 0] > ex and abs(ey) < 10.0 else 0.0
        elif rule_code == int(MapRuleCode.TRAFFIC_LIGHT_RED):
            q[17] = 1.0 if action[-1, 0] > ex and abs(ey) < 10.0 else 0.0
        elif rule_code == int(MapRuleCode.CROSSWALK):
            q[18] = 1.0 if min_dist < 4.0 else 0.0
        elif rule_code == int(MapRuleCode.DRIVABLE_AREA):
            q[19] = 0.0
        q[19] = max(q[19], max(0.0, max_abs_y - 5.25))
        if rule_code == int(MapRuleCode.LANE_BOUNDARY) or lane_side != 0.0:
            boundary_y = abs(ey) if abs(ey) > 0.1 else 5.25
            side = lane_side if lane_side != 0.0 else (1.0 if ey >= 0.0 else -1.0)
            q[20] = 1.0 if np.any(side * action[:, 1] > boundary_y) else 0.0
        q[21] = max(0.0, max_speed - 13.4)
        q[22] = max(q[22], max(0.0, -float(action[-1, 0])))
        _map_rule_exact(q, feature, action, dt, metadata, route_info, use_future_traffic=use_future_traffic)
    q[23] = 1.0
    return q


def compute_geometry_query(evidence_features: np.ndarray, evidence_type: np.ndarray, actions: np.ndarray,
                           evidence_mask: np.ndarray, action_mask: np.ndarray, dt: float = 0.5,
                           future_agents: np.ndarray | None = None,
                           future_agent_mask: np.ndarray | None = None,
                           evidence_metadata: Sequence[dict[str, Any]] | None = None,
                           route_info: dict[str, Any] | None = None,
                           use_future_traffic: bool = False) -> np.ndarray:
    n, k = evidence_features.shape[0], actions.shape[0]
    out = np.zeros((n, k, QUERY_DIM), dtype=np.float32)
    for i in range(n):
        if not evidence_mask[i]:
            continue
        meta = evidence_metadata[i] if evidence_metadata is not None and i < len(evidence_metadata) else None
        for a in range(k):
            if not action_mask[a]:
                continue
            out[i, a] = query_one(evidence_features[i], int(evidence_type[i]), actions[a], dt, future_agents,
                                  future_agent_mask, metadata=meta, route_info=route_info,
                                  use_future_traffic=use_future_traffic)
    return out


def query_active_score(query: np.ndarray) -> np.ndarray:
    d = query[..., 0]
    intersects = query[..., 2]
    overlap = query[..., 3] + query[..., 15]
    rule = query[..., 17] + query[..., 18] + query[..., 19] + query[..., 20] + query[..., 21] + query[..., 22]
    return (np.exp(-np.clip(d, 0.0, 50.0) / 10.0) + intersects + overlap + rule).astype(np.float32)
