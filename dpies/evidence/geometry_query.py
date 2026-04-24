from __future__ import annotations

import numpy as np

from dpies.common.geometry import box_corners, pairwise_min_distance, polygon_overlap_area, time_to_collision_1d, wrap_angle
from dpies.common.types import EvidenceType, QUERY_DIM


def _dynamic_agent_positions(feature: np.ndarray, action_steps: int, dt: float, future_agents: np.ndarray | None = None) -> np.ndarray:
    idx = int(round(float(feature[11])))
    if future_agents is not None and idx >= 0 and idx < future_agents.shape[0]:
        fut = future_agents[idx, :action_steps, :2]
        valid = np.any(np.abs(fut) > 1e-4, axis=-1)
        if valid.any():
            return fut.astype(np.float32)
    x, y, vx, vy = float(feature[1]), float(feature[2]), float(feature[3]), float(feature[4])
    t = np.arange(1, action_steps + 1, dtype=np.float32) * dt
    return np.stack([x + vx * t, y + vy * t], axis=-1).astype(np.float32)


def query_one(feature: np.ndarray, type_id: int, action: np.ndarray, dt: float = 0.5,
              future_agents: np.ndarray | None = None) -> np.ndarray:
    q = np.zeros((QUERY_DIM,), dtype=np.float32)
    xy = action[:, :2]
    speed = action[:, 3]
    typ = EvidenceType(int(type_id)) if int(type_id) in [e.value for e in EvidenceType] else EvidenceType.PADDING
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
        agent_xy = _dynamic_agent_positions(feature, action.shape[0], dt, future_agents)
        dmin, ai, aj = pairwise_min_distance(xy, agent_xy)
        length = max(float(feature[5]), 4.5)
        width = max(float(feature[6]), 2.0)
        ego_radius = 2.4
        agent_radius = max(1.0, 0.25 * (length + width))
        signed_dist = float(dmin) - ego_radius - agent_radius
        q[0] = min(signed_dist, 99.0)
        q[1] = float(ai + 1) * dt if ai >= 0 else 99.0
        q[2] = 1.0 if signed_dist < 0.0 else 0.0
        q[3] = max(0.0, -signed_dist)
        q[4] = max(0.0, -signed_dist)
        q[5] = float(ai + 1) * dt if ai >= 0 else 99.0
        q[9] = float(abs(ai - aj) * dt) if ai >= 0 and aj >= 0 else 99.0
        vx, vy = float(feature[3]), float(feature[4])
        ego_v = np.asarray([np.cos(action[max(ai, 0), 2]) * action[max(ai, 0), 3], np.sin(action[max(ai, 0), 2]) * action[max(ai, 0), 3]], dtype=np.float32)
        rel_v = ego_v - np.asarray([vx, vy], dtype=np.float32)
        rel_p = xy[max(ai, 0)] - agent_xy[max(aj, 0)]
        q[10] = float(np.linalg.norm(rel_v))
        q[11] = float(wrap_angle(action[max(ai, 0), 2] - np.arctan2(vy, vx))) if abs(vx) + abs(vy) > 1e-3 else 0.0
        q[12] = 1.0 if ai < aj else 0.0
        q[13] = 1.0 if aj < ai else 0.0
        ttc = time_to_collision_1d(rel_p, rel_v, radius=4.0)
        q[14] = min(float(ttc), 99.0)
        # Approximate footprint overlap at closest time if shapely is available.
        if ai >= 0 and aj >= 0:
            ego_poly = box_corners(float(xy[ai, 0]), float(xy[ai, 1]), float(action[ai, 2]), 4.8, 2.1)
            ag_poly = box_corners(float(agent_xy[aj, 0]), float(agent_xy[aj, 1]), 0.0, length, width)
            q[15] = polygon_overlap_area(ego_poly, ag_poly)
    elif typ == EvidenceType.GAP:
        side = float(feature[12]) if len(feature) > 12 else 0.0
        front_gap = float(feature[13]) if len(feature) > 13 else 80.0
        rear_gap = float(feature[14]) if len(feature) > 14 else 80.0
        rear_relv = float(feature[15]) if len(feature) > 15 else 0.0
        terminal_y = float(action[-1, 1])
        merge_time_idx = int(np.argmax(np.abs(action[:, 1]) > 0.5 * abs(side) * 3.5)) if abs(side) > 0 else 0
        q[16] = front_gap
        q[17] = rear_gap
        q[18] = rear_relv
        q[19] = float(merge_time_idx + 1) * dt
        q[20] = 1.0 if side * terminal_y > 1.0 else 0.0
    elif typ == EvidenceType.MAP_RULE:
        rule_code = float(feature[12]) if len(feature) > 12 else 0.0
        lane_boundary = float(feature[13]) if len(feature) > 13 else 0.0
        max_abs_y = float(np.max(np.abs(action[:, 1])))
        max_speed = float(np.max(speed))
        q[16] = 1.0 if np.min(np.linalg.norm(xy - point, axis=-1)) < 2.5 else 0.0
        q[17] = 1.0 if rule_code == 1.0 and action[-1, 0] > ex else 0.0  # stop-line crossing proxy
        q[18] = 1.0 if rule_code == 2.0 and np.min(np.linalg.norm(xy - point, axis=-1)) < 4.0 else 0.0
        q[19] = max(0.0, max_abs_y - 5.25)
        q[20] = 1.0 if lane_boundary != 0.0 and np.any(lane_boundary * action[:, 1] > abs(ey)) else 0.0
        q[21] = max(0.0, max_speed - 13.4)
        q[22] = max(0.0, -float(action[-1, 0]))
    q[23] = 1.0
    return q


def compute_geometry_query(evidence_features: np.ndarray, evidence_type: np.ndarray, actions: np.ndarray,
                           evidence_mask: np.ndarray, action_mask: np.ndarray, dt: float = 0.5,
                           future_agents: np.ndarray | None = None) -> np.ndarray:
    n, k = evidence_features.shape[0], actions.shape[0]
    out = np.zeros((n, k, QUERY_DIM), dtype=np.float32)
    for i in range(n):
        if not evidence_mask[i]:
            continue
        for a in range(k):
            if not action_mask[a]:
                continue
            out[i, a] = query_one(evidence_features[i], int(evidence_type[i]), actions[a], dt, future_agents)
    return out


def query_active_score(query: np.ndarray) -> np.ndarray:
    """Return [N,K] scalar activity used for active masking."""
    d = query[..., 0]
    intersects = query[..., 2]
    overlap = query[..., 3] + query[..., 15]
    rule = query[..., 17] + query[..., 18] + query[..., 19] + query[..., 20] + query[..., 21]
    return (np.exp(-np.clip(d, 0.0, 50.0) / 10.0) + intersects + overlap + rule).astype(np.float32)
