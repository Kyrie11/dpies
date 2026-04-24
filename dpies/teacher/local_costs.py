from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpies.common.types import EvidenceType


@dataclass
class LocalCostWeights:
    lambda_overlap: float = 20.0
    lambda_distance: float = 5.0
    sigma_distance: float = 2.0
    lambda_ttc: float = 10.0
    tau0_ttc: float = 3.0
    lambda_yield: float = 10.0
    front_safe_distance: float = 8.0
    rear_safe_distance: float = 10.0
    lambda_front: float = 3.0
    lambda_rear: float = 4.0
    lambda_rear_rel_speed: float = 2.0
    lambda_drivable: float = 30.0
    lambda_lane_boundary: float = 10.0
    lambda_red_light: float = 50.0
    lambda_stop_line: float = 20.0
    lambda_crosswalk: float = 20.0
    lambda_speed_limit: float = 5.0


def local_teacher_contribution(evidence_features: np.ndarray, evidence_type: np.ndarray, geometry_query: np.ndarray,
                               evidence_mask: np.ndarray, action_mask: np.ndarray,
                               weights: LocalCostWeights | None = None) -> np.ndarray:
    w = weights or LocalCostWeights()
    n, k = geometry_query.shape[:2]
    out = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        if not evidence_mask[i]:
            continue
        p = float(np.clip(evidence_features[i, 10], 0.0, 2.0))
        typ = int(evidence_type[i])
        for a in range(k):
            if not action_mask[a]:
                continue
            q = geometry_query[i, a]
            cost = 0.0
            if typ in (EvidenceType.DYNAMIC_AGENT, EvidenceType.CONFLICT_POINT, EvidenceType.LOW_TTC_RISK):
                d = max(float(q[0]), 0.0)
                overlap = float(q[3] + q[15])
                ttc = float(q[14])
                cost += w.lambda_overlap * overlap
                cost += w.lambda_distance * np.exp(-d / max(w.sigma_distance, 1e-3))
                if ttc < w.tau0_ttc:
                    cost += w.lambda_ttc * max((w.tau0_ttc - ttc) / w.tau0_ttc, 0.0)
                # Precedence proxy: ego arrives much earlier and too close.
                if q[12] > 0.5 and q[0] < 2.0:
                    cost += w.lambda_yield
            elif typ == EvidenceType.GAP:
                front_gap = float(q[16])
                rear_gap = float(q[17])
                rear_relv = float(q[18])
                applies = float(q[20])
                cost += applies * w.lambda_front * max(w.front_safe_distance - front_gap, 0.0)
                cost += applies * w.lambda_rear * max(w.rear_safe_distance - rear_gap, 0.0)
                cost += applies * w.lambda_rear_rel_speed * max(-rear_relv, 0.0)
            elif typ == EvidenceType.MAP_RULE:
                stop_cross = float(q[17])
                crosswalk = float(q[18])
                drv = float(q[19])
                lane = float(q[20])
                speed = float(q[21])
                cost += w.lambda_stop_line * stop_cross
                cost += w.lambda_crosswalk * crosswalk
                cost += w.lambda_drivable * drv
                cost += w.lambda_lane_boundary * lane
                cost += w.lambda_speed_limit * speed
            out[i, a] = p * float(cost)
    return out
