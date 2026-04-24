from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpies.teacher.local_costs import LocalCostWeights, local_teacher_contribution


@dataclass
class TeacherWeights:
    collision: float = 100.0
    overlap_area: float = 50.0
    ttc: float = 20.0
    drivable: float = 50.0
    lane_boundary: float = 20.0
    traffic_light: float = 50.0
    stop_line: float = 30.0
    crosswalk: float = 30.0
    route_progress: float = -5.0
    speed_limit: float = 5.0
    comfort_accel: float = 1.0
    comfort_jerk: float = 0.5
    comfort_curvature: float = 0.5
    imitation_ade: float = 1.0
    imitation_fde: float = 1.0


class TeacherEvaluator:
    def __init__(self, weights: TeacherWeights | None = None, local_weights: LocalCostWeights | None = None):
        self.weights = weights or TeacherWeights()
        self.local_weights = local_weights or LocalCostWeights()

    def evaluate(self, actions: np.ndarray, action_mask: np.ndarray, logged_ego_future: np.ndarray,
                 agent_future: np.ndarray, agent_mask: np.ndarray, evidence_features: np.ndarray,
                 evidence_type: np.ndarray, evidence_mask: np.ndarray, teacher_geometry_query: np.ndarray) -> np.ndarray:
        k = actions.shape[0]
        costs = np.full((k,), 1e6, dtype=np.float32)
        local = local_teacher_contribution(evidence_features, evidence_type, teacher_geometry_query,
                                           evidence_mask, action_mask, self.local_weights)
        local_sum = local.sum(axis=0)
        for a in range(k):
            if not action_mask[a]:
                continue
            traj = actions[a]
            h = min(len(traj), len(logged_ego_future))
            diff = traj[:h, :2] - logged_ego_future[:h, :2]
            dist = np.linalg.norm(diff, axis=-1)
            ade = float(dist.mean()) if h > 0 else 0.0
            fde = float(dist[-1]) if h > 0 else 0.0
            progress = float(traj[-1, 0])
            speed_over = float(np.maximum(traj[:, 3] - 13.4, 0.0).mean())
            accel_cost = float(np.mean(np.maximum(np.abs(traj[:, 4]) - 2.5, 0.0) ** 2))
            jerk = np.gradient(traj[:, 4], 0.5)
            jerk_cost = float(np.mean(np.maximum(np.abs(jerk) - 5.0, 0.0) ** 2))
            curv_cost = float(np.mean(np.maximum(np.abs(traj[:, 5]) - 0.25, 0.0) ** 2))
            drivable = float(np.maximum(np.max(np.abs(traj[:, 1])) - 5.25, 0.0))
            # Direct logged-future collision/proximity proxy.
            col = 0.0
            prox = 0.0
            if agent_future.size > 0 and agent_mask.any():
                valid = np.where(agent_mask)[0]
                for idx in valid:
                    fut = agent_future[idx, :len(traj), :2]
                    valid_fut = np.any(np.abs(fut) > 1e-4, axis=-1)
                    if not valid_fut.any():
                        continue
                    dd = np.linalg.norm(traj[:len(fut), :2] - fut, axis=-1)
                    m = float(dd[valid_fut].min())
                    if m < 2.2:
                        col += 1.0 + (2.2 - m)
                    prox += np.exp(-m / 3.0)
            cost = 0.0
            w = self.weights
            cost += w.imitation_ade * ade + w.imitation_fde * fde
            cost += w.route_progress * (progress / 50.0)
            cost += w.speed_limit * speed_over
            cost += w.comfort_accel * accel_cost + w.comfort_jerk * jerk_cost + w.comfort_curvature * curv_cost
            cost += w.drivable * drivable
            cost += w.collision * col + w.ttc * prox
            cost += float(local_sum[a])
            costs[a] = float(cost)
        return costs
