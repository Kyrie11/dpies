from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpies.teacher.local_costs import LocalCostWeights, local_teacher_contribution


@dataclass
class TeacherWeights:
    """Weights for the candidate-set hindsight teacher.

    The teacher is deliberately split into a global oracle cost and a local
    evidence cost.  By default direct future-agent collision/proximity is off
    because dynamic/conflict evidence already covers the same interaction risk;
    this avoids double counting and removes an O(K*A*T) preprocessing loop.
    """
    route_progress: float = -5.0
    speed_limit: float = 5.0
    comfort_accel: float = 1.0
    comfort_jerk: float = 0.5
    comfort_curvature: float = 0.5
    imitation_ade: float = 1.0
    imitation_fde: float = 1.0
    local_evidence: float = 1.0

    # Optional legacy direct future-agent penalties. Keep disabled in the main
    # implementation unless running an ablation that intentionally double checks
    # future-agent distance outside evidence units.
    future_collision: float = 0.0
    future_proximity: float = 0.0
    collision_radius_m: float = 2.2
    proximity_sigma_m: float = 3.0


class TeacherEvaluator:
    def __init__(self, weights: TeacherWeights | None = None, local_weights: LocalCostWeights | None = None, dt: float = 0.5):
        self.weights = weights or TeacherWeights()
        self.local_weights = local_weights or LocalCostWeights()
        self.dt = float(dt)

    def evaluate(self, actions: np.ndarray, action_mask: np.ndarray, logged_ego_future: np.ndarray,
                 agent_future: np.ndarray, agent_mask: np.ndarray, evidence_features: np.ndarray,
                 evidence_type: np.ndarray, evidence_mask: np.ndarray, teacher_geometry_query: np.ndarray,
                 agent_future_mask: np.ndarray | None = None, dt: float | None = None,
                 local_cost: np.ndarray | None = None) -> np.ndarray:
        costs, _ = self.evaluate_with_components(actions, action_mask, logged_ego_future, agent_future, agent_mask,
                                                  evidence_features, evidence_type, evidence_mask, teacher_geometry_query,
                                                  agent_future_mask=agent_future_mask, dt=dt, local_cost=local_cost)
        return costs

    def evaluate_with_components(self, actions: np.ndarray, action_mask: np.ndarray, logged_ego_future: np.ndarray,
                                 agent_future: np.ndarray, agent_mask: np.ndarray, evidence_features: np.ndarray,
                                 evidence_type: np.ndarray, evidence_mask: np.ndarray, teacher_geometry_query: np.ndarray,
                                 agent_future_mask: np.ndarray | None = None, dt: float | None = None,
                                 local_cost: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        dt = float(self.dt if dt is None else dt)
        k = actions.shape[0]
        costs = np.full((k,), 1e6, dtype=np.float32)
        # columns: imitation_ade, imitation_fde, progress_reward, speed, accel,
        # jerk, curvature, future_collision, future_proximity, local_sum,
        # global_without_local, total
        components = np.zeros((k, 12), dtype=np.float32)

        if local_cost is None:
            local_cost = local_teacher_contribution(
                evidence_features, evidence_type, teacher_geometry_query,
                evidence_mask, action_mask, self.local_weights,
            )
        local_sum = local_cost.sum(axis=0).astype(np.float32)

        valid_idx = np.where(action_mask)[0]
        if len(valid_idx) == 0:
            return costs, components

        h = min(actions.shape[1], len(logged_ego_future))
        if h > 0:
            diff = actions[:, :h, :2] - logged_ego_future[:h, :2][None, :, :]
            dist = np.linalg.norm(diff, axis=-1)
            ade_all = dist.mean(axis=1)
            fde_all = dist[:, -1]
        else:
            ade_all = np.zeros((k,), dtype=np.float32)
            fde_all = np.zeros((k,), dtype=np.float32)

        progress_all = actions[:, -1, 0] / 50.0
        speed_over_all = np.maximum(actions[:, :, 3] - 13.4, 0.0).mean(axis=1)
        accel_cost_all = np.maximum(np.abs(actions[:, :, 4]) - 2.5, 0.0) ** 2
        accel_cost_all = accel_cost_all.mean(axis=1)
        jerk = np.gradient(actions[:, :, 4], dt, axis=1)
        jerk_cost_all = (np.maximum(np.abs(jerk) - 5.0, 0.0) ** 2).mean(axis=1)
        curv_cost_all = (np.maximum(np.abs(actions[:, :, 5]) - 0.25, 0.0) ** 2).mean(axis=1)

        # Optional legacy direct future-agent check. Disabled by default because
        # it duplicates dynamic/conflict local evidence and is often one of the
        # heavier preprocessing loops.
        future_col = np.zeros((k,), dtype=np.float32)
        future_prox = np.zeros((k,), dtype=np.float32)
        w = self.weights

        if (w.future_collision != 0.0 or w.future_proximity != 0.0) and agent_future.size > 0 and agent_mask.any():
            valid_agents = np.where(agent_mask)[0]
            for idx in valid_agents:
                fut = agent_future[idx, :actions.shape[1], :2]
                if agent_future_mask is not None and idx < agent_future_mask.shape[0]:
                    valid_fut = agent_future_mask[idx, :actions.shape[1]].astype(bool)
                else:
                    valid_fut = np.any(np.abs(fut) > 1e-4, axis=-1)
                if not valid_fut.any():
                    continue
                d = np.linalg.norm(actions[:, :len(fut), :2] - fut[None, :, :], axis=-1)
                d_valid = np.where(valid_fut[None, :], d, np.inf)
                m = d_valid.min(axis=1)
                future_col += np.where(m < w.collision_radius_m, 1.0 + (w.collision_radius_m - m), 0.0)
                future_prox += np.exp(-m / max(w.proximity_sigma_m, 1e-3)).astype(np.float32)

        global_without_local = (
                w.imitation_ade * ade_all
                + w.imitation_fde * fde_all
                + w.route_progress * progress_all
                + w.speed_limit * speed_over_all
                + w.comfort_accel * accel_cost_all
                + w.comfort_jerk * jerk_cost_all
                + w.comfort_curvature * curv_cost_all
                + w.future_collision * future_col
                + w.future_proximity * future_prox
        )
        total = global_without_local + w.local_evidence * local_sum
        costs[valid_idx] = total[valid_idx].astype(np.float32)
        components[:, :] = np.stack([
            ade_all, fde_all, progress_all, speed_over_all, accel_cost_all,
            jerk_cost_all, curv_cost_all, future_col, future_prox, local_sum,
            global_without_local, total,
        ], axis=1).astype(np.float32)
        return costs, components
