from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpies.teacher.local_costs import LocalCostWeights, local_teacher_contribution


@dataclass
class TeacherWeights:
    route_progress: float = -35.0
    speed_limit: float = 5.0
    comfort_accel: float = 1.0
    comfort_jerk: float = 0.5
    comfort_curvature: float = 0.5

    imitation_ade: float = 0.25
    imitation_fde: float = 0.25

    local_evidence: float = 0.05

    low_progress: float = 20.0
    stop_when_should_move: float = 60.0
    hard_comfort: float = 5.0

    future_collision: float = 50.0
    future_proximity: float = 10.0

    absolute_low_progress: float = 80.0
    progress_floor_m: float = 20.0
    progress_floor_speed_frac: float = 0.65
    progress_gate_risk_margin: float = 80.0


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

        max_abs_accel = np.max(np.abs(actions[:, :, 4]), axis=1)

        jerk = np.gradient(actions[:, :, 4], dt, axis=1)
        max_abs_jerk = np.max(np.abs(jerk), axis=1)

        max_abs_curv = np.max(np.abs(actions[:, :, 5]), axis=1)

        hard_comfort_cost = (
                np.maximum(max_abs_accel - 3.0, 0.0) ** 2
                + np.maximum(max_abs_jerk - 5.0, 0.0) ** 2
                + np.maximum(max_abs_curv - 0.25, 0.0) ** 2
        ).astype(np.float32)

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

        expert_progress = float(max(logged_ego_future[-1, 0], 1.0)) if len(logged_ego_future) else 1.0
        action_progress = actions[:, -1, 0]
        final_speed = actions[:, -1, 3]

        progress_ratio = action_progress / max(expert_progress, 1e-3)

        low_progress_cost = np.maximum(0.75 - progress_ratio, 0.0) ** 2

        # 如果 log future 明显在走，candidate 却几乎停住，强惩罚
        expert_final_speed = 0.0
        if len(logged_ego_future) >= 2:
            diffs = np.diff(logged_ego_future[:, :2], axis=0)
            expert_final_speed = float(np.linalg.norm(diffs[-1]) / max(dt, 1e-3))

        current_speed_est = float(np.percentile(actions[valid_idx, 0, 3], 80))
        horizon_s = float(actions.shape[1] * dt)

        risk_gate = (
                local_sum
                + 200.0 * future_col
                + 30.0 * future_prox
                + 50.0 * (hard_comfort_cost > 0.0).astype(np.float32)
        )

        min_risk = float(np.min(risk_gate[valid_idx]))

        safe_move_exists = bool(np.any(
            action_mask
            & (action_progress >= w.progress_floor_m)
            & (risk_gate <= min_risk + w.progress_gate_risk_margin)
        ))

        log_should_move = bool(expert_progress > 10.0 or expert_final_speed > 1.5)
        should_move = float(log_should_move or safe_move_exists)

        desired_progress = max(
            w.progress_floor_m,
            w.progress_floor_speed_frac * current_speed_est * horizon_s,
            0.75 * expert_progress,
        )
        desired_progress = min(desired_progress, 60.0)

        absolute_low_progress_cost = should_move * (
                np.maximum(desired_progress - action_progress, 0.0)
                / max(desired_progress, 1.0)
        ) ** 2

        stop_when_should_move_cost = should_move * np.maximum(1.0 - final_speed, 0.0) ** 2

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
                + w.low_progress * low_progress_cost
                + w.stop_when_should_move * stop_when_should_move_cost
                + w.hard_comfort * hard_comfort_cost
                + w.absolute_low_progress * absolute_low_progress_cost
        )

        total = global_without_local + w.local_evidence * local_sum
        costs[valid_idx] = total[valid_idx].astype(np.float32)
        components[:, :] = np.stack([
            ade_all, fde_all, progress_all, speed_over_all, accel_cost_all,
            jerk_cost_all, curv_cost_all, future_col, future_prox, local_sum,
            global_without_local, total,
        ], axis=1).astype(np.float32)
        return costs, components
