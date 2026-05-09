from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dpies.teacher.local_costs import LocalCostWeights, local_teacher_contribution


@dataclass
class TeacherWeights:
    route_progress: float = -18.0
    speed_limit: float = 5.0
    comfort_accel: float = 1.0
    comfort_jerk: float = 0.5
    comfort_curvature: float = 0.5
    imitation_ade: float = 0.25
    imitation_fde: float = 0.25
    local_evidence: float = 0.08

    # Relative imitation/progress still matters, but should not be the only
    # signal deciding whether the oracle creeps or moves.
    low_progress: float = 30.0
    stop_when_should_move: float = 120.0

    # Absolute motion priors.  These are gated by per-action future risk so
    # they discourage unnecessary stop/creep without forcing blind rushing.
    absolute_progress_floor: float = 22.0
    absolute_progress_weight: float = 120.0
    speed_floor: float = 2.5
    speed_floor_weight: float = 18.0
    target_speed: float = 7.0
    target_speed_weight: float = 2.0
    excessive_progress_cap: float = 70.0
    excessive_progress: float = 8.0
    hard_comfort: float = 8.0

    future_collision: float = 60.0
    future_proximity: float = 8.0
    collision_radius_m: float = 2.2
    proximity_sigma_m: float = 3.0
    risk_gate_proximity_scale: float = 0.05


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
        # columns:
        # ade, fde, progress_norm, speed_over, accel, jerk, curvature,
        # future_collision, future_proximity, local_sum,
        # rel_low_progress, abs_low_progress, speed_floor, target_speed, excessive_progress,
        # move_gate, global_without_local, total
        components = np.zeros((k, 18), dtype=np.float32)

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

        expert_final_speed = 0.0
        if len(logged_ego_future) >= 2:
            diffs = np.diff(logged_ego_future[:, :2], axis=0)
            expert_final_speed = float(np.linalg.norm(diffs[-1]) / max(dt, 1e-3))

        valid_progress = action_progress[valid_idx]
        max_valid_progress = float(np.max(valid_progress)) if len(valid_progress) else 0.0

        # Do not use the logged future alone to decide whether the ego should move:
        # nuPlan logs often contain slow/creep segments, and blindly imitating them
        # creates a stop-biased oracle.  If the candidate set contains reasonable
        # motion, low-progress candidates should pay a cost unless that candidate
        # itself has high future risk.
        candidate_has_move = max_valid_progress >= 25.0
        expert_suggests_move = expert_progress > 6.0 or expert_final_speed > 1.0
        should_move = float(candidate_has_move and (expert_suggests_move or max_valid_progress >= 35.0))

        # Adaptive floor: require moderate progress, but cap the floor so that the
        # teacher does not chase the very aggressive 120m/180m actions.
        base_floor = float(getattr(w, "absolute_progress_floor", 22.0))
        adaptive_floor = min(35.0, max(base_floor, 0.45 * max_valid_progress))
        abs_low_progress_cost = np.maximum(adaptive_floor - action_progress, 0.0) / max(adaptive_floor, 1e-3)
        abs_low_progress_cost = abs_low_progress_cost ** 2

        # Per-action risk gate.  Collision can fully suppress the move prior;
        # proximity only weakly suppresses it, otherwise "there is a nearby car"
        # makes the oracle creep in dense but normal traffic.
        risk_gate = np.clip(
            future_col + float(getattr(w, "risk_gate_proximity_scale", 0.05)) * future_prox,
            0.0,
            1.0,
        )
        move_gate = should_move * (1.0 - risk_gate)
        abs_low_progress_cost = move_gate * abs_low_progress_cost

        # Speed floor discourages stopping when movement is available.  The target
        # speed term is intentionally weak and centered at a city-driving speed;
        # hard comfort/speed-limit terms handle the over-aggressive actions.
        speed_floor = float(getattr(w, "speed_floor", 2.5))
        speed_floor_cost = move_gate * (np.maximum(speed_floor - final_speed, 0.0) / max(speed_floor, 1e-3)) ** 2

        target_speed = float(getattr(w, "target_speed", 7.0))
        target_speed_cost = move_gate * ((final_speed - target_speed) / max(target_speed, 1e-3)) ** 2

        excessive_progress_cap = float(getattr(w, "excessive_progress_cap", 70.0))
        excessive_progress_cost = (np.maximum(action_progress - excessive_progress_cap, 0.0) / max(excessive_progress_cap, 1e-3)) ** 2

        stop_when_should_move_cost = move_gate * np.maximum(1.0 - final_speed, 0.0) ** 2

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
                + w.absolute_progress_weight * abs_low_progress_cost
                + w.speed_floor_weight * speed_floor_cost
                + w.stop_when_should_move * stop_when_should_move_cost
                + w.target_speed_weight * target_speed_cost
                + w.excessive_progress * excessive_progress_cost
                + w.hard_comfort * hard_comfort_cost
        )

        total = global_without_local + w.local_evidence * local_sum
        costs[valid_idx] = total[valid_idx].astype(np.float32)
        components[:, :] = np.stack([
            ade_all, fde_all, progress_all, speed_over_all, accel_cost_all,
            jerk_cost_all, curv_cost_all, future_col, future_prox, local_sum,
            low_progress_cost, abs_low_progress_cost, speed_floor_cost, target_speed_cost, excessive_progress_cost,
            move_gate, global_without_local, total,
        ], axis=1).astype(np.float32)
        return costs, components
