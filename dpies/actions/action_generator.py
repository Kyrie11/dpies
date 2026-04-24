from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from dpies.actions.action_filter import diversity_filter
from dpies.actions.rollout import action_feasible, kinematic_rollout, stop_rollout
from dpies.common.types import ACTION_META_DIM, ACTION_STATE_DIM, ActionMode


@dataclass
class ActionGeneratorConfig:
    max_actions: int = 32
    horizon_s: float = 8.0
    dt: float = 0.5
    speed_limit_mps: float = 13.4
    max_accel: float = 3.0
    min_accel: float = -4.0
    max_curvature: float = 0.35


class ActionGenerator:
    def __init__(self, cfg: ActionGeneratorConfig):
        self.cfg = cfg

    def _make(self, mode: ActionMode, traj: np.ndarray, target_speed: float, terminal_lateral: float, progress: float) -> dict:
        return {
            "mode": int(mode),
            "trajectory": traj.astype(np.float32),
            "meta": np.asarray([
                int(mode), target_speed, terminal_lateral, progress,
                float(traj[-1, 0]), float(traj[-1, 1]), float(traj[-1, 3]), self.cfg.horizon_s,
            ], dtype=np.float32),
        }

    def generate(self, ego_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return padded action trajectories, meta, and validity mask.

        ego_history is ego-centric [H, 8]; the last row is current ego state,
        whose last value stores a speed proxy from the DB reader.
        """
        cur = ego_history[-1]
        current_speed = float(max(cur[7], np.linalg.norm(cur[3:5])))
        if current_speed < 0.05:
            current_speed = float(np.linalg.norm(cur[3:5]))
        speed_candidates = sorted(set([
            0.0,
            max(0.0, current_speed - 3.0),
            max(0.0, current_speed - 1.0),
            current_speed,
            min(self.cfg.speed_limit_mps, current_speed + 1.5),
            min(self.cfg.speed_limit_mps, current_speed + 3.0),
            min(self.cfg.speed_limit_mps, 8.0),
        ]))
        actions: List[dict] = []
        # Keep/follow and proceed candidates.
        for vtar in speed_candidates:
            for progress in (20.0, 35.0, 50.0):
                traj = kinematic_rollout(current_speed, vtar, 0.0, self.cfg.horizon_s, self.cfg.dt, progress)
                if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                    mode = ActionMode.PROCEED if vtar > current_speed + 1.0 else ActionMode.KEEP
                    actions.append(self._make(mode, traj, vtar, 0.0, progress))
        # Yield/stop and creep.
        for stop_d in (4.0, 8.0, 12.0, 18.0):
            traj = stop_rollout(current_speed, stop_d, self.cfg.horizon_s, self.cfg.dt)
            if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                actions.append(self._make(ActionMode.STOP, traj, 0.0, 0.0, stop_d))
        for progress in (5.0, 10.0, 15.0):
            traj = kinematic_rollout(current_speed, min(2.0, self.cfg.speed_limit_mps), 0.0, self.cfg.horizon_s, self.cfg.dt, progress)
            actions.append(self._make(ActionMode.CREEP, traj, 2.0, 0.0, progress))
        # Lane changes and nudges. They are intentionally kept even if later
        # rule/evidence modules penalize them.
        for lat, mode in ((3.5, ActionMode.LANE_CHANGE_LEFT), (-3.5, ActionMode.LANE_CHANGE_RIGHT)):
            for vtar in (max(1.0, current_speed), min(self.cfg.speed_limit_mps, current_speed + 2.0)):
                for progress in (25.0, 40.0, 55.0):
                    traj = kinematic_rollout(current_speed, vtar, lat, self.cfg.horizon_s, self.cfg.dt, progress)
                    if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                        actions.append(self._make(mode, traj, vtar, lat, progress))
        for lat, mode in ((1.0, ActionMode.NUDGE_LEFT), (-1.0, ActionMode.NUDGE_RIGHT)):
            for progress in (20.0, 35.0):
                traj = kinematic_rollout(current_speed, max(2.0, current_speed), lat, self.cfg.horizon_s, self.cfg.dt, progress)
                if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                    actions.append(self._make(mode, traj, current_speed, lat, progress))
        if not actions:
            traj = stop_rollout(current_speed, 5.0, self.cfg.horizon_s, self.cfg.dt)
            actions.append(self._make(ActionMode.STOP, traj, 0.0, 0.0, 5.0))
        actions = diversity_filter(actions, self.cfg.max_actions)
        k = self.cfg.max_actions
        steps = int(round(self.cfg.horizon_s / self.cfg.dt))
        trajs = np.zeros((k, steps, ACTION_STATE_DIM), dtype=np.float32)
        meta = np.zeros((k, ACTION_META_DIM), dtype=np.float32)
        mask = np.zeros((k,), dtype=bool)
        for i, a in enumerate(actions[:k]):
            trajs[i] = a["trajectory"]
            meta[i] = a["meta"]
            mask[i] = True
        return trajs, meta, mask
