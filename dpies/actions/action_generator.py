from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from dpies.actions.action_filter import diversity_filter
from dpies.actions.rollout import action_feasible, kinematic_rollout, stop_rollout
from dpies.common.types import ACTION_META_DIM, ACTION_STATE_DIM, ActionMode, MapRuleCode


@dataclass
class ActionGeneratorConfig:
    max_actions: int = 32
    horizon_s: float = 8.0
    dt: float = 0.5
    speed_limit_mps: float = 13.4
    max_accel: float = 3.0
    min_accel: float = -4.0
    max_curvature: float = 0.35
    lane_width_m: float = 3.5
    allow_topology_fallback_lane_changes: bool = True


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

    @staticmethod
    def _current_speed(ego_history: np.ndarray) -> float:
        cur = ego_history[-1]
        speed_xy = float(np.linalg.norm(cur[3:5])) if cur.shape[-1] >= 5 else 0.0
        speed_col = float(cur[8]) if cur.shape[-1] >= 9 else speed_xy
        # Column 7 is yaw_rate in the new schema and must not be used as speed.
        return max(speed_xy, speed_col, 0.0)

    def _stop_distances_from_rules(self, rule_units: list[dict] | None) -> list[float]:
        stops: list[float] = []
        for ru in rule_units or []:
            code = int(ru.get("rule_code", 0))
            if code in (int(MapRuleCode.STOP_LINE), int(MapRuleCode.TRAFFIC_LIGHT_RED)):
                xy = np.asarray(ru.get("xy", [0.0, 0.0]), dtype=np.float32)
                x, y = float(xy[0]), float(xy[1])
                if 1.0 < x < 60.0 and abs(y) < 8.0:
                    stops.append(max(1.0, x - 2.0))
        if not stops:
            stops = [4.0, 8.0, 12.0, 18.0]
        return sorted(set(round(float(s), 2) for s in stops))[:6]

    def _lane_change_sides(self, rule_units: list[dict] | None) -> list[tuple[float, ActionMode]]:
        # Without reliable topology, keep old fallback lane-change candidates. If map
        # boundary tokens exist very close to one side, avoid the grossly invalid side.
        if self.cfg.allow_topology_fallback_lane_changes:
            return [(self.cfg.lane_width_m, ActionMode.LANE_CHANGE_LEFT), (-self.cfg.lane_width_m, ActionMode.LANE_CHANGE_RIGHT)]
        return []

    def generate(self, ego_history: np.ndarray, agent_history: np.ndarray | None = None, agent_mask: np.ndarray | None = None,
                 map_context: Any | None = None, rule_units: list[dict] | None = None,
                 traffic_lights: Any | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return padded action trajectories, meta, and validity mask.

        New ego schema: [x,y,yaw,vx,vy,ax,ay,yaw_rate,speed]. The generator accepts
        optional map/rule inputs so preprocessing can stop at real stop lines when
        available while preserving a map-free fallback for smoke tests.
        """
        current_speed = self._current_speed(ego_history)
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
        for vtar in speed_candidates:
            for progress in (20.0, 35.0, 50.0):
                traj = kinematic_rollout(current_speed, vtar, 0.0, self.cfg.horizon_s, self.cfg.dt, progress)
                if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                    mode = ActionMode.PROCEED if vtar > current_speed + 1.0 else ActionMode.KEEP
                    actions.append(self._make(mode, traj, vtar, 0.0, progress))
        for stop_d in self._stop_distances_from_rules(rule_units):
            traj = stop_rollout(current_speed, stop_d, self.cfg.horizon_s, self.cfg.dt)
            if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                actions.append(self._make(ActionMode.STOP, traj, 0.0, 0.0, stop_d))
        for progress in (5.0, 10.0, 15.0):
            traj = kinematic_rollout(current_speed, min(2.0, self.cfg.speed_limit_mps), 0.0, self.cfg.horizon_s, self.cfg.dt, progress)
            if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                actions.append(self._make(ActionMode.CREEP, traj, 2.0, 0.0, progress))
        for lat, mode in self._lane_change_sides(rule_units):
            for vtar in (max(1.0, current_speed), min(self.cfg.speed_limit_mps, current_speed + 2.0)):
                for progress in (25.0, 40.0, 55.0):
                    traj = kinematic_rollout(current_speed, vtar, lat, self.cfg.horizon_s, self.cfg.dt, progress)
                    if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                        actions.append(self._make(mode, traj, vtar, lat, progress))
        # Lightweight merge candidates: lateral transition toward the same adjacent-lane geometry,
        # tagged separately so gap evidence can learn merge-specific preferences.
        for lat in (self.cfg.lane_width_m, -self.cfg.lane_width_m):
            for progress in (30.0, 45.0):
                traj = kinematic_rollout(current_speed, max(2.0, current_speed), lat, self.cfg.horizon_s, self.cfg.dt, progress)
                if action_feasible(traj, self.cfg.max_accel, self.cfg.min_accel, self.cfg.max_curvature):
                    actions.append(self._make(ActionMode.MERGE, traj, max(2.0, current_speed), lat, progress))
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
