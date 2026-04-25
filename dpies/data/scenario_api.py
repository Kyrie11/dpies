from __future__ import annotations

"""Optional official nuPlan Scenario API integration.

This module is intentionally defensive: it is imported in environments without
nuPlan devkit for unit/smoke tests, but only does real work when the official
v1.1 devkit is installed.  It avoids guessing traffic-light/route SQL schemas by
using NuPlanScenario methods such as get_route_roadblock_ids(),
get_traffic_light_status_at_iteration(), and
get_future_traffic_light_status_history().
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from dpies.common.geometry import global_to_ego_points, transform_agent_state_global_to_ego, transform_ego_state_global_to_ego
from dpies.data.nuplan_db import stable_int


@dataclass
class ScenarioAPIContext:
    map_api: Any | None = None
    route_roadblock_ids: list[str] = field(default_factory=list)
    traffic_lights_current: list[Any] = field(default_factory=list)
    traffic_lights_future: list[list[Any]] = field(default_factory=list)
    mission_goal: Any | None = None
    scenario: Any | None = None
    error: str = ""


def traffic_light_to_dict(tl: Any) -> dict[str, Any]:
    if isinstance(tl, dict):
        status = tl.get("status", tl.get("status_name", "UNKNOWN"))
        status_name = str(getattr(status, "name", status)).upper()
        out = {
            "lane_connector_id": str(tl.get("lane_connector_id", tl.get("lane_connector", ""))),
            "status": status_name,
            "timestamp": int(tl.get("timestamp", tl.get("timestamp_us", 0)) or 0),
        }
        if "relative_time_s" in tl:
            out["relative_time_s"] = float(tl.get("relative_time_s") or 0.0)
        return out
    try:
        status = getattr(tl, "status")
        status_name = str(getattr(status, "name", status)).upper()
    except Exception:
        status_name = "UNKNOWN"
    try:
        lane_connector_id = str(getattr(tl, "lane_connector_id"))
    except Exception:
        lane_connector_id = ""
    try:
        timestamp = int(getattr(tl, "timestamp"))
    except Exception:
        timestamp = 0
    return {"lane_connector_id": lane_connector_id, "status": status_name, "timestamp": timestamp}


def traffic_light_history_to_json(history: list[list[Any]]) -> list[list[dict[str, Any]]]:
    return [[traffic_light_to_dict(tl) for tl in step] for step in history]


def traffic_light_list_to_json(items: list[Any]) -> list[dict[str, Any]]:
    return [traffic_light_to_dict(tl) for tl in items]


class ScenarioAPIExtractor:
    """Build a one-step NuPlanScenario around a lidar token and extract map-data."""

    def __init__(self, data_root: str | Path, map_root: str | Path, map_version: str = "nuplan-maps-v1.0",
                 sensor_root: str | Path | None = None):
        self.data_root = str(data_root)
        self.map_root = str(map_root)
        self.map_version = map_version
        self.sensor_root = None if sensor_root is None else str(sensor_root)
        self.available = False
        self._init_error = ""
        try:
            from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario  # type: ignore
            from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters  # type: ignore
            self._NuPlanScenario = NuPlanScenario
            self._get_pacifica_parameters = get_pacifica_parameters
            self.available = True
        except Exception as exc:  # pragma: no cover
            self._init_error = str(exc)

    @staticmethod
    def _token_to_hex(token: Any) -> str:
        if isinstance(token, bytes):
            return token.hex()
        return str(token)

    def _make_scenario(self, db_path: str | Path, lidar_token: Any, timestamp_us: int, map_name: str) -> Any:
        if not self.available:
            raise ImportError(f"nuPlan devkit is unavailable: {self._init_error}")
        return self._NuPlanScenario(
            data_root=self.data_root,
            log_file_load_path=str(db_path),
            initial_lidar_token=self._token_to_hex(lidar_token),
            initial_lidar_timestamp=int(timestamp_us),
            scenario_type="dpies_preprocess",
            map_root=self.map_root,
            map_version=self.map_version,
            map_name=map_name,
            scenario_extraction_info=None,
            ego_vehicle_parameters=self._get_pacifica_parameters(),
            sensor_root=self.sensor_root,
        )

    def extract_for_lidar_row(self, db_path: str | Path, lidar_row: Any, map_name: str,
                              future_seconds: float, future_steps: int) -> ScenarioAPIContext:
        try:
            scenario = self._make_scenario(db_path, lidar_row["token"], int(lidar_row["timestamp_us"]), map_name)
            route_ids = []
            try:
                route_ids = [str(x) for x in scenario.get_route_roadblock_ids()]
            except Exception:
                # Some early/buggy scenarios may miss route ids. Keep going.
                route_ids = []
            current_tl = []
            try:
                current_tl = list(scenario.get_traffic_light_status_at_iteration(0))
            except Exception:
                current_tl = []
            future_tl: list[list[Any]] = []
            try:
                rel_dt = float(future_seconds) / max(int(future_steps), 1)
                for step_idx, status in enumerate(scenario.get_future_traffic_light_status_history(0, float(future_seconds), int(future_steps)), start=1):
                    raw_step = list(getattr(status, "traffic_lights", status))
                    converted = []
                    for tl in raw_step:
                        d = traffic_light_to_dict(tl)
                        d["relative_time_s"] = float(step_idx) * rel_dt
                        converted.append(d)
                    future_tl.append(converted)
            except Exception:
                future_tl = []
            mission_goal = None
            try:
                mission_goal = scenario.get_mission_goal()
            except Exception:
                mission_goal = None
            return ScenarioAPIContext(
                map_api=scenario.map_api,
                route_roadblock_ids=route_ids,
                traffic_lights_current=current_tl,
                traffic_lights_future=future_tl,
                mission_goal=mission_goal,
                scenario=scenario,
            )
        except Exception as exc:
            return ScenarioAPIContext(error=str(exc))


# ------------------------- closed-loop state conversion -------------------------


def ego_state_to_global_array(ego_state: Any) -> np.ndarray:
    """Convert official EgoState to [x,y,yaw,vx,vy,ax,ay,yaw_rate,speed]."""
    pose = getattr(ego_state, "rear_axle", getattr(ego_state, "center", None))
    x = float(getattr(pose, "x", 0.0))
    y = float(getattr(pose, "y", 0.0))
    yaw = float(getattr(pose, "heading", 0.0))
    dyn = getattr(ego_state, "dynamic_car_state", None)
    vel = getattr(dyn, "rear_axle_velocity_2d", getattr(dyn, "center_velocity_2d", None)) if dyn is not None else None
    acc = getattr(dyn, "rear_axle_acceleration_2d", getattr(dyn, "center_acceleration_2d", None)) if dyn is not None else None
    vx = float(getattr(vel, "x", 0.0)) if vel is not None else 0.0
    vy = float(getattr(vel, "y", 0.0)) if vel is not None else 0.0
    ax = float(getattr(acc, "x", 0.0)) if acc is not None else 0.0
    ay = float(getattr(acc, "y", 0.0)) if acc is not None else 0.0
    yaw_rate = float(getattr(dyn, "angular_velocity", 0.0)) if dyn is not None else 0.0
    speed = float(np.hypot(vx, vy))
    return np.asarray([x, y, yaw, vx, vy, ax, ay, yaw_rate, speed], dtype=np.float32)


def tracked_object_to_global_array(obj: Any) -> tuple[np.ndarray, str, int]:
    """Convert official tracked object to [x,y,yaw,vx,vy,length,width,type_id]."""
    box = getattr(obj, "box", getattr(obj, "oriented_box", None))
    center = getattr(box, "center", None)
    x = float(getattr(center, "x", 0.0))
    y = float(getattr(center, "y", 0.0))
    yaw = float(getattr(center, "heading", 0.0))
    length = float(getattr(box, "length", 4.5))
    width = float(getattr(box, "width", 2.0))
    vel = getattr(obj, "velocity", None)
    vx = float(getattr(vel, "x", 0.0)) if vel is not None else 0.0
    vy = float(getattr(vel, "y", 0.0)) if vel is not None else 0.0
    typ = getattr(obj, "tracked_object_type", None)
    type_name = str(getattr(typ, "name", typ)).lower()
    if "ped" in type_name:
        type_id = 1
    elif "bike" in type_name or "bicycle" in type_name or "cycl" in type_name:
        type_id = 2
    elif "vehicle" in type_name or "car" in type_name or "truck" in type_name or "bus" in type_name:
        type_id = 3
    else:
        type_id = 0
    metadata = getattr(obj, "metadata", None)
    track_token = str(getattr(metadata, "track_token", getattr(metadata, "token", ""))) if metadata is not None else ""
    return np.asarray([x, y, yaw, vx, vy, length, width, type_id], dtype=np.float32), track_token, type_id


def observation_to_objects(observation: Any) -> list[Any]:
    tracked = getattr(observation, "tracked_objects", observation)
    objs = getattr(tracked, "tracked_objects", None)
    if objs is None:
        try:
            return list(tracked)
        except Exception:
            return []
    return list(objs)


def build_history_tensors_from_simulation(
    ego_states: list[Any],
    observations: list[Any],
    max_agents: int,
    history_steps: int,
    ego_xy: np.ndarray,
    ego_yaw: float,
    agent_radius_m: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ego_history, agent_history and masks from SimulationHistoryBuffer."""
    if not ego_states:
        raise ValueError("empty ego_states in SimulationHistoryBuffer")
    # Use the latest history_steps, left-pad with the first available state.
    states = ego_states[-history_steps:]
    if len(states) < history_steps:
        states = [states[0]] * (history_steps - len(states)) + states
    ego_global = np.stack([ego_state_to_global_array(s) for s in states], axis=0)
    ego_history = transform_ego_state_global_to_ego(ego_global, ego_xy, ego_yaw)

    current_objs = observation_to_objects(observations[-1]) if observations else []
    current_records: list[tuple[str, np.ndarray, int, float]] = []
    for obj in current_objs:
        arr_g, token, type_id = tracked_object_to_global_array(obj)
        arr_e = transform_agent_state_global_to_ego(arr_g[None, :], ego_xy, ego_yaw)[0]
        dist = float(np.hypot(arr_e[0], arr_e[1]))
        if dist <= agent_radius_m:
            current_records.append((token, arr_e, type_id, dist))
    current_records.sort(key=lambda z: z[3])
    current_records = current_records[:max_agents]

    agent_history = np.zeros((max_agents, history_steps, 8), dtype=np.float32)
    agent_history_mask = np.zeros((max_agents, history_steps), dtype=bool)
    agent_mask = np.zeros((max_agents,), dtype=bool)
    agent_track_id = np.zeros((max_agents,), dtype=np.int64)
    agent_type = np.zeros((max_agents,), dtype=np.int64)
    token_to_idx = {tok: i for i, (tok, _, _, _) in enumerate(current_records)}
    for i, (tok, arr_e, type_id, _) in enumerate(current_records):
        agent_history[i, -1] = arr_e
        agent_history_mask[i, -1] = True
        agent_mask[i] = True
        agent_type[i] = int(type_id)
        agent_track_id[i] = stable_int(tok) if tok else i

    obs_hist = observations[-history_steps:] if observations else []
    if len(obs_hist) < history_steps and obs_hist:
        obs_hist = [obs_hist[0]] * (history_steps - len(obs_hist)) + obs_hist
    for h, obs in enumerate(obs_hist):
        for obj in observation_to_objects(obs):
            arr_g, tok, _ = tracked_object_to_global_array(obj)
            if tok not in token_to_idx:
                continue
            i = token_to_idx[tok]
            agent_history[i, h] = transform_agent_state_global_to_ego(arr_g[None, :], ego_xy, ego_yaw)[0]
            agent_history_mask[i, h] = True
    return ego_history.astype(np.float32), agent_history, agent_history_mask, agent_mask, agent_track_id, agent_type
