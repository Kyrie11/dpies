from __future__ import annotations

"""Optional nuPlan-devkit helpers.

The main DPIES package should remain importable in environments without nuPlan.
All official-devkit imports therefore live inside functions in this module.
The helpers support both the direct DB preprocessing path and the closed-loop
Scenario/PlannerInput path used by nuPlan simulation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

from dpies.common.geometry import global_to_ego_points, transform_agent_state_global_to_ego, wrap_angle
from dpies.data.nuplan_db import stable_int, token_to_str


@dataclass
class DevkitTrafficLightRecord:
    status: str
    lane_connector_id: str
    timestamp_us: int | None = None
    relative_time_s: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "lane_connector_id": self.lane_connector_id,
            "timestamp_us": self.timestamp_us,
            "relative_time_s": self.relative_time_s,
        }


def _status_to_string(status: Any) -> str:
    if status is None:
        return "UNKNOWN"
    for attr in ("name", "value"):
        try:
            val = getattr(status, attr)
            if val is not None:
                return str(val).upper()
        except Exception:
            pass
    return str(status).upper()


def is_red_status(status: Any) -> bool:
    s = _status_to_string(status)
    return "RED" in s or s.endswith("STOP")


def _record_from_devkit(obj: Any, relative_time_s: float | None = None) -> DevkitTrafficLightRecord:
    status = _status_to_string(getattr(obj, "status", None))
    lane_connector_id = getattr(obj, "lane_connector_id", getattr(obj, "lane_connector_token", ""))
    timestamp = getattr(obj, "timestamp", getattr(obj, "timestamp_us", None))
    try:
        timestamp = int(timestamp) if timestamp is not None else None
    except Exception:
        timestamp = None
    return DevkitTrafficLightRecord(
        status=status,
        lane_connector_id=str(lane_connector_id),
        timestamp_us=timestamp,
        relative_time_s=relative_time_s,
    )


def records_to_json(records: Sequence[DevkitTrafficLightRecord]) -> list[dict[str, Any]]:
    return [r.to_json() for r in records]


def flatten_traffic_statuses(statuses: Any, relative_time_s: float | None = None) -> list[DevkitTrafficLightRecord]:
    """Convert nuPlan TrafficLightStatusData / TrafficLightStatuses containers to plain records.

    Accepts a single TrafficLightStatusData, a TrafficLightStatuses wrapper, a
    dict/list of dicts, or a nested future history list.
    """
    if statuses is None:
        return []
    if isinstance(statuses, DevkitTrafficLightRecord):
        return [statuses]
    if isinstance(statuses, dict):
        ts = statuses.get("timestamp_us", statuses.get("timestamp", None))
        try:
            ts = int(ts) if ts is not None else None
        except Exception:
            ts = None
        return [DevkitTrafficLightRecord(
            status=_status_to_string(statuses.get("status")),
            lane_connector_id=str(statuses.get("lane_connector_id", statuses.get("connector_id", ""))),
            timestamp_us=ts,
            relative_time_s=statuses.get("relative_time_s", relative_time_s),
        )]
    # The devkit TrafficLightStatuses wrapper usually exposes .traffic_lights.
    for attr in ("traffic_lights", "statuses", "data"):
        try:
            val = getattr(statuses, attr)
            if val is not None and val is not statuses:
                return flatten_traffic_statuses(val, relative_time_s)
        except Exception:
            pass
    if isinstance(statuses, (list, tuple)):
        out: list[DevkitTrafficLightRecord] = []
        for idx, item in enumerate(statuses):
            # For nested future histories, preserve per-step relative time if the
            # inner items do not already carry one.
            item_rel = relative_time_s
            if isinstance(item, (list, tuple)) and relative_time_s is None:
                item_rel = float(idx)
            out.extend(flatten_traffic_statuses(item, item_rel))
        return out
    try:
        return [_record_from_devkit(x, relative_time_s=relative_time_s) for x in list(statuses)]
    except Exception:
        return [_record_from_devkit(statuses, relative_time_s=relative_time_s)]

def red_times_by_connector(records: Sequence[DevkitTrafficLightRecord]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for r in records:
        if is_red_status(r.status):
            out.setdefault(str(r.lane_connector_id), []).append(float(r.relative_time_s or 0.0))
    return out


def latest_status_by_connector(records: Sequence[DevkitTrafficLightRecord]) -> dict[str, str]:
    out: dict[str, str] = {}
    for r in records:
        out[str(r.lane_connector_id)] = r.status
    return out


def read_route_roadblock_ids_from_db(db_path: str | Path, lidar_token: Any) -> list[str]:
    """Official-devkit route extraction from a lidar token, with safe fallback."""
    try:
        from nuplan.database.nuplan_db.nuplan_scenario_queries import get_roadblock_ids_for_lidarpc_token_from_db  # type: ignore
    except Exception:
        return []
    try:
        ids = get_roadblock_ids_for_lidarpc_token_from_db(str(db_path), token_to_str(lidar_token))
        if ids is None:
            return []
        return [str(x) for x in ids]
    except Exception:
        # Some DB interfaces expect raw bytes instead of hex strings.
        try:
            ids = get_roadblock_ids_for_lidarpc_token_from_db(str(db_path), lidar_token)
            return [] if ids is None else [str(x) for x in ids]
        except Exception:
            return []


def read_traffic_light_records_from_db(db_path: str | Path, lidar_token: Any, relative_time_s: float | None = None) -> list[DevkitTrafficLightRecord]:
    """Official-devkit traffic-light status extraction from a lidar token, with safe fallback."""
    try:
        from nuplan.database.nuplan_db.nuplan_scenario_queries import get_traffic_light_status_for_lidarpc_token_from_db  # type: ignore
    except Exception:
        return []
    try:
        records = list(get_traffic_light_status_for_lidarpc_token_from_db(str(db_path), token_to_str(lidar_token)))
    except Exception:
        try:
            records = list(get_traffic_light_status_for_lidarpc_token_from_db(str(db_path), lidar_token))
        except Exception:
            return []
    return flatten_traffic_statuses(records, relative_time_s=relative_time_s)


def traffic_from_planner_input(current_input: Any) -> list[DevkitTrafficLightRecord]:
    return flatten_traffic_statuses(getattr(current_input, "traffic_light_data", None), relative_time_s=0.0)


def traffic_from_scenario(scenario: Any, iteration: int, future_seconds: float = 0.0, dt: float = 0.5) -> tuple[list[DevkitTrafficLightRecord], list[DevkitTrafficLightRecord]]:
    current: list[DevkitTrafficLightRecord] = []
    future: list[DevkitTrafficLightRecord] = []
    try:
        current = flatten_traffic_statuses(list(scenario.get_traffic_light_status_at_iteration(iteration)), relative_time_s=0.0)
    except Exception:
        current = []
    if future_seconds > 0.0:
        num = max(1, int(round(future_seconds / max(dt, 1e-3))))
        try:
            for h, statuses in enumerate(scenario.get_future_traffic_light_status_history(iteration, future_seconds, num_samples=num), start=1):
                future.extend(flatten_traffic_statuses(statuses, relative_time_s=h * dt))
        except Exception:
            pass
    return current, future


def route_from_scenario_or_init(scenario: Any | None = None, initialization: Any | None = None) -> list[str]:
    if initialization is not None:
        try:
            ids = getattr(initialization, "route_roadblock_ids")
            if ids:
                return [str(x) for x in ids]
        except Exception:
            pass
    if scenario is not None:
        try:
            return [str(x) for x in scenario.get_route_roadblock_ids()]
        except Exception:
            pass
    return []


def ego_state_to_global_array(ego: Any) -> np.ndarray:
    """Convert a nuPlan EgoState to DPIES global ego schema.

    Output: [x,y,yaw,vx,vy,ax,ay,yaw_rate,speed]. Uses rear axle pose to be
    consistent with nuPlan planner trajectories.
    """
    pose = getattr(ego, "rear_axle", getattr(ego, "center", None))
    if pose is None:
        raise ValueError("EgoState has neither rear_axle nor center")
    dyn = getattr(ego, "dynamic_car_state", None)
    vel = getattr(dyn, "rear_axle_velocity_2d", getattr(dyn, "center_velocity_2d", None)) if dyn is not None else None
    acc = getattr(dyn, "rear_axle_acceleration_2d", getattr(dyn, "center_acceleration_2d", None)) if dyn is not None else None
    vx = float(getattr(vel, "x", 0.0))
    vy = float(getattr(vel, "y", 0.0))
    ax = float(getattr(acc, "x", 0.0))
    ay = float(getattr(acc, "y", 0.0))
    yaw_rate = float(getattr(dyn, "angular_velocity", 0.0)) if dyn is not None else 0.0
    speed = float(np.hypot(vx, vy))
    return np.asarray([float(pose.x), float(pose.y), float(pose.heading), vx, vy, ax, ay, yaw_rate, speed], dtype=np.float32)


def _tracked_object_token(obj: Any, fallback_idx: int = 0) -> str:
    meta = getattr(obj, "metadata", None)
    for attr in ("track_token", "token", "track_id"):
        try:
            val = getattr(meta, attr)
            if val is not None:
                return str(val)
        except Exception:
            pass
    return f"object_{fallback_idx}"


def _tracked_object_type_id(obj: Any) -> int:
    typ = getattr(obj, "tracked_object_type", None)
    name = _status_to_string(typ)
    if "PEDESTRIAN" in name:
        return 1
    if "BICYCLE" in name or "CYCLIST" in name:
        return 2
    if "VEHICLE" in name or "CAR" in name or "BUS" in name or "TRUCK" in name:
        return 3
    if "BARRIER" in name or "CONE" in name:
        return 4
    return stable_int(name, mod=1000) + 10


def tracked_object_to_global_state(obj: Any) -> tuple[str, np.ndarray]:
    box = getattr(obj, "box", getattr(obj, "oriented_box", None))
    if box is None:
        raise ValueError("Tracked object has no oriented box")
    center = getattr(box, "center", None)
    if center is None:
        raise ValueError("Tracked object box has no center")
    vel = getattr(obj, "velocity", None)
    vx = float(getattr(vel, "x", 0.0))
    vy = float(getattr(vel, "y", 0.0))
    length = float(getattr(box, "length", 4.5))
    width = float(getattr(box, "width", 2.0))
    typ = _tracked_object_type_id(obj)
    token = _tracked_object_token(obj)
    state = np.asarray([float(center.x), float(center.y), float(center.heading), vx, vy, length, width, float(typ)], dtype=np.float32)
    return token, state


def observation_to_tracked_objects(observation: Any) -> list[Any]:
    if observation is None:
        return []
    objs = getattr(observation, "tracked_objects", observation)
    if hasattr(objs, "tracked_objects"):
        return list(objs.tracked_objects)
    try:
        return list(objs)
    except Exception:
        return []


def current_agents_from_observation(observation: Any, ego_global_state: np.ndarray, max_agents: int, radius_m: float) -> tuple[list[str], np.ndarray, np.ndarray]:
    ego_xy = ego_global_state[:2]
    ego_yaw = float(ego_global_state[2])
    rows: list[tuple[float, str, np.ndarray]] = []
    for idx, obj in enumerate(observation_to_tracked_objects(observation)):
        try:
            token, state = tracked_object_to_global_state(obj)
            local = transform_agent_state_global_to_ego(state, ego_xy, ego_yaw)
        except Exception:
            continue
        dist = float(np.linalg.norm(local[:2]))
        if dist <= radius_m:
            rows.append((dist, token, local))
    rows.sort(key=lambda x: x[0])
    tokens = [r[1] for r in rows[:max_agents]]
    arr = np.zeros((max_agents, 8), dtype=np.float32)
    mask = np.zeros((max_agents,), dtype=bool)
    for i, (_, _, local) in enumerate(rows[:max_agents]):
        arr[i] = local
        mask[i] = True
    return tokens, arr, mask


def agent_history_from_observations(observations: Sequence[Any], tokens: Sequence[str], ego_global_state: np.ndarray,
                                    hist_steps: int, max_agents: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_agents, hist_steps, 8), dtype=np.float32)
    mask = np.zeros((max_agents, hist_steps), dtype=bool)
    token_to_idx = {str(tok): i for i, tok in enumerate(tokens[:max_agents])}
    ego_xy = ego_global_state[:2]
    ego_yaw = float(ego_global_state[2])
    obs_list = list(observations)[-hist_steps:]
    # Left-pad if the simulation buffer is shorter than the configured history.
    start_h = hist_steps - len(obs_list)
    for j, obs in enumerate(obs_list):
        h = start_h + j
        for obj_idx, obj in enumerate(observation_to_tracked_objects(obs)):
            try:
                token, state = tracked_object_to_global_state(obj)
            except Exception:
                continue
            idx = token_to_idx.get(str(token))
            if idx is None:
                continue
            out[idx, h] = transform_agent_state_global_to_ego(state, ego_xy, ego_yaw)
            mask[idx, h] = True
    return out, mask
