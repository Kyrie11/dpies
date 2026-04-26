from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as np

from dpies.common.geometry import pairwise_min_distance, time_to_collision_1d
from dpies.common.types import EVIDENCE_DIM, EvidenceType, MapRuleCode


def _jsonable_points(value: Any) -> list[list[float]]:
    if value is None:
        return []
    try:
        arr = np.asarray(value, dtype=np.float32).reshape((-1, 2))
        return [[float(x), float(y)] for x, y in arr]
    except Exception:
        return []

def _jsonable_polygons(value: Any) -> list[list[list[float]]]:
    if value is None:
        return []
    out: list[list[list[float]]] = []
    try:
        for poly in value:
            pts = _jsonable_points(poly)
            if pts:
                out.append(pts)
    except Exception:
        pts = _jsonable_points(value)
        if pts:
            out.append(pts)
    return out


@dataclass
class EvidenceBuilderConfig:
    max_units: int = 128
    max_dynamic: int = 48
    max_conflict: int = 32
    max_gap: int = 16
    max_map_rule: int = 32
    max_risk: int = 16
    radius_m: float = 80.0
    conflict_distance_m: float = 5.0
    low_ttc_s: float = 3.0
    lane_width_m: float = 3.5


class EvidenceBuilder:
    def __init__(self, cfg: EvidenceBuilderConfig):
        self.cfg = cfg
        self.last_metadata: list[dict[str, Any]] = []

    def _feature(self, typ: EvidenceType, x: float = 0.0, y: float = 0.0, vx: float = 0.0, vy: float = 0.0,
                 length: float = 0.0, width: float = 0.0, distance: float = 0.0, ttc: float = 99.0,
                 confidence: float = 1.0, agent_index: int = -1, aux: Sequence[float] | None = None) -> np.ndarray:
        f = np.zeros((EVIDENCE_DIM,), dtype=np.float32)
        speed = float(np.hypot(vx, vy))
        f[:12] = np.asarray([
            int(typ), x, y, vx, vy, length, width, distance, speed, min(ttc, 99.0), confidence, float(agent_index)
        ], dtype=np.float32)
        if aux is not None:
            arr = np.asarray(list(aux), dtype=np.float32)
            f[12:12 + min(len(arr), EVIDENCE_DIM - 12)] = arr[: EVIDENCE_DIM - 12]
        return f

    def _relevance(self, f: np.ndarray, typ: EvidenceType, hard_keep: bool = False) -> float:
        d = max(float(f[7]), 0.0)
        ttc = float(f[9])
        conf = float(f[10])
        score = 1.0 * np.exp(-d / 35.0) + 0.8 * np.exp(-ttc / 3.0) + 0.2 * conf
        if typ in (EvidenceType.MAP_RULE, EvidenceType.LOW_TTC_RISK, EvidenceType.CONFLICT_POINT):
            score += 0.5
        if hard_keep:
            score += 100.0
        return float(score)

    @staticmethod
    def _last_valid_agent_state(agent_history: np.ndarray, agent_history_mask: np.ndarray | None, idx: int) -> np.ndarray:
        if agent_history_mask is not None and idx < agent_history_mask.shape[0] and agent_history_mask[idx].any():
            h = int(np.where(agent_history_mask[idx])[0][-1])
            return agent_history[idx, h]
        return agent_history[idx, -1]

    def build(self, agent_history: np.ndarray, agent_mask: np.ndarray, actions: np.ndarray,
              action_mask: np.ndarray, rule_units: list[dict] | None = None, dt: float = 0.5,
              agent_history_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build high-recall evidence from current/history and optional map rule units.

        Sets `last_metadata` with a JSON-serializable description of retained units.
        """
        units: List[tuple[float, bool, EvidenceType, np.ndarray, dict[str, Any]]] = []
        valid_agents = np.where(agent_mask)[0]

        dyn_count = 0
        for idx in valid_agents:
            st = self._last_valid_agent_state(agent_history, agent_history_mask, int(idx))
            x, y, yaw, vx, vy, length, width, obj_type = st.tolist()
            dist = float(np.hypot(x, y))
            if dist > self.cfg.radius_m:
                continue
            speed = float(np.hypot(vx, vy))
            heading_align = 1.0 if speed < 0.2 else float(max(np.cos(float(yaw) - np.arctan2(vy, vx)), 0.0))
            conf = float(np.clip(0.35 + 0.35 * np.exp(-dist / 40.0) + 0.3 * heading_align, 0.1, 1.0))
            ttc = time_to_collision_1d(np.asarray([x, y], dtype=np.float32), np.asarray([vx, vy], dtype=np.float32), radius=4.0)
            feat = self._feature(EvidenceType.DYNAMIC_AGENT, x, y, vx, vy, length, width, dist, ttc, conf, int(idx), aux=[0.0, yaw, obj_type])
            meta = {"type": "dynamic_agent", "agent_ids": [int(idx)], "motion_mode": "constant_velocity", "time_interval": [0.0, float(actions.shape[1] * dt)]}
            units.append((self._relevance(feat, EvidenceType.DYNAMIC_AGENT), False, EvidenceType.DYNAMIC_AGENT, feat, meta))
            dyn_count += 1
            if dyn_count >= self.cfg.max_dynamic:
                break

        # Conflict-point evidence from ego candidates and predicted CV agents.
        conf_count = 0
        time_steps = np.arange(1, actions.shape[1] + 1, dtype=np.float32) * dt
        action_union = actions[action_mask, :, :2].reshape(-1, 2) if action_mask.any() else np.zeros((0, 2), dtype=np.float32)
        for idx in valid_agents:
            if conf_count >= self.cfg.max_conflict:
                break
            st = self._last_valid_agent_state(agent_history, agent_history_mask, int(idx))
            x, y, yaw, vx, vy, length, width, obj_type = st.tolist()
            pred = np.stack([x + vx * time_steps, y + vy * time_steps], axis=-1)
            if len(action_union) == 0:
                continue
            dmin, ego_i, ag_i = pairwise_min_distance(action_union, pred)
            footprint_threshold = self.cfg.conflict_distance_m + 0.25 * (float(length) + float(width))
            if dmin < footprint_threshold:
                p = pred[ag_i]
                aux = [float(ego_i), float(ag_i), float(dmin), yaw, obj_type]
                feat = self._feature(EvidenceType.CONFLICT_POINT, float(p[0]), float(p[1]), vx, vy, length, width,
                                     float(np.hypot(p[0], p[1])), float(ag_i) * dt, 1.0, int(idx), aux)
                meta = {"type": "conflict_point", "agent_ids": [int(idx)], "time_interval": [max(0.0, (ag_i - 1) * dt), float((ag_i + 1) * dt)], "min_distance": float(dmin)}
                units.append((self._relevance(feat, EvidenceType.CONFLICT_POINT, hard_keep=True), True, EvidenceType.CONFLICT_POINT, feat, meta))
                conf_count += 1

        # Gap units for left/right target corridors. Prefer map-supported corridors;
        # fall back only when no rule_units exist. This prevents fake gaps in places
        # without adjacent lane support.
        gap_count = 0
        for side, lane_y in ((1.0, self.cfg.lane_width_m), (-1.0, -self.cfg.lane_width_m)):
            if rule_units:
                supported = False
                for ru in rule_units:
                    layer = str(ru.get("layer", "")).upper()
                    if not any(k in layer for k in ("LANE", "CONNECTOR", "ROADBLOCK", "ROUTE_CORRIDOR", "DRIVABLE")):
                        continue
                    pts = np.asarray(ru.get("polyline", ru.get("polygon", ru.get("xy", []))), dtype=np.float32).reshape(
                        -1, 2)
                    if len(pts) and np.count_nonzero((pts[:, 0] > 0.0) & (pts[:, 0] < 70.0) & (
                            np.abs(pts[:, 1] - lane_y) < self.cfg.lane_width_m)) >= 2:
                        supported = True
                        break
                if not supported:
                    continue
            near = []
            for idx in valid_agents:
                st = self._last_valid_agent_state(agent_history, agent_history_mask, int(idx))
                x, y, yaw, vx, vy, length, width, obj_type = st.tolist()
                if abs(y - lane_y) < self.cfg.lane_width_m * 0.75 and -25.0 < x < 90.0:
                    near.append((x, idx, st))
            if not near:
                continue
            front = min([n for n in near if n[0] >= 0.0], default=None, key=lambda z: z[0])
            rear = max([n for n in near if n[0] < 0.0], default=None, key=lambda z: z[0])
            front_gap = float(front[0]) if front else 80.0
            rear_gap = float(-rear[0]) if rear else 80.0
            front_relv = float(front[2][3]) if front else 0.0
            rear_relv = float(rear[2][3]) if rear else 0.0
            x = front_gap if front else 20.0
            aux = [side, front_gap, rear_gap, rear_relv, front_relv, 0.0, 0.0, min(front_gap, rear_gap)]
            agent_id = int(front[1] if front else rear[1] if rear else -1)
            feat = self._feature(EvidenceType.GAP, x, lane_y, 0.0, 0.0, 0.0, self.cfg.lane_width_m,
                                 min(front_gap, rear_gap), 99.0, 1.0, agent_id, aux)
            meta = {"type": "gap", "agent_ids": [agent_id] if agent_id >= 0 else [], "side": float(side), "front_gap": front_gap, "rear_gap": rear_gap}
            units.append((self._relevance(feat, EvidenceType.GAP), False, EvidenceType.GAP, feat, meta))
            gap_count += 1
            if gap_count >= self.cfg.max_gap:
                break

        # Map-rule units from optional map API. Rule metadata preserves exact
        # ego-frame polylines/polygons for GeometryQuery, while the fixed feature
        # tensor carries compact numeric hints.
        map_count = 0
        for ru in rule_units or []:
            if map_count >= self.cfg.max_map_rule:
                break
            xy = np.asarray(ru.get("xy", [0.0, 0.0]), dtype=np.float32)
            if xy.shape[0] < 2:
                xy = np.asarray([0.0, 0.0], dtype=np.float32)
            layer = str(ru.get("layer", "rule"))
            code = int(ru.get("rule_code", 0))
            hard = bool(ru.get("hard_keep", False)) or code in (
                int(MapRuleCode.STOP_LINE), int(MapRuleCode.TRAFFIC_LIGHT_RED),
                int(MapRuleCode.CROSSWALK), int(MapRuleCode.DRIVABLE_AREA),
            )
            polyline = np.asarray(ru.get("polyline", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            polygon = np.asarray(ru.get("polygon", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            polygons = ru.get("polygons", None)
            if polyline.ndim != 2:
                polyline = np.zeros((0, 2), dtype=np.float32)
            if polygon.ndim != 2:
                polygon = np.zeros((0, 2), dtype=np.float32)
            extent_pts = polyline if len(polyline) else polygon
            extent = float(np.linalg.norm(extent_pts.max(axis=0) - extent_pts.min(axis=0))) if extent_pts.ndim == 2 and len(extent_pts) else 0.0
            speed_limit = float(ru.get("speed_limit_mps", 0.0) or 0.0)
            lane_side = float(ru.get("lane_side", 0.0) or 0.0)
            is_route = 1.0 if bool(ru.get("is_route", False)) else 0.0
            feat = self._feature(
                EvidenceType.MAP_RULE, float(xy[0]), float(xy[1]), 0.0, 0.0, extent, 0.0,
                float(np.linalg.norm(xy)), 99.0, 1.0, -1,
                [float(code), lane_side, extent, speed_limit, is_route],
            )
            meta = {
                "type": "map_rule",
                "map_object_ids": [str(ru.get("map_object_id", ""))],
                "layer": layer,
                "rule_code": code,
                "xy": [float(xy[0]), float(xy[1])],
                "polyline": _jsonable_points(polyline),
                "polygon": _jsonable_points(polygon),
                "polygons": _jsonable_polygons(polygons),
                "speed_limit_mps": speed_limit,
                "lane_side": lane_side,
                "lane_connector_id": str(ru.get("lane_connector_id", "")),
                "traffic_light_status": str(ru.get("traffic_light_status", "")),
                "traffic_light_timestamp": int(ru.get("traffic_light_timestamp", 0) or 0),
                "red_times_s": [float(x) for x in ru.get("red_times_s", [])],
                "geometry_type": str(ru.get("geometry_type", "polyline")),
                "roadblock_id": str(ru.get("roadblock_id", "")),
                "is_route": bool(ru.get("is_route", False)),
            }
            units.append((self._relevance(feat, EvidenceType.MAP_RULE, hard_keep=hard), hard, EvidenceType.MAP_RULE, feat, meta))
            map_count += 1

        # Optional: include coarse boundaries only if map-rule quota has room.
        coarse_count = 0
        # Always include coarse drivable/lane-boundary rule tokens in the ego frame.
        for side, y in ((1.0, self.cfg.lane_width_m * 1.5), (-1.0, -self.cfg.lane_width_m * 1.5)):
            if map_count + coarse_count >= self.cfg.max_map_rule:
                break
            feat = self._feature(EvidenceType.MAP_RULE, 25.0, y, 0.0, 0.0, 0.0, 0.0, abs(y), 99.0, 0.75, -1,
                                 [float(MapRuleCode.LANE_BOUNDARY), side])
            meta = {"type": "map_rule", "layer": "coarse_lane_boundary", "rule_code": int(MapRuleCode.LANE_BOUNDARY), "side": float(side)}
            units.append((self._relevance(feat, EvidenceType.MAP_RULE), False, EvidenceType.MAP_RULE, feat, meta))
            coarse_count += 1
        # Stronger per-type cap after hard_keep scoring, so one evidence family cannot
        # consume the whole max_units budget. Critical hard units still sort first
        # within their family.
        per_type_limit = {
            EvidenceType.DYNAMIC_AGENT: self.cfg.max_dynamic,
            EvidenceType.CONFLICT_POINT: self.cfg.max_conflict,
            EvidenceType.GAP: self.cfg.max_gap,
            EvidenceType.MAP_RULE: self.cfg.max_map_rule,
            EvidenceType.LOW_TTC_RISK: self.cfg.max_risk,
        }
        # Low-TTC/proximity risk units.
        risk_count = 0
        for idx in valid_agents:
            st = self._last_valid_agent_state(agent_history, agent_history_mask, int(idx))
            x, y, yaw, vx, vy, length, width, obj_type = st.tolist()
            dist = float(np.hypot(x, y))
            ttc = time_to_collision_1d(np.asarray([x, y], dtype=np.float32), np.asarray([vx, vy], dtype=np.float32), radius=5.0)
            candidate_risk = False
            if action_mask.any():
                dmin, _, _ = pairwise_min_distance(actions[action_mask, :, :2].reshape(-1, 2), np.asarray([[x, y]], dtype=np.float32))
                candidate_risk = dmin < 6.0
            if dist < 12.0 or ttc < self.cfg.low_ttc_s or candidate_risk:
                feat = self._feature(EvidenceType.LOW_TTC_RISK, x, y, vx, vy, length, width, dist, ttc, 1.0, int(idx), aux=[yaw, obj_type])
                meta = {"type": "low_ttc_risk", "agent_ids": [int(idx)], "ttc": float(min(ttc, 99.0)), "distance": dist}
                units.append((self._relevance(feat, EvidenceType.LOW_TTC_RISK, hard_keep=True), True, EvidenceType.LOW_TTC_RISK, feat, meta))
                risk_count += 1
                if risk_count >= self.cfg.max_risk:
                    break

        # Two-stage pruning: keep hard critical units first, then score-rank the rest.
        hard_units = [u for u in units if u[1]]
        soft_units = [u for u in units if not u[1]]
        hard_units.sort(key=lambda u: u[0], reverse=True)
        soft_units.sort(key=lambda u: u[0], reverse=True)
        kept = []
        type_counts = {k: 0 for k in per_type_limit}
        for u in hard_units + soft_units:
            typ = u[2]
            if type_counts.get(typ, 0) >= per_type_limit.get(typ, self.cfg.max_units):
                continue
            kept.append(u)
            type_counts[typ] = type_counts.get(typ, 0) + 1
            if len(kept) >= self.cfg.max_units:
                break
        features = np.zeros((self.cfg.max_units, EVIDENCE_DIM), dtype=np.float32)
        type_ids = np.full((self.cfg.max_units,), int(EvidenceType.PADDING), dtype=np.int64)
        costs = np.ones((self.cfg.max_units,), dtype=np.float32)
        mask = np.zeros((self.cfg.max_units,), dtype=bool)
        self.last_metadata = []
        for i, (_, hard, typ, feat, meta) in enumerate(kept):
            features[i] = feat
            type_ids[i] = int(typ)
            mask[i] = True
            meta = dict(meta)
            meta.update({"unit_id": i, "type_id": int(typ), "confidence": float(feat[10]), "cost": 1.0, "hard_keep": bool(hard)})
            self.last_metadata.append(meta)
        return features, type_ids, costs, mask
