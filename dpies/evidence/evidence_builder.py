from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from dpies.common.geometry import pairwise_min_distance, time_to_collision_1d
from dpies.common.types import EVIDENCE_DIM, EvidenceType


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

    def _relevance(self, f: np.ndarray, typ: EvidenceType) -> float:
        d = max(float(f[7]), 0.0)
        ttc = float(f[9])
        conf = float(f[10])
        score = 1.0 * np.exp(-d / 35.0) + 0.8 * np.exp(-ttc / 3.0) + 0.2 * conf
        if typ in (EvidenceType.MAP_RULE, EvidenceType.LOW_TTC_RISK, EvidenceType.CONFLICT_POINT):
            score += 0.5
        return float(score)

    def build(self, agent_history: np.ndarray, agent_mask: np.ndarray, actions: np.ndarray,
              action_mask: np.ndarray, rule_units: list[dict] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build high-recall evidence from current/history and optional map rule units."""
        units: List[tuple[float, EvidenceType, np.ndarray]] = []
        current = agent_history[:, -1]
        valid_agents = np.where(agent_mask)[0]
        # Dynamic-agent occupancy units.
        dyn_count = 0
        for idx in valid_agents:
            st = current[idx]
            x, y, yaw, vx, vy, length, width, typ = st.tolist()
            dist = float(np.hypot(x, y))
            if dist > self.cfg.radius_m:
                continue
            speed = float(np.hypot(vx, vy))
            conf = float(np.clip(0.5 + 0.5 * np.exp(-dist / 40.0), 0.1, 1.0))
            rel = np.asarray([x, y], dtype=np.float32)
            ttc = time_to_collision_1d(rel, np.asarray([vx, vy], dtype=np.float32), radius=4.0)
            feat = self._feature(EvidenceType.DYNAMIC_AGENT, x, y, vx, vy, length, width, dist, ttc, conf, int(idx))
            units.append((self._relevance(feat, EvidenceType.DYNAMIC_AGENT), EvidenceType.DYNAMIC_AGENT, feat))
            dyn_count += 1
            if dyn_count >= self.cfg.max_dynamic:
                break
        # Conflict-point evidence from ego candidates and predicted CV agents.
        conf_count = 0
        time_steps = np.arange(1, actions.shape[1] + 1, dtype=np.float32) * 0.5
        action_union = actions[action_mask, :, :2].reshape(-1, 2) if action_mask.any() else np.zeros((0, 2), dtype=np.float32)
        for idx in valid_agents:
            if conf_count >= self.cfg.max_conflict:
                break
            st = current[idx]
            x, y, _, vx, vy, length, width, _ = st.tolist()
            pred = np.stack([x + vx * time_steps, y + vy * time_steps], axis=-1)
            if len(action_union) == 0:
                continue
            dmin, ego_i, ag_i = pairwise_min_distance(action_union, pred)
            if dmin < self.cfg.conflict_distance_m:
                p = pred[ag_i]
                aux = [float(ego_i), float(ag_i), float(dmin)]
                feat = self._feature(EvidenceType.CONFLICT_POINT, float(p[0]), float(p[1]), vx, vy, length, width,
                                     float(np.hypot(p[0], p[1])), float(ag_i) * 0.5, 1.0, int(idx), aux)
                units.append((self._relevance(feat, EvidenceType.CONFLICT_POINT), EvidenceType.CONFLICT_POINT, feat))
                conf_count += 1
        # Gap units for left/right target corridors.
        gap_count = 0
        for side, lane_y in ((1.0, self.cfg.lane_width_m), (-1.0, -self.cfg.lane_width_m)):
            near = []
            for idx in valid_agents:
                st = current[idx]
                x, y, _, vx, vy, length, width, _ = st.tolist()
                if abs(y - lane_y) < self.cfg.lane_width_m * 0.75 and -20.0 < x < 80.0:
                    near.append((x, idx, st))
            if not near:
                continue
            front = min([n for n in near if n[0] >= 0.0], default=None, key=lambda z: z[0])
            rear = max([n for n in near if n[0] < 0.0], default=None, key=lambda z: z[0])
            front_gap = float(front[0]) if front else 80.0
            rear_gap = float(-rear[0]) if rear else 80.0
            relv_rear = float(rear[2][3]) if rear else 0.0
            x = front_gap if front else 20.0
            aux = [side, front_gap, rear_gap, relv_rear]
            feat = self._feature(EvidenceType.GAP, x, lane_y, 0.0, 0.0, 0.0, self.cfg.lane_width_m,
                                 min(front_gap, rear_gap), 99.0, 1.0, int(front[1] if front else rear[1] if rear else -1), aux)
            units.append((self._relevance(feat, EvidenceType.GAP), EvidenceType.GAP, feat))
            gap_count += 1
            if gap_count >= self.cfg.max_gap:
                break
        # Map-rule units from optional map API.
        map_count = 0
        for ru in rule_units or []:
            xy = np.asarray(ru.get("xy", [0.0, 0.0]), dtype=np.float32)
            layer = str(ru.get("layer", "rule"))
            layer_code = 1.0 if "STOP" in layer else 2.0 if "CROSS" in layer or "WALK" in layer else 0.0
            feat = self._feature(EvidenceType.MAP_RULE, float(xy[0]), float(xy[1]), 0.0, 0.0, 0.0, 0.0,
                                 float(np.linalg.norm(xy)), 99.0, 1.0, -1, [layer_code])
            units.append((self._relevance(feat, EvidenceType.MAP_RULE), EvidenceType.MAP_RULE, feat))
            map_count += 1
            if map_count >= self.cfg.max_map_rule:
                break
        # Always include coarse drivable/lane-boundary rule tokens in the ego frame.
        for side, y in ((1.0, self.cfg.lane_width_m * 1.5), (-1.0, -self.cfg.lane_width_m * 1.5)):
            feat = self._feature(EvidenceType.MAP_RULE, 25.0, y, 0.0, 0.0, 0.0, 0.0, abs(y), 99.0, 0.75, -1, [3.0, side])
            units.append((self._relevance(feat, EvidenceType.MAP_RULE), EvidenceType.MAP_RULE, feat))
        # Low-TTC/proximity risk units.
        risk_count = 0
        for idx in valid_agents:
            st = current[idx]
            x, y, _, vx, vy, length, width, _ = st.tolist()
            dist = float(np.hypot(x, y))
            rel = np.asarray([x, y], dtype=np.float32)
            ttc = time_to_collision_1d(rel, np.asarray([vx, vy], dtype=np.float32), radius=5.0)
            if dist < 12.0 or ttc < self.cfg.low_ttc_s:
                feat = self._feature(EvidenceType.LOW_TTC_RISK, x, y, vx, vy, length, width, dist, ttc, 1.0, int(idx))
                units.append((self._relevance(feat, EvidenceType.LOW_TTC_RISK), EvidenceType.LOW_TTC_RISK, feat))
                risk_count += 1
                if risk_count >= self.cfg.max_risk:
                    break
        units.sort(key=lambda u: u[0], reverse=True)
        kept = units[: self.cfg.max_units]
        features = np.zeros((self.cfg.max_units, EVIDENCE_DIM), dtype=np.float32)
        type_ids = np.full((self.cfg.max_units,), int(EvidenceType.PADDING), dtype=np.int64)
        costs = np.ones((self.cfg.max_units,), dtype=np.float32)
        mask = np.zeros((self.cfg.max_units,), dtype=bool)
        for i, (_, typ, feat) in enumerate(kept):
            features[i] = feat
            type_ids[i] = int(typ)
            mask[i] = True
        return features, type_ids, costs, mask
