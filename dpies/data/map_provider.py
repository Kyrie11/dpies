from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from dpies.common.geometry import global_to_ego_points, pad_or_trim
from dpies.common.types import MAP_DIM


@dataclass
class MapObjects:
    polylines: np.ndarray
    masks: np.ndarray
    rule_units: List[dict]


class NullMapProvider:
    def __init__(self, max_polylines: int = 256, max_points: int = 20):
        self.max_polylines = max_polylines
        self.max_points = max_points

    def extract(self, map_name: str, ego_xy: np.ndarray, ego_yaw: float, radius_m: float) -> MapObjects:
        return MapObjects(
            polylines=np.zeros((self.max_polylines, self.max_points, MAP_DIM), dtype=np.float32),
            masks=np.zeros((self.max_polylines, self.max_points), dtype=bool),
            rule_units=[],
        )


class NuPlanMapProvider(NullMapProvider):
    """Defensive optional wrapper around nuPlan's map API.

    If nuPlan devkit is not installed or map extraction fails, this class
    returns empty map tensors. The rest of the pipeline still works with dynamic
    evidence, which is useful for smoke tests and direct DB preprocessing.
    """

    def __init__(self, map_root: str | Path | None, max_polylines: int = 256, max_points: int = 20):
        super().__init__(max_polylines=max_polylines, max_points=max_points)
        self.map_root = Path(map_root) if map_root else None
        self._apis: Dict[str, Any] = {}
        self.available = False
        try:
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # type: ignore
            from nuplan.common.maps.maps_datatypes import Point2D, SemanticMapLayer  # type: ignore
            self._get_maps_api = get_maps_api
            self._Point2D = Point2D
            self._Layer = SemanticMapLayer
            self.available = True
        except Exception:
            self.available = False

    def _api(self, map_name: str) -> Any:
        if map_name in self._apis:
            return self._apis[map_name]
        if not self.available or self.map_root is None:
            raise RuntimeError("nuPlan map API unavailable")
        # Common nuPlan map root contains a version directory. Most installs use nuplan-maps-v1.0.
        map_version = "nuplan-maps-v1.0"
        try:
            api = self._get_maps_api(str(self.map_root), map_version, map_name)
        except TypeError:
            api = self._get_maps_api(str(self.map_root), map_name)
        self._apis[map_name] = api
        return api

    @staticmethod
    def _coords_from_obj(obj: Any) -> Optional[np.ndarray]:
        candidates = []
        for attr in ("baseline_path", "centerline", "linestring", "polygon"):
            if hasattr(obj, attr):
                try:
                    candidates.append(getattr(obj, attr))
                except Exception:
                    pass
        candidates.append(obj)
        for cand in candidates:
            try:
                if hasattr(cand, "discrete_path"):
                    pts = cand.discrete_path
                    xy = np.asarray([[float(p.x), float(p.y)] for p in pts], dtype=np.float32)
                    if len(xy) >= 2:
                        return xy
                if hasattr(cand, "coords"):
                    xy = np.asarray(cand.coords, dtype=np.float32)[:, :2]
                    if len(xy) >= 2:
                        return xy
                if hasattr(cand, "exterior") and hasattr(cand.exterior, "coords"):
                    xy = np.asarray(cand.exterior.coords, dtype=np.float32)[:, :2]
                    if len(xy) >= 2:
                        return xy
            except Exception:
                continue
        return None

    def extract(self, map_name: str, ego_xy: np.ndarray, ego_yaw: float, radius_m: float) -> MapObjects:
        if not self.available:
            return super().extract(map_name, ego_xy, ego_yaw, radius_m)
        try:
            api = self._api(map_name)
            layer_names = [
                "LANE", "LANE_CONNECTOR", "ROADBLOCK", "ROADBLOCK_CONNECTOR",
                "CROSSWALK", "STOP_LINE", "WALKWAYS", "CARPARK_AREA",
            ]
            layers = []
            for name in layer_names:
                if hasattr(self._Layer, name):
                    layers.append(getattr(self._Layer, name))
            objects = api.get_proximal_map_objects(self._Point2D(float(ego_xy[0]), float(ego_xy[1])), radius_m, layers)
        except Exception:
            return super().extract(map_name, ego_xy, ego_yaw, radius_m)

        polylines = np.zeros((self.max_polylines, self.max_points, MAP_DIM), dtype=np.float32)
        masks = np.zeros((self.max_polylines, self.max_points), dtype=bool)
        rule_units: List[dict] = []
        n = 0
        try:
            iterable = []
            if isinstance(objects, dict):
                for layer, vals in objects.items():
                    for obj in vals:
                        iterable.append((str(layer), obj))
            else:
                iterable = [("unknown", obj) for obj in objects]
            for layer, obj in iterable:
                xy = self._coords_from_obj(obj)
                if xy is None or len(xy) < 2:
                    continue
                xy_e = global_to_ego_points(xy, ego_xy, ego_yaw)
                keep = xy_e[np.linspace(0, len(xy_e) - 1, min(len(xy_e), self.max_points)).astype(int)]
                if n < self.max_polylines:
                    count = min(len(keep), self.max_points)
                    polylines[n, :count, 0:2] = keep[:count]
                    polylines[n, :count, 2] = 1.0 if "LANE" in layer else 0.0
                    polylines[n, :count, 3] = 1.0 if any(k in layer for k in ("STOP", "CROSS", "WALK")) else 0.0
                    masks[n, :count] = True
                    n += 1
                if any(k in layer for k in ("STOP", "CROSS", "TRAFFIC", "WALK")):
                    centroid = xy_e.mean(axis=0)
                    rule_units.append({"layer": layer, "xy": centroid.astype(np.float32), "polyline": xy_e.astype(np.float32)})
        except Exception:
            pass
        return MapObjects(polylines=polylines, masks=masks, rule_units=rule_units)
