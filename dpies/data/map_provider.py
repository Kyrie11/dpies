from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from dpies.common.geometry import global_to_ego_points
from dpies.common.types import MAP_DIM, MapRuleCode
from dpies.data.devkit_utils import (
    DevkitTrafficLightRecord,
    flatten_traffic_statuses,
    is_red_status,
    latest_status_by_connector,
    records_to_json,
    red_times_by_connector,
)


@dataclass
class MapObjects:
    polylines: np.ndarray
    masks: np.ndarray
    rule_units: List[dict]
    success: bool = False
    error: str = ""
    route_info: Dict[str, Any] = field(default_factory=dict)
    traffic_lights: List[dict] = field(default_factory=list)
    future_traffic_lights: List[dict] = field(default_factory=list)


class NullMapProvider:
    def __init__(self, max_polylines: int = 256, max_points: int = 20):
        self.max_polylines = max_polylines
        self.max_points = max_points

    def extract(self, map_name: str, ego_xy: np.ndarray, ego_yaw: float, radius_m: float, **_: Any) -> MapObjects:
        return MapObjects(
            polylines=np.zeros((self.max_polylines, self.max_points, MAP_DIM), dtype=np.float32),
            masks=np.zeros((self.max_polylines, self.max_points), dtype=bool),
            rule_units=[],
            success=False,
            error="nuPlan map API unavailable",
            route_info={},
        )

    def extract_from_api(self, map_api: Any, ego_xy: np.ndarray, ego_yaw: float, radius_m: float, **kwargs: Any) -> MapObjects:
        return self.extract("__api_unavailable__", ego_xy, ego_yaw, radius_m, **kwargs)


class NuPlanMapProvider(NullMapProvider):
    """nuPlan HD map extraction for DPIES.

    The class supports two entry points:
      * extract(map_name, ...): loads map_api from map_root/map_name;
      * extract_from_api(map_api, ...): uses the official devkit Scenario/Planner
        API map object directly, which is preferred for route/closed-loop use.

    Returned rule_units retain ego-frame geometry in JSON-serializable form so
    GeometryQuery can compute exact drivable-area, lane-boundary, stop-line,
    crosswalk, traffic-light, and route-deviation features offline and online.
    """

    def __init__(self, map_root: str | Path | None, max_polylines: int = 256, max_points: int = 20,
                 map_version: str = "nuplan-maps-v1.0"):
        super().__init__(max_polylines=max_polylines, max_points=max_points)
        self.map_root = Path(map_root) if map_root else None
        self.map_version = map_version
        self._apis: Dict[str, Any] = {}
        self._warned: set[str] = set()
        self.available = False
        self._init_error = ""
        try:
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
            from nuplan.common.actor_state.state_representation import Point2D
            from nuplan.common.maps.maps_datatypes import SemanticMapLayer

            self._get_maps_api = get_maps_api
            self._Point2D = Point2D
            self._Layer = SemanticMapLayer
            self.available = True
        except Exception as exc:
            self._get_maps_api = None
            self._Point2D = None
            self._Layer = None
            self._init_error = str(exc)
            self.available = False

    def _normalized_map_root(self) -> str:
        if self.map_root is None:
            raise RuntimeError("map_root is None")

        root = self.map_root

        # official get_maps_api expects the parent directory that contains map_version.
        # If user passes /.../maps/nuplan-maps-v1.0, use /.../maps.
        if root.name == self.map_version:
            root = root.parent

        return str(root)

    def _api(self, map_name: str) -> Any:
        map_name = str(map_name).strip()
        if map_name in self._apis:
            return self._apis[map_name]
        if not self.available or self.map_root is None:
            raise RuntimeError(f"nuPlan map API unavailable: {getattr(self, '_init_error', '')}")

        map_root = self._normalized_map_root()

        try:
            api = self._get_maps_api(map_root, self.map_version, map_name)
        except TypeError:
            api = self._get_maps_api(map_root, map_name)

        self._apis[map_name] = api
        return api

    @staticmethod
    def _obj_id(obj: Any) -> str:
        for attr in ("id", "token", "fid", "lane_id", "roadblock_id"):
            try:
                if hasattr(obj, attr):
                    return str(getattr(obj, attr))
            except Exception:
                pass
        return hashlib.sha1(repr(obj).encode("utf-8", errors="ignore")).hexdigest()[:16]

    @staticmethod
    def _roadblock_id(obj: Any) -> str | None:
        for attr in ("get_roadblock_id", "roadblock_id", "roadblock_token"):
            try:
                val = getattr(obj, attr)
                val = val() if callable(val) else val
                if val is not None:
                    return str(val)
            except Exception:
                pass
        return None

    @staticmethod
    def _speed_limit(obj: Any) -> float | None:
        for attr in ("speed_limit_mps", "speed_limit", "speed_limit_mps_or_none"):
            try:
                val = getattr(obj, attr)
                val = val() if callable(val) else val
                if val is not None:
                    return float(val)
            except Exception:
                pass
        return None

    @staticmethod
    def _coords_from_candidate(cand: Any) -> Optional[np.ndarray]:
        try:
            if cand is None:
                return None

            # Some nuPlan geometry accessors are methods.
            if callable(cand):
                cand = cand()

            # Shapely Polygon / LineString.
            if hasattr(cand, "exterior") and hasattr(cand.exterior, "coords"):
                xy = np.asarray(cand.exterior.coords, dtype=np.float32)[:, :2]
                return xy if len(xy) >= 3 else None

            if hasattr(cand, "coords"):
                xy = np.asarray(cand.coords, dtype=np.float32)[:, :2]
                return xy if len(xy) >= 2 else None

            # NuPlan BaselinePath-like.
            if hasattr(cand, "discrete_path"):
                pts = cand.discrete_path
                xy = np.asarray([[float(p.x), float(p.y)] for p in pts], dtype=np.float32)
                return xy if len(xy) >= 2 else None

            # Some path wrappers expose poses/states.
            for attr in ("poses", "states", "points"):
                if hasattr(cand, attr):
                    pts = getattr(cand, attr)
                    pts = pts() if callable(pts) else pts
                    xy = []
                    for p in pts:
                        if hasattr(p, "x") and hasattr(p, "y"):
                            xy.append([float(p.x), float(p.y)])
                        elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                            xy.append([float(p[0]), float(p[1])])
                    if len(xy) >= 2:
                        return np.asarray(xy, dtype=np.float32)

            # Raw list/tuple/ndarray of points.
            if isinstance(cand, np.ndarray):
                xy = np.asarray(cand, dtype=np.float32)
                if xy.ndim == 2 and xy.shape[1] >= 2 and len(xy) >= 2:
                    return xy[:, :2]

            if isinstance(cand, (list, tuple)):
                xy = []
                for p in cand:
                    if hasattr(p, "x") and hasattr(p, "y"):
                        xy.append([float(p.x), float(p.y)])
                    elif isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                        xy.append([float(p[0]), float(p[1])])
                if len(xy) >= 2:
                    return np.asarray(xy, dtype=np.float32)

        except Exception:
            return None

        return None

    @classmethod
    def _geometry_from_obj(cls, obj: Any, layer: str) -> tuple[str, Optional[np.ndarray]]:
        """Return ('polygon'|'polyline', global xy coords) when possible."""
        u = layer.upper()
        # Prefer true polygons for area-like objects; prefer lines for stop/boundary.
        polygon_attrs = (
            "polygon",
            "exterior_polygon",
        )
        line_attrs = (
            "baseline_path",
            "centerline",
            "linestring",
            "line_string",
            "left_boundary",
            "right_boundary",
        )
        if "STOP" in u:
            # nuPlan stop-line objects can be represented as polygon-like objects.
            # Try polygon attrs too, then line attrs.
            attr_order = polygon_attrs + line_attrs
        elif "BOUNDARY" in u:
            attr_order = line_attrs + polygon_attrs
        else:
            attr_order = polygon_attrs + line_attrs
        for attr in attr_order:
            try:
                if hasattr(obj, attr):
                    cand = getattr(obj, attr)
                    xy = cls._coords_from_candidate(cand)
                    if xy is not None:
                        kind = "polygon" if attr in polygon_attrs or (len(xy) >= 4 and np.allclose(xy[0], xy[-1])) else "polyline"
                        return kind, xy.astype(np.float32)
            except Exception:
                continue
        xy = cls._coords_from_candidate(obj)
        if xy is not None:
            kind = "polygon" if (len(xy) >= 4 and np.allclose(xy[0], xy[-1])) else "polyline"
            return kind, xy.astype(np.float32)
        return "point", None

    @staticmethod
    def _rule_code(layer: str, obj: Any | None = None) -> int:
        u = layer.upper()
        if "TRAFFIC" in u:
            return int(MapRuleCode.TRAFFIC_LIGHT_RED)
        if "STOP" in u:
            return int(MapRuleCode.STOP_LINE)
        if "CROSS" in u or "WALK" in u:
            return int(MapRuleCode.CROSSWALK)
        if "BOUNDARY" in u:
            return int(MapRuleCode.LANE_BOUNDARY)
        if "DRIVABLE" in u or "CARPARK" in u:
            return int(MapRuleCode.DRIVABLE_AREA)
        if "CONNECTOR" in u:
            return int(MapRuleCode.LANE_CONNECTOR)
        if "INTERSECTION" in u:
            return int(MapRuleCode.INTERSECTION)
        return int(MapRuleCode.NONE)

    def _layers_by_name(self, names: Sequence[str]) -> list[Any]:
        layers: list[Any] = []
        for name in names:
            if hasattr(self._Layer, name):
                layers.append(getattr(self._Layer, name))
        return layers

    def _layer_list(self) -> list[Any]:
        """Object-backed layers safe to query in nuPlan v1.1.

        Do not request DRIVABLE_AREA here. In official nuPlan v1.1 devkit,
        SemanticMapLayer.DRIVABLE_AREA may exist as an enum but often has no
        object representation through get_proximal_map_objects().
        """
        return self._layers_by_name([
            "LANE",
            "LANE_CONNECTOR",
            "ROADBLOCK",
            "ROADBLOCK_CONNECTOR",
            "CROSSWALK",
            "STOP_LINE",
            "CARPARK_AREA",
            "LANE_BOUNDARY",
            "TRAFFIC_LIGHT_CONNECTOR",
            "INTERSECTION",
        ])

    def _query_map_layer(self, map_api: Any, center: Any, radius_m: float, layer: Any) -> list[Any]:
        """Query one nuPlan semantic map layer and normalize return formats."""
        got = map_api.get_proximal_map_objects(center, radius_m, [layer])
        if isinstance(got, dict):
            vals = got.get(layer, None)
            if vals is None:
                # Some devkit versions / wrappers use layer.name or str(layer) keys.
                vals = got.get(getattr(layer, "name", str(layer)), [])
            return list(vals or [])
        return list(got or [])



    def _safe_get_proximal_map_objects(
            self,
            map_api: Any,
            ego_xy: np.ndarray,
            radius_m: float,
    ) -> dict[Any, list[Any]]:
        """Layer-wise map query for official nuPlan v1.1 devkit.

        One unsupported layer should not make the whole local HD map empty.
        """
        center = self._Point2D(float(ego_xy[0]), float(ego_xy[1]))
        objects: dict[Any, list[Any]] = {}

        for layer in self._layer_list():
            lname = getattr(layer, "name", str(layer))
            try:
                objects[layer] = self._query_map_layer(map_api, center, radius_m, layer)
            except Exception as exc:
                self._warn_once(
                    f"map_layer:{lname}",
                    f"Skipping nuPlan map layer {lname}: {exc}",
                )
                objects[layer] = []

        return objects

    @staticmethod
    def _ensure_closed_polygon(xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        if len(xy) >= 3 and not np.allclose(xy[0], xy[-1]):
            xy = np.concatenate([xy, xy[:1]], axis=0)
        return xy

    @staticmethod
    def _downsample_points(xy: np.ndarray, max_points: int) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float32)
        if max_points <= 0:
            return xy
        if len(xy) <= max_points:
            return xy
        idx = np.linspace(0, len(xy) - 1, max_points).astype(np.int64)
        return xy[idx].astype(np.float32)

    def _drivable_union_rule(self, polygons: list[np.ndarray]) -> dict[str, Any] | None:
        """Build local drivable-area evidence from polygon-backed nuPlan layers.

        Do not query SemanticMapLayer.DRIVABLE_AREA directly in nuPlan v1.1,
        because it can be raster-backed / non-object-backed. Instead, synthesize
        a local drivable proxy from lane, lane_connector, roadblock,
        roadblock_connector and carpark polygons.
        """
        clean: list[np.ndarray] = []
        for poly in polygons:
            if poly is None:
                continue
            arr = np.asarray(poly, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] < 2 or len(arr) < 3:
                continue
            clean.append(self._ensure_closed_polygon(arr[:, :2]))

        if not clean:
            return None

        pts = np.concatenate(clean, axis=0)
        xy = pts.mean(axis=0).astype(np.float32)

        return {
            "layer": "DRIVABLE_AREA_UNION",
            "rule_code": int(MapRuleCode.DRIVABLE_AREA),
            "xy": xy.astype(float).tolist(),
            "polyline": [],
            "polygon": [],
            "polygons": [p.astype(float).tolist() for p in clean[:128]],
            "geometry_type": "multi_polygon",
            "map_object_id": "drivable_area_union",
            "hard_keep": True,
        }

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            warnings.warn(msg, RuntimeWarning)
            self._warned.add(key)

    def extract(self, map_name: str, ego_xy: np.ndarray, ego_yaw: float, radius_m: float,
                route_roadblock_ids: Sequence[str] | None = None,
                traffic_lights: Sequence[Any] | None = None,
                future_traffic_lights: Sequence[Any] | None = None) -> MapObjects:
        if not self.available:
            err = getattr(self, "_init_error", "nuPlan map API unavailable")
            self._warn_once("unavailable", f"nuPlan map API unavailable; map tensors will be empty ({err})")
            obj = super().extract(map_name, ego_xy, ego_yaw, radius_m)
            obj.error = err
            return obj
        try:
            api = self._api(map_name)
        except Exception as exc:
            err = str(exc)
            self._warn_once(f"api:{map_name}", f"Map API creation failed for {map_name}: {err}")
            obj = super().extract(map_name, ego_xy, ego_yaw, radius_m)
            obj.error = err
            return obj
        return self.extract_from_api(
            api, ego_xy, ego_yaw, radius_m,
            route_roadblock_ids=route_roadblock_ids,
            traffic_lights=traffic_lights,
            future_traffic_lights=future_traffic_lights,
        )

    def extract_from_api(self, map_api: Any, ego_xy: np.ndarray, ego_yaw: float, radius_m: float,
                         route_roadblock_ids: Sequence[str] | None = None,
                         traffic_lights: Sequence[Any] | None = None,
                         future_traffic_lights: Sequence[Any] | None = None) -> MapObjects:
        if not self.available:
            # Closed-loop can pass an already-created map_api even when map_root is not set.
            try:
                from nuplan.common.actor_state.state_representation import Point2D  # type: ignore
                from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # type: ignore
                self._Point2D = Point2D
                self._Layer = SemanticMapLayer
                self.available = True
            except Exception as exc:
                obj = super().extract("__api_unavailable__", ego_xy, ego_yaw, radius_m)
                obj.error = str(exc)
                return obj
        try:
            objects = self._safe_get_proximal_map_objects(map_api, ego_xy, radius_m)
        except Exception as exc:
            err = str(exc)
            self._warn_once("extract_from_api", f"Map extraction from supplied API failed: {err}")
            obj = super().extract("__api_failed__", ego_xy, ego_yaw, radius_m)
            obj.error = err
            return obj

        current_tl = flatten_traffic_statuses(traffic_lights, relative_time_s=0.0)
        future_tl = flatten_traffic_statuses(future_traffic_lights, relative_time_s=None)
        status_by_connector = latest_status_by_connector(current_tl)
        red_times = red_times_by_connector(current_tl + future_tl)
        route_ids = {str(x) for x in (route_roadblock_ids or [])}

        polylines = np.zeros((self.max_polylines, self.max_points, MAP_DIM), dtype=np.float32)
        masks = np.zeros((self.max_polylines, self.max_points), dtype=bool)
        rule_units: List[dict] = []
        route_polygons: list[list[list[float]]] = []
        route_polyline_centers: list[list[list[float]]] = []
        drivable_polygons: list[np.ndarray] = []
        n = 0

        raw_layer_counts: dict[str, int] = {}
        geom_layer_counts: dict[str, int] = {}
        raw_object_count = 0
        geom_object_count = 0
        try:
            iterable: list[tuple[str, Any]] = []
            if isinstance(objects, dict):
                for layer, vals in objects.items():
                    lname = str(getattr(layer, "name", str(layer)))
                    vals = list(vals or [])
                    raw_layer_counts[lname] = len(vals)
                    raw_object_count += len(vals)
                    for obj in vals:
                        iterable.append((lname, obj))
            else:
                iterable = [("unknown", obj) for obj in objects]
                raw_layer_counts["unknown"] = len(iterable)
                raw_object_count = len(iterable)

            for layer, obj in iterable:
                obj_id = self._obj_id(obj)
                roadblock_id = self._roadblock_id(obj)
                is_route = bool(route_ids and (obj_id in route_ids or (roadblock_id is not None and roadblock_id in route_ids)))
                kind, xy_global = self._geometry_from_obj(obj, layer)
                if xy_global is None or len(xy_global) < 2:
                    continue

                geom_object_count += 1
                geom_layer_counts[layer] = geom_layer_counts.get(layer, 0) + 1
                xy_e_full = global_to_ego_points(xy_global, ego_xy, ego_yaw).astype(np.float32)

                if kind == "polygon":
                    xy_e_full = self._ensure_closed_polygon(xy_e_full)

                # Encoder tensor can stay compact.
                keep = self._downsample_points(xy_e_full, self.max_points)

                code = self._rule_code(layer, obj)
                speed_limit = self._speed_limit(obj)
                layer_upper = layer.upper()

                # Synthesize drivable area from polygon-backed map objects.
                if kind == "polygon" and (
                        "LANE" in layer_upper
                        or "ROADBLOCK" in layer_upper
                        or "CARPARK" in layer_upper
                        or "DRIVABLE" in layer_upper
                ):
                    drivable_polygons.append(xy_e_full)

                # Rule geometry should be denser than max_points=20, because GeometryQuery
                # uses it for exact intersections / containment.
                rule_poly = (
                    self._downsample_points(xy_e_full, 80)
                    if kind == "polygon"
                    else np.zeros((0, 2), dtype=np.float32)
                )
                rule_line = (
                    self._downsample_points(xy_e_full, 80)
                    if kind != "polygon"
                    else np.zeros((0, 2), dtype=np.float32)
                )

                if n < self.max_polylines:
                    count = min(len(keep), self.max_points)
                    polylines[n, :count, 0:2] = keep[:count]
                    polylines[n, :count, 2] = 1.0 if "LANE" in layer_upper else 0.0
                    polylines[n, :count, 3] = 1.0 if (code != int(MapRuleCode.NONE) or is_route) else 0.0
                    masks[n, :count] = True
                    n += 1

                if is_route:
                    if kind == "polygon" and len(xy_e_full) >= 3:
                        route_polygons.append(
                            self._downsample_points(xy_e_full, 80).astype(float).tolist()
                        )
                    else:
                        route_polyline_centers.append(
                            self._downsample_points(xy_e_full, 80).astype(float).tolist()
                        )

                # Traffic-light state is keyed by lane_connector_id in nuPlan. The map
                # does not expose physical signal geometry, so we attach the state to
                # the corresponding lane-connector geometry when available.
                lc_id = obj_id
                has_red = lc_id in red_times and len(red_times[lc_id]) > 0
                has_tl_state = lc_id in status_by_connector or has_red
                if has_tl_state and "LANE_CONNECTOR" in layer_upper:
                    status = status_by_connector.get(lc_id, "FUTURE_RED")
                    if has_red or is_red_status(status):
                        centroid = keep.mean(axis=0)
                        rule_units.append({
                            "layer": "TRAFFIC_LIGHT_CONNECTOR",
                            "rule_code": int(MapRuleCode.TRAFFIC_LIGHT_RED),
                            "xy": centroid.astype(float).tolist(),
                            "polyline": rule_line.astype(float).tolist() if len(rule_line) else keep.astype(float).tolist(),
                            "polygon": rule_poly.astype(float).tolist(),
                            "geometry_type": kind,
                            "map_object_id": obj_id,
                            "roadblock_id": roadblock_id,
                            "lane_connector_id": lc_id,
                            "traffic_light_status": status,
                            "red_times_s": [float(x) for x in red_times.get(lc_id, [0.0])],
                        })

                if code != int(MapRuleCode.NONE):
                    centroid = keep.mean(axis=0)
                    rule_units.append({
                        "layer": layer,
                        "rule_code": int(code),
                        "xy": centroid.astype(float).tolist(),
                        "polyline": rule_line.astype(float).tolist(),
                        "polygon": rule_poly.astype(float).tolist(),
                        "geometry_type": kind,
                        "map_object_id": obj_id,
                        "roadblock_id": roadblock_id,
                        "is_route": bool(is_route),
                    })
                if speed_limit is not None and ("LANE" in layer_upper or "CONNECTOR" in layer_upper):
                    centroid = keep.mean(axis=0)
                    rule_units.append({
                        "layer": "SPEED_LIMIT",
                        "rule_code": int(MapRuleCode.SPEED_LIMIT),
                        "xy": centroid.astype(float).tolist(),
                        "polyline": rule_line.astype(float).tolist(),
                        "polygon": rule_poly.astype(float).tolist(),
                        "geometry_type": kind,
                        "map_object_id": obj_id,
                        "speed_limit_mps": float(speed_limit),
                        "is_route": bool(is_route),
                    })

            drivable_rule = self._drivable_union_rule(drivable_polygons)
            if drivable_rule is not None:
                rule_units.append(drivable_rule)

            if route_polygons or route_polyline_centers:
                # A route-deviation unit gives GeometryQuery a single evidence unit
                # whose geometry is the on-route local corridor around ego.
                pts = np.asarray([p for poly in (route_polygons or route_polyline_centers) for p in poly], dtype=np.float32)
                xy = pts.mean(axis=0) if len(pts) else np.asarray([0.0, 0.0], dtype=np.float32)
                rule_units.append({
                    "layer": "ROUTE_CORRIDOR",
                    "rule_code": int(MapRuleCode.ROUTE_DEVIATION),
                    "xy": xy.astype(float).tolist(),
                    "polyline": [],
                    "polygon": [],
                    "polygons": route_polygons,
                    "polylines": route_polyline_centers,
                    "geometry_type": "multi_polygon" if route_polygons else "multi_polyline",
                    "map_object_id": "route_corridor",
                })
        except Exception as exc:
            return MapObjects(
                polylines=polylines, masks=masks, rule_units=rule_units,
                success=False, error=str(exc), route_info={},
                traffic_lights=records_to_json(current_tl), future_traffic_lights=records_to_json(future_tl),
            )

        route_info: dict[str, Any] = {
            "route_roadblock_ids": [str(x) for x in route_ids],
            "num_route_polygons_local": len(route_polygons),
            "num_route_polylines_local": len(route_polyline_centers),
            "route_polygons": route_polygons[:64],
            "route_polylines": route_polyline_centers[:64],
            "traffic_lights_current": records_to_json(current_tl),
            "traffic_lights_future": records_to_json(future_tl),
            "raw_map_object_count": int(raw_object_count),
            "geom_map_object_count": int(geom_object_count),
            "raw_layer_counts": raw_layer_counts,
            "geom_layer_counts": geom_layer_counts,
        }

        if n > 0:
            map_error = ""
        elif raw_object_count == 0:
            map_error = f"no raw proximal map objects returned; map_name={getattr(map_api, 'map_name', '')}; radius_m={radius_m}"
        else:
            map_error = f"raw map objects returned but geometry extraction failed; raw_layer_counts={raw_layer_counts}"

        return MapObjects(
            polylines=polylines,
            masks=masks,
            rule_units=rule_units,
            success=bool(n > 0),
            error=map_error,
            route_info=route_info,
            traffic_lights=records_to_json(current_tl),
            future_traffic_lights=records_to_json(future_tl),
        )
