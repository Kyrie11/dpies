from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dpies.common.geometry import quaternion_yaw, transform_agent_state_global_to_ego


def token_to_str(token: Any) -> str:
    if token is None:
        return ""
    if isinstance(token, bytes):
        return token.hex()
    return str(token)


def stable_int(token: Any, mod: int = 2_147_483_647) -> int:
    h = hashlib.sha1(token_to_str(token).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


def _type_to_id(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, np.integer)):
        return int(value)
    s = str(value).lower()
    if "ped" in s:
        return 1
    if "bike" in s or "bicycle" in s or "cycl" in s:
        return 2
    if "vehicle" in s or "car" in s or "truck" in s or "bus" in s:
        return 3
    if "barrier" in s or "cone" in s or "traffic" in s:
        return 4
    return stable_int(s, mod=1000) + 10

def canonical_nuplan_map_name(value: str, db_name: str = "") -> str:
    s = str(value or "").strip().lower()
    d = str(db_name or "").strip().lower()
    combined = f"{s} {d}"

    if s in {
        "us-ma-boston",
        "us-nv-las-vegas-strip",
        "us-pa-pittsburgh-hazelwood",
        "sg-one-north",
    }:
        return s

    # Boston
    if (
            "boston" in combined
            or "seaport" in combined
            or "us-ma" in combined
            or "ma-boston" in combined
    ):
        return "us-ma-boston"

    # Las Vegas
    if (
            "vegas" in combined
            or "las_vegas" in combined
            or "las-vegas" in combined
            or "us-nv" in combined
            or "nv-las" in combined
    ):
        return "us-nv-las-vegas-strip"

    # Pittsburgh
    if (
            "pittsburgh" in combined
            or "hazelwood" in combined
            or "us-pa" in combined
            or "pa-pittsburgh" in combined
    ):
        return "us-pa-pittsburgh-hazelwood"

    # Singapore one-north
    if (
            "singapore" in combined
            or "one-north" in combined
            or "one_north" in combined
            or "onenorth" in combined
            or "sg-one" in combined
    ):
        return "sg-one-north"

    return s or "unknown"

class NuPlanSQLite:
    """Small direct SQLite reader for nuPlan DB files.

    This reader avoids a hard devkit dependency and uses schema introspection.
    The expensive lidar timestamp lookup is cached as an in-memory timeline; this
    replaces repeated `ORDER BY ABS(timestamp - ?)` full scans during preprocessing.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._tables: Optional[set[str]] = None
        self._columns: Dict[str, List[str]] = {}
        self._lidar_rows: Optional[List[sqlite3.Row]] = None
        self._lidar_times: Optional[np.ndarray] = None
        self._box_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._box_cols_cache: Optional[Dict[str, str]] = None

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "NuPlanSQLite":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def create_fast_indexes(self) -> None:
        """Best-effort indexes for writable DB copies. In-memory timestamp caching is used regardless."""
        try:
            token_col, ts_col, _ = self._lidar_pc_base_cols()
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_dpies_lidar_pc_ts ON lidar_pc({ts_col})")
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_dpies_lidar_pc_token ON lidar_pc({token_col})")
            if self.has_table("ego_pose"):
                pose_token = "token" if "token" in self.columns("ego_pose") else self.columns("ego_pose")[0]
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_dpies_ego_pose_token ON ego_pose({pose_token})")
            if self.has_table("lidar_box"):
                c = self._box_cols()
                if c.get("lidar_pc_token"):
                    self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_dpies_lidar_box_pc ON lidar_box({c['lidar_pc_token']})")
            self.conn.commit()
        except Exception as exc:
            print(f"[WARN] could not create sqlite indexes for {self.db_path.name}: {exc}")

    @property
    def tables(self) -> set[str]:
        if self._tables is None:
            rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            self._tables = {str(r[0]) for r in rows}
        return self._tables

    def columns(self, table: str) -> List[str]:
        if table not in self._columns:
            rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
            self._columns[table] = [str(r[1]) for r in rows]
        return self._columns[table]

    def has_table(self, table: str) -> bool:
        return table in self.tables

    def has_column(self, table: str, col: str) -> bool:
        return self.has_table(table) and col in self.columns(table)

    def describe(self) -> Dict[str, List[str]]:
        return {t: self.columns(t) for t in sorted(self.tables)}

    def _timestamp_col(self, table: str) -> str:
        cols = self.columns(table)
        for c in ("timestamp", "timestamp_us", "time_us"):
            if c in cols:
                return c
        raise KeyError(f"No timestamp column in table {table}: {cols}")

    def _lidar_pc_base_cols(self) -> Tuple[str, str, str]:
        cols = self.columns("lidar_pc")
        token_col = "token" if "token" in cols else cols[0]
        ts_col = self._timestamp_col("lidar_pc")
        ego_col = "ego_pose_token" if "ego_pose_token" in cols else "ego_pose_id" if "ego_pose_id" in cols else ""
        return token_col, ts_col, ego_col

    def _ensure_lidar_timeline(self) -> None:
        if self._lidar_rows is not None:
            return
        token_col, ts_col, ego_col = self._lidar_pc_base_cols()
        if not ego_col:
            raise KeyError("lidar_pc table has no ego_pose_token/ego_pose_id column")
        q = f"SELECT {token_col} AS token, {ts_col} AS timestamp_us, {ego_col} AS ego_pose_token FROM lidar_pc ORDER BY {ts_col} ASC"
        self._lidar_rows = list(self.conn.execute(q))
        self._lidar_times = np.asarray([int(r["timestamp_us"]) for r in self._lidar_rows], dtype=np.int64)

    def iter_lidar_pc_rows(self, sample_interval_s: float = 1.0, limit: Optional[int] = None) -> Iterable[sqlite3.Row]:
        self._ensure_lidar_timeline()
        assert self._lidar_rows is not None
        last_ts: Optional[int] = None
        emitted = 0
        min_delta = int(sample_interval_s * 1e6)
        for row in self._lidar_rows:
            ts = int(row["timestamp_us"])
            if last_ts is not None and ts - last_ts < min_delta:
                continue
            last_ts = ts
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    def lidar_rows_between(self, start_us: int, end_us: int) -> List[sqlite3.Row]:
        self._ensure_lidar_timeline()
        assert self._lidar_rows is not None and self._lidar_times is not None
        lo = int(np.searchsorted(self._lidar_times, int(start_us), side="left"))
        hi = int(np.searchsorted(self._lidar_times, int(end_us), side="right"))
        return self._lidar_rows[lo:hi]

    def nearest_lidar_row(self, timestamp_us: int) -> Optional[sqlite3.Row]:
        self._ensure_lidar_timeline()
        assert self._lidar_rows is not None and self._lidar_times is not None
        if len(self._lidar_rows) == 0:
            return None
        idx = int(np.searchsorted(self._lidar_times, int(timestamp_us), side="left"))
        if idx <= 0:
            return self._lidar_rows[0]
        if idx >= len(self._lidar_times):
            return self._lidar_rows[-1]
        left = idx - 1
        if abs(int(self._lidar_times[left]) - int(timestamp_us)) <= abs(int(self._lidar_times[idx]) - int(timestamp_us)):
            return self._lidar_rows[left]
        return self._lidar_rows[idx]

    def ego_pose_by_token(self, token: Any) -> Optional[sqlite3.Row]:
        if not self.has_table("ego_pose"):
            return None
        token_col = "token" if "token" in self.columns("ego_pose") else self.columns("ego_pose")[0]
        q = f"SELECT * FROM ego_pose WHERE {token_col}=? LIMIT 1"
        return self.conn.execute(q, (token,)).fetchone()

    def ego_state_from_pose_row(self, row: sqlite3.Row) -> np.ndarray:
        keys = set(row.keys())
        x = float(row["x"] if "x" in keys else row["pos_x"] if "pos_x" in keys else 0.0)
        y = float(row["y"] if "y" in keys else row["pos_y"] if "pos_y" in keys else 0.0)
        if {"qw", "qx", "qy", "qz"}.issubset(keys):
            yaw = quaternion_yaw(float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"]))
        elif "yaw" in keys:
            yaw = float(row["yaw"])
        else:
            yaw = 0.0
        vx = float(row["vx"] if "vx" in keys else row["velocity_x"] if "velocity_x" in keys else 0.0)
        vy = float(row["vy"] if "vy" in keys else row["velocity_y"] if "velocity_y" in keys else 0.0)
        ax = float(row["acceleration_x"] if "acceleration_x" in keys else row["accel_x"] if "accel_x" in keys else 0.0)
        ay = float(row["acceleration_y"] if "acceleration_y" in keys else row["accel_y"] if "accel_y" in keys else 0.0)
        yaw_rate = float(row["angular_rate_z"] if "angular_rate_z" in keys else row["yaw_rate"] if "yaw_rate" in keys else 0.0)
        speed = math.hypot(vx, vy)
        return np.asarray([x, y, yaw, vx, vy, ax, ay, yaw_rate, speed], dtype=np.float32)

    def ego_state_at_lidar_row(self, lidar_row: sqlite3.Row) -> Optional[np.ndarray]:
        pose = self.ego_pose_by_token(lidar_row["ego_pose_token"])
        if pose is None:
            return None
        return self.ego_state_from_pose_row(pose)

    def ego_series(self, center_us: int, seconds_before: float, seconds_after: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        target_offsets = np.arange(-seconds_before, seconds_after + 1e-6, dt, dtype=np.float32)
        target_times = (center_us + target_offsets * 1e6).astype(np.int64)
        rows = [self.nearest_lidar_row(int(t)) for t in target_times]
        states: List[np.ndarray] = []
        times: List[int] = []
        for r in rows:
            if r is None:
                continue
            st = self.ego_state_at_lidar_row(r)
            if st is None:
                continue
            states.append(st)
            times.append(int(r["timestamp_us"]))
        if len(states) != len(target_times):
            raise ValueError("missing ego history/future")
        return np.stack(states, axis=0), np.asarray(times, dtype=np.int64)

    def _box_cols(self) -> Dict[str, str]:
        if self._box_cols_cache is not None:
            return self._box_cols_cache
        if not self.has_table("lidar_box"):
            raise KeyError("No lidar_box table")
        cols = self.columns("lidar_box")
        def first(names: Sequence[str], default: str = "") -> str:
            for n in names:
                if n in cols:
                    return n
            return default
        self._box_cols_cache = {
            "lidar_pc_token": first(["lidar_pc_token", "lidar_pc_id"]),
            "track_token": first(["track_token", "track_id", "token"]),
            "x": first(["x", "center_x", "tx"]),
            "y": first(["y", "center_y", "ty"]),
            "yaw": first(["yaw", "heading"]),
            "length": first(["length", "size_x", "dx"], "length"),
            "width": first(["width", "size_y", "dy"], "width"),
            "vx": first(["vx", "velocity_x"], ""),
            "vy": first(["vy", "velocity_y"], ""),
            "type": first(["category_name", "tracked_object_type", "object_type", "type", "classification"], ""),
        }
        return self._box_cols_cache

    def boxes_at_lidar_token(self, lidar_token: Any) -> List[Dict[str, Any]]:
        key = token_to_str(lidar_token)
        if key in self._box_cache:
            return self._box_cache[key]
        if not self.has_table("lidar_box"):
            self._box_cache[key] = []
            return []
        c = self._box_cols()
        if not c["lidar_pc_token"]:
            self._box_cache[key] = []
            return []
        rows = self.conn.execute(f"SELECT * FROM lidar_box WHERE {c['lidar_pc_token']}=?", (lidar_token,)).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            keys = set(r.keys())
            try:
                x = float(r[c["x"]]) if c["x"] else 0.0
                y = float(r[c["y"]]) if c["y"] else 0.0
            except Exception:
                continue
            yaw = float(r[c["yaw"]]) if c["yaw"] else 0.0
            length = float(r[c["length"]]) if c["length"] in keys and r[c["length"]] is not None else 4.5
            width = float(r[c["width"]]) if c["width"] in keys and r[c["width"]] is not None else 2.0
            vx = float(r[c["vx"]]) if c["vx"] and c["vx"] in keys and r[c["vx"]] is not None else 0.0
            vy = float(r[c["vy"]]) if c["vy"] and c["vy"] in keys and r[c["vy"]] is not None else 0.0
            track_token = r[c["track_token"]] if c["track_token"] else r[0]
            typ = _type_to_id(r[c["type"]]) if c["type"] and c["type"] in keys else 0
            out.append({
                "track_token": track_token,
                "track_id": stable_int(track_token),
                "type_id": int(typ),
                "state": np.asarray([x, y, yaw, vx, vy, length, width, float(typ)], dtype=np.float32),
            })
        self._box_cache[key] = out
        return out

    def current_agents(self, lidar_row: sqlite3.Row, ego_state: np.ndarray, max_agents: int, radius_m: float) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        boxes = self.boxes_at_lidar_token(lidar_row["token"])
        if not boxes:
            return [], np.zeros((max_agents, 8), dtype=np.float32), np.zeros((max_agents,), dtype=bool)
        ego_xy = ego_state[:2]
        ego_yaw = float(ego_state[2])
        transformed: List[Tuple[float, Any, np.ndarray]] = []
        for b in boxes:
            st = transform_agent_state_global_to_ego(b["state"], ego_xy, ego_yaw)
            dist = float(np.linalg.norm(st[:2]))
            if dist <= radius_m:
                transformed.append((dist, b["track_token"], st))
        transformed.sort(key=lambda x: x[0])
        tokens = [t for _, t, _ in transformed[:max_agents]]
        arr = np.zeros((max_agents, 8), dtype=np.float32)
        mask = np.zeros((max_agents,), dtype=bool)
        for i, (_, _, st) in enumerate(transformed[:max_agents]):
            arr[i] = st
            mask[i] = True
        return tokens, arr, mask

    def _agent_series(self, center_us: int, tokens: Sequence[Any], ego_state: np.ndarray,
                      offsets_s: np.ndarray, return_mask: bool = False) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        times = (center_us + offsets_s.astype(np.float32) * 1e6).astype(np.int64)
        out = np.zeros((len(tokens), len(times), 8), dtype=np.float32)
        mask = np.zeros((len(tokens), len(times)), dtype=bool)
        token_keys = {token_to_str(t): i for i, t in enumerate(tokens)}
        ego_xy, ego_yaw = ego_state[:2], float(ego_state[2])
        for h, ts in enumerate(times):
            row = self.nearest_lidar_row(int(ts))
            if row is None:
                continue
            for b in self.boxes_at_lidar_token(row["token"]):
                idx = token_keys.get(token_to_str(b["track_token"]))
                if idx is not None:
                    out[idx, h] = transform_agent_state_global_to_ego(b["state"], ego_xy, ego_yaw)
                    mask[idx, h] = True
        return (out, mask) if return_mask else out

    def agent_history(self, center_us: int, tokens: Sequence[Any], ego_state: np.ndarray, history_s: float,
                      dt: float, return_mask: bool = False) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        offsets = np.arange(-history_s, 1e-6, dt, dtype=np.float32)
        return self._agent_series(center_us, tokens, ego_state, offsets, return_mask=return_mask)

    def agent_future(self, center_us: int, tokens: Sequence[Any], ego_state: np.ndarray, future_s: float,
                     dt: float, return_mask: bool = False) -> Tuple[np.ndarray, np.ndarray] | np.ndarray:
        offsets = np.arange(dt, future_s + 1e-6, dt, dtype=np.float32)
        return self._agent_series(center_us, tokens, ego_state, offsets, return_mask=return_mask)


    def route_roadblock_ids_for_lidar_token(self, lidar_token: Any) -> List[str]:
        """Return v1.1 route roadblock ids using the official devkit query when available."""
        token = token_to_str(lidar_token)
        try:
            from nuplan.database.nuplan_db.nuplan_scenario_queries import get_roadblock_ids_for_lidarpc_token_from_db  # type: ignore
            ids = get_roadblock_ids_for_lidarpc_token_from_db(str(self.db_path), token)
            if ids is None:
                return []
            return [str(x) for x in ids]
        except Exception:
            pass
        for table in ("lidar_pc", "scene", "scenario"):
            if not self.has_table(table):
                continue
            cols = self.columns(table)
            rb_cols = [c for c in cols if "roadblock" in c.lower() and "route" in c.lower()]
            token_cols = [c for c in cols if c in ("token", "lidar_pc_token", "initial_lidar_pc_token")]
            if not rb_cols or not token_cols:
                continue
            try:
                row = self.conn.execute(f"SELECT {rb_cols[0]} AS route FROM {table} WHERE {token_cols[0]}=? LIMIT 1", (lidar_token,)).fetchone()
                if row and row["route"]:
                    raw = row["route"]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8", errors="ignore")
                    return [x.strip() for x in str(raw).replace(";", ",").split(",") if x.strip()]
            except Exception:
                continue
        return []

    @staticmethod
    def _traffic_light_to_dict(raw: Any) -> Dict[str, Any]:
        def status_name(status: Any) -> str:
            if hasattr(status, "name"):
                return str(status.name).upper()
            s = str(status).split(".")[-1].upper()
            if s in {"0", "GREEN"}:
                return "GREEN"
            if s in {"1", "YELLOW"}:
                return "YELLOW"
            if s in {"2", "RED"}:
                return "RED"
            return s or "UNKNOWN"
        if isinstance(raw, dict):
            return {
                "status": status_name(raw.get("status")),
                "lane_connector_id": str(raw.get("lane_connector_id", raw.get("connector_id", ""))),
                "timestamp": int(raw.get("timestamp", raw.get("timestamp_us", 0)) or 0),
            }
        return {
            "status": status_name(getattr(raw, "status", None)),
            "lane_connector_id": str(getattr(raw, "lane_connector_id", "")),
            "timestamp": int(getattr(raw, "timestamp", 0) or 0),
        }

    def traffic_light_statuses_for_lidar_token(self, lidar_token: Any) -> List[Dict[str, Any]]:
        """Return current traffic light status records via the official nuPlan query."""
        token = token_to_str(lidar_token)
        try:
            from nuplan.database.nuplan_db.nuplan_scenario_queries import get_traffic_light_status_for_lidarpc_token_from_db  # type: ignore
            return [self._traffic_light_to_dict(x) for x in get_traffic_light_status_for_lidarpc_token_from_db(str(self.db_path), token)]
        except Exception:
            return []

    def future_traffic_light_status_history(self, center_us: int, future_s: float, dt: float) -> List[List[Dict[str, Any]]]:
        offsets = np.arange(dt, future_s + 1e-6, dt, dtype=np.float32)
        out: List[List[Dict[str, Any]]] = []
        for off in offsets:
            row = self.nearest_lidar_row(int(center_us + float(off) * 1e6))
            
            records = self.traffic_light_statuses_for_lidar_token(row["token"]) if row is not None else []
            for rec in records:
                rec["relative_time_s"] = float(off)
            out.append(records)
        return out

    def get_log_metadata(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"db_path": str(self.db_path), "db_name": self.db_path.name}

        if self.has_table("log"):
            rows = self.conn.execute("SELECT * FROM log LIMIT 1").fetchall()
            if rows:
                r = rows[0]
                for k in r.keys():
                    try:
                        v = r[k]
                        if isinstance(v, bytes):
                            v = v.hex()
                        meta[k] = v
                    except Exception:
                        pass

        # Build a rich context string. nuPlan DBs often expose location rather than map_name.
        candidates = []
        for k in (
                "map_name",
                "location",
                "city",
                "map_location",
                "logfile",
                "log_file",
                "log_name",
                "vehicle_name",
        ):
            v = meta.get(k, "")
            if v is not None:
                candidates.append(str(v))

        candidates.append(str(self.db_path.parent.name))
        candidates.append(str(self.db_path.name))

        raw_context = " ".join(candidates)
        raw_map_name = str(meta.get("map_name", "")) or str(meta.get("location", "")) or raw_context

        meta["raw_map_name"] = raw_map_name
        meta["map_name"] = canonical_nuplan_map_name(raw_map_name, raw_context)

        return meta
