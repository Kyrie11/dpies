from __future__ import annotations

import hashlib
import math
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from dpies.common.geometry import global_to_ego_points, quaternion_yaw, transform_state_global_to_ego, wrap_angle


def token_to_str(token: Any) -> str:
    if token is None:
        return ""
    if isinstance(token, bytes):
        return token.hex()
    return str(token)


def stable_int(token: Any, mod: int = 2_147_483_647) -> int:
    h = hashlib.sha1(token_to_str(token).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % mod


class NuPlanSQLite:
    """Small direct SQLite reader for nuPlan DB files.

    The official nuPlan devkit is still recommended when available, but this
    reader avoids a hard dependency during preprocessing. It uses schema
    introspection and conservative fallbacks because public nuPlan DB releases
    differ slightly in column naming.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._tables: Optional[set[str]] = None
        self._columns: Dict[str, List[str]] = {}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "NuPlanSQLite":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

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

    def iter_lidar_pc_rows(self, sample_interval_s: float = 1.0, limit: Optional[int] = None) -> Iterable[sqlite3.Row]:
        token_col, ts_col, ego_col = self._lidar_pc_base_cols()
        if not ego_col:
            raise KeyError("lidar_pc table has no ego_pose_token/ego_pose_id column")
        q = f"SELECT {token_col} AS token, {ts_col} AS timestamp_us, {ego_col} AS ego_pose_token FROM lidar_pc ORDER BY {ts_col} ASC"
        last_ts: Optional[int] = None
        emitted = 0
        min_delta = int(sample_interval_s * 1e6)
        for row in self.conn.execute(q):
            ts = int(row["timestamp_us"])
            if last_ts is not None and ts - last_ts < min_delta:
                continue
            last_ts = ts
            yield row
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    def lidar_rows_between(self, start_us: int, end_us: int) -> List[sqlite3.Row]:
        token_col, ts_col, ego_col = self._lidar_pc_base_cols()
        q = f"SELECT {token_col} AS token, {ts_col} AS timestamp_us, {ego_col} AS ego_pose_token FROM lidar_pc WHERE {ts_col} BETWEEN ? AND ? ORDER BY {ts_col} ASC"
        return list(self.conn.execute(q, (int(start_us), int(end_us))))

    def nearest_lidar_row(self, timestamp_us: int) -> Optional[sqlite3.Row]:
        token_col, ts_col, ego_col = self._lidar_pc_base_cols()
        q = f"SELECT {token_col} AS token, {ts_col} AS timestamp_us, {ego_col} AS ego_pose_token FROM lidar_pc ORDER BY ABS({ts_col} - ?) ASC LIMIT 1"
        return self.conn.execute(q, (int(timestamp_us),)).fetchone()

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
        return np.asarray([x, y, yaw, vx, vy, ax, ay, yaw_rate if yaw_rate != 0.0 else speed], dtype=np.float32)

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
        if not self.has_table("lidar_box"):
            raise KeyError("No lidar_box table")
        cols = self.columns("lidar_box")
        def first(names: Sequence[str], default: str = "") -> str:
            for n in names:
                if n in cols:
                    return n
            return default
        return {
            "lidar_pc_token": first(["lidar_pc_token", "lidar_pc_id"]),
            "track_token": first(["track_token", "track_id", "token"]),
            "x": first(["x", "center_x", "tx"]),
            "y": first(["y", "center_y", "ty"]),
            "yaw": first(["yaw", "heading"]),
            "length": first(["length", "size_x", "dx"], "length"),
            "width": first(["width", "size_y", "dy"], "width"),
            "vx": first(["vx", "velocity_x"], ""),
            "vy": first(["vy", "velocity_y"], ""),
        }

    def boxes_at_lidar_token(self, lidar_token: Any) -> List[Dict[str, Any]]:
        if not self.has_table("lidar_box"):
            return []
        c = self._box_cols()
        if not c["lidar_pc_token"]:
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
            length = float(r[c["length"]]) if c["length"] in keys else 4.5
            width = float(r[c["width"]]) if c["width"] in keys else 2.0
            vx = float(r[c["vx"]]) if c["vx"] and c["vx"] in keys and r[c["vx"]] is not None else 0.0
            vy = float(r[c["vy"]]) if c["vy"] and c["vy"] in keys and r[c["vy"]] is not None else 0.0
            track_token = r[c["track_token"]] if c["track_token"] else r[0]
            out.append({
                "track_token": track_token,
                "track_id": stable_int(track_token),
                "state": np.asarray([x, y, yaw, vx, vy, length, width, 0.0], dtype=np.float32),
            })
        return out

    def current_agents(self, lidar_row: sqlite3.Row, ego_state: np.ndarray, max_agents: int, radius_m: float) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        boxes = self.boxes_at_lidar_token(lidar_row["token"])
        if not boxes:
            return [], np.zeros((max_agents, 8), dtype=np.float32), np.zeros((max_agents,), dtype=bool)
        ego_xy = ego_state[:2]
        ego_yaw = float(ego_state[2])
        transformed: List[Tuple[float, Any, np.ndarray]] = []
        for b in boxes:
            st = transform_state_global_to_ego(b["state"], ego_xy, ego_yaw)
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

    def agent_history(self, center_us: int, tokens: Sequence[Any], ego_state: np.ndarray, history_s: float, dt: float) -> np.ndarray:
        times = (center_us + np.arange(-history_s, 1e-6, dt, dtype=np.float32) * 1e6).astype(np.int64)
        out = np.zeros((len(tokens), len(times), 8), dtype=np.float32)
        token_keys = {token_to_str(t): i for i, t in enumerate(tokens)}
        ego_xy, ego_yaw = ego_state[:2], float(ego_state[2])
        for h, ts in enumerate(times):
            row = self.nearest_lidar_row(int(ts))
            if row is None:
                continue
            for b in self.boxes_at_lidar_token(row["token"]):
                idx = token_keys.get(token_to_str(b["track_token"]))
                if idx is not None:
                    out[idx, h] = transform_state_global_to_ego(b["state"], ego_xy, ego_yaw)
        return out

    def agent_future(self, center_us: int, tokens: Sequence[Any], ego_state: np.ndarray, future_s: float, dt: float) -> np.ndarray:
        times = (center_us + np.arange(dt, future_s + 1e-6, dt, dtype=np.float32) * 1e6).astype(np.int64)
        out = np.zeros((len(tokens), len(times), 8), dtype=np.float32)
        token_keys = {token_to_str(t): i for i, t in enumerate(tokens)}
        ego_xy, ego_yaw = ego_state[:2], float(ego_state[2])
        for h, ts in enumerate(times):
            row = self.nearest_lidar_row(int(ts))
            if row is None:
                continue
            for b in self.boxes_at_lidar_token(row["token"]):
                idx = token_keys.get(token_to_str(b["track_token"]))
                if idx is not None:
                    out[idx, h] = transform_state_global_to_ego(b["state"], ego_xy, ego_yaw)
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
        if "map_name" not in meta:
            name = self.db_path.name.lower()
            if "boston" in name:
                meta["map_name"] = "us-ma-boston"
            elif "vegas" in name or "las" in name:
                meta["map_name"] = "us-nv-las-vegas-strip"
            elif "singapore" in name or "sg" in name:
                meta["map_name"] = "sg-one-north"
            elif "pittsburgh" in name or "pa" in name:
                meta["map_name"] = "us-pa-pittsburgh-hazelwood"
            else:
                meta["map_name"] = "unknown"
        return meta
