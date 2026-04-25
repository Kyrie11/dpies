from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

from dpies.actions.action_generator import ActionGenerator, ActionGeneratorConfig
from dpies.actions.coverage_metrics import min_ade_fde
from dpies.common.geometry import ego_pose_to_transform, transform_ego_state_global_to_ego
from dpies.common.io import ensure_dir, write_json
from dpies.data.map_provider import NuPlanMapProvider, NullMapProvider
from dpies.data.nuplan_db import NuPlanSQLite, stable_int, token_to_str
from dpies.data.scenario_index import find_db_files
from dpies.data.scenario_api import ScenarioAPIExtractor, traffic_light_history_to_json, traffic_light_list_to_json
from dpies.evidence.evidence_builder import EvidenceBuilder, EvidenceBuilderConfig
from dpies.evidence.geometry_query import compute_geometry_query
from dpies.teacher.labels import oracle_action, rival_labels, signed_evidence_active_mask, signed_evidence_labels
from dpies.teacher.local_costs import local_teacher_contribution
from dpies.teacher.teacher_evaluator import TeacherEvaluator


def ego_series_to_local(series: np.ndarray, current: np.ndarray) -> np.ndarray:
    ego_xy, ego_yaw = current[:2], float(current[2])
    rel = transform_ego_state_global_to_ego(series, ego_xy, ego_yaw)
    return rel.astype(np.float32)


def _pad_agents(arr: np.ndarray, mask: np.ndarray, max_agents: int, steps: int) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((max_agents, steps, 8), dtype=np.float32)
    out_mask = np.zeros((max_agents, steps), dtype=bool)
    n = min(len(arr), max_agents)
    if n:
        out[:n, :arr.shape[1]] = arr[:n, :steps]
        out_mask[:n, :mask.shape[1]] = mask[:n, :steps]
    return out, out_mask


def make_sample(db: NuPlanSQLite, lidar_row: Any, args: argparse.Namespace, map_provider: NuPlanMapProvider,
                action_gen: ActionGenerator, evidence_builder: EvidenceBuilder,
                teacher: TeacherEvaluator, scenario_extractor: ScenarioAPIExtractor | None = None) -> Dict[str, np.ndarray] | None:
    center_us = int(lidar_row["timestamp_us"])
    current = db.ego_state_at_lidar_row(lidar_row)
    if current is None:
        return None
    hist_global, _ = db.ego_series(center_us, args.history_seconds, 0.0, args.dt)
    future_global, _ = db.ego_series(center_us, 0.0, args.future_seconds, args.dt)
    logged_future = ego_series_to_local(future_global[1:], current)
    ego_history = ego_series_to_local(hist_global, current)

    tokens, current_agents, agent_mask = db.current_agents(lidar_row, current, args.max_agents, args.agent_radius_m)
    agent_hist_small, agent_hist_mask_small = db.agent_history(center_us, tokens, current, args.history_seconds, args.dt, return_mask=True)
    agent_future_small, agent_future_mask_small = db.agent_future(center_us, tokens, current, args.future_seconds, args.dt, return_mask=True)
    hist_steps = int(round(args.history_seconds / args.dt)) + 1
    fut_steps = int(round(args.future_seconds / args.dt))
    agent_history, agent_history_mask = _pad_agents(agent_hist_small, agent_hist_mask_small, args.max_agents, hist_steps)
    agent_future, agent_future_mask = _pad_agents(agent_future_small, agent_future_mask_small, args.max_agents, fut_steps)
    agent_track_id = np.zeros((args.max_agents,), dtype=np.int64)
    agent_type = np.zeros((args.max_agents,), dtype=np.int64)
    for i, tok in enumerate(tokens[:args.max_agents]):
        agent_track_id[i] = stable_int(tok)
        if i < current_agents.shape[0]:
            agent_type[i] = int(current_agents[i, 7])

    meta = db.get_log_metadata()
    map_name = str(meta.get("map_name", "unknown"))
    route_roadblock_ids = db.route_roadblock_ids_for_lidar_token(lidar_row["token"])
    traffic_lights_current = db.traffic_light_statuses_for_lidar_token(lidar_row["token"])
    traffic_lights_future = db.future_traffic_light_status_history(center_us, args.future_seconds, args.dt)
    scenario_ctx = None
    if scenario_extractor is not None and scenario_extractor.available:
        scenario_ctx = scenario_extractor.extract_for_lidar_row(db.db_path, lidar_row, map_name, args.future_seconds, fut_steps)
        if not scenario_ctx.error:
            if scenario_ctx.route_roadblock_ids:
                route_roadblock_ids = scenario_ctx.route_roadblock_ids
            traffic_lights_current = traffic_light_list_to_json(scenario_ctx.traffic_lights_current)
            traffic_lights_future = traffic_light_history_to_json(scenario_ctx.traffic_lights_future)

    if args.disable_map:
        map_obj = NullMapProvider(args.max_map_polylines, args.max_map_points).extract("__disabled__", current[:2], float(current[2]), args.map_radius_m)
        map_obj.error = "map disabled by --disable-map"
    elif scenario_ctx is not None and scenario_ctx.map_api is not None and not scenario_ctx.error:
        map_obj = map_provider.extract_from_api(
            scenario_ctx.map_api, current[:2], float(current[2]), args.map_radius_m,
            route_roadblock_ids=route_roadblock_ids,
            traffic_lights=scenario_ctx.traffic_lights_current,
            future_traffic_lights=scenario_ctx.traffic_lights_future,
        )
    else:
        map_obj = map_provider.extract(
            map_name, current[:2], float(current[2]), args.map_radius_m,
            route_roadblock_ids=route_roadblock_ids,
            traffic_lights=traffic_lights_current,
            future_traffic_lights=traffic_lights_future,
        )
    map_obj.route_info.setdefault("route_roadblock_ids", [str(x) for x in route_roadblock_ids])
    map_obj.route_info.setdefault("traffic_lights_current", traffic_lights_current)
    map_obj.route_info["traffic_lights_future"] = traffic_lights_future
    if scenario_ctx is not None and scenario_ctx.error:
        map_obj.route_info["scenario_api_error"] = scenario_ctx.error
    if args.require_map and not map_obj.success:
        if args.continue_on_error:
            return None
        raise RuntimeError(f"required map extraction failed: {map_obj.error}")

    actions, action_meta, action_mask = action_gen.generate(
        ego_history,
        agent_history=agent_history,
        agent_mask=agent_mask,
        map_context=map_obj,
        rule_units=map_obj.rule_units,
        traffic_lights=traffic_lights_current,
    )
    if not action_mask.any():
        return None

    evidence_features, evidence_type, evidence_cost, evidence_mask = evidence_builder.build(
        agent_history, agent_mask, actions, action_mask, rule_units=map_obj.rule_units,
        dt=args.dt, agent_history_mask=agent_history_mask,
    )
    evidence_metadata = list(evidence_builder.last_metadata)
    geometry_query = compute_geometry_query(
        evidence_features, evidence_type, actions, evidence_mask, action_mask, args.dt,
        evidence_metadata=evidence_metadata, route_info=map_obj.route_info,
    )
    teacher_geometry_query = compute_geometry_query(
        evidence_features, evidence_type, actions, evidence_mask, action_mask, args.dt,
        future_agents=agent_future, future_agent_mask=agent_future_mask, evidence_metadata=evidence_metadata, route_info=map_obj.route_info,
        use_future_traffic=True,
    )
    local_cost = local_teacher_contribution(evidence_features, evidence_type, teacher_geometry_query, evidence_mask, action_mask)
    teacher_cost, teacher_components = teacher.evaluate_with_components(
        actions, action_mask, logged_future, agent_future, agent_mask,
        evidence_features, evidence_type, evidence_mask, teacher_geometry_query,
        agent_future_mask=agent_future_mask, dt=args.dt,
    )
    oracle = oracle_action(teacher_cost, action_mask)
    rival = rival_labels(teacher_cost, action_mask, args.rival_top_rank_l, args.rival_margin_delta)
    signed = signed_evidence_labels(local_cost, action_mask, args.s_max)
    active = signed_evidence_active_mask(local_cost, teacher_geometry_query, action_mask, evidence_mask,
                                         args.active_cost_threshold, args.active_query_threshold)
    ade, fde = min_ade_fde(actions, action_mask, logged_future)
    local_sum = local_cost.sum(axis=0).astype(np.float32)
    sample_meta = {
        "scenario_id": f"{Path(db.db_path).stem}_{token_to_str(lidar_row['token'])[:16]}",
        "db_path": str(db.db_path),
        "timestamp_us": center_us,
        "map_name": map_name,
        "map_success": bool(map_obj.success),
        "map_error": str(map_obj.error),
        "num_map_polylines": int(map_obj.masks.any(axis=1).sum()),
        "num_rule_units": int(len(map_obj.rule_units)),
        "num_route_roadblocks": int(len(route_roadblock_ids)),
        "num_traffic_lights_current": int(len(traffic_lights_current)),
        "num_traffic_light_future_steps": int(len(traffic_lights_future)),
        "oracle_action_index": int(oracle),
        "min_ade": float(ade),
        "min_fde": float(fde),
        "candidate_oracle_teacher_cost": float(np.min(teacher_cost[action_mask])),
        "valid_action_count": int(action_mask.sum()),
        "evidence_count": int(evidence_mask.sum()),
        "dt": float(args.dt),
        "ego_dim": int(ego_history.shape[-1]),
    }
    return {
        "ego_history": ego_history.astype(np.float32),
        "ego_global_state": current.astype(np.float32),
        "ego_to_global": ego_pose_to_transform(current[:2], float(current[2])).astype(np.float32),
        "agent_history": agent_history.astype(np.float32),
        "agent_history_mask": agent_history_mask.astype(bool),
        "agent_future": agent_future.astype(np.float32),
        "agent_future_mask": agent_future_mask.astype(bool),
        "agent_mask": agent_mask.astype(bool),
        "agent_track_id": agent_track_id.astype(np.int64),
        "agent_type": agent_type.astype(np.int64),
        "map_polylines": map_obj.polylines.astype(np.float32),
        "map_masks": map_obj.masks.astype(bool),
        "actions": actions.astype(np.float32),
        "action_meta": action_meta.astype(np.float32),
        "action_mask": action_mask.astype(bool),
        "evidence_features": evidence_features.astype(np.float32),
        "evidence_type": evidence_type.astype(np.int64),
        "evidence_cost": evidence_cost.astype(np.float32),
        "evidence_mask": evidence_mask.astype(bool),
        "geometry_query": geometry_query.astype(np.float32),
        "teacher_cost": teacher_cost.astype(np.float32),
        "teacher_components": teacher_components.astype(np.float32),
        "local_cost_sum": local_sum,
        "oracle_action_index": np.asarray(oracle, dtype=np.int64),
        "rival_label": rival.astype(bool),
        "signed_evidence_label": signed.astype(np.float32),
        "signed_evidence_mask": active.astype(bool),
        "logged_ego_future": logged_future.astype(np.float32),
        "metadata_json": json_bytes(sample_meta),
        "evidence_metadata_json": json_bytes(evidence_metadata),
        "route_info_json": json_bytes(map_obj.route_info),
        "traffic_lights_json": json_bytes({"current": traffic_lights_current, "future": traffic_lights_future}),
    }


def save_sample(path: Path, sample: Dict[str, np.ndarray], compress: bool = False) -> None:
    if compress:
        np.savez_compressed(path, **sample)
    else:
        np.savez(path, **sample)

def json_bytes(obj: object) -> np.ndarray:
    return np.asarray(json.dumps(obj, ensure_ascii=False).encode("utf-8"), dtype=np.bytes_)

def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess nuPlan DB files into DPIES training cache.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--map-root", default=None)
    p.add_argument("--output-dir", "--cache-dir", dest="output_dir", required=True)
    p.add_argument("--subdirs", nargs="*", default=None)
    p.add_argument("--max-dbs", type=int, default=None)
    p.add_argument("--max-samples-per-db", type=int, default=None)
    p.add_argument("--max-samples-total", type=int, default=None)
    p.add_argument("--sample-interval-s", type=float, default=1.0)
    p.add_argument("--history-seconds", type=float, default=2.0)
    p.add_argument("--future-seconds", type=float, default=8.0)
    p.add_argument("--dt", type=float, default=0.5)
    p.add_argument("--max-agents", type=int, default=64)
    p.add_argument("--agent-radius-m", type=float, default=80.0)
    p.add_argument("--max-actions", type=int, default=32)
    p.add_argument("--max-evidence-units", type=int, default=128)
    p.add_argument("--max-map-polylines", type=int, default=256)
    p.add_argument("--max-map-points", type=int, default=20)
    p.add_argument("--map-radius-m", type=float, default=80.0)
    p.add_argument("--map-version", default="nuplan-maps-v1.0")
    p.add_argument("--sensor-root", default=None)
    p.add_argument("--use-scenario-api", action="store_true", help="use official NuPlanScenario API for route, traffic-light, and map extraction")
    p.add_argument("--require-map", action="store_true", help="skip/raise when HD-map extraction fails instead of silently using empty maps")
    p.add_argument("--disable-map", action="store_true", help="force empty maps for fast dynamic-only debugging")
    p.add_argument("--rival-top-rank-l", type=int, default=8)
    p.add_argument("--rival-margin-delta", type=float, default=0.5)
    p.add_argument("--s-max", type=float, default=10.0)
    p.add_argument("--active-cost-threshold", type=float, default=0.05)
    p.add_argument("--active-query-threshold", type=float, default=0.1)
    p.add_argument("--compress", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="do not overwrite existing sample_*.npz when resuming an interrupted cache build")
    p.add_argument("--create-sqlite-indexes", action="store_true", help="best-effort CREATE INDEX for writable DB copies; in-memory caching is always used")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    out_dir = ensure_dir(args.output_dir)
    db_files = find_db_files(args.data_root, args.subdirs, args.max_dbs)
    if not db_files:
        raise FileNotFoundError(f"No .db files found under {args.data_root} with subdirs={args.subdirs}")
    write_json(out_dir / "preprocess_args.json", vars(args) | {"num_db_files": len(db_files)})
    map_provider = NuPlanMapProvider(args.map_root, args.max_map_polylines, args.max_map_points, map_version=args.map_version)
    scenario_extractor = None
    if args.use_scenario_api:
        if args.map_root is None:
            raise ValueError("--use-scenario-api requires --map-root")
        scenario_extractor = ScenarioAPIExtractor(args.data_root, args.map_root, map_version=args.map_version, sensor_root=args.sensor_root)
        if not scenario_extractor.available:
            raise ImportError(f"--use-scenario-api requested, but nuPlan devkit is unavailable: {scenario_extractor._init_error}")
    action_gen = ActionGenerator(ActionGeneratorConfig(max_actions=args.max_actions, horizon_s=args.future_seconds, dt=args.dt))
    evidence_builder = EvidenceBuilder(EvidenceBuilderConfig(max_units=args.max_evidence_units, radius_m=args.agent_radius_m))
    teacher = TeacherEvaluator(dt=args.dt)

    manifest_rows = []
    existing = sorted(out_dir.glob("sample_*.npz")) if args.skip_existing else []
    sample_id = len(existing) if args.skip_existing else 0
    failures = 0
    skipped_missing_map = 0
    for db_path in tqdm(db_files, desc="db"):
        if args.max_samples_total is not None and sample_id >= args.max_samples_total:
            break
        try:
            with NuPlanSQLite(db_path) as db:
                if args.create_sqlite_indexes:
                    db.create_fast_indexes()
                emitted = 0
                for lidar_row in db.iter_lidar_pc_rows(args.sample_interval_s, args.max_samples_per_db):
                    if args.max_samples_total is not None and sample_id >= args.max_samples_total:
                        break
                    try:
                        sample = make_sample(db, lidar_row, args, map_provider, action_gen, evidence_builder, teacher, scenario_extractor)
                        if sample is None:
                            if args.require_map:
                                skipped_missing_map += 1
                            continue
                        name = f"sample_{sample_id:09d}.npz"
                        if args.skip_existing and (out_dir / name).exists():
                            sample_id += 1
                            continue
                        save_sample(out_dir / name, sample, args.compress)
                        meta = json.loads(str(sample["metadata_json"]))
                        meta["file"] = name
                        manifest_rows.append(meta)
                        sample_id += 1
                        emitted += 1
                    except Exception as exc:
                        failures += 1
                        if not args.continue_on_error:
                            raise
                        if failures <= 10:
                            print(f"[WARN] failed sample in {db_path}: {exc}")
                tqdm.write(f"{db_path.name}: wrote {emitted} samples")
        except Exception as exc:
            failures += 1
            if not args.continue_on_error:
                traceback.print_exc()
                raise
            print(f"[WARN] failed db {db_path}: {exc}")
    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for row in manifest_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out_dir / "summary.json", {
        "num_samples": sample_id,
        "num_failures": failures,
        "num_dbs": len(db_files),
        "skipped_missing_map": skipped_missing_map,
    })
    print(f"Wrote {sample_id} samples to {out_dir}; failures={failures}; skipped_missing_map={skipped_missing_map}")


if __name__ == "__main__":
    main()
