from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

from dpies.actions.action_generator import ActionGenerator, ActionGeneratorConfig
from dpies.actions.coverage_metrics import min_ade_fde
from dpies.common.geometry import transform_state_global_to_ego
from dpies.common.io import ensure_dir, write_json
from dpies.data.map_provider import NuPlanMapProvider
from dpies.data.nuplan_db import NuPlanSQLite, token_to_str
from dpies.data.scenario_index import find_db_files
from dpies.evidence.evidence_builder import EvidenceBuilder, EvidenceBuilderConfig
from dpies.evidence.geometry_query import compute_geometry_query
from dpies.teacher.labels import oracle_action, rival_labels, signed_evidence_active_mask, signed_evidence_labels
from dpies.teacher.local_costs import local_teacher_contribution
from dpies.teacher.teacher_evaluator import TeacherEvaluator


def ego_series_to_local(series: np.ndarray, current: np.ndarray) -> np.ndarray:
    out = np.zeros((len(series), 8), dtype=np.float32)
    ego_xy, ego_yaw = current[:2], float(current[2])
    rel = transform_state_global_to_ego(series, ego_xy, ego_yaw)
    out[:, :7] = rel[:, :7]
    speed = np.linalg.norm(rel[:, 3:5], axis=-1)
    raw_speed = np.where(np.abs(series[:, 7]) > 0.0, series[:, 7], speed)
    out[:, 7] = raw_speed.astype(np.float32)
    return out


def make_sample(db: NuPlanSQLite, lidar_row: Any, args: argparse.Namespace, map_provider: NuPlanMapProvider,
                action_gen: ActionGenerator, evidence_builder: EvidenceBuilder,
                teacher: TeacherEvaluator) -> Dict[str, np.ndarray] | None:
    center_us = int(lidar_row["timestamp_us"])
    current = db.ego_state_at_lidar_row(lidar_row)
    if current is None:
        return None
    # Ego history and future.
    hist_global, _ = db.ego_series(center_us, args.history_seconds, 0.0, args.dt)
    future_global, _ = db.ego_series(center_us, 0.0, args.future_seconds, args.dt)
    # future_global includes t=0 as first entry; labels use future steps only.
    logged_future = ego_series_to_local(future_global[1:], current)
    ego_history = ego_series_to_local(hist_global, current)
    # Agents.
    tokens, current_agents, agent_mask = db.current_agents(lidar_row, current, args.max_agents, args.agent_radius_m)
    agent_hist_small = db.agent_history(center_us, tokens, current, args.history_seconds, args.dt)
    agent_future_small = db.agent_future(center_us, tokens, current, args.future_seconds, args.dt)
    agent_history = np.zeros((args.max_agents, int(round(args.history_seconds / args.dt)) + 1, 8), dtype=np.float32)
    agent_future = np.zeros((args.max_agents, int(round(args.future_seconds / args.dt)), 8), dtype=np.float32)
    if len(tokens):
        agent_history[:len(tokens)] = agent_hist_small[:args.max_agents]
        agent_future[:len(tokens)] = agent_future_small[:args.max_agents]
    # Map.
    meta = db.get_log_metadata()
    map_obj = map_provider.extract(str(meta.get("map_name", "unknown")), current[:2], float(current[2]), args.map_radius_m)
    # Actions.
    actions, action_meta, action_mask = action_gen.generate(ego_history)
    if not action_mask.any():
        return None
    # Evidence and queries.
    evidence_features, evidence_type, evidence_cost, evidence_mask = evidence_builder.build(
        agent_history, agent_mask, actions, action_mask, rule_units=map_obj.rule_units
    )
    geometry_query = compute_geometry_query(evidence_features, evidence_type, actions, evidence_mask, action_mask, args.dt)
    teacher_geometry_query = compute_geometry_query(
        evidence_features, evidence_type, actions, evidence_mask, action_mask, args.dt, future_agents=agent_future
    )
    local_cost = local_teacher_contribution(evidence_features, evidence_type, teacher_geometry_query, evidence_mask, action_mask)
    teacher_cost = teacher.evaluate(actions, action_mask, logged_future, agent_future, agent_mask,
                                    evidence_features, evidence_type, evidence_mask, teacher_geometry_query)
    oracle = oracle_action(teacher_cost, action_mask)
    rival = rival_labels(teacher_cost, action_mask, args.rival_top_rank_l, args.rival_margin_delta)
    signed = signed_evidence_labels(local_cost, action_mask, args.s_max)
    active = signed_evidence_active_mask(local_cost, teacher_geometry_query, action_mask, evidence_mask,
                                         args.active_cost_threshold, args.active_query_threshold)
    ade, fde = min_ade_fde(actions, action_mask, logged_future)
    sample_meta = {
        "scenario_id": f"{Path(db.db_path).stem}_{token_to_str(lidar_row['token'])[:16]}",
        "db_path": str(db.db_path),
        "timestamp_us": center_us,
        "map_name": str(meta.get("map_name", "unknown")),
        "oracle_action_index": int(oracle),
        "min_ade": float(ade),
        "min_fde": float(fde),
        "valid_action_count": int(action_mask.sum()),
        "evidence_count": int(evidence_mask.sum()),
    }
    return {
        "ego_history": ego_history.astype(np.float32),
        "agent_history": agent_history.astype(np.float32),
        "agent_mask": agent_mask.astype(bool),
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
        "oracle_action_index": np.asarray(oracle, dtype=np.int64),
        "rival_label": rival.astype(bool),
        "signed_evidence_label": signed.astype(np.float32),
        "signed_evidence_mask": active.astype(bool),
        "logged_ego_future": logged_future.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(sample_meta), dtype="<U4096"),
    }


def save_sample(path: Path, sample: Dict[str, np.ndarray], compress: bool = False) -> None:
    if compress:
        np.savez_compressed(path, **sample)
    else:
        np.savez(path, **sample)


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess nuPlan DB files into DPIES training cache.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--map-root", default=None)
    p.add_argument("--output-dir", "--cache-dir", dest="output_dir", required=True)
    p.add_argument("--subdirs", nargs="*", default=None)
    p.add_argument("--max-dbs", type=int, default=None)
    p.add_argument("--max-samples-per-db", type=int, default=None)
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
    p.add_argument("--rival-top-rank-l", type=int, default=8)
    p.add_argument("--rival-margin-delta", type=float, default=0.5)
    p.add_argument("--s-max", type=float, default=10.0)
    p.add_argument("--active-cost-threshold", type=float, default=0.05)
    p.add_argument("--active-query-threshold", type=float, default=0.1)
    p.add_argument("--compress", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    args = p.parse_args()

    out_dir = ensure_dir(args.output_dir)
    db_files = find_db_files(args.data_root, args.subdirs, args.max_dbs)
    if not db_files:
        raise FileNotFoundError(f"No .db files found under {args.data_root} with subdirs={args.subdirs}")
    write_json(out_dir / "preprocess_args.json", vars(args) | {"num_db_files": len(db_files)})
    map_provider = NuPlanMapProvider(args.map_root, args.max_map_polylines, args.max_map_points)
    action_gen = ActionGenerator(ActionGeneratorConfig(max_actions=args.max_actions, horizon_s=args.future_seconds, dt=args.dt))
    evidence_builder = EvidenceBuilder(EvidenceBuilderConfig(max_units=args.max_evidence_units, radius_m=args.agent_radius_m))
    teacher = TeacherEvaluator()

    manifest_rows = []
    sample_id = 0
    failures = 0
    for db_path in tqdm(db_files, desc="db"):
        try:
            with NuPlanSQLite(db_path) as db:
                emitted = 0
                for lidar_row in db.iter_lidar_pc_rows(args.sample_interval_s, args.max_samples_per_db):
                    try:
                        sample = make_sample(db, lidar_row, args, map_provider, action_gen, evidence_builder, teacher)
                        if sample is None:
                            continue
                        name = f"sample_{sample_id:09d}.npz"
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
    write_json(out_dir / "summary.json", {"num_samples": sample_id, "num_failures": failures, "num_dbs": len(db_files)})
    print(f"Wrote {sample_id} samples to {out_dir}; failures={failures}")


if __name__ == "__main__":
    main()
