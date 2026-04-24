from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

from dpies.actions.action_generator import ActionGenerator, ActionGeneratorConfig
from dpies.common.io import ensure_dir, write_json
from dpies.evidence.evidence_builder import EvidenceBuilder, EvidenceBuilderConfig
from dpies.evidence.geometry_query import compute_geometry_query
from dpies.teacher.labels import oracle_action, rival_labels, signed_evidence_active_mask, signed_evidence_labels
from dpies.teacher.local_costs import local_teacher_contribution
from dpies.teacher.teacher_evaluator import TeacherEvaluator


def make_one(seed: int, max_agents: int, max_actions: int, max_evidence: int, dt: float, hist_s: float, fut_s: float):
    rng = np.random.default_rng(seed)
    h = int(round(hist_s / dt)) + 1
    t = int(round(fut_s / dt))
    ego_history = np.zeros((h, 8), dtype=np.float32)
    speed = rng.uniform(2.0, 8.0)
    times = np.arange(-hist_s, 1e-6, dt)
    ego_history[:, 0] = speed * times
    ego_history[:, 7] = speed
    agent_history = np.zeros((max_agents, h, 8), dtype=np.float32)
    agent_mask = np.zeros((max_agents,), dtype=bool)
    num_agents = rng.integers(3, min(max_agents, 12))
    for i in range(num_agents):
        agent_mask[i] = True
        x0 = rng.uniform(5, 55)
        y0 = rng.choice([-3.5, 0.0, 3.5]) + rng.normal(0, 0.3)
        vx = rng.uniform(-1.0, 6.0)
        vy = rng.normal(0, 0.2)
        for j, tt in enumerate(times):
            agent_history[i, j] = [x0 + vx * tt, y0 + vy * tt, 0.0, vx, vy, 4.5, 2.0, 0.0]
    action_gen = ActionGenerator(ActionGeneratorConfig(max_actions=max_actions, horizon_s=fut_s, dt=dt))
    actions, action_meta, action_mask = action_gen.generate(ego_history)
    agent_future = np.zeros((max_agents, t, 8), dtype=np.float32)
    ftimes = np.arange(1, t + 1) * dt
    for i in range(num_agents):
        cur = agent_history[i, -1]
        for j, tt in enumerate(ftimes):
            agent_future[i, j] = [cur[0] + cur[3] * tt, cur[1] + cur[4] * tt, cur[2], cur[3], cur[4], cur[5], cur[6], cur[7]]
    # Logged ego future: one of the generated actions plus small noise.
    valid = np.where(action_mask)[0]
    expert_idx = int(rng.choice(valid[: min(len(valid), 8)]))
    logged_ego_future = actions[expert_idx].copy()
    logged_ego_future[:, :2] += rng.normal(0, 0.1, size=logged_ego_future[:, :2].shape)
    eb = EvidenceBuilder(EvidenceBuilderConfig(max_units=max_evidence))
    evidence_features, evidence_type, evidence_cost, evidence_mask = eb.build(agent_history, agent_mask, actions, action_mask)
    geometry_query = compute_geometry_query(evidence_features, evidence_type, actions, evidence_mask, action_mask, dt)
    teacher_geometry_query = compute_geometry_query(evidence_features, evidence_type, actions, evidence_mask, action_mask, dt, future_agents=agent_future)
    local_cost = local_teacher_contribution(evidence_features, evidence_type, teacher_geometry_query, evidence_mask, action_mask)
    teacher = TeacherEvaluator()
    teacher_cost = teacher.evaluate(actions, action_mask, logged_ego_future, agent_future, agent_mask,
                                    evidence_features, evidence_type, evidence_mask, teacher_geometry_query)
    oracle = oracle_action(teacher_cost, action_mask)
    rival = rival_labels(teacher_cost, action_mask)
    signed = signed_evidence_labels(local_cost, action_mask)
    active = signed_evidence_active_mask(local_cost, teacher_geometry_query, action_mask, evidence_mask)
    map_polylines = np.zeros((256, 20, 4), dtype=np.float32)
    map_masks = np.zeros((256, 20), dtype=bool)
    return {
        "ego_history": ego_history,
        "agent_history": agent_history,
        "agent_mask": agent_mask,
        "map_polylines": map_polylines,
        "map_masks": map_masks,
        "actions": actions,
        "action_meta": action_meta,
        "action_mask": action_mask,
        "evidence_features": evidence_features,
        "evidence_type": evidence_type,
        "evidence_cost": evidence_cost,
        "evidence_mask": evidence_mask,
        "geometry_query": geometry_query,
        "teacher_cost": teacher_cost,
        "oracle_action_index": np.asarray(oracle, dtype=np.int64),
        "rival_label": rival,
        "signed_evidence_label": signed,
        "signed_evidence_mask": active,
        "logged_ego_future": logged_ego_future,
        "metadata_json": np.asarray(json.dumps({"toy_seed": seed, "oracle": int(oracle)}), dtype="<U4096"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Create a synthetic cache for smoke-testing the code without nuPlan data.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--max-agents", type=int, default=64)
    p.add_argument("--max-actions", type=int, default=32)
    p.add_argument("--max-evidence-units", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = ensure_dir(args.output_dir)
    rows = []
    for i in tqdm(range(args.num_samples), desc="toy"):
        sample = make_one(args.seed + i, args.max_agents, args.max_actions, args.max_evidence_units, 0.5, 2.0, 8.0)
        name = f"sample_{i:09d}.npz"
        np.savez(out / name, **sample)
        rows.append({"file": name, "sample_id": i})
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    write_json(out / "summary.json", {"num_samples": args.num_samples, "toy": True})
    print(f"Wrote toy cache to {out}")


if __name__ == "__main__":
    main()
