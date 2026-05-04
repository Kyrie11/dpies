from __future__ import annotations

"""nuPlan closed-loop planner adapter for DPIES.

This module implements the official devkit planner interface while keeping all
DPIES preprocessing logic schema-consistent with offline cache generation:
PlannerInitialization/PlannerInput -> ego/agent/map/action/evidence/query tensors
-> DPIESNetwork -> capped evidence selection -> max-min action -> InterpolatedTrajectory.
"""
import json
from pathlib import Path

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:  # Keep the file importable without nuPlan for syntax tests.
    from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner  # type: ignore
except Exception:  # pragma: no cover
    class AbstractPlanner:  # type: ignore
        pass

from dpies.actions.action_generator import ActionGenerator, ActionGeneratorConfig
from dpies.actions.trajectory_quality import batch_action_quality
from dpies.common.geometry import ego_to_global_points
from dpies.data.devkit_utils import traffic_from_planner_input
from dpies.data.map_provider import NuPlanMapProvider
from dpies.data.scenario_api import build_history_tensors_from_simulation, ego_state_to_global_array
from dpies.evidence.evidence_builder import EvidenceBuilder, EvidenceBuilderConfig
from dpies.evidence.geometry_query import compute_geometry_query
from dpies.model.network import DPIESConfig, DPIESNetwork
from dpies.selection.capped_greedy import capped_greedy_select_batch, compute_q_scores, make_directed_pair_mask


@dataclass
class DPIESClosedLoopConfig:
    history_seconds: float = 2.0
    future_seconds: float = 8.0
    dt: float = 0.5
    max_agents: int = 64
    agent_radius_m: float = 60.0
    max_actions: int = 32
    max_evidence_units: int = 64
    max_map_polylines: int = 128
    max_map_points: int = 20
    map_radius_m: float = 50.0
    top_m: int = 4
    budget: float = 24.0
    eta_e: float = 0.05
    gamma0: float = 1.0
    device: str = "cuda"
    fallback_on_error: bool = True
    exact_online_map_query: bool = False
    enable_timing_debug: bool = True
    progress_rerank_weight: float = 0.03
    comfort_rerank_penalty: float = 5.0
    selection_policy: str = "model"


class DPIESPlannerCore:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", top_m: int = 4,
                 budget: float = 32, eta_e: float = 0.05, gamma0: float = 1.0):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg = ckpt.get("config", {}).get("model", {}) if isinstance(ckpt, dict) else {}
        self.model = DPIESNetwork(DPIESConfig(**cfg)).to(self.device)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        self.model.load_state_dict(state)
        self.model.eval()
        self.top_m = int(top_m)
        self.budget = float(budget)
        self.eta_e = float(eta_e)
        self.gamma0 = float(gamma0)

    @torch.no_grad()
    def choose_action(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        dev_batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = self.model(dev_batch)
        pair_mask = make_directed_pair_mask(out["rival_scores"], dev_batch["action_mask"], self.top_m)
        selected = capped_greedy_select_batch(out["signed_evidence"], out["rival_scores"], pair_mask,
                                             dev_batch["evidence_mask"], dev_batch["evidence_cost"],
                                             self.budget, self.eta_e, self.gamma0)
        q, _ = compute_q_scores(out["signed_evidence"], selected, pair_mask, dev_batch["action_mask"])
        pred = q.masked_fill(~dev_batch["action_mask"].bool(), -1e9).argmax(dim=-1)
        return pred.detach().cpu(), q.detach().cpu(), selected


class DPIESNuPlanPlanner(AbstractPlanner):  # type: ignore[misc]
    """Official nuPlan closed-loop planner for DPIES.

    Hydra target:
        dpies.evaluation.closed_loop_planner.DPIESNuPlanPlanner
    """

    requires_scenario = False

    def __init__(self, checkpoint: str, **kwargs: Any):
        cfg_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in DPIESClosedLoopConfig.__dataclass_fields__}
        self.cfg = DPIESClosedLoopConfig(**cfg_kwargs)
        self.core = DPIESPlannerCore(
            checkpoint=checkpoint,
            device=self.cfg.device,
            top_m=self.cfg.top_m,
            budget=self.cfg.budget,
            eta_e=self.cfg.eta_e,
            gamma0=self.cfg.gamma0,
        )
        self.map_provider = NuPlanMapProvider(None, self.cfg.max_map_polylines, self.cfg.max_map_points)
        self.action_gen = ActionGenerator(ActionGeneratorConfig(
            max_actions=self.cfg.max_actions,
            horizon_s=self.cfg.future_seconds,
            dt=self.cfg.dt,
        ))
        self.evidence_builder = EvidenceBuilder(EvidenceBuilderConfig(
            max_units=self.cfg.max_evidence_units,
            radius_m=self.cfg.agent_radius_m,
        ))
        self.initialization = None
        self.last_debug: dict[str, Any] = {}

    def name(self) -> str:
        return "dpies_planner"

    def observation_type(self):
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks  # type: ignore
        return DetectionsTracks

    def initialize(self, initialization: Any) -> None:
        self.initialization = initialization

    def _planner_input_to_batch(self, current_input: Any) -> dict[str, torch.Tensor]:
        import time
        t_all = time.perf_counter()
        timings = {}
        t = time.perf_counter()

        history = current_input.history
        ego_states = list(history.ego_states)
        observations = list(history.observations)
        current_ego, _ = history.current_state
        current_global = ego_state_to_global_array(current_ego)
        ego_xy, ego_yaw = current_global[:2], float(current_global[2])
        hist_steps = int(round(self.cfg.history_seconds / self.cfg.dt)) + 1
        ego_history, agent_history, agent_history_mask, agent_mask, agent_track_id, agent_type = build_history_tensors_from_simulation(
            ego_states=ego_states,
            observations=observations,
            max_agents=self.cfg.max_agents,
            history_steps=hist_steps,
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            agent_radius_m=self.cfg.agent_radius_m,
        )
        timings["history_s"] = time.perf_counter() - t
        t = time.perf_counter()

        if self.initialization is None:
            raise RuntimeError("DPIESNuPlanPlanner.initialize() must be called before compute_planner_trajectory")
        route_ids = [str(x) for x in getattr(self.initialization, "route_roadblock_ids", [])]
        map_api = getattr(self.initialization, "map_api", None)
        if map_api is None:
            raise RuntimeError("PlannerInitialization.map_api is required for DPIES closed-loop inference")
        traffic_light_records = traffic_from_planner_input(current_input)
        map_obj = self.map_provider.extract_from_api(
            map_api,
            ego_xy,
            ego_yaw,
            self.cfg.map_radius_m,
            route_roadblock_ids=route_ids,
            traffic_lights=traffic_light_records,
            future_traffic_lights=None,
        )
        timings["map_extract_s"] = time.perf_counter() - t
        t = time.perf_counter()

        actions, action_meta, action_mask = self.action_gen.generate(
            ego_history,
            agent_history=agent_history,
            agent_mask=agent_mask,
            map_context=map_obj,
            rule_units=map_obj.rule_units,
            traffic_lights=traffic_light_records,
        )
        timings["action_gen_s"] = time.perf_counter() - t
        t = time.perf_counter()

        evidence_features, evidence_type, evidence_cost, evidence_mask = self.evidence_builder.build(
            agent_history,
            agent_mask,
            actions,
            action_mask,
            rule_units=map_obj.rule_units,
            dt=self.cfg.dt,
            agent_history_mask=agent_history_mask,
        )
        timings["evidence_build_s"] = time.perf_counter() - t
        t = time.perf_counter()

        evidence_metadata = list(self.evidence_builder.last_metadata)
        geometry_query = compute_geometry_query(
            evidence_features,
            evidence_type,
            actions,
            evidence_mask,
            action_mask,
            self.cfg.dt,
            evidence_metadata=evidence_metadata,
            route_info=map_obj.route_info,
            exact_map_rules=self.cfg.exact_online_map_query,
        )
        timings["geometry_query_s"] = time.perf_counter() - t
        timings["planner_input_total_s"] = time.perf_counter() - t_all
        self.last_debug = {
            "map_success": bool(map_obj.success),
            "map_error": str(map_obj.error),
            "valid_action_count": int(action_mask.sum()),
            "evidence_count": int(evidence_mask.sum()),
            "route_roadblocks": len(route_ids),
            "traffic_lights": len(traffic_light_records),
            "route_info": map_obj.route_info,
            "timings": timings,
        }

        def ft(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(x).float().unsqueeze(0)

        def bt(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(x.astype(bool)).bool().unsqueeze(0)

        def lt(x: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(x).long().unsqueeze(0)

        return {
            "ego_history": ft(ego_history.astype(np.float32)),
            "agent_history": ft(agent_history.astype(np.float32)),
            "agent_history_mask": bt(agent_history_mask),
            "agent_mask": bt(agent_mask),
            "map_polylines": ft(map_obj.polylines.astype(np.float32)),
            "map_masks": bt(map_obj.masks),
            "actions": ft(actions.astype(np.float32)),
            "action_meta": ft(action_meta.astype(np.float32)),
            "action_mask": bt(action_mask),
            "evidence_features": ft(evidence_features.astype(np.float32)),
            "evidence_type": lt(evidence_type.astype(np.int64)),
            "evidence_cost": ft(evidence_cost.astype(np.float32)),
            "evidence_mask": bt(evidence_mask),
            "geometry_query": ft(geometry_query.astype(np.float32)),
        }

    def _trajectory_from_action(self, current_ego: Any, action: np.ndarray) -> Any:
        from nuplan.common.actor_state.ego_state import EgoState  # type: ignore
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint  # type: ignore
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory  # type: ignore

        current_global = ego_state_to_global_array(current_ego)
        ego_xy, ego_yaw = current_global[:2], float(current_global[2])
        vehicle = current_ego.car_footprint.vehicle_parameters
        base_time_us = int(current_ego.time_us)
        traj_states = [current_ego]
        xy_global = ego_to_global_points(action[:, :2], ego_xy, ego_yaw)
        for i, st in enumerate(action):
            yaw = float(ego_yaw + st[2])
            speed = float(max(st[3], 0.0))
            accel = float(st[4]) if action.shape[-1] > 4 else 0.0
            vx = speed * np.cos(yaw)
            vy = speed * np.sin(yaw)
            ax = accel * np.cos(yaw)
            ay = accel * np.sin(yaw)
            time_us = base_time_us + int(round((i + 1) * self.cfg.dt * 1e6))
            traj_states.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(xy_global[i, 0]), float(xy_global[i, 1]), yaw),
                rear_axle_velocity_2d=StateVector2D(float(vx), float(vy)),
                rear_axle_acceleration_2d=StateVector2D(float(ax), float(ay)),
                tire_steering_angle=float(getattr(current_ego, "tire_steering_angle", 0.0)),
                time_point=TimePoint(time_us),
                vehicle_parameters=vehicle,
                is_in_auto_mode=True,
            ))
        return InterpolatedTrajectory(traj_states)

    def _fallback_trajectory(self, current_ego: Any) -> Any:
        from dpies.actions.rollout import stop_rollout
        current = ego_state_to_global_array(current_ego)
        speed = float(current[8])
        action = stop_rollout(speed, max(3.0, speed * 1.5), self.cfg.future_seconds, self.cfg.dt)
        return self._trajectory_from_action(current_ego, action)

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        current_ego, _ = current_input.history.current_state
        try:
            import time
            t = time.perf_counter()
            batch = self._planner_input_to_batch(current_input)
            input_s = time.perf_counter() - t
            t = time.perf_counter()
            pred, q, selected = self.core.choose_action(batch)
            model_select_s = time.perf_counter() - t
            idx = int(pred[0].item())
            actions = batch["actions"][0].cpu().numpy()
            action_mask = batch["action_mask"][0].cpu().numpy().astype(bool)

            qual = batch_action_quality(actions, action_mask, self.cfg.dt)

            rerank_reason = "model_argmax"
            raw_idx = idx
            min_progress = None
            selected_comfort = None
            selected_rerank_score = None
            # Optional deployment rerank: keep DPIES q as the base score, but avoid
            # low-progress or uncomfortable trajectories when q is close.
            if self.cfg.progress_rerank_weight != 0.0 or self.cfg.comfort_rerank_penalty != 0.0:
                q0 = q[0].clone()
                valid = batch["action_mask"][0].bool()

                progress = torch.as_tensor(qual["progress"], dtype=q0.dtype)
                comfort = torch.as_tensor(qual["comfort_violation"], dtype=q0.dtype)

                current_speed = float(batch["ego_history"][0, -1, 8].item())
                min_progress = max(8.0, 0.65 * current_speed * self.cfg.future_seconds)


                score = q0.clone()
                score = score.masked_fill(~valid, -1e9)

                # 1. 优先筛掉明显不前进的动作
                progress_ok = progress >= min_progress

                # 2. 优先使用舒适动作
                comfort_ok = comfort < 0.5

                preferred = valid & progress_ok & comfort_ok
                fallback1 = valid & progress_ok
                fallback2 = valid & comfort_ok

                if preferred.any():
                    cand = preferred
                    rerank_reason = "preferred_progress_and_comfort"
                elif fallback1.any():
                    cand = fallback1
                    rerank_reason = "fallback_progress_only"
                elif fallback2.any():
                    cand = fallback2
                    rerank_reason = "fallback_comfort_only"
                else:
                    cand = valid
                    rerank_reason = "fallback_any_valid"
                # 3. 在候选集合内再用 q + route/progress bias 排序
                score = (
                        q0
                        + float(self.cfg.progress_rerank_weight) * progress
                        - float(self.cfg.comfort_rerank_penalty) * comfort
                )
                score = score.masked_fill(~cand, -1e9)
                idx = int(score.argmax().item())
                selected_comfort = float(comfort[idx].item())
                selected_rerank_score = float(score[idx].item())
            if self.cfg.selection_policy == "progress":
                score = torch.as_tensor(qual["progress"], dtype=q[0].dtype)
                score = score.masked_fill(~batch["action_mask"][0].bool(), -1e9)
                idx = int(score.argmax().item())
                rerank_reason = "progress_baseline"
            if idx < 0 or idx >= actions.shape[0] or not action_mask[idx]:
                raise RuntimeError(f"invalid DPIES selected action index {idx}")
            mode = int(batch["action_meta"][0, idx, 0].item())
            progress = float(batch["actions"][0, idx, -1, 0].item())
            final_speed = float(batch["actions"][0, idx, -1, 3].item())

            debug_path = Path("runs/closed_loop_action_debug.jsonl")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            q_np = q[0].numpy()
            order = np.argsort(np.where(action_mask, q_np, -1e9))[::-1][:8]
            meta_np = batch["action_meta"][0].cpu().numpy()
            with debug_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "selected_action": idx,
                    "selected_mode": mode,
                    "selected_progress": progress,
                    "selected_final_speed": final_speed,
                    "q_selected": float(q[0, idx].item()),
                    "valid_action_count": int(batch["action_mask"][0].sum().item()),
                    "evidence_count": int(batch["evidence_mask"][0].sum().item()),
                    "debug": self.last_debug,
                    "raw_model_action": raw_idx,
                    "rerank_reason": rerank_reason,
                    "min_progress": None if min_progress is None else float(min_progress),
                    "selected_comfort_violation": selected_comfort,
                    "selected_rerank_score": selected_rerank_score,
                    "model_top8_actions": [int(x) for x in order],
                    "model_top8_q": [float(q_np[x]) for x in order],
                    "model_top8_mode": [int(meta_np[x, 0]) for x in order],
                    "model_top8_progress": [float(actions[x, -1, 0]) for x in order],
                    "model_top8_final_speed": [float(actions[x, -1, 3]) for x in order],
                    "model_top8_terminal_lateral": [float(meta_np[x, 2]) for x in order],
                }, ensure_ascii=False) + "\n")
                fh.flush()

            self.last_debug.update({"selected_action": idx, "q_selected": float(q[0, idx].item()), "selected_evidence": selected[0], "input_s": input_s, "model_select_s": model_select_s})


            return self._trajectory_from_action(current_ego, actions[idx])
        except Exception as exc:
            self.last_debug.update({"fallback_reason": str(exc)})
            debug_path = Path("runs/closed_loop_action_debug.jsonl")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "fallback": True,
                    "fallback_reason": str(exc),
                    "debug": self.last_debug,
                }, ensure_ascii=False)+"\n")
            if not self.cfg.fallback_on_error:
                raise
            return self._fallback_trajectory(current_ego)


def build_nuplan_planner_class():
    """Return the devkit-compatible DPIES planner class.

    Kept for backward compatibility with earlier skeleton versions.
    """
    return DPIESNuPlanPlanner
