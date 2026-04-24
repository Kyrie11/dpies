from __future__ import annotations

"""nuPlan closed-loop planner adapter skeleton.

The offline cache/training path is fully implemented. Closed-loop evaluation in
nuPlan requires the exact devkit version used in your environment. This adapter
keeps all DPIES-specific logic in one place and can be connected to the devkit's
AbstractPlanner by mapping nuPlan PlannerInput -> the same tensors produced by
preprocess_nuplan.py.
"""

from pathlib import Path
from typing import Any

import torch

from dpies.model.network import DPIESConfig, DPIESNetwork
from dpies.selection.capped_greedy import capped_greedy_select_batch, compute_q_scores, make_directed_pair_mask


class DPIESPlannerCore:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", top_m: int = 4,
                 budget: float = 32, eta_e: float = 0.05, gamma0: float = 1.0):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg = ckpt.get("config", {}).get("model", {})
        self.model = DPIESNetwork(DPIESConfig(**cfg)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.top_m = top_m
        self.budget = budget
        self.eta_e = eta_e
        self.gamma0 = gamma0

    @torch.no_grad()
    def choose_action(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        out = self.model(batch)
        pair_mask = make_directed_pair_mask(out["rival_scores"], batch["action_mask"], self.top_m)
        selected = capped_greedy_select_batch(out["signed_evidence"], out["rival_scores"], pair_mask,
                                             batch["evidence_mask"], batch["evidence_cost"],
                                             self.budget, self.eta_e, self.gamma0)
        q, _ = compute_q_scores(out["signed_evidence"], selected, pair_mask, batch["action_mask"])
        pred = q.masked_fill(~batch["action_mask"].bool(), -1e9).argmax(dim=-1)
        return pred, q


def build_nuplan_planner_class():
    """Return a devkit-compatible planner class if nuPlan is installed.

    Usage pattern:
        Planner = build_nuplan_planner_class()
        planner = Planner(checkpoint="runs/dpies_main/best.pt", ...)

    You still need to fill planner_input_to_batch for your nuPlan devkit version.
    """
    try:
        from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("nuPlan devkit is not installed in this environment") from exc

    class DPIESNuPlanPlanner(AbstractPlanner):  # type: ignore
        def __init__(self, checkpoint: str, **kwargs: Any):
            self.core = DPIESPlannerCore(checkpoint, **kwargs)

        def name(self) -> str:
            return "dpies_planner"

        def observation_type(self):
            try:
                from nuplan.planning.simulation.observation.observation_type import ObservationType  # type: ignore
                return ObservationType.DETECTIONS_TRACKS
            except Exception:
                return None

        def initialize(self, initialization):
            self.initialization = initialization

        def compute_planner_trajectory(self, current_input):
            raise NotImplementedError(
                "Map nuPlan PlannerInput to the cache-style DPIES batch, call self.core.choose_action, "
                "then convert the selected nominal action trajectory to InterpolatedTrajectory. "
                "The offline model/action-selection code is complete; this method is version-specific."
            )

    return DPIESNuPlanPlanner
