from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dpies.common.torch_utils import to_device
from dpies.evaluation.offline_metrics import (
    batch_action_metrics,
    decisive_rival_miss_rate,
    evidence_prediction_metrics,
    screening_precision_at_m,
    screening_recall_at_m,
)
from dpies.selection.capped_greedy import capped_greedy_select_batch, compute_q_scores, make_directed_pair_mask


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, selection_cfg: dict) -> Dict[str, float]:
    model.eval()
    sums = defaultdict(float)
    count = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch)
        pair_mask = make_directed_pair_mask(out["rival_scores"], batch["action_mask"], int(selection_cfg.get("top_m", 4)))
        selected = capped_greedy_select_batch(out["signed_evidence"], out["rival_scores"], pair_mask,
                                             batch["evidence_mask"], batch["evidence_cost"],
                                             float(selection_cfg.get("budget", 32)),
                                             float(selection_cfg.get("eta_e", 0.05)),
                                             float(selection_cfg.get("gamma0", 1.0)))
        q, _ = compute_q_scores(out["signed_evidence"], selected, pair_mask, batch["action_mask"])
        metrics = batch_action_metrics(q, batch["oracle_action_index"], batch["teacher_cost"], batch["action_mask"])
        metrics["screen_recall_at_m"] = screening_recall_at_m(pair_mask, batch["rival_label"], batch["action_mask"])
        metrics["screen_precision_at_m"] = screening_precision_at_m(pair_mask, batch["rival_label"], batch["action_mask"])
        metrics["decisive_rival_miss_rate"] = decisive_rival_miss_rate(pair_mask, batch["rival_label"], batch["oracle_action_index"], batch["action_mask"])
        metrics.update(evidence_prediction_metrics(out["signed_evidence"], batch["signed_evidence_label"], batch["signed_evidence_mask"], batch["evidence_mask"], batch["action_mask"]))
        metrics["selected_count"] = float(selected.float().sum(dim=1).mean().item())
        bs = int(batch["actions"].shape[0])
        for key, val in metrics.items():
            sums[key] += float(val) * bs
        count += bs
    return {k: v / max(count, 1) for k, v in sums.items()}
