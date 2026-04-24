from __future__ import annotations

import torch


def batch_action_metrics(q_scores: torch.Tensor, oracle: torch.Tensor, teacher_cost: torch.Tensor,
                         action_mask: torch.Tensor) -> dict[str, float]:
    pred = q_scores.masked_fill(~action_mask.bool(), -1e9).argmax(dim=-1)
    match = (pred == oracle).float().mean().item()
    b = q_scores.shape[0]
    regrets = []
    unresolved = []
    for i in range(b):
        regrets.append(float(teacher_cost[i, pred[i]] - teacher_cost[i, oracle[i]]))
        unresolved.append(float(q_scores[i, pred[i]] <= 0.0))
    return {
        "action_match": match,
        "teacher_regret": float(sum(regrets) / max(len(regrets), 1)),
        "unresolved_rate": float(sum(unresolved) / max(len(unresolved), 1)),
    }


def screening_recall_at_m(pair_mask: torch.Tensor, rival_label: torch.Tensor, action_mask: torch.Tensor) -> float:
    valid = rival_label.bool() & action_mask[:, :, None] & action_mask[:, None, :]
    denom = valid.sum().item()
    if denom == 0:
        return 1.0
    hit = (pair_mask & valid).sum().item()
    return float(hit / denom)
