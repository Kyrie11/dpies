from __future__ import annotations

import torch
import torch.nn.functional as F

def neg_large_like(x: torch.Tensor) -> float:
    if x.dtype == torch.float16:
        return -1.0e4
    if x.dtype == torch.bfloat16:
        return -1.0e6
    return -1.0e9

def screening_loss(rival_logits: torch.Tensor, rival_label: torch.Tensor, action_mask: torch.Tensor,
                   positive_weight_alpha: float = 5.0) -> torch.Tensor:
    b, k, _ = rival_logits.shape
    eye = torch.eye(k, dtype=torch.bool, device=rival_logits.device)[None]
    valid = action_mask[:, :, None] & action_mask[:, None, :] & (~eye)
    if valid.sum() == 0:
        return rival_logits.sum() * 0.0
    labels = rival_label.float()
    weights = torch.ones_like(labels)
    weights = torch.where(labels > 0.5, torch.full_like(weights, positive_weight_alpha), weights)
    loss = F.binary_cross_entropy_with_logits(rival_logits, labels, weight=weights, reduction="none")
    return loss[valid].mean()


def evidence_loss(pred_signed: torch.Tensor, target_signed: torch.Tensor, active_mask: torch.Tensor,
                  evidence_mask: torch.Tensor, action_mask: torch.Tensor, pair_mask: torch.Tensor | None = None,
                  huber_delta: float = 1.0, loss_on_all_pairs: bool = False) -> torch.Tensor:
    b, n, k, _ = pred_signed.shape
    eye = torch.eye(k, dtype=torch.bool, device=pred_signed.device)[None, None]
    valid = evidence_mask[:, :, None, None] & action_mask[:, None, :, None] & action_mask[:, None, None, :] & (~eye)
    valid = valid & active_mask.bool()
    if pair_mask is not None and not loss_on_all_pairs:
        valid = valid & pair_mask[:, None, :, :].bool()
    if valid.sum() == 0:
        return pred_signed.sum() * 0.0
    loss = F.huber_loss(pred_signed, target_signed, delta=huber_delta, reduction="none")
    return loss[valid].mean()


def action_identity_loss(
    q_scores: torch.Tensor,
    oracle: torch.Tensor,
    action_mask: torch.Tensor,
    tau_q: float = 1.0,) -> torch.Tensor:
    q_scores = q_scores.float()
    logits = q_scores / max(tau_q, 1e-6)
    logits = logits.masked_fill(~action_mask.bool(), neg_large_like(logits))
    return F.cross_entropy(logits, oracle.long())


def hard_negative_loss(
    q_scores: torch.Tensor,
    oracle: torch.Tensor,
    action_mask: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    q_scores = q_scores.float()
    b, k = q_scores.shape
    losses = []
    neg_large = neg_large_like(q_scores)

    for i in range(b):
        valid = action_mask[i].bool().clone()
        if valid.sum() <= 1:
            losses.append(q_scores[i].sum() * 0.0)
            continue

        o = int(oracle[i].item())
        competitor_scores = q_scores[i].clone()
        competitor_scores[~valid] = neg_large
        competitor_scores[o] = neg_large

        neg = competitor_scores.max()
        pos = q_scores[i, o]
        losses.append(
            torch.relu(
                torch.tensor(margin, dtype=q_scores.dtype, device=q_scores.device) + neg - pos
            )
        )

    return torch.stack(losses).mean()


def compute_total_loss(outputs: dict, batch: dict, pair_mask: torch.Tensor, q_scores: torch.Tensor,
                       weights: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l_scr = screening_loss(outputs["rival_logits"], batch["rival_label"], batch["action_mask"], weights.get("positive_weight_alpha", 5.0))
    l_evi = evidence_loss(outputs["signed_evidence"], batch["signed_evidence_label"], batch["signed_evidence_mask"],
                          batch["evidence_mask"], batch["action_mask"], pair_mask,
                          weights.get("huber_delta", 1.0), bool(weights.get("loss_on_all_pairs", False)))
    l_act = action_identity_loss(q_scores, batch["oracle_action_index"], batch["action_mask"], weights.get("tau_q", 1.0))
    l_hn = hard_negative_loss(q_scores, batch["oracle_action_index"], batch["action_mask"], weights.get("action_margin", 0.5))
    total = (weights.get("lambda_scr", 1.0) * l_scr + weights.get("lambda_evi", 1.0) * l_evi +
             weights.get("lambda_act", 1.0) * l_act + weights.get("lambda_hn", 0.5) * l_hn)
    logs = {"loss": total.detach(), "loss_scr": l_scr.detach(), "loss_evi": l_evi.detach(),
            "loss_act": l_act.detach(), "loss_hn": l_hn.detach()}
    return total, logs
