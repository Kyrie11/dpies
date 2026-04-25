from __future__ import annotations

import torch


def batch_action_metrics(q_scores: torch.Tensor, oracle: torch.Tensor, teacher_cost: torch.Tensor,
                         action_mask: torch.Tensor, topk: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    masked_q = q_scores.masked_fill(~action_mask.bool(), -1e9)
    pred = masked_q.argmax(dim=-1)
    b, k = q_scores.shape
    out: dict[str, float] = {}
    for kk in topk:
        kk_eff = min(kk, k)
        top = torch.topk(masked_q, k=kk_eff, dim=-1).indices
        hit = (top == oracle[:, None]).any(dim=-1).float().mean().item()
        out[f"action_top{kk}_match"] = float(hit)
    out["action_match"] = out.get("action_top1_match", float((pred == oracle).float().mean().item()))
    regrets = []
    norm_regrets = []
    unresolved = []
    q_margins = []
    pair_correct = []
    for i in range(b):
        valid = action_mask[i].bool()
        o = int(oracle[i].item())
        p = int(pred[i].item())
        reg = float(teacher_cost[i, p] - teacher_cost[i, o])
        regrets.append(reg)
        tc = teacher_cost[i][valid]
        denom = float(torch.quantile(tc.float(), 0.75) - torch.quantile(tc.float(), 0.25) + 1e-6) if tc.numel() else 1.0
        norm_regrets.append(reg / denom)
        unresolved.append(float(q_scores[i, p] <= 0.0))
        comp = masked_q[i].clone()
        comp[o] = -1e9
        q_margins.append(float(q_scores[i, o] - comp.max()))
        idx = torch.where(valid)[0]
        if len(idx) > 1:
            qt = q_scores[i, idx]
            tt = -teacher_cost[i, idx]  # larger is better
            qa = qt[:, None] - qt[None, :]
            ta = tt[:, None] - tt[None, :]
            tri = torch.triu(torch.ones_like(qa, dtype=torch.bool), diagonal=1)
            pair_correct.append(float((torch.sign(qa[tri]) == torch.sign(ta[tri])).float().mean().item()))
    out.update({
        "teacher_regret": float(sum(regrets) / max(len(regrets), 1)),
        "normalized_teacher_regret": float(sum(norm_regrets) / max(len(norm_regrets), 1)),
        "unresolved_rate": float(sum(unresolved) / max(len(unresolved), 1)),
        "q_margin": float(sum(q_margins) / max(len(q_margins), 1)),
        "pairwise_ranking_accuracy": float(sum(pair_correct) / max(len(pair_correct), 1)) if pair_correct else 0.0,
    })
    return out


def screening_recall_at_m(pair_mask: torch.Tensor, rival_label: torch.Tensor, action_mask: torch.Tensor) -> float:
    valid = rival_label.bool() & action_mask[:, :, None] & action_mask[:, None, :]
    denom = valid.sum().item()
    if denom == 0:
        return 1.0
    hit = (pair_mask & valid).sum().item()
    return float(hit / denom)


def screening_precision_at_m(pair_mask: torch.Tensor, rival_label: torch.Tensor, action_mask: torch.Tensor) -> float:
    selected = pair_mask.bool() & action_mask[:, :, None] & action_mask[:, None, :]
    denom = selected.sum().item()
    if denom == 0:
        return 1.0
    hit = (selected & rival_label.bool()).sum().item()
    return float(hit / denom)


def decisive_rival_miss_rate(pair_mask: torch.Tensor, rival_label: torch.Tensor, oracle: torch.Tensor,
                              action_mask: torch.Tensor) -> float:
    misses = []
    for i in range(pair_mask.shape[0]):
        o = int(oracle[i].item())
        valid_rivals = rival_label[i, o].bool() & action_mask[i].bool()
        if valid_rivals.sum() == 0:
            misses.append(0.0)
        else:
            misses.append(float(((~pair_mask[i, o]) & valid_rivals).any().item()))
    return float(sum(misses) / max(len(misses), 1))


def evidence_prediction_metrics(pred_signed: torch.Tensor, target_signed: torch.Tensor, active_mask: torch.Tensor,
                                evidence_mask: torch.Tensor, action_mask: torch.Tensor) -> dict[str, float]:
    b, n, k, _ = pred_signed.shape
    eye = torch.eye(k, dtype=torch.bool, device=pred_signed.device)[None, None]
    valid = evidence_mask[:, :, None, None] & action_mask[:, None, :, None] & action_mask[:, None, None, :] & (~eye) & active_mask.bool()
    if valid.sum() == 0:
        return {"evidence_mae": 0.0, "evidence_rmse": 0.0, "evidence_sign_accuracy": 0.0, "evidence_pos_frac": 0.0, "evidence_neg_frac": 0.0}
    diff = pred_signed[valid] - target_signed[valid]
    targ = target_signed[valid]
    sign_active = targ.abs() > 1e-6
    sign_acc = float((torch.sign(pred_signed[valid][sign_active]) == torch.sign(targ[sign_active])).float().mean().item()) if sign_active.any() else 0.0
    return {
        "evidence_mae": float(diff.abs().mean().item()),
        "evidence_rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "evidence_sign_accuracy": sign_acc,
        "evidence_pos_frac": float((targ > 0).float().mean().item()),
        "evidence_neg_frac": float((targ < 0).float().mean().item()),
    }
