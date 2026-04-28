from __future__ import annotations

import torch

def neg_large_like(x: torch.Tensor) -> float:
    """A large negative value representable by x.dtype."""
    if x.dtype == torch.float16:
        return -1.0e4
    if x.dtype == torch.bfloat16:
        return -1.0e6
    return -1.0e9

def make_directed_pair_mask(rival_scores: torch.Tensor, action_mask: torch.Tensor, top_m: int) -> torch.Tensor:
    """Top-M screened rivals per action. Returns [B,K,K] bool directed pair mask."""
    b, k, _ = rival_scores.shape
    scores = rival_scores.detach().clone()
    valid = action_mask.bool()
    eye = torch.eye(k, dtype=torch.bool, device=scores.device)[None]
    pair_valid = valid[:, :, None] & valid[:, None, :] & (~eye)
    scores = scores.float()
    scores = scores.masked_fill(~pair_valid, neg_large_like(scores))
    m = min(top_m, max(k - 1, 1))
    idx = torch.topk(scores, k=m, dim=-1).indices
    mask = torch.zeros((b, k, k), dtype=torch.bool, device=scores.device)
    mask.scatter_(2, idx, True)
    mask &= pair_valid
    # Fallback: if a valid action has no rival, select its best valid competitor.
    empty = valid & (mask.sum(dim=-1) == 0)
    if empty.any():
        best = torch.argmax(scores, dim=-1)
        mask[empty, best[empty]] = True
    return mask


def greedy_select_single(signed: torch.Tensor, pair_scores: torch.Tensor, pair_mask: torch.Tensor,
                         evidence_mask: torch.Tensor, costs: torch.Tensor, budget: float,
                         eta_e: float = 0.05, gamma0: float = 1.0) -> torch.Tensor:
    """Greedy capped selection for one sample.

    signed: [N,K,K], pair_scores: [K,K], pair_mask: [K,K].
    Returns boolean selected evidence mask [N].
    """
    device = signed.device
    n, k, _ = signed.shape
    selected = torch.zeros((n,), dtype=torch.bool, device=device)
    valid_e = evidence_mask.bool()
    if pair_mask.sum() == 0 or valid_e.sum() == 0 or budget <= 0:
        return selected
    # Candidate pool by max absolute evidence over screened pairs.
    absmax = torch.zeros((n,), device=device)
    absmax[valid_e] = signed.detach().abs().masked_fill(~pair_mask[None], 0.0).amax(dim=(1, 2))[valid_e]
    candidate = valid_e & (absmax > eta_e)
    if candidate.sum() == 0:
        candidate = valid_e
    # Unordered pair list induced by directed pair mask.
    unordered = pair_mask | pair_mask.T
    pair_list = []
    weights = []
    for a in range(k):
        for b in range(a + 1, k):
            if bool(unordered[a, b]):
                pair_list.append((a, b))
                weights.append(float(torch.maximum(pair_scores[a, b], pair_scores[b, a]).detach().clamp_min(0.0).item()))
    if not pair_list:
        return selected
    p = len(pair_list)
    weights_t = torch.tensor(weights, dtype=signed.dtype, device=device).clamp_min(1e-4)
    pair_scores_e = torch.stack([signed[:, a, b] for a, b in pair_list], dim=1).detach()  # [N,P]
    pos_sum = torch.zeros((p,), dtype=signed.dtype, device=device)
    neg_sum = torch.zeros((p,), dtype=signed.dtype, device=device)
    used = 0.0
    for _ in range(int(candidate.sum().item())):
        feasible = candidate & (~selected) & ((used + costs.detach()) <= budget + 1e-6)
        if feasible.sum() == 0:
            break
        s = pair_scores_e[feasible]
        pos_gain = torch.minimum(pos_sum[None] + torch.relu(s), torch.tensor(gamma0, device=device)) - torch.minimum(pos_sum[None], torch.tensor(gamma0, device=device))
        neg_gain = torch.minimum(neg_sum[None] + torch.relu(-s), torch.tensor(gamma0, device=device)) - torch.minimum(neg_sum[None], torch.tensor(gamma0, device=device))
        gain = ((pos_gain + neg_gain) * weights_t[None]).sum(dim=1)
        feasible_idx = torch.where(feasible)[0]
        cost = costs.detach()[feasible_idx].clamp_min(1e-6)
        ratio = gain / cost
        best_local = int(torch.argmax(ratio).item())
        if float(ratio[best_local].item()) <= 0.0 and selected.any():
            break
        best = feasible_idx[best_local]
        selected[best] = True
        used += float(costs.detach()[best].item())
        s_best = pair_scores_e[best]
        pos_sum = torch.minimum(pos_sum + torch.relu(s_best), torch.tensor(gamma0, device=device))
        neg_sum = torch.minimum(neg_sum + torch.relu(-s_best), torch.tensor(gamma0, device=device))
    return selected


def capped_greedy_select_batch(signed: torch.Tensor, pair_scores: torch.Tensor, pair_mask: torch.Tensor,
                               evidence_mask: torch.Tensor, costs: torch.Tensor, budget: float,
                               eta_e: float = 0.05, gamma0: float = 1.0) -> torch.Tensor:
    b = signed.shape[0]
    selected = []
    for i in range(b):
        selected.append(greedy_select_single(signed[i], pair_scores[i], pair_mask[i], evidence_mask[i], costs[i], budget, eta_e, gamma0))
    return torch.stack(selected, dim=0)


def compute_q_scores(signed: torch.Tensor, selected_mask: torch.Tensor, pair_mask: torch.Tensor,
                     action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute D_SB and max-min Q for screened rivals.

    signed [B,N,K,K], selected [B,N], pair_mask [B,K,K].
    """
    signed = signed.float()
    dmat = (signed * selected_mask[:, :, None, None].to(signed.dtype)).sum(dim=1)
    b, k, _ = dmat.shape

    neg_large = torch.tensor(neg_large_like(signed), dtype=signed.dtype, device=signed.device)
    q = torch.full((b, k), neg_large_like(signed), dtype=signed.dtype, device=signed.device)
    for i in range(b):
        for a in range(k):
            if not bool(action_mask[i, a]):
                continue
            rivals = pair_mask[i, a] & action_mask[i]
            if rivals.sum() == 0:
                q[i, a] = neg_large
            else:
                q[i, a] = dmat[i, a][rivals].min()
    return q, dmat
