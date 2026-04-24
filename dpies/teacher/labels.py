from __future__ import annotations

import numpy as np

from dpies.evidence.geometry_query import query_active_score


def oracle_action(costs: np.ndarray, action_mask: np.ndarray) -> int:
    c = costs.copy()
    c[~action_mask] = np.inf
    return int(np.argmin(c))


def normalize_costs(costs: np.ndarray, action_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    valid = costs[action_mask]
    out = np.zeros_like(costs, dtype=np.float32)
    if len(valid) == 0:
        return out
    q75, q25 = np.percentile(valid, [75, 25])
    iqr = float(q75 - q25)
    med = float(np.median(valid))
    out[action_mask] = (costs[action_mask] - med) / (iqr + eps)
    return out


def rival_labels(costs: np.ndarray, action_mask: np.ndarray, top_rank_l: int = 8, margin_delta: float = 0.5) -> np.ndarray:
    k = len(costs)
    labels = np.zeros((k, k), dtype=bool)
    valid_idx = np.where(action_mask)[0]
    if len(valid_idx) < 2:
        return labels
    norm = normalize_costs(costs, action_mask)
    order = valid_idx[np.argsort(costs[valid_idx])]
    rank = np.full((k,), 10_000, dtype=np.int64)
    for r, idx in enumerate(order):
        rank[idx] = r
    for i in valid_idx:
        for j in valid_idx:
            if i == j:
                continue
            competitive = min(rank[i], rank[j]) < top_rank_l
            near = abs(float(norm[i] - norm[j])) <= margin_delta
            labels[i, j] = bool(competitive and near)
    return labels


def signed_evidence_labels(local_cost: np.ndarray, action_mask: np.ndarray, s_max: float = 10.0) -> np.ndarray:
    # s_i(a,b)=g_i(b)-g_i(a), positive favors a over b.
    s = local_cost[:, None, :] - local_cost[:, :, None]
    # Current shape [N,K,K] has g(b)-g(a) because axis 1=a, axis2=b.
    valid = action_mask[None, :, None] & action_mask[None, None, :]
    s = np.clip(s, -s_max, s_max).astype(np.float32)
    s *= valid.astype(np.float32)
    return s


def signed_evidence_active_mask(local_cost: np.ndarray, geometry_query: np.ndarray, action_mask: np.ndarray,
                                evidence_mask: np.ndarray, cost_threshold: float = 0.05,
                                query_threshold: float = 0.1) -> np.ndarray:
    n, k = local_cost.shape
    active = np.zeros((n, k, k), dtype=bool)
    q_score = query_active_score(geometry_query)
    for i in range(n):
        if not evidence_mask[i]:
            continue
        for a in range(k):
            if not action_mask[a]:
                continue
            for b in range(k):
                if a == b or not action_mask[b]:
                    continue
                cost_active = max(float(local_cost[i, a]), float(local_cost[i, b])) > cost_threshold
                query_active = max(float(q_score[i, a]), float(q_score[i, b])) > query_threshold
                active[i, a, b] = cost_active or query_active
    return active
