from __future__ import annotations

import numpy as np


def diversity_filter(actions: list[dict], max_actions: int) -> list[dict]:
    """Mode-aware endpoint/speed diversity filter."""
    if len(actions) <= max_actions:
        return actions
    by_mode: dict[int, list[dict]] = {}
    for a in actions:
        by_mode.setdefault(int(a["mode"]), []).append(a)
    selected: list[dict] = []
    quotas = {m: max(1, max_actions // max(len(by_mode), 1)) for m in by_mode}
    while len(selected) < max_actions and any(by_mode.values()):
        for mode in sorted(list(by_mode)):
            if len(selected) >= max_actions:
                break
            bucket = by_mode[mode]
            if not bucket:
                continue
            # Pick the action farthest from already chosen endpoints in this mode.
            if not selected:
                chosen = bucket.pop(0)
            else:
                endpoints = np.asarray([s["trajectory"][-1, :2] for s in selected], dtype=np.float32)
                dists = []
                for cand in bucket:
                    p = cand["trajectory"][-1, :2]
                    dists.append(float(np.min(np.linalg.norm(endpoints - p[None, :], axis=1))))
                idx = int(np.argmax(dists))
                chosen = bucket.pop(idx)
            selected.append(chosen)
    return selected[:max_actions]
