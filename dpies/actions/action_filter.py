from __future__ import annotations

import numpy as np


def diversity_filter(actions: list[dict], max_actions: int, return_trace: bool = False):
    """Mode-aware endpoint/speed diversity filter."""
    if len(actions) <= max_actions:
        if return_trace:
            return actions, {"pre_count": len(actions), "post_count": len(actions), "dropped": []}
        return actions

    original = list(actions)
    by_mode: dict[int, list[dict]] = {}
    for idx, a in enumerate(actions):
        a = dict(a)
        a["_pre_filter_index"] = idx
        by_mode.setdefault(int(a["mode"]), []).append(a)

    selected: list[dict] = []

    while len(selected) < max_actions and any(by_mode.values()):
        for mode in sorted(list(by_mode)):
            if len(selected) >= max_actions:
                break
            bucket = by_mode[mode]
            if not bucket:
                continue

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

    selected = selected[:max_actions]
    selected_ids = {int(a.get("_pre_filter_index", -1)) for a in selected}
    dropped = []
    for idx, a in enumerate(original):
        if idx not in selected_ids:
            traj = a["trajectory"]
            dropped.append({
                "pre_filter_index": idx,
                "mode": int(a["mode"]),
                "final_x": float(traj[-1, 0]),
                "final_y": float(traj[-1, 1]),
                "final_speed": float(traj[-1, 3]) if traj.shape[-1] > 3 else 0.0,
            })

    for a in selected:
        a.pop("_pre_filter_index", None)

    trace = {
        "pre_count": len(original),
        "post_count": len(selected),
        "dropped": dropped,
    }

    if return_trace:
        return selected, trace
    return selected
