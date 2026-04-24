from __future__ import annotations

import numpy as np


def min_ade_fde(actions: np.ndarray, action_mask: np.ndarray, logged_future: np.ndarray) -> tuple[float, float]:
    valid = np.where(action_mask)[0]
    if len(valid) == 0:
        return float("inf"), float("inf")
    tgt = logged_future[: actions.shape[1], :2]
    ades = []
    fdes = []
    for k in valid:
        diff = actions[k, : len(tgt), :2] - tgt
        dist = np.linalg.norm(diff, axis=-1)
        ades.append(float(dist.mean()))
        fdes.append(float(dist[-1]))
    return min(ades), min(fdes)
