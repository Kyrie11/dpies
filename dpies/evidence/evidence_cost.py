from __future__ import annotations

import numpy as np


def unit_costs(type_ids: np.ndarray, cost_type: str = "unit") -> np.ndarray:
    if cost_type == "unit":
        return np.ones_like(type_ids, dtype=np.float32)
    # Simple complexity-aware placeholder: dynamic/conflict units are slightly
    # more expensive than scalar rule units. This is for ablations only.
    costs = np.ones_like(type_ids, dtype=np.float32)
    costs += np.where(type_ids == 0, 0.5, 0.0)
    costs += np.where(type_ids == 1, 0.75, 0.0)
    costs += np.where(type_ids == 2, 0.25, 0.0)
    return costs.astype(np.float32)
