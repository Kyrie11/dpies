from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = samples[0].keys()
    for k in keys:
        vals = [s[k] for s in samples]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    # Flatten scalar oracle from [B] rather than [B,]. stack handles scalars.
    out["oracle_action_index"] = out["oracle_action_index"].long().view(-1)
    return out
