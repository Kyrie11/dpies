from __future__ import annotations

from typing import Any, Dict, List

import torch


OPTIONAL_STRING_KEYS = {
    "metadata_json",
    "evidence_metadata_json",
    "evidence_units_json",
    "route_info_json",
    "traffic_lights_json",
    "action_filter_trace_json",
    "cache_file",
}


def collate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = set(samples[0].keys())

    for k in OPTIONAL_STRING_KEYS:
        if any(k in s for s in samples):
            keys.add(k)

    for k in sorted(keys):
        if k in OPTIONAL_STRING_KEYS:
            vals = [s.get(k, "") for s in samples]
        else:
            vals = [s[k] for s in samples]

        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals

    out["oracle_action_index"] = out["oracle_action_index"].long().view(-1)
    return out