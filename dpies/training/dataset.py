from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from dpies.common.types import EGO_DIM
from dpies.data.cache_reader import list_cache_files, load_npz


FLOAT_KEYS = {
    "ego_history", "ego_global_state", "ego_to_global",
    "agent_history", "agent_future", "map_polylines", "actions", "action_meta",
    "evidence_features", "evidence_cost", "geometry_query", "teacher_cost", "teacher_components",
    "local_cost_sum", "signed_evidence_label", "logged_ego_future",
}
BOOL_KEYS = {
    "agent_mask", "agent_history_mask", "agent_future_mask", "map_masks", "action_mask",
    "evidence_mask", "rival_label", "signed_evidence_mask",
}
LONG_KEYS = {"evidence_type", "oracle_action_index", "agent_track_id", "agent_type"}
STRING_KEYS = {"metadata_json", "evidence_metadata_json", "evidence_units_json", "route_info_json", "traffic_lights_json"}


def _upgrade_ego_history(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.shape[-1] == EGO_DIM:
        return arr.astype(np.float32)
    if arr.shape[-1] == 8:
        speed = np.linalg.norm(arr[..., 3:5], axis=-1, keepdims=True).astype(np.float32)
        return np.concatenate([arr.astype(np.float32), speed], axis=-1)
    raise ValueError(f"Unsupported ego_history dim {arr.shape[-1]}; expected legacy 8 or {EGO_DIM}")


class EvidenceCacheDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = list_cache_files(self.root)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        z = load_npz(self.files[idx])
        out: Dict[str, Any] = {}
        for k, v in z.items():
            if k in STRING_KEYS:
                out[k] = str(v)
            elif k == "ego_history":
                out[k] = torch.from_numpy(_upgrade_ego_history(np.asarray(v))).float()
            elif k in FLOAT_KEYS:
                out[k] = torch.from_numpy(np.asarray(v)).float()
            elif k in BOOL_KEYS:
                out[k] = torch.from_numpy(np.asarray(v)).bool()
            elif k in LONG_KEYS:
                out[k] = torch.as_tensor(v).long()
            else:
                arr = np.asarray(v)
                if arr.dtype.kind in {"U", "S", "O"}:
                    out[k] = str(v)
                elif np.issubdtype(arr.dtype, np.floating):
                    out[k] = torch.from_numpy(arr).float()
                elif np.issubdtype(arr.dtype, np.integer):
                    out[k] = torch.from_numpy(arr).long()
                elif np.issubdtype(arr.dtype, np.bool_):
                    out[k] = torch.from_numpy(arr).bool()
        # Backward compatibility for old caches.
        if "agent_history_mask" not in out and "agent_history" in out and "agent_mask" in out:
            h = out["agent_history"].shape[1]
            out["agent_history_mask"] = out["agent_mask"][:, None].expand(-1, h).clone()
        if "agent_future_mask" not in out and "agent_future" in out and "agent_mask" in out:
            t = out["agent_future"].shape[1]
            out["agent_future_mask"] = out["agent_mask"][:, None].expand(-1, t).clone()
        out["cache_file"] = str(self.files[idx])
        return out
