from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from dpies.data.cache_reader import list_cache_files, load_npz


FLOAT_KEYS = {
    "ego_history", "agent_history", "map_polylines", "actions", "action_meta",
    "evidence_features", "evidence_cost", "geometry_query", "teacher_cost",
    "signed_evidence_label", "logged_ego_future",
}
BOOL_KEYS = {"agent_mask", "map_masks", "action_mask", "evidence_mask", "rival_label", "signed_evidence_mask"}
LONG_KEYS = {"evidence_type", "oracle_action_index"}


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
            if k == "metadata_json":
                out[k] = str(v)
            elif k in FLOAT_KEYS:
                out[k] = torch.from_numpy(np.asarray(v)).float()
            elif k in BOOL_KEYS:
                out[k] = torch.from_numpy(np.asarray(v)).bool()
            elif k in LONG_KEYS:
                out[k] = torch.as_tensor(v).long()
            else:
                # Keep unknown numeric arrays if any.
                if np.issubdtype(np.asarray(v).dtype, np.floating):
                    out[k] = torch.from_numpy(np.asarray(v)).float()
                elif np.issubdtype(np.asarray(v).dtype, np.integer):
                    out[k] = torch.from_numpy(np.asarray(v)).long()
                elif np.issubdtype(np.asarray(v).dtype, np.bool_):
                    out[k] = torch.from_numpy(np.asarray(v)).bool()
        out["cache_file"] = str(self.files[idx])
        return out
