from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def list_cache_files(root: str | Path) -> List[Path]:
    root = Path(root)
    files = sorted(root.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz cache files found under {root}")
    return files


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
