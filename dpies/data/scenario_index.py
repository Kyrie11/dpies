from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence


def find_db_files(data_root: str | Path, subdirs: Sequence[str] | None = None, max_dbs: int | None = None) -> List[Path]:
    root = Path(data_root)
    paths: list[Path] = []
    if subdirs:
        for sd in subdirs:
            paths.extend(sorted((root / sd).rglob("*.db")))
    else:
        paths.extend(sorted(root.rglob("*.db")))
    paths = sorted(set(paths))
    if max_dbs is not None:
        paths = paths[:max_dbs]
    return paths
