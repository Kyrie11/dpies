from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _decode_json_array(x: Any):
    try:
        if isinstance(x, np.ndarray):
            if x.shape == ():
                x = x.item()
            else:
                x = x.tobytes()
        if isinstance(x, bytes):
            return json.loads(x.decode('utf-8'))
        if isinstance(x, str):
            return json.loads(x)
    except Exception:
        return None
    return None


def summarize_npz(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"file": str(path), "arrays": {}}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            arr = z[k]
            item = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
            if arr.size and np.issubdtype(arr.dtype, np.number):
                a = arr.astype(np.float64, copy=False)
                finite = np.isfinite(a)
                item.update({
                    "finite_frac": float(finite.mean()),
                    "min": float(np.nanmin(a)) if finite.any() else None,
                    "max": float(np.nanmax(a)) if finite.any() else None,
                    "mean": float(np.nanmean(a)) if finite.any() else None,
                })
            out["arrays"][k] = item
        if "metadata_json" in z.files:
            out["metadata"] = _decode_json_array(z["metadata_json"])
        if "route_info_json" in z.files:
            ri = _decode_json_array(z["route_info_json"])
            if isinstance(ri, dict):
                out["route_info_keys"] = sorted(ri.keys())
                out["route_info_sizes"] = {k: len(v) if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) else None for k, v in ri.items()}
        if "evidence_metadata_json" in z.files:
            em = _decode_json_array(z["evidence_metadata_json"])
            if isinstance(em, list):
                counts = {}
                for m in em:
                    typ = str(m.get("type", "unknown")) if isinstance(m, dict) else "unknown"
                    counts[typ] = counts.get(typ, 0) + 1
                out["evidence_metadata_type_counts"] = counts
    return out


def main() -> None:
    p = argparse.ArgumentParser("Inspect DPIES npz cache schema and basic stats.")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--file", default=None, help="Optional specific .npz file. If omitted, use first file under cache-dir.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    path = Path(args.file) if args.file else next(iter(sorted(Path(args.cache_dir).rglob("*.npz"))))
    info = summarize_npz(path)
    text = json.dumps(info, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
