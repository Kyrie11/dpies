from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib import format as npfmt


def files_from_manifest(root: Path) -> list[Path]:
    manifest = root / "manifest.jsonl"
    out: list[Path] = []
    if not manifest.exists():
        return out
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rel = row.get("file", "")
                if rel:
                    out.append(root / rel)
            except Exception:
                continue
    return out


def list_npz_files(root: Path, use_manifest: bool) -> list[Path]:
    files = files_from_manifest(root) if use_manifest else []
    if not files:
        files = list(root.rglob("*.npz"))
    return [p for p in files if p.suffix == ".npz" and ".tmp." not in p.name and not p.name.startswith(".")]


def _read_npy_header(fp):
    version = npfmt.read_magic(fp)
    if version == (1, 0):
        shape, fortran_order, dtype = npfmt.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, fortran_order, dtype = npfmt.read_array_header_2_0(fp)
    elif version == (3, 0):
        shape, fortran_order, dtype = npfmt.read_array_header_2_0(fp)
    else:
        raise ValueError(f"unsupported npy version {version}")
    return shape, fortran_order, np.dtype(dtype)


def check_npz_header(path_str: str, deep_crc: bool = False) -> tuple[str, bool, str]:
    path = Path(path_str)
    try:
        if not path.exists():
            return path_str, False, "missing file"
        if path.stat().st_size <= 0:
            return path_str, False, "empty file"
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
            if not infos:
                return path_str, False, "empty zip/npz"
            for info in infos:
                if info.is_dir():
                    continue
                if not info.filename.endswith(".npy"):
                    return path_str, False, f"non-npy member {info.filename}"
                with zf.open(info, "r") as fp:
                    shape, _, dtype = _read_npy_header(fp)
                    if dtype.hasobject:
                        return path_str, False, f"object dtype member {info.filename}, dtype={dtype}, shape={shape}"
            if deep_crc:
                bad = zf.testzip()
                if bad is not None:
                    return path_str, False, f"crc failed member {bad}"
        return path_str, True, ""
    except Exception as e:
        return path_str, False, repr(e)


def _worker(args):
    path_str, deep_crc = args
    return check_npz_header(path_str, deep_crc)


def delete_from_list(list_path: Path) -> None:
    deleted = 0
    missing = 0
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            path = Path(line.split("\t", 1)[0])
            try:
                if path.exists():
                    path.unlink()
                    deleted += 1
                    print(f"[DELETE] {path}", flush=True)
                else:
                    missing += 1
            except Exception as e:
                print(f"[DELETE_FAIL] {path}\t{e!r}", flush=True)
    print(f"delete_done deleted={deleted} missing={missing}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast npz cache validator. Header-only by default.")
    ap.add_argument("--cache-dir", type=str, help="Cache root containing manifest.jsonl and npz files")
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--progress-every", type=int, default=5000)
    ap.add_argument("--delete", action="store_true", help="Delete bad files during scan")
    ap.add_argument("--bad-list", type=str, default=None, help="Output txt path for bad files")
    ap.add_argument("--delete-from-list", type=str, default=None, help="Delete files from a previous bad_npz_files.txt")
    ap.add_argument("--no-manifest", action="store_true", help="Ignore manifest.jsonl and scan by rglob")
    ap.add_argument("--deep-crc", action="store_true", help="Also read members to verify zip CRC; slower")
    args = ap.parse_args()

    if args.delete_from_list:
        delete_from_list(Path(args.delete_from_list))
        return
    if not args.cache_dir:
        raise SystemExit("--cache-dir is required unless --delete-from-list is used")

    root = Path(args.cache_dir)
    files = list_npz_files(root, use_manifest=not args.no_manifest)
    total = len(files)
    print(f"cache_dir={root}")
    print(f"total_npz={total}")
    print(f"workers={args.workers} delete={args.delete} deep_crc={args.deep_crc}")

    bad_list_path = Path(args.bad_list) if args.bad_list else root / "bad_npz_files.txt"
    bad_list_path.parent.mkdir(parents=True, exist_ok=True)

    bad = 0
    deleted = 0
    start = time.time()
    with bad_list_path.open("w", encoding="utf-8") as out:
        work = ((str(p), bool(args.deep_crc)) for p in files)
        if args.workers <= 1:
            iterator = map(_worker, work)
        else:
            pool = Pool(processes=args.workers)
            iterator = pool.imap_unordered(_worker, work, chunksize=64)
        try:
            for i, (path_str, ok, err) in enumerate(iterator, 1):
                if not ok:
                    bad += 1
                    out.write(f"{path_str}\t{err}\n")
                    out.flush()
                    print(f"[BAD] {path_str}\t{err}", flush=True)
                    if args.delete:
                        try:
                            Path(path_str).unlink(missing_ok=True)
                            deleted += 1
                            print(f"[DELETE] {path_str}", flush=True)
                        except Exception as e:
                            print(f"[DELETE_FAIL] {path_str}\t{e!r}", flush=True)
                if i % max(1, args.progress_every) == 0 or i == total:
                    elapsed = max(time.time() - start, 1e-6)
                    rate = i / elapsed
                    eta = (total - i) / max(rate, 1e-6)
                    print(f"progress {i}/{total} bad={bad} deleted={deleted} rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)
        finally:
            if args.workers > 1:
                pool.close()
                pool.join()

    print(f"done total={total} bad={bad} deleted={deleted} bad_list={bad_list_path}")


if __name__ == "__main__":
    main()
