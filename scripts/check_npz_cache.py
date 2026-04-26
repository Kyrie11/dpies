#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _as_json(value):
    if isinstance(value, np.ndarray):
        value = value.item() if value.shape == () else value.tolist()

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if not isinstance(value, str):
        value = str(value)

    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse JSON field. len={len(value)}, "
            f"error={exc}. The field is likely truncated by fixed-length numpy dtype "
            f"such as <U32768. Regenerate this npz after changing JSON saving to "
            f"object dtype or UTF-8 bytes."
        ) from exc


def _shape(d, key):
    return tuple(d[key].shape) if key in d.files else None


def check_file(path: Path):
    d = np.load(path, allow_pickle=True)

    meta = _as_json(d["metadata_json"]) if "metadata_json" in d.files else {}
    ev_meta = _as_json(d["evidence_metadata_json"]) if "evidence_metadata_json" in d.files else []

    layers = []
    rule_codes = Counter()
    has_drivable = False
    has_route = False
    has_crosswalk = False
    has_stop = False
    has_tl = False
    has_speed = False

    for m in ev_meta:
        if not isinstance(m, dict):
            continue
        layer = str(m.get("layer", ""))
        layers.append(layer)
        code = m.get("rule_code", None)
        if code is not None:
            rule_codes[str(code)] += 1
        has_drivable |= layer == "DRIVABLE_AREA_UNION"
        has_route |= layer == "ROUTE_CORRIDOR"
        has_crosswalk |= "CROSSWALK" in layer
        has_stop |= "STOP" in layer
        has_tl |= "TRAFFIC_LIGHT" in layer
        has_speed |= layer == "SPEED_LIMIT"

    out = {
        "file": str(path),
        "scenario_id": meta.get("scenario_id", ""),
        "map_name": meta.get("map_name", ""),
        "map_success": bool(meta.get("map_success", False)),
        "map_error": str(meta.get("map_error", "")),
        "num_map_rule_units": int(meta.get("num_rule_units", -1)),
        "num_route_roadblocks": int(meta.get("num_route_roadblocks", -1)),
        "num_traffic_lights_current": int(meta.get("num_traffic_lights_current", -1)),
        "num_traffic_lights_future_steps": int(meta.get("num_traffic_lights_future_steps", -1)),
        "has_drivable_area_union": has_drivable,
        "has_route_corridor": has_route,
        "has_crosswalk": has_crosswalk,
        "has_stop_line": has_stop,
        "has_traffic_light": has_tl,
        "has_speed_limit": has_speed,
        "layers": Counter(layers),
        "shapes": {k: _shape(d, k) for k in d.files},
        "action_valid_count": int(np.asarray(d["action_mask"]).sum()) if "action_mask" in d.files else int(np.asarray(d["action_valid_mask"]).sum()) if "action_valid_mask" in d.files else -1,
        "evidence_valid_count": int(np.asarray(d["evidence_mask"]).sum()) if "evidence_mask" in d.files else -1,
        "teacher_cost_finite": bool(np.isfinite(d["teacher_cost"]).all()) if "teacher_cost" in d.files else False,
        "geometry_query_finite": bool(np.isfinite(d["geometry_query"]).all()) if "geometry_query" in d.files else False,
        "signed_evidence_finite": bool(np.isfinite(d["signed_evidence_label"]).all()) if "signed_evidence_label" in d.files else False,
    }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--show-files", type=int, default=3)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    files = sorted(cache_dir.glob("sample_*.npz"))
    if args.limit > 0:
        files = files[:args.limit]

    print(f"cache_dir: {cache_dir}")
    print(f"checked_files: {len(files)}")

    if not files:
        print("No sample_*.npz files found.")
        return

    totals = Counter()
    layer_totals = Counter()
    map_errors = Counter()
    shape_examples = {}
    bad_files = []

    for i, f in enumerate(files):
        r = check_file(f)

        totals["files"] += 1
        totals["map_success"] += int(r["map_success"])
        totals["has_drivable_area_union"] += int(r["has_drivable_area_union"])
        totals["has_route_corridor"] += int(r["has_route_corridor"])
        totals["has_crosswalk"] += int(r["has_crosswalk"])
        totals["has_stop_line"] += int(r["has_stop_line"])
        totals["has_traffic_light"] += int(r["has_traffic_light"])
        totals["has_speed_limit"] += int(r["has_speed_limit"])
        totals["teacher_cost_finite"] += int(r["teacher_cost_finite"])
        totals["geometry_query_finite"] += int(r["geometry_query_finite"])
        totals["signed_evidence_finite"] += int(r["signed_evidence_finite"])

        totals["action_valid_sum"] += r["action_valid_count"]
        totals["evidence_valid_sum"] += r["evidence_valid_count"]

        layer_totals.update(r["layers"])
        if r["map_error"]:
            map_errors[r["map_error"]] += 1

        if not r["map_success"] or not r["has_drivable_area_union"] or not r["teacher_cost_finite"] or not r["geometry_query_finite"]:
            bad_files.append((f, r))

        if i < args.show_files:
            print("\n--- example file ---")
            print("file:", f.name)
            print("map_success:", r["map_success"], "map_error:", r["map_error"])
            print("map_name:", r["map_name"])
            print("action_valid_count:", r["action_valid_count"])
            print("evidence_valid_count:", r["evidence_valid_count"])
            print("has_drivable_area_union:", r["has_drivable_area_union"])
            print("has_route_corridor:", r["has_route_corridor"])
            print("has_crosswalk:", r["has_crosswalk"])
            print("has_stop_line:", r["has_stop_line"])
            print("has_traffic_light:", r["has_traffic_light"])
            print("has_speed_limit:", r["has_speed_limit"])
            print("top layers:", r["layers"].most_common(20))
            print("important shapes:")
            for k in [
                "ego_history",
                "agent_history",
                "map_polylines",
                "actions",
                "evidence_features",
                "geometry_query",
                "teacher_cost",
                "rival_label",
                "signed_evidence_label",
                "signed_evidence_mask",
            ]:
                print(f"  {k}: {r['shapes'].get(k)}")

    n = totals["files"]
    print("\n=== aggregate ===")
    for k in [
        "map_success",
        "has_drivable_area_union",
        "has_route_corridor",
        "has_crosswalk",
        "has_stop_line",
        "has_traffic_light",
        "has_speed_limit",
        "teacher_cost_finite",
        "geometry_query_finite",
        "signed_evidence_finite",
    ]:
        print(f"{k}: {totals[k]} / {n} = {totals[k] / max(n, 1):.3f}")

    print("avg action_valid_count:", totals["action_valid_sum"] / max(n, 1))
    print("avg evidence_valid_count:", totals["evidence_valid_sum"] / max(n, 1))

    print("\n=== top layers ===")
    for layer, c in layer_totals.most_common(40):
        print(f"{c:8d} {repr(layer)}")

    print("\n=== top map errors ===")
    if not map_errors:
        print("(none)")
    else:
        for err, c in map_errors.most_common(20):
            print(f"{c:8d} {repr(err)}")

    print("\n=== suspicious files ===")
    if not bad_files:
        print("(none among checked files)")
    else:
        for f, r in bad_files[:20]:
            print(
                f.name,
                "map_success=", r["map_success"],
                "drivable=", r["has_drivable_area_union"],
                "finite_q=", r["geometry_query_finite"],
                "error=", repr(r["map_error"]),
            )


if __name__ == "__main__":
    main()