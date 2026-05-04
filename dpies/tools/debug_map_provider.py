import argparse
import json
from pathlib import Path

import numpy as np

from dpies.data.nuplan_db import NuPlanSQLite
from dpies.data.map_provider import NuPlanMapProvider


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--map-root", required=True)
    p.add_argument("--map-version", default="nuplan-maps-v1.0")
    p.add_argument("--sample-interval-s", type=float, default=10.0)
    p.add_argument("--map-radius-m", type=float, default=80.0)
    p.add_argument("--max-rows", type=int, default=5)
    args = p.parse_args()

    provider = NuPlanMapProvider(
        args.map_root,
        max_polylines=256,
        max_points=20,
        map_version=args.map_version,
    )

    with NuPlanSQLite(args.db) as db:
        meta = db.get_log_metadata()
        print("db:", args.db)
        print("metadata:", json.dumps(meta, indent=2, ensure_ascii=False))

        rows = list(db.iter_lidar_pc_rows(args.sample_interval_s, args.max_rows))
        print("rows:", len(rows))

        for i, row in enumerate(rows):
            current = db.ego_state_at_lidar_row(row)
            if current is None:
                print(i, "no ego state")
                continue

            map_name = str(meta.get("map_name", "unknown"))
            obj = provider.extract(
                map_name,
                current[:2],
                float(current[2]),
                args.map_radius_m,
                route_roadblock_ids=db.route_roadblock_ids_for_lidar_token(row["token"]),
                traffic_lights=db.traffic_light_statuses_for_lidar_token(row["token"]),
            )

            print("\nrow", i)
            print("timestamp_us:", int(row["timestamp_us"]))
            print("ego_xy:", current[:2].tolist())
            print("map_name:", map_name)
            print("success:", obj.success)
            print("error:", obj.error)
            print("num_polylines:", int(obj.masks.any(axis=1).sum()))
            print("num_rule_units:", len(obj.rule_units))
            print("route_info:", json.dumps({
                "raw_map_object_count": obj.route_info.get("raw_map_object_count"),
                "geom_map_object_count": obj.route_info.get("geom_map_object_count"),
                "raw_layer_counts": obj.route_info.get("raw_layer_counts"),
                "geom_layer_counts": obj.route_info.get("geom_layer_counts"),
            }, indent=2, ensure_ascii=False))
            print("layers:", sorted(set(str(x.get("layer", "")) for x in obj.rule_units))[:30])


if __name__ == "__main__":
    main()