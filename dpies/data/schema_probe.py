from __future__ import annotations

import argparse
from pathlib import Path

from dpies.data.nuplan_db import NuPlanSQLite


def find_dbs(data_root: str, limit: int | None = None):
    paths = sorted(Path(data_root).rglob("*.db"))
    return paths[:limit] if limit else paths


def main() -> None:
    p = argparse.ArgumentParser(description="Print nuPlan SQLite schemas for a few DB files.")
    p.add_argument("--data-root", required=True)
    p.add_argument("--limit", type=int, default=1)
    args = p.parse_args()
    for db_path in find_dbs(args.data_root, args.limit):
        print(f"\n# {db_path}")
        with NuPlanSQLite(db_path) as db:
            for table, cols in db.describe().items():
                print(f"{table}: {', '.join(cols)}")


if __name__ == "__main__":
    main()
