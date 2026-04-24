from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Launch common budget ablations by calling evaluate.py.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-root", default="runs/ablations")
    p.add_argument("--budgets", nargs="*", default=["4", "8", "16", "24", "32", "48", "64"])
    args = p.parse_args()
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "dpies.evaluation.evaluate",
        "--checkpoint", args.checkpoint,
        "--cache-dir", args.cache_dir,
        "--output-dir", str(out / "budget_curve"),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
