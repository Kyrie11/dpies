from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize cached actions/evidence in ego-centric coordinates.")
    p.add_argument("sample")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    with np.load(args.sample, allow_pickle=False) as z:
        actions = z["actions"]
        action_mask = z["action_mask"]
        evidence = z["evidence_features"]
        evidence_mask = z["evidence_mask"]
        oracle = int(z["oracle_action_index"])
        logged = z["logged_ego_future"]
    plt.figure(figsize=(8, 6))
    for k in np.where(action_mask)[0]:
        lw = 2.5 if k == oracle else 0.8
        plt.plot(actions[k, :, 0], actions[k, :, 1], linewidth=lw, alpha=0.8)
    plt.plot(logged[:, 0], logged[:, 1], "k--", linewidth=2, label="logged")
    ev = evidence[evidence_mask]
    if len(ev):
        plt.scatter(ev[:, 1], ev[:, 2], s=12, marker="x", label="evidence")
    plt.scatter([0], [0], marker="o", label="ego")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(Path(args.sample).name)
    if args.output:
        plt.savefig(args.output, dpi=160, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
