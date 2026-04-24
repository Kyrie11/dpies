from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

from dpies.common.config import load_yaml
from dpies.common.io import ensure_dir, write_json
from dpies.common.torch_utils import to_device
from dpies.evaluation.offline_metrics import batch_action_metrics, screening_recall_at_m
from dpies.model.network import DPIESConfig, DPIESNetwork
from dpies.selection.capped_greedy import capped_greedy_select_batch, compute_q_scores, make_directed_pair_mask
from dpies.training.collate import collate_samples
from dpies.training.dataset import EvidenceCacheDataset


@torch.no_grad()
def evaluate_budget(model, loader, device, top_m: int, budget: float, eta_e: float, gamma0: float) -> dict[str, float]:
    sums = {"action_match": 0.0, "teacher_regret": 0.0, "unresolved_rate": 0.0, "screen_recall_at_m": 0.0, "selected_count": 0.0}
    count = 0
    model.eval()
    for batch in tqdm(loader, desc=f"B={budget}"):
        batch = to_device(batch, device)
        out = model(batch)
        pair_mask = make_directed_pair_mask(out["rival_scores"], batch["action_mask"], top_m)
        selected = capped_greedy_select_batch(out["signed_evidence"], out["rival_scores"], pair_mask,
                                             batch["evidence_mask"], batch["evidence_cost"], budget, eta_e, gamma0)
        q, _ = compute_q_scores(out["signed_evidence"], selected, pair_mask, batch["action_mask"])
        metrics = batch_action_metrics(q, batch["oracle_action_index"], batch["teacher_cost"], batch["action_mask"])
        metrics["screen_recall_at_m"] = screening_recall_at_m(pair_mask, batch["rival_label"], batch["action_mask"])
        metrics["selected_count"] = float(selected.float().sum(dim=1).mean().item())
        bs = int(batch["actions"].shape[0])
        count += bs
        for k, v in metrics.items():
            sums[k] += float(v) * bs
    return {k: v / max(count, 1) for k, v in sums.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="Offline evaluate DPIES checkpoints on cached samples.")
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    cfg = load_yaml(args.config)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.cache_dir:
        cfg["cache_dir"] = args.cache_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    out_dir = ensure_dir(cfg.get("output_dir", "eval"))
    ckpt = torch.load(cfg["checkpoint"], map_location=args.device)
    train_cfg = ckpt.get("config", {})
    model_cfg = train_cfg.get("model", {})
    model = DPIESNetwork(DPIESConfig(**model_cfg)).to(args.device)
    model.load_state_dict(ckpt["model"])
    ds = EvidenceCacheDataset(cfg["cache_dir"])
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 4)), shuffle=False,
                        num_workers=int(cfg.get("num_workers", 4)), pin_memory=True, collate_fn=collate_samples)
    rows = []
    for budget in cfg.get("budgets", [32]):
        metrics = evaluate_budget(model, loader, torch.device(args.device), int(cfg.get("top_m", 4)),
                                  float(budget), float(cfg.get("eta_e", 0.05)), float(cfg.get("gamma0", 1.0)))
        row = {"budget": budget, **metrics}
        rows.append(row)
        print(json.dumps(row, indent=2))
    write_json(out_dir / "metrics.json", rows)
    with open(out_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
