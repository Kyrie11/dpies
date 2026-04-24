from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else (lambda x: x)

from dpies.common.config import deep_update, load_yaml, parse_override_items
from dpies.common.io import ensure_dir, write_json
from dpies.common.torch_utils import set_seed, to_device
from dpies.model.network import DPIESConfig, DPIESNetwork
from dpies.selection.capped_greedy import capped_greedy_select_batch, compute_q_scores, make_directed_pair_mask
from dpies.training.collate import collate_samples
from dpies.training.dataset import EvidenceCacheDataset
from dpies.training.losses import compute_total_loss
from dpies.training.validate import validate


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    cfg: dict, metrics: dict) -> None:
    ensure_dir(path.parent)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "metrics": metrics,
    }, path)


def main() -> None:
    p = argparse.ArgumentParser(description="Train DPIES model.")
    p.add_argument("--config", default="configs/train.yaml")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--val-cache-dir", default=None)
    p.add_argument("--output-dir", default="runs/dpies_main")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", default=None)
    p.add_argument("--override", nargs="*", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_override_items(args.override))
    if args.cache_dir:
        cfg.setdefault("data", {})["train_cache_dir"] = args.cache_dir
    if args.val_cache_dir:
        cfg.setdefault("data", {})["val_cache_dir"] = args.val_cache_dir
    out_dir = ensure_dir(args.output_dir)
    write_json(out_dir / "config.json", cfg)
    set_seed(int(cfg.get("training", {}).get("seed", 7)))
    device = torch.device(args.device)

    train_ds = EvidenceCacheDataset(cfg["data"]["train_cache_dir"])
    val_ds = EvidenceCacheDataset(cfg["data"].get("val_cache_dir", cfg["data"]["train_cache_dir"]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["data"].get("batch_size", 4)), shuffle=True,
                              num_workers=int(cfg["data"].get("num_workers", 4)), pin_memory=True,
                              collate_fn=collate_samples, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["data"].get("batch_size", 4)), shuffle=False,
                            num_workers=int(cfg["data"].get("num_workers", 4)), pin_memory=True,
                            collate_fn=collate_samples, drop_last=False)
    model = DPIESNetwork(DPIESConfig(**cfg.get("model", {}))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("lr", 1e-4)),
                            weight_decay=float(cfg["training"].get("weight_decay", 1e-4)))
    start_epoch = 0
    best_match = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_match = float(ckpt.get("metrics", {}).get("action_match", -1.0))
    use_amp = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_cfg = cfg.get("loss", {})
    sel_cfg = cfg.get("selection", {})
    epochs = int(cfg["training"].get("epochs", 20))
    log_interval = int(cfg["training"].get("log_interval", 20))
    history = []
    for epoch in range(start_epoch, epochs):
        model.train()
        sums = defaultdict(float)
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch)
                pair_mask = make_directed_pair_mask(out["rival_scores"], batch["action_mask"], int(sel_cfg.get("top_m", 4)))
                selected = capped_greedy_select_batch(out["signed_evidence"], out["rival_scores"], pair_mask,
                                                     batch["evidence_mask"], batch["evidence_cost"],
                                                     float(sel_cfg.get("budget", 32)),
                                                     float(sel_cfg.get("eta_e", 0.05)),
                                                     float(sel_cfg.get("gamma0", 1.0)))
                q, _ = compute_q_scores(out["signed_evidence"], selected, pair_mask, batch["action_mask"])
                loss, logs = compute_total_loss(out, batch, pair_mask, q, loss_cfg)
            scaler.scale(loss).backward()
            if float(cfg["training"].get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip_norm", 5.0)))
            scaler.step(opt)
            scaler.update()
            bs = int(batch["actions"].shape[0])
            seen += bs
            for key, val in logs.items():
                sums[key] += float(val.item()) * bs
            if step % max(log_interval, 1) == 0:
                pbar.set_postfix({k: f"{v / max(seen, 1):.4f}" for k, v in sums.items() if k.startswith("loss")})
        train_logs = {k: v / max(seen, 1) for k, v in sums.items()}
        val_metrics = validate(model, val_loader, device, sel_cfg)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_logs.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, indent=2))
        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        save_checkpoint(out_dir / "last.pt", model, opt, epoch, cfg, val_metrics)
        if val_metrics.get("action_match", 0.0) > best_match:
            best_match = val_metrics.get("action_match", 0.0)
            save_checkpoint(out_dir / "best.pt", model, opt, epoch, cfg, val_metrics)
    print(f"done. best action_match={best_match:.4f}")


if __name__ == "__main__":
    main()
