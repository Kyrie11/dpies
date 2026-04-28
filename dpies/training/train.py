from __future__ import annotations

import os
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
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

def decision_outputs_fp32(out: dict) -> dict:
    """Cast tensors used by screening/selection/loss to fp32.

    AMP is useful for the neural forward pass, but the downstream
    discrete screening, masked top-k, q-score computation, and losses
    should run in fp32 for numerical stability.
    """
    out = dict(out)
    if "rival_logits" in out:
        out["rival_logits"] = out["rival_logits"].float()
        out["rival_scores"] = torch.sigmoid(out["rival_logits"])
    elif "rival_scores" in out:
        out["rival_scores"] = out["rival_scores"].float()
    if "signed_evidence" in out:
        out["signed_evidence"] = out["signed_evidence"].float()
    return out

def init_distributed() -> tuple[bool, int, int, int]:
    """Initialize torchrun/DDP if LOCAL_RANK is present."""
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        return False, 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA in this training script.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return True, local_rank, rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def reduce_train_logs(
    sums: dict,
    seen: int,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    if not distributed:
        return {k: v / max(seen, 1) for k, v in sums.items()}

    keys = sorted(sums.keys())
    values = [float(seen)] + [float(sums[k]) for k in keys]
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)

    total_seen = max(float(t[0].item()), 1.0)
    return {k: float(t[i + 1].item() / total_seen) for i, k in enumerate(keys)}

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
    distributed, local_rank, rank, world_size = init_distributed()

    cfg = load_yaml(args.config)
    cfg = deep_update(cfg, parse_override_items(args.override))
    if args.cache_dir:
        cfg.setdefault("data", {})["train_cache_dir"] = args.cache_dir
    if args.val_cache_dir:
        cfg.setdefault("data", {})["val_cache_dir"] = args.val_cache_dir
    out_dir = ensure_dir(args.output_dir)
    if is_main_process(rank):
        write_json(out_dir/"config.json", cfg)

    set_seed(int(cfg.get("training", {}).get("seed", 7)))
    if bool(cfg["training"].get("allow_tf32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    train_ds = EvidenceCacheDataset(cfg["data"]["train_cache_dir"])
    val_ds = EvidenceCacheDataset(cfg["data"].get("val_cache_dir", cfg["data"]["train_cache_dir"]))

    num_workers = int(cfg["data"].get("num_workers", 4))
    batch_size = int(cfg["data"].get("batch_size", 4))

    train_sampler = (
        DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )
        if distributed
        else None
    )

    loader_common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_samples,
        drop_last=False,
    )

    if num_workers > 0:
        loader_common["persistent_workers"] = True
        loader_common["prefetch_factor"] = int(cfg["data"].get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_common,
    )

    # Only rank 0 uses val_loader, but constructing it on all ranks is harmless.
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_common,
    )

    raw_model = DPIESNetwork(DPIESConfig(**cfg.get("model", {}))).to(device)

    if distributed:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        model = raw_model

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )

    start_epoch = 0
    best_match = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        unwrap_model(model).load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_match = float(ckpt.get("metrics", {}).get("action_match", -1.0))
    use_amp = bool(cfg["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    loss_cfg = cfg.get("loss", {})
    sel_cfg = cfg.get("selection", {})
    epochs = int(cfg["training"].get("epochs", 20))
    log_interval = int(cfg["training"].get("log_interval", 20))
    history = []

    for epoch in range(start_epoch, epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        sums = defaultdict(float)
        seen = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main_process(rank))
        two_stage_signed_evidence = bool(cfg["training"].get("two_stage_signed_evidence", True))

        if two_stage_signed_evidence and bool(loss_cfg.get("loss_on_all_pairs", False)):
            raise ValueError("training.two_stage_signed_evidence=true requires loss.loss_on_all_pairs=false")

        for step, batch in enumerate(pbar):
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if two_stage_signed_evidence:
                    out = unwrap_model(model).forward_rival(batch)
                else:
                    out = model(batch, mode="rival")

            out = decision_outputs_fp32(out)

            pair_mask = make_directed_pair_mask(
                out["rival_scores"],
                batch["action_mask"],
                int(sel_cfg.get("top_m", 4)),
            )

            if two_stage_signed_evidence:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    signed = unwrap_model(model).signed_evidence_for_pair_mask(batch, out, pair_mask)
                out["signed_evidence"] = signed.float()

            with torch.no_grad():
                selected = capped_greedy_select_batch(
                    out["signed_evidence"],
                    out["rival_scores"],
                    pair_mask,
                    batch["evidence_mask"],
                    batch["evidence_cost"],
                    float(sel_cfg.get("budget", 32)),
                    float(sel_cfg.get("eta_e", 0.05)),
                    float(sel_cfg.get("gamma0", 1.0)),
                )

            q, _ = compute_q_scores(
                out["signed_evidence"],
                selected,
                pair_mask,
                batch["action_mask"],
            )

            loss, logs = compute_total_loss(out, batch, pair_mask, q, loss_cfg)

            scaler.scale(loss).backward()

            if float(cfg["training"].get("grad_clip_norm", 0.0)) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(cfg["training"].get("grad_clip_norm", 5.0)),
                )

            scaler.step(opt)
            scaler.update()

            bs = int(batch["actions"].shape[0])
            seen += bs
            for key, val in logs.items():
                sums[key] += float(val.item()) * bs

            if is_main_process(rank) and step % max(log_interval, 1) == 0:
                pbar.set_postfix({k: f"{v / max(seen, 1):.4f}" for k, v in sums.items() if k.startswith("loss")})

        train_logs = reduce_train_logs(sums, seen, device, distributed)

        if distributed:
            dist.barrier()

        validate_every = int(cfg["training"].get("validate_every_epochs", 1))
        val_metrics = {}

        if is_main_process(rank):
            if (epoch + 1) % validate_every == 0:
                val_metrics = validate(
                    unwrap_model(model),
                    val_loader,
                    device,
                    sel_cfg,
                    max_batches=cfg["training"].get("max_val_batches", None),
                    use_amp=use_amp,
                    two_stage_signed_evidence=two_stage_signed_evidence,
                )

            row = {
                "epoch": epoch,
                "world_size": world_size,
                **{f"train_{k}": v for k, v in train_logs.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }

            history.append(row)
            print(json.dumps(row, indent=2), flush=True)

            with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
                f.flush()

            save_checkpoint(
                out_dir / "last.pt",
                unwrap_model(model),
                opt,
                epoch,
                cfg,
                val_metrics,
            )

            if val_metrics.get("action_match", 0.0) > best_match:
                best_match = val_metrics.get("action_match", 0.0)
                save_checkpoint(
                    out_dir / "best.pt",
                    unwrap_model(model),
                    opt,
                    epoch,
                    cfg,
                    val_metrics,
                )

        if distributed:
            dist.barrier()
    if distributed:
        dist.destroy_process_group()

    if is_main_process(rank):
        print(f"done. best action_match={best_match:.4f}")


if __name__ == "__main__":
    main()
