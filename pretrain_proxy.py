from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from src.adaptation_data import build_proxy_loader
from src.config import load_config, project_path, set_seed
from src.distributed import cleanup, gather_objects, reduce_pair, setup_distributed
from src.losses import build_loss
from train import cosine_with_warmup, make_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Domain-balanced thin-structure proxy pretraining.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _state(model):
    return model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()


def train_epoch(model, loader, sampler, criterion, optimizer, scheduler, scaler, device, epoch, amp):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total = 0.0
    count = 0
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            loss = criterion(model(image), target)["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite proxy loss at epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += float(loss.detach())
        count += 1
    total, count = reduce_pair(total, count, device)
    return total / max(count, 1)


@torch.no_grad()
def validate(model, loader, runtime, device, amp):
    model.eval()
    cases = []
    for batch in loader:
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            probability = model(batch["image"].to(device, non_blocking=True))["prob"][:, 0]
        target = batch["mask"].to(device, non_blocking=True).float()
        intersection = (probability * target).flatten(1).sum(1)
        dice = (2 * intersection + 1e-6) / (
            probability.flatten(1).sum(1) + target.flatten(1).sum(1) + 1e-6
        )
        cases.extend(zip(batch["sample_id"], batch["domain"], dice.cpu().tolist()))
    shards = gather_objects(cases, runtime["world_size"])
    unique = {sample: (domain, value) for shard in shards for sample, domain, value in shard}
    by_domain: dict[str, list[float]] = {}
    for domain, value in unique.values():
        by_domain.setdefault(domain, []).append(value)
    domain_dice = {domain: sum(values) / len(values) for domain, values in by_domain.items()}
    return sum(domain_dice.values()) / len(domain_dice), domain_dice


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = setup_distributed()
    device = runtime["device"]
    set_seed(int(cfg["project"]["seed"]) + runtime["rank"])
    output_dir = project_path(cfg["project"]["output_dir"])
    conflict = (output_dir / "best.pt").exists() if runtime["is_main"] else False
    if runtime["distributed"]:
        flag = torch.tensor(int(conflict), device=device)
        dist.broadcast(flag, 0)
        conflict = bool(flag.item())
    if conflict:
        cleanup()
        raise FileExistsError(f"Refusing to overwrite completed run: {output_dir}")
    if runtime["is_main"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, output_dir / "requested_config.yaml")
        (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if runtime["distributed"]:
        dist.barrier()
    logger = make_logger(output_dir, runtime["is_main"])
    train_loader, train_sampler = build_proxy_loader(
        cfg, "train", runtime["distributed"], runtime["rank"], runtime["world_size"]
    )
    val_loader, _ = build_proxy_loader(
        cfg, "val", runtime["distributed"], runtime["rank"], runtime["world_size"]
    )
    model = build_model(cfg).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    if runtime["distributed"]:
        model = DistributedDataParallel(model, device_ids=[runtime["local_rank"]])
    criterion = build_loss(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    epochs = int(cfg["optim"]["epochs"])
    total_steps = epochs * len(train_loader)
    warmup = int(cfg["optim"]["warmup_epochs"]) * len(train_loader)
    scheduler = LambdaLR(optimizer, lambda step: cosine_with_warmup(step, warmup, total_steps))
    amp = device.type == "cuda" and bool(cfg["optim"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -math.inf
    stale = 0
    best_epoch = 0
    patience = int(cfg["optim"].get("early_stopping_patience", epochs))
    history = []
    try:
        for epoch in range(1, epochs + 1):
            loss = train_epoch(model, train_loader, train_sampler, criterion, optimizer, scheduler, scaler, device, epoch, amp)
            score, domain_dice = validate(model, val_loader, runtime, device, amp)
            improved = score > best
            if improved:
                best = score
                best_epoch = epoch
                stale = 0
                if runtime["is_main"]:
                    torch.save({"model": _state(model), "config": cfg, "epoch": epoch, "proxy_macro_soft_dice": score, "domain_soft_dice": domain_dice}, output_dir / "best.pt")
            else:
                stale += 1
            row = {"epoch": epoch, "train_loss": loss, "proxy_macro_soft_dice": score, **{f"dice_{k}": v for k, v in domain_dice.items()}}
            history.append(row)
            logger.info("epoch=%03d loss=%.4f macro_soft_dice=%.4f%s", epoch, loss, score, " best" if improved else "")
            stop = stale >= patience
            if runtime["distributed"]:
                flag = torch.tensor(int(stop), device=device)
                dist.broadcast(flag, 0)
                stop = bool(flag.item())
            if stop:
                logger.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
                break
        if runtime["is_main"]:
            with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader(); writer.writerows(history)
            (output_dir / "train_summary.json").write_text(json.dumps({"status": "completed", "parameter_count": parameters, "best_epoch": best_epoch, "best_macro_soft_dice": best, "epochs_ran": len(history)}, indent=2), encoding="utf-8")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
