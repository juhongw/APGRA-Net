from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model
from src.config import load_config, project_path, set_seed
from src.data import build_loader
from src.distributed import cleanup, gather_objects, reduce_pair, setup_distributed
from src.losses import build_loss
from src.metrics import aggregate_case_metrics, compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train APGRA-Net or a matched segmentation baseline.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def make_logger(output_dir: Path, enabled: bool) -> logging.Logger:
    logger = logging.getLogger(f"fmi_new_{output_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if enabled:
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(stream)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def cosine_with_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_epoch(
    model,
    loader,
    sampler,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    epoch: int,
    is_main: bool,
    amp_enabled: bool,
    grad_clip: float,
) -> dict:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss = 0.0
    batches = 0
    skipped_nonfinite = 0
    progress = tqdm(
        loader,
        desc=f"train {epoch:03d}",
        leave=False,
        disable=not is_main or not sys.stderr.isatty(),
    )
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(image)
            losses = criterion(outputs, target)
            loss = losses["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}: {loss.detach().item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
            error_if_nonfinite=False,
        )
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            skipped_nonfinite += 1
            continue
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        value = float(loss.detach().item())
        total_loss += value
        batches += 1
        if is_main:
            progress.set_postfix(loss=f"{total_loss / batches:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    total_loss, batches = reduce_pair(total_loss, batches, device)
    skipped_nonfinite, _ = reduce_pair(float(skipped_nonfinite), 0, device)
    return {
        "loss": total_loss / max(batches, 1),
        "skipped_nonfinite": int(skipped_nonfinite),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, cfg: dict, runtime: dict, amp_enabled: bool) -> dict:
    model.eval()
    total_loss = 0.0
    batches = 0
    cases = []
    progress = tqdm(
        loader,
        desc="validation",
        leave=False,
        disable=not runtime["is_main"] or not sys.stderr.isatty(),
    )
    for batch in progress:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(image)
            losses = criterion(outputs, target)
        probability = outputs["prob"].float().cpu().numpy()[:, 0]
        targets = target.cpu().numpy()
        for index in range(len(probability)):
            metrics = compute_metrics(
                probability[index] >= 0.5,
                targets[index],
                boundary_tolerance=int(cfg["eval"]["boundary_tolerance"]),
            )
            metrics.update(
                sample_id=batch["sample_id"][index],
                group_id=batch["group_id"][index],
                session_id=batch["session_id"][index],
            )
            probability_tensor = probability[index].astype(np.float64)
            target_tensor = targets[index].astype(np.float64)
            metrics["soft_dice"] = float(
                (2.0 * (probability_tensor * target_tensor).sum() + 1.0e-7)
                / (probability_tensor.sum() + target_tensor.sum() + 1.0e-7)
            )
            cases.append(metrics)
        total_loss += float(losses["loss"].item())
        batches += 1
    total_loss, batches = reduce_pair(total_loss, batches, device)
    gathered = gather_objects(cases, runtime["world_size"])
    cases = [case for shard in gathered for case in shard]
    cases = list({case["sample_id"]: case for case in cases}.values())
    metrics = aggregate_case_metrics(cases)
    metrics["loss"] = total_loss / max(batches, 1)
    metrics["composite"] = metrics["dice"] + 0.20 * metrics["cldice"] + 0.10 * metrics["boundary_f1"]
    return metrics


def manifest_fingerprint(cfg: dict) -> str:
    manifest_dir = Path(str(cfg["data"]["manifest_dir"]))
    if not manifest_dir.is_absolute():
        manifest_dir = PROJECT_ROOT / manifest_dir
    value = hashlib.sha256()
    for split in ("train", "val", "test"):
        path = manifest_dir / f"{split}.jsonl"
        value.update(split.encode("utf-8"))
        value.update(path.read_bytes())
    return value.hexdigest()


def save_checkpoint(
    path: Path,
    model,
    cfg: dict,
    epoch: int,
    metrics: dict,
    manifest_sha256: str,
) -> None:
    state = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
    torch.save(
        {
            "model": state,
            "config": cfg,
            "epoch": epoch,
            "validation": metrics,
            "manifest_sha256": manifest_sha256,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    runtime = setup_distributed()
    device = runtime["device"]
    set_seed(int(cfg["project"]["seed"]) + runtime["rank"])
    output_dir = project_path(cfg["project"]["output_dir"])
    conflict = False
    if runtime["is_main"]:
        output_dir.mkdir(parents=True, exist_ok=True)
        conflict = (output_dir / "best.pt").exists()
    if runtime["distributed"]:
        conflict_tensor = torch.tensor(int(conflict), device=device)
        dist.broadcast(conflict_tensor, src=0)
        conflict = bool(conflict_tensor.item())
    if conflict:
        cleanup()
        raise FileExistsError(f"Refusing to overwrite completed run: {output_dir}")
    if runtime["is_main"]:
        shutil.copy2(args.config, output_dir / "requested_config.yaml")
        (output_dir / "resolved_config.json").write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if runtime["distributed"]:
        dist.barrier()
    logger = make_logger(output_dir, runtime["is_main"])

    train_loader, train_sampler = build_loader(
        cfg, "train", runtime["distributed"], runtime["rank"], runtime["world_size"]
    )
    val_loader, _ = build_loader(
        cfg, "val", runtime["distributed"], runtime["rank"], runtime["world_size"]
    )
    model = build_model(cfg).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if runtime["distributed"]:
        model = DistributedDataParallel(model, device_ids=[runtime["local_rank"]])
    criterion = build_loss(cfg).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["optim"]["lr"]),
        weight_decay=float(cfg["optim"]["weight_decay"]),
    )
    total_steps = int(cfg["optim"]["epochs"]) * len(train_loader)
    warmup_steps = int(cfg["optim"]["warmup_epochs"]) * len(train_loader)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_with_warmup(step, warmup_steps, total_steps),
    )
    amp_enabled = device.type == "cuda" and bool(cfg["optim"].get("amp", False))
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
        init_scale=1024.0,
        growth_interval=2000,
    )

    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []
    epochs = int(cfg["optim"]["epochs"])
    patience = int(cfg["optim"]["early_stopping_patience"])
    selection_metric = str(cfg["eval"].get("selection_metric", "composite"))
    manifest_sha256 = manifest_fingerprint(cfg)
    logger.info(
        "run=%s model=%s params=%d device=%s world_size=%d train=%d val=%d",
        cfg["project"]["name"],
        cfg["model"]["name"],
        parameter_count,
        device,
        runtime["world_size"],
        len(train_loader.dataset),
        len(val_loader.dataset),
    )
    try:
        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(
                model,
                train_loader,
                train_sampler,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                epoch,
                runtime["is_main"],
                amp_enabled,
                float(cfg["optim"].get("grad_clip", 1.0)),
            )
            val_metrics = validate(model, val_loader, criterion, device, cfg, runtime, amp_enabled)
            score = float(val_metrics[selection_metric])
            improved = score > best_score + 1.0e-6
            if improved:
                best_score = score
                best_epoch = epoch
                stale_epochs = 0
                if runtime["is_main"]:
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        cfg,
                        epoch,
                        val_metrics,
                        manifest_sha256,
                    )
            else:
                stale_epochs += 1
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_skipped_nonfinite": train_metrics["skipped_nonfinite"],
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(row)
            logger.info(
                "epoch=%03d train_loss=%.4f val_dice=%.4f val_cldice=%.4f val_boundary=%.4f score=%.4f%s",
                epoch,
                train_metrics["loss"],
                val_metrics["dice"],
                val_metrics["cldice"],
                val_metrics["boundary_f1"],
                score,
                " best" if improved else "",
            )
            stop = stale_epochs >= patience
            if runtime["distributed"]:
                flag = torch.tensor(int(stop), device=device)
                dist.broadcast(flag, src=0)
                stop = bool(flag.item())
            if stop:
                logger.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
                break
        if runtime["is_main"]:
            keys = list(history[0])
            with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(history)
            summary = {
                "status": "completed",
                "parameter_count": parameter_count,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "epochs_ran": len(history),
            }
            (output_dir / "train_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    finally:
        cleanup()


if __name__ == "__main__":
    main()
