from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model, load_model_state
from src.adaptation_data import build_unlabeled_loader
from src.config import load_config, project_path, set_seed
from src.data import build_loader
from src.distributed import cleanup, gather_objects, reduce_pair, setup_distributed
from src.losses import build_loss
from train import cosine_with_warmup, make_logger, validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label-efficient FMI adaptation with azimuth-cycle consistency.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def roll_batch(value: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.roll(item, int(shift), dims=-1) for item, shift in zip(value, shifts)])


def depth_stratified_prototype_margin(
    features: torch.Tensor,
    probabilities: torch.Tensor,
    reliability: torch.Tensor,
    depth_bins: int,
    shrinkage_tau: float = 0.0,
) -> torch.Tensor:
    """Support-adaptive global/local response--host prototype discrimination.

    ``shrinkage_tau=0`` reproduces independent measured-depth prototypes. A
    positive value shrinks each class-conditional local prototype toward its
    image-global counterpart according to its reliability-weighted support.
    """
    features = F.normalize(features.float(), dim=1, eps=1e-6)
    height = features.shape[-2]
    margin = features.new_zeros((features.shape[0], 1, height, features.shape[-1]))
    global_weights = probabilities.float() * reliability.float()
    global_denominator = global_weights.flatten(2).sum(-1, keepdim=True).clamp_min(1e-6)
    global_prototypes = torch.einsum("bkhw,bchw->bkc", global_weights, features) / global_denominator
    global_prototypes = F.normalize(global_prototypes, dim=-1, eps=1e-6)
    for bin_index in range(depth_bins):
        top = round(bin_index * height / depth_bins)
        bottom = round((bin_index + 1) * height / depth_bins)
        local_features = features[:, :, top:bottom]
        local_weights = probabilities[:, :, top:bottom] * reliability[:, :, top:bottom]
        denominator = local_weights.flatten(2).sum(-1, keepdim=True).clamp_min(1e-6)
        local_prototypes = torch.einsum("bkhw,bchw->bkc", local_weights, local_features) / denominator
        local_prototypes = F.normalize(local_prototypes, dim=-1, eps=1e-6)
        if shrinkage_tau > 0.0:
            effective_support = denominator.squeeze(-1)
            alpha = effective_support / (effective_support + shrinkage_tau)
            prototypes = alpha.unsqueeze(-1) * local_prototypes + (1.0 - alpha.unsqueeze(-1)) * global_prototypes
            prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)
        else:
            prototypes = local_prototypes
        similarities = torch.einsum("bchw,bkc->bkhw", local_features, prototypes)
        margin[:, :, top:bottom] = (similarities[:, 1:2] - similarities[:, 0:1]).abs()
    return margin


def geological_response_transport(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    probabilities: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Reliability-weighted diagonal Gaussian transport for the two response populations."""
    student_features = F.normalize(student_features.float(), dim=1, eps=1e-6)
    teacher_features = F.normalize(teacher_features.float(), dim=1, eps=1e-6)
    total = student_features.new_zeros(())
    for class_index in range(2):
        weight = probabilities[:, class_index : class_index + 1] * reliability
        denominator = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        student_mean = (student_features * weight).sum(dim=(2, 3), keepdim=True) / denominator
        teacher_mean = (teacher_features * weight).sum(dim=(2, 3), keepdim=True) / denominator
        student_var = ((student_features - student_mean).square() * weight).sum((2, 3), keepdim=True) / denominator
        teacher_var = ((teacher_features - teacher_mean).square() * weight).sum((2, 3), keepdim=True) / denominator
        total = total + (student_mean - teacher_mean).square().mean()
        total = total + (student_var.sqrt() - teacher_var.sqrt()).square().mean()
    return total / 2.0


@torch.no_grad()
def update_teacher(student, teacher, momentum: float) -> None:
    for teacher_parameter, student_parameter in zip(teacher.parameters(), unwrap(student).parameters()):
        teacher_parameter.mul_(momentum).add_(student_parameter.detach(), alpha=1.0 - momentum)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), unwrap(student).buffers()):
        teacher_buffer.copy_(student_buffer)


def pseudo_objective(student_outputs: dict, teacher_outputs: dict, weak: torch.Tensor, strong: torch.Tensor, shifts: torch.Tensor, cfg: dict):
    adaptation = cfg["adaptation"]
    teacher_probability = roll_batch(teacher_outputs["prob"].detach(), shifts)
    teacher_features = teacher_outputs["decoder_features"].detach()
    reliability_weak = weak[:, 2:3].clamp(0.0, 1.0)
    probabilities = torch.cat([1.0 - teacher_outputs["prob"].detach(), teacher_outputs["prob"].detach()], dim=1)
    rolled_teacher_features = roll_batch(teacher_features, shifts)
    rolled_reliability = roll_batch(reliability_weak, shifts)
    prototype_margin = depth_stratified_prototype_margin(
        rolled_teacher_features,
        torch.cat([1.0 - teacher_probability, teacher_probability], dim=1),
        rolled_reliability,
        depth_bins=int(adaptation.get("prototype_depth_bins", 4)),
        shrinkage_tau=float(adaptation.get("prototype_shrinkage_tau", 0.0)),
    )
    confidence = torch.maximum(teacher_probability, 1.0 - teacher_probability)
    reliable = strong[:, 2:3] >= float(adaptation.get("reliability_threshold", 0.5))
    accepted = (
        (confidence >= float(adaptation.get("confidence_threshold", 0.80)))
        & (prototype_margin >= float(adaptation.get("prototype_margin", 0.05)))
        & reliable
    )
    pseudo_target = (teacher_probability[:, 0] >= 0.5).long()
    pixel_ce = F.cross_entropy(student_outputs["logits"].float(), pseudo_target, reduction="none")
    mask = accepted[:, 0].float()
    pseudo_ce = (pixel_ce * mask).sum() / mask.sum().clamp_min(1.0)
    student_probability = student_outputs["prob"].float()
    consistency = ((student_probability - teacher_probability).square() * accepted.float()).sum()
    consistency = consistency / accepted.float().sum().clamp_min(1.0)
    transport_weight = float(adaptation.get("georesponse_transport_weight", 0.0))
    if transport_weight > 0.0:
        transport = geological_response_transport(
            student_outputs["decoder_features"],
            rolled_teacher_features,
            torch.cat([1.0 - teacher_probability, teacher_probability], dim=1),
            strong[:, 2:3].clamp(0.0, 1.0),
        )
    else:
        transport = pseudo_ce.new_zeros(())
    loss = (
        pseudo_ce
        + float(adaptation.get("cycle_consistency_weight", 0.5)) * consistency
        + transport_weight * transport
    )
    return loss, accepted.float().mean()


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_epoch(model, teacher, labeled_loader, unlabeled_loader, labeled_sampler, unlabeled_sampler, criterion, optimizer, scheduler, scaler, device, epoch, cfg, amp):
    model.train(); teacher.eval()
    if labeled_sampler is not None: labeled_sampler.set_epoch(epoch)
    if unlabeled_sampler is not None: unlabeled_sampler.set_epoch(epoch)
    labeled_iterator = iter(labeled_loader)
    unlabeled_iterator = iter(unlabeled_loader) if unlabeled_loader is not None else None
    steps = max(len(labeled_loader), len(unlabeled_loader) if unlabeled_loader is not None else 0)
    total = supervised_total = unsupervised_total = coverage_total = 0.0
    ramp_epochs = max(1, int(cfg["adaptation"].get("ramp_epochs", 10)))
    unsupervised_weight = float(cfg["adaptation"].get("unsupervised_weight", 1.0)) * min(1.0, epoch / ramp_epochs)
    for _ in range(steps):
        labeled, labeled_iterator = next_batch(labeled_iterator, labeled_loader)
        image = labeled["image"].to(device, non_blocking=True)
        target = labeled["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            supervised = criterion(model(image), target)["loss"]
            if unlabeled_loader is not None:
                unlabeled, unlabeled_iterator = next_batch(unlabeled_iterator, unlabeled_loader)
                weak = unlabeled["weak"].to(device, non_blocking=True)
                strong = unlabeled["strong"].to(device, non_blocking=True)
                shifts = unlabeled["shift"].to(device)
                with torch.no_grad():
                    teacher_outputs = teacher(weak)
                student_outputs = model(strong)
                unsupervised, coverage = pseudo_objective(student_outputs, teacher_outputs, weak, strong, shifts, cfg)
            else:
                unsupervised = supervised.new_zeros(())
                coverage = supervised.new_zeros(())
            loss = supervised + unsupervised_weight * unsupervised
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite adaptation loss at epoch {epoch}")
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["optim"].get("grad_clip", 1.0)))
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            scheduler.step()
            update_teacher(model, teacher, float(cfg["adaptation"].get("ema_momentum", 0.996)))
        total += float(loss.detach()); supervised_total += float(supervised.detach())
        unsupervised_total += float(unsupervised.detach()); coverage_total += float(coverage.detach())
    values = []
    for value in (total, supervised_total, unsupervised_total, coverage_total):
        reduced, count = reduce_pair(value, steps, device); values.append(reduced / max(count, 1))
    return {"loss": values[0], "supervised_loss": values[1], "unsupervised_loss": values[2], "pseudo_coverage": values[3], "unsupervised_weight": unsupervised_weight}


@torch.no_grad()
def fast_selection_validate(model, loader, device, runtime, amp):
    """Compute the registered checkpoint-selection metric without costly morphology."""
    model.eval()
    local_cases = []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True).float()
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            probability = model(image)["prob"][:, 0].float()
        intersection = (probability * target).flatten(1).sum(1)
        soft = (2.0 * intersection + 1.0e-7) / (probability.flatten(1).sum(1) + target.flatten(1).sum(1) + 1.0e-7)
        hard_prediction = (probability >= 0.5).float()
        hard = (2.0 * (hard_prediction * target).flatten(1).sum(1) + 1.0e-7) / (hard_prediction.flatten(1).sum(1) + target.flatten(1).sum(1) + 1.0e-7)
        local_cases.extend(zip(batch["sample_id"], soft.cpu().tolist(), hard.cpu().tolist()))
    cases = {sample_id: (soft, hard) for shard in gather_objects(local_cases, runtime["world_size"]) for sample_id, soft, hard in shard}
    return {"soft_dice": sum(value[0] for value in cases.values()) / len(cases), "dice": sum(value[1] for value in cases.values()) / len(cases)}


def main() -> None:
    args = parse_args(); cfg = load_config(args.config); runtime = setup_distributed(); device = runtime["device"]
    set_seed(int(cfg["project"]["seed"]) + runtime["rank"])
    output_dir = project_path(cfg["project"]["output_dir"])
    conflict = (output_dir / "best.pt").exists() if runtime["is_main"] else False
    if runtime["distributed"]:
        flag = torch.tensor(int(conflict), device=device); dist.broadcast(flag, 0); conflict = bool(flag.item())
    if conflict:
        cleanup(); raise FileExistsError(f"Refusing to overwrite completed run: {output_dir}")
    if runtime["is_main"]:
        output_dir.mkdir(parents=True, exist_ok=True); shutil.copy2(args.config, output_dir / "requested_config.yaml")
        (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if runtime["distributed"]: dist.barrier()
    logger = make_logger(output_dir, runtime["is_main"])
    labeled_loader, labeled_sampler = build_loader(cfg, "train", runtime["distributed"], runtime["rank"], runtime["world_size"])
    val_loader, _ = build_loader(cfg, "val", runtime["distributed"], runtime["rank"], runtime["world_size"])
    manifest_dir = project_path(cfg["data"]["manifest_dir"])
    has_unlabeled = (
        bool(cfg["adaptation"].get("use_unlabeled", True))
        and (manifest_dir / "unlabeled.jsonl").exists()
        and (manifest_dir / "unlabeled.jsonl").stat().st_size > 0
    )
    unlabeled_loader, unlabeled_sampler = build_unlabeled_loader(cfg, runtime["distributed"], runtime["rank"], runtime["world_size"]) if has_unlabeled else (None, None)
    model = build_model(cfg).to(device)
    initialization = cfg["adaptation"].get("initial_checkpoint")
    if initialization:
        checkpoint_path = project_path(initialization); checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        load_model_state(model, checkpoint["model"], strict=True)
    teacher = copy.deepcopy(model).to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    parameters = sum(p.numel() for p in model.parameters())
    if runtime["distributed"]: model = DistributedDataParallel(model, device_ids=[runtime["local_rank"]])
    criterion = build_loss(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg["optim"]["lr"]), weight_decay=float(cfg["optim"]["weight_decay"]))
    epochs = int(cfg["optim"]["epochs"]); steps = max(len(labeled_loader), len(unlabeled_loader) if unlabeled_loader is not None else 0)
    scheduler = LambdaLR(optimizer, lambda step: cosine_with_warmup(step, int(cfg["optim"]["warmup_epochs"]) * steps, epochs * steps))
    amp = device.type == "cuda" and bool(cfg["optim"].get("amp", True)); scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = -math.inf; best_epoch = 0; stale = 0; history = []
    try:
        for epoch in range(1, epochs + 1):
            train_metrics = train_epoch(model, teacher, labeled_loader, unlabeled_loader, labeled_sampler, unlabeled_sampler, criterion, optimizer, scheduler, scaler, device, epoch, cfg, amp)
            val_metrics = (fast_selection_validate(teacher, val_loader, device, runtime, amp)
                           if cfg["eval"].get("fast_selection", False)
                           else validate(teacher, val_loader, criterion, device, cfg, runtime, amp))
            score = float(val_metrics[str(cfg["eval"].get("selection_metric", "soft_dice"))]); improved = score > best + 1e-6
            if improved:
                best, best_epoch, stale = score, epoch, 0
                if runtime["is_main"]:
                    fingerprint = hashlib.sha256((manifest_dir / "train.jsonl").read_bytes()).hexdigest()
                    torch.save({"model": teacher.state_dict(), "student_model": unwrap(model).state_dict(), "config": cfg, "epoch": epoch, "validation": val_metrics, "labeled_manifest_sha256": fingerprint}, output_dir / "best.pt")
            else: stale += 1
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}; history.append(row)
            logger.info("epoch=%03d loss=%.4f coverage=%.3f val_dice=%.4f score=%.4f%s", epoch, train_metrics["loss"], train_metrics["pseudo_coverage"], val_metrics["dice"], score, " best" if improved else "")
            stop = stale >= int(cfg["optim"]["early_stopping_patience"])
            if runtime["distributed"]:
                flag = torch.tensor(int(stop), device=device); dist.broadcast(flag, 0); stop = bool(flag.item())
            if stop: break
        if runtime["is_main"]:
            with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
            (output_dir / "train_summary.json").write_text(json.dumps({"status": "completed", "parameter_count": parameters, "best_epoch": best_epoch, "best_score": best, "epochs_ran": len(history), "initial_checkpoint": initialization, "used_unlabeled_data": has_unlabeled}, indent=2), encoding="utf-8")
    finally: cleanup()


if __name__ == "__main__":
    main()
