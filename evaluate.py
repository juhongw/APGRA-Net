from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import build_model, load_model_state
from src.config import load_config, project_path, set_seed
from src.data import build_loader
from src.metrics import aggregate_case_metrics, compute_metrics


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


def assert_checkpoint_compatibility(cfg: dict, checkpoint: dict) -> None:
    checkpoint_cfg = checkpoint.get("config", {})
    for section in ("data", "model", "loss"):
        if checkpoint_cfg.get(section) != cfg.get(section):
            raise ValueError(f"Checkpoint/config mismatch in section '{section}'")
    if checkpoint_cfg.get("project", {}).get("name") != cfg.get("project", {}).get("name"):
        raise ValueError("Checkpoint/config project name mismatch")
    expected_manifest = checkpoint.get("manifest_sha256")
    current_manifest = manifest_fingerprint(cfg)
    if expected_manifest and expected_manifest != current_manifest:
        raise ValueError("Manifest fingerprint differs from the training checkpoint")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a threshold on validation data and evaluate once on test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional evidence directory; the checkpoint still comes from the canonical run.",
    )
    parser.add_argument(
        "--tta-rolls",
        nargs="*",
        type=int,
        default=(),
        help="Additional azimuth rolls whose inverse-aligned probabilities are averaged.",
    )
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def infer(
    model,
    loader,
    device: torch.device,
    amp_enabled: bool = False,
    tta_rolls: tuple[int, ...] = (),
) -> list[dict]:
    model.eval()
    records = []
    for batch in tqdm(loader, desc="inference", leave=False, disable=not sys.stderr.isatty()):
        image = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            probability_sum = model(image)["prob"]
            ensemble_count = 1
            for requested_shift in tta_rolls:
                shift = int(requested_shift) % image.shape[-1]
                if shift == 0:
                    continue
                rolled_probability = model(torch.roll(image, shifts=shift, dims=-1))["prob"]
                probability_sum = probability_sum + torch.roll(
                    rolled_probability, shifts=-shift, dims=-1
                )
                ensemble_count += 1
            probability = (probability_sum / ensemble_count)[:, 0]
        probability = probability.float().cpu().numpy()
        target = batch["mask"].numpy()
        input_gray = batch["image"][:, 0].numpy()
        for index in range(len(probability)):
            records.append(
                {
                    "sample_id": batch["sample_id"][index],
                    "group_id": batch["group_id"][index],
                    "session_id": batch["session_id"][index],
                    "probability": probability[index],
                    "target": target[index],
                    "input_gray": input_gray[index],
                }
            )
    return records


def score_records(records: list[dict], threshold: float, tolerance: int) -> tuple[dict, list[dict]]:
    cases = []
    for record in records:
        metrics = compute_metrics(
            record["probability"] >= threshold,
            record["target"],
            boundary_tolerance=tolerance,
        )
        metrics.update(
            sample_id=record["sample_id"],
            group_id=record["group_id"],
            session_id=record["session_id"],
        )
        cases.append(metrics)
    return aggregate_case_metrics(cases), cases


def fast_macro_dice(records: list[dict], threshold: float) -> float:
    values = []
    for record in records:
        prediction = record["probability"] >= threshold
        target = record["target"].astype(bool)
        intersection = float((prediction & target).sum())
        denominator = float(prediction.sum() + target.sum())
        values.append((2.0 * intersection + 1.0e-7) / (denominator + 1.0e-7))
    return float(np.mean(values))


def save_case_csv(path: Path, cases: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)


def save_predictions(output_dir: Path, records: list[dict], threshold: float) -> None:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        prediction = (record["probability"] >= threshold).astype(np.uint8)
        cv2.imwrite(str(pred_dir / f"{record['sample_id']}_pred.png"), prediction * 255)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    canonical_output_dir = project_path(cfg["project"]["output_dir"])
    output_dir = project_path(args.output_dir) if args.output_dir else canonical_output_dir
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else canonical_output_dir / "best.pt"
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert_checkpoint_compatibility(cfg, checkpoint)
    load_model_state(model, checkpoint["model"], strict=True)

    val_loader, _ = build_loader(cfg, "val")
    test_loader, _ = build_loader(cfg, "test")
    amp_enabled = device.type == "cuda" and bool(cfg["optim"].get("amp", False))
    tta_rolls = tuple(dict.fromkeys(int(value) for value in args.tta_rolls))
    validation_records = infer(model, val_loader, device, amp_enabled, tta_rolls)
    eval_cfg = cfg["eval"]
    thresholds = np.arange(
        float(eval_cfg["threshold_min"]),
        float(eval_cfg["threshold_max"]) + 0.5 * float(eval_cfg["threshold_step"]),
        float(eval_cfg["threshold_step"]),
    )
    tolerance = int(eval_cfg["boundary_tolerance"])
    threshold_metric = str(eval_cfg.get("threshold_metric", "dice"))
    sweep = []
    best = None
    for threshold in thresholds:
        if threshold_metric != "dice":
            raise ValueError("Fast threshold selection currently supports threshold_metric=dice")
        row = {"threshold": float(threshold), "dice": fast_macro_dice(validation_records, float(threshold))}
        sweep.append(row)
        if best is None or row[threshold_metric] > best[threshold_metric]:
            best = row
    assert best is not None
    selected_threshold = float(best["threshold"])
    validation_metrics, validation_cases = score_records(validation_records, selected_threshold, tolerance)

    if args.validation_only:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_case_csv(output_dir / "validation_cases.csv", validation_cases)
        with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
            writer.writeheader()
            writer.writerows(sweep)
        result = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "threshold_selected_on_validation": selected_threshold,
            "threshold_metric": threshold_metric,
            "tta_rolls": list(tta_rolls),
            "validation": validation_metrics,
            "test_evaluated": False,
        }
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if (output_dir / "metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing test evidence package: {output_dir}")

    # Test inference happens only after the validation checkpoint and threshold are fixed.
    test_records = infer(model, test_loader, device, amp_enabled, tta_rolls)
    test_metrics, test_cases = score_records(test_records, selected_threshold, tolerance)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_case_csv(output_dir / "validation_cases.csv", validation_cases)
    save_case_csv(output_dir / "test_cases.csv", test_cases)
    with (output_dir / "threshold_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep[0]))
        writer.writeheader()
        writer.writerows(sweep)
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "manifest_sha256": manifest_fingerprint(cfg),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "threshold_selected_on_validation": selected_threshold,
        "threshold_metric": threshold_metric,
        "tta_rolls": list(tta_rolls),
        "validation": validation_metrics,
        "test": test_metrics,
        "test_records": len(test_cases),
        "test_sessions": len({case["session_id"] for case in test_cases}),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if bool(eval_cfg.get("save_predictions", True)):
        save_predictions(output_dir, test_records, selected_threshold)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
