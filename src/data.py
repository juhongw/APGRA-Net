from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_fmi_root(configured: str = "auto") -> Path:
    candidates = []
    if configured != "auto":
        path = Path(configured)
        candidates.append(path if path.is_absolute() else (PROJECT_ROOT / path))
    if os.environ.get("FMI_DATA_ROOT"):
        candidates.append(Path(os.environ["FMI_DATA_ROOT"]))
    candidates.append(PROJECT_ROOT / "data" / "fmi")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("FMI data root was not found; set FMI_DATA_ROOT or data.root")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _robust_unit_scale(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low, high = np.percentile(image, (1.0, 99.0))
    if high - low < 1.0e-6:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def _detect_reliability(image: np.ndarray, mode: str = "global") -> np.ndarray:
    """Flag near-constant vertical regions using image statistics alone."""
    if mode == "global":
        column_std = image.std(axis=0)
        # Historical final model: one decision for the entire depth column.
        unreliable = (column_std < 0.015).astype(np.uint8)[None, :]
        unreliable = cv2.morphologyEx(unreliable, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
        unreliable = np.repeat(unreliable, image.shape[0], axis=0)
    elif mode == "local32":
        # Reviewer-check experiment: a 32-row variance window permits local
        # vertical losses; opening removes isolated low-variance speckles.
        mean = cv2.boxFilter(image, -1, (1, 32), borderType=cv2.BORDER_REFLECT_101)
        mean_sq = cv2.boxFilter(image * image, -1, (1, 32), borderType=cv2.BORDER_REFLECT_101)
        std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
        unreliable = (std < 0.015).astype(np.uint8)
        unreliable = cv2.morphologyEx(unreliable, cv2.MORPH_OPEN, np.ones((12, 3), np.uint8))
    else:
        raise ValueError(f"Unknown reliability detector mode: {mode}")
    return 1.0 - unreliable.astype(np.float32)


def _local_contrast(image: np.ndarray) -> np.ndarray:
    # Wrap azimuthally before smoothing so preprocessing does not create an
    # artificial left/right seam in the cylindrical image.
    horizontal_pad = 8
    padded = cv2.copyMakeBorder(
        image,
        0,
        0,
        horizontal_pad,
        horizontal_pad,
        borderType=cv2.BORDER_WRAP,
    )
    smooth = cv2.GaussianBlur(padded, (0, 0), sigmaX=2.0, sigmaY=2.0)
    smooth = smooth[:, horizontal_pad:-horizontal_pad]
    residual = np.abs(image - smooth)
    scale = float(np.percentile(residual, 99.0))
    return np.clip(residual / max(scale, 1.0e-6), 0.0, 1.0).astype(np.float32)


def _periodic_elastic_warp(
    image: np.ndarray,
    mask: np.ndarray,
    alpha_azimuth: float,
    alpha_depth: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth joint image/mask deformation on R_depth x S1_azimuth."""
    height, width = image.shape
    noise_x = np.random.normal(0.0, 1.0, (height, width)).astype(np.float32)
    noise_y = np.random.normal(0.0, 1.0, (height, width)).astype(np.float32)
    # Tile azimuth before smoothing so the random displacement is continuous
    # across the 0/360-degree seam.
    tiled_x = np.concatenate([noise_x, noise_x, noise_x], axis=1)
    tiled_y = np.concatenate([noise_y, noise_y, noise_y], axis=1)
    tiled_x = cv2.GaussianBlur(tiled_x, (0, 0), sigmaX=sigma, sigmaY=sigma)
    tiled_y = cv2.GaussianBlur(tiled_y, (0, 0), sigmaX=sigma, sigmaY=sigma)
    displacement_x = tiled_x[:, width : 2 * width]
    displacement_y = tiled_y[:, width : 2 * width]
    displacement_x /= max(float(displacement_x.std()), 1.0e-6)
    displacement_y /= max(float(displacement_y.std()), 1.0e-6)
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = np.mod(grid_x + alpha_azimuth * displacement_x, width).astype(np.float32)
    map_y = np.clip(grid_y + alpha_depth * displacement_y, 0, height - 1).astype(np.float32)
    warped_image = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask = cv2.remap(
        mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return warped_image, (warped_mask > 0).astype(np.uint8)


def _smooth_periodic_column_drift(image: np.ndarray, strength: float) -> np.ndarray:
    """Low-frequency multiplicative pad/column response variation."""
    width = image.shape[1]
    noise = np.random.normal(0.0, 1.0, (1, width)).astype(np.float32)
    tiled = np.concatenate([noise, noise, noise], axis=1)
    tiled = cv2.GaussianBlur(tiled, (0, 0), sigmaX=max(width / 18.0, 1.0), sigmaY=0.0)
    smooth = tiled[:, width : 2 * width]
    smooth = smooth / max(float(smooth.std()), 1.0e-6)
    gain = np.clip(1.0 + strength * smooth, 0.70, 1.30)
    return np.clip(image * gain, 0.0, 1.0).astype(np.float32)


class FMISegmentationDataset(Dataset):
    def __init__(self, cfg: dict, split: str, training: bool = False) -> None:
        data_cfg = cfg["data"]
        self.root = resolve_fmi_root(str(data_cfg.get("root", "auto")))
        manifest_dir = Path(str(data_cfg["manifest_dir"]))
        if not manifest_dir.is_absolute():
            manifest_dir = PROJECT_ROOT / manifest_dir
        self.records = load_manifest(manifest_dir / f"{split}.jsonl")
        self.height = int(data_cfg["height"])
        self.width = int(data_cfg["width"])
        self.training = training
        self.circular_roll = bool(data_cfg.get("circular_roll", True))
        self.blank_probability = float(data_cfg.get("blank_augment_probability", 0.0)) if training else 0.0
        self.partial_blank_probability = float(data_cfg.get("partial_blank_augment_probability", 0.0)) if training else 0.0
        self.reliability_channel = bool(data_cfg.get("reliability_channel", True))
        self.input_features = str(data_cfg.get("input_features", "standard"))
        self.reliability_mode = str(data_cfg.get("reliability_mode", "global"))
        self.depth_mix_probability = float(data_cfg.get("depth_mix_probability", 0.0)) if training else 0.0
        self.depth_mix_min_fraction = float(data_cfg.get("depth_mix_min_fraction", 0.20))
        self.depth_mix_max_fraction = float(data_cfg.get("depth_mix_max_fraction", 0.45))
        self.elastic_probability = float(data_cfg.get("elastic_probability", 0.0)) if training else 0.0
        self.elastic_alpha_azimuth = float(data_cfg.get("elastic_alpha_azimuth", 6.0))
        self.elastic_alpha_depth = float(data_cfg.get("elastic_alpha_depth", 4.0))
        self.elastic_sigma = float(data_cfg.get("elastic_sigma", 20.0))
        self.depth_scale_probability = float(data_cfg.get("depth_scale_probability", 0.0)) if training else 0.0
        self.depth_scale_min = float(data_cfg.get("depth_scale_min", 0.75))
        self.column_drift_probability = float(data_cfg.get("column_drift_probability", 0.0)) if training else 0.0
        self.column_drift_strength = float(data_cfg.get("column_drift_strength", 0.10))
        self.clahe_probability = float(data_cfg.get("clahe_probability", 0.0)) if training else 0.0

    def __len__(self) -> int:
        return len(self.records)

    def _read(self, relative: str, grayscale: bool = True) -> np.ndarray:
        flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        value = cv2.imread(str(self.root / relative), flags)
        if value is None:
            raise FileNotFoundError(self.root / relative)
        return value

    def _load_resized_pair(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        record = self.records[index]
        image = _robust_unit_scale(self._read(record["image_rel"]))
        mask = (self._read(record["mask_rel"]) > 127).astype(np.uint8)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return image.astype(np.float32), (mask > 0).astype(np.uint8)

    def _depth_interval_mix(
        self, image: np.ndarray, mask: np.ndarray, index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self.records) < 2 or random.random() >= self.depth_mix_probability:
            return image, mask
        partner = random.randrange(len(self.records) - 1)
        if partner >= index:
            partner += 1
        donor_image, donor_mask = self._load_resized_pair(partner)
        donor_shift = random.randrange(self.width)
        donor_image = np.roll(donor_image, donor_shift, axis=1)
        donor_mask = np.roll(donor_mask, donor_shift, axis=1)
        low = max(1, int(round(self.height * self.depth_mix_min_fraction)))
        high = max(low, int(round(self.height * self.depth_mix_max_fraction)))
        interval = random.randint(low, min(high, self.height))
        foreground_rows = np.flatnonzero(donor_mask.any(axis=1))
        if len(foreground_rows) and random.random() < 0.75:
            centre = int(random.choice(foreground_rows.tolist()))
            top = int(np.clip(centre - interval // 2, 0, self.height - interval))
        else:
            top = random.randint(0, self.height - interval)
        bottom = top + interval
        mixed_image = image.copy()
        mixed_mask = mask.copy()
        mixed_image[top:bottom] = donor_image[top:bottom]
        mixed_mask[top:bottom] = donor_mask[top:bottom]
        return mixed_image, mixed_mask

    def _augment(self, image: np.ndarray, mask: np.ndarray, reliability: np.ndarray) -> tuple[np.ndarray, ...]:
        if self.circular_roll:
            shift = random.randrange(self.width)
            image = np.roll(image, shift, axis=1)
            mask = np.roll(mask, shift, axis=1)
            reliability = np.roll(reliability, shift, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
            reliability = np.flip(reliability, axis=1).copy()

        if random.random() < self.depth_scale_probability:
            crop_height = max(16, int(round(self.height * random.uniform(self.depth_scale_min, 1.0))))
            top = random.randint(0, self.height - crop_height)
            image = cv2.resize(
                image[top : top + crop_height],
                (self.width, self.height),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask[top : top + crop_height],
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST,
            )

        if random.random() < self.elastic_probability:
            image, mask = _periodic_elastic_warp(
                image,
                mask,
                alpha_azimuth=self.elastic_alpha_azimuth,
                alpha_depth=self.elastic_alpha_depth,
                sigma=self.elastic_sigma,
            )

        gamma = random.uniform(0.80, 1.25)
        gain = random.uniform(0.90, 1.10)
        bias = random.uniform(-0.06, 0.06)
        image = np.clip(gain * np.power(np.clip(image, 0.0, 1.0), gamma) + bias, 0.0, 1.0)
        if random.random() < self.column_drift_probability:
            image = _smooth_periodic_column_drift(image, self.column_drift_strength)
        if random.random() < self.clahe_probability:
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
            enhanced = clahe.apply(np.round(image * 255.0).astype(np.uint8)).astype(np.float32) / 255.0
            image = np.clip(0.65 * image + 0.35 * enhanced, 0.0, 1.0)
        if random.random() < 0.25:
            sigma = random.uniform(0.005, 0.025)
            image = np.clip(image + np.random.normal(0.0, sigma, image.shape), 0.0, 1.0)
        if random.random() < 0.15:
            padded = np.concatenate([image[:, -1:], image, image[:, :1]], axis=1)
            image = cv2.GaussianBlur(padded, (3, 3), 0)[:, 1:-1]

        if random.random() < self.blank_probability:
            for _ in range(random.randint(1, 2)):
                strip_width = random.randint(max(3, self.width // 64), max(5, self.width // 12))
                left = random.randint(0, self.width - strip_width)
                fill = random.choice((0.0, float(np.median(image))))
                image[:, left : left + strip_width] = fill
        if random.random() < self.partial_blank_probability:
            # Controlled ablation: simulate a local vertical signal loss that spans only part of the image.
            for _ in range(random.randint(1, 2)):
                strip_width = random.randint(max(3, self.width // 64), max(5, self.width // 12))
                left = random.randint(0, self.width - strip_width)
                strip_height = random.randint(max(2, self.height // 4), max(3, self.height // 2))
                top = random.randint(0, self.height - strip_height)
                fill = random.choice((0.0, float(np.median(image))))
                image[top : top + strip_height, left : left + strip_width] = fill
        # Re-run the same image-only detector used at inference; synthetic
        # corruption coordinates are never supplied as an oracle input.
        reliability = _detect_reliability(image, mode=self.reliability_mode)
        return image.astype(np.float32), mask, reliability

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image, mask = self._load_resized_pair(index)
        if self.training:
            image, mask = self._depth_interval_mix(image, mask, index)
        reliability = _detect_reliability(image, mode=self.reliability_mode)
        if self.training:
            image, mask, reliability = self._augment(image, mask, reliability)
        if not self.reliability_channel:
            reliability = np.ones_like(reliability, dtype=np.float32)
        if self.input_features == "eilnet_four":
            bilateral = cv2.bilateralFilter(image, 5, 0.1, 3.0)
            sharpened = np.clip(image - 0.25 * cv2.Laplacian(image, cv2.CV_32F, ksize=3), 0.0, 1.0)
            frequency_y = np.fft.fftfreq(self.height)[:, None] * self.height
            frequency_x = np.fft.fftfreq(self.width)[None, :] * self.width
            gaussian = np.exp(-0.5 * (frequency_x**2 + frequency_y**2) / 20.0**2)
            low_pass = np.fft.ifft2(np.fft.fft2(image) * gaussian).real
            inputs = np.stack([image, bilateral, sharpened, np.clip(low_pass, 0.0, 1.0)], axis=0).astype(np.float32)
        else:
            contrast = _local_contrast(image)
            inputs = np.stack([image, contrast, reliability], axis=0).astype(np.float32)
        return {
            "image": torch.from_numpy(inputs),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "sample_id": record["sample_id"],
            "group_id": record["group_id"],
            "session_id": record["session_id"],
        }


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(
    cfg: dict,
    split: str,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DistributedSampler | None]:
    training = split == "train"
    dataset = FMISegmentationDataset(cfg, split, training=training)
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=training,
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(int(cfg["project"]["seed"]) + rank)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["data"]["batch_size"]),
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=int(cfg["data"]["num_workers"]),
        pin_memory=True,
        drop_last=False,
        persistent_workers=int(cfg["data"]["num_workers"]) > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return loader, sampler
