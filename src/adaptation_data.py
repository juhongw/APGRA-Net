from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .data import (
    PROJECT_ROOT,
    FMISegmentationDataset,
    _detect_reliability,
    _local_contrast,
    _robust_unit_scale,
    _seed_worker,
    load_manifest,
)


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _three_channel_input(
    image: np.ndarray, reliability: np.ndarray | None = None, periodic: bool = True,
    reliability_mode: str = "global",
) -> np.ndarray:
    reliability = _detect_reliability(image, mode=reliability_mode) if reliability is None else reliability
    if periodic:
        contrast = _local_contrast(image)
    else:
        smooth = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0, sigmaY=2.0)
        residual = np.abs(image - smooth)
        contrast = np.clip(residual / max(float(np.percentile(residual, 99.0)), 1e-6), 0, 1)
    return np.stack([image, contrast, reliability], axis=0).astype(np.float32)


def _paired_files(root: Path, dataset: str, split: str) -> list[tuple[Path, Path]]:
    image_dir = root / dataset / split / "images"
    mask_dir = root / dataset / split / "masks"
    images = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    missing_mask = sorted(set(images) - set(masks))
    missing_image = sorted(set(masks) - set(images))
    if missing_mask or missing_image:
        raise RuntimeError(
            f"Unpaired files in {dataset}/{split}: "
            f"missing_masks={missing_mask[:5]}, missing_images={missing_image[:5]}"
        )
    return [(images[key], masks[key]) for key in sorted(images)]


class BalancedProxyDataset(Dataset):
    """Domain-balanced planar thin-structure dataset; no label-derived priors are read."""

    def __init__(self, cfg: dict, split: str, training: bool) -> None:
        source_cfg = cfg["source_data"]
        root = Path(str(source_cfg.get("root", "dataset")))
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        self.domains = list(source_cfg["datasets"])
        self.records = {name: _paired_files(root, name, split) for name in self.domains}
        if any(not records for records in self.records.values()):
            raise RuntimeError(f"An empty proxy domain was selected: {self.records}")
        self.max_domain_size = max(len(records) for records in self.records.values())
        requested = int(source_cfg.get("samples_per_domain", self.max_domain_size))
        self.samples_per_domain = min(self.max_domain_size, requested)
        self.height = int(source_cfg["height"])
        self.width = int(source_cfg["width"])
        self.training = training

    def __len__(self) -> int:
        # Every domain contributes equally in every epoch.
        return self.samples_per_domain * len(self.domains)

    @lru_cache(maxsize=128)
    def _read_resized_pair(self, image_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Bound repeated decoding of small, high-resolution proxy domains."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(image_path if image is None else mask_path)
        image = _robust_unit_scale(image)
        mask = (mask > 127).astype(np.uint8)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return image.astype(np.float32), mask.astype(np.uint8)

    def __getitem__(self, index: int) -> dict[str, Any]:
        domain_index = index % len(self.domains)
        domain = self.domains[domain_index]
        domain_records = self.records[domain]
        balanced_index = index // len(self.domains)
        if len(domain_records) > self.samples_per_domain:
            record_index = (balanced_index * len(domain_records)) // self.samples_per_domain
        else:
            record_index = balanced_index % len(domain_records)
        image_path, mask_path = domain_records[record_index]
        image, mask = self._read_resized_pair(image_path, mask_path)
        if self.training:
            if random.random() < 0.5:
                image, mask = np.flip(image, 1).copy(), np.flip(mask, 1).copy()
            if random.random() < 0.5:
                image, mask = np.flip(image, 0).copy(), np.flip(mask, 0).copy()
            gamma = random.uniform(0.75, 1.35)
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-0.08, 0.08)
            image = np.clip(gain * np.power(image, gamma) + bias, 0.0, 1.0)
            if random.random() < 0.30:
                image = np.clip(
                    image + np.random.normal(0.0, random.uniform(0.005, 0.03), image.shape),
                    0.0,
                    1.0,
                )
        inputs = _three_channel_input(
            image.astype(np.float32), np.ones_like(image, np.float32), periodic=False
        )
        sample_id = f"{domain}:{image_path.stem}"
        return {
            "image": torch.from_numpy(inputs),
            "mask": torch.from_numpy(mask.astype(np.int64)),
            "sample_id": sample_id,
            "group_id": domain,
            "session_id": domain,
            "domain": domain,
        }


class FMIUnlabeledMultiViewDataset(FMISegmentationDataset):
    """Weak/strong FMI views with an exactly known azimuthal cycle transform."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg, "train", training=False)
        data_cfg = cfg["data"]
        manifest_dir = Path(str(data_cfg["manifest_dir"]))
        if not manifest_dir.is_absolute():
            manifest_dir = PROJECT_ROOT / manifest_dir
        self.records = load_manifest(manifest_dir / "unlabeled.jsonl")
        self.strong_noise_max = float(cfg.get("adaptation", {}).get("strong_noise_max", 0.035))
        self.azimuth_cycle_enabled = bool(
            cfg.get("adaptation", {}).get("azimuth_cycle_enabled", True)
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        # The unlabeled branch never opens mask_rel, even though the canonical
        # manifest retains it for later auditability.
        image = _robust_unit_scale(self._read(record["image_rel"]))
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        weak = _three_channel_input(image, reliability_mode=self.reliability_mode)
        shift = random.randrange(self.width) if self.azimuth_cycle_enabled else 0
        strong_image = np.roll(image, shift, axis=1).copy()
        gamma = random.uniform(0.70, 1.40)
        gain = random.uniform(0.82, 1.18)
        bias = random.uniform(-0.10, 0.10)
        strong_image = np.clip(gain * np.power(strong_image, gamma) + bias, 0.0, 1.0)
        if random.random() < 0.5:
            strong_image = np.clip(
                strong_image
                + np.random.normal(0.0, random.uniform(0.005, self.strong_noise_max), strong_image.shape),
                0.0,
                1.0,
            )
        strong = _three_channel_input(strong_image.astype(np.float32), reliability_mode=self.reliability_mode)
        return {
            "weak": torch.from_numpy(weak),
            "strong": torch.from_numpy(strong),
            "shift": shift,
            "sample_id": record["sample_id"],
        }


def _loader(
    dataset: Dataset,
    cfg: dict,
    training: bool,
    distributed: bool,
    rank: int,
    world_size: int,
    batch_size: int,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=training, drop_last=False
        )
    generator = torch.Generator().manual_seed(int(cfg["project"]["seed"]) + rank)
    loader_cfg = cfg["source_data"] if "source_data" in cfg else cfg["data"]
    workers = int(loader_cfg.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=training and len(dataset) >= batch_size,
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    ), sampler


def build_proxy_loader(cfg: dict, split: str, distributed: bool, rank: int, world_size: int):
    dataset = BalancedProxyDataset(cfg, split, training=split == "train")
    batch_size = int(cfg["source_data"]["batch_size"])
    return _loader(dataset, cfg, split == "train", distributed, rank, world_size, batch_size)


def build_unlabeled_loader(cfg: dict, distributed: bool, rank: int, world_size: int):
    dataset = FMIUnlabeledMultiViewDataset(cfg)
    batch_size = int(cfg["data"]["batch_size"])
    return _loader(dataset, cfg, True, distributed, rank, world_size, batch_size)
