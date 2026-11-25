"""Create deterministic synthetic FMI-like images for a public smoke example.

These images are generated from equations and random noise. They contain no
field logs, well identifiers, or expert annotations.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
HEIGHT, WIDTH = 96, 64
SPLITS = {
    "train": (0, 1, 2, 3),
    "unlabeled": (4, 5),
    "val": (6, 7),
    "test": (8, 9),
}


def make_pair(index: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3407 + index)
    y, x = np.mgrid[:HEIGHT, :WIDTH]
    phase = index * 0.37
    image = (
        0.54
        + 0.075 * np.sin(2 * np.pi * x / WIDTH + phase)
        + 0.055 * np.cos(y / 8.0 + phase)
        + rng.normal(0.0, 0.035, (HEIGHT, WIDTH))
    )
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for centre, amplitude in ((27 + index % 3, 7), (66 - index % 4, 5)):
        curve = centre + amplitude * np.sin(2 * np.pi * np.arange(WIDTH) / WIDTH + phase)
        points = np.column_stack((np.arange(WIDTH), np.rint(curve).astype(np.int32)))
        cv2.polylines(mask, [points.astype(np.int32)], False, 255, thickness=2)
    image -= 0.32 * (mask > 0)
    for left in (15, 45):
        image[:, left : left + 2] *= 0.85
    return np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8), mask


def main() -> None:
    image_dir = ROOT / "images"
    mask_dir = ROOT / "masks"
    manifest_dir = ROOT / "manifests"
    for directory in (image_dir, mask_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for split, indices in SPLITS.items():
        records = []
        for index in indices:
            name = f"synthetic_{index:02d}.png"
            image, mask = make_pair(index)
            if not cv2.imwrite(str(image_dir / name), image):
                raise OSError(image_dir / name)
            if not cv2.imwrite(str(mask_dir / name), mask):
                raise OSError(mask_dir / name)
            records.append(
                {
                    "sample_id": f"synthetic_{index:02d}",
                    "group_id": f"synthetic_{split}",
                    "session_id": f"synthetic_{split}",
                    "image_rel": f"images/{name}",
                    "mask_rel": f"masks/{name}",
                }
            )
        (manifest_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
    print("Generated 10 synthetic images, masks, and split manifests in demo/.")


if __name__ == "__main__":
    main()
