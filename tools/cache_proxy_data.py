from __future__ import annotations

import argparse
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DOMAINS = ["CFD", "Crack500", "CrackTree200", "DeepCrack", "HRF", "STARE"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache resized proxy images without derived priors.")
    parser.add_argument("--source", default="dataset")
    parser.add_argument("--output", default="cache/proxy_320x256")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if not source.is_absolute(): source = ROOT / source
    if not output.is_absolute(): output = ROOT / output
    written = 0
    for domain in DOMAINS:
        for split in ("train", "val"):
            image_dir = source / domain / split / "images"
            mask_dir = source / domain / split / "masks"
            images = {path.stem: path for path in image_dir.iterdir() if path.suffix.lower() in SUFFIXES}
            masks = {path.stem: path for path in mask_dir.iterdir() if path.suffix.lower() in SUFFIXES}
            if set(images) != set(masks):
                raise RuntimeError(f"Unpaired proxy files in {domain}/{split}")
            out_images = output / domain / split / "images"
            out_masks = output / domain / split / "masks"
            out_images.mkdir(parents=True, exist_ok=True)
            out_masks.mkdir(parents=True, exist_ok=True)
            for stem in sorted(images):
                out_image = out_images / f"{stem}.png"
                out_mask = out_masks / f"{stem}.png"
                if out_image.exists() and out_mask.exists():
                    continue
                image = cv2.imread(str(images[stem]), cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
                if image is None or mask is None:
                    raise FileNotFoundError(images[stem] if image is None else masks[stem])
                image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(out_image), image)
                cv2.imwrite(str(out_mask), (mask > 127).astype("uint8") * 255)
                written += 1
    print(f"proxy_cache={output} newly_written_pairs={written}")


if __name__ == "__main__":
    main()
