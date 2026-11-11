from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build session-disjoint labeled/unlabeled FMI splits.")
    parser.add_argument("--source", default="manifests/session_v1")
    parser.add_argument("--output", default="manifests/label_efficiency")
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--seeds", nargs="+", type=int, default=[3407, 2025, 2026])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def closest_session_subset(records: list[dict], fraction: float, seed: int) -> set[str]:
    if fraction >= 1.0:
        return {r["session_id"] for r in records}
    counts: dict[str, int] = {}
    for record in records:
        counts[record["session_id"]] = counts.get(record["session_id"], 0) + 1
    items = list(counts.items())
    random.Random(seed).shuffle(items)
    target = max(1, round(len(records) * fraction))
    # Subset-sum dynamic programming gives a reproducible closest sample count,
    # while entire acquisition sessions remain indivisible.
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for session, count in items:
        updates = {total + count: chosen + (session,) for total, chosen in list(reachable.items())}
        for total, chosen in updates.items():
            reachable.setdefault(total, chosen)
    best = min((n for n in reachable if n > 0), key=lambda n: (abs(n - target), n < target, n))
    return set(reachable[best])


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    train = read_jsonl(source / "train.jsonl")
    validation = read_jsonl(source / "val.jsonl")
    test = read_jsonl(source / "test.jsonl")
    summary = []
    for seed in args.seeds:
        for fraction in args.fractions:
            selected = closest_session_subset(train, fraction, seed)
            labeled = [r for r in train if r["session_id"] in selected]
            unlabeled = [r for r in train if r["session_id"] not in selected]
            name = f"p{round(100 * fraction):03d}_seed{seed}"
            destination = output / name
            destination.mkdir(parents=True, exist_ok=True)
            write_jsonl(destination / "train.jsonl", labeled)
            write_jsonl(destination / "unlabeled.jsonl", unlabeled)
            write_jsonl(destination / "val.jsonl", validation)
            write_jsonl(destination / "test.jsonl", test)
            digest = hashlib.sha256((destination / "train.jsonl").read_bytes()).hexdigest()
            summary.append(
                {
                    "split": name,
                    "requested_fraction": fraction,
                    "labeled_images": len(labeled),
                    "unlabeled_images": len(unlabeled),
                    "labeled_sessions": len(selected),
                    "train_manifest_sha256": digest,
                }
            )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
