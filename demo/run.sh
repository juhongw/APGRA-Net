#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ -e results/synthetic_demo/best.pt ]]; then
  echo "results/synthetic_demo already exists; move it aside before rerunning." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES=""
python_bin="${PYTHON_BIN:-python}"
"$python_bin" demo/generate.py
"$python_bin" train_adaptation.py --config demo/config.yaml
"$python_bin" evaluate.py --config demo/config.yaml --device cpu --validation-only
"$python_bin" evaluate.py --config demo/config.yaml --device cpu
"$python_bin" - <<'PY'
import json
from pathlib import Path

result = json.loads(Path('results/synthetic_demo/metrics.json').read_text())
print('Synthetic test images:', result['test_records'])
print('Synthetic test Dice:', round(result['test']['dice'], 4))
PY
