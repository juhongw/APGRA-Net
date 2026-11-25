#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

config="${1:?Usage: bash tools/run_experiment.sh configs/path/to/experiment.yaml}"
if [[ ! -f "$config" ]]; then
  echo "Configuration not found: $config" >&2
  exit 2
fi

python_bin="${PYTHON_BIN:-python}"
"$python_bin" train_adaptation.py --config "$config"
"$python_bin" evaluate.py --config "$config" --validation-only
"$python_bin" evaluate.py --config "$config"
