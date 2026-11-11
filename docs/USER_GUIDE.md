# User guide

## 1. Install

Use Python 3.10 or newer. Create a virtual environment and install the packages
listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install a CPU or CUDA PyTorch build compatible with your machine. The
[README](../README.md#dependencies-and-compute) lists the versions used to
verify this release and the hardware used for the paper experiments.

## 2. Run the public example

From the repository root:

```bash
bash demo/run.sh
```

The script uses the ten generated examples already included in `demo/`. It
regenerates them deterministically, trains a reduced APGRA-Net for one epoch on
CPU, selects a threshold using only the synthetic validation split, then
evaluates the synthetic test split. The output is
`results/synthetic_demo/metrics.json`; predictions are in
`results/synthetic_demo/predictions/`. The script refuses to overwrite an
existing checkpoint. Move `results/synthetic_demo/` aside to run it again.

The demo checks the software workflow. It does not reproduce the paper's FMI
scores because the field images and trained field checkpoints are restricted.

## 3. Prepare FMI data for research runs

Place your own grayscale images and binary masks under `data/fmi/`, or set
`FMI_DATA_ROOT` to another directory. Create
`manifests/session_v1/train.jsonl`, `val.jsonl`, and `test.jsonl`. Each line
must have `sample_id`, `group_id`, `session_id`, `image_rel`, and `mask_rel`, as
shown in `manifests/example.jsonl`. Image and mask paths are relative to the
FMI data root. Encode fracture pixels as 255 and background as 0; the loader
thresholds masks at 127. Keep each acquisition
session in one partition. The paper's cohort contained 398 training, 86
validation, and 70 test images; its split was session disjoint, not well
disjoint.

Build labeled and unlabeled training subsets by acquisition session:

```bash
python tools/build_label_efficiency_manifests.py
```

This creates the 10%, 25%, 50%, and 100% label fractions for seeds 3407, 2025,
and 2026. It writes JSONL manifests only and never copies the private images.

## 4. Prepare structural proxy data

The complete method initializes from CFD, Crack500, CrackTree200, DeepCrack,
HRF, and STARE. Obtain these datasets from their original providers and place
paired images and masks under
`data/proxy/<dataset>/<train-or-val>/{images,masks}/`. Proxy data provide
structural pretraining only; they are not FMI test cases.

```bash
python tools/cache_proxy_data.py --source data/proxy
python pretrain_proxy.py --config configs/pretrain.yaml
```

The checkpoint is written to `results/proxy_pretrain/structure_all/best.pt`.
For the crack-only ablation, run
`python pretrain_proxy.py --config configs/proxy_pretrain/structure_cracks.yaml`.

## 5. Train and evaluate

For the 25% APGRA-Net experiment:

```bash
python train_adaptation.py --config configs/apgra_25.yaml
python evaluate.py --config configs/apgra_25.yaml --validation-only
python evaluate.py --config configs/apgra_25.yaml
```

Use `configs/apgra_100.yaml` for 100% training labels. The 25% validation
comparison also uses the 2025 and 2026 seed configurations in
`configs/adaptation/confirmed/`. Comparison configurations are under
`configs/comparisons/`; APGRA-Net component ablations are under
`configs/adaptation/final_ablations/`. To run a single comparison or ablation
configuration, use `bash tools/run_experiment.sh <config.yaml>` after preparing
the needed proxy checkpoint and manifests.

`evaluate.py` selects the prediction threshold on the validation split before
scoring the test split. It refuses to overwrite an existing `metrics.json`.
Each run saves `best.pt`, `validation_metrics.json`, `metrics.json`,
`test_cases.csv`, and prediction masks under its configured output directory.
For CPU evaluation, add `--device cpu`; for CPU training, set
`CUDA_VISIBLE_DEVICES=""` before the command.

## 6. Read the manuscript tables

- `results.csv`: seed 3407 FMI test comparison at 25% and 100% training labels.
- `validation.csv`: three-seed validation means and sample standard deviations.
- `ablation.csv`: component ablations scored on the same 70 FMI test images.
- `labels.csv`: APGRA-Net validation results by training-label fraction.

These CSVs match the rounded values in the submitted manuscript. The
private 70-image test cohort was finalized after preliminary model outputs
were examined, so the scores describe that cohort rather than an untouched
confirmatory holdout. The synthetic demo uses separate, generated data and is
not a source for these table values.
