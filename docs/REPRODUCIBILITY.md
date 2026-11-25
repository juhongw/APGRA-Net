# Reproducibility guide

## What can be reproduced publicly

The synthetic example validates the complete software path on CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash demo/run.sh
```

The command regenerates the synthetic data, trains for one epoch, selects a
threshold on the validation split, evaluates the test split, and writes
`results/synthetic_demo/metrics.json` and prediction masks.

The released paper checkpoint can be evaluated on an authorized cohort or on
another cohort using the same manifest schema:

```bash
python evaluate.py --config configs/apgra_25.yaml \
  --checkpoint checkpoints/apgranet_p025_seed3407.pt \
  --output-dir results/released_p025_seed3407
```

Verify the checkpoint before use:

```bash
sha256sum -c checkpoints/SHA256SUMS
```

## Manuscript experiment map

| Manuscript result | Configuration or file | Output |
| --- | --- | --- |
| 10%, 25%, 50%, 100% APGRA-Net validation | `configs/adaptation/confirmed/` | `labels.csv` |
| 25% three-seed architecture comparison | `configs/comparisons/*_p025_seed*.yaml` | `validation.csv` |
| 25% and 100% unseen-well test comparison | seed-3407 files in `configs/comparisons/` and `configs/adaptation/confirmed/` | `results.csv` |
| Eight APGRA-Net component ablations | `configs/adaptation/final_ablations/` | `ablation.csv` |
| Full structural-prior pretraining | `configs/proxy_pretrain/structure_all.yaml` | proxy checkpoint |
| Crack-only prior ablation | `configs/proxy_pretrain/structure_cracks.yaml` | proxy checkpoint |

Run a configured training, validation, and test sequence with:

```bash
bash tools/run_experiment.sh <configuration.yaml>
```

All label percentages refer only to training annotations. Validation and test
splits remain fixed. Operating thresholds are selected from validation images;
test images are not used for checkpoint or threshold selection.

## Limits imposed by confidentiality

Exact recomputation of the FMI numerical tables requires the proprietary 554-
image cohort and the private well-to-session key. Those files cannot be shared
under the industry agreement. The repository supplies the reported tables, all
configurations and implementations, a trained primary checkpoint, a synthetic
end-to-end test, and deidentified examples so that the software can be inspected
and executed independently within that boundary.
