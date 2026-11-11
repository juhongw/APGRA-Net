# APGRA-Net

APGRA-Net segments expert-labeled fractures in formation microresistivity images (FMI). This repository contains the research code, experiment configurations, a runnable synthetic example, and the tables reported in the manuscript. Field FMI images, expert masks, and trained field checkpoints are not distributed.

## Quick start: synthetic CPU example

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash demo/run.sh
```

The script generates ten synthetic images and masks, trains a small APGRA-Net for one epoch, selects a threshold on two validation images, and scores two test images. It writes `results/synthetic_demo/metrics.json` and prediction masks. No FMI data or GPU is needed. These demo scores are **not** the paper's FMI results. See [the synthetic example](demo/README.md) and [the user guide](docs/USER_GUIDE.md).

## Dependencies and compute

- Python 3.10 or newer, PyTorch, torchvision, NumPy, OpenCV, SciPy, scikit-image, PyYAML, and tqdm; install the version ranges in `requirements.txt`.
- The synthetic example is CPU runnable at 96 × 64 pixels and one training epoch.
- The paper experiments used 384 × 256 images, batch size 8 for APGRA-Net, up to 120 adaptation epochs, and an NVIDIA RTX 4090 (24 GB) in the reference environment. APGRA-Net has 1.189 M parameters and approximately 1.687 G MACs per image. A GPU is recommended for the full training runs.
- The verified environment used Python 3.10.20, PyTorch 2.12.0, torchvision 0.27.0, NumPy 2.2.6, SciPy 1.15.3, OpenCV 4.13.0, scikit-image 0.25.2, PyYAML 6.0.3, and tqdm 4.67.3. Install a PyTorch build appropriate for your CPU or CUDA setup.

## License

See [LICENSE](LICENSE). The external proxy datasets and private FMI data retain their own access and licensing terms.

## Files and directories

| Path | Purpose |
| --- | --- |
| `README.md` | Project overview and instructions. |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | License for the code in this repository. |
| `.gitignore` | Excludes data, caches, checkpoints, and run outputs from Git. |
| `docs/USER_GUIDE.md` | Data format, training, evaluation, and result interpretation. |
| `demo/` | Generated synthetic images, split manifests, mini configuration, and a CPU run script. |
| `results.csv` | The manuscript's 25% and 100% FMI test comparison, including parameters, MACs, and segmentation metrics. |
| `validation.csv` | The manuscript's 25% validation comparison (three-seed mean and sample standard deviation). |
| `ablation.csv` | The manuscript's 25% component ablations on the same 70 test images. |
| `labels.csv` | APGRA-Net validation results at 10%, 25%, 50%, and 100% training-label budgets. |
| `pretrain_proxy.py` | Pretrains on external curvilinear datasets and saves a checkpoint. |
| `train_adaptation.py` | Adapts the pretrained model to FMI using labeled and, when available, unlabeled training images; updates a teacher model from the student. |
| `train.py` | Standard supervised training entry point for experiments without the adaptation pipeline. |
| `evaluate.py` | Selects a prediction threshold on validation data, then computes test metrics and saves prediction masks. |
| `models/apgranet.py` | APGRA-Net model entry point. |
| `models/cylfbnet.py` | Periodic convolutions, strip context, foreground/background interaction, and encoder-decoder architecture. |
| `models/baselines.py` | General segmentation baselines, including U-Net. |
| `models/eilnet.py` | EILNet comparison model. |
| `models/__init__.py` | Builds models from configurations and loads model weights. |
| `src/config.py` | Loads YAML configurations, merges `base` configurations, and sets random seeds. |
| `src/data.py` | Loads FMI images, masks, and manifests; applies preprocessing and augmentation. |
| `src/adaptation_data.py` | Loads proxy data and unlabeled FMI images and builds training views. |
| `src/losses.py` | Training losses, including Dice, clDice, and contrastive losses. |
| `src/metrics.py` | Evaluation metrics, including Dice, IoU, clDice, and boundary F1. |
| `src/distributed.py` | Multi-GPU setup and result aggregation. |
| `src/__init__.py` | Initializes the `src` package. |
| `tools/cache_proxy_data.py` | Resizes and caches the external proxy datasets. |
| `tools/build_label_efficiency_manifests.py` | Creates training-label subsets by acquisition session and random seed. |
| `tools/run_experiment.sh` | Trains one configured experiment, selects its threshold on validation images, and evaluates the test images. |
| `configs/pretrain.yaml` | Default entry point for APGRA-Net proxy pretraining. |
| `configs/apgra_25.yaml`, `configs/apgra_100.yaml` | Use 25% and 100% of the FMI **training labels**, respectively. |
| `configs/proxy_pretrain/structure_all.yaml` | Full configuration for APGRA-Net proxy pretraining. |
| `configs/proxy_pretrain/structure_cracks.yaml` | Crack-only proxy pretraining for one ablation. |
| `configs/adaptation/base.yaml` | Shared parameters for FMI adaptation. |
| `configs/adaptation/main_p025_seed3407.yaml` | Base configuration for the 25% label experiment. |
| `configs/adaptation/confirmed/apgra_final_p{010,025,050,100}_seed{3407,2025,2026}.yaml` | APGRA-Net configurations for each label fraction and seed combination. |
| `configs/comparisons/` | The U-Net, U-Net++, UNETR-like, DeepLabV3-ResNet50, SegFormer-B0, and EILNet configurations. |
| `configs/adaptation/final_ablations/` | Eight APGRA-Net component variants from the manuscript's test ablation table. |
| `manifests/example.jsonl` | Example of one image-manifest record. |
| `manifests/README.md` | Manifest fields and split instructions. |
| `examples/example_01.png`, `example_02.png`, `example_03.png` | FMI input, expert mask, model prediction, and overlay. |
| `examples/README.md` | Color legend for the visualizations. |

Configuration files inherit shared parameters through `base`. In filenames, `p010`, `p025`, `p050`, and `p100` indicate the fraction of labeled **training** images; `seed3407` and similar suffixes identify random seeds. These settings do not change the test split. The four result CSVs contain the values displayed in the manuscript; they are not recalculated from the three released examples.

## Prepare the data

1. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Put your own FMI images and masks under `data/fmi/`. Following `manifests/example.jsonl`, create `manifests/session_v1/train.jsonl`, `val.jsonl`, and `test.jsonl`. `image_rel` and `mask_rel` are paths relative to `data/fmi/`; `sample_id` identifies a sample, and `group_id` and `session_id` record its acquisition grouping. Keep each acquisition session in only one split. Alternatively, set `FMI_DATA_ROOT` to the FMI data directory.

3. Generate the training-label subsets:

   ```bash
   python tools/build_label_efficiency_manifests.py
   ```

   This command writes lists for labeled training, unlabeled training, validation, and test images under `manifests/label_efficiency/`; it does not copy image files. By default, it generates several label fractions and seeds. `configs/apgra_25.yaml` uses `p025_seed3407/`.

4. To train from scratch, prepare CFD, Crack500, CrackTree200, DeepCrack, HRF, and STARE under `data/proxy/<dataset>/<train-or-val>/{images,masks}/`. Match each image and mask by filename.

   | Proxy dataset | Images used for pretraining | Role |
   | --- | --- | --- |
   | CFD | Road-surface cracks | Thin crack morphology |
   | Crack500 | Pavement cracks | Cracks across varying widths and backgrounds |
   | CrackTree200 | Branching pavement cracks | Dendritic crack morphology |
   | DeepCrack | Surface cracks | Crack morphology across scenes and scales |
   | HRF | Retinal vessel images | Branching, connected line structures |
   | STARE | Retinal vessel images | Branching, connected line structures |

   These datasets provide only structural pretraining examples. FMI adaptation and primary test evaluation use FMI data. Proxy datasets are not distributed in this repository; obtain them from their original providers and organize each training and validation split as above. `tools/cache_proxy_data.py` pairs images and masks by filename stem, converts them to grayscale, and resizes them to `320 × 256` for pretraining.

## Run the pipeline

From the repository root, run these commands in order:

```bash
# 1. Resize and cache proxy images and masks in cache/proxy_320x256/.
python tools/cache_proxy_data.py --source data/proxy

# 2. Learn general curvilinear features; save results/proxy_pretrain/structure_all/best.pt.
python pretrain_proxy.py --config configs/pretrain.yaml

# 3. Adapt with 25% of the FMI training labels and the remaining unlabeled training images.
#    Save best.pt and training records in the run directory.
python train_adaptation.py --config configs/apgra_25.yaml

# 4. Evaluate the validation split only and save the threshold sweep and validation metrics.
python evaluate.py --config configs/apgra_25.yaml --validation-only

# 5. Select a threshold on validation data, evaluate the test split, and save
#    metrics.json, test_cases.csv, and predictions/.
python evaluate.py --config configs/apgra_25.yaml
```

Steps 3–5 write to `results/apgra/final/p025_seed3407/` by default. To use all training labels, replace their configuration with `configs/apgra_100.yaml`. Every `evaluate.py` run selects its threshold on the validation split before scoring the test split. It will not overwrite an existing `metrics.json`.

For each comparison model, use the corresponding file under `configs/comparisons/` in steps 3–5. For example:

```bash
bash tools/run_experiment.sh configs/comparisons/unetpp_p025_seed3407.yaml
bash tools/run_experiment.sh configs/comparisons/unetpp_p100_seed3407.yaml
```

The same pattern applies to `unet`, `unetr`, `deeplabv3_resnet50`, `segformer_b0`, and `eilnet`. The 25% validation comparison additionally uses seeds 2025 and 2026. The paper's test comparison uses seed 3407 at both label fractions.

For an ablation, use one file under `configs/adaptation/final_ablations/`. All eight ablations use 25% training labels and the same 70-image test split:

```bash
bash tools/run_experiment.sh configs/adaptation/final_ablations/no_unlabeled.yaml
```

The `crack_only_proxy` ablation first requires `python pretrain_proxy.py --config configs/proxy_pretrain/structure_cracks.yaml`. The `no_proxy` ablation starts without a proxy checkpoint. All other ablations use the full structural proxy checkpoint created in step 2. The three files in `examples/` are visualizations, not a training dataset.

The paper's ablation names map to configuration filenames as follows:

| `ablation.csv` row | Configuration under `configs/adaptation/final_ablations/` |
| --- | --- |
| Without unlabeled target adaptation | `no_unlabeled.yaml` |
| Without structural proxy initialization | `no_proxy.yaml` |
| Crack-only structural proxies | `crack_only_proxy.yaml` |
| Without azimuth-cycle equivariance | `no_azimuth_cycle.yaml` |
| Independent hard depth-bin prototypes | `independent_hard_bins.yaml` |
| Image-global prototypes | `global_prototype.yaml` |
| Without prototype-margin gating | `no_prototype_filter.yaml` |
| Without acquisition-support gating | `no_reliability_filter.yaml` |

The APGRA-Net row uses `configs/apgra_25.yaml`. All comparison and ablation scores in the CSV files are rounded exactly as in the manuscript.
