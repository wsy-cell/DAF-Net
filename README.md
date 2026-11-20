# DAF-Net: Aligning Enhancement with Degradation-Dependent Needs for Underwater Image Enhancement

**Official project repository for DAF-Net (Degradation-Adaptive Dual-Branch Fusion Network).**

This repository provides the official source code and final model checkpoints for DAF-Net, including training, inference, and evaluation workflows for UIEB, EUVP, and LSUI.

## Introduction

Underwater images often exhibit color casts, reduced contrast, and blurred details due to wavelength-dependent absorption and scattering. The relative enhancement needs differ across images. DAF-Net addresses this variation through input-adaptive collaboration between two complementary branches.

The framework includes:

- **Three-view shared feature extraction:** the original image, a contrast-limited adaptive histogram equalization (CLAHE) view, and a Gray-World view are concatenated to extract shared shallow features.
- **Complementary enhancement branches:** a **Scattering Perception Branch (SPB)** and a **Color Correction Branch (CCB)** learn scattering-related and color-correction features, respectively, with physics-inspired auxiliary supervision during training.
- **Input-adaptive branch collaboration:** an **Adaptive Synergy Controller (ASC)** estimates image-level branch weights using global shared features and RGB statistics of the original image. **Synergistic Feature Modulation (SFM)** aggregates the shared features with the weighted branch features and refines their fusion before reconstruction.
- **Differentiated low-bit weight configuration:** different functional components use different weight bit widths to reduce parameter storage.

**CCB (Color Correction Branch) in the paper and SCB in the code refer to the same branch, with no structural or functional difference.** Original SCB identifiers are retained for checkpoint compatibility.

## Installation

Use Python 3.10 in a clean environment. Install the recommended PyTorch/torchvision CUDA 11.8 pair separately, then install the project dependencies:

```bash
python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt torch==2.5.1+cu118 torchvision==0.20.1+cu118
python -m pip check
```

The second command keeps the installed PyTorch pair fixed during dependency resolution. Other package versions follow the locally validated evaluation baseline; a fresh Linux CUDA installation of this combination remains to be verified. LPIPS and NIQE may download metric assets on first use.

Users may adjust dependency versions according to their computing environment, provided that the versions remain compatible.

## Dataset Preparation

Datasets are not included. Set the input and ground-truth directories in `configs/uieb.yaml`, `configs/euvp.yaml`, or `configs/lsui.yaml`:

- `training`: training/validation paths, degradation-label CSV, and training settings.
- `evaluation`: test input/GT paths, image output directory, result file, and efficiency settings.
- `checkpoint`: the corresponding final model checkpoint.

Relative YAML paths resolve against the project root. Absolute paths are also supported. Input and GT files are paired in independently sorted filename order; ensure matching counts and correspondence. Training resizes images to 256 × 256. Testing preserves the supplied image resolution without additional resizing or cropping; use the original prepared test split.

The original degradation CSV and normalization JSON files are included in `degradation_labels/`. To generate labels and a distribution plot for new training data:

```bash
python scripts/stage01.py --config configs/uieb.yaml
```

Set `stage01.input_dir` and `stage01.output_dir` as needed. Generated labels default to `outputs/<dataset>/degradation_labels`; update `training.csv_path` to use them. Bundled labels remain unchanged.

## Model Checkpoints

These are the final best checkpoints from independent DAF-Net training runs on UIEB, EUVP, and LSUI. Each is used for inference and quantitative evaluation on its corresponding dataset; they are not external pretrained initialization weights.

| Dataset | Checkpoint |
| --- | --- |
| UIEB | `checkpoints/uieb/best_model_ema.pth` |
| EUVP | `checkpoints/euvp/best_model_ema.pth` |
| LSUI | `checkpoints/lsui/best_model_ema.pth` |

The original checkpoint files and parameter names are preserved. Loading extracts `model_state` and uses `strict=True`. LSUI also contains EMA and configuration metadata; inference uses its saved `model_state` directly.

## Training

```bash
python scripts/train.py --config configs/uieb.yaml
python scripts/train.py --config configs/euvp.yaml
python scripts/train.py --config configs/lsui.yaml
```

The YAML files retain the original hyperparameters. UIEB and EUVP train from scratch. LSUI additionally supports restoring the model, optimizer, EMA, epoch, and RNG state through `auto_resume` and `resume_path`; interruption recovery is epoch-level. The retained `pretrained_weight_path` field is inactive and does not initialize training from the distributed checkpoints. Training outputs are saved under `outputs/`.

## Inference

```bash
python scripts/inference.py --config configs/uieb.yaml --input path/to/image.png --output outputs/single
python scripts/inference.py --config configs/euvp.yaml --input path/to/images --output outputs/batch
```

`--input` accepts one image or a directory. Choose the matching dataset config; `--output` selects the destination. Both modes use the original three-view preprocessing and save RGB predictions at the input resolution. Output filenames append `.png` to the full input filename.

## Evaluation

Test the complete paired dataset, compute quality metrics, and save enhanced images:

```bash
python scripts/test.py --config configs/uieb.yaml
python scripts/test.py --config configs/euvp.yaml
python scripts/test.py --config configs/lsui.yaml
```

Supported quality metrics are **PSNR, SSIM, LPIPS, UIQM, and NIQE**, using the original dataset-specific implementations. Metrics are computed from model outputs before image-file quantization. Aggregate results are written as JSON to `evaluation.SAVE_PATH`; enhanced images are saved to `evaluation.image_dir`.

Run the original quality and efficiency evaluation with text result export:

```bash
python scripts/evaluate.py --config configs/uieb.yaml
python scripts/evaluate.py --config configs/euvp.yaml
python scripts/evaluate.py --config configs/lsui.yaml
```

Efficiency evaluation retains parameter count, THOP profiling, and FPS measurement. UIEB also includes its original batch-size sweep, latency, and peak-memory reporting. EUVP/LSUI efficiency measurement requires CUDA. Use `test.py` for quality-only evaluation on CPU. The two entry points share the original quality metric implementations.

## Citation

Citation details will be updated when the paper's publication information is available.
