# DAF-Net: Aligning Enhancement with Degradation-Dependent Needs for Underwater Image Enhancement

**Official project repository for DAF-Net (Degradation-Adaptive Dual-Branch Fusion Network).**

This repository provides project information and will host the pretrained model weights for **DAF-Net**, a network for underwater image enhancement that adjusts the relative contributions of scattering-related enhancement and color correction according to the input image.

## Overview

Underwater images often exhibit color casts, reduced contrast, and blurred details due to wavelength-dependent absorption and scattering. The relative enhancement needs differ across images. DAF-Net addresses this variation through input-adaptive collaboration between two complementary branches.

The framework includes:

- **Three-view shared feature extraction:** the original image, a contrast-limited adaptive histogram equalization (CLAHE) view, and a Gray-World view are concatenated to extract shared shallow features.
- **Complementary enhancement branches:** a **Scattering Perception Branch (SPB)** and a **Color Correction Branch (CCB)** learn scattering-related and color-correction features, respectively, with physics-inspired auxiliary supervision during training.
- **Input-adaptive branch collaboration:** an **Adaptive Synergy Controller (ASC)** estimates image-level branch weights using global shared features and RGB statistics of the original image. **Synergistic Feature Modulation (SFM)** aggregates the shared features with the weighted branch features and refines their fusion before reconstruction.
- **Differentiated low-bit weight configuration:** different functional components use different weight bit widths to reduce parameter storage.

## Quantitative Results

DAF-Net achieves the following PSNR results on three paired underwater image enhancement benchmarks:

| Dataset | PSNR (dB) |
| :------ | --------: |
| UIEB    | **23.96** |
| LSUI    | **25.89** |
| EUVP    | **27.90** |

On LSUI and EUVP, DAF-Net also achieves the best SSIM and LPIPS results among the evaluated comparison methods. Compared with the full-precision configuration, **parameter storage is reduced by 49.34%**.

## Pretrained Weights

**Status:** The pretrained checkpoint has not yet been uploaded to this repository.

The model weights will be listed in this section once the file has been uploaded. This section will then be updated with the exact checkpoint filename and download link.

## Code Availability

The repository currently serves as a project page for the paper and its pretrained weights. **Source code is not included in the current release.**

## Citation

Citation details will be updated when the paper's publication information is available.
