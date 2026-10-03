# LeviDigit 3D nnU-Net

## Intended use

Research segmentation of animal digit micro-CT volumes for downstream morphometric analysis. The model is intended for data acquired and reconstructed similarly to its training data.

## Model and task

- Architecture: nnU-Net v2 3D full-resolution ensemble
- Task: binary digit segmentation
- Input: one 3D micro-CT channel, NIfTI format, 0.004 mm isotropic spacing, 0-255 intensities
- Output: binary NIfTI mask (`0` background, `1` digit)
- Ensemble: five cross-validation folds using final checkpoints

## Data and preprocessing

The model was trained on 55 manually segmented digit volumes. nnU-Net performs its configured resampling, cropping, Z-score normalization, sliding-window inference, mirroring, and Gaussian blending. The released pipeline validates input spacing and preserves original output geometry.

## Limitations

The model has not been established for different species, anatomy, reconstruction kernels, voxel spacing, intensity scaling, substantial artifacts, or acquisition protocols outside the development data. Segmentation and derived measurements require visual and quantitative quality control. A plausible mask does not guarantee a valid biological measurement.

## Ethical and clinical considerations

This is a research model, not a clinical device. It must not be used for diagnosis or treatment decisions. Users are responsible for data governance and for confirming that shared inputs contain no identifying information.

## Software

The reference release uses nnU-Net v2.5.1. Install a hardware-compatible PyTorch build separately, followed by the repository requirements.

## Version

Initial collaborator release.
