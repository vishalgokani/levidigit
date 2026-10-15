# LeviDigit

LeviDigit segments animal digit micro-CT volumes with a five-fold 3D nnU-Net model and reports bone volume (BV), total segmented volume (TV), BV/TV, and digit length. Inputs are 8-bit NIfTI volumes with 0.004 mm isotropic spacing. Postprocessing also exports quantitative masks, audit landmarks, and publication-oriented 2D/3D figures.

## Install

Create an environment and install a PyTorch build compatible with the workstation's CUDA version. Then install the remaining packages:

```powershell
conda create -n levidigit python=3.12 -y
conda activate levidigit
# Install the appropriate PyTorch build first.
pip install -r requirements.txt
```

The model weights are distributed separately. Contact the authors for access, download `levidigit_nnunet_model.zip`, and keep it outside this Git repository.

## Prepare BMP stacks

If the scanner output is a folder of BMP slices, convert it before inference. The input may be one BMP stack or a folder containing one BMP-stack subfolder per case.

```powershell
python tools\bmp_stack_to_nifti.py `
  --input-dir "<BMP_INPUT_DIR>" `
  --output-dir "<NIFTI_INPUT_DIR>" `
  --spacing-mm 0.004
```

Each case is written as `<case>_0000.nii.gz`. Slice dimensions must agree; the converter never pads or resizes data.

## Run inference and measurements

Place one NIfTI volume per case directly in a local input folder and choose the inclusive 0-255 bone threshold:

```powershell
python inference\run_inference.py `
  --input-dir "<NIFTI_INPUT_DIR>" `
  --model "<PATH_TO_MODEL_ZIP_OR_EXTRACTED_FOLDER>" `
  --bone-threshold 60
```

The pipeline automatically uses CUDA when available and otherwise uses CPU. It creates a new timestamped `levidigit_output_*` folder inside the input folder containing:

```text
masks/<case>.nii.gz
derived_masks/bv/<case>.nii.gz
derived_masks/concavity/<case>.nii.gz
derived_masks/digit_length_counted/<case>.nii.gz
visualization_labelmaps/standard/<case>.nii.gz
visualization_labelmaps/audit/<case>.nii.gz
landmarks/<case>.json
figures/2d/<case>_2d_{standard,audit}.{png,tiff}
figures/3d/<case>_3d_{standard,audit}.{png,tiff}
measurements.csv
measurements.json
label_definitions.json
run_manifest.json
```

`measurements.csv` contains one row per sample. Volumes are reported in voxels and mm³, BV/TV as percent, digit length in mm and µm, and landmark coordinates in voxel and NIfTI world coordinates. A failed concavity/length calculation remains a clearly marked row rather than being silently reported as zero.

The combined labelmaps are visualization products, not quantitative masks. `label_definitions.json` records their label values and colors. The independent binary masks under `derived_masks/` are authoritative for analysis. The original binary nnU-Net ROI remains unchanged under `masks/`.

## Re-run postprocessing only

Changing the threshold does not require another segmentation run:

```powershell
python postprocessing\analyze_bvtv.py `
  --input-dir "<NIFTI_INPUT_DIR>" `
  --mask-dir "<LEVIDIGIT_OUTPUT_DIR>\masks" `
  --bone-threshold 60
```

See [training/README.md](training/README.md) for model training and [model_release/README.md](model_release/README.md) for preparing collaborator model packages.
