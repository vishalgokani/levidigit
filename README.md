# LeviDigit

LeviDigit segments animal digit micro-CT volumes with a five-fold 3D nnU-Net model and reports bone volume (BV), total segmented volume (TV), BV/TV, and digit length. Inputs are 8-bit NIfTI volumes with 0.004 mm isotropic spacing. Postprocessing also exports quantitative masks, audit landmarks, and publication-oriented 2D/3D figures.

## Install

Create an environment and install the required packages:

```powershell
conda create -n levidigit python=3.12 -y
conda activate levidigit
pip install -r requirements.txt
# Then install the CUDA-enabled PyTorch build appropriate for your NVIDIA GPU and driver.
```

The model weights are distributed separately through the project OneDrive. Download `levidigit_nnunet_model.zip`, keep it outside this Git repository, and pass its local path to `--model`. This model is not distributed through Hugging Face.

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
  --scratch-dir "<LOCAL_SCRATCH_DIR>" `
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
figures/<case>_publication.{png,tiff}
measurements.csv
measurements.json
label_definitions.json
run_manifest.json
```

`measurements.csv` contains one row per sample. Volumes are reported in voxels and mm³, BV/TV as percent, digit length in mm and µm, and landmark coordinates in voxel and NIfTI world coordinates. A failed concavity/length calculation remains a clearly marked row rather than being silently reported as zero.

The combined labelmaps are visualization products, not quantitative masks. `label_definitions.json` records their label values and colors. The independent binary masks under `derived_masks/` are authoritative for analysis. The original binary nnU-Net ROI remains unchanged under `masks/`. When `--scratch-dir` is supplied, inference and postprocessing run in a unique local workspace, the completed timestamped output is copied back beside the input directory, and that run's scratch workspace is removed after a successful export.

## Re-run postprocessing only

Changing the threshold does not require another segmentation run:

```powershell
python postprocessing\analyze_bvtv.py `
  --input-dir "<NIFTI_INPUT_DIR>" `
  --mask-dir "<LEVIDIGIT_OUTPUT_DIR>\masks" `
  --bone-threshold 60
```

See [training/README.md](training/README.md) for model training and [model_release/README.md](model_release/README.md) for preparing collaborator model packages.
