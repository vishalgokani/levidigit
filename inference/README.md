# Inference

Run from the repository root:

```powershell
python inference\run_inference.py `
  --input-dir "<NIFTI_INPUT_DIR>" `
  --model "<MODEL_ZIP_OR_EXTRACTED_MODEL_FOLDER>" `
  --bone-threshold 60
```

The input folder must contain one 3D `.nii` or `.nii.gz` volume per case. Files may use either `<case>.nii.gz` or nnU-Net's `<case>_0000.nii.gz` naming. Images must use 0.004 mm isotropic spacing and 0-255 intensities.

The script uses all folds in the model package. It does not overwrite earlier runs; every run receives a new `levidigit_output_<UTC timestamp>` folder under the input directory. Postprocessing writes the original binary ROI, separate binary BV/concavity/counted-length masks, standard and audit visualization labelmaps, one aggregate CSV, and 300-dpi PNG/TIFF figures. Use `--no-figures` for quantitative outputs only or `--figure-dpi` to change raster resolution.
