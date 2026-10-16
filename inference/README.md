# Inference

Run from the repository root:

```powershell
python inference\run_inference.py `
  --input-dir "<NIFTI_INPUT_DIR>" `
  --model "<MODEL_ZIP_OR_EXTRACTED_MODEL_FOLDER>" `
  --scratch-dir "<LOCAL_SCRATCH_DIR>" `
  --bone-threshold 60
```

The input folder must contain one 3D `.nii` or `.nii.gz` volume per case. Files may use either `<case>.nii.gz` or nnU-Net's `<case>_0000.nii.gz` naming. Images must use 0.004 mm isotropic spacing and 0-255 intensities.

The script uses all folds in the model package. It does not overwrite earlier runs; every run receives a new `levidigit_output_<UTC timestamp>` folder under the input directory. With `--scratch-dir`, the NIfTI inputs and model are staged in a unique local workspace, inference is followed automatically by postprocessing there, results are copied back to the timestamped output folder, and the run workspace is removed after successful export. Source data are never deleted.

Postprocessing writes the original binary ROI, separate binary BV/concavity/counted-length masks, standard and audit visualization labelmaps, one aggregate CSV, and one 300-dpi PNG/TIFF publication figure per sample. Its four columns are original bone, TV segmentation, digit length, and BV/TV. Rows show the consistent 3D view, the clockwise-rotated longitudinal measurement plane, and the cross section. Use `--no-figures` for quantitative outputs only or `--figure-dpi` to change raster resolution.
