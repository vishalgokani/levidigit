# BV/TV postprocessing

BV/TV can be recalculated from existing masks without rerunning the neural network:

```powershell
python postprocessing\analyze_bvtv.py `
  --input-dir "<NIFTI_INPUT_DIR>" `
  --mask-dir "<LEVIDIGIT_OUTPUT_DIR>\masks" `
  --bone-threshold 60
```

TV is every foreground voxel in the predicted digit mask. BV is every foreground-mask voxel whose original image intensity is greater than or equal to the supplied threshold. Physical volume is derived from each NIfTI affine. The command writes `measurements.csv` and `measurements.json` beside the masks folder.

The `digit_length_um` column is intentionally blank. Phase two will detect the largest concavity, use its physical-space centroid, connect it to the most distant bone point, and sum only line intervals inside the bone mask. The initial concavity gap and every later out-of-mask interval will be excluded.
