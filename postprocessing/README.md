# Postprocessing

Recalculate BV/TV, digit length, masks, landmarks, and figures from existing nnU-Net masks without rerunning inference:

```powershell
python postprocessing\analyze_bvtv.py `
  --input-dir "<NIFTI_INPUT_DIR>" `
  --mask-dir "<LEVIDIGIT_OUTPUT_DIR>\masks" `
  --output-dir "<POSTPROCESSING_OUTPUT_DIR>" `
  --bone-threshold 60
```

TV is every foreground voxel in the binary nnU-Net ROI. BV is every TV voxel whose source-image intensity is greater than or equal to the supplied threshold. Volumes use the NIfTI affine determinant and are reported in mm³.

Digit length follows the largest-concavity approach adapted from the `perios2` v5 pipeline: slice-wise hull, 95th-percentile shrink wrap, 3D convex-hull difference, and matched erosion/dilation. The concavity centroid is connected to the most distant ROI voxel. The straight path is densely sampled, and only subsegments whose midpoints lie inside the ROI contribute to length. The originating concavity and any later outside-ROI gaps therefore contribute zero. Length is reported in mm and µm.

Outputs are written to `--output-dir`, or beside the supplied `masks` directory when that option is omitted. `measurements.csv` is one aggregate table for all samples. The original mask and independent binary masks are the quantitative artifacts; standard/audit combined labelmaps are visualization-only. Each sample receives one combined three-row, four-column figure in PNG and LZW-compressed TIFF. Columns show original bone, TV segmentation, digit length, and BV/TV. Rows show the consistent 3D view, the clockwise-rotated longitudinal measurement plane, and the cross section through the concavity centroid. The digit points right in both 2D rows. The 3D camera aligns the projected digit direction in native XY and tilts 30° away from face-on, producing consistent automatic views; the original-bone surface uses directional shading to preserve depth and contrast.

Use `--no-figures` to skip raster rendering or `--figure-dpi <DPI>` to change its resolution.
