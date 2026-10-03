LeviDigit collaborator model package
=====================================

Files
-----
levidigit_nnunet_model.zip
    Five-fold 3D nnU-Net model used by the LeviDigit inference pipeline.

levidigit_nnunet_model.zip.sha256
    SHA-256 checksum for verifying the downloaded model archive.

Use
---
1. Download or clone the LeviDigit code repository.
2. Create the environment described in its README.md.
3. Keep this ZIP outside the Git repository.
4. Run from the code repository root:

   python inference\run_inference.py `
     --input-dir "<NIFTI_INPUT_DIR>" `
     --model "<PATH>\levidigit_nnunet_model.zip" `
     --bone-threshold 60

The input directory must contain one 3D NIfTI volume per case. Images must
have 0.004 mm isotropic spacing and 0-255 intensities. The threshold is
inclusive: bone voxels are image values greater than or equal to the selected
threshold inside the predicted digit mask.

Integrity
---------
Compare the model ZIP's SHA-256 hash with the value in the .sha256 file before
use. On Windows PowerShell:

   Get-FileHash -Algorithm SHA256 levidigit_nnunet_model.zip

The model and derived measurements are for research use only.
