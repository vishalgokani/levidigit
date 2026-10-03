# Model release

Model weights are not stored in Git or hosted publicly. Contact the authors for access to the collaborator package.

To create a shareable ZIP from a completed five-fold training folder:

```powershell
python tools\export_nnunet_model.py `
  --model-dir "<NNUNET_RESULTS>\Dataset501_LeviDigit\nnUNetTrainer__nnUNetPlans__3d_fullres" `
  --output-zip "<DESTINATION>\levidigit_nnunet_model.zip"
```

The tool includes only inference-required metadata and final fold checkpoints, removes legacy public-facing dataset terminology, and writes a SHA-256 checksum beside the ZIP. Upload both files to the controlled OneDrive location.
