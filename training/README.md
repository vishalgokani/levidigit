# Training

Training expects a standard nnU-Net v2 dataset:

```text
<NNUNET_ROOT>/
└── nnUNet_raw/
    └── Dataset501_LeviDigit/
        ├── imagesTr/<case>_0000.nii.gz
        ├── labelsTr/<case>.nii.gz
        └── dataset.json
```

The dataset JSON should describe channel 0 as `microCT` and labels as `background: 0` and `digit: 1`. Images and labels must be 3D NIfTI files with matching geometry. Subject-level train/validation separation must be preserved when multiple samples originate from one animal.

Run five-fold 3D full-resolution training:

```powershell
python training\train.py --nnunet-root "<NNUNET_ROOT>" --dataset-id 501
```

The wrapper verifies and preprocesses the dataset, trains folds 0-4 with `nnUNetTrainer`, selects the best configuration from validation data, and writes a training manifest under `nnUNet_results/Dataset501_LeviDigit/`.

Do not use the held-out test set for model selection. Keep raw data, preprocessed arrays, and checkpoints outside Git.
