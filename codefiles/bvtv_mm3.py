import os
import numpy as np
import nibabel as nib
import pandas as pd

# ============================================================
# Configuration
# ============================================================
images_dir = r"C:\ML\3dunet\TCIA\nnunet\nnUNet_raw\Dataset501_Glioblastoma\imagesTs"
labels_dir = r"C:\ML\3dunet\TCIA\nnunet\nnUNet_results\Dataset501_Glioblastoma\inference"
output_csv = r"C:\ML\3dunet\TCIA\nnunet\nnUNet_raw\Dataset501_Glioblastoma\bvtv_cv_mm3.csv"

BV_THRESHOLD_MIN = 81
BV_THRESHOLD_MAX = 255

EXPECTED_SPACING_MM = np.array([0.004, 0.004, 0.004])
TOL = 1e-6

# ============================================================
# Utilities
# ============================================================
def list_nifti_files(directory):
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".nii") or f.endswith(".nii.gz")
    )

def load_nifti(path):
    nii = nib.load(path)
    return nii, nii.get_fdata()

def spacing_from_affine(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

# ============================================================
# Main processing
# ============================================================
image_files = list_nifti_files(images_dir)
label_files = list_nifti_files(labels_dir)

if len(image_files) != len(label_files):
    raise RuntimeError(
        f"Image/label count mismatch: {len(image_files)} vs {len(label_files)}"
    )

results = []

print(f"\nProcessing {len(image_files)} aligned image/label pairs\n")

for i, (image_name, label_name) in enumerate(zip(image_files, label_files), start=1):

    print("=" * 80)
    print(f"PAIR {i}")
    print(f"Image: {image_name}")
    print(f"Label: {label_name}")

    image_nii, image_data = load_nifti(os.path.join(images_dir, image_name))
    label_nii, label_data = load_nifti(os.path.join(labels_dir, label_name))

    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------
    if image_data.shape != label_data.shape:
        raise RuntimeError(
            f"Shape mismatch: image {image_data.shape}, label {label_data.shape}"
        )

    # --------------------------------------------------------
    # Spacing check
    # --------------------------------------------------------
    image_spacing = spacing_from_affine(image_nii.affine)
    label_spacing = spacing_from_affine(label_nii.affine)

    print(f"Voxel spacing (mm): {image_spacing}")

    if not np.allclose(image_spacing, label_spacing, atol=TOL):
        raise RuntimeError("Image/label spacing mismatch")

    if not np.allclose(image_spacing, EXPECTED_SPACING_MM, atol=TOL):
        raise RuntimeError(
            f"Unexpected spacing {image_spacing} mm "
            f"(expected {EXPECTED_SPACING_MM} mm)"
        )

    # --------------------------------------------------------
    # Voxel geometry
    # --------------------------------------------------------
    voxel_volume_mm3 = np.prod(image_spacing)

    print(f"Voxel volume: {voxel_volume_mm3:.8e} mm³")

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------
    mask = label_data > 0
    tv_voxels = np.count_nonzero(mask)

    if tv_voxels == 0:
        raise RuntimeError("Label mask is empty")

    print(f"Mask voxels: {tv_voxels}")

    # --------------------------------------------------------
    # Total Volume (TV) — mm³
    # --------------------------------------------------------
    tv_mm3 = tv_voxels * voxel_volume_mm3
    print(f"Total volume (TV): {tv_mm3:.6e} mm³")

    # --------------------------------------------------------
    # Blood Volume (BV) — mm³
    # --------------------------------------------------------
    masked_image = image_data[mask]
    bv_voxels = np.count_nonzero(
        (masked_image >= BV_THRESHOLD_MIN) &
        (masked_image <= BV_THRESHOLD_MAX)
    )

    bv_mm3 = bv_voxels * voxel_volume_mm3
    print(f"Blood volume (BV): {bv_mm3:.6e} mm³")

    # --------------------------------------------------------
    # BV / TV
    # --------------------------------------------------------
    bv_tv_percent = (bv_mm3 / tv_mm3) * 100.0

    print(f"BV/TV: {bv_tv_percent:.2f} %")

    results.append([
        image_name,
        label_name,
        bv_mm3,
        tv_mm3,
        bv_tv_percent
    ])

# ============================================================
# Save CSV
# ============================================================
df = pd.DataFrame(
    results,
    columns=[
        "image_nifti",
        "label_nifti",
        "bv_mm3",
        "tv_mm3",
        "bv_tv_percent"
    ]
)

df.to_csv(output_csv, index=False)

print("\nProcessing complete.")
print(f"CSV written to: {output_csv}")
