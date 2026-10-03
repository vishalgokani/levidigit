"""Bone-volume and total-volume measurements."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from .nifti import validate_spacing
from .paths import discover_nifti_images, mask_path_for_case


CSV_COLUMNS = [
    "case_id",
    "image_file",
    "mask_file",
    "bone_threshold",
    "voxel_volume_mm3",
    "tv_voxels",
    "bv_voxels",
    "tv_mm3",
    "bv_mm3",
    "bv_tv_percent",
    "digit_length_um",
]


def validate_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 255.0:
        raise ValueError(f"Bone threshold must be between 0 and 255; got {value}")
    return threshold


def measure_case(case_id: str, image_path: Path, mask_path: Path, bone_threshold: float) -> dict[str, object]:
    threshold = validate_threshold(bone_threshold)
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    validate_spacing(image_nii, image_path)

    if image_nii.shape != mask_nii.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: image {image_nii.shape}, mask {mask_nii.shape}"
        )
    if not np.allclose(image_nii.affine, mask_nii.affine, rtol=1e-5, atol=1e-7):
        raise ValueError(f"Image and mask affines differ for {case_id}")

    image = np.asanyarray(image_nii.dataobj)
    mask_data = np.asanyarray(mask_nii.dataobj)
    if not np.all(np.isfinite(image)):
        raise ValueError(f"Image contains non-finite values: {image_path}")
    image_min = float(np.min(image))
    image_max = float(np.max(image))
    if image_min < 0.0 or image_max > 255.0:
        raise ValueError(
            f"Expected 0-255 image intensities for {case_id}; observed {image_min:g}-{image_max:g}"
        )

    mask_values = np.unique(mask_data)
    binary_values = np.isclose(mask_values, 0.0, atol=1e-5) | np.isclose(
        mask_values, 1.0, atol=1e-5
    )
    if not np.all(binary_values):
        raise ValueError(
            f"Predicted mask for {case_id} must be binary 0/1; found values {mask_values[:10]}"
        )
    # NIfTI slope/intercept scaling can decode a stored value of 1 as a value
    # microscopically above or below 1. A 0.5 decision boundary preserves the
    # intended binary mask without accepting probabilistic segmentations.
    mask = mask_data > 0.5
    tv_voxels = int(np.count_nonzero(mask))
    if tv_voxels == 0:
        raise ValueError(f"Predicted mask is empty for {case_id}")

    bv_voxels = int(np.count_nonzero(mask & (image >= threshold)))
    voxel_volume_mm3 = float(abs(np.linalg.det(image_nii.affine[:3, :3])))
    tv_mm3 = tv_voxels * voxel_volume_mm3
    bv_mm3 = bv_voxels * voxel_volume_mm3
    return {
        "case_id": case_id,
        "image_file": image_path.name,
        "mask_file": mask_path.name,
        "bone_threshold": threshold,
        "voxel_volume_mm3": voxel_volume_mm3,
        "tv_voxels": tv_voxels,
        "bv_voxels": bv_voxels,
        "tv_mm3": tv_mm3,
        "bv_mm3": bv_mm3,
        "bv_tv_percent": (bv_voxels / tv_voxels) * 100.0,
        # Reserved for phase two. The future implementation will measure only
        # in-mask line intervals and will exclude the initial concavity gap and
        # every later out-of-mask interval.
        "digit_length_um": "",
    }


def analyze_bvtv(input_dir: Path, mask_dir: Path, bone_threshold: float, output_csv: Path) -> list[dict[str, object]]:
    cases = discover_nifti_images(input_dir)
    mask_dir = mask_dir.expanduser().resolve()
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    rows = [
        measure_case(case_id, image_path, mask_path_for_case(mask_dir, case_id), bone_threshold)
        for case_id, image_path in sorted(cases.items())
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bone_threshold": float(bone_threshold),
        "threshold_rule": "image intensity >= bone_threshold inside predicted mask",
        "num_cases": len(rows),
        "digit_length_status": "reserved for phase two; not calculated",
    }
    summary_path = output_csv.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return rows
