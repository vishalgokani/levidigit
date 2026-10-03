"""Convert one or more 8-bit BMP slice stacks to 3D NIfTI volumes."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def discover_stacks(input_dir: Path) -> list[tuple[str, Path, list[Path]]]:
    direct = sorted(input_dir.glob("*.bmp"), key=natural_key)
    if direct:
        return [(input_dir.name, input_dir, direct)]
    stacks = []
    for folder in sorted((path for path in input_dir.iterdir() if path.is_dir()), key=natural_key):
        files = sorted(folder.glob("*.bmp"), key=natural_key)
        if files:
            stacks.append((folder.name, folder, files))
    if not stacks:
        raise FileNotFoundError(f"No BMP files or BMP-containing case folders found in {input_dir}")
    return stacks


def read_slice(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D BMP slice: {path}")
    return array


def convert_stack(files: list[Path], output_path: Path, spacing_mm: float) -> tuple[int, tuple[int, int, int]]:
    first = read_slice(files[0])
    volume = np.empty((first.shape[0], first.shape[1], len(files)), dtype=np.uint8)
    volume[:, :, 0] = first
    for index, path in enumerate(files[1:], start=1):
        current = read_slice(path)
        if current.shape != first.shape:
            raise ValueError(
                f"Inconsistent BMP dimensions in {path.parent}: {path.name} is {current.shape}, expected {first.shape}"
            )
        volume[:, :, index] = current

    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    image = nib.Nifti1Image(volume, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, str(output_path))
    return len(files), volume.shape


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="One BMP stack or a folder of per-case BMP stacks.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New or empty folder for NIfTI volumes.")
    parser.add_argument("--spacing-mm", required=True, type=float, help="Isotropic voxel spacing in millimeters, e.g. 0.004.")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)
    if args.spacing_mm <= 0:
        raise ValueError("--spacing-mm must be greater than zero")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id, source_dir, files in discover_stacks(input_dir):
        output = output_dir / f"{case_id}_0000.nii.gz"
        count, shape = convert_stack(files, output, float(args.spacing_mm))
        rows.append({
            "case_id": case_id,
            "source_dir": str(source_dir),
            "first_slice": files[0].name,
            "last_slice": files[-1].name,
            "num_slices": count,
            "shape": "x".join(str(value) for value in shape),
            "spacing_mm": float(args.spacing_mm),
            "output_file": output.name,
        })
        print(f"{case_id}: {count} slices -> {output.name} {shape}")

    with (output_dir / "conversion_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

