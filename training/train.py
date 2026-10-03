"""Train the five-fold LeviDigit nnU-Net v2 3D full-resolution model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def executable(name: str) -> str:
    scripts = Path(sys.executable).resolve().parent / "Scripts" / f"{name}.exe"
    return str(scripts) if scripts.is_file() else (shutil.which(name) or name)


def run(command: list[str], env: dict[str, str]) -> None:
    resolved = [executable(command[0]), *command[1:]]
    print("\n" + " ".join(f'"{part}"' if " " in part else part for part in resolved), flush=True)
    subprocess.run(resolved, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nnunet-root", required=True, type=Path,
        help="Working directory containing nnUNet_raw and receiving preprocessing/results.",
    )
    parser.add_argument("--dataset-id", type=int, default=501, help="nnU-Net dataset number (default: 501).")
    args = parser.parse_args()

    root = args.nnunet_root.expanduser().resolve()
    if root.anchor == str(root):
        raise ValueError(f"Refusing to use a filesystem root as nnunet-root: {root}")
    raw = root / "nnUNet_raw"
    preprocessed = root / "nnUNet_preprocessed"
    results = root / "nnUNet_results"
    for path in (raw, preprocessed, results):
        path.mkdir(parents=True, exist_ok=True)

    expected_dataset = raw / f"Dataset{args.dataset_id:03d}_LeviDigit"
    if not expected_dataset.is_dir():
        raise FileNotFoundError(
            f"Expected training dataset at {expected_dataset}. See training/README.md for its layout."
        )
    dataset_json = expected_dataset / "dataset.json"
    if not dataset_json.is_file():
        raise FileNotFoundError(dataset_json)

    env = os.environ.copy()
    env.update({
        "nnUNet_raw": str(raw),
        "nnUNet_preprocessed": str(preprocessed),
        "nnUNet_results": str(results),
    })
    run([
        "nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id),
        "--verify_dataset_integrity", "-c", "3d_fullres",
    ], env)
    for fold in range(5):
        run([
            "nnUNetv2_train", str(args.dataset_id), "3d_fullres", str(fold),
            "-tr", "nnUNetTrainer", "-p", "nnUNetPlans",
        ], env)
    run([
        "nnUNetv2_find_best_configuration", str(args.dataset_id),
        "-c", "3d_fullres", "-f", "0", "1", "2", "3", "4",
    ], env)

    try:
        import torch
        torch_info = {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_info = {"torch_version": None}
    manifest = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": args.dataset_id,
        "dataset_folder": expected_dataset.name,
        "configuration": "3d_fullres",
        "folds": [0, 1, 2, 3, 4],
        "nnunetv2_version": importlib.metadata.version("nnunetv2"),
        "python_version": sys.version,
        **torch_info,
    }
    manifest_path = results / expected_dataset.name / "training_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Training complete. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

