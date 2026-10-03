"""Create a compact, collaborator-ready LeviDigit model ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_ROOT_FILES = ("plans.json", "dataset.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_dataset_dict(data: dict[str, object]) -> dict[str, object]:
    data = dict(data)
    data.update({
        "channel_names": {"0": "microCT"},
        "labels": {"background": 0, "digit": 1},
        "name": "LeviDigit",
        "description": "Binary segmentation of animal digit micro-CT volumes.",
        # Preserve the original model's region-label structure while replacing
        # its legacy project terminology.
        "region_class_order": {"background": 0, "digit": 1},
    })
    return data


def sanitized_dataset_json(source: Path) -> dict[str, object]:
    return sanitize_dataset_dict(json.loads(source.read_text(encoding="utf-8")))


def sanitized_plans_json(source: Path) -> dict[str, object]:
    data = json.loads(source.read_text(encoding="utf-8"))
    data["dataset_name"] = "LeviDigit"
    return data


def sanitize_checkpoint(source: Path, destination: Path) -> None:
    """Remove obsolete project names from checkpoint metadata, not weights."""
    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    init_args = checkpoint.get("init_args")
    if not isinstance(init_args, dict):
        raise ValueError(f"Checkpoint has no nnU-Net init_args: {source}")
    plans = init_args.get("plans")
    dataset_json = init_args.get("dataset_json")
    if not isinstance(plans, dict) or not isinstance(dataset_json, dict):
        raise ValueError(f"Checkpoint is missing plans or dataset_json metadata: {source}")
    plans["dataset_name"] = "LeviDigit"
    init_args["dataset_json"] = sanitize_dataset_dict(dataset_json)
    torch.save(checkpoint, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path, help="Trained nnUNetTrainer__...__3d_fullres folder.")
    parser.add_argument("--output-zip", required=True, type=Path, help="Destination levidigit_nnunet_model.zip.")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    output_zip = args.output_zip.expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite existing model package: {output_zip}")
    for filename in REQUIRED_ROOT_FILES:
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(model_dir / filename)
    folds = sorted(path for path in model_dir.glob("fold_*") if path.is_dir())
    if not folds:
        raise FileNotFoundError(f"No fold directories found in {model_dir}")
    for fold in folds:
        if not (fold / "checkpoint_final.pth").is_file():
            raise FileNotFoundError(fold / "checkpoint_final.pth")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    package_root = "levidigit_nnunet_model/model"
    manifest = {
        "model_name": "LeviDigit 3D nnU-Net",
        "task": "binary animal digit micro-CT segmentation",
        "configuration": "3d_fullres",
        "checkpoint": "checkpoint_final.pth",
        "folds": [fold.name.removeprefix("fold_") for fold in folds],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.TemporaryDirectory(prefix="levidigit_export_") as temp:
        temp_dir = Path(temp)
        dataset_json = temp_dir / "dataset.json"
        plans_json = temp_dir / "plans.json"
        dataset_json.write_text(json.dumps(sanitized_dataset_json(model_dir / "dataset.json"), indent=2) + "\n", encoding="utf-8")
        plans_json.write_text(json.dumps(sanitized_plans_json(model_dir / "plans.json"), indent=2) + "\n", encoding="utf-8")
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
            archive.write(dataset_json, f"{package_root}/dataset.json")
            archive.write(plans_json, f"{package_root}/plans.json")
            archive.writestr("levidigit_nnunet_model/manifest.json", json.dumps(manifest, indent=2) + "\n")
            for fold in folds:
                checkpoint = fold / "checkpoint_final.pth"
                sanitized_checkpoint = temp_dir / f"{fold.name}_checkpoint_final.pth"
                sanitize_checkpoint(checkpoint, sanitized_checkpoint)
                archive.write(
                    sanitized_checkpoint,
                    f"{package_root}/{fold.name}/checkpoint_final.pth",
                )
                sanitized_checkpoint.unlink()

    checksum = sha256(output_zip)
    output_zip.with_suffix(output_zip.suffix + ".sha256").write_text(
        f"{checksum}  {output_zip.name}\n", encoding="utf-8"
    )
    print(f"Created: {output_zip}")
    print(f"SHA-256: {checksum}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
