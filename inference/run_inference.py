"""Run LeviDigit 3D segmentation and BV/TV analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from levidigit.bvtv import analyze_bvtv, validate_threshold
from levidigit.inference import run_prediction
from levidigit.model import prepare_model
from levidigit.nifti import stage_single_channel_inputs
from levidigit.paths import discover_nifti_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment digit micro-CT NIfTI volumes and calculate BV/TV."
    )
    parser.add_argument(
        "--input-dir", required=True, type=Path,
        help="Local folder containing one 3D NIfTI image per case.",
    )
    parser.add_argument(
        "--model", required=True, type=Path,
        help="Downloaded LeviDigit model .zip or extracted model folder.",
    )
    parser.add_argument(
        "--bone-threshold", required=True, type=float,
        help="Inclusive 0-255 image-intensity threshold used to define bone volume.",
    )
    return parser.parse_args()


def new_output_dir(input_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = input_dir / f"levidigit_output_{timestamp}"
    output = base
    sequence = 1
    while output.exists():
        output = input_dir / f"{base.name}_{sequence:02d}"
        sequence += 1
    output.mkdir()
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    threshold = validate_threshold(args.bone_threshold)
    input_dir = args.input_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    cases = discover_nifti_images(input_dir)
    output_dir = new_output_dir(input_dir)
    staged_input = output_dir / "_nnunet_input"
    model_extract = output_dir / "_model_extract"
    masks_dir = output_dir / "masks"

    print(f"Cases: {len(cases)}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    status = "failed"
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_dir": str(input_dir),
        "model_source": str(model_path),
        "bone_threshold": threshold,
        "threshold_rule": "image intensity >= bone_threshold inside predicted mask",
        "digit_length_status": "reserved for phase two; not calculated",
        "cases": sorted(cases),
    }
    if model_path.is_file():
        manifest["model_sha256"] = sha256(model_path)

    try:
        input_manifest = stage_single_channel_inputs(cases, staged_input)
        manifest["inputs"] = input_manifest
        model_folder, _ = prepare_model(model_path, model_extract)
        manifest["model_folder"] = model_folder.name
        folds = run_prediction(model_folder, staged_input, masks_dir)
        manifest["folds"] = folds

        expected = {f"{case_id}.nii.gz" for case_id in cases}
        observed = {path.name for path in masks_dir.glob("*.nii.gz")}
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise RuntimeError(f"Prediction set mismatch; missing={missing}, extra={extra}")

        rows = analyze_bvtv(
            input_dir=input_dir,
            mask_dir=masks_dir,
            bone_threshold=threshold,
            output_csv=output_dir / "measurements.csv",
        )
        manifest["num_predictions"] = len(observed)
        manifest["num_measurements"] = len(rows)
        manifest["status"] = "complete"
        status = "complete"
        print(f"Measurements: {output_dir / 'measurements.csv'}", flush=True)
        print("Digit length is reserved for phase two and is blank in the CSV.", flush=True)
        return 0
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = status
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        for temporary in (staged_input, model_extract):
            if temporary.exists():
                shutil.rmtree(temporary)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

