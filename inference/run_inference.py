"""Run LeviDigit segmentation, measurements, and publication visualizations."""

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
        description="Segment digit micro-CT volumes and calculate BV/TV and digit length."
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
        "--scratch-dir", type=Path,
        help=(
            "Optional local workspace. Inputs and the model are staged here, results are "
            "copied back beside the input directory, and this run's scratch folder is removed."
        ),
    )
    parser.add_argument(
        "--bone-threshold", required=True, type=float,
        help="Inclusive 0-255 image-intensity threshold used to define bone volume.",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Write measurements and NIfTI artifacts without PNG/TIFF figures.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=300,
        help="PNG/TIFF resolution (default: 300).",
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


def new_work_dir(scratch_dir: Path, run_name: str) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    work_dir = scratch_dir / f"{run_name}_work"
    sequence = 1
    while work_dir.exists():
        work_dir = scratch_dir / f"{run_name}_work_{sequence:02d}"
        sequence += 1
    work_dir.mkdir()
    return work_dir


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
    scratch_dir = args.scratch_dir.expanduser().resolve() if args.scratch_dir else None
    work_root = new_work_dir(scratch_dir, output_dir.name) if scratch_dir else output_dir
    work_output = work_root / "output" if scratch_dir else output_dir
    work_output.mkdir(exist_ok=True)
    staged_input = work_root / "_nnunet_input"
    model_extract = work_root / "_model_extract"
    masks_dir = work_output / "masks"

    print(f"Cases: {len(cases)}", flush=True)
    print(f"Output: {output_dir}", flush=True)
    if scratch_dir:
        print(f"Scratch workspace: {work_root}", flush=True)
    status = "failed"
    manifest: dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "input_dir": str(input_dir),
        "scratch_dir": str(scratch_dir) if scratch_dir else None,
        "model_source": str(model_path),
        "bone_threshold": threshold,
        "threshold_rule": "image intensity >= bone_threshold inside predicted mask",
        "digit_length_method": "largest concavity centroid to most distant ROI point; count only in-ROI line intervals",
        "figures_requested": not args.no_figures,
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
            input_dir=staged_input,
            mask_dir=masks_dir,
            bone_threshold=threshold,
            output_csv=work_output / "measurements.csv",
            make_figures=not args.no_figures,
            figure_dpi=args.figure_dpi,
        )
        manifest["num_predictions"] = len(observed)
        manifest["num_measurements"] = len(rows)
        manifest["status"] = "complete"
        status = "complete"
        print(f"Measurements: {output_dir / 'measurements.csv'}", flush=True)
        complete = sum(row["digit_length_status"] == "complete" for row in rows)
        print(f"Digit lengths complete: {complete}/{len(rows)}", flush=True)
        return 0
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = status
        (work_output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if scratch_dir:
            exported = False
            try:
                shutil.copytree(work_output, output_dir, dirs_exist_ok=True)
                exported = True
            finally:
                if exported:
                    shutil.rmtree(work_root)
        else:
            for temporary in (staged_input, model_extract):
                if temporary.exists():
                    shutil.rmtree(temporary)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

