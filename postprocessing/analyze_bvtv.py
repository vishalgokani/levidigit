"""Calculate BV/TV and digit length from images and predicted masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from levidigit.bvtv import analyze_bvtv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing original NIfTI images.")
    parser.add_argument("--mask-dir", required=True, type=Path, help="Folder containing predicted binary masks.")
    parser.add_argument("--bone-threshold", required=True, type=float, help="Inclusive 0-255 bone threshold.")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output folder (default: parent of --mask-dir).",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Write measurements and NIfTI artifacts without PNG/TIFF figures.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=300,
        help="PNG/TIFF resolution (default: 300).",
    )
    args = parser.parse_args()

    mask_dir = args.mask_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else mask_dir.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "measurements.csv"
    rows = analyze_bvtv(
        args.input_dir, mask_dir, args.bone_threshold, output_csv,
        make_figures=not args.no_figures, figure_dpi=args.figure_dpi,
    )
    print(f"Processed {len(rows)} case(s).")
    print(f"Measurements: {output_csv}")
    complete = sum(row["digit_length_status"] == "complete" for row in rows)
    print(f"Digit lengths complete: {complete}/{len(rows)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

