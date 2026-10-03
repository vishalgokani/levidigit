"""Calculate BV, TV, and BV/TV from images and existing predicted masks."""

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
    args = parser.parse_args()

    mask_dir = args.mask_dir.expanduser().resolve()
    output_csv = mask_dir.parent / "measurements.csv"
    rows = analyze_bvtv(args.input_dir, mask_dir, args.bone_threshold, output_csv)
    print(f"Processed {len(rows)} case(s).")
    print(f"Measurements: {output_csv}")
    print("digit_length_um is reserved for phase two and remains blank.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)

