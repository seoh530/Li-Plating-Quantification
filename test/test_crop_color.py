# CLI smoke test for CropEngine.
#
# Usage:
#   python test/test_crop_color.py
#   python test/test_crop_color.py --customer H --new-dir tests/new_images --out-dir tests/results
#
# Input:
#   tests/new_images/FRONT/   raw electrode images
#   tests/new_images/BACK/    raw electrode images
#
# Output:
#   tests/results/FRONT_pass/  cropped PNG
#   tests/results/BACK_pass/   cropped PNG

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.qc.crop_engine import CropEngine, CropResult

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "thresholds.yaml"
DEFAULT_NEW = Path(__file__).parent.parent / "tests" / "new_images"
DEFAULT_OUT = Path(__file__).parent.parent / "tests" / "results"


def process_face(
    face: str,
    input_folder: Path,
    output_folder: Path,
    customer_code: str,
    crop_engine: CropEngine,
) -> list[CropResult]:
    """Run CropEngine on all images in a face folder."""
    pass_folder = output_folder / f"{face}_pass"

    print(f"\n{'-' * 55}")
    print(f"[{face}]  {input_folder}")
    print(f"      -> {pass_folder}")
    print(f"{'-' * 55}")

    if not input_folder.exists():
        print(f"  Folder not found: {input_folder}")
        return []

    results = crop_engine.process_folder(input_folder, customer_code, pass_folder)
    if not results:
        print("  No images found.")
        return []

    passed = [result for result in results if result.status == "PASS"]
    failed = [result for result in results if result.status == "FAIL"]

    print(f"\n  Result: total={len(results)}  pass={len(passed)}  fail={len(failed)}")
    for result in failed:
        reasons = " | ".join(result.fail_reasons)
        print(f"  FAIL  {result.image_path.name:<40}  {reasons}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="CropEngine CLI smoke test")
    parser.add_argument(
        "--customer",
        default="H",
        choices=["H", "F", "V", "S", "T"],
        help="Customer code. Default: H",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=DEFAULT_NEW,
        help=f"Input image root. Default: {DEFAULT_NEW}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output folder. Default: {DEFAULT_OUT}",
    )
    args = parser.parse_args()

    crop_engine = CropEngine(CONFIG_PATH)
    print(f"CropEngine initialized (config: {CONFIG_PATH})")

    all_results: list[CropResult] = []
    for face in ("FRONT", "BACK"):
        input_folder = args.new_dir / face
        results = process_face(
            face,
            input_folder,
            args.out_dir,
            args.customer,
            crop_engine,
        )
        all_results.extend(results)

    total = len(all_results)
    passed = sum(1 for result in all_results if result.status == "PASS")
    failed = total - passed

    print(f"\n{'=' * 55}")
    print(f"Overall result  total={total}  pass={passed}  fail={failed}")
    if failed:
        print("\nFAIL list:")
        for result in all_results:
            if result.status == "FAIL":
                for reason in result.fail_reasons:
                    print(f"  {result.image_path.name:<40}  {reason}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
