# test_crop_color.py — CLI test for CropEngine + ColorNormalizer
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
#   tests/results/FRONT_pass/  cropped + normalized PNG
#   tests/results/BACK_pass/   cropped + normalized PNG
#
# Expected PASS / FAIL outcomes are documented in GUIDE.md.

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.qc.crop_engine    import CropEngine,    CropResult
from modules.qc.color_normalizer import ColorNormalizer

import cv2

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH  = Path(__file__).parent.parent / "config" / "thresholds.yaml"
DEFAULT_NEW  = Path(__file__).parent.parent / "tests" / "new_images"
DEFAULT_OUT  = Path(__file__).parent.parent / "tests" / "results"


def process_face(
    face:          str,           # "FRONT" or "BACK"
    input_folder:  Path,
    output_folder: Path,
    customer_code: str,
    crop_engine:   CropEngine,
    normalizer:    ColorNormalizer | None,
) -> list[CropResult]:
    """Run CropEngine (then ColorNormalizer if available) on all images in a face folder."""
    pass_folder = output_folder / f"{face}_pass"

    print(f"\n{'─' * 55}")
    print(f"[{face}]  {input_folder}")
    print(f"      → {pass_folder}")
    print(f"{'─' * 55}")

    if not input_folder.exists():
        print(f"  폴더 없음: {input_folder}")
        return []

    results = crop_engine.process_folder(input_folder, customer_code, pass_folder)

    if not results:
        print("  이미지 없음")
        return []

    # Apply ColorNormalizer to each PASS output
    if normalizer is not None:
        for r in results:
            if r.status == "PASS" and r.output_path and r.output_path.exists():
                bgra = cv2.imread(str(r.output_path), cv2.IMREAD_UNCHANGED)
                if bgra is None:
                    logger.warning("ColorNormalizer: 파일 로드 실패 — %s", r.output_path.name)
                    continue
                normalized = normalizer.normalize(bgra)
                cv2.imwrite(str(r.output_path), normalized)
                logger.debug("색상 정규화 완료: %s", r.output_path.name)

    # Summary
    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]

    print(f"\n  결과: total={len(results)}  pass={len(passed)}  fail={len(failed)}")

    for r in failed:
        reasons = " | ".join(r.fail_reasons)
        print(f"  FAIL  {r.image_path.name:<40}  {reasons}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CropEngine + ColorNormalizer CLI 테스트"
    )
    parser.add_argument(
        "--customer", default="H",
        choices=["H", "F", "V", "S", "T"],
        help="고객 코드 (기본값: H = HKMC)",
    )
    parser.add_argument(
        "--new-dir", type=Path, default=DEFAULT_NEW,
        help=f"처리 대상 이미지 루트 폴더 (기본값: {DEFAULT_NEW})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT,
        help=f"결과 저장 폴더 (기본값: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help="ColorNormalizer 건너뜀 (CropEngine만 실행)",
    )
    args = parser.parse_args()

    # ── Init CropEngine ────────────────────────────────────────────────────
    crop_engine = CropEngine(CONFIG_PATH)
    print(f"CropEngine 초기화 완료  (config: {CONFIG_PATH})")

    # ── Init ColorNormalizer (optional) ────────────────────────────────────
    normalizer: ColorNormalizer | None = None
    if not args.no_normalize:
        try:
            normalizer = ColorNormalizer(CONFIG_PATH)
            print("ColorNormalizer 초기화 완료")
        except RuntimeError as e:
            print(f"\n⚠  ColorNormalizer 비활성화: {e}")
            print("   --no-normalize 옵션을 사용하거나 compute_reference.py 를 먼저 실행하세요.\n")

    # ── Process FRONT and BACK ─────────────────────────────────────────────
    all_results: list[CropResult] = []

    for face in ("FRONT", "BACK"):
        input_folder  = args.new_dir / face
        results = process_face(
            face, input_folder, args.out_dir,
            args.customer, crop_engine, normalizer,
        )
        all_results.extend(results)

    # ── Overall summary ────────────────────────────────────────────────────
    total  = len(all_results)
    passed = sum(1 for r in all_results if r.status == "PASS")
    failed = total - passed

    print(f"\n{'=' * 55}")
    print(f"전체 결과  total={total}  pass={passed}  fail={failed}")
    if failed:
        print(f"\nFAIL 목록:")
        for r in all_results:
            if r.status == "FAIL":
                for reason in r.fail_reasons:
                    print(f"  {r.image_path.name:<40}  {reason}")
    print(f"{'=' * 55}")

    if passed:
        print(f"\n저장 위치: {args.out_dir}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()