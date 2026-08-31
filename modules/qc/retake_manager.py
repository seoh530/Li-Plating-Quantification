# retake_manager.py — NG image deletion and per-image re-scan after retake

import logging
from dataclasses import dataclass, field
from pathlib import Path

from modules.qc.crop_engine import CropEngine, CropResult

logger = logging.getLogger(__name__)


# ── RetakeItem ─────────────────────────────────────────────────────────────────

@dataclass
class RetakeItem:
    """One retake task for a single FAIL image."""
    image_path: Path
    fail_reasons: list[str]
    folder_path: Path           # FRONT or BACK folder (context for re-scan)
    customer_code: str          # single-char code (H / F / V / S / T)
    output_folder: Path         # _pass folder path
    output_path: Path | None = None
    skip_size_validation: bool = False   # True if image was first/last in folder
    fixed_skip_size_validation: bool = False
    status: str = "PENDING"             # "PENDING" | "PASS" | "FAIL"


# ── RetakeManager ──────────────────────────────────────────────────────────────

class RetakeManager:
    """
    Tracks CropEngine FAIL images and re-processes them after operator retake.

    Usage:
      1. Call load() after CropEngine.process_folder() to register FAIL items.
      2. Call get_pending_items() to populate the GUI retake list.
      3. Operator replaces/overwrites the FAIL image file.
      4. Call retry_image(item) to re-process only that single image.
      5. Repeat until get_pending_items() == [].
    """

    def __init__(self, config_path) -> None:
        self._config_path = Path(config_path)
        self._crop_engine = CropEngine(config_path)
        self._items: list[RetakeItem] = []

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(
        self,
        crop_results: list[CropResult],
        folder_path: Path,
        customer_code: str,
        output_folder: Path,
        output_paths: dict[Path, Path] | None = None,
        skip_size_validation_for_all: bool = False,
    ) -> None:
        """Register FAIL results from CropEngine as RetakeItems."""
        folder_path   = Path(folder_path)
        output_folder = Path(output_folder)

        # Determine first/last image paths for skip_size_validation
        first_path = crop_results[0].image_path  if crop_results             else None
        last_path  = crop_results[-1].image_path if len(crop_results) > 1    else None

        for result in crop_results:
            if result.status == "FAIL":
                skip_size = skip_size_validation_for_all or (
                    result.image_path == first_path or
                    result.image_path == last_path
                )
                self._items.append(RetakeItem(
                    image_path=result.image_path,
                    fail_reasons=list(result.fail_reasons),
                    folder_path=folder_path,
                    customer_code=customer_code,
                    output_folder=output_folder,
                    output_path=(
                        output_paths.get(result.image_path)
                        if output_paths is not None
                        else None
                    ),
                    skip_size_validation=skip_size,
                    fixed_skip_size_validation=skip_size_validation_for_all,
                ))
                logger.debug("RetakeManager registered: %s", result.image_path.name)

    def retry_image(self, item: RetakeItem) -> CropResult:
        """
        Re-process a single image after the operator has replaced the file.

        Determines whether the image is still first/last in the folder
        (for skip_size_validation consistency), runs CropEngine, and applies
        On PASS  → item is removed from the pending list.
        On FAIL  → item.fail_reasons is updated; item stays PENDING.
        """
        # Check if image is first/last in its folder
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        folder_images = sorted(
            p for p in item.folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )
        skip_size = item.skip_size_validation
        if folder_images and not item.fixed_skip_size_validation:
            skip_size = (
                item.image_path == folder_images[0] or
                item.image_path == folder_images[-1]
            )

        out_path = item.output_path or item.output_folder / (item.image_path.stem + ".png")

        logger.info("retry_image: %s  (skip_size=%s)", item.image_path.name, skip_size)
        result = self._crop_engine.process_image(
            image_path=item.image_path,
            customer_code=item.customer_code,
            output_path=out_path,
            skip_size_validation=skip_size,
            image_postprocess=None,
        )

        if result.status == "PASS":
            # Remove from pending list
            self._items = [i for i in self._items if i is not item]
            logger.info("retry_image PASS — removed from retake list: %s", item.image_path.name)
        else:
            item.fail_reasons = list(result.fail_reasons)
            item.status = "PENDING"
            logger.warning(
                "retry_image FAIL: %s  %s",
                item.image_path.name,
                " | ".join(result.fail_reasons),
            )

        return result

    def force_crop_image(self, item: RetakeItem) -> CropResult:
        """
        Save the cropped output for a pending FAIL image when the operator accepts it.

        Fatal crop failures such as unreadable image, missing contour, or empty crop
        still remain FAIL because there is no reliable crop to save.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        folder_images = sorted(
            p for p in item.folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )
        skip_size = item.skip_size_validation
        if folder_images and not item.fixed_skip_size_validation:
            skip_size = (
                item.image_path == folder_images[0] or
                item.image_path == folder_images[-1]
            )

        out_path = item.output_path or item.output_folder / (item.image_path.stem + ".png")
        logger.info(
            "force_crop_image: %s  (skip_size=%s)", item.image_path.name, skip_size
        )
        result = self._crop_engine.process_image(
            image_path=item.image_path,
            customer_code=item.customer_code,
            output_path=out_path,
            skip_size_validation=skip_size,
            image_postprocess=None,
            force_save_failed=True,
        )

        if result.status == "PASS":
            self._items = [i for i in self._items if i is not item]
            logger.info(
                "force_crop_image PASS — removed from retake list: %s",
                item.image_path.name,
            )
        else:
            item.fail_reasons = list(result.fail_reasons)
            item.status = "PENDING"
            logger.warning(
                "force_crop_image FAIL: %s  %s",
                item.image_path.name,
                " | ".join(result.fail_reasons),
            )
        return result

    def retry_folder(self, folder_path: Path) -> list[CropResult]:
        """
        Re-scan an entire folder and re-run CropEngine.
        Replaces existing RetakeItems for this folder with new FAIL results.
        (Kept for compatibility — prefer retry_image() for single-image retake.)
        """
        folder_path  = Path(folder_path)
        folder_items = [i for i in self._items if i.folder_path == folder_path]

        if not folder_items:
            logger.warning("retry_folder: no registered items for %s", folder_path)
            return []

        customer_code = folder_items[0].customer_code
        output_folder = folder_items[0].output_folder

        logger.info("retry_folder: %s", folder_path)
        new_results = self._crop_engine.process_folder(
            folder_path=folder_path,
            customer_code=customer_code,
            output_folder=output_folder,
            image_postprocess=None,
        )

        first_path = new_results[0].image_path  if new_results             else None
        last_path  = new_results[-1].image_path if len(new_results) > 1    else None

        self._items = [i for i in self._items if i.folder_path != folder_path]
        for r in new_results:
            if r.status == "FAIL":
                skip_size = (r.image_path == first_path or r.image_path == last_path)
                self._items.append(RetakeItem(
                    image_path=r.image_path,
                    fail_reasons=list(r.fail_reasons),
                    folder_path=folder_path,
                    customer_code=customer_code,
                    output_folder=output_folder,
                    skip_size_validation=skip_size,
                ))

        passed = sum(1 for r in new_results if r.status == "PASS")
        logger.info(
            "retry_folder done: pass=%d  fail=%d",
            passed, len(new_results) - passed,
        )
        return new_results

    def get_pending_items(self) -> list[RetakeItem]:
        """Return all items with PENDING status."""
        return [i for i in self._items if i.status == "PENDING"]

    def remove_by_analysis(self, analysis_path: Path) -> None:
        """Remove all items under analysis_path (used before a full refresh)."""
        analysis_path = Path(analysis_path)
        before = len(self._items)
        self._items = [
            i for i in self._items
            if not _is_subpath(i.folder_path, analysis_path)
        ]
        removed = before - len(self._items)
        if removed:
            logger.debug(
                "remove_by_analysis: %d items removed (%s)", removed, analysis_path.name
            )

    def clear(self) -> None:
        """Clear all items."""
        self._items.clear()
        logger.debug("RetakeManager: cleared")


# ── Utility ────────────────────────────────────────────────────────────────────

def _is_subpath(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False
