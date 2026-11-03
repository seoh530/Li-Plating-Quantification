# crop_engine.py — Electrode contour detection, padded crop, tolerance validation

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from modules.utils.image_io import atomic_imwrite_png, quiet_imread

logger = logging.getLogger(__name__)


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class CropResult:
    image_path: Path
    output_path: Optional[Path]
    status: str                           # "PASS" | "FAIL"
    fail_reasons: list[str] = field(default_factory=list)
    pixel_count: int = 0                  # non-transparent pixels; 0 if pipeline aborted
    qc_exempt: bool = False               # True for first/last images excluded from QC/ML


# ─── Atomic PNG write ──────────────────────────────────────────────────────────

def _atomic_imwrite(path: Path, img: np.ndarray) -> bool:
    return atomic_imwrite_png(path, img, compression=1)


# ─── CropEngine ───────────────────────────────────────────────────────────────

class CropEngine:
    """
    Electrode image crop engine.

    Pipeline per image:
      1. White paper background mask
      2. Largest contour detection
      3. Angle validation using minAreaRect  (non-blocking — records FAIL)
      4. Contour bounding box + padding crop
      5. Source-image boundary touch check   (non-blocking — records FAIL)
      6. Alpha masking — white bg → transparent PNG
      7. Aspect ratio tolerance check
         (skipped for the first and last image in each FRONT/BACK folder)

    Folder-level (after all images):
      8. Pixel count deviation check (±pixel_count_deviation across folder)

    Output: <output_folder>/<stem>.png  (BGRA, alpha=0 for background pixels)
    """

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        c = cfg["crop_engine"]

        self._angle_tol: float = float(c["crop_angle_tolerance"])
        self._size_dev:  float = float(c["size_deviation"])
        self._pixel_dev: float = float(c["pixel_count_deviation"])

        self._crop_padding_px: int = int(c.get("crop_padding_px", 30))
        self._source_edge_margin_px: float = float(c.get("source_edge_margin_px", 3))
        self._source_edge_contact_min_px: int = int(
            c.get("source_edge_contact_min_px", 25)
        )

        pm = c["paper_mask_hsv"]
        self._paper_lower = np.array(pm["lower"], dtype=np.uint8)
        self._paper_upper = np.array(pm["upper"], dtype=np.uint8)

        self._ref_ratios: dict = c["customer_reference_ratios"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def process_folder(
        self,
        folder_path: str | Path,
        customer_code: str,
        output_folder: str | Path,
        image_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> list[CropResult]:
        """
        Process all images in a FRONT or BACK folder.

        The first and last image (sorted by filename) skip aspect ratio
        validation — outermost sheets are often curved and would produce
        false failures.
        """
        folder_path   = Path(folder_path)
        output_folder = Path(output_folder)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_paths = sorted(
            p for p in folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in image_extensions
        )

        if not image_paths:
            logger.warning("No images found: %s", folder_path)
            return []

        output_folder.mkdir(parents=True, exist_ok=True)

        n = len(image_paths)
        results: list[CropResult] = []
        for i, img_path in enumerate(image_paths):
            out_path  = output_folder / (img_path.stem + ".png")
            if (n > 1) and (i == 0 or i == n - 1):
                if out_path.exists():
                    out_path.unlink()
                logger.info(
                    "QC-EXEMPT PASS  %-40s  first/last image excluded from ML",
                    img_path.name,
                )
                results.append(
                    CropResult(
                        image_path=img_path,
                        output_path=None,
                        status="PASS",
                        qc_exempt=True,
                    )
                )
                continue

            result    = self.process_image(
                img_path, customer_code, out_path,
                skip_size_validation=False,
                image_postprocess=image_postprocess,
            )
            results.append(result)

        self._check_pixel_count_deviation(results)

        passed = sum(1 for r in results if r.status == "PASS")
        logger.info(
            "process_folder  %-30s  total=%d  pass=%d  fail=%d",
            folder_path.name, n, passed, n - passed,
        )
        return results

    def process_image(
        self,
        image_path: str | Path,
        customer_code: str,
        output_path: str | Path,
        skip_size_validation: bool = False,
        image_postprocess: Callable[[np.ndarray], np.ndarray] | None = None,
        force_save_failed: bool = False,
    ) -> CropResult:
        """
        Process a single electrode image end-to-end.

        Parameters
        ----------
        skip_size_validation : When True, step 7 (aspect ratio check) is skipped.
                               Used for first/last images in a folder.
        """
        image_path  = Path(image_path)
        output_path = Path(output_path)
        result = CropResult(image_path=image_path, output_path=None, status="PASS")

        # ── Load ────────────────────────────────────────────────────────────
        bgr = quiet_imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            result.status = "FAIL"
            result.fail_reasons.append(f"Image load failed: {image_path.name}")
            logger.error("Image load failed: %s", image_path)
            return result

        img_h, img_w = bgr.shape[:2]

        # ── Step 1: paper mask ──────────────────────────────────────────────
        paper_mask = self._extract_paper_mask(bgr)

        # ── Step 2: largest contour ─────────────────────────────────────────
        contour = self._find_largest_contour(paper_mask)
        if contour is None:
            result.status = "FAIL"
            result.fail_reasons.append(
                "Contour extraction failed — insufficient white background or invalid image"
            )
            return result

        # ── Step 3: angle validation (non-blocking) ─────────────────────────
        angle = self._compute_contour_angle(contour)
        if abs(angle) > self._angle_tol:
            result.status = "FAIL"
            result.fail_reasons.append(
                f"Angle tolerance exceeded: {angle:+.2f}°  (allowed: ±{self._angle_tol}°)"
            )

        # ── Step 4: bounding box + padding crop ─────────────────────────────
        x1, y1, x2, y2 = self._get_padded_crop_rect(contour, img_w, img_h)
        cropped = bgr[y1:y2, x1:x2].copy()

        if cropped.size == 0:
            result.status = "FAIL"
            result.fail_reasons.append("Crop result is empty")
            return result

        # ── Step 5: source boundary touch check (non-blocking) ──────────────
        if self._is_contour_touching_source_boundary(contour, img_w, img_h):
            result.status = "FAIL"
            result.fail_reasons.append(
                "Electrode touches source image boundary — may be partially out of frame"
            )

        # ── Step 6: alpha mask + pixel count ────────────────────────────────
        bgra = self._apply_alpha_mask(cropped)
        result.pixel_count = int(np.count_nonzero(bgra[:, :, 3]))

        # ── Step 7: aspect ratio tolerance ──────────────────────────────────
        actual_ratio = self._compute_contour_ratio(contour)
        if skip_size_validation:
            logger.debug(
                "Size validation skipped (first/last image)  ratio=%.4f", actual_ratio
            )
        else:
            ref_ratio = self._ref_ratios.get(customer_code)
            if ref_ratio is not None and actual_ratio > 0:
                deviation = abs(actual_ratio - ref_ratio) / ref_ratio
                if deviation > self._size_dev:
                    result.status = "FAIL"
                    result.fail_reasons.append(
                        f"Aspect ratio out of tolerance: {actual_ratio:.4f}  "
                        f"(ref: {ref_ratio:.3f}, deviation: {deviation * 100:.2f}%  "
                        f"allowed: ±{self._size_dev * 100:.1f}%)"
                    )

        # ── Save (PASS only) ────────────────────────────────────────────────
        should_save = result.status == "PASS" or force_save_failed
        if should_save:
            if image_postprocess is not None:
                bgra = image_postprocess(bgra)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not _atomic_imwrite(output_path, bgra):
                result.status = "FAIL"
                result.fail_reasons.append(f"Failed to write output: {output_path.name}")
                logger.error("PNG encode failed: %s", output_path)
                return result
            result.output_path = output_path
            if result.status != "PASS":
                logger.warning(
                    "MANUAL OVERRIDE  %-40s  saved despite: %s",
                    image_path.name,
                    " | ".join(result.fail_reasons),
                )
                result.status = "PASS"
            logger.info(
                "PASS  %-40s  pixels=%7d  ratio=%.4f  angle=%+.2f°  crop=%dx%d%s",
                image_path.name,
                result.pixel_count,
                actual_ratio,
                angle,
                cropped.shape[1],
                cropped.shape[0],
                "  [size skip]" if skip_size_validation else "",
            )
        else:
            logger.warning(
                "FAIL  %-40s  %s",
                image_path.name,
                " | ".join(result.fail_reasons),
            )

        return result

    # ── Pipeline steps ─────────────────────────────────────────────────────────

    def _extract_paper_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Return binary mask of white paper background pixels."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, self._paper_lower, self._paper_upper)

    def _find_largest_contour(self, paper_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        Invert paper mask → electrode region → return the largest external contour.
        Uses CHAIN_APPROX_SIMPLE for speed (corner extraction no longer needed).
        """
        electrode_mask = cv2.bitwise_not(paper_mask)
        contours, _ = cv2.findContours(
            electrode_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 10_000:
            logger.warning("Largest contour area too small: %.0f px²", area)
            return None
        return largest

    def _compute_contour_angle(self, contour: np.ndarray) -> float:
        """
        Compute electrode tilt angle from cv2.minAreaRect.
        Result is normalized to approximately [-45, +45].
        """
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), angle = rect

        if w < h:
            angle -= 90.0
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        if angle > 45.0:
            angle -= 90.0
        elif angle < -45.0:
            angle += 90.0

        logger.debug("Contour angle: %+.3f°", angle)
        return float(angle)

    def _get_padded_crop_rect(
        self,
        contour: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> tuple[int, int, int, int]:
        """Return padded bounding rectangle of the contour, clamped to image bounds."""
        x, y, w, h = cv2.boundingRect(contour)
        pad = self._crop_padding_px
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        logger.debug(
            "Padded crop: (%d,%d)-(%d,%d)  size=%dx%d",
            x1, y1, x2, y2, x2 - x1, y2 - y1,
        )
        return x1, y1, x2, y2

    def _is_contour_touching_source_boundary(
        self,
        contour: np.ndarray,
        img_w: int,
        img_h: int,
    ) -> bool:
        """Return True if enough contour pixels touch the source image edge."""
        x, y, w, h = cv2.boundingRect(contour)
        m = self._source_edge_margin_px
        edge_width = max(1, int(np.ceil(max(m, 0.0))))
        min_contact = max(1, self._source_edge_contact_min_px)

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        left_count = int(np.count_nonzero(mask[:, :edge_width]))
        top_count = int(np.count_nonzero(mask[:edge_width, :]))
        right_count = int(np.count_nonzero(mask[:, img_w - edge_width:]))
        bottom_count = int(np.count_nonzero(mask[img_h - edge_width:, :]))

        touch_left = x <= m and left_count >= min_contact
        touch_top = y <= m and top_count >= min_contact
        touch_right = (x + w) >= (img_w - 1 - m) and right_count >= min_contact
        touch_bottom = (y + h) >= (img_h - 1 - m) and bottom_count >= min_contact
        touched = touch_left or touch_top or touch_right or touch_bottom
        if touched:
            logger.warning(
                "Source boundary contact: left=%s(%dpx) top=%s(%dpx) right=%s(%dpx) bottom=%s(%dpx) margin=%.2f min_contact=%d bbox=(%d,%d,%d,%d) image=%dx%d",
                touch_left,
                left_count,
                touch_top,
                top_count,
                touch_right,
                right_count,
                touch_bottom,
                bottom_count,
                m,
                min_contact,
                x,
                y,
                w,
                h,
                img_w,
                img_h,
            )
        return touched

    def _compute_contour_ratio(self, contour: np.ndarray) -> float:
        """
        Compute electrode aspect ratio from minAreaRect (long/short side).
        Padding does not affect this since it is contour-based.
        """
        rect = cv2.minAreaRect(contour)
        (_, _), (w, h), _ = rect
        w = float(w)
        h = float(h)
        short_side = max(min(w, h), 1.0)
        long_side  = max(w, h)
        ratio = long_side / short_side
        logger.debug("Contour aspect ratio: %.4f", ratio)
        return ratio

    def _apply_alpha_mask(self, cropped_bgr: np.ndarray) -> np.ndarray:
        """
        Build alpha channel from the largest electrode contour (filled).
        Morphological close/open removes small gaps at edges before fill.
        Anything outside the largest contour becomes fully transparent.
        """
        hsv          = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2HSV)
        paper_mask   = cv2.inRange(hsv, self._paper_lower, self._paper_upper)
        elec_mask    = cv2.bitwise_not(paper_mask)

        kernel = np.ones((5, 5), np.uint8)
        elec_mask = cv2.morphologyEx(elec_mask, cv2.MORPH_CLOSE, kernel)
        elec_mask = cv2.morphologyEx(elec_mask, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(
            elec_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        alpha = np.zeros(elec_mask.shape, dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(alpha, [largest], -1, 255, thickness=cv2.FILLED)

        bgra = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        return bgra

    # ── Folder-level deviation check ───────────────────────────────────────────

    def _check_pixel_count_deviation(self, results: list[CropResult]) -> None:
        """
        Mark images FAIL if pixel count deviates more than pixel_count_deviation
        from the folder mean.  Deletes previously saved output files for newly-failed images.
        """
        valid = [r for r in results if r.pixel_count > 0]
        if len(valid) < 2:
            return

        counts     = np.array([r.pixel_count for r in valid], dtype=np.float64)
        mean_count = float(counts.mean())

        for r in valid:
            deviation = abs(r.pixel_count - mean_count) / mean_count
            if deviation > self._pixel_dev:
                r.status = "FAIL"
                r.fail_reasons.append(
                    f"Pixel count deviation exceeded: {r.pixel_count:,}px  "
                    f"(folder mean: {mean_count:,.0f}px, "
                    f"deviation: {deviation * 100:.1f}%  "
                    f"allowed: ±{self._pixel_dev * 100:.0f}%)"
                )
                if r.output_path and r.output_path.exists():
                    r.output_path.unlink()
                    r.output_path = None
                logger.warning(
                    "Pixel count deviation FAIL — file deleted  %s  "
                    "(count=%d, mean=%.0f, dev=%.1f%%)",
                    r.image_path.name, r.pixel_count, mean_count, deviation * 100,
                )
