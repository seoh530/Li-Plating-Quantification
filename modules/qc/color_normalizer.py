# color_normalizer.py — Mean/Std color normalization using reference image statistics

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class ColorNormalizer:
    """
    Per-channel affine color normalization (Mean/Std Matching).

    Color space
    -----------
    When use_lab=True  (recommended): correction is computed and applied in
    CIELAB space, which separates lightness from chrominance and produces
    more natural-looking results than BGR.
    When use_lab=False: the original BGR affine approach is used.

    Coefficient clamping
    --------------------
    Per-channel scale is clamped to coeff_scale_clamp and shift to
    coeff_shift_clamp to prevent extreme over-correction.

    Blending
    --------
    output = blend_factor * corrected + (1 - blend_factor) * original
    blend_factor=1.0 is full correction; 0.7 gives a subtle, natural shift.

    Input
    -----
    Accepts BGR (uint8, H×W×3) or BGRA (uint8, H×W×4).
    Alpha-zero pixels are excluded from statistics; alpha channel is
    preserved unchanged in the output.

    Prerequisites
    -------------
    Run tools/compute_reference.py once (with matching use_lab setting) to
    populate color_normalizer.reference_*_mean_std in config/thresholds.yaml.
    """

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        c = cfg["color_normalizer"]

        self._use_lab: bool  = bool(c.get("use_lab", True))
        self._blend:   float = float(c.get("blend_factor", 0.7))

        clamp_s  = c.get("coeff_scale_clamp",  [0.8,   1.2])
        clamp_sh = c.get("coeff_shift_clamp",  [-25.0, 25.0])
        self._scale_min = float(clamp_s[0])
        self._scale_max = float(clamp_s[1])
        self._shift_min = float(clamp_sh[0])
        self._shift_max = float(clamp_sh[1])

        margin = int(c["hsv_mask_margin"])
        self._inactive_lower, self._inactive_upper = self._build_hsv_range(
            c["hsv_inactive_range"], margin
        )
        self._bg_lower, self._bg_upper = self._build_hsv_range(
            c["hsv_background_range"], margin
        )

        ref_i = c["reference_inactive_mean_std"]
        ref_b = c["reference_background_mean_std"]

        if ref_i["mean"] is None or ref_b["mean"] is None:
            raise RuntimeError(
                "ColorNormalizer: reference statistics not set in config. "
                "Run tools/compute_reference.py first, then copy the output "
                "values into config/thresholds.yaml."
            )

        self._ref_inactive_mean = np.array(ref_i["mean"], dtype=np.float64)
        self._ref_inactive_std  = np.array(ref_i["std"],  dtype=np.float64)
        self._ref_bg_mean       = np.array(ref_b["mean"], dtype=np.float64)
        self._ref_bg_std        = np.array(ref_b["std"],  dtype=np.float64)

        logger.info(
            "ColorNormalizer init — color_space=%s  blend=%.2f  "
            "scale_clamp=[%.2f, %.2f]  shift_clamp=[%.1f, %.1f]",
            "LAB" if self._use_lab else "BGR",
            self._blend,
            self._scale_min, self._scale_max,
            self._shift_min, self._shift_max,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Apply affine color normalization.

        Returns a copy of the original if masked regions are empty.
        """
        has_alpha = image.ndim == 3 and image.shape[2] == 4
        if has_alpha:
            bgr   = image[:, :, :3].copy()
            alpha = image[:, :, 3]
        else:
            bgr   = image.copy()
            alpha = None

        valid_mask    = (alpha > 0) if alpha is not None else np.ones(bgr.shape[:2], bool)
        inactive_mask = self._extract_hsv_mask(bgr, self._inactive_lower, self._inactive_upper) & valid_mask
        bg_mask       = self._extract_hsv_mask(bgr, self._bg_lower,       self._bg_upper)       & valid_mask

        if self._use_lab:
            result_bgr = self._normalize_lab(bgr, inactive_mask, bg_mask)
        else:
            result_bgr = self._normalize_bgr(bgr, inactive_mask, bg_mask)

        if result_bgr is None:
            return image.copy()

        # Blend corrected with original for a natural, subtle result
        blend      = self._blend
        result_bgr = np.clip(
            blend * result_bgr.astype(np.float64)
            + (1.0 - blend) * bgr.astype(np.float64),
            0, 255,
        ).astype(np.uint8)

        if has_alpha:
            return np.dstack([result_bgr, alpha])
        return result_bgr

    # ── Color-space-specific normalization ─────────────────────────────────────

    def _normalize_lab(
        self,
        bgr: np.ndarray,
        inactive_mask: np.ndarray,
        bg_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Normalize in CIELAB space.
        OpenCV float32 LAB convention: L ∈ [0,100], a/b ∈ [-127, 127].
        """
        bgr_f32 = bgr.astype(np.float32) / 255.0
        lab      = cv2.cvtColor(bgr_f32, cv2.COLOR_BGR2Lab)   # float32 LAB

        inactive_lab = lab[inactive_mask].astype(np.float64)
        bg_lab       = lab[bg_mask].astype(np.float64)

        if inactive_lab.size == 0 or bg_lab.size == 0:
            logger.warning(
                "ColorNormalizer LAB: inactive(%d) or background(%d) mask empty — "
                "skipping normalization",
                int(inactive_mask.sum()), int(bg_mask.sum()),
            )
            return None

        src_i_mean = inactive_lab.mean(axis=0)
        src_i_std  = inactive_lab.std(axis=0)
        src_b_mean = bg_lab.mean(axis=0)
        src_b_std  = bg_lab.std(axis=0)

        logger.debug(
            "LAB src  inactive mean=%s | bg mean=%s",
            src_i_mean.round(2), src_b_mean.round(2),
        )
        logger.debug(
            "LAB ref  inactive mean=%s | bg mean=%s",
            self._ref_inactive_mean.round(2), self._ref_bg_mean.round(2),
        )

        scale, shift = self._compute_coefficients(
            src_i_mean, src_i_std, src_b_mean, src_b_std
        )
        logger.debug("LAB affine  scale=%s  shift=%s", scale.round(4), shift.round(4))

        result_lab = lab.astype(np.float64) * scale + shift

        # Clip to valid LAB ranges before converting back
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0],    0.0,  100.0)   # L
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], -128.0,  127.0)   # a
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], -128.0,  127.0)   # b

        result_bgr = cv2.cvtColor(result_lab.astype(np.float32), cv2.COLOR_Lab2BGR)
        result_bgr = np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)
        return result_bgr

    def _normalize_bgr(
        self,
        bgr: np.ndarray,
        inactive_mask: np.ndarray,
        bg_mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Normalize in BGR space (fallback when use_lab=False)."""
        inactive_px = bgr[inactive_mask].astype(np.float64)
        bg_px       = bgr[bg_mask].astype(np.float64)

        if inactive_px.size == 0 or bg_px.size == 0:
            logger.warning(
                "ColorNormalizer BGR: inactive(%d) or background(%d) mask empty — "
                "skipping normalization",
                int(inactive_mask.sum()), int(bg_mask.sum()),
            )
            return None

        src_i_mean = inactive_px.mean(axis=0)
        src_i_std  = inactive_px.std(axis=0)
        src_b_mean = bg_px.mean(axis=0)
        src_b_std  = bg_px.std(axis=0)

        logger.debug(
            "BGR src  inactive mean=%s | bg mean=%s",
            src_i_mean.round(1), src_b_mean.round(1),
        )

        scale, shift = self._compute_coefficients(
            src_i_mean, src_i_std, src_b_mean, src_b_std
        )
        logger.debug("BGR affine  scale=%s  shift=%s", scale.round(4), shift.round(4))

        result = np.clip(bgr.astype(np.float64) * scale + shift, 0, 255).astype(np.uint8)
        return result

    # ── Shared coefficient computation ─────────────────────────────────────────

    def _compute_coefficients(
        self,
        src_i_mean: np.ndarray, src_i_std: np.ndarray,
        src_b_mean: np.ndarray, src_b_std: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Derive clamped per-channel (scale, shift) from two region pairs.
        Scale is the average of std-ratio estimates; shift matches the means.
        """
        safe_i_std  = np.where(src_i_std  < 1e-6, 1.0, src_i_std)
        safe_b_std  = np.where(src_b_std  < 1e-6, 1.0, src_b_std)
        safe_ri_std = np.where(self._ref_inactive_std < 1e-6, safe_i_std, self._ref_inactive_std)
        safe_rb_std = np.where(self._ref_bg_std        < 1e-6, safe_b_std, self._ref_bg_std)

        scale_i = safe_ri_std / safe_i_std
        scale_b = safe_rb_std / safe_b_std
        scale   = (scale_i + scale_b) / 2.0
        scale   = np.clip(scale, self._scale_min, self._scale_max)

        shift_i = self._ref_inactive_mean - scale * src_i_mean
        shift_b = self._ref_bg_mean       - scale * src_b_mean
        shift   = (shift_i + shift_b) / 2.0
        shift   = np.clip(shift, self._shift_min, self._shift_max)

        return scale, shift

    # ── Static helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_hsv_range(
        hsv_range: dict, margin: int
    ) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array(hsv_range["lower"], dtype=np.int32)
        hi = np.array(hsv_range["upper"], dtype=np.int32)
        lo -= margin
        hi += margin
        lo[0]  = int(np.clip(lo[0],  0,  179))
        hi[0]  = int(np.clip(hi[0],  0,  179))
        lo[1:] = np.clip(lo[1:], 0, 255)
        hi[1:] = np.clip(hi[1:], 0, 255)
        return lo.astype(np.uint8), hi.astype(np.uint8)

    @staticmethod
    def _extract_hsv_mask(
        bgr: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> np.ndarray:
        hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return mask.astype(bool)


