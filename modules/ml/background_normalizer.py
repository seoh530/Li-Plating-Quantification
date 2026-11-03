from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalizationResult:
    image: np.ndarray
    correction_applied: bool
    l_shift: float
    a_shift: float
    b_shift: float


class BackgroundLABNormalizer:
    """Apply clamped LAB channel shifts while preserving alpha."""

    def __init__(
        self,
        *,
        max_l_shift: float,
        max_a_shift: float,
        max_b_shift: float,
    ) -> None:
        self._max_l_shift = float(max_l_shift)
        self._max_a_shift = float(max_a_shift)
        self._max_b_shift = float(max_b_shift)

    def normalize(
        self,
        image: np.ndarray,
        *,
        l_shift: float | None,
        a_shift: float | None,
        b_shift: float | None,
    ) -> NormalizationResult:
        if l_shift is None or a_shift is None or b_shift is None:
            return NormalizationResult(image=image.copy(), correction_applied=False, l_shift=0.0, a_shift=0.0, b_shift=0.0)

        applied_l = float(np.clip(l_shift, -self._max_l_shift, self._max_l_shift))
        applied_a = float(np.clip(a_shift, -self._max_a_shift, self._max_a_shift))
        applied_b = float(np.clip(b_shift, -self._max_b_shift, self._max_b_shift))

        if max(abs(applied_l), abs(applied_a), abs(applied_b)) <= 1e-6:
            return NormalizationResult(image=image.copy(), correction_applied=False, l_shift=0.0, a_shift=0.0, b_shift=0.0)

        has_alpha = image.ndim == 3 and image.shape[2] == 4
        bgr = image[:, :, :3] if has_alpha else image
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + applied_l, 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + applied_a, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + applied_b, 0, 255)
        corrected_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        if has_alpha:
            return NormalizationResult(
                image=np.dstack([corrected_bgr, image[:, :, 3]]),
                correction_applied=True,
                l_shift=applied_l,
                a_shift=applied_a,
                b_shift=applied_b,
            )

        return NormalizationResult(
            image=corrected_bgr,
            correction_applied=True,
            l_shift=applied_l,
            a_shift=applied_a,
            b_shift=applied_b,
        )
