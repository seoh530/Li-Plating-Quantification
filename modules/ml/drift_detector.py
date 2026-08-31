import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackgroundBaseline:
    l_mean: float
    a_mean: float
    b_mean: float
    l_std: float
    a_std: float
    b_std: float


@dataclass(frozen=True)
class BackgroundDrift:
    raw_bg_l: float | None
    raw_bg_a: float | None
    raw_bg_b_lab: float | None
    baseline_bg_l: float | None
    baseline_bg_a: float | None
    baseline_bg_b_lab: float | None
    l_shift: float | None
    a_shift: float | None
    b_shift: float | None
    drift_score_before: float | None
    high_conf_bg_ratio: float | None
    skipped_reason: str | None


class DriftDetector:
    """Compute background LAB drift against optional training baseline stats."""

    def __init__(
        self,
        baseline_path: str | Path,
        *,
        min_high_conf_bg_ratio: float,
        drift_threshold: float,
    ) -> None:
        self._baseline_path = Path(baseline_path)
        self._min_high_conf_bg_ratio = float(min_high_conf_bg_ratio)
        self._drift_threshold = float(drift_threshold)
        self._baseline = self._load_baseline()

    @property
    def baseline(self) -> BackgroundBaseline | None:
        return self._baseline

    @property
    def has_baseline(self) -> bool:
        return self._baseline is not None

    @property
    def drift_threshold(self) -> float:
        return self._drift_threshold

    def evaluate(
        self,
        lab_image: np.ndarray,
        valid_mask: np.ndarray,
        background_mask: np.ndarray | None,
    ) -> BackgroundDrift:
        valid_count = int(np.count_nonzero(valid_mask))
        if valid_count <= 0 or background_mask is None:
            reason = "missing_baseline_stats" if self._baseline is None else "no_valid_pixels"
            return self._empty_with_baseline(reason, None)

        trusted_bg = background_mask & valid_mask
        bg_count = int(np.count_nonzero(trusted_bg))
        high_conf_bg_ratio = bg_count / valid_count
        if high_conf_bg_ratio < self._min_high_conf_bg_ratio:
            return self._empty_with_baseline(
                "not_enough_high_confidence_background",
                high_conf_bg_ratio,
            )

        bg_lab = lab_image[trusted_bg].astype(np.float64)
        raw_l, raw_a, raw_b = [float(value) for value in bg_lab.mean(axis=0)]
        if self._baseline is None:
            return BackgroundDrift(
                raw_bg_l=raw_l,
                raw_bg_a=raw_a,
                raw_bg_b_lab=raw_b,
                baseline_bg_l=None,
                baseline_bg_a=None,
                baseline_bg_b_lab=None,
                l_shift=None,
                a_shift=None,
                b_shift=None,
                drift_score_before=None,
                high_conf_bg_ratio=high_conf_bg_ratio,
                skipped_reason="missing_baseline_stats",
            )

        baseline = self._baseline
        l_shift = baseline.l_mean - raw_l
        a_shift = baseline.a_mean - raw_a
        b_shift = baseline.b_mean - raw_b
        drift_score = max(
            self._z_score(raw_l, baseline.l_mean, baseline.l_std),
            self._z_score(raw_a, baseline.a_mean, baseline.a_std),
            self._z_score(raw_b, baseline.b_mean, baseline.b_std),
        )

        return BackgroundDrift(
            raw_bg_l=raw_l,
            raw_bg_a=raw_a,
            raw_bg_b_lab=raw_b,
            baseline_bg_l=baseline.l_mean,
            baseline_bg_a=baseline.a_mean,
            baseline_bg_b_lab=baseline.b_mean,
            l_shift=l_shift,
            a_shift=a_shift,
            b_shift=b_shift,
            drift_score_before=drift_score,
            high_conf_bg_ratio=high_conf_bg_ratio,
            skipped_reason=None,
        )

    def _empty_with_baseline(
        self,
        reason: str,
        high_conf_bg_ratio: float | None,
    ) -> BackgroundDrift:
        baseline = self._baseline
        return BackgroundDrift(
            raw_bg_l=None,
            raw_bg_a=None,
            raw_bg_b_lab=None,
            baseline_bg_l=baseline.l_mean if baseline else None,
            baseline_bg_a=baseline.a_mean if baseline else None,
            baseline_bg_b_lab=baseline.b_mean if baseline else None,
            l_shift=None,
            a_shift=None,
            b_shift=None,
            drift_score_before=None,
            high_conf_bg_ratio=high_conf_bg_ratio,
            skipped_reason=reason,
        )

    def _load_baseline(self) -> BackgroundBaseline | None:
        if not self._baseline_path.exists():
            logger.info("Baseline stats not found: %s", self._baseline_path)
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            global_stats = data.get("global", data)
            bg_lab = global_stats.get("background_lab", {})
            mean = bg_lab.get("mean", {})
            std = bg_lab.get("std", {})
            return BackgroundBaseline(
                l_mean=float(mean["l"]),
                a_mean=float(mean["a"]),
                b_mean=float(mean["b"]),
                l_std=float(std["l"]),
                a_std=float(std["a"]),
                b_std=float(std["b"]),
            )
        except Exception:
            logger.exception("Failed to load baseline stats: %s", self._baseline_path)
            return None

    @staticmethod
    def _z_score(value: float, mean: float, std: float) -> float:
        safe_std = std if abs(std) > 1e-6 else 1.0
        return abs(value - mean) / safe_std
