import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import yaml

from modules.ml.drift_detector import BackgroundDrift, DriftDetector
from modules.ml.feature_extractor import FeatureExtractor
from modules.ml.inference import InferenceEngine, LABEL_BACKGROUND
from modules.ml.quantifier import Quantifier
from modules.utils.analysis_id import is_test_analysis_id
from modules.utils.image_io import quiet_imread

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceTask:
    analysis_id: str
    analysis_path: Path
    cell: int
    face: str
    image_num: int
    image_path: Path
    customer: str | None = None


@dataclass
class InferencePipelineResult:
    task: InferenceTask
    result_row: dict
    monitoring_record: dict
    mask_path: Path | None
    elapsed_s: float


class InferencePipeline:
    """Run image feature extraction, inference, quantification, drift checks, and mask output."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        model_dir: str | Path | None = None,
        save_mask_images: bool | None = None,
        customer_codes: Mapping[str, str] | None = None,
    ) -> None:
        self._config_path = Path(config_path)
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        ml_cfg = cfg.get("ml", {}) or {}
        normalization_cfg = dict(cfg.get("normalization", {}) or {})
        resolved_model_dir = Path(model_dir) if model_dir is not None else Path(ml_cfg["model_path"])

        self._feature_extractor = FeatureExtractor(self._config_path)
        self._inference = InferenceEngine(self._config_path, model_path=resolved_model_dir)
        self._quantifier = Quantifier(self._config_path)
        self._save_mask_images = (
            bool(ml_cfg.get("save_mask_images", True))
            if save_mask_images is None
            else bool(save_mask_images)
        )
        self._normalization_cfg = normalization_cfg
        self._low_confidence_threshold = float(
            normalization_cfg.get("low_confidence_threshold", 0.60)
        )
        self._customer_codes = dict(customer_codes or {})
        self._drift_detector = DriftDetector(
            resolved_model_dir / "baseline_stats.json",
            min_high_conf_bg_ratio=float(
                normalization_cfg.get("min_high_conf_bg_ratio", 0.05)
            ),
            drift_threshold=float(normalization_cfg.get("drift_threshold", 2.0)),
        )

    @property
    def model_dir(self) -> Path:
        return self._inference.model_dir

    def run_image(
        self,
        task: InferenceTask,
        *,
        mask_path: str | Path | None = None,
        extra_result_fields: Mapping[str, object] | None = None,
    ) -> InferencePipelineResult:
        started = time.perf_counter()
        source_image = quiet_imread(task.image_path, cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise RuntimeError(f"Failed to read image: {task.image_path}")

        image_stats = self._image_statistics(source_image)
        label_mask, confidence_mask = self._predict_image(source_image)
        monitoring_info = self._monitoring_info(
            source_image=source_image,
            label_mask=label_mask,
            confidence_mask=confidence_mask,
        )
        result_row = self._quantifier.quantify(
            label_mask,
            analysis_id=task.analysis_id,
            cell=task.cell,
            face=task.face,
            image_num=task.image_num,
        )
        if extra_result_fields:
            result_row.update(extra_result_fields)

        resolved_mask_path = Path(mask_path) if mask_path is not None else None
        if self._save_mask_images and resolved_mask_path is not None:
            self._quantifier.save_mask_image(label_mask, resolved_mask_path)

        monitoring_record = self._prediction_record(
            task=task,
            result_row=result_row,
            image_stats=image_stats,
            confidence_stats=self._confidence_statistics(confidence_mask),
            monitoring_info=monitoring_info,
            mask_path=resolved_mask_path if self._save_mask_images else None,
        )

        del label_mask
        del confidence_mask
        del source_image

        return InferencePipelineResult(
            task=task,
            result_row=result_row,
            monitoring_record=monitoring_record,
            mask_path=resolved_mask_path if self._save_mask_images else None,
            elapsed_s=time.perf_counter() - started,
        )

    def _predict_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        started = time.perf_counter()
        feature_batch = self._feature_extractor.extract(
            image,
            downsample_factor=self._inference.downsample_factor,
            feature_columns=self._inference.feature_columns,
        )
        extract_elapsed = time.perf_counter() - started
        predict_started = time.perf_counter()
        try:
            label_mask, confidence_mask = self._inference.predict_mask_with_confidence(feature_batch)
            predict_elapsed = time.perf_counter() - predict_started
            total_elapsed = time.perf_counter() - started
            if total_elapsed >= 5.0 or logger.isEnabledFor(logging.DEBUG):
                logger.info(
                    "InferencePipeline timing: model=%s type=%s family=%s feature_set=%s "
                    "image_shape=%s downsample=%.2f valid_pixels=%d features=%.2fs inference=%.2fs total=%.2fs",
                    self._inference.model_dir,
                    self._inference.model_type,
                    self._inference.model_family,
                    self._inference.feature_set,
                    image.shape[:2],
                    self._inference.downsample_factor,
                    int(feature_batch.features.shape[0]),
                    extract_elapsed,
                    predict_elapsed,
                    total_elapsed,
                )
            return label_mask, confidence_mask
        finally:
            del feature_batch

    def _valid_mask_for_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, 3] > 0
        return np.ones(image.shape[:2], dtype=bool)

    def _bgr_from_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3]
        return image

    def _image_statistics(self, image: np.ndarray) -> dict[str, float | None]:
        valid_mask = self._valid_mask_for_image(image)
        if not np.any(valid_mask):
            return {
                "mean_r": None,
                "mean_g": None,
                "mean_b": None,
                "mean_h": None,
                "mean_s": None,
                "mean_v": None,
                "mean_l": None,
                "mean_a": None,
                "mean_b_lab": None,
            }

        bgr = self._bgr_from_image(image)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        bgr_values = bgr[valid_mask].astype(np.float64)
        hsv_values = hsv[valid_mask].astype(np.float64)
        lab_values = lab[valid_mask].astype(np.float64)

        return {
            "mean_r": float(bgr_values[:, 2].mean()),
            "mean_g": float(bgr_values[:, 1].mean()),
            "mean_b": float(bgr_values[:, 0].mean()),
            "mean_h": float(hsv_values[:, 0].mean()),
            "mean_s": float(hsv_values[:, 1].mean()),
            "mean_v": float(hsv_values[:, 2].mean()),
            "mean_l": float(lab_values[:, 0].mean()),
            "mean_a": float(lab_values[:, 1].mean()),
            "mean_b_lab": float(lab_values[:, 2].mean()),
        }

    def _confidence_statistics(
        self,
        confidence_mask: np.ndarray | None,
    ) -> dict[str, float | None]:
        if confidence_mask is None:
            return {"mean_confidence": None, "low_confidence_ratio": None}

        valid_confidence = confidence_mask[np.isfinite(confidence_mask)]
        if valid_confidence.size == 0:
            return {"mean_confidence": None, "low_confidence_ratio": None}

        return {
            "mean_confidence": float(valid_confidence.mean()),
            "low_confidence_ratio": float(
                np.count_nonzero(valid_confidence < self._low_confidence_threshold)
                / valid_confidence.size
            ),
        }

    def _monitoring_info(
        self,
        *,
        source_image: np.ndarray,
        label_mask: np.ndarray,
        confidence_mask: np.ndarray | None,
    ) -> dict[str, float | int | str | None]:
        info: dict[str, float | int | str | None] = {
            "high_conf_bg_ratio": None,
            "raw_bg_l": None,
            "raw_bg_a": None,
            "raw_bg_b_lab": None,
            "baseline_bg_l": None,
            "baseline_bg_a": None,
            "baseline_bg_b_lab": None,
            "l_shift": None,
            "a_shift": None,
            "b_shift": None,
            "drift_score_before": None,
            "drift_score_after": None,
            "correction_applied": 0,
            "normalization_skipped_reason": None,
            "retraining_candidate": 0,
            "qc_warning": 0,
        }

        detector = self._drift_detector
        if detector is None:
            info["normalization_skipped_reason"] = "missing_baseline_stats"
            return info

        valid_mask = self._valid_mask_for_image(source_image)
        bgr = self._bgr_from_image(source_image)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        background_mask = self._high_confidence_background_mask(
            label_mask,
            confidence_mask,
        )
        drift = detector.evaluate(lab, valid_mask, background_mask)
        self._apply_drift_info(info, drift)
        info["normalization_skipped_reason"] = self._normalization_skip_reason(drift)
        if info["normalization_skipped_reason"] == "drift_review_required":
            info["qc_warning"] = 1
        return info

    def _high_confidence_background_mask(
        self,
        label_mask: np.ndarray,
        confidence_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        if confidence_mask is None:
            return None
        threshold = float(self._normalization_cfg.get("bg_confidence_threshold", 0.90))
        return (
            (label_mask == LABEL_BACKGROUND)
            & np.isfinite(confidence_mask)
            & (confidence_mask >= threshold)
        )

    def _apply_drift_info(
        self,
        info: dict[str, float | int | str | None],
        drift: BackgroundDrift,
    ) -> None:
        info.update(
            {
                "high_conf_bg_ratio": drift.high_conf_bg_ratio,
                "raw_bg_l": drift.raw_bg_l,
                "raw_bg_a": drift.raw_bg_a,
                "raw_bg_b_lab": drift.raw_bg_b_lab,
                "baseline_bg_l": drift.baseline_bg_l,
                "baseline_bg_a": drift.baseline_bg_a,
                "baseline_bg_b_lab": drift.baseline_bg_b_lab,
                "l_shift": drift.l_shift,
                "a_shift": drift.a_shift,
                "b_shift": drift.b_shift,
                "drift_score_before": drift.drift_score_before,
            }
        )

    def _normalization_skip_reason(self, drift: BackgroundDrift) -> str | None:
        if drift.skipped_reason is not None:
            return drift.skipped_reason
        score = drift.drift_score_before
        detector = self._drift_detector
        if score is None or detector is None:
            return None
        if score <= detector.drift_threshold:
            return "drift_below_threshold"
        return "drift_review_required"

    def _prediction_record(
        self,
        *,
        task: InferenceTask,
        result_row: dict,
        image_stats: dict[str, float | None],
        confidence_stats: dict[str, float | None],
        monitoring_info: dict[str, float | int | str | None],
        mask_path: Path | None,
    ) -> dict:
        record = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "analysis_id": task.analysis_id,
            "cell": str(task.cell),
            "face": task.face,
            "image_num": task.image_num,
            "image_name": task.image_path.name,
            "customer": task.customer or self._customer_for_analysis(task.analysis_id),
            "model_version": self._inference.model_version,
            "total_pixel": int(result_row["total_pixel"]),
            "background_ratio": float(result_row["background_ratio"]),
            "nonactive_ratio": float(result_row["nonactive_ratio"]),
            "li_salt_ratio": float(result_row["li_salt_ratio"]),
            "mask_path": str(mask_path) if mask_path is not None else None,
            "pass_image_path": str(task.image_path),
        }
        record.update(image_stats)
        record.update(confidence_stats)
        record.update(monitoring_info)
        low_conf = record.get("low_confidence_ratio")
        drift_after = record.get("drift_score_after")
        drift_before = record.get("drift_score_before")
        drift_score = drift_after if drift_after is not None else drift_before
        retraining_low_conf = float(
            self._normalization_cfg.get("retraining_low_confidence_threshold", 0.20)
        )
        if (
            isinstance(drift_score, (int, float))
            and isinstance(low_conf, (int, float))
            and drift_score > self._drift_detector.drift_threshold
            and low_conf > retraining_low_conf
        ):
            record["retraining_candidate"] = 1
        return record

    def _customer_for_analysis(self, analysis_id: str) -> str:
        if is_test_analysis_id(analysis_id):
            return "TEST"
        if len(analysis_id) < 2:
            return ""
        customer_code = analysis_id[1]
        return str(self._customer_codes.get(customer_code, customer_code))
