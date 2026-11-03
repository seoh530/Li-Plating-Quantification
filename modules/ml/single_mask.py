from dataclasses import dataclass
from pathlib import Path
import tempfile

import cv2
import numpy as np
import yaml

from modules.ml.background_normalizer import BackgroundLABNormalizer
from modules.ml.drift_detector import DriftDetector
from modules.ml.feature_extractor import FeatureExtractor
from modules.ml.inference import InferenceEngine, LABEL_BACKGROUND
from modules.ml.model_registry import resolve_model_path
from modules.ml.quantifier import Quantifier
from modules.qc.crop_engine import CropEngine
from modules.utils.image_io import quiet_imread


@dataclass(frozen=True)
class SingleMaskResult:
    "Result."
    image_path: Path
    label_mask: np.ndarray
    mask_image: np.ndarray


class SingleMaskEngine:
    def __init__(self, config_path) -> None:
        self._config_path = Path(config_path)
        with open(self._config_path, encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh) or {}
        model_dir = resolve_model_path('models/V2F')
        normal_cfg = dict(cfg.get('normalization', {}) or {})
        self._feature_extractor = FeatureExtractor(self._config_path)
        self._inference = InferenceEngine(self._config_path, model_path=model_dir)
        self._quantifier = Quantifier(self._config_path)
        self._crop_engine = CropEngine(self._config_path)
        self._color_normalizer = None
        try:
            from modules.qc.color_normalizer import ColorNormalizer
            self._color_normalizer = ColorNormalizer(self._config_path)
        except RuntimeError:
            pass
        self._normalization_cfg = normal_cfg
        self._normalization_enabled = bool(normal_cfg.get('enabled', False))
        self._drift_detector = DriftDetector(
            model_dir / 'baseline_stats.json',
            min_high_conf_bg_ratio=float(
                normal_cfg.get('min_high_conf_bg_ratio', 0.05)
            ),
            drift_threshold=float(normal_cfg.get('drift_threshold', 2.0)),
        )
        self._background_normalizer = BackgroundLABNormalizer(
            max_l_shift=float(normal_cfg.get('max_l_shift', 25)),
            max_a_shift=float(normal_cfg.get('max_a_shift', 5)),
            max_b_shift=float(normal_cfg.get('max_b_shift', 5)),
        )

    def process_path(self, image_path) -> SingleMaskResult:
        image_path = Path(image_path)
        source = self._crop_source(image_path)

        labels, confidence = self._predict(source)
        if self._should_normalize(source, labels, confidence):
            drift = self._measure_drift(source, labels, confidence)
            normalized = self._background_normalizer.normalize(
                source,
                l_shift=drift.l_shift,
                a_shift=drift.a_shift,
                b_shift=drift.b_shift,
            )
            if normalized.correction_applied:
                labels, _confidence = self._predict(normalized.image)

        return SingleMaskResult(
            image_path=image_path,
            label_mask=labels,
            mask_image=self._quantifier.build_mask_image(labels),
        )

    def _crop_source(self, image_path):
        with tempfile.TemporaryDirectory(prefix='single_mask_') as temp_dir:
            cropped_path = Path(temp_dir) / f'{image_path.stem}.png'
            postprocess = (
                self._color_normalizer.normalize
                if self._color_normalizer is not None
                else None
            )
            crop_result = self._crop_engine.process_image(
                image_path=image_path,
                customer_code='ETC',
                output_path=cropped_path,
                skip_size_validation=True,
                image_postprocess=postprocess,
                force_save_failed=True,
            )
            cropped = quiet_imread(cropped_path, cv2.IMREAD_UNCHANGED)
            if cropped is None:
                reasons = ' | '.join(crop_result.fail_reasons)
                detail = f': {reasons}' if reasons else ''
                raise RuntimeError(f'Background crop failed{detail}')
            return cropped

    def _predict(self, image):
        features = self._feature_extractor.extract(
            image,
            downsample_factor=self._inference.downsample_factor,
            feature_columns=self._inference.feature_columns,
        )
        return self._inference.predict_mask_with_confidence(features)

    def _should_normalize(self, image, labels, confidence) -> bool:
        if not self._normalization_enabled:
            return False
        drift = self._measure_drift(image, labels, confidence)
        return (
            drift.skipped_reason is None
            and drift.drift_score_before is not None
            and drift.drift_score_before > self._drift_detector.drift_threshold
            and drift.l_shift is not None
            and drift.a_shift is not None
            and drift.b_shift is not None
        )

    def _measure_drift(self, image, labels, confidence):
        valid = (
            image[:, :, 3] > 0
            if image.shape[2] == 4
            else np.ones(image.shape[:2], dtype=bool)
        )
        bgr = image[:, :, :3] if image.shape[2] == 4 else image
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        background = None
        if confidence is not None:
            threshold = float(
                self._normalization_cfg.get('bg_confidence_threshold', 0.90)
            )
            background = (
                (labels == LABEL_BACKGROUND)
                & np.isfinite(confidence)
                & (confidence >= threshold)
            )
        return self._drift_detector.evaluate(lab, valid, background)
