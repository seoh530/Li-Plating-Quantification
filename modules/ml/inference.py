import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import lightgbm as lgb
import lightgbm.basic as lgb_basic
import numpy as np
import yaml

from modules.ml.feature_extractor import FeatureBatch, FeatureExtractor

logger = logging.getLogger(__name__)


LABEL_BACKGROUND = 0
LABEL_LI_SALT = 1
LABEL_INACTIVE = 2
LABEL_EDGE = 3

LABEL_CODE_TO_NAME = {
    LABEL_BACKGROUND: "background",
    LABEL_LI_SALT: "li-salt",
    LABEL_INACTIVE: "inactive",
    LABEL_EDGE: "edge",
}

TRAINING_LABEL_TO_CODE = {
    "b": LABEL_BACKGROUND,
    "l": LABEL_LI_SALT,
    "n": LABEL_INACTIVE,
}


@dataclass(frozen=True)
class EdgePostprocessDebug:
    before_edge_mask: np.ndarray
    after_edge_mask: np.ndarray
    fixed_edge_count: int
    dynamic_candidate_count: int
    dynamic_connected_count: int
    dynamic_added_count: int
    nonactive_override_count: int
    background_reference_count: int
    fixed_edge_limit: float
    dynamic_search_limit: float


class InferenceEngine:
    """Load a trained pixel classifier and classify QC-ready pixels."""

    def __init__(self, config_path: str | Path, model_path: str | Path | None = None) -> None:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        ml_cfg = cfg["ml"]
        self._model_dir = Path(model_path) if model_path is not None else Path(ml_cfg["model_path"])
        self._num_threads = int(ml_cfg.get("num_threads", -1))
        self._downsample_factor = float(ml_cfg.get("downsample_factor", 1.0))
        self._predict_chunk_rows = int(ml_cfg.get("predict_chunk_rows", 250000))

        metadata_file = self._model_dir / "metadata.json"
        if not metadata_file.exists():
            raise RuntimeError(f"Model metadata file not found: {metadata_file}")

        self._metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        self._model_type = str(self._metadata.get("model_type", "lightgbm_booster"))
        self._feature_columns = tuple(self._metadata["feature_columns"])
        self._feature_set = str(
            self._metadata.get("feature_set")
            or ("enhanced" if self._model_dir.name.startswith("logical_lightgbm_") else "legacy")
        )
        if self._model_dir.name.startswith("logical_lightgbm_") and self._feature_set != "enhanced":
            raise RuntimeError(
                f"Logical model {self._model_dir.name} must declare feature_set=enhanced in metadata.json"
            )
        self._classes = tuple(self._metadata["classes"])
        self._best_iteration = int(self._metadata.get("best_iteration", -1) or -1)
        self._logged_feature_rebuild = False
        self._booster = None
        self._predictor = None
        self._estimator = None
        if self._model_type == "lightgbm_booster":
            model_file = self._model_dir / "model.txt"
            if not model_file.exists():
                raise RuntimeError(f"LightGBM model file not found: {model_file}")
            self._booster = lgb.Booster(model_file=str(model_file))
            self._predictor = lgb_basic._InnerPredictor.from_booster(
                booster=self._booster,
                pred_parameter={"num_threads": self._num_threads},
            )
        elif self._model_type == "sklearn_classifier":
            import joblib

            model_file = self._model_dir / "model.joblib"
            if not model_file.exists():
                raise RuntimeError(f"Model joblib file not found: {model_file}")
            self._estimator = joblib.load(model_file)
        else:
            raise RuntimeError(f"Unsupported model_type in metadata.json: {self._model_type}")

        edge_cfg = ml_cfg["edge_hsv_range"]
        nonactive_cfg = ml_cfg["nonactive_hsv_range"]
        self._edge_hsv_lower = tuple(int(v) for v in edge_cfg["lower"])
        self._edge_hsv_upper = tuple(int(v) for v in edge_cfg["upper"])
        self._nonactive_hsv_lower = tuple(int(v) for v in nonactive_cfg["lower"])
        self._nonactive_hsv_upper = tuple(int(v) for v in nonactive_cfg["upper"])
        self._edge_margin_px = int(ml_cfg["edge_margin_px"])
        self._dynamic_edge_search_px = float(
            ml_cfg.get("dynamic_edge_search_px", max(3.0, float(self._edge_margin_px) * 2.0))
        )
        self._dynamic_edge_bg_confidence_threshold = float(
            ml_cfg.get("dynamic_edge_bg_confidence_threshold", 0.90)
        )
        self._dynamic_edge_preserve_confidence_threshold = float(
            ml_cfg.get("dynamic_edge_preserve_confidence_threshold", 0.90)
        )
        self._dynamic_edge_bg_min_pixels = int(ml_cfg.get("dynamic_edge_bg_min_pixels", 64))
        bg_padding = ml_cfg.get("dynamic_edge_background_hsv_padding", [8, 25, 25])
        self._dynamic_edge_bg_hsv_padding = tuple(float(v) for v in bg_padding)
        if len(self._dynamic_edge_bg_hsv_padding) != 3:
            self._dynamic_edge_bg_hsv_padding = (8.0, 25.0, 25.0)
        self._last_edge_postprocess_debug: EdgePostprocessDebug | None = None

        logger.info(
            "InferenceEngine loaded model from %s (model_type=%s, feature_set=%s, num_threads=%d, edge_margin_px=%d, downsample_factor=%.2f, best_iteration=%d, predict_chunk_rows=%d)",
            self._model_dir,
            self._model_type,
            self._feature_set,
            self._num_threads,
            self._edge_margin_px,
            self._downsample_factor,
            self._best_iteration,
            self._predict_chunk_rows,
        )

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def downsample_factor(self) -> float:
        return self._downsample_factor

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self._feature_columns

    @property
    def feature_set(self) -> str:
        return self._feature_set

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def model_family(self) -> str:
        return str(self._metadata.get("model_family") or self._model_type)

    @property
    def model_version(self) -> str | None:
        value = (
            self._metadata.get("model_version")
            or self._metadata.get("version")
            or self._metadata.get("training_run_id")
        )
        return str(value) if value else None

    @property
    def last_edge_postprocess_debug(self) -> EdgePostprocessDebug | None:
        return self._last_edge_postprocess_debug

    def predict_codes(self, features: np.ndarray) -> np.ndarray:
        """Classify all valid pixels."""
        predicted_codes, _confidence = self.predict_codes_with_confidence(features)
        return predicted_codes

    def _predict_sklearn_scores(self, features: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("Sklearn estimator is not loaded.")
        if hasattr(self._estimator, "predict_proba"):
            scores = self._estimator.predict_proba(features)
            if isinstance(scores, list):
                scores = np.column_stack([score[:, 1] for score in scores])
            return np.asarray(scores)
        return np.asarray(self._estimator.predict(features)).reshape(-1)

    def _predict_scores(self, features: np.ndarray) -> np.ndarray:
        if self._model_type == "lightgbm_booster":
            if self._predictor is None:
                raise RuntimeError("LightGBM predictor is not loaded.")
            return self._predictor.predict(
                features,
                num_iteration=self._best_iteration if self._best_iteration > 0 else -1,
                validate_features=False,
            )
        return self._predict_sklearn_scores(features)

    def predict_codes_with_confidence(
        self,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Classify valid pixels and return confidence when raw scores are probabilistic."""
        if features.size == 0:
            return np.empty((0,), dtype=np.uint8), None

        if features.ndim != 2:
            raise RuntimeError(
                f"Expected 2D feature matrix for inference, got shape={features.shape}"
            )

        expected_feature_count = len(self._feature_columns)
        if features.shape[1] != expected_feature_count:
            raise RuntimeError(
                "Inference feature count mismatch: "
                f"model expects {expected_feature_count}, got {features.shape[1]}"
        )

        features = np.ascontiguousarray(features, dtype=np.float32)

        if (
            self._predict_chunk_rows > 0
            and features.shape[0] > self._predict_chunk_rows
        ):
            predicted_codes = np.empty((features.shape[0],), dtype=np.uint8)
            confidence = np.empty((features.shape[0],), dtype=np.float32)
            has_confidence = True
            for start in range(0, features.shape[0], self._predict_chunk_rows):
                end = min(start + self._predict_chunk_rows, features.shape[0])
                scores = self._predict_scores(features[start:end])
                if scores.ndim == 1:
                    predicted_codes[start:end] = scores.astype(np.uint8, copy=False)
                    has_confidence = False
                else:
                    predicted_codes[start:end] = np.argmax(scores, axis=1).astype(
                        np.uint8,
                        copy=False,
                    )
                    confidence[start:end] = np.max(scores, axis=1).astype(
                        np.float32,
                        copy=False,
                    )
            return predicted_codes, confidence if has_confidence else None

        scores = self._predict_scores(features)
        if scores.ndim == 1:
            return scores.astype(np.uint8), None
        return (
            np.argmax(scores, axis=1).astype(np.uint8),
            np.max(scores, axis=1).astype(np.float32),
        )

    def _get_available_columns(self, feature_batch: FeatureBatch) -> tuple[str, ...]:
        columns = getattr(feature_batch, "feature_columns", None)
        if columns is not None:
            return tuple(columns)

        feature_dim = int(feature_batch.features.shape[1])
        if feature_dim == len(FeatureExtractor.FEATURE_COLUMNS):
            logger.warning(
                "FeatureBatch is missing feature_columns; assuming current extractor columns."
            )
            return FeatureExtractor.FEATURE_COLUMNS
        if feature_dim == 6:
            logger.warning(
                "FeatureBatch is missing feature_columns; assuming legacy RGB/HSV columns."
            )
            return ("r", "g", "b", "h", "s", "v")

        raise RuntimeError(
            "FeatureBatch object has no feature_columns attribute and the feature "
            f"width ({feature_dim}) does not match a known extractor layout."
        )

    def _rebuild_current_features(
        self,
        feature_batch: FeatureBatch,
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        valid_mask = getattr(feature_batch, "valid_mask", None)
        bgr = getattr(feature_batch, "bgr", None)
        hsv = getattr(feature_batch, "hsv", None)
        if valid_mask is None or bgr is None or hsv is None:
            raise RuntimeError(
                "Cannot rebuild feature columns because FeatureBatch is missing "
                "valid_mask, bgr, or hsv."
            )

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        yy, xx = np.indices(bgr.shape[:2], dtype=np.float32)
        rebuilt = np.column_stack(
            [
                bgr[:, :, 2][valid_mask],
                bgr[:, :, 1][valid_mask],
                bgr[:, :, 0][valid_mask],
                hsv[:, :, 0][valid_mask],
                hsv[:, :, 1][valid_mask],
                hsv[:, :, 2][valid_mask],
                xx[valid_mask],
                yy[valid_mask],
                lab[:, :, 0][valid_mask],
                lab[:, :, 1][valid_mask],
                lab[:, :, 2][valid_mask],
            ]
        ).astype(np.float32, copy=False)
        return rebuilt, FeatureExtractor.FEATURE_COLUMNS

    def _select_model_features(self, feature_batch: FeatureBatch) -> np.ndarray:
        features = feature_batch.features
        available_columns = self._get_available_columns(feature_batch)
        missing = [
            column for column in self._feature_columns if column not in available_columns
        ]
        if missing:
            if not self._logged_feature_rebuild:
                logger.warning(
                    "Inference input batch reported columns %s, but model expects %s. "
                    "Rebuilding the current 11-column feature set from image data.",
                    available_columns,
                    self._feature_columns,
                )
                self._logged_feature_rebuild = True
            features, available_columns = self._rebuild_current_features(feature_batch)
            missing = [
                column
                for column in self._feature_columns
                if column not in available_columns
            ]
        if missing:
            raise RuntimeError(
                "FeatureExtractor output is missing model feature columns: "
                f"{missing}"
            )

        if available_columns == self._feature_columns:
            return features

        column_to_index = {
            column: idx for idx, column in enumerate(available_columns)
        }
        indices = [column_to_index[column] for column in self._feature_columns]
        return features[:, indices]

    def predict_mask(self, feature_batch: FeatureBatch) -> np.ndarray:
        """Run model inference then edge post-processing. Returns finalized label mask."""
        label_mask, _confidence_mask = self.predict_mask_with_confidence(feature_batch)
        return label_mask

    def predict_mask_with_confidence(
        self,
        feature_batch: FeatureBatch,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Run model inference and return finalized label mask plus optional confidence map."""
        started = time.perf_counter()
        last = started
        label_mask = np.full(feature_batch.image_shape, fill_value=-1, dtype=np.int16)
        model_features = self._select_model_features(feature_batch)
        select_elapsed = time.perf_counter() - last
        last = time.perf_counter()
        predicted_codes, confidence = self.predict_codes_with_confidence(model_features)
        predict_elapsed = time.perf_counter() - last
        last = time.perf_counter()
        label_mask[feature_batch.valid_mask] = predicted_codes.astype(np.int16, copy=False)
        confidence_mask = None
        if confidence is not None:
            confidence_mask = np.full(feature_batch.image_shape, np.nan, dtype=np.float32)
            confidence_mask[feature_batch.valid_mask] = confidence
        assign_elapsed = time.perf_counter() - last
        last = time.perf_counter()
        label_mask = self.apply_edge_postprocess(
            label_mask,
            feature_batch.hsv,
            confidence_mask=confidence_mask,
        )
        edge_elapsed = time.perf_counter() - last
        last = time.perf_counter()
        resize_elapsed = 0.0
        if feature_batch.original_image_shape != feature_batch.image_shape:
            target_h, target_w = feature_batch.original_image_shape
            encoded = (label_mask + 1).astype(np.uint8, copy=False)
            upsampled = cv2.resize(
                encoded,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int16)
            label_mask = upsampled - 1
            label_mask[~feature_batch.original_valid_mask] = -1
            if confidence_mask is not None:
                confidence_mask = cv2.resize(
                    confidence_mask,
                    (target_w, target_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                confidence_mask[~feature_batch.original_valid_mask] = np.nan
            resize_elapsed = time.perf_counter() - last
        total_elapsed = time.perf_counter() - started
        if total_elapsed >= 5.0 or logger.isEnabledFor(logging.DEBUG):
            logger.info(
                "Inference timing: model=%s type=%s family=%s feature_set=%s pixels=%d "
                "select=%.2fs predict=%.2fs assign=%.2fs edge=%.2fs resize=%.2fs total=%.2fs",
                self._model_dir,
                self._model_type,
                self.model_family,
                self._feature_set,
                int(model_features.shape[0]),
                select_elapsed,
                predict_elapsed,
                assign_elapsed,
                edge_elapsed,
                resize_elapsed,
                total_elapsed,
            )
        return label_mask, confidence_mask

    def apply_edge_postprocess(
        self,
        label_mask: np.ndarray,
        hsv_image: np.ndarray,
    ) -> np.ndarray:
        """Apply geometry-first edge handling and near-edge nonactive overrides."""
        result = label_mask.copy()
        valid = label_mask >= 0
        if not np.any(valid):
            logger.warning("apply_edge_postprocess: no valid pixels found, skipping.")
            return result

        nonactive_lower = np.array(self._nonactive_hsv_lower, dtype=np.uint8)
        nonactive_upper = np.array(self._nonactive_hsv_upper, dtype=np.uint8)
        nonactive_hsv_match = cv2.inRange(
            hsv_image, nonactive_lower, nonactive_upper
        ).astype(bool)

        valid_u8 = valid.astype(np.uint8)
        distance_from_outer = cv2.distanceTransform(valid_u8, cv2.DIST_L2, 3)

        search_limit = max(3.0, float(self._edge_margin_px) * 2.0)
        near_outer = valid & (distance_from_outer <= search_limit)

        inactive_candidate = near_outer & nonactive_hsv_match
        if inactive_candidate.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            inactive_candidate = cv2.morphologyEx(
                inactive_candidate.astype(np.uint8),
                cv2.MORPH_CLOSE,
                kernel,
            ).astype(bool)
            inactive_candidate &= near_outer

        min_candidate_px = max(32, int(valid.sum()) // 2000)
        candidate_count = int(inactive_candidate.sum())
        if candidate_count >= min_candidate_px:
            candidate_distance = distance_from_outer[inactive_candidate]
            inactive_start = float(np.percentile(candidate_distance, 20))
            inactive_end = float(np.percentile(candidate_distance, 90))
            edge_limit = max(1.0, inactive_start - 1.0)
            inactive_limit = min(
                search_limit,
                max(edge_limit + 1.0, inactive_end + 1.0),
            )
        else:
            edge_limit = max(1.0, min(float(self._edge_margin_px), search_limit))
            inactive_limit = min(
                search_limit,
                edge_limit + max(1.0, float(self._edge_margin_px) * 0.75),
            )

        edge_band = valid & (distance_from_outer <= edge_limit)
        inactive_zone = (
            valid
            & (distance_from_outer > edge_limit)
            & (distance_from_outer <= inactive_limit)
        )

        result[edge_band] = LABEL_EDGE

        remaining_li_salt = (result == LABEL_LI_SALT) & inactive_zone
        nonactive_override = remaining_li_salt & nonactive_hsv_match
        result[nonactive_override] = LABEL_INACTIVE

        logger.debug(
            "apply_edge_postprocess: edge_px=%d inactive_zone_px=%d nonactive_candidate_px=%d nonactive_override_px=%d edge_limit=%.2f inactive_limit=%.2f",
            int(np.count_nonzero(result == LABEL_EDGE)),
            int(inactive_zone.sum()),
            candidate_count,
            int(nonactive_override.sum()),
            edge_limit,
            inactive_limit,
        )
        return result


