import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureBatch:
    """Vectorized per-pixel features for one QC-ready image."""

    features: np.ndarray
    feature_columns: tuple[str, ...]
    valid_mask: np.ndarray
    image_shape: tuple[int, int]
    original_valid_mask: np.ndarray
    original_image_shape: tuple[int, int]
    bgr: np.ndarray
    hsv: np.ndarray


class FeatureExtractor:
    """
    Build per-pixel features from a QC-ready cropped cell image.

    All coordinates, local residuals, robust statistics, and edge distances are
    computed after crop/preprocessing. Transparent pixels (alpha == 0) are
    excluded from the feature matrix and from robust image statistics.
    """

    LEGACY_FEATURE_COLUMNS = (
        "r",
        "g",
        "b",
        "h",
        "s",
        "v",
        "x",
        "y",
        "l_lab",
        "a_lab",
        "b_lab",
    )
    BASELINE_FEATURE_COLUMNS = (
        "r",
        "g",
        "b",
        "h",
        "s",
        "v",
        "l_lab",
        "a_lab",
        "b_lab",
        "x_norm",
        "y_norm",
    )
    ENHANCED_FEATURE_COLUMNS = (
        *BASELINE_FEATURE_COLUMNS,
        "l_res",
        "a_res",
        "b_res",
        "s_res",
        "v_res",
        "l_robust",
        "a_robust",
        "b_robust",
        "s_robust",
        "v_robust",
        "distance_to_edge_px",
        "distance_to_edge_mm",
        "edge_band_5mm",
    )

    # Preserve the historical default used by existing callers and old metadata.
    FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS
    ALL_FEATURE_COLUMNS = tuple(
        dict.fromkeys(
            (*LEGACY_FEATURE_COLUMNS, *BASELINE_FEATURE_COLUMNS, *ENHANCED_FEATURE_COLUMNS)
        )
    )

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        local_residual_sigma_px: float | None = None,
        robust_eps: float | None = None,
        mm_per_pixel: float | None = None,
        edge_band_mm: float | None = None,
        feature_config: dict | None = None,
    ) -> None:
        cfg = self._load_feature_config(config_path)
        # Persisted training parameters take precedence over mutable UI config.
        cfg.update(feature_config or {})
        self.local_residual_sigma_px = float(
            local_residual_sigma_px
            if local_residual_sigma_px is not None
            else cfg.get("local_residual_sigma_px", 50.0)
        )
        self.robust_eps = float(
            robust_eps if robust_eps is not None else cfg.get("robust_eps", 1.0e-6)
        )
        self.mm_per_pixel = float(
            mm_per_pixel if mm_per_pixel is not None else cfg.get("mm_per_pixel", 1.0)
        )
        self.edge_band_mm = float(
            edge_band_mm if edge_band_mm is not None else cfg.get("edge_band_mm", 5.0)
        )

    @staticmethod
    def _load_feature_config(config_path: str | Path | None) -> dict:
        if config_path is None:
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except FileNotFoundError:
            logger.warning("FeatureExtractor config not found: %s", config_path)
            return {}
        return dict(cfg.get("ml", {}).get("feature_extraction", {}) or {})

    @classmethod
    def columns_for_feature_set(cls, feature_set: str) -> tuple[str, ...]:
        normalized = str(feature_set or "legacy").lower()
        if normalized == "legacy":
            return cls.LEGACY_FEATURE_COLUMNS
        if normalized == "baseline":
            return cls.BASELINE_FEATURE_COLUMNS
        if normalized == "enhanced":
            return cls.ENHANCED_FEATURE_COLUMNS
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    def extract_from_path(
        self,
        image_path: str | Path,
        downsample_factor: float = 1.0,
        feature_columns: tuple[str, ...] | None = None,
    ) -> FeatureBatch:
        image_path = Path(image_path)
        from modules.utils.image_io import quiet_imread

        image = quiet_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        return self.extract(
            image,
            downsample_factor=downsample_factor,
            feature_columns=feature_columns,
        )

    def extract(
        self,
        image: np.ndarray,
        downsample_factor: float = 1.0,
        feature_columns: tuple[str, ...] | None = None,
    ) -> FeatureBatch:
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("FeatureExtractor expects an HxWx3 or HxWx4 uint8 image.")

        requested_columns = tuple(feature_columns or self.FEATURE_COLUMNS)
        unsupported = [
            column for column in requested_columns if column not in self.ALL_FEATURE_COLUMNS
        ]
        if unsupported:
            raise ValueError(f"Unsupported feature columns requested: {unsupported}")

        original_shape = image.shape[:2]
        if image.shape[2] == 4:
            bgr = image[:, :, :3]
            original_valid_mask = image[:, :, 3] > 0
        else:
            bgr = image
            original_valid_mask = np.ones(image.shape[:2], dtype=bool)

        factor = float(downsample_factor)
        if not 0.0 < factor <= 1.0:
            factor = 1.0

        if factor < 1.0:
            new_w = max(1, int(round(bgr.shape[1] * factor)))
            new_h = max(1, int(round(bgr.shape[0] * factor)))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            valid_mask = cv2.resize(
                original_valid_mask.astype(np.uint8),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            valid_mask = original_valid_mask

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        valid_count = int(valid_mask.sum())
        features = np.empty((valid_count, len(requested_columns)), dtype=np.float32)

        flat_indices = np.flatnonzero(valid_mask)
        height, width = bgr.shape[:2]
        valid_y = flat_indices // width
        valid_x = flat_indices - (valid_y * width)
        valid_x_f = valid_x.astype(np.float32, copy=False)
        valid_y_f = valid_y.astype(np.float32, copy=False)
        x_norm = valid_x_f / max(float(width - 1), 1.0)
        y_norm = valid_y_f / max(float(height - 1), 1.0)

        bgr_pixels = bgr.reshape(-1, 3)[flat_indices]
        hsv_pixels = hsv.reshape(-1, 3)[flat_indices]
        lab_pixels = lab.reshape(-1, 3)[flat_indices]

        residual_maps: dict[str, np.ndarray] | None = None
        robust_maps: dict[str, np.ndarray] | None = None
        edge_maps: dict[str, np.ndarray] | None = None

        needs_residual = any(column.endswith("_res") for column in requested_columns)
        needs_robust = any(column.endswith("_robust") for column in requested_columns)
        needs_edge = any(column.startswith("distance_to_edge") or column.startswith("edge_band") for column in requested_columns)

        if needs_residual:
            residual_maps = self._build_residual_maps(lab, hsv, valid_mask)
        if needs_robust:
            robust_maps = self._build_robust_maps(lab, hsv, valid_mask)
        if needs_edge:
            edge_maps = self._build_edge_maps(height, width)

        for idx, column in enumerate(requested_columns):
            if column == "r":
                values = bgr_pixels[:, 2]
            elif column == "g":
                values = bgr_pixels[:, 1]
            elif column == "b":
                values = bgr_pixels[:, 0]
            elif column == "h":
                values = hsv_pixels[:, 0]
            elif column == "s":
                values = hsv_pixels[:, 1]
            elif column == "v":
                values = hsv_pixels[:, 2]
            elif column == "x":
                values = valid_x
            elif column == "y":
                values = valid_y
            elif column == "x_norm":
                values = x_norm
            elif column == "y_norm":
                values = y_norm
            elif column in {"l_lab", "a_lab", "b_lab"}:
                lab_index = {"l_lab": 0, "a_lab": 1, "b_lab": 2}[column]
                values = lab_pixels[:, lab_index]
            elif column in {"l_res", "a_res", "b_res", "s_res", "v_res"}:
                assert residual_maps is not None
                values = residual_maps[column].reshape(-1)[flat_indices]
            elif column in {"l_robust", "a_robust", "b_robust", "s_robust", "v_robust"}:
                assert robust_maps is not None
                values = robust_maps[column].reshape(-1)[flat_indices]
            elif column in {"distance_to_edge_px", "distance_to_edge_mm", "edge_band_5mm"}:
                assert edge_maps is not None
                values = edge_maps[column].reshape(-1)[flat_indices]
            else:  # pragma: no cover - guarded by unsupported check above.
                raise ValueError(f"Unsupported feature column requested: {column}")
            features[:, idx] = values

        logger.debug(
            "FeatureExtractor: shape=%s valid_pixels=%d feature_dim=%d columns=%s",
            bgr.shape[:2],
            int(valid_mask.sum()),
            features.shape[1] if features.ndim == 2 else 0,
            requested_columns,
        )
        return FeatureBatch(
            features=features,
            feature_columns=requested_columns,
            valid_mask=valid_mask,
            image_shape=bgr.shape[:2],
            original_valid_mask=original_valid_mask,
            original_image_shape=original_shape,
            bgr=bgr,
            hsv=hsv,
        )

    def _build_residual_maps(
        self,
        lab: np.ndarray,
        hsv: np.ndarray,
        valid_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        sigma = max(float(self.local_residual_sigma_px), 0.1)
        return {
            "l_res": self._residual(lab[:, :, 0], valid_mask, sigma),
            "a_res": self._residual(lab[:, :, 1], valid_mask, sigma),
            "b_res": self._residual(lab[:, :, 2], valid_mask, sigma),
            "s_res": self._residual(hsv[:, :, 1], valid_mask, sigma),
            "v_res": self._residual(hsv[:, :, 2], valid_mask, sigma),
        }

    def _residual(
        self,
        channel: np.ndarray,
        valid_mask: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        channel_f = channel.astype(np.float32, copy=False)
        valid_f = valid_mask.astype(np.float32, copy=False)
        blurred_value = cv2.GaussianBlur(channel_f * valid_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        blurred_weight = cv2.GaussianBlur(valid_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
        background = np.divide(
            blurred_value,
            np.maximum(blurred_weight, self.robust_eps),
            out=np.zeros_like(channel_f, dtype=np.float32),
            where=blurred_weight > 0,
        )
        return channel_f - background

    def _build_robust_maps(
        self,
        lab: np.ndarray,
        hsv: np.ndarray,
        valid_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "l_robust": self._robust(lab[:, :, 0], valid_mask),
            "a_robust": self._robust(lab[:, :, 1], valid_mask),
            "b_robust": self._robust(lab[:, :, 2], valid_mask),
            "s_robust": self._robust(hsv[:, :, 1], valid_mask),
            "v_robust": self._robust(hsv[:, :, 2], valid_mask),
        }

    def _robust(self, channel: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        channel_f = channel.astype(np.float32, copy=False)
        values = channel_f[valid_mask]
        if values.size == 0:
            return np.zeros_like(channel_f, dtype=np.float32)
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        iqr = float(q75 - q25)
        return (channel_f - median) / (iqr + self.robust_eps)

    def _build_edge_maps(self, height: int, width: int) -> dict[str, np.ndarray]:
        yy, xx = np.indices((height, width), dtype=np.float32)
        distance_px = np.minimum.reduce(
            [
                xx,
                np.maximum(float(width - 1), 0.0) - xx,
                yy,
                np.maximum(float(height - 1), 0.0) - yy,
            ]
        ).astype(np.float32, copy=False)
        distance_mm = distance_px * float(self.mm_per_pixel)
        return {
            "distance_to_edge_px": distance_px,
            "distance_to_edge_mm": distance_mm.astype(np.float32, copy=False),
            "edge_band_5mm": (distance_mm <= float(self.edge_band_mm)).astype(np.float32),
        }

    def select_point_features(
        self,
        feature_batch: FeatureBatch,
        points: Iterable[tuple[int, int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return feature rows and validity mask for crop-image pixel coordinates."""
        points_array = np.asarray(list(points), dtype=np.int64)
        if points_array.size == 0:
            return np.empty((0, feature_batch.features.shape[1]), dtype=np.float32), np.empty((0,), dtype=bool)
        if points_array.ndim != 2 or points_array.shape[1] != 2:
            raise ValueError("points must be an iterable of (x, y) pairs")

        height, width = feature_batch.image_shape
        lookup = np.full((height, width), -1, dtype=np.int64)
        flat_indices = np.flatnonzero(feature_batch.valid_mask)
        ys = flat_indices // width
        xs = flat_indices - ys * width
        lookup[ys, xs] = np.arange(flat_indices.size, dtype=np.int64)

        x = points_array[:, 0]
        y = points_array[:, 1]
        in_bounds = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        selected = np.full(points_array.shape[0], -1, dtype=np.int64)
        selected[in_bounds] = lookup[y[in_bounds], x[in_bounds]]
        valid = selected >= 0
        output = np.full((points_array.shape[0], feature_batch.features.shape[1]), np.nan, dtype=np.float32)
        output[valid] = feature_batch.features[selected[valid]]
        return output, valid
