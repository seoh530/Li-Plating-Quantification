from pathlib import Path

import cv2
import numpy as np
import yaml

from modules.ml.inference import (
    LABEL_BACKGROUND,
    LABEL_EDGE,
    LABEL_INACTIVE,
    LABEL_LI_SALT,
)

MASK_COLORS = {
    LABEL_BACKGROUND: (0, 255, 0),
    LABEL_INACTIVE: (255, 0, 0),
    LABEL_EDGE: (128, 128, 128),
    LABEL_LI_SALT: (0, 0, 255),
}


class Quantifier:
    """Compute per-image class ratios from a label mask."""

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._png_compression_level = int(cfg["ml"].get("png_compression_level", 1))

    def quantify(
        self,
        label_mask: np.ndarray,
        *,
        analysis_id: str,
        cell: int,
        face: str,
        image_num: int,
    ) -> dict:
        valid_mask = label_mask >= 0
        total_pixel = int(valid_mask.sum())
        if total_pixel <= 0:
            raise RuntimeError("Quantifier received an empty valid-pixel mask.")

        counts = np.bincount(label_mask[valid_mask], minlength=LABEL_EDGE + 1)
        background_count = int(counts[LABEL_BACKGROUND])
        nonactive_count = int(counts[LABEL_INACTIVE])
        li_salt_count = int(counts[LABEL_LI_SALT])

        background_ratio = round(background_count / total_pixel, 2)
        nonactive_ratio = round(nonactive_count / total_pixel, 2)
        li_salt_ratio = round(li_salt_count / total_pixel, 2)

        return {
            "AnalysisID": analysis_id,
            "Cell": cell,
            "Face": face,
            "ImageNum": image_num,
            "total_pixel": total_pixel,
            "background_ratio": background_ratio,
            "nonactive_ratio": nonactive_ratio,
            "li_salt_ratio": li_salt_ratio,
        }

    def build_mask_image(self, label_mask: np.ndarray) -> np.ndarray:
        palette = np.zeros((LABEL_EDGE + 2, 4), dtype=np.uint8)
        palette[LABEL_BACKGROUND + 1] = (*MASK_COLORS[LABEL_BACKGROUND], 255)
        palette[LABEL_INACTIVE + 1] = (*MASK_COLORS[LABEL_INACTIVE], 255)
        palette[LABEL_EDGE + 1] = (*MASK_COLORS[LABEL_EDGE], 255)
        palette[LABEL_LI_SALT + 1] = (*MASK_COLORS[LABEL_LI_SALT], 255)
        encoded = np.clip(label_mask + 1, 0, LABEL_EDGE + 1).astype(
            np.uint8,
            copy=False,
        )
        return palette[encoded]

    def save_mask_image(self, label_mask: np.ndarray, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_image = self.build_mask_image(label_mask)
        success = cv2.imwrite(
            str(output_path),
            mask_image,
            [cv2.IMWRITE_PNG_COMPRESSION, self._png_compression_level],
        )
        if not success:
            raise RuntimeError(f"Failed to write mask image: {output_path}")
