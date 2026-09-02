from pathlib import Path

import numpy as np
import pytest

from modules.ml.inference import LABEL_BACKGROUND, LABEL_INACTIVE, LABEL_LI_SALT
from modules.ml.quantifier import Quantifier


CONFIG_PATH = Path(__file__).parent.parent / "config" / "thresholds.yaml"


def test_mask_image_uses_transparent_invalid_pixels() -> None:
    quantifier = Quantifier(CONFIG_PATH)
    label_mask = np.array(
        [
            [-1, LABEL_BACKGROUND],
            [LABEL_LI_SALT, LABEL_INACTIVE],
        ],
        dtype=np.int16,
    )

    mask_image = quantifier.build_mask_image(label_mask)

    assert mask_image.shape == (2, 2, 4)
    assert mask_image[0, 0, 3] == 0
    assert np.all(mask_image[label_mask >= 0, 3] == 255)


def test_quantify_preserves_small_li_salt_ratio() -> None:
    quantifier = Quantifier(CONFIG_PATH)
    label_mask = np.full((10, 100), LABEL_BACKGROUND, dtype=np.int16)
    label_mask[0, 0] = LABEL_LI_SALT

    result = quantifier.quantify(
        label_mask,
        analysis_id="AHG00001",
        cell=1,
        face="FRONT",
        image_num=1,
    )

    assert result["li_salt_ratio"] == pytest.approx(0.001)
