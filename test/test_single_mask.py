from pathlib import Path

import cv2
import numpy as np

from gui.single_mask_tab import SingleMaskTab
from modules.ml.single_mask import SingleMaskEngine
from modules.qc.crop_engine import CropEngine


CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'thresholds.yaml'


class _Quantifier:
    def build_mask_image(self, labels):
        return np.dstack([labels, labels, labels, np.full_like(labels, 255)])


def test_single_mask_processes_one_image_without_writing_outputs(tmp_path):
    source_path = tmp_path / 'sample.png'
    ok, encoded = cv2.imencode('.png', np.zeros((3, 4, 3), dtype=np.uint8))
    assert ok
    source_path.write_bytes(encoded.tobytes())

    engine = SingleMaskEngine.__new__(SingleMaskEngine)
    engine._normalization_enabled = False
    engine._crop_source = lambda path: np.zeros((2, 3, 4), dtype=np.uint8)
    engine._predict = lambda image: (
        np.ones(image.shape[:2], dtype=np.int16),
        None,
    )
    engine._quantifier = _Quantifier()

    result = engine.process_path(source_path)

    assert result.image_path == source_path
    assert result.label_mask.shape == (2, 3)
    assert result.mask_image.shape == (2, 3, 4)
    assert list(tmp_path.iterdir()) == [source_path]


def test_single_mask_normalizes_file_uri():
    path = SingleMaskTab._normalize_path('file:///C:/images/sample.png')
    assert path == Path('C:/images/sample.png')


def test_single_mask_crops_background_before_inference(tmp_path):
    source_path = tmp_path / 'raw.png'
    raw = np.full((400, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(raw, (100, 100), (700, 300), (20, 30, 40), thickness=-1)
    ok, encoded = cv2.imencode('.png', raw)
    assert ok
    source_path.write_bytes(encoded.tobytes())

    engine = SingleMaskEngine.__new__(SingleMaskEngine)
    engine._crop_engine = CropEngine(CONFIG_PATH)
    engine._color_normalizer = None

    cropped = engine._crop_source(source_path)

    assert cropped.shape[0] < raw.shape[0]
    assert cropped.shape[1] < raw.shape[1]
    assert cropped.shape[2] == 4
    assert np.count_nonzero(cropped[:, :, 3] == 0) > 0
    assert list(tmp_path.iterdir()) == [source_path]
