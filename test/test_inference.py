import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from modules.ml.inference import InferenceEngine


def write_config(path: Path) -> None:
    path.write_text(
        """
ml:
  model_path: models/model
  num_threads: 1
  downsample_factor: 1.0
  predict_chunk_rows: 2
  edge_hsv_range:
    lower: [0, 0, 0]
    upper: [179, 255, 255]
  nonactive_hsv_range:
    lower: [0, 0, 0]
    upper: [179, 255, 255]
  edge_margin_px: 1
""",
        encoding="utf-8",
    )


def test_inference_engine_loads_joblib_classifier(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    write_config(config_path)
    model_dir = tmp_path / "models" / "logistic_regression_1"
    model_dir.mkdir(parents=True)

    estimator = DummyClassifier(strategy="constant", constant=1)
    x_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([0, 1, 2], dtype=np.int32)
    estimator.fit(x_train, y_train)
    joblib.dump(estimator, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "sklearn_classifier",
                "model_version": model_dir.name,
                "model_family": "logistic_regression",
                "feature_set": "basic",
                "feature_columns": ["r"],
                "classes": ["b", "l", "n"],
            }
        ),
        encoding="utf-8",
    )

    engine = InferenceEngine(config_path, model_path=model_dir)
    predictions, confidence = engine.predict_codes_with_confidence(
        np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    )

    assert predictions.tolist() == [1, 1, 1]
    assert confidence is not None
