import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import train_lightGBM as common
from train import training_utils as train_utils

logger = logging.getLogger(__name__)


MODEL_SPECS = {
    "logistic_regression": {
        "prefix": "logistic_regression",
        "description": "Train a Logistic Regression pixel classifier.",
    },
    "random_forest": {
        "prefix": "random_forest",
        "description": "Train a Random Forest pixel classifier.",
    },
    "xgboost": {
        "prefix": "xgboost",
        "description": "Train an XGBoost pixel classifier.",
    },
    "catboost": {
        "prefix": "catboost",
        "description": "Train a CatBoost pixel classifier.",
    },
}


def parse_args(model_kind: str) -> argparse.Namespace:
    spec = MODEL_SPECS[model_kind]
    parser = argparse.ArgumentParser(description=spec["description"])
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV files or directories containing labeled CSV files. Defaults to data/labeled/.",
    )
    parser.add_argument("--config", default="config/thresholds.yaml")
    parser.add_argument("--feature-set", choices=("basic", "enhanced"), default=None)
    parser.add_argument("--csv-selection", choices=("all", "custom"), default=None)
    parser.add_argument("--csv-indices", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--test-dir",
        default=str(common.DEFAULT_TEST_DIR),
        help="Directory of test images to mask after training. Defaults to data/test.",
    )
    parser.add_argument(
        "--skip-test-masks",
        action="store_true",
        help="Train only; do not run data/test masking after saving the model.",
    )
    return parser.parse_args()


def resolve_model_dir_for_kind(
    cfg: dict,
    output_dir: str | None,
    model_name: str | None,
    model_kind: str,
) -> Path:
    if output_dir and model_name:
        raise RuntimeError("--output-dir and --model-name cannot be used together.")
    if output_dir:
        path = Path(output_dir)
        return path if path.is_absolute() else common.PROJECT_ROOT / path

    models_root, _configured_name = common.models_root_from_config(cfg)
    if model_name:
        model_dir = models_root / common.validate_model_name(model_name)
        if model_dir.exists():
            raise RuntimeError(f"Model output folder already exists: {model_dir}")
        return model_dir
    return common.next_versioned_model_dir(models_root, MODEL_SPECS[model_kind]["prefix"])


def build_estimator(model_kind: str, seed: int, num_threads: int):
    if model_kind == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=num_threads if num_threads > 0 else None,
                random_state=seed,
                multi_class="auto",
            ),
        )
    if model_kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=num_threads,
            random_state=seed,
        )
    if model_kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed. Run: pip install xgboost") from exc
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=num_threads,
            random_state=seed,
        )
    if model_kind == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("catboost is not installed. Run: pip install catboost") from exc
        return CatBoostClassifier(
            loss_function="MultiClass",
            iterations=350,
            depth=6,
            learning_rate=0.05,
            random_seed=seed,
            thread_count=num_threads,
            verbose=False,
            allow_writing_files=False,
        )
    raise RuntimeError(f"Unsupported model kind: {model_kind}")


def predict_codes(estimator, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        scores = estimator.predict_proba(features)
        if isinstance(scores, list):
            scores = np.column_stack([score[:, 1] for score in scores])
        return np.argmax(scores, axis=1).astype(np.int32)
    return np.asarray(estimator.predict(features), dtype=np.int32).reshape(-1)


def feature_importance_frame(estimator, feature_columns: list[str]) -> pd.DataFrame | None:
    model = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    values = getattr(model, "feature_importances_", None)
    if values is None:
        coef = getattr(model, "coef_", None)
        if coef is None:
            return None
        values = np.mean(np.abs(coef), axis=0)
    frame = pd.DataFrame({"feature": feature_columns, "importance": values})
    return frame.sort_values("importance", ascending=False, kind="mergesort")


def train_model(model_kind: str, args: argparse.Namespace) -> Path:
    config_path = common.resolve_config_path(args.config)
    cfg = common.load_config(config_path)
    feature_set = common.resolve_feature_set(args, cfg)
    csv_selection = common.resolve_csv_selection(args)
    all_csv_files = common.collect_all_csv_files(args.inputs)
    csv_files = common.eligible_csv_files(all_csv_files, feature_set)
    common.print_csv_filter_summary(all_csv_files, csv_files, feature_set)
    csv_files = common.select_training_csv_files(csv_files, csv_selection, args.csv_indices)

    model_name = args.model_name
    if not model_name and not args.output_dir and sys.stdin.isatty():
        model_name = common.prompt_model_name()
    model_dir = resolve_model_dir_for_kind(cfg, args.output_dir, model_name, model_kind)

    data = common.load_training_frame(csv_files)
    feature_columns = common.select_feature_columns(data, feature_set)
    features = data.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    labels = data["label"].map(common.TRAINING_LABEL_TO_CODE).to_numpy(dtype=np.int32)

    training_cfg = cfg["ml"]["training"]
    seed = int(training_cfg.get("seed", 0))
    num_threads = int(training_cfg.get("num_threads", -1))
    validation_fraction = float(training_cfg["validation_fraction"])
    test_fraction = train_utils.test_fraction_from_config(cfg)
    split = train_utils.build_cell_group_split(
        data,
        validation_fraction,
        test_fraction,
        seed,
    )

    logger.info("Training %s model with %d feature(s)", model_kind, len(feature_columns))
    estimator = build_estimator(model_kind, seed, num_threads)
    estimator.fit(features[split.train], labels[split.train])
    valid_pred = predict_codes(estimator, features[split.valid])
    test_pred = predict_codes(estimator, features[split.test])
    validation_metrics = train_utils.evaluate_predictions(labels[split.valid], valid_pred)
    test_metrics = train_utils.evaluate_predictions(labels[split.test], test_pred)

    report_text = train_utils.format_training_report(
        data,
        split,
        validation_metrics,
        test_metrics,
    )
    print(report_text)

    source_csvs = [
        csv_path.relative_to(common.PROJECT_ROOT).as_posix()
        if csv_path.is_relative_to(common.PROJECT_ROOT)
        else str(csv_path)
        for csv_path in csv_files
    ]
    feature_cfg = dict(cfg.get("ml", {}).get("feature_extraction", {}) or {})
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "sklearn_classifier",
                "model_version": model_dir.name,
                "model_family": model_kind,
                "feature_set": feature_set,
                "feature_columns": feature_columns,
                "feature_extraction": feature_cfg,
                "source_csvs": source_csvs,
                "classes": train_utils.CLASS_CODES,
                "class_names": train_utils.CLASS_NAMES,
                "split_mode": "cell_group",
                "selected_split": train_utils.split_metadata(data, split),
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "metrics": {
                    "validation": train_utils.metrics_to_json(validation_metrics),
                    "test": train_utils.metrics_to_json(test_metrics),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / "report.txt").write_text(report_text + "\n", encoding="utf-8")
    (model_dir / "source_csvs.txt").write_text(
        common.format_source_csvs(csv_files),
        encoding="utf-8",
    )
    importance = feature_importance_frame(estimator, feature_columns)
    if importance is not None:
        importance.to_csv(model_dir / "feature_importance.csv", index=False)

    logger.info("Saved %s model to %s", model_kind, model_dir)
    if not args.skip_test_masks:
        test_dir = Path(args.test_dir)
        if not test_dir.is_absolute():
            test_dir = common.PROJECT_ROOT / test_dir
        saved_masks = common.save_test_masks(
            config_path,
            model_dir,
            test_dir,
            overwrite_output=True,
        )
        if saved_masks:
            print(f"Saved {saved_masks} test mask image(s) to {test_dir / model_dir.name}")
    return model_dir


def main(model_kind: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    train_model(model_kind, parse_args(model_kind))
