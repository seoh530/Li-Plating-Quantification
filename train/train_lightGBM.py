import argparse
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

# Run through: python train/train_model.py
# This script assumes the current working directory is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ml.feature_extractor import FeatureExtractor
from modules.ml.inference import (
    InferenceEngine,
    LABEL_BACKGROUND,
    LABEL_EDGE,
    LABEL_INACTIVE,
    LABEL_LI_SALT,
    TRAINING_LABEL_TO_CODE,
)
from modules.ml.quantifier import Quantifier
from modules.utils.image_io import quiet_imread
from train import training_utils as train_utils

logger = logging.getLogger(__name__)

FILENAME_COLUMNS = ("filename", "file_name")
MIN_REQUIRED_COLUMNS = ("x", "y", "label")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "labeled"
DEFAULT_TEST_DIR = PROJECT_ROOT / "data" / "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ENHANCED_PREFIX = "enhanced_"
INVALID_MODEL_NAME_CHARS = set('<>:"/\\|?*')
COLOR_FEATURE_COLUMNS = (
    "r",
    "g",
    "b",
    "h",
    "s",
    "v",
    "l_lab",
    "a_lab",
    "b_lab",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LightGBM pixel classifier from labeled or enhanced CSV files."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV files or directories containing labeled CSV files. Defaults to data/labeled/.",
    )
    parser.add_argument(
        "--config",
        default="config/thresholds.yaml",
        help="Path to thresholds.yaml",
    )
    parser.add_argument(
        "--feature-set",
        choices=("basic", "enhanced"),
        default=None,
        help=(
            "Training feature set. basic uses RGB/HSV/LAB color only; enhanced adds "
            "residual/robust/edge features. If omitted, prompts at runtime."
        ),
    )
    parser.add_argument(
        "--csv-selection",
        choices=("all", "custom"),
        default=None,
        help="Use all eligible CSVs or choose a custom subset. If omitted, prompts at runtime.",
    )
    parser.add_argument(
        "--csv-indices",
        default=None,
        help="Comma-separated 1-based CSV list indices for --csv-selection custom. Example: 1,2,4",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model folder name to create under models/. If omitted, prompts at runtime.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Explicit model output directory. If omitted, the next versioned folder "
            "under models/ is selected automatically. Enhanced models use logical_lightgbm_."
        ),
    )
    parser.add_argument(
        "--test-dir",
        default=str(DEFAULT_TEST_DIR),
        help="Directory of test images to mask after training. Defaults to data/test.",
    )
    parser.add_argument(
        "--skip-test-masks",
        action="store_true",
        help="Train only; do not run data/test masking after saving the model.",
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict:
    with open(resolve_config_path(config_path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def normalize_feature_set(feature_set: str | None) -> str:
    normalized = str(feature_set or "basic").strip().lower()
    if normalized in {"basic", "enhanced"}:
        return normalized
    raise RuntimeError(f"Unsupported feature set: {feature_set}")


def prompt_feature_set() -> str:
    while True:
        raw = input("Select feature set [basic/enhanced] (default: basic): ").strip().lower()
        if not raw:
            return "basic"
        if raw == "basic":
            return raw
        if raw == "enhanced":
            return "enhanced"
        print("Enter basic or enhanced.")


def normalize_csv_selection(selection: str | None) -> str:
    normalized = str(selection or "all").strip().lower()
    if normalized in {"all", "custom"}:
        return normalized
    raise RuntimeError(f"Unsupported CSV selection: {selection}")


def prompt_csv_selection() -> str:
    while True:
        raw = input("CSV selection [all/custom] (default: all): ").strip().lower()
        if not raw:
            return "all"
        if raw in {"all", "custom"}:
            return raw
        print("Enter all or custom.")


def validate_model_name(model_name: str) -> str:
    name = model_name.strip()
    if not name:
        raise RuntimeError("Model name cannot be empty.")
    if name in {".", ".."} or any(char in INVALID_MODEL_NAME_CHARS for char in name):
        raise RuntimeError(
            "Model name must be a folder name only; do not include path separators or Windows-reserved characters."
        )
    if any(ord(char) < 32 for char in name):
        raise RuntimeError("Model name contains an invalid control character.")
    return name


def prompt_model_name() -> str | None:
    raw = input("Model name under models/ (blank = auto version): ").strip()
    return validate_model_name(raw) if raw else None


def resolve_feature_set(args: argparse.Namespace, cfg: dict) -> str:
    if args.feature_set:
        return normalize_feature_set(args.feature_set)
    if sys.stdin.isatty():
        return prompt_feature_set()
    configured = (
        cfg.get("ml", {})
        .get("feature_extraction", {})
        .get("feature_set", "basic")
    )
    try:
        return normalize_feature_set(str(configured).lower())
    except RuntimeError:
        return "basic"


def resolve_csv_selection(args: argparse.Namespace) -> str:
    if args.csv_selection:
        return normalize_csv_selection(args.csv_selection)
    if sys.stdin.isatty():
        return prompt_csv_selection()
    return "all"


def csv_display_path(csv_path: Path) -> str:
    try:
        return csv_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(csv_path)


def print_csv_choices(csv_files: list[Path]) -> None:
    print("Eligible training CSV files:")
    for index, csv_path in enumerate(csv_files, start=1):
        print(f"  {index:>2}. {csv_display_path(csv_path)}")


def parse_csv_indices(raw: str, csv_count: int) -> list[int]:
    selected: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        try:
            index = int(part)
        except ValueError as exc:
            raise RuntimeError("CSV indices must be numbers separated by commas.") from exc
        if not 1 <= index <= csv_count:
            raise RuntimeError(f"CSV index out of range: {index}")
        if index not in seen:
            selected.append(index)
            seen.add(index)
    if not selected:
        raise RuntimeError("At least one CSV index must be selected.")
    return selected


def prompt_csv_indices(csv_count: int) -> list[int]:
    while True:
        raw = input("Select CSV numbers (example: 1,2,4): ").strip()
        try:
            return parse_csv_indices(raw, csv_count)
        except RuntimeError as exc:
            print(str(exc))


def select_training_csv_files(
    csv_files: list[Path],
    selection: str,
    csv_indices: str | None = None,
) -> list[Path]:
    if selection == "all":
        return csv_files

    print_csv_choices(csv_files)
    if csv_indices:
        selected_indices = parse_csv_indices(csv_indices, len(csv_files))
    elif sys.stdin.isatty():
        selected_indices = prompt_csv_indices(len(csv_files))
    else:
        raise RuntimeError("--csv-indices is required when --csv-selection custom is used non-interactively.")
    return [csv_files[index - 1] for index in selected_indices]


def models_root_from_config(cfg: dict) -> tuple[Path, str]:
    configured_model_path = Path(cfg.get("ml", {}).get("model_path", "models/lightgbm"))
    if configured_model_path.is_absolute():
        return configured_model_path.parent, configured_model_path.name
    return PROJECT_ROOT / configured_model_path.parent, configured_model_path.name


def resolve_model_dir(
    cfg: dict,
    output_dir: str | None,
    feature_set: str,
    model_name: str | None = None,
) -> Path:
    if output_dir and model_name:
        raise RuntimeError("--output-dir and --model-name cannot be used together.")
    if output_dir:
        model_dir = Path(output_dir)
        return model_dir if model_dir.is_absolute() else PROJECT_ROOT / model_dir

    models_root, configured_name = models_root_from_config(cfg)
    if model_name:
        model_dir = models_root / validate_model_name(model_name)
        if model_dir.exists():
            raise RuntimeError(f"Model output folder already exists: {model_dir}")
        return model_dir

    prefix = "logical_lightgbm" if feature_set == "enhanced" else model_prefix(configured_name)
    return next_versioned_model_dir(models_root, prefix)


def model_prefix(model_dir_name: str) -> str:
    match = re.fullmatch(r"(.+)_\d+(?:\.\d+)?", model_dir_name)
    return match.group(1) if match else model_dir_name


def next_versioned_model_dir(models_root: Path, prefix: str) -> Path:
    models_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)(?:\.(\d+))?$")
    versions: list[tuple[int, int]] = []
    for child in models_root.iterdir():
        if not child.is_dir():
            continue
        match = pattern.fullmatch(child.name)
        if match is None:
            continue
        major = int(match.group(1))
        minor = int(match.group(2) or 0)
        versions.append((major, minor))

    if not versions:
        candidate = models_root / f"{prefix}_1"
        if not candidate.exists():
            return candidate
        versions.append((1, 0))

    major = max(version[0] for version in versions)
    minor = max(version[1] for version in versions if version[0] == major) + 1
    candidate = models_root / f"{prefix}_{major}.{minor}"
    while candidate.exists():
        minor += 1
        candidate = models_root / f"{prefix}_{major}.{minor}"
    return candidate


def is_enhanced_csv(path: Path) -> bool:
    return path.name.lower().startswith(ENHANCED_PREFIX)


def csv_has_columns(path: Path, columns: tuple[str, ...] | list[str]) -> bool:
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return False
    if "filename" in columns:
        required = [column for column in columns if column != "filename"]
        return (
            all(column in header.columns for column in required)
            and any(column in header.columns for column in FILENAME_COLUMNS)
        )
    return all(column in header.columns for column in columns)


def has_basic_label_columns(path: Path) -> bool:
    return csv_has_columns(path, ["filename", *MIN_REQUIRED_COLUMNS])


def resolve_input_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def collect_all_csv_files(inputs: list[str]) -> list[Path]:
    if not inputs:
        inputs = [str(DEFAULT_INPUT_DIR)]

    csv_files: list[Path] = []
    for item in inputs:
        path = resolve_input_path(item)
        if path.is_dir():
            csv_files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            csv_files.append(path)
        else:
            raise RuntimeError(f"Input path is not a CSV file or directory: {path}")

    unique_files = sorted({path.resolve() for path in csv_files})
    if not unique_files:
        raise RuntimeError("No labeled CSV files found.")
    return unique_files


def eligible_csv_files(unique_files: list[Path], feature_set: str) -> list[Path]:
    if feature_set == "enhanced":
        enhanced_files = [
            path
            for path in unique_files
            if is_enhanced_csv(path)
            and csv_has_columns(path, FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
        ]
        if enhanced_files:
            return enhanced_files
        raise RuntimeError(
            "No enhanced CSV files found. Enhanced training requires enhanced_*.csv files "
            "with residual/robust/edge columns such as l_res, l_robust, and edge_band_5mm. "
            "Run train/train_model.py to auto-generate enhanced CSVs."
        )

    source_files = [
        path
        for path in unique_files
        if not is_enhanced_csv(path) and has_basic_label_columns(path)
    ]
    if source_files:
        return source_files
    raise RuntimeError(
        "No basic label CSV files found. Basic training requires CSV files with "
        "filename/file_name, x, y, and label columns."
    )


def csv_skip_reason(csv_path: Path, feature_set: str) -> str | None:
    if feature_set == "enhanced":
        if not is_enhanced_csv(csv_path):
            return "filename does not start with enhanced_"
        if not csv_has_columns(csv_path, FeatureExtractor.ENHANCED_FEATURE_COLUMNS):
            return "missing enhanced columns"
        return None

    if is_enhanced_csv(csv_path):
        return "enhanced_ file excluded for basic training"
    if not has_basic_label_columns(csv_path):
        return "missing filename/x/y/label columns"
    return None


def print_csv_filter_summary(all_csv_files: list[Path], eligible_files: list[Path], feature_set: str) -> None:
    eligible_set = {path.resolve() for path in eligible_files}
    skipped = [path for path in all_csv_files if path.resolve() not in eligible_set]
    print(
        f"CSV scan summary: found {len(all_csv_files)} CSV file(s), "
        f"eligible for {feature_set}: {len(eligible_files)}, skipped: {len(skipped)}"
    )
    if not skipped:
        return

    reason_counts: dict[str, int] = {}
    for csv_path in skipped:
        reason = csv_skip_reason(csv_path, feature_set) or "not eligible"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    for reason, count in sorted(reason_counts.items()):
        print(f"  skipped {count}: {reason}")


def collect_csv_files(inputs: list[str], feature_set: str = "basic") -> list[Path]:
    unique_files = collect_all_csv_files(inputs)
    return eligible_csv_files(unique_files, feature_set)


def expected_test_mask_path(test_dir: Path, output_dir: Path, image_path: Path) -> Path:
    relative_path = image_path.relative_to(test_dir)
    return output_dir / relative_path.parent / f"{relative_path.stem}_mask.png"


def collect_test_images(
    test_dir: Path,
    output_dir: Path,
    *,
    skip_existing_masks: bool = False,
) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        path
        for path in test_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.name.lower().endswith("_mask.png")
        and output_dir not in path.parents
        and (not skip_existing_masks or not expected_test_mask_path(test_dir, output_dir, path).exists())
    )


def test_result_row(
    label_mask: np.ndarray,
    image_path: Path,
    test_dir: Path,
    mask_path: Path,
) -> dict:
    relative_image = image_path.relative_to(test_dir).as_posix()
    valid_mask = label_mask >= 0
    total_pixel = int(valid_mask.sum())
    if total_pixel <= 0:
        raise RuntimeError(f"Test image has no valid pixels: {image_path}")

    counts = np.bincount(label_mask[valid_mask], minlength=LABEL_EDGE + 1)
    background_count = int(counts[LABEL_BACKGROUND])
    nonactive_count = int(counts[LABEL_INACTIVE])
    li_salt_count = int(counts[LABEL_LI_SALT])
    edge_count = int(counts[LABEL_EDGE])

    def percent(count: int) -> float:
        return round((count / total_pixel) * 100.0, 4)

    return {
        "filename": relative_image,
        "mask_file": mask_path.relative_to(test_dir).as_posix(),
        "total_pixel": total_pixel,
        "background_pixel": background_count,
        "nonactive_pixel": nonactive_count,
        "li_salt_pixel": li_salt_count,
        "edge_pixel": edge_count,
        "background_percent": percent(background_count),
        "nonactive_percent": percent(nonactive_count),
        "li_salt_percent": percent(li_salt_count),
        "edge_percent": percent(edge_count),
    }


def write_test_results(output_dir: Path, rows: list[dict], *, append: bool = False) -> Path | None:
    if not rows:
        return None
    output_path = output_dir / "test_results.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(rows)
    if append and output_path.exists():
        try:
            existing_frame = pd.read_csv(output_path)
        except pd.errors.EmptyDataError:
            existing_frame = pd.DataFrame()
        if not existing_frame.empty:
            result_frame = pd.concat([existing_frame, result_frame], ignore_index=True, sort=False)
            if "filename" in result_frame.columns:
                result_frame = result_frame.drop_duplicates(subset=["filename"], keep="last")
    result_frame.to_csv(output_path, index=False)
    return output_path


def overwrite_test_output_dir(test_dir: Path, model_dir: Path) -> Path:
    output_dir = test_dir / model_dir.name
    resolved_output = output_dir.resolve()
    resolved_test = test_dir.resolve()
    if resolved_output == resolved_test or resolved_test not in resolved_output.parents:
        raise RuntimeError(f"Refusing to overwrite unsafe test output path: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_test_masks(
    config_path: str | Path,
    model_dir: Path,
    test_dir: Path,
    *,
    overwrite_output: bool = False,
    skip_existing_masks: bool = False,
) -> int:
    output_dir = overwrite_test_output_dir(test_dir, model_dir) if overwrite_output else test_dir / model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_images = collect_test_images(test_dir, output_dir)
    images = collect_test_images(test_dir, output_dir, skip_existing_masks=skip_existing_masks)
    if not images:
        if skip_existing_masks and all_images:
            logger.info(
                "All %d test image(s) already have masks under %s; skipping inference.",
                len(all_images),
                output_dir,
            )
        else:
            logger.info("No test images found under %s; skipping test mask generation.", test_dir)
        return 0

    extractor = FeatureExtractor(config_path)
    inference = InferenceEngine(config_path, model_path=model_dir)
    quantifier = Quantifier(config_path)
    saved_count = 0
    result_rows: list[dict] = []

    for image_path in images:
        image = quiet_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning("Skipping unreadable test image: %s", image_path)
            continue
        feature_batch = extractor.extract(
            image,
            downsample_factor=inference.downsample_factor,
            feature_columns=inference.feature_columns,
        )
        try:
            label_mask = inference.predict_mask(feature_batch)
        finally:
            del feature_batch

        mask_path = expected_test_mask_path(test_dir, output_dir, image_path)
        quantifier.save_mask_image(label_mask, mask_path)
        result_rows.append(
            test_result_row(
                label_mask,
                image_path,
                test_dir,
                mask_path,
            )
        )
        saved_count += 1
        logger.info("Saved test mask: %s", mask_path)

    result_csv = write_test_results(output_dir, result_rows, append=skip_existing_masks)
    if result_csv is not None:
        logger.info("Saved test result CSV: %s", result_csv)
    logger.info("Saved %d test mask image(s) to %s", saved_count, output_dir)
    return saved_count


def normalize_training_frame_columns(frame: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    missing = [column for column in MIN_REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")
    if "filename" not in frame.columns:
        if "file_name" not in frame.columns:
            raise RuntimeError(
                f"Missing required image filename column in {csv_path}: expected filename or file_name"
            )
        frame = frame.rename(columns={"file_name": "filename"})
    return frame


def load_training_frame(csv_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        frame = normalize_training_frame_columns(frame, csv_path)
        frame = frame.copy()
        frame["source_csv"] = str(csv_path)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(
        by=["source_csv", "filename", "x", "y"],
        kind="mergesort",
    ).reset_index(drop=True)

    invalid_labels = sorted(set(data["label"]) - set(TRAINING_LABEL_TO_CODE))
    if invalid_labels:
        raise RuntimeError(f"Unsupported labels found: {invalid_labels}")

    return train_utils.add_split_group_columns(data)


def select_feature_columns(data: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "enhanced":
        columns = list(FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise RuntimeError(
                "Enhanced training requires enhanced feature columns. "
                "Run train/train_model.py to auto-generate enhanced CSVs. Missing: "
                f"{missing}"
            )
        return columns

    if feature_set == "basic":
        columns = list(COLOR_FEATURE_COLUMNS)
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise RuntimeError(
                "Basic training requires RGB/HSV/LAB color columns. "
                f"Missing: {missing}"
            )
        return columns

    raise RuntimeError(f"Unsupported feature set: {feature_set}")


def build_params(cfg: dict) -> tuple[dict, int, int]:
    training_cfg = cfg["ml"]["training"]
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": float(training_cfg["learning_rate"]),
        "num_leaves": int(training_cfg["num_leaves"]),
        "max_depth": int(training_cfg["max_depth"]),
        "min_data_in_leaf": int(training_cfg["min_data_in_leaf"]),
        "feature_fraction": float(training_cfg["feature_fraction"]),
        "bagging_fraction": float(training_cfg["bagging_fraction"]),
        "bagging_freq": int(training_cfg["bagging_freq"]),
        "lambda_l1": float(training_cfg["lambda_l1"]),
        "lambda_l2": float(training_cfg["lambda_l2"]),
        "min_gain_to_split": float(training_cfg["min_gain_to_split"]),
        "max_bin": int(training_cfg["max_bin"]),
        "num_threads": int(training_cfg["num_threads"]),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": int(training_cfg["seed"]),
        "data_random_seed": int(training_cfg["seed"]),
        "feature_fraction_seed": int(training_cfg["seed"]),
        "bagging_seed": int(training_cfg["seed"]),
    }
    return (
        params,
        int(training_cfg["num_boost_round"]),
        int(training_cfg["early_stopping_rounds"]),
    )


def train_lightgbm_booster(
    features: np.ndarray,
    labels: np.ndarray,
    feature_columns: list[str],
    split: train_utils.SplitMasks,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        features[split.train],
        label=labels[split.train],
        feature_name=feature_columns,
    )
    valid_set = lgb.Dataset(
        features[split.valid],
        label=labels[split.valid],
        reference=train_set,
        feature_name=feature_columns,
    )

    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    return booster


def predict_lightgbm_codes(booster: lgb.Booster, features: np.ndarray) -> np.ndarray:
    scores = booster.predict(features, num_iteration=booster.best_iteration)
    return np.argmax(scores, axis=1).astype(np.int32)


def format_source_csvs(csv_files: list[Path]) -> str:
    lines = []
    for csv_path in csv_files:
        try:
            lines.append(csv_path.relative_to(PROJECT_ROOT).as_posix())
        except ValueError:
            lines.append(str(csv_path))
    return "\n".join(lines) + "\n"


def save_feature_importance(booster: lgb.Booster, feature_columns: list[str], model_dir: Path) -> None:
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_gain": booster.feature_importance(importance_type="gain"),
            "importance_split": booster.feature_importance(importance_type="split"),
        }
    )
    importance = importance.sort_values(
        by=["importance_gain", "importance_split"],
        ascending=False,
        kind="mergesort",
    )
    importance.to_csv(model_dir / "feature_importance.csv", index=False)


def train_model(args: argparse.Namespace) -> Path:
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    feature_set = resolve_feature_set(args, cfg)
    csv_selection = resolve_csv_selection(args)
    all_csv_files = collect_all_csv_files(args.inputs)
    csv_files = eligible_csv_files(all_csv_files, feature_set)
    print_csv_filter_summary(all_csv_files, csv_files, feature_set)
    csv_files = select_training_csv_files(csv_files, csv_selection, args.csv_indices)

    model_name = args.model_name
    if not model_name and not args.output_dir and sys.stdin.isatty():
        model_name = prompt_model_name()
    model_dir = resolve_model_dir(cfg, args.output_dir, feature_set, model_name)
    data = load_training_frame(csv_files)
    feature_columns = select_feature_columns(data, feature_set)

    features = data.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    labels = data["label"].map(TRAINING_LABEL_TO_CODE).to_numpy(dtype=np.int32)

    training_cfg = cfg["ml"]["training"]
    validation_fraction = float(training_cfg["validation_fraction"])
    test_fraction = train_utils.test_fraction_from_config(cfg)
    seed = int(training_cfg.get("seed", 0))

    split = train_utils.build_cell_group_split(
        data,
        validation_fraction,
        test_fraction,
        seed,
    )

    params, num_boost_round, early_stopping_rounds = build_params(cfg)
    logger.info("Training %s model with %d feature(s): %s", feature_set, len(feature_columns), feature_columns)
    booster = train_lightgbm_booster(
        features,
        labels,
        feature_columns,
        split,
        params,
        num_boost_round,
        early_stopping_rounds,
    )
    valid_pred = predict_lightgbm_codes(booster, features[split.valid])
    test_pred = predict_lightgbm_codes(booster, features[split.test])
    validation_metrics = train_utils.evaluate_predictions(labels[split.valid], valid_pred)
    test_metrics = train_utils.evaluate_predictions(labels[split.test], test_pred)

    report_text = train_utils.format_training_report(
        data,
        split,
        validation_metrics,
        test_metrics,
    )
    print(report_text)

    feature_cfg = dict(cfg.get("ml", {}).get("feature_extraction", {}) or {})
    source_csvs = [
        csv_path.relative_to(PROJECT_ROOT).as_posix()
        if csv_path.is_relative_to(PROJECT_ROOT)
        else str(csv_path)
        for csv_path in csv_files
    ]
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "model.txt"))
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_type": "lightgbm_booster",
                "model_version": model_dir.name,
                "model_family": "logical_lightgbm" if feature_set == "enhanced" else "lightgbm",
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
                "best_iteration": booster.best_iteration,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (model_dir / "report.txt").write_text(report_text + "\n", encoding="utf-8")
    (model_dir / "source_csvs.txt").write_text(
        format_source_csvs(csv_files),
        encoding="utf-8",
    )
    save_feature_importance(booster, feature_columns, model_dir)

    logger.info("Saved model to %s", model_dir)
    logger.info(
        "To use this model in the app, set ml.model_path to: %s",
        model_dir.relative_to(PROJECT_ROOT) if model_dir.is_relative_to(PROJECT_ROOT) else model_dir,
    )

    if not args.skip_test_masks:
        test_dir = Path(args.test_dir)
        if not test_dir.is_absolute():
            test_dir = PROJECT_ROOT / test_dir
        saved_masks = save_test_masks(config_path, model_dir, test_dir, overwrite_output=True)
        if saved_masks:
            print(f"Saved {saved_masks} test mask image(s) to {test_dir / model_dir.name}")
    return model_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    train_model(parse_args())


if __name__ == "__main__":
    main()
