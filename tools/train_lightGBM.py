import argparse
import json
import logging
import re
import sys
from pathlib import Path

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

# Run as: python tools/train_lightGBM.py
# This script assumes the current working directory is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ml.feature_extractor import FeatureExtractor
from modules.ml.label_data import validate_label_rows, validation_groups
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

logger = logging.getLogger(__name__)

MIN_REQUIRED_COLUMNS = ("filename", "x", "y", "label")
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
    normalized = str(feature_set or "enhanced").strip().lower()
    if normalized in {"basic", "enhanced"}:
        return normalized
    raise RuntimeError(f"Unsupported feature set: {feature_set}")


def prompt_feature_set() -> str:
    while True:
        raw = input("Select feature set [basic/enhanced] (default: enhanced): ").strip().lower()
        if not raw:
            return "enhanced"
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
        .get("feature_set", "enhanced")
    )
    try:
        return normalize_feature_set(str(configured).lower())
    except RuntimeError:
        return "enhanced"


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
    return all(column in header.columns for column in columns)


def collect_all_csv_files(inputs: list[str]) -> list[Path]:
    if not inputs:
        inputs = [str(DEFAULT_INPUT_DIR)]

    csv_files: list[Path] = []
    for item in inputs:
        path = Path(item)
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
            "Run tools/enhance_training_features.py for legacy label CSVs."
        )

    source_files = [path for path in unique_files if not is_enhanced_csv(path)]
    return source_files or unique_files


def csv_skip_reason(csv_path: Path, feature_set: str) -> str | None:
    if feature_set == "enhanced":
        if not is_enhanced_csv(csv_path):
            return "filename does not start with enhanced_"
        if not csv_has_columns(csv_path, FeatureExtractor.ENHANCED_FEATURE_COLUMNS):
            return "missing enhanced columns"
        return None

    if is_enhanced_csv(csv_path):
        return "enhanced_ file excluded for basic training"
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
    # Preserve unrelated files and prior masks if new inference fails.
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

    inference = InferenceEngine(config_path, model_path=model_dir)
    quantifier = Quantifier(config_path)
    saved_count = 0
    result_rows: list[dict] = []

    for image_path in images:
        image = quiet_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning("Skipping unreadable test image: %s", image_path)
            continue
        feature_batch = inference.extract_features(image)
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


def load_training_frame(csv_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        missing = [column for column in MIN_REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")
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

    return validate_label_rows(data)


def select_feature_columns(data: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "enhanced":
        columns = list(FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise RuntimeError(
                "Enhanced training requires enhanced feature columns. "
                "Run tools/enhance_training_features.py first. Missing: "
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


def deterministic_validation_mask(labels: np.ndarray, validation_fraction: float) -> np.ndarray:
    if not 0.0 < validation_fraction < 1.0:
        raise RuntimeError("ml.training.validation_fraction must be between 0 and 1.")

    mask = np.zeros(labels.shape[0], dtype=bool)
    for label_code in sorted(np.unique(labels)):
        indices = np.flatnonzero(labels == label_code)
        if len(indices) <= 1:
            continue

        val_count = int(round(len(indices) * validation_fraction))
        val_count = max(1, min(val_count, len(indices) - 1))
        selected_offsets = np.unique(
            np.linspace(0, len(indices) - 1, num=val_count, dtype=int)
        )
        mask[indices[selected_offsets]] = True

    if not mask.any():
        raise RuntimeError("Validation split is empty. Add more labeled data.")
    if mask.all():
        raise RuntimeError("Training split is empty. Add more labeled data.")
    return mask


def grouped_validation_mask(
    labels: np.ndarray,
    validation_fraction: float,
    groups: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Hold out whole cells/images, never neighboring labeled pixels."""
    if not 0.0 < validation_fraction < 1.0:
        raise RuntimeError("validation_fraction must be between 0 and 1.")
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        raise RuntimeError(
            "Grouped validation needs at least two independent groups. Add more "
            "cells, or explicitly set validation_group_by=image for an exploratory "
            "image-level comparison (not a new-cell performance estimate)."
        )
    classes, class_indices = np.unique(labels, return_inverse=True)
    group_counts = np.zeros((len(unique_groups), len(classes)), dtype=np.int64)
    np.add.at(group_counts, (group_indices, class_indices), 1)
    total_counts = group_counts.sum(axis=0)
    splitter = GroupShuffleSplit(n_splits=128, test_size=validation_fraction, random_state=seed)
    best = None
    best_gap = float("inf")
    for _train_groups, valid_groups in splitter.split(unique_groups, groups=unique_groups):
        valid_counts = group_counts[valid_groups].sum(axis=0)
        if np.any(valid_counts == 0) or np.any(total_counts - valid_counts == 0):
            continue
        gap = abs(valid_counts.sum() / len(labels) - validation_fraction)
        if gap < best_gap:
            best = valid_groups
            best_gap = gap
    if best is None:
        raise RuntimeError(
            "Cannot build a group-disjoint split with every labeled class in both "
            "sets. Add independent labeled groups for rare classes. No pixel-level "
            "fallback was used."
        )
    return np.isin(group_indices, best)


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


def format_report(accuracy: float, per_class_metrics: np.ndarray, confusion: np.ndarray) -> str:
    labels = ["b", "l", "n"]
    lines = [f"Overall accuracy: {accuracy:.6f}", "", "Per-class metrics:"]
    for label, metrics in zip(labels, per_class_metrics):
        precision, recall, f1_score, support = metrics
        lines.append(
            f"  {label}: precision={precision:.6f} recall={recall:.6f} "
            f"f1={f1_score:.6f} support={int(support)}"
        )

    lines.extend(["", "Confusion matrix (rows=true, cols=pred):"])
    lines.append("      b      l      n")
    for label, row in zip(labels, confusion):
        lines.append(f"  {label}  {int(row[0]):6d} {int(row[1]):6d} {int(row[2]):6d}")
    return "\n".join(lines)


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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

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
    if model_dir.exists() and any(model_dir.iterdir()):
        raise RuntimeError(f"Model output folder is not empty: {model_dir}")
    data = load_training_frame(csv_files)
    feature_columns = select_feature_columns(data, feature_set)

    features = data.loc[:, feature_columns].to_numpy(dtype=np.float32, copy=True)
    labels = data["label"].map(TRAINING_LABEL_TO_CODE).to_numpy(dtype=np.int32)

    training_cfg = cfg["ml"]["training"]
    group_by = str(training_cfg.get("validation_group_by", "cell"))
    groups = validation_groups(data, group_by)
    validation_mask = grouped_validation_mask(
        labels,
        float(training_cfg["validation_fraction"]),
        groups,
        seed=int(training_cfg["seed"]),
    )

    x_train = features[~validation_mask]
    y_train = labels[~validation_mask]
    x_valid = features[validation_mask]
    y_valid = labels[validation_mask]

    params, num_boost_round, early_stopping_rounds = build_params(cfg)
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_columns)
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, feature_name=feature_columns)

    callbacks = []
    if early_stopping_rounds > 0:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    logger.info("Training %s model with %d feature(s): %s", feature_set, len(feature_columns), feature_columns)
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    valid_scores = booster.predict(x_valid, num_iteration=booster.best_iteration)
    valid_pred = np.argmax(valid_scores, axis=1)

    accuracy = accuracy_score(y_valid, valid_pred)
    per_class = np.column_stack(
        precision_recall_fscore_support(y_valid, valid_pred, labels=[0, 1, 2], zero_division=0)
    )
    confusion = confusion_matrix(y_valid, valid_pred, labels=[0, 1, 2])

    report_text = format_report(accuracy, per_class, confusion)
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
                "classes": ["b", "l", "n"],
                "best_iteration": booster.best_iteration,
                "validation_group_by": group_by,
                "training_groups": sorted(set(groups[~validation_mask])),
                "validation_groups": sorted(set(groups[validation_mask])),
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


if __name__ == "__main__":
    main()
