from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ENHANCED_PREFIX = "enhanced_"
CLASS_LABELS = [0, 1, 2]
CLASS_CODES = ["b", "l", "n"]
CLASS_NAMES = ["background", "lithium", "nonactive"]
LABEL_TO_CODE = {"b": 0, "l": 1, "n": 2}


@dataclass(frozen=True)
class SplitMasks:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray
    group_column: str = "cell_group_id"


def csv_identity_stem(source_csv: str | Path) -> str:
    stem = Path(source_csv).stem
    if stem.lower().startswith(ENHANCED_PREFIX):
        return stem[len(ENHANCED_PREFIX):]
    return stem


def image_basename(filename: str) -> str:
    normalized = str(filename).replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1]


def source_image_id(source_csv: str | Path, filename: str) -> str:
    return f"{csv_identity_stem(source_csv)}__{image_basename(filename)}"


def cell_group_id(source_csv: str | Path) -> str:
    stem = csv_identity_stem(source_csv)
    parts = stem.split("_")
    if len(parts) > 1 and parts[-1].upper() in {"F", "B", "FRONT", "BACK"}:
        parts = parts[:-1]
    return "_".join(parts) if parts else stem


def split_cell_group_id(value: str) -> tuple[str, str]:
    parts = str(value).rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return str(value), ""


def add_split_group_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["source_image_id"] = [
        source_image_id(source_csv, filename)
        for source_csv, filename in zip(data["source_csv"], data["filename"])
    ]
    data["cell_group_id"] = [cell_group_id(source_csv) for source_csv in data["source_csv"]]
    data[["analysis_id", "cell"]] = pd.DataFrame(
        [split_cell_group_id(value) for value in data["cell_group_id"]],
        index=data.index,
    )
    return data


def test_fraction_from_config(cfg: dict) -> float:
    training_cfg = cfg["ml"]["training"]
    return float(training_cfg.get("test_fraction", training_cfg["validation_fraction"]))


def validate_split_fractions(validation_fraction: float, test_fraction: float) -> None:
    if not 0.0 < validation_fraction < 1.0:
        raise RuntimeError("ml.training.validation_fraction must be between 0 and 1.")
    if not 0.0 < test_fraction < 1.0:
        raise RuntimeError("ml.training.test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1.0:
        raise RuntimeError("validation_fraction + test_fraction must be less than 1.")


def split_counts(total_count: int, validation_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    if total_count < 3:
        raise RuntimeError("Cell Group Split requires at least 3 unique cell_group_id values.")
    test_count = max(1, int(round(total_count * test_fraction)))
    test_count = min(test_count, total_count - 2)
    valid_count = max(1, int(round(total_count * validation_fraction)))
    valid_count = min(valid_count, total_count - test_count - 1)
    train_count = total_count - valid_count - test_count
    if train_count <= 0 or valid_count <= 0 or test_count <= 0:
        raise RuntimeError("Train/validation/test split would be empty. Add more cell data.")
    return train_count, valid_count, test_count


def build_cell_group_split(
    data: pd.DataFrame,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> SplitMasks:
    validate_split_fractions(validation_fraction, test_fraction)
    group_values = data["cell_group_id"].astype(str).to_numpy()
    unique_groups = np.array(sorted(set(group_values)), dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    _train_count, valid_count, test_count = split_counts(
        len(shuffled),
        validation_fraction,
        test_fraction,
    )

    test_groups = set(shuffled[:test_count])
    valid_groups = set(shuffled[test_count:test_count + valid_count])
    train_groups = set(shuffled[test_count + valid_count:])

    split = SplitMasks(
        train=np.array([value in train_groups for value in group_values], dtype=bool),
        valid=np.array([value in valid_groups for value in group_values], dtype=bool),
        test=np.array([value in test_groups for value in group_values], dtype=bool),
    )
    validate_split_integrity(data, split)
    return split


def group_sets(groups: pd.Series, split: SplitMasks) -> dict[str, set[str]]:
    group_values = groups.astype(str).to_numpy()
    return {
        "train": set(group_values[split.train]),
        "validation": set(group_values[split.valid]),
        "test": set(group_values[split.test]),
    }


def group_overlap_counts(groups: pd.Series, split: SplitMasks) -> dict[str, int]:
    sets = group_sets(groups, split)
    return {
        "train_validation": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "validation_test": len(sets["validation"] & sets["test"]),
    }


def assert_no_group_overlap(data: pd.DataFrame, split: SplitMasks, group_column: str) -> None:
    overlaps = group_overlap_counts(data[group_column], split)
    duplicates = {name: count for name, count in overlaps.items() if count}
    if duplicates:
        raise RuntimeError(f"{group_column} leakage across splits: {duplicates}")


def validate_split_integrity(data: pd.DataFrame, split: SplitMasks) -> None:
    for split_name, mask in (
        ("train", split.train),
        ("validation", split.valid),
        ("test", split.test),
    ):
        if int(mask.sum()) <= 0:
            raise RuntimeError(f"{split_name} split is empty. Add more labeled cell data.")
    assert_no_group_overlap(data, split, "cell_group_id")
    assert_no_group_overlap(data, split, "source_image_id")


def missing_classes_by_split(data: pd.DataFrame, split: SplitMasks) -> list[str]:
    warnings: list[str] = []
    for split_name, mask in (
        ("train", split.train),
        ("validation", split.valid),
        ("test", split.test),
    ):
        present = set(data.loc[mask, "label"])
        missing = [code for code in CLASS_CODES if code not in present]
        if missing:
            warnings.append(f"{split_name} split missing class(es): {', '.join(missing)}")
    return warnings


def split_metadata(data: pd.DataFrame, split: SplitMasks) -> dict:
    metadata = {
        "name": "Cell Group Split",
        "group_column": split.group_column,
        "rows": {
            "train": int(split.train.sum()),
            "validation": int(split.valid.sum()),
            "test": int(split.test.sum()),
        },
        "groups": {},
        "warnings": missing_classes_by_split(data, split),
    }
    for group_column in ("cell_group_id", "source_image_id"):
        sets = group_sets(data[group_column], split)
        metadata["groups"][group_column] = {
            "train": len(sets["train"]),
            "validation": len(sets["validation"]),
            "test": len(sets["test"]),
            "overlap": group_overlap_counts(data[group_column], split),
        }
    return metadata


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": float(np.mean(f1)),
        "per_class": np.column_stack([precision, recall, f1, support]),
        "confusion": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS),
    }


def metrics_to_json(metrics: dict) -> dict:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class": {
            code: {
                "class_name": name,
                "precision": float(row[0]),
                "recall": float(row[1]),
                "f1": float(row[2]),
                "support": int(row[3]),
            }
            for code, name, row in zip(CLASS_CODES, CLASS_NAMES, metrics["per_class"])
        },
        "confusion_matrix": metrics["confusion"].astype(int).tolist(),
    }


def format_metrics(metrics: dict) -> str:
    lines = [
        f"Overall accuracy: {metrics['accuracy']:.6f}",
        f"Macro F1: {metrics['macro_f1']:.6f}",
        "",
        "Per-class metrics:",
    ]
    for code, name, row in zip(CLASS_CODES, CLASS_NAMES, metrics["per_class"]):
        precision, recall, f1_score, support = row
        lines.append(
            f"  {code} ({name}): precision={precision:.6f} recall={recall:.6f} "
            f"f1={f1_score:.6f} support={int(support)}"
        )
    lines.extend(["", "Confusion matrix (rows=true, cols=pred):", "      b      l      n"])
    for code, row in zip(CLASS_CODES, metrics["confusion"]):
        lines.append(f"  {code}  {int(row[0]):6d} {int(row[1]):6d} {int(row[2]):6d}")
    return "\n".join(lines)


def class_count_lines(frame: pd.DataFrame) -> list[str]:
    total = len(frame)
    lines = []
    for code, name in zip(CLASS_CODES, CLASS_NAMES):
        count = int((frame["label"] == code).sum())
        ratio = (count / total) if total else 0.0
        lines.append(f"    {code} ({name}): {count} ({ratio:.2%})")
    return lines


def format_split_section(data: pd.DataFrame, split_name: str, mask: np.ndarray) -> list[str]:
    frame = data.loc[mask]
    analysis_ids = sorted(frame["analysis_id"].astype(str).unique())
    cells = sorted(frame["cell_group_id"].astype(str).unique())
    lines = [
        f"[{split_name}]",
        f"  Rows: {len(frame)}",
        f"  Analysis IDs ({len(analysis_ids)}): {', '.join(analysis_ids)}",
        f"  Cells ({len(cells)}): {', '.join(cells)}",
        "  Class rows:",
    ]
    lines.extend(class_count_lines(frame))
    return lines


def format_training_report(
    data: pd.DataFrame,
    split: SplitMasks,
    validation_metrics: dict,
    test_metrics: dict,
) -> str:
    warnings = missing_classes_by_split(data, split)
    cell_overlap = group_overlap_counts(data["cell_group_id"], split)
    source_overlap = group_overlap_counts(data["source_image_id"], split)
    lines = [
        "Selected training split: Cell Group Split",
        "Classes: b=background, l=lithium, n=nonactive",
        "",
        "Group leakage checks:",
        (
            "  cell_group_id overlap: "
            f"train-validation={cell_overlap['train_validation']}, "
            f"train-test={cell_overlap['train_test']}, "
            f"validation-test={cell_overlap['validation_test']}"
        ),
        (
            "  source_image_id overlap: "
            f"train-validation={source_overlap['train_validation']}, "
            f"train-test={source_overlap['train_test']}, "
            f"validation-test={source_overlap['validation_test']}"
        ),
        "",
        "Split summary:",
    ]
    for split_name, mask in (
        ("Train", split.train),
        ("Validation", split.valid),
        ("Test", split.test),
    ):
        lines.extend(format_split_section(data, split_name, mask))
        lines.append("")
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"  {warning}" for warning in warnings)
        lines.append("")
    lines.extend(["Validation metrics:", format_metrics(validation_metrics), "", "Test metrics:"])
    lines.append(format_metrics(test_metrics))
    return "\n".join(lines)
