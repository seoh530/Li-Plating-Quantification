from __future__ import annotations

from itertools import combinations
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from tools.train_lightGBM import DEFAULT_TEST_DIR, IMAGE_EXTENSIONS


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = DEFAULT_TEST_DIR
RESULT_FILE = "test_results.csv"
SOURCE_CSV_FILE = "source_csvs.txt"
REVIEW_FILE = "_prediction_review.csv"
REVIEW_STATUSES = ("Pending", "Pass", "Fail")
REVIEW_INPUT_STATUSES = ("Pass", "Fail")
FAIL_BORDER_COLOR = (220, 38, 38)

CLASS_COLUMNS = {
    "Lithium Salt": "li_salt_percent",
    "Background": "background_percent",
    "Nonactive": "nonactive_percent",
}
PIXEL_COLUMNS = {
    "Lithium Salt": "li_salt_pixel",
    "Background": "background_pixel",
    "Nonactive": "nonactive_pixel",
}
MASK_BGR_TO_LABEL = {
    (0, 255, 0): 0,      # background
    (0, 0, 255): 1,      # lithium salt
    (255, 0, 0): 2,      # nonactive
    (128, 128, 128): 3,  # edge
}
LABEL_TO_RGB = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (128, 128, 128),
}
LABEL_NAMES = {
    0: "Background",
    1: "Lithium Salt",
    2: "Nonactive",
    3: "Edge",
}


st.set_page_config(page_title="Test Model Comparison", layout="wide")


def path_mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return 0


def normalize_enhanced_csv_entry(value: str) -> str | None:
    normalized = value.strip().strip('"').replace("\\", "/")
    if not normalized:
        return None
    name = normalized.rsplit("/", 1)[-1]
    if not name.lower().endswith(".csv") or not name.lower().startswith("enhanced_"):
        return None
    return normalized


def find_model_source_csv_file(model_name: str, models_root: Path | None = None) -> Path | None:
    root = models_root or PROJECT_ROOT / "models"
    direct = root / model_name / SOURCE_CSV_FILE
    if direct.is_file():
        return direct
    if not root.is_dir():
        return None
    matches = sorted(
        (
            path
            for path in root.rglob(SOURCE_CSV_FILE)
            if path.parent.name == model_name and path.is_file()
        ),
        key=lambda path: path.as_posix().lower(),
    )
    return matches[0] if matches else None


def model_source_csv_signature(
    model_names: list[str],
    models_root: Path | None = None,
) -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (
            model_name,
            str(source_path) if source_path is not None else "",
            path_mtime_ns(source_path) if source_path is not None else 0,
        )
        for model_name in model_names
        for source_path in [find_model_source_csv_file(model_name, models_root)]
    )


@st.cache_data(show_spinner=False)
def load_enhanced_source_csv_matrix(
    signature: tuple[tuple[str, str, int], ...],
) -> tuple[pd.DataFrame, list[str]]:
    sources_by_model: dict[str, set[str]] = {}
    missing_models: list[str] = []
    all_sources: set[str] = set()

    for model_name, source_path_text, _mtime in signature:
        model_sources: set[str] = set()
        if not source_path_text:
            missing_models.append(model_name)
            sources_by_model[model_name] = model_sources
            continue
        try:
            lines = Path(source_path_text).read_text(encoding="utf-8").splitlines()
        except OSError:
            missing_models.append(model_name)
            sources_by_model[model_name] = model_sources
            continue
        for line in lines:
            entry = normalize_enhanced_csv_entry(line)
            if entry is None:
                continue
            model_sources.add(entry)
            all_sources.add(entry)
        sources_by_model[model_name] = model_sources

    rows = []
    for source_csv in sorted(all_sources, key=lambda value: value.lower()):
        row = {"Enhanced CSV": source_csv}
        for model_name, _source_path, _mtime in signature:
            row[model_name] = "●" if source_csv in sources_by_model.get(model_name, set()) else ""
        rows.append(row)

    columns = ["Enhanced CSV", *[model_name for model_name, _source_path, _mtime in signature]]
    return pd.DataFrame(rows, columns=columns), missing_models


def discover_result_dirs(test_dir: Path = TEST_DIR) -> list[Path]:
    if not test_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in test_dir.iterdir()
            if path.is_dir() and (path / RESULT_FILE).is_file()
        ],
        key=lambda path: path.name.lower(),
    )


def discover_original_images(test_dir: Path, result_dirs: list[Path]) -> list[Path]:
    result_names = {path.name for path in result_dirs}
    if not test_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in test_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and not path.name.lower().endswith("_mask.png")
            and not any(part in result_names for part in path.relative_to(test_dir).parts[:-1])
        ],
        key=lambda path: path.relative_to(test_dir).as_posix().lower(),
    )


@st.cache_data(show_spinner=False)
def load_test_results(test_dir_text: str, signature: tuple[tuple[str, int], ...]) -> pd.DataFrame:
    test_dir = Path(test_dir_text)
    frames: list[pd.DataFrame] = []
    for model_name, _mtime in signature:
        result_path = test_dir / model_name / RESULT_FILE
        if not result_path.is_file():
            continue
        try:
            frame = pd.read_csv(result_path)
        except Exception:
            continue
        frame["model_name"] = model_name
        frame["result_dir"] = str(result_path.parent)
        frames.append(frame)

    if not frames:
        return empty_results_frame()

    data = pd.concat(frames, ignore_index=True)
    for column in [
        "filename",
        "mask_file",
        "total_pixel",
        "background_pixel",
        "nonactive_pixel",
        "li_salt_pixel",
        "edge_pixel",
        "background_percent",
        "nonactive_percent",
        "li_salt_percent",
        "edge_percent",
    ]:
        if column not in data.columns:
            data[column] = np.nan

    for column in [
        "total_pixel",
        "background_pixel",
        "nonactive_pixel",
        "li_salt_pixel",
        "edge_pixel",
        "background_percent",
        "nonactive_percent",
        "li_salt_percent",
        "edge_percent",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["filename"] = data["filename"].astype(str)
    data["mask_file"] = data["mask_file"].astype(str)
    data["original_path"] = data["filename"].map(lambda value: str(test_dir / value))
    data["mask_path"] = data["mask_file"].map(lambda value: str(test_dir / value))
    data["original_exists"] = data["original_path"].map(lambda value: Path(value).is_file())
    data["mask_exists"] = data["mask_path"].map(lambda value: Path(value).is_file())
    data["percent_sum"] = (
        data["background_percent"].fillna(0)
        + data["nonactive_percent"].fillna(0)
        + data["li_salt_percent"].fillna(0)
        + data["edge_percent"].fillna(0)
    )
    return data


def empty_results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "filename",
            "mask_file",
            "model_name",
            "result_dir",
            "original_path",
            "mask_path",
            "original_exists",
            "mask_exists",
            "total_pixel",
            "background_percent",
            "nonactive_percent",
            "li_salt_percent",
            "edge_percent",
            "percent_sum",
        ]
    )


def result_signature(result_dirs: list[Path]) -> tuple[tuple[str, int], ...]:
    return tuple((path.name, path_mtime_ns(path / RESULT_FILE)) for path in result_dirs)


def review_path(test_dir: Path = TEST_DIR) -> Path:
    return test_dir / REVIEW_FILE


def empty_review_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["model_name", "filename", "review_status", "reviewed_at"]
    )


def load_prediction_reviews(test_dir: Path = TEST_DIR) -> pd.DataFrame:
    path = review_path(test_dir)
    if not path.is_file():
        return empty_review_frame()
    try:
        frame = pd.read_csv(path)
    except Exception:
        return empty_review_frame()
    for column in ["model_name", "filename", "review_status", "reviewed_at"]:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[["model_name", "filename", "review_status", "reviewed_at"]].copy()
    frame["model_name"] = frame["model_name"].astype(str)
    frame["filename"] = frame["filename"].astype(str)
    frame["review_status"] = frame["review_status"].where(
        frame["review_status"].isin(REVIEW_STATUSES),
        "Pending",
    )
    return frame.drop_duplicates(subset=["model_name", "filename"], keep="last")


def save_prediction_review(
    test_dir: Path,
    model_name: str,
    filename: str,
    review_status: str,
) -> None:
    if review_status not in REVIEW_STATUSES:
        raise ValueError(f"Unsupported review status: {review_status}")
    test_dir.mkdir(parents=True, exist_ok=True)
    reviews = load_prediction_reviews(test_dir)
    key = (reviews["model_name"] == model_name) & (reviews["filename"] == filename)
    row = {
        "model_name": model_name,
        "filename": filename,
        "review_status": review_status,
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if key.any():
        reviews.loc[key, list(row)] = list(row.values())
    else:
        reviews = pd.concat([reviews, pd.DataFrame([row])], ignore_index=True)
    reviews.sort_values(["model_name", "filename"]).to_csv(
        review_path(test_dir),
        index=False,
    )


def save_prediction_review_from_state(
    test_dir_text: str,
    model_name: str,
    filename: str,
    state_key: str,
) -> None:
    save_prediction_review(
        Path(test_dir_text),
        model_name,
        filename,
        str(st.session_state[state_key]),
    )


def attach_review_status(results: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    data = results.copy()
    if data.empty:
        data["review_status"] = pd.Series(dtype=str)
        data["reviewed_at"] = pd.Series(dtype=str)
        return data
    merged = data.merge(
        reviews,
        on=["model_name", "filename"],
        how="left",
    )
    merged["review_status"] = merged["review_status"].fillna("Pending")
    merged["reviewed_at"] = merged["reviewed_at"].fillna("")
    return merged


def parse_source_csv_lines(source_path: Path | None) -> list[str]:
    if source_path is None or not source_path.is_file():
        return []
    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    datasets: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = line.strip().strip('"').replace("\\", "/")
        if not normalized or not normalized.lower().endswith(".csv"):
            continue
        if normalized not in seen:
            datasets.append(normalized)
            seen.add(normalized)
    return datasets


def model_training_datasets(
    model_names: list[str],
    models_root: Path | None = None,
) -> dict[str, list[str]]:
    return {
        model_name: parse_source_csv_lines(
            find_model_source_csv_file(model_name, models_root)
        )
        for model_name in model_names
    }


def compact_dataset_list(datasets: list[str], limit: int = 3) -> str:
    if not datasets:
        return "-"
    names = [Path(value).name for value in datasets]
    shown = names[:limit]
    suffix = f" (+{len(names) - limit})" if len(names) > limit else ""
    return ", ".join(shown) + suffix


def model_summary_table(
    frame: pd.DataFrame,
    model_names: list[str],
    datasets_by_model: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        model_frame = frame[frame["model_name"] == model_name]
        effective_status = model_frame["review_status"].where(
            model_frame["review_status"] == "Fail",
            "Pass",
        )
        pass_count = int((effective_status == "Pass").sum())
        fail_count = int((model_frame["review_status"] == "Fail").sum())
        reviewed = pass_count + fail_count
        pass_rate = (pass_count / reviewed * 100.0) if reviewed else np.nan
        rows.append(
            {
                "Model Name": model_name,
                "Train Dataset": compact_dataset_list(
                    datasets_by_model.get(model_name, [])
                ),
                "#Pass": pass_count,
                "#Fail": fail_count,
                "Pass Rate": "-" if pd.isna(pass_rate) else f"{pass_rate:.1f}%",
            }
        )
    return pd.DataFrame(
        rows,
        columns=["Model Name", "Train Dataset", "#Pass", "#Fail", "Pass Rate"],
    )


def model_groups_from_summary(summary: pd.DataFrame) -> tuple[list[str], list[str]]:
    if summary.empty:
        return [], []
    pass_models = summary[
        (summary["#Pass"] > 0)
        & (summary["#Fail"] == 0)
    ]["Model Name"].astype(str).tolist()
    fail_models = summary[summary["#Fail"] > 0]["Model Name"].astype(str).tolist()
    return pass_models, fail_models


def dataset_intersection(
    model_names: list[str],
    datasets_by_model: dict[str, list[str]],
) -> list[str]:
    sets = [set(datasets_by_model.get(model_name, [])) for model_name in model_names]
    if not sets:
        return []
    return sorted(set.intersection(*sets), key=lambda value: value.lower())


def dataset_influence_table(
    pass_models: list[str],
    fail_models: list[str],
    datasets_by_model: dict[str, list[str]],
) -> pd.DataFrame:
    all_datasets = sorted(
        {
            dataset
            for datasets in datasets_by_model.values()
            for dataset in datasets
        },
        key=lambda value: value.lower(),
    )
    rows = []
    for dataset in all_datasets:
        pass_count = sum(dataset in datasets_by_model.get(model, []) for model in pass_models)
        fail_count = sum(dataset in datasets_by_model.get(model, []) for model in fail_models)
        pass_ratio = pass_count / len(pass_models) if pass_models else 0.0
        fail_ratio = fail_count / len(fail_models) if fail_models else 0.0
        rows.append(
            {
                "Dataset": dataset,
                "Pass Model Count": pass_count,
                "Fail Model Count": fail_count,
                "Pass Inclusion": pass_ratio,
                "Fail Inclusion": fail_ratio,
                "Pass Advantage": pass_ratio - fail_ratio,
                "Fail Advantage": fail_ratio - pass_ratio,
            }
        )
    return pd.DataFrame(rows)


def initialize_selected_image_state(all_images: list[str]) -> None:
    if not all_images:
        st.session_state["selected_image_index"] = 0
        st.session_state["selected_image"] = ""
        return
    current = st.session_state.get("selected_image")
    if current not in all_images:
        st.session_state["selected_image_index"] = 0
        st.session_state["selected_image"] = all_images[0]
        return
    st.session_state["selected_image_index"] = all_images.index(str(current))


def set_selected_image_by_index(all_images: list[str], index: int) -> None:
    if not all_images:
        return
    bounded = max(0, min(index, len(all_images) - 1))
    st.session_state["selected_image_index"] = bounded
    st.session_state["selected_image"] = all_images[bounded]


def sync_selected_image_from_selectbox(all_images: list[str]) -> None:
    selected = str(st.session_state.get("selected_image", ""))
    if selected in all_images:
        st.session_state["selected_image_index"] = all_images.index(selected)


@st.cache_data(show_spinner=False)
def load_display_image(path_text: str, mtime_ns: int) -> np.ndarray | None:
    path = Path(path_text)
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def load_label_mask(path_text: str, mtime_ns: int) -> np.ndarray | None:
    path = Path(path_text)
    if not path.is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None
    bgr = image[:, :, :3]
    label_mask = np.full(bgr.shape[:2], -1, dtype=np.int16)
    known = np.zeros(bgr.shape[:2], dtype=bool)
    for color, label in MASK_BGR_TO_LABEL.items():
        matches = np.all(bgr == np.array(color, dtype=np.uint8), axis=2)
        label_mask[matches] = label
        known |= matches
    if not known.any() or np.count_nonzero(~known) > 0:
        return None
    return label_mask


def model_sort_values(frame: pd.DataFrame, image_name: str, sort_by: str, class_name: str) -> list[str]:
    models = sorted(frame["model_name"].dropna().unique())
    image_frame = frame[frame["filename"] == image_name].copy()
    if sort_by == "Model name" or image_frame.empty:
        return models

    sort_map = {
        "Lithium Salt %": "li_salt_percent",
        "Background %": "background_percent",
        "Nonactive %": "nonactive_percent",
    }
    if sort_by in sort_map:
        column = sort_map[sort_by]
        ordered = image_frame.sort_values(column, ascending=False)["model_name"].tolist()
        return ordered + [model for model in models if model not in ordered]

    if sort_by == "Difference from model average":
        columns = list(CLASS_COLUMNS.values()) if class_name == "All classes" else [CLASS_COLUMNS[class_name]]
        means = image_frame[columns].mean(numeric_only=True)
        image_frame["_deviation"] = image_frame[columns].sub(means).abs().mean(axis=1)
        ordered = image_frame.sort_values("_deviation", ascending=False)["model_name"].tolist()
        return ordered + [model for model in models if model not in ordered]

    return models


def filtered_results(frame: pd.DataFrame, models: list[str], image_name: str, scope: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame[frame["model_name"].isin(models)].copy()
    if scope == "Selected image":
        data = data[data["filename"] == image_name].copy()
    return data


def selected_image_frame(frame: pd.DataFrame, image_name: str, models: list[str]) -> pd.DataFrame:
    return frame[(frame["filename"] == image_name) & (frame["model_name"].isin(models))].copy()


def long_percent_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["filename", "model_name", "class_name", "percent"])
    parts = []
    for class_name, column in CLASS_COLUMNS.items():
        part = frame[["filename", "model_name", column]].copy()
        part = part.rename(columns={column: "percent"})
        part["class_name"] = class_name
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def variability_stats(frame: pd.DataFrame) -> pd.DataFrame:
    long_frame = long_percent_frame(frame)
    if long_frame.empty:
        return pd.DataFrame()
    grouped = long_frame.groupby(["filename", "class_name"])["percent"]
    stats = grouped.agg(["mean", "min", "max", "std"]).reset_index()
    stats["std"] = stats["std"].fillna(0)
    stats["range"] = stats["max"] - stats["min"]
    stats["coefficient_of_variation"] = np.where(
        stats["mean"].abs() > 1.0e-9,
        stats["std"] / stats["mean"].abs(),
        np.nan,
    )
    stats["max_absolute_difference"] = stats["range"]
    return stats


def add_table_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    long_frame = long_percent_frame(frame)
    if long_frame.empty:
        return long_frame
    stats = variability_stats(frame)[
        ["filename", "class_name", "mean", "std", "range"]
    ].rename(
        columns={
            "mean": "image_class_mean",
            "std": "image_level_standard_deviation",
            "range": "image_level_range",
        }
    )
    merged = long_frame.merge(stats, on=["filename", "class_name"], how="left")
    merged["difference_from_image_mean"] = merged["percent"] - merged["image_class_mean"]
    merged["absolute_difference_from_image_mean"] = merged["difference_from_image_mean"].abs()
    merged["variability_status"] = np.where(
        (merged["image_level_standard_deviation"] > threshold)
        | (merged["image_level_range"] > threshold),
        "High variability",
        "Normal",
    )
    return merged


def most_variable_image(stats: pd.DataFrame) -> str:
    if stats.empty:
        return "-"
    by_image = stats.groupby("filename")["range"].mean().sort_values(ascending=False)
    return str(by_image.index[0]) if not by_image.empty else "-"


def model_average_deviation(frame: pd.DataFrame) -> pd.DataFrame:
    long_frame = long_percent_frame(frame)
    if long_frame.empty:
        return pd.DataFrame(columns=["model_name", "average_absolute_deviation"])
    long_frame["image_class_mean"] = long_frame.groupby(["filename", "class_name"])["percent"].transform("mean")
    long_frame["absolute_deviation"] = (long_frame["percent"] - long_frame["image_class_mean"]).abs()
    return (
        long_frame.groupby("model_name")["absolute_deviation"]
        .mean()
        .reset_index(name="average_absolute_deviation")
        .sort_values("average_absolute_deviation", ascending=False)
    )


def model_reliability_scores(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "model_name",
                "reliability_score",
                "average_absolute_deviation",
                "missing_result_count",
                "bad_percent_sum_count",
                "result_count",
            ]
        )

    deviations = model_average_deviation(frame)
    models = pd.DataFrame({"model_name": sorted(frame["model_name"].dropna().unique())})
    summary = frame.groupby("model_name").agg(
        result_count=("filename", "count"),
        missing_result_count=("mask_exists", lambda values: int((~values.fillna(False)).sum())),
        bad_percent_sum_count=(
            "percent_sum",
            lambda values: int(((values < 99.5) | (values > 100.5)).sum()),
        ),
    ).reset_index()
    scores = models.merge(deviations, on="model_name", how="left").merge(
        summary,
        on="model_name",
        how="left",
    )
    scores["average_absolute_deviation"] = scores["average_absolute_deviation"].fillna(0.0)
    scores["missing_result_count"] = scores["missing_result_count"].fillna(0).astype(int)
    scores["bad_percent_sum_count"] = scores["bad_percent_sum_count"].fillna(0).astype(int)
    scores["result_count"] = scores["result_count"].fillna(0).astype(int)
    scores["reliability_score"] = (
        100.0
        - scores["average_absolute_deviation"].clip(lower=0) * 5.0
        - scores["missing_result_count"] * 10.0
        - scores["bad_percent_sum_count"] * 10.0
    ).clip(lower=0.0, upper=100.0)
    return scores.sort_values(
        ["reliability_score", "average_absolute_deviation", "model_name"],
        ascending=[False, True, True],
    )


def most_inconsistent_model(frame: pd.DataFrame) -> str:
    deviations = model_average_deviation(frame)
    if deviations.empty:
        return "-"
    return str(deviations.iloc[0]["model_name"])


def pairwise_model_differences(frame: pd.DataFrame, class_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if class_name == "Overall":
        columns = list(CLASS_COLUMNS.values())
    else:
        columns = [CLASS_COLUMNS[class_name]]

    models = sorted(frame["model_name"].dropna().unique())
    matrix = pd.DataFrame(0.0, index=models, columns=models)
    for model_a, model_b in combinations(models, 2):
        a = frame[frame["model_name"] == model_a][["filename", *columns]]
        b = frame[frame["model_name"] == model_b][["filename", *columns]]
        merged = a.merge(b, on="filename", suffixes=("_a", "_b"))
        if merged.empty:
            diff = np.nan
        else:
            diffs = [
                (merged[f"{column}_a"] - merged[f"{column}_b"]).abs()
                for column in columns
            ]
            diff = float(pd.concat(diffs, axis=1).mean(axis=1).mean())
        matrix.loc[model_a, model_b] = diff
        matrix.loc[model_b, model_a] = diff
    return matrix


def build_consensus(mask_items: list[tuple[str, np.ndarray]]) -> dict | None:
    if len(mask_items) < 2:
        return None
    shapes = {mask.shape for _model, mask in mask_items}
    if len(shapes) != 1:
        return None

    stack = np.stack([mask for _model, mask in mask_items], axis=0)
    valid = np.all(stack >= 0, axis=0)
    if not valid.any():
        return None

    consensus = np.full(stack.shape[1:], -1, dtype=np.int16)
    max_votes = np.zeros(stack.shape[1:], dtype=np.int16)
    for label in sorted(LABEL_NAMES):
        votes = np.sum(stack == label, axis=0)
        update = votes > max_votes
        consensus[update] = label
        max_votes[update] = votes[update]

    disagreement_score = 1.0 - (max_votes.astype(np.float32) / float(stack.shape[0]))
    agreement_pixels = valid & (max_votes == stack.shape[0])
    total_valid = int(valid.sum())
    agreement_ratio = float(agreement_pixels.sum() / total_valid)
    class_rows = []
    for label, name in LABEL_NAMES.items():
        class_pixels = valid & (consensus == label)
        if class_pixels.any():
            ratio = float(np.count_nonzero(class_pixels & ~agreement_pixels) / np.count_nonzero(class_pixels))
        else:
            ratio = np.nan
        class_rows.append({"class_name": name, "disagreement_ratio": ratio})

    return {
        "consensus": consensus,
        "disagreement": np.where(valid, disagreement_score, np.nan),
        "agreement_ratio": agreement_ratio,
        "disagreement_ratio": 1.0 - agreement_ratio,
        "class_disagreement": pd.DataFrame(class_rows),
    }


def consensus_to_rgb(consensus: np.ndarray) -> np.ndarray:
    image = np.zeros((*consensus.shape, 3), dtype=np.uint8)
    for label, color in LABEL_TO_RGB.items():
        image[consensus == label] = color
    return image


def status_counts(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    data["missing_data"] = ~(data["original_exists"].fillna(False) & data["mask_exists"].fillna(False))
    data["bad_percent_sum"] = (data["percent_sum"] < 99.5) | (data["percent_sum"] > 100.5)
    long_metrics = add_table_metrics(data, threshold)
    high = long_metrics[long_metrics["variability_status"] == "High variability"]
    return pd.DataFrame(
        {
            "status": ["Missing image/data", "Percent sum outside normal range", "High variability rows"],
            "count": [int(data["missing_data"].sum()), int(data["bad_percent_sum"].sum()), int(len(high))],
        }
    )


def model_card_flags(image_frame: pd.DataFrame, threshold: float) -> dict[str, dict[str, bool | float]]:
    if image_frame.empty:
        return {}

    li_values = pd.to_numeric(image_frame["li_salt_percent"], errors="coerce")
    valid_li = li_values.dropna()
    show_extremes = len(valid_li) >= 2
    li_max = float(valid_li.max()) if show_extremes else np.nan
    li_min = float(valid_li.min()) if show_extremes else np.nan
    li_mean = float(valid_li.mean()) if not valid_li.empty else np.nan
    flags: dict[str, dict[str, bool | float]] = {}

    for row_index, row in image_frame.reset_index(drop=True).iterrows():
        li_value = li_values.iloc[row_index]
        diff = float(li_value) - li_mean if pd.notna(li_value) and pd.notna(li_mean) else np.nan
        flags[str(row["model_name"])] = {
            "highest_li": bool(show_extremes and np.isclose(float(li_value), li_max)),
            "lowest_li": bool(show_extremes and np.isclose(float(li_value), li_min)),
            "variability_warning": bool(pd.notna(diff) and abs(diff) > threshold),
            "li_diff": diff,
        }
    return flags


def render_metric_cards(frame: pd.DataFrame, stats: pd.DataFrame) -> None:
    cols = st.columns(6)
    cols[0].metric("Test images", int(frame["filename"].nunique()) if not frame.empty else 0)
    cols[1].metric("Compared models", int(frame["model_name"].nunique()) if not frame.empty else 0)
    cols[2].metric("Avg Lithium Salt %", f"{frame['li_salt_percent'].mean():.3f}" if not frame.empty else "-")
    cols[3].metric("Avg Nonactive %", f"{frame['nonactive_percent'].mean():.3f}" if not frame.empty else "-")
    cols[4].metric("Most variable image", most_variable_image(stats))
    cols[5].metric("Most inconsistent model", most_inconsistent_model(frame))


def render_original_image(test_dir: Path, image_name: str) -> None:
    image_path = test_dir / image_name
    st.subheader("Model Input Image")
    image = load_display_image(str(image_path), path_mtime_ns(image_path))
    if image is None:
        st.warning(f"Original image is missing: {image_name}")
        return
    st.image(image, caption=image_name, use_container_width=True)


def format_percent(value: float) -> str:
    return "-" if pd.isna(value) else f"{float(value):.4f}%"


def format_signed_pp(value: float) -> str:
    return "-" if pd.isna(value) else f"{float(value):+.4f} pp"


def missing_image_placeholder(label: str) -> np.ndarray:
    image = np.full((420, 420, 3), 248, dtype=np.uint8)
    cv2.putText(
        image,
        "Image missing",
        (86, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (100, 116, 139),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label[:32],
        (36, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 116, 139),
        1,
        cv2.LINE_AA,
    )
    return image


def add_review_border(
    image: np.ndarray,
    review_status: str,
    border_px: int = 12,
) -> np.ndarray:
    if review_status != "Fail":
        return image
    color = FAIL_BORDER_COLOR
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3:
        return image

    channels = image.shape[2]
    if channels == 4:
        fill = np.array([*color, 255], dtype=image.dtype)
    else:
        fill = np.array(color[:channels], dtype=image.dtype)

    h, w = image.shape[:2]
    bordered = np.empty((h + border_px * 2, w + border_px * 2, channels), dtype=image.dtype)
    bordered[:, :] = fill
    bordered[border_px:border_px + h, border_px:border_px + w] = image
    return bordered


def prediction_caption(row: pd.Series, flags: dict[str, bool | float]) -> str:
    parts = [str(row["model_name"])]
    status = str(row.get("review_status", "Pending"))
    if status == "Fail":
        parts.append(status)
    if flags.get("highest_li"):
        parts.append("Highest Li")
    if flags.get("lowest_li"):
        parts.append("Lowest Li")
    if flags.get("variability_warning"):
        parts.append("Variability warning")
    return " / ".join(parts)


def render_model_cards(
    image_frame: pd.DataFrame,
    threshold: float,
    test_dir: Path,
    selected_image: str,
) -> None:
    if image_frame.empty:
        st.info("No model results are available for this image.")
        return
    flags_by_model = model_card_flags(image_frame, threshold)
    original_path = test_dir / selected_image
    original = load_display_image(str(original_path), path_mtime_ns(original_path))
    images = [
        original
        if original is not None
        else missing_image_placeholder(selected_image)
    ]
    captions = [f"Model Input Image / {selected_image}"]
    stats_rows = []

    for _, row in image_frame.iterrows():
        model_name = str(row["model_name"])
        mask_path = Path(str(row["mask_path"]))
        mask = load_display_image(str(mask_path), path_mtime_ns(mask_path))
        review_status = str(row.get("review_status", "Pending"))
        if review_status not in REVIEW_STATUSES:
            review_status = "Pending"
        display_mask = (
            mask
            if mask is not None
            else missing_image_placeholder(model_name)
        )
        images.append(
            add_review_border(display_mask, review_status)
        )
        flags = flags_by_model.get(model_name, {})
        captions.append(prediction_caption(row, flags))
        stats_rows.append(
            {
                "Model": model_name,
                "Background %": format_percent(row["background_percent"]),
                "Li %": format_percent(row["li_salt_percent"]),
                "Nonactive %": format_percent(row["nonactive_percent"]),
            }
        )

    st.image(images, caption=captions, width=420)

    st.subheader("Prediction Review")
    control_columns = st.columns(min(6, max(1, len(image_frame))))
    for index, (_, row) in enumerate(image_frame.iterrows()):
        col = control_columns[index % len(control_columns)]
        with col:
            st.markdown(f"**{row['model_name']}**")
            current_status = str(row.get("review_status", "Pending"))
            if current_status not in REVIEW_STATUSES:
                current_status = "Pending"
            input_status = current_status if current_status in REVIEW_INPUT_STATUSES else "Pass"
            review_key = f"review::{row['model_name']}::{row['filename']}"
            if review_key not in st.session_state:
                st.session_state[review_key] = input_status
            elif st.session_state[review_key] not in REVIEW_INPUT_STATUSES:
                st.session_state[review_key] = input_status
            st.radio(
                f"Review {row['model_name']}",
                REVIEW_INPUT_STATUSES,
                index=REVIEW_INPUT_STATUSES.index(st.session_state[review_key]),
                horizontal=True,
                key=review_key,
                label_visibility="collapsed",
                on_change=save_prediction_review_from_state,
                args=(
                    str(test_dir),
                    str(row["model_name"]),
                    str(row["filename"]),
                    review_key,
                ),
            )

    st.subheader("Prediction Percentages")
    st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)


def render_percentage_comparison(image_frame: pd.DataFrame) -> None:
    long_frame = long_percent_frame(image_frame)
    if long_frame.empty:
        st.info("No percentage data available.")
        return
    avg_rows = (
        long_frame.groupby("class_name")["percent"]
        .mean()
        .reset_index()
        .assign(model_name="Selected model mean")
    )
    chart_data = pd.concat([long_frame, avg_rows], ignore_index=True)
    fig = px.bar(
        chart_data,
        x="model_name",
        y="percent",
        color="class_name",
        barmode="group",
        title="Class Percentages by Model",
        labels={"model_name": "Model", "percent": "Percent", "class_name": "Class"},
    )
    st.plotly_chart(fig, use_container_width=True)

    table = long_frame.copy()
    table["selected_model_mean"] = table.groupby("class_name")["percent"].transform("mean")
    table["difference_from_mean"] = table["percent"] - table["selected_model_mean"]
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_heatmap(frame: pd.DataFrame, class_name: str) -> None:
    column = CLASS_COLUMNS[class_name]
    pivot = frame.pivot_table(index="filename", columns="model_name", values=column, aggfunc="mean")
    if pivot.empty:
        st.info(f"No {class_name} data available.")
        return
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Viridis",
        title=f"Image x Model Heatmap: {class_name} %",
        labels={"x": "Model", "y": "Image", "color": "%"},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_variability(frame: pd.DataFrame, threshold: float) -> None:
    stats = variability_stats(frame)
    if stats.empty:
        st.info("No variability data available.")
        return
    st.subheader("Class Heatmaps")
    for class_name in CLASS_COLUMNS:
        render_heatmap(frame, class_name)

    st.subheader("Per-image Variability")
    image_var = stats.groupby("filename")[["std", "range"]].mean().reset_index()
    fig = px.bar(
        image_var,
        x="filename",
        y=["std", "range"],
        barmode="group",
        title="Average Standard Deviation and Range by Image",
        labels={"value": "Percentage points", "filename": "Image"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Per-model Average Deviation")
    deviations = model_average_deviation(frame)
    fig = px.bar(
        deviations,
        x="model_name",
        y="average_absolute_deviation",
        title="Average Absolute Deviation from Image/Class Mean",
        labels={"model_name": "Model", "average_absolute_deviation": "Percentage points"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Class Percentage Distribution")
    long_frame = long_percent_frame(frame)
    fig = px.box(
        long_frame,
        x="class_name",
        y="percent",
        color="class_name",
        points="all",
        title="Class Percentage Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pairwise Model Difference")
    pairwise_class = st.selectbox(
        "Pairwise difference class",
        ["Overall", *CLASS_COLUMNS.keys()],
        key="pairwise_class",
    )
    matrix = pairwise_model_differences(frame, pairwise_class)
    if matrix.empty:
        st.info("At least two models are required for pairwise comparison.")
    else:
        fig = px.imshow(
            matrix,
            aspect="auto",
            color_continuous_scale="Reds",
            title="Pairwise Mean Absolute Difference",
            labels={"x": "Model", "y": "Model", "color": "pp"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Unstable Result Detection")
    status = status_counts(frame, threshold)
    st.dataframe(status, use_container_width=True, hide_index=True)
    st.dataframe(
        stats.assign(
            variability_status=np.where(
                (stats["std"] > threshold) | (stats["range"] > threshold),
                "High variability",
                "Normal",
            )
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_disagreement(image_frame: pd.DataFrame) -> None:
    mask_items = []
    missing = []
    for row in image_frame.itertuples(index=False):
        mask_path = Path(row.mask_path)
        label_mask = load_label_mask(str(mask_path), path_mtime_ns(mask_path))
        if label_mask is None:
            missing.append(row.model_name)
        else:
            mask_items.append((row.model_name, label_mask))

    if missing:
        st.warning(
            "Pixel-level disagreement is disabled for models with non-recoverable or missing masks: "
            + ", ".join(missing)
        )
    consensus = build_consensus(mask_items)
    if consensus is None:
        st.info("Pixel-level disagreement requires at least two same-size class mask images.")
        return

    cols = st.columns(3)
    cols[0].metric("Agreement ratio", f"{consensus['agreement_ratio'] * 100.0:.3f}%")
    cols[1].metric("Disagreement ratio", f"{consensus['disagreement_ratio'] * 100.0:.3f}%")
    cols[2].metric("Compared masks", len(mask_items))

    left, right = st.columns(2)
    with left:
        st.subheader("Consensus Prediction Mask")
        st.image(consensus_to_rgb(consensus["consensus"]), use_container_width=True)
    with right:
        st.subheader("Pixel Disagreement Heatmap")
        fig = px.imshow(
            consensus["disagreement"],
            color_continuous_scale="Magma",
            title="Per-pixel Disagreement Score",
            labels={"color": "Disagreement"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Class Disagreement Ratio")
    table = consensus["class_disagreement"].copy()
    table["disagreement_percent"] = table["disagreement_ratio"] * 100.0
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_raw_data(frame: pd.DataFrame, threshold: float) -> None:
    st.subheader("Long-format Result Table")
    search = st.text_input("Search table", "")
    table = add_table_metrics(frame, threshold)
    if search:
        query = search.lower()
        table = table[
            table.astype(str).apply(lambda row: row.str.lower().str.contains(query).any(), axis=1)
        ]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="filtered_test_model_results.csv",
        mime="text/csv",
    )


def render_model_analysis(frame: pd.DataFrame, selected_image: str, threshold: float) -> None:
    st.subheader("Selected Image Percentage Comparison")
    image_frame = frame[frame["filename"] == selected_image].copy()
    render_percentage_comparison(image_frame)

    st.subheader("Model-level Summary")
    if frame.empty:
        st.info("No model data available.")
        return
    summary = frame.groupby("model_name")[
        ["li_salt_percent", "background_percent", "nonactive_percent", "edge_percent"]
    ].mean().reset_index()
    st.dataframe(summary, use_container_width=True, hide_index=True)

    deviations = model_average_deviation(frame)
    if not deviations.empty:
        deviations["status"] = np.where(
            deviations["average_absolute_deviation"] > threshold,
            "High deviation",
            "Normal",
        )
        st.dataframe(deviations, use_container_width=True, hide_index=True)

    st.subheader("Reliability Ranking")
    reliability = model_reliability_scores(frame)
    if reliability.empty:
        st.info("No reliability data available.")
        return
    best = reliability.iloc[0]
    st.info(
        "Best current candidate: "
        f"{best['model_name']} "
        f"(score {best['reliability_score']:.1f}/100, "
        f"avg deviation {best['average_absolute_deviation']:.4f} pp). "
        "This is a consistency score, not ground-truth accuracy."
    )
    st.dataframe(reliability, use_container_width=True, hide_index=True)


def render_training_csv_matrix(model_names: list[str]) -> None:
    st.subheader("Enhanced Training CSV Coverage")
    if not model_names:
        st.info("Select at least one model to inspect training CSV coverage.")
        return

    signature = model_source_csv_signature(model_names)
    matrix, missing_models = load_enhanced_source_csv_matrix(signature)
    if missing_models:
        st.caption(
            "source_csvs.txt not found for: "
            + ", ".join(sorted(missing_models, key=str.lower))
        )

    if matrix.empty:
        st.info("No enhanced CSV entries were found in the selected models' source_csvs.txt files.")
        return

    model_columns = [column for column in matrix.columns if column != "Enhanced CSV"]
    styled = matrix.style.set_properties(
        subset=model_columns,
        **{"text-align": "center", "font-weight": "700"},
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("Only source CSV entries whose file name starts with enhanced_ are shown.")


def render_model_summary(summary: pd.DataFrame) -> None:
    st.subheader("Model Summary Table")
    if summary.empty:
        st.info("No selected model results are available.")
        return

    display = summary.copy()
    st.dataframe(display, use_container_width=True, hide_index=True)

    reviewed = int(display["#Pass"].sum() + display["#Fail"].sum())
    best = display.copy()
    best["_pass_rate_value"] = np.where(
        best["#Pass"] + best["#Fail"] > 0,
        best["#Pass"] / (best["#Pass"] + best["#Fail"]),
        -1.0,
    )
    best = best.sort_values(
        ["_pass_rate_value", "#Pass", "#Fail", "Model Name"],
        ascending=[False, False, True, True],
    )
    cols = st.columns(4)
    cols[0].metric("Reviewed Predictions", reviewed)
    cols[1].metric("Pass", int(display["#Pass"].sum()))
    cols[2].metric("Fail", int(display["#Fail"].sum()))
    cols[3].metric("Best Model", str(best.iloc[0]["Model Name"]) if not best.empty else "-")


def render_fail_image_table(frame: pd.DataFrame) -> None:
    st.subheader("Fail Analysis")
    fail_frame = frame[frame["review_status"] == "Fail"].copy()
    if fail_frame.empty:
        st.info("No manually marked Fail predictions yet.")
        return
    table = fail_frame[
        [
            "filename",
            "model_name",
            "li_salt_percent",
            "background_percent",
            "nonactive_percent",
            "edge_percent",
            "mask_exists",
            "reviewed_at",
        ]
    ].sort_values(["filename", "model_name"])
    st.dataframe(table, use_container_width=True, hide_index=True)

    counts = (
        fail_frame.groupby("filename")["model_name"]
        .nunique()
        .reset_index(name="Fail Model Count")
        .sort_values(["Fail Model Count", "filename"], ascending=[False, True])
    )
    st.caption("Images at the top are failing across more selected models.")
    st.dataframe(counts, use_container_width=True, hide_index=True)


def render_dataset_list(title: str, datasets: list[str]) -> None:
    st.markdown(f"**{title}**")
    if not datasets:
        st.caption("No dataset matched this condition.")
        return
    st.dataframe(
        pd.DataFrame({"Dataset": datasets}),
        use_container_width=True,
        hide_index=True,
    )


def render_dataset_analysis(
    summary: pd.DataFrame,
    datasets_by_model: dict[str, list[str]],
) -> None:
    st.subheader("Dataset Analysis")
    pass_models, fail_models = model_groups_from_summary(summary)
    st.caption(
        "Pass models are models with at least one manual Pass and no manual Fail. "
        "Fail models are models with one or more manual Fail predictions."
    )

    cols = st.columns(2)
    with cols[0]:
        st.metric("Pass Models", len(pass_models))
        st.caption(", ".join(pass_models) if pass_models else "-")
    with cols[1]:
        st.metric("Fail Models", len(fail_models))
        st.caption(", ".join(fail_models) if fail_models else "-")

    common_pass = dataset_intersection(pass_models, datasets_by_model)
    common_fail = dataset_intersection(fail_models, datasets_by_model)
    left, right = st.columns(2)
    with left:
        render_dataset_list("Common datasets in Pass models", common_pass)
    with right:
        render_dataset_list("Common datasets in Fail models", common_fail)

    influence = dataset_influence_table(pass_models, fail_models, datasets_by_model)
    if influence.empty:
        st.info("No source_csvs.txt dataset information found for selected models.")
        return

    st.subheader("Dataset Influence Candidates")
    pass_heavy = influence[
        influence["Pass Model Count"] > influence["Fail Model Count"]
    ].sort_values(
        ["Pass Advantage", "Pass Model Count", "Dataset"],
        ascending=[False, False, True],
    )
    fail_heavy = influence[
        influence["Fail Model Count"] > influence["Pass Model Count"]
    ].sort_values(
        ["Fail Advantage", "Fail Model Count", "Dataset"],
        ascending=[False, False, True],
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**More common in Pass models**")
        st.dataframe(
            pass_heavy[
                ["Dataset", "Pass Model Count", "Fail Model Count", "Pass Advantage"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    with right:
        st.markdown("**More common in Fail models**")
        st.dataframe(
            fail_heavy[
                ["Dataset", "Pass Model Count", "Fail Model Count", "Fail Advantage"]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("All dataset coverage"):
        coverage = influence.copy()
        coverage["Pass Inclusion"] = (coverage["Pass Inclusion"] * 100.0).round(1)
        coverage["Fail Inclusion"] = (coverage["Fail Inclusion"] * 100.0).round(1)
        st.dataframe(coverage, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Test Model Comparison Dashboard")
    st.caption(
        "Manual Pass/Fail review for test_model outputs, model comparison, and training dataset analysis."
    )

    result_dirs = discover_result_dirs(TEST_DIR)
    signature = result_signature(result_dirs)
    results = load_test_results(str(TEST_DIR), signature)
    reviews = load_prediction_reviews(TEST_DIR)
    results = attach_review_status(results, reviews)
    original_images = discover_original_images(TEST_DIR, result_dirs)

    if results.empty:
        st.warning("No test model results found. Expected data/test/<model-name>/test_results.csv.")
        return

    all_images = sorted(set(results["filename"].dropna().astype(str)))
    for path in original_images:
        all_images.append(path.relative_to(TEST_DIR).as_posix())
    all_images = sorted(set(all_images))
    all_models = sorted(results["model_name"].dropna().unique())
    initialize_selected_image_state(all_images)

    with st.sidebar:
        st.header("Filters")
        nav_prev, nav_mid, nav_next = st.columns([1, 1.2, 1])
        with nav_prev:
            st.button(
                "Previous",
                use_container_width=True,
                disabled=st.session_state["selected_image_index"] <= 0,
                on_click=set_selected_image_by_index,
                args=(all_images, st.session_state["selected_image_index"] - 1),
            )
        with nav_mid:
            st.caption(
                f"{st.session_state['selected_image_index'] + 1} / {len(all_images)}"
                if all_images
                else "0 / 0"
            )
        with nav_next:
            st.button(
                "Next",
                use_container_width=True,
                disabled=st.session_state["selected_image_index"] >= len(all_images) - 1,
                on_click=set_selected_image_by_index,
                args=(all_images, st.session_state["selected_image_index"] + 1),
            )
        selected_image = st.selectbox(
            "Test image selector",
            all_images,
            key="selected_image",
            on_change=sync_selected_image_from_selectbox,
            args=(all_images,),
        )
        selected_models = st.multiselect(
            "Model multiselect",
            all_models,
            default=all_models,
        )
        class_selection = st.selectbox(
            "Class selector",
            ["All classes", *CLASS_COLUMNS.keys()],
        )
        sort_by = st.selectbox(
            "Sort models by",
            [
                "Model name",
                "Lithium Salt %",
                "Background %",
                "Nonactive %",
                "Difference from model average",
            ],
        )
        scope = st.radio(
            "Analysis scope",
            ["Selected image", "All test images"],
            horizontal=False,
        )
        threshold = st.slider(
            "Variability threshold (percentage points)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )

    if not selected_models:
        st.warning("Select at least one model.")
        return

    ordered_models = model_sort_values(results, selected_image, sort_by, class_selection)
    selected_models = [model for model in ordered_models if model in selected_models]
    scoped = filtered_results(results, selected_models, selected_image, scope)
    selected_results = results[results["model_name"].isin(selected_models)].copy()
    datasets_by_model = model_training_datasets(selected_models)
    summary = model_summary_table(selected_results, selected_models, datasets_by_model)

    render_model_summary(summary)
    render_fail_image_table(selected_results)

    tab_image, tab_dataset = st.tabs(
        [
            "Image Comparison",
            "Dataset Analysis",
        ]
    )

    image_frame = selected_image_frame(results, selected_image, selected_models)

    with tab_image:
        st.subheader("Prediction Results")
        render_model_cards(image_frame, threshold, TEST_DIR, selected_image)

    with tab_dataset:
        render_dataset_analysis(summary, datasets_by_model)


if __name__ == "__main__":
    main()
