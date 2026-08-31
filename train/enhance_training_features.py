import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.ml.feature_extractor import FeatureExtractor
from modules.utils.image_io import quiet_imread

logger = logging.getLogger(__name__)

REQUIRED_LABEL_COLUMNS = ("filename", "x", "y", "label")
OUTPUT_FEATURE_COLUMNS = FeatureExtractor.ENHANCED_FEATURE_COLUMNS
OUTPUT_COLUMNS = ("filename", "x", "y", "label", *OUTPUT_FEATURE_COLUMNS)
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data"
DEFAULT_IMAGE_ROOTS = (
    PROJECT_ROOT / "lit_cells_image",
    PROJECT_ROOT / "data",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ENHANCED_PREFIX = "enhanced_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate enhanced per-point training CSVs from existing label CSVs. "
            "By default, scans data/ and writes enhanced_<source>.csv next to each source CSV."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV files or directories containing label CSV files. Defaults to data/.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        help="Root directory containing cropped/QC-ready images. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional single combined output CSV path. If omitted, each source CSV is "
            "written next to itself as enhanced_<source>.csv."
        ),
    )
    parser.add_argument(
        "--config",
        default="config/thresholds.yaml",
        help="Path to thresholds.yaml for feature extraction parameters.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing enhanced_*.csv outputs.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def is_enhanced_csv(path: Path) -> bool:
    return path.name.lower().startswith(ENHANCED_PREFIX)


def enhanced_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{ENHANCED_PREFIX}{csv_path.name}")


def has_required_label_columns(csv_path: Path) -> bool:
    try:
        header = pd.read_csv(csv_path, nrows=0)
    except Exception as exc:
        logger.warning("Skipping unreadable CSV: %s (%s)", csv_path, exc)
        return False
    return all(column in header.columns for column in REQUIRED_LABEL_COLUMNS)


def collect_csv_files(inputs: list[str]) -> list[Path]:
    if not inputs:
        inputs = [str(DEFAULT_INPUT_DIR)]

    csv_files: list[Path] = []
    for item in inputs:
        path = resolve_project_path(item)
        if path.is_dir():
            csv_files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            csv_files.append(path)
        else:
            raise RuntimeError(f"Input path is not a CSV file or directory: {path}")

    unique_files = sorted({path.resolve() for path in csv_files})
    source_files: list[Path] = []
    for csv_path in unique_files:
        if is_enhanced_csv(csv_path):
            logger.debug("Skipping generated enhanced CSV: %s", csv_path)
            continue
        if not has_required_label_columns(csv_path):
            logger.warning("Skipping non-label CSV without %s: %s", REQUIRED_LABEL_COLUMNS, csv_path)
            continue
        source_files.append(csv_path)

    if not source_files:
        raise RuntimeError("No source label CSV files found.")
    return source_files


def load_label_frame(csv_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        missing = [column for column in REQUIRED_LABEL_COLUMNS if column not in frame.columns]
        if missing:
            raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")
        frame = frame.loc[:, REQUIRED_LABEL_COLUMNS].copy()
        frame["source_csv"] = str(csv_path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["x"] = pd.to_numeric(data["x"], errors="raise").astype(int)
    data["y"] = pd.to_numeric(data["y"], errors="raise").astype(int)
    return data.sort_values(
        by=["source_csv", "filename", "x", "y"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_image_roots(csv_files: list[Path], explicit_roots: list[str]) -> list[Path]:
    roots: list[Path] = []
    for root in explicit_roots:
        roots.append(resolve_project_path(root))
    roots.extend(csv_path.parent for csv_path in csv_files)
    roots.extend(DEFAULT_IMAGE_ROOTS)

    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_image_index(image_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in image_roots:
        if root.is_file() and root.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(root.name, root)
            continue
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                index.setdefault(path.name, path)
    return index


def resolve_image_path(filename: str, csv_path: Path, image_index: dict[str, Path]) -> Path:
    raw_path = Path(str(filename))
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            Path.cwd() / raw_path,
            csv_path.parent / raw_path,
        ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    basename = raw_path.name
    if basename in image_index:
        return image_index[basename]
    raise RuntimeError(
        f"Image not found for filename='{filename}'. Pass --image-root pointing to cropped/QC-ready images."
    )


def build_enhanced_frame(
    labels: pd.DataFrame,
    image_index: dict[str, Path],
    config_path: str | Path,
) -> pd.DataFrame:
    extractor = FeatureExtractor(config_path)
    output_parts: list[pd.DataFrame] = []

    for (source_csv, filename), group in labels.groupby(["source_csv", "filename"], sort=False):
        csv_path = Path(source_csv)
        image_path = resolve_image_path(str(filename), csv_path, image_index)
        image = quiet_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")

        batch = extractor.extract(image, feature_columns=OUTPUT_FEATURE_COLUMNS)
        points = list(zip(group["x"].astype(int), group["y"].astype(int)))
        feature_values, valid = extractor.select_point_features(batch, points)
        if not np.all(valid):
            bad_rows = group.loc[~valid, ["filename", "x", "y"]].head(10)
            raise RuntimeError(
                "Some label points are outside the cropped image or transparent alpha region:\n"
                + bad_rows.to_string(index=False)
            )

        feature_frame = pd.DataFrame(feature_values, columns=OUTPUT_FEATURE_COLUMNS, index=group.index)
        base_frame = group.loc[:, ["filename", "x", "y", "label"]].copy()
        output_parts.append(pd.concat([base_frame, feature_frame], axis=1))

    return pd.concat(output_parts, ignore_index=True).loc[:, OUTPUT_COLUMNS]


def write_per_source_outputs(
    csv_files: list[Path],
    image_index: dict[str, Path],
    config_path: str | Path,
    *,
    overwrite: bool,
) -> list[Path]:
    written: list[Path] = []
    for csv_path in csv_files:
        output_path = enhanced_output_path(csv_path)
        if output_path.exists() and not overwrite:
            logger.info("Skipping existing output: %s (use --overwrite to regenerate)", output_path)
            continue
        labels = load_label_frame([csv_path])
        enhanced = build_enhanced_frame(labels, image_index, config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.to_csv(output_path, index=False, float_format="%.6f")
        written.append(output_path)
        logger.info("Saved %s (%d rows)", output_path, len(enhanced))
    return written


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    csv_files = collect_csv_files(args.inputs)
    image_roots = build_image_roots(csv_files, args.image_root)
    image_index = build_image_index(image_roots)
    config_path = resolve_project_path(args.config)
    logger.info("Found %d source label CSV file(s)", len(csv_files))
    logger.info("Indexed %d image file(s) from %d root(s)", len(image_index), len(image_roots))

    if args.output:
        labels = load_label_frame(csv_files)
        enhanced = build_enhanced_frame(labels, image_index, config_path)
        output_path = resolve_project_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.to_csv(output_path, index=False, float_format="%.6f")
        logger.info("Saved combined enhanced training features to %s (%d rows)", output_path, len(enhanced))
        return

    written = write_per_source_outputs(
        csv_files,
        image_index,
        config_path,
        overwrite=bool(args.overwrite),
    )
    logger.info("Done. Wrote %d enhanced CSV file(s).", len(written))


if __name__ == "__main__":
    main()
