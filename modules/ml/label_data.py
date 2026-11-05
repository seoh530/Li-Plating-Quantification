"""Validate sparse annotations before feature generation or model training."""

from pathlib import Path
from functools import lru_cache
import re

import numpy as np
import pandas as pd


@lru_cache(maxsize=8192)
def image_key(filename: str, source_csv: str) -> str:
    path = Path(str(filename).replace("\\", "/"))
    if not path.is_absolute():
        path = Path(source_csv).parent / path
    return str(path.resolve())


def validate_label_rows(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        raise RuntimeError("No labeled pixels found.")
    result = data.copy()
    if result[["filename", "label", "source_csv"]].isna().any().any():
        raise RuntimeError("Label CSV contains an empty filename, label, or source.")
    if not result["filename"].astype(str).str.strip().ne("").all():
        raise RuntimeError("Label CSV contains an empty filename.")
    if not result["label"].isin(["b", "l", "n"]).all():
        raise RuntimeError("Labels must be b, l, or n.")
    for column in ("x", "y"):
        values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=float)
        if not np.all(np.isfinite(values) & (values >= 0) & (values == np.floor(values))):
            raise RuntimeError(f"Label coordinate {column} must be a non-negative integer.")
        result[column] = values.astype(np.int64)
    result["_image_key"] = [
        image_key(filename, source)
        for filename, source in zip(result["filename"], result["source_csv"])
    ]
    keys = ["_image_key", "x", "y"]
    conflicts = result.groupby(keys, sort=False)["label"].nunique()
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index.tolist()[:3]
        raise RuntimeError(f"Conflicting labels for the same image pixel: {examples}")
    return result.drop_duplicates(keys, keep="first").reset_index(drop=True)


def validation_groups(data: pd.DataFrame, group_by: str = "cell") -> np.ndarray:
    """Group both faces/layers of a cell together; fall back to source CSV."""
    if group_by not in {"cell", "analysis", "image", "csv"}:
        raise RuntimeError("validation_group_by must be cell, analysis, image, or csv.")
    groups = []
    identities = data[["source_csv", "filename"]].drop_duplicates()
    for row in identities.itertuples(index=False):
        source = Path(row.source_csv)
        key = image_key(row.filename, row.source_csv)
        text = str(source).replace("\\", "/") + "/" + str(row.filename).replace("\\", "/")
        analysis = re.search(r"(?<![A-Za-z0-9])([A-Z]{3}\d{5}|TEST(?:\d{2})?)(?![A-Za-z0-9])", text, re.I)
        cell = re.search(r"(?:^|/)cell\s*#\s*(\d+)(?:/|$)", text, re.I)
        if cell is None and analysis:
            cell = re.search(re.escape(analysis.group(1)) + r"_(\d+)_[FB](?:\.|_)", text, re.I)
        if group_by == "image":
            group = key
        elif group_by == "csv":
            group = str(source.resolve())
        elif analysis:
            # Include the analysis root to distinguish identically named datasets.
            normalized = str(source.resolve()).replace("\\", "/")
            root_match = re.search(re.escape(analysis.group(1)), normalized, re.I)
            root = normalized[:root_match.end()] if root_match else analysis.group(1).upper()
            group = root.casefold()
            if group_by == "cell" and cell:
                group += "/cell#" + str(int(cell.group(1)))
        else:
            group = str(source.resolve())
        groups.append(group)
    lookup = pd.MultiIndex.from_frame(identities)
    positions = lookup.get_indexer(pd.MultiIndex.from_frame(data[["source_csv", "filename"]]))
    return np.asarray(groups, dtype=str)[positions]
