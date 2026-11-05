"""Validated, GUI-independent inputs for PDF reports and manual combines."""

from dataclasses import dataclass
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd
import yaml

from modules.report.report_combiner import IMAGE_EXTENSIONS
from modules.utils.analysis_id import compile_cell_pattern


@dataclass(frozen=True)
class ReportSide:
    cell: int
    face: str
    source: Path
    passes: tuple[Path, ...]
    masks: tuple[Path, ...]
    layers: tuple[int, ...]
    combined: tuple[Path, ...]
    mask_combined: tuple[Path, ...]


@dataclass
class ReportDataset:
    analysis_id: str
    path: Path
    csv_path: Path
    frame: pd.DataFrame
    sides: list[ReportSide]

    @property
    def cells(self):
        return sorted({side.cell for side in self.sides})


def analysis_paths(root: Path, pattern=r"^[A-Z]{3}[0-9]{5}$") -> list[Path]:
    """Find IDs at any depth, stopping at each ID and skipping generated folders."""
    root = Path(root)
    if not root.is_dir():
        return []
    valid = re.compile(pattern, re.I)
    if valid.fullmatch(root.name):
        return [root]
    found = []
    for parent, dirs, _ in os.walk(root, followlinks=False):
        for name in list(dirs):
            if valid.fullmatch(name):
                found.append(Path(parent) / name)
                dirs.remove(name)
            elif (
                name.startswith(".")
                or name.lower() in {"models", "reports", "captures"}
                or name.lower().endswith(("_pass", "_mask", "_combined"))
            ):
                dirs.remove(name)
    return sorted(found, key=lambda p: (p.name, str(p)))


def image_paths(folder: Path, png_only=False) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        (
            p
            for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in ({".png"} if png_only else IMAGE_EXTENSIONS)
        ),
        key=lambda p: p.name,
    )


def side_sources(analysis: Path, cell_pattern=r"^cell\s*#\d+$"):
    pattern = compile_cell_pattern(cell_pattern)
    cells = sorted(
        (p for p in analysis.iterdir() if p.is_dir() and pattern.fullmatch(p.name)),
        key=lambda p: int(re.search(r"\d+", p.name)[0]),
    )
    for cell in cells or [analysis]:
        number = int(re.search(r"\d+", cell.name)[0]) if cell != analysis else 1
        folders = {
            p.name.upper(): p
            for p in cell.iterdir()
            if p.is_dir() and p.name.upper() in {"FRONT", "BACK"}
        }
        if folders:
            for face in ("FRONT", "BACK"):
                if face in folders:
                    yield (
                        number,
                        face,
                        folders[face],
                        cell / f"{face}_pass",
                        cell / f"{face}_mask",
                        cell,
                    )
        elif image_paths(cell):
            yield number, "SOURCE", cell, cell / "_pass", cell / "_mask", cell


def find_result_csv(analysis: Path) -> Path:
    aid = analysis.name.lower()
    files = [
        p
        for p in analysis.rglob("*.csv")
        if p.name.lower() == f"{aid}_result.csv"
        or p.name.lower().startswith((f"{aid}_li-salt_", f"{aid}_li_salt_"))
    ]
    if not files:
        raise ValueError("Result CSV is missing.")
    return max(files, key=lambda p: (p.stat().st_mtime_ns, p.name))


def load_dataset(analysis: Path, config_path: Path) -> ReportDataset:
    """Require every source, layer, cell and present face, never a partial report."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8-sig"))
    analysis = Path(analysis)
    csv = find_result_csv(analysis)
    frame = pd.read_csv(csv)
    needed = {"Cell", "Face", "ImageNum", "li_salt_ratio", "nonactive_ratio"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"CSV columns missing: {sorted(needed - set(frame.columns))}")
    if frame.empty:
        raise ValueError("Result CSV is empty.")
    for key in ("Cell", "ImageNum"):
        frame[key] = pd.to_numeric(frame[key], errors="raise")
        if (
            not np.isfinite(frame[key]).all()
            or (frame[key] < 1).any()
            or (frame[key] % 1 != 0).any()
        ):
            raise ValueError(f"Invalid {key} in CSV.")
        frame[key] = frame[key].astype(int)
    frame["Face"] = frame.Face.astype(str).str.upper().str.strip()
    if (
        "AnalysisID" in frame
        and not frame.AnalysisID.astype(str).str.upper().eq(analysis.name.upper()).all()
    ):
        raise ValueError("CSV contains another Analysis ID.")
    if frame.duplicated(["Cell", "Face", "ImageNum"]).any():
        raise ValueError("CSV has duplicate cell / face / layer rows.")
    for key in ("li_salt_ratio", "nonactive_ratio"):
        frame[key] = pd.to_numeric(frame[key], errors="raise")
        if not np.isfinite(frame[key]).all() or not frame[key].between(0, 1).all():
            raise ValueError(f"{key} must contain fractions from 0 to 1.")
    if (frame.li_salt_ratio + frame.nonactive_ratio > 1.000001).any():
        raise ValueError("CSV class ratios exceed 100%.")
    sides = []
    expected_keys = set()
    for cell, face, source, pass_dir, mask_dir, parent in side_sources(
        analysis, cfg["folder_qc"].get("cell_folder_pattern", r"^cell\s*#\d+$")
    ):
        originals = image_paths(source)
        if not originals:
            raise ValueError(f"Cell #{cell} {face}: source images missing.")
        if len({p.stem for p in originals}) != len(originals):
            raise ValueError(
                "Source files share the same stem; layer mapping is ambiguous."
            )
        passes = [pass_dir / (p.stem + ".png") for p in originals]
        if not all(p.is_file() and p.stat().st_size > 0 for p in passes):
            raise ValueError(f"Cell #{cell} {face}: PASS images incomplete.")
        layers = list(range(2, len(originals))) if len(originals) > 1 else [1]
        if not layers:
            raise ValueError(f"Cell #{cell} {face}: no ML layers.")
        masks = [mask_dir / (originals[i - 1].stem + "_mask.png") for i in layers]
        if not all(p.is_file() and p.stat().st_size > 0 for p in masks):
            raise ValueError(f"Cell #{cell} {face}: masks incomplete.")
        for i in layers:
            expected_keys.add((cell, face, i))
        combined = tuple(
            p
            for p in image_paths(parent / f"{face}_combined", True)
            if "_combined_" in p.stem
        )
        mask_combined = tuple(
            p
            for p in image_paths(parent / f"{face}_mask_combined", True)
            if "_combined_" in p.stem
        )
        if not combined or not mask_combined or len(combined) != len(mask_combined):
            raise ValueError(f"Cell #{cell} {face}: combined outputs missing.")
        for sheets in (combined, mask_combined):
            numbers = [
                int(re.search(r"_combined_(\d+)$", p.stem)[1])
                if re.search(r"_combined_(\d+)$", p.stem)
                else -1
                for p in sheets
            ]
            if numbers != list(range(1, len(sheets) + 1)) or any(
                p.stat().st_size == 0 for p in sheets
            ):
                raise ValueError(f"Cell #{cell} {face}: combined pages incomplete.")
        sides.append(
            ReportSide(
                cell,
                face,
                source,
                tuple(passes),
                tuple(masks),
                tuple(layers),
                combined,
                mask_combined,
            )
        )
    if not sides:
        raise ValueError("No cell / face source folders found.")
    actual = set(frame[["Cell", "Face", "ImageNum"]].itertuples(index=False, name=None))
    if actual != expected_keys:
        raise ValueError(
            "CSV layer coverage differs from source / mask coverage. Rerun ML."
        )
    # A persisted QC hold must not appear as ready just because old outputs exist.
    for hold in analysis.glob("*qc*hold*.json"):
        import json

        payload = json.loads(hold.read_text(encoding="utf-8"))
        if payload.get("failures"):
            raise ValueError("Unresolved QC hold images remain.")
    ordered = (
        frame.assign(_face_order=frame.Face.map({"FRONT": 0, "BACK": 1, "SOURCE": 2}))
        .sort_values(["Cell", "_face_order", "ImageNum"])
        .drop(columns="_face_order")
    )
    return ReportDataset(analysis.name, analysis, csv, ordered, sides)


def summary_frame(dataset: ReportDataset) -> pd.DataFrame:
    rows = []
    for (cell, face), group in dataset.frame.groupby(["Cell", "Face"], sort=False):
        for metric, column in [
            ("Li-salt", "li_salt_ratio"),
            ("Nonactive", "nonactive_ratio"),
        ]:
            values = group[column] * 100
            rows.append(
                {
                    "Cell": cell,
                    "Face": face,
                    "Metric": metric,
                    "N": len(values),
                    "Mean": values.mean(),
                    "Median": values.median(),
                    "Min": values.min(),
                    "Max": values.max(),
                }
            )
    return pd.DataFrame(rows)
