import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import yaml

try:
    from PIL import ImageGrab
except ImportError:  # pragma: no cover - depends on deployment package.
    ImageGrab = None

from modules.utils.analysis_id import (
    find_test_source_folders,
    get_test_mask_folder_for_pass,
    get_test_output_folders,
    is_test_analysis_id,
)

logger = logging.getLogger(__name__)

_RESULT_PATTERN_OLD = "{analysis_id}_result.csv"
_RESULT_PATTERN_NEW = "{analysis_id}_LI-SALT_*.csv"
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_C_AXIS = "#555555"
_C_GRID = "#dddddd"
_C_MUTED = "#777777"
_C_NORMAL = "#000000"
_C_BG = "#ffffff"
_C_ISSUE = "#cc0000"
_C_SELECTED = "#dbeafe"

_MASK_COLOR_LI_SALT = np.array([0, 0, 255], dtype=np.uint8)
_MASK_COLOR_NONACTIVE = np.array([255, 0, 0], dtype=np.uint8)

_HEATMAP_MODES = (
    "Li-salt",
    "Nonactive",
    "Li-salt + Nonactive",
)

_SERIES_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#d62728",
    "#bcbd22",
)

_FACE_MODES = (
    "FRONT",
    "BACK",
    "TEST",
    "FRONT + BACK",
    "Average",
)


@dataclass(frozen=True)
class _ResultOption:
    key: str
    date_name: str
    analysis_id: str
    analysis_path: Path
    csv_path: Path
    cell: int
    label: str
    result_summary: str


@dataclass(frozen=True)
class _SeriesPoint:
    option_key: str
    label: str
    face: str
    image_num: int
    percent: float
    is_issue: bool
    row: dict
    pass_image: Path | None
    mask_image: Path | None


@dataclass(frozen=True)
class _ResultRow:
    key: str
    date_name: str
    work_label: str
    display_path: str
    analysis_id: str
    analysis_path: Path
    csv_path: Path
    cell: int
    face: str
    image_num: int
    li_salt_percent: float
    nonactive_percent: float | None
    pass_image: Path | None
    mask_image: Path | None
    row: dict


class AnalysisTab(ttk.Frame):
    """Visual analysis view for ML result CSVs."""

    def __init__(self, parent, config_path: str | Path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._load_config()

        self._date_values: list[str] = []
        self._result_options: list[_ResultOption] = []
        self._option_by_item: dict[str, _ResultOption] = {}
        self._option_rows_by_item: dict[str, list[_ResultRow]] = {}
        self._row_by_item: dict[str, _ResultRow] = {}
        self._selected_options: dict[str, _ResultOption] = {}
        self._all_series_points: list[_SeriesPoint] = []
        self._canvas_points: list[tuple[float, float, _SeriesPoint]] = []
        self._overlap_points: list[_SeriesPoint] = []
        self._shown_point: _SeriesPoint | None = None
        self._active_result_item: str | None = None
        self._heatmap_option_key: str | None = None
        self._heatmap_counts_cache: dict[
            tuple[str, str, tuple[tuple[str, int, int], ...]],
            tuple[np.ndarray, np.ndarray, int],
        ] = {}
        self._photo_refs: list[tk.PhotoImage] = []

        self._build_ui()
        self._load_dates()

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        folder_cfg = cfg.get("folder_qc", {}) or {}
        self._root_path = Path(cfg["app"]["root_path"])
        self._analysis_id_re = re.compile(folder_cfg["analysis_id_pattern"])
        self._date_formats = list(folder_cfg.get("date_formats") or ["%Y-%m-%d"])

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=(10, 6))
        header.columnconfigure(4, weight=1)

        self._root_path_var = tk.StringVar(value=str(self._root_path))

        ttk.Label(header, text="Date").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self._date_var = tk.StringVar()
        self._date_combo = ttk.Combobox(
            header,
            textvariable=self._date_var,
            state="readonly",
            width=18,
        )
        self._date_combo.grid(row=0, column=1, sticky=tk.W)
        self._date_combo.bind("<<ComboboxSelected>>", self._on_date_change)

        ttk.Label(header, text="View").grid(row=0, column=2, sticky=tk.W, padx=(10, 6))
        self._face_mode_var = tk.StringVar(value="FRONT")
        self._face_mode_combo = ttk.Combobox(
            header,
            textvariable=self._face_mode_var,
            values=_FACE_MODES,
            state="readonly",
            width=14,
        )
        self._face_mode_combo.grid(row=0, column=3, sticky=tk.W)
        self._face_mode_combo.bind("<<ComboboxSelected>>", self._on_face_mode_change)

        ttk.Button(header, text="Refresh", command=self._refresh).grid(
            row=0, column=4, sticky=tk.W, padx=(8, 0)
        )
        ttk.Button(header, text="Clear graph", command=self._clear_graph).grid(
            row=0, column=5, sticky=tk.W, padx=(8, 0)
        )
        ttk.Button(header, text="Export data", command=self._export_data).grid(
            row=0, column=6, sticky=tk.W, padx=(8, 0)
        )
        ttk.Button(header, text="Capture screen", command=self._capture_screen).grid(
            row=0, column=7, sticky=tk.W, padx=(8, 0)
        )
        self._status_label = ttk.Label(header, text="", foreground=_C_MUTED)
        self._status_label.grid(row=0, column=8, sticky=tk.W, padx=(12, 0))

        left = ttk.LabelFrame(self, text="ML Results", padding=8)
        left.grid(row=1, column=0, sticky=tk.NS, padx=(10, 6), pady=(0, 8))
        left.rowconfigure(0, weight=1)

        columns = ("face", "li_salt", "result")
        self._result_tree = ttk.Treeview(
            left,
            columns=columns,
            show="tree headings",
            height=18,
            selectmode="none",
        )
        self._result_tree.heading("#0", text="Date / Work Folder / Image")
        self._result_tree.heading("face", text="Face", command=lambda: self._sort_results("face"))
        self._result_tree.heading("li_salt", text="Li-salt %", command=lambda: self._sort_results("li_salt"))
        self._result_tree.heading("result", text="Li-salt / Nonactive")
        self._result_tree.column("#0", width=220, anchor=tk.W)
        self._result_tree.column("face", width=70, anchor=tk.W)
        self._result_tree.column("li_salt", width=75, anchor=tk.E)
        self._result_tree.column("result", width=175, anchor=tk.W)
        self._result_tree.grid(row=0, column=0, sticky=tk.NS)
        self._result_tree.tag_configure("selected", background=_C_SELECTED)
        self._result_tree.bind("<Button-1>", self._on_result_click)

        tree_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._result_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._result_tree.configure(yscrollcommand=tree_scroll.set)

        graph_frame = ttk.LabelFrame(self, text="Analysis view", padding=8)
        graph_frame.grid(row=1, column=1, sticky=tk.NSEW, padx=(6, 10), pady=(0, 8))
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)

        self._analysis_notebook = ttk.Notebook(graph_frame)
        self._analysis_notebook.grid(row=0, column=0, sticky=tk.NSEW)
        self._analysis_notebook.bind("<<NotebookTabChanged>>", self._on_analysis_tab_change)

        graph_tab = ttk.Frame(self._analysis_notebook)
        graph_tab.columnconfigure(0, weight=1)
        graph_tab.rowconfigure(0, weight=1)
        self._analysis_notebook.add(graph_tab, text="Graph")

        self._graph = tk.Canvas(graph_tab, bg=_C_BG, height=360, highlightthickness=0)
        self._graph.grid(row=0, column=0, sticky=tk.NSEW)
        self._graph.bind("<Configure>", lambda _event: self._redraw_graph())
        self._graph.bind("<Button-1>", self._on_graph_click)

        heatmap_tab = ttk.Frame(self._analysis_notebook)
        heatmap_tab.columnconfigure(0, weight=1)
        heatmap_tab.rowconfigure(1, weight=1)
        self._analysis_notebook.add(heatmap_tab, text="Heatmap")

        heatmap_controls = ttk.Frame(heatmap_tab)
        heatmap_controls.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        heatmap_controls.columnconfigure(3, weight=1)
        ttk.Label(heatmap_controls, text="Mask").grid(row=0, column=0, sticky=tk.W)
        self._heatmap_mode_var = tk.StringVar(value="Li-salt")
        self._heatmap_mode_combo = ttk.Combobox(
            heatmap_controls,
            textvariable=self._heatmap_mode_var,
            values=_HEATMAP_MODES,
            state="readonly",
            width=20,
        )
        self._heatmap_mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(6, 12))
        self._heatmap_mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._redraw_heatmap())
        self._heatmap_label = ttk.Label(
            heatmap_controls,
            text="Click a cell result on the left to build a heatmap.",
            foreground=_C_MUTED,
        )
        self._heatmap_label.grid(row=0, column=2, columnspan=2, sticky=tk.W)

        self._heatmap_canvas = tk.Canvas(heatmap_tab, bg=_C_BG, height=360, highlightthickness=0)
        self._heatmap_canvas.grid(row=1, column=0, sticky=tk.NSEW)
        self._heatmap_canvas.bind("<Configure>", lambda _event: self._redraw_heatmap())

        images = ttk.LabelFrame(self, text="Selected layer images", padding=8)
        images.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=(0, 10))
        images.columnconfigure(0, weight=1)
        images.columnconfigure(1, weight=1)

        self._selection_label = ttk.Label(images, text="Click a graph point to inspect images.")
        self._selection_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 6))

        self._overlap_var = tk.StringVar()
        self._overlap_combo = ttk.Combobox(
            images,
            textvariable=self._overlap_var,
            state="readonly",
            width=36,
        )
        self._overlap_combo.grid(row=0, column=1, sticky=tk.E, pady=(0, 6))
        self._overlap_combo.bind("<<ComboboxSelected>>", self._on_overlap_select)
        self._overlap_combo.grid_remove()

        self._crop_canvas = tk.Canvas(images, bg="#f5f5f5", height=230, highlightthickness=1)
        self._crop_canvas.grid(row=1, column=0, sticky=tk.EW, padx=(0, 5))
        self._mask_canvas = tk.Canvas(images, bg="#f5f5f5", height=230, highlightthickness=1)
        self._mask_canvas.grid(row=1, column=1, sticky=tk.EW, padx=(5, 0))

    def reload_settings(self) -> None:
        self._load_config()
        self._load_dates()

    def _load_dates(self) -> None:
        self._date_values = self._list_date_folders()
        self._date_combo["values"] = self._date_values
        selected = self._pick_default_date(self._date_values)
        self._date_var.set(selected)
        self._load_results()

    def _refresh(self) -> None:
        self._load_config()
        self._load_dates()

    def _on_date_change(self, _event=None) -> None:
        self._load_results()

    def _on_face_mode_change(self, _event=None) -> None:
        self._clear_images()
        self._redraw_graph()
        self._redraw_heatmap()

    def _on_analysis_tab_change(self, _event=None) -> None:
        self._redraw_graph()
        self._redraw_heatmap()

    def _is_heatmap_tab_active(self) -> bool:
        try:
            selected = self._analysis_notebook.select()
            return self._analysis_notebook.tab(selected, "text") == "Heatmap"
        except tk.TclError:
            return False

    def _list_date_folders(self) -> list[str]:
        if not self._root_path.exists():
            return []
        dated: list[tuple[datetime, str]] = []
        for path in self._root_path.iterdir():
            if not path.is_dir():
                continue
            parsed = self._parse_date_folder(path.name)
            if parsed is not None:
                dated.append((parsed, path.name))
        dated.sort(key=lambda item: item[0], reverse=True)
        return [name for _, name in dated]

    def _parse_date_folder(self, folder_name: str) -> datetime | None:
        for fmt in self._date_formats:
            try:
                return datetime.strptime(folder_name, fmt)
            except ValueError:
                continue
        return None

    def _pick_default_date(self, date_names: list[str]) -> str:
        if not date_names:
            return ""
        today_value = date.today()
        for name in date_names:
            parsed = self._parse_date_folder(name)
            if parsed is not None and parsed.date() == today_value:
                return name
        return date_names[0]

    def _load_root_path(self, clear_cache: bool = False) -> None:
        del clear_cache
        raw_path = self._root_path_var.get().strip()
        if raw_path:
            self._root_path = Path(raw_path)
        self._date_values = self._list_date_folders()
        self._date_combo["values"] = self._date_values
        if self._date_values and self._date_var.get() not in self._date_values:
            self._date_var.set(self._pick_default_date(self._date_values))
        rows = self._discover_result_rows(self._root_path)
        self._set_result_rows(rows)
        self._status_label.config(text=f"Loaded {len(rows)} result row(s) from {self._root_path}")

    def _load_results(self) -> None:
        if hasattr(self, "_root_path_var"):
            self._load_root_path()
            return
        for item in self._result_tree.get_children():
            self._result_tree.delete(item)
        self._option_by_item.clear()
        self._result_options.clear()
        self._selected_options.clear()
        self._all_series_points.clear()
        self._overlap_points.clear()
        self._shown_point = None
        self._heatmap_option_key = None
        self._heatmap_counts_cache.clear()
        self._clear_images()

        date_name = self._date_var.get().strip()
        date_path = self._root_path / date_name if date_name else None
        if date_path is None or not date_path.is_dir():
            self._status_label.config(text="No date selected")
            self._redraw_graph()
            return

        options = self._discover_result_options(date_name, date_path)
        self._result_options = options
        for option in options:
            item_id = self._result_tree.insert(
                "",
                tk.END,
                values=(option.analysis_id, f"#{option.cell}", option.result_summary),
            )
            self._option_by_item[item_id] = option

        self._status_label.config(text=f"{len(options)} cell result(s)")
        self._redraw_graph()
        self._redraw_heatmap()

    def _discover_result_rows(self, root_path: Path) -> list[_ResultRow]:
        rows: list[_ResultRow] = []
        seen_mask_keys: set[tuple[Path, str]] = set()
        if not root_path.exists():
            return rows
        for csv_path in sorted(root_path.rglob("*.csv"), key=lambda path: str(path).lower()):
            if not self._is_generated_li_salt_csv(csv_path) and not csv_path.name.endswith("_result.csv"):
                continue
            rows.extend(self._rows_from_csv(root_path, csv_path))
            for row in rows:
                if row.mask_image is not None:
                    seen_mask_keys.add((row.mask_image.parent, row.mask_image.stem))
        for mask_dir in self._iter_mask_folders(root_path):
            for mask_image in sorted(mask_dir.glob("*_mask.png"), key=lambda path: path.name.lower()):
                if (mask_image.parent, mask_image.stem) in seen_mask_keys:
                    continue
                derived = self._row_from_mask(root_path, mask_image)
                if derived is not None:
                    rows.append(derived)
        return rows

    def _set_result_rows(self, rows: list[_ResultRow]) -> None:
        for item in self._result_tree.get_children():
            self._result_tree.delete(item)
        self._option_by_item.clear()
        self._option_rows_by_item.clear()
        self._row_by_item.clear()
        self._result_options.clear()
        self._selected_options.clear()
        self._all_series_points.clear()
        self._overlap_points.clear()
        self._shown_point = None
        self._active_result_item = None
        self._heatmap_option_key = None
        self._heatmap_counts_cache.clear()
        self._clear_images()

        grouped: dict[str, dict[str, list[_ResultRow]]] = {}
        row_order = {row.key: index for index, row in enumerate(rows)}
        for row in rows:
            grouped.setdefault(row.date_name, {}).setdefault(row.work_label, []).append(row)

        for date_name in sorted(grouped.keys(), key=lambda name: min(row_order[row.key] for group in grouped[name].values() for row in group)):
            date_id = self._result_tree.insert("", tk.END, text=date_name, open=True, values=("", "", ""))
            for work_label in sorted(grouped[date_name].keys(), key=lambda label: min(row_order[row.key] for row in grouped[date_name][label])):
                work_rows = sorted(grouped[date_name][work_label], key=lambda row: row_order[row.key])
                option = self._option_from_rows(work_rows)
                option_id = self._result_tree.insert(date_id, tk.END, text=work_label, open=True, values=("", "", option.result_summary))
                self._option_by_item[option_id] = option
                self._option_rows_by_item[option_id] = work_rows
                self._result_options.append(option)
                for row in work_rows:
                    item_id = self._result_tree.insert(
                        option_id,
                        tk.END,
                        text=row.display_path,
                        values=(
                            row.face,
                            self._format_li_salt_percent(row.li_salt_percent),
                            self._row_result_summary(row),
                        ),
                    )
                    self._row_by_item[item_id] = row
        self._redraw_graph()
        self._redraw_heatmap()

    def _option_from_rows(self, rows: list[_ResultRow]) -> _ResultOption:
        first = rows[0]
        li_avg = sum(row.li_salt_percent for row in rows) / len(rows)
        nonactive_values = [row.nonactive_percent for row in rows if row.nonactive_percent is not None]
        if nonactive_values:
            summary = f"Li {li_avg:.2f}% / Nonactive {sum(nonactive_values) / len(nonactive_values):.2f}%"
        else:
            summary = f"Li {li_avg:.2f}% / Nonactive -"
        return _ResultOption(
            key=f"{first.date_name}|{first.work_label}",
            date_name=first.date_name,
            analysis_id=first.analysis_id,
            analysis_path=first.analysis_path,
            csv_path=first.csv_path,
            cell=first.cell,
            label=first.work_label,
            result_summary=summary,
        )

    def _discover_result_options(self, date_name: str, date_path: Path) -> list[_ResultOption]:
        options: list[_ResultOption] = []
        for analysis_path in sorted(
            [
                p for p in date_path.iterdir()
                if p.is_dir() and self._is_analysis_folder(p.name)
            ],
            key=lambda p: p.name,
        ):
            csv_path = self._find_result_csv(analysis_path, analysis_path.name)
            if csv_path is None:
                continue
            try:
                frame = pd.read_csv(csv_path)
            except Exception:
                logger.exception("Failed to read result CSV: %s", csv_path)
                continue
            if "Cell" not in frame.columns or "li_salt_ratio" not in frame.columns:
                continue
            numeric_cells = pd.to_numeric(frame["Cell"], errors="coerce")
            cells = sorted({int(value) for value in numeric_cells.dropna().unique()})
            for cell in cells:
                key = f"{date_name}|{analysis_path.name}|{cell}|{csv_path.name}"
                result_summary = self._cell_result_summary(frame, cell)
                options.append(
                    _ResultOption(
                        key=key,
                        date_name=date_name,
                        analysis_id=analysis_path.name,
                        analysis_path=analysis_path,
                        csv_path=csv_path,
                        cell=cell,
                        label=f"{analysis_path.name} Cell #{cell}",
                        result_summary=result_summary,
                    )
                )
        return options

    def _rows_from_csv(self, root_path: Path, csv_path: Path) -> list[_ResultRow]:
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Failed to read result CSV: %s", csv_path)
            return []
        if frame.empty:
            frame = pd.DataFrame([{}])
        rows: list[_ResultRow] = []
        for index, record in frame.iterrows():
            row_dict = record.to_dict()
            li_percent = self._li_salt_percent_from_row(row_dict, csv_path)
            if li_percent is None:
                continue
            nonactive_percent = self._nonactive_percent_from_row(row_dict)
            face = self._clean_text(row_dict.get("Face")) or self._face_from_path(csv_path)
            image_num = self._int_value(row_dict.get("ImageNum"), index + 1)
            cell = self._int_value(row_dict.get("Cell"), 1)
            mask_image = self._path_from_value(row_dict.get("mask_image"))
            crop_image = self._path_from_value(row_dict.get("crop_image"))
            source_image = self._path_from_value(row_dict.get("source_image"))
            analysis_path = self._analysis_path_for_csv(csv_path)
            if mask_image is None:
                mask_image = self._mask_image_for_csv(csv_path, crop_image, source_image, image_num)
            if crop_image is None:
                crop_image = self._pass_image_for_mask(mask_image)
            date_name = self._date_label_for_path(root_path, analysis_path)
            work_label = self._work_folder_label(root_path, analysis_path, date_name)
            analysis_id = self._clean_text(row_dict.get("AnalysisID")) or self._analysis_id_for_path(analysis_path)
            display_path = self._display_path(root_path, crop_image or source_image or mask_image or csv_path)
            rows.append(
                _ResultRow(
                    key=f"{csv_path}|{index}",
                    date_name=date_name,
                    work_label=work_label,
                    display_path=display_path,
                    analysis_id=analysis_id,
                    analysis_path=analysis_path,
                    csv_path=csv_path,
                    cell=cell,
                    face=face,
                    image_num=image_num,
                    li_salt_percent=li_percent,
                    nonactive_percent=nonactive_percent,
                    pass_image=crop_image,
                    mask_image=mask_image,
                    row=row_dict,
                )
            )
        return rows

    def _row_from_mask(self, root_path: Path, mask_image: Path) -> _ResultRow | None:
        li_percent, nonactive_percent = self._percentages_from_mask_image(mask_image)
        pass_image = self._pass_image_for_mask(mask_image)
        analysis_path = self._analysis_path_for_mask_folder(mask_image.parent)
        date_name = self._date_label_for_path(root_path, analysis_path)
        work_label = self._work_folder_label(root_path, analysis_path, date_name)
        image_num = self._image_num_from_name(mask_image.name)
        face = self._face_from_path(mask_image)
        analysis_id = self._analysis_id_for_path(analysis_path)
        row = {
            "AnalysisID": analysis_id,
            "Cell": 1,
            "Face": face,
            "ImageNum": image_num,
            "nonactive_ratio": nonactive_percent / 100.0,
            "li_salt_ratio": li_percent / 100.0,
        }
        return _ResultRow(
            key=str(mask_image),
            date_name=date_name,
            work_label=work_label,
            display_path=self._display_path(root_path, pass_image or mask_image),
            analysis_id=analysis_id,
            analysis_path=analysis_path,
            csv_path=mask_image.parent / "_derived_from_masks.csv",
            cell=1,
            face=face,
            image_num=image_num,
            li_salt_percent=li_percent,
            nonactive_percent=nonactive_percent,
            pass_image=pass_image,
            mask_image=mask_image,
            row=row,
        )

    def _iter_mask_folders(self, root_path: Path) -> list[Path]:
        folders: list[Path] = []
        for path in root_path.rglob("*"):
            if path.is_dir() and self._is_mask_folder(path):
                folders.append(path)
        return sorted(folders, key=lambda path: str(path).lower())

    def _analysis_path_for_csv(self, csv_path: Path) -> Path:
        if self._is_mask_folder(csv_path.parent):
            return self._analysis_path_for_mask_folder(csv_path.parent)
        return csv_path.parent

    def _is_mask_folder(self, path: Path) -> bool:
        name = path.name.upper()
        return name in {"_MASK", "_MASKS"} or name.endswith("_MASK") or name.endswith("_MASKS")

    def _analysis_path_for_mask_folder(self, mask_folder: Path) -> Path:
        name = mask_folder.name
        upper = name.upper()
        if upper in {"_MASK", "_MASKS"}:
            return mask_folder.parent
        for suffix in ("_MASKS", "_masks", "_MASK", "_mask"):
            if name.endswith(suffix):
                source = mask_folder.with_name(name[: -len(suffix)])
                if source.exists():
                    return source
        return mask_folder.parent

    def _date_label_for_path(self, root_path: Path, path: Path) -> str:
        for candidate in [path, *path.parents]:
            if candidate == root_path.parent:
                break
            if self._parse_date_folder(candidate.name) is not None:
                return candidate.name
        return "No date"

    def _work_folder_label(self, root_path: Path, analysis_path: Path, date_name: str) -> str:
        base = root_path / date_name if date_name != "No date" else root_path
        try:
            rel = analysis_path.relative_to(base)
            return rel.as_posix() or analysis_path.name
        except ValueError:
            return analysis_path.name

    def _analysis_id_for_path(self, analysis_path: Path) -> str:
        for candidate in [analysis_path, *analysis_path.parents]:
            if self._is_analysis_folder(candidate.name):
                return candidate.name
        return re.sub(r"[^A-Za-z0-9_]+", "_", analysis_path.name).strip("_") or "WORK"

    def _display_path(self, root_path: Path, path: Path) -> str:
        try:
            return str(path.relative_to(root_path))
        except ValueError:
            return str(path)

    def _li_salt_percent_from_row(self, row: dict, csv_path: Path) -> float | None:
        if "li_salt_ratio" in row and not pd.isna(row["li_salt_ratio"]):
            try:
                return self._ratio_to_percent(float(row["li_salt_ratio"]))
            except (TypeError, ValueError):
                pass
        match = re.search(r"LI[-_]SALT_([0-9.]+)%", csv_path.name, re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _nonactive_percent_from_row(self, row: dict) -> float | None:
        if "nonactive_ratio" not in row or pd.isna(row["nonactive_ratio"]):
            return None
        try:
            return self._ratio_to_percent(float(row["nonactive_ratio"]))
        except (TypeError, ValueError):
            return None

    def _path_from_value(self, value) -> Path | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return Path(text) if text else None

    def _int_value(self, value, default: int) -> int:
        try:
            if value is None or pd.isna(value):
                return default
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def _face_from_path(self, path: Path) -> str:
        upper_parts = [part.upper() for part in path.parts]
        for face in ("FRONT", "BACK", "TEST"):
            if face in upper_parts or any(part.endswith(f"_{face}") or part.startswith(f"{face}_") for part in upper_parts):
                return face
        return "WORK"

    def _image_num_from_name(self, filename: str) -> int:
        match = re.search(r"(\d+)", filename)
        return int(match.group(1)) if match else 1

    def _mask_image_for_csv(self, csv_path: Path, crop_image: Path | None, source_image: Path | None, image_num: int) -> Path | None:
        if crop_image is not None:
            candidate = self._find_mask_image(crop_image, self._face_from_path(crop_image))
            if candidate is not None:
                return candidate
        stem = (crop_image or source_image).stem if (crop_image or source_image) is not None else csv_path.stem.split("_LI")[0]
        candidates = sorted(csv_path.parent.glob(f"{stem}*_mask.png"), key=lambda path: path.name.lower())
        if candidates:
            return candidates[0]
        masks = sorted(csv_path.parent.glob("*_mask.png"), key=lambda path: path.name.lower())
        index = max(0, image_num - 1)
        return masks[index] if index < len(masks) else None

    def _pass_image_for_mask(self, mask_image: Path | None) -> Path | None:
        if mask_image is None:
            return None
        mask_folder = mask_image.parent
        stem = mask_image.name[:-9] if mask_image.name.endswith("_mask.png") else mask_image.stem
        pass_dirs = []
        name = mask_folder.name
        upper = name.upper()
        if upper in {"_MASK", "_MASKS"}:
            pass_dirs.append(mask_folder.with_name("_PASS"))
        for suffix in ("_MASKS", "_masks", "_MASK", "_mask"):
            if name.endswith(suffix):
                pass_dirs.append(mask_folder.with_name(f"{name[: -len(suffix)]}_PASS"))
                pass_dirs.append(mask_folder.with_name(f"{name[: -len(suffix)]}_pass"))
        for pass_dir in pass_dirs:
            for ext in _IMAGE_EXTENSIONS:
                candidate = pass_dir / f"{stem}{ext}"
                if candidate.exists():
                    return candidate
        return None

    def _percentages_from_mask_image(self, mask_image: Path) -> tuple[float, float]:
        image = cv2.imread(str(mask_image), cv2.IMREAD_UNCHANGED)
        if image is None:
            return 0.0, 0.0
        channels = image[:, :, :3]
        valid = image[:, :, 3] > 0 if image.ndim == 3 and image.shape[2] >= 4 else np.ones(image.shape[:2], dtype=bool)
        total = int(valid.sum())
        if total <= 0:
            return 0.0, 0.0
        li_mask = np.all(channels == _MASK_COLOR_LI_SALT, axis=2) & valid
        nonactive_mask = np.all(channels == _MASK_COLOR_NONACTIVE, axis=2) & valid
        return float(li_mask.sum()) / total * 100.0, float(nonactive_mask.sum()) / total * 100.0

    @staticmethod
    def _is_generated_li_salt_csv(path: Path) -> bool:
        name = path.name.upper()
        return path.suffix.lower() == ".csv" and ("LI-SALT" in name or "LI_SALT" in name)

    @staticmethod
    def _format_li_salt_percent(value: float) -> str:
        if value == 0:
            return "0.00"
        if abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.2f}"

    def _row_result_summary(self, row: _ResultRow) -> str:
        if row.nonactive_percent is None:
            return f"Li {row.li_salt_percent:.2f}% / Nonactive -"
        return f"Li {row.li_salt_percent:.2f}% / Nonactive {row.nonactive_percent:.2f}%"

    def _is_analysis_folder(self, folder_name: str) -> bool:
        return bool(self._analysis_id_re.match(folder_name)) or is_test_analysis_id(folder_name)

    def _cell_result_summary(self, frame: pd.DataFrame, cell: int) -> str:
        required = {"Cell", "Face", "ImageNum", "li_salt_ratio", "nonactive_ratio"}
        if not required.issubset(frame.columns):
            return "Li - / Nonactive -"

        numeric_cells = pd.to_numeric(frame["Cell"], errors="coerce")
        cell_frame = frame[numeric_cells == cell].copy()
        if cell_frame.empty:
            return "Li - / Nonactive -"

        included_rows: list[pd.Series] = []
        for _face, face_frame in cell_frame.groupby("Face"):
            face_frame = face_frame.copy()
            face_frame["ImageNum"] = pd.to_numeric(face_frame["ImageNum"], errors="coerce")
            face_frame = face_frame.dropna(subset=["ImageNum"])
            ordered = face_frame.sort_values("ImageNum")
            if str(_face).upper() == "TEST":
                included_rows.extend(row for _, row in ordered.iterrows())
                continue
            if len(ordered) > 2:
                included_rows.extend(row for _, row in ordered.iloc[1:-1].iterrows())

        if not included_rows:
            included_rows = [row for _, row in cell_frame.iterrows()]

        li_values = [
            self._ratio_to_percent(float(row["li_salt_ratio"]))
            for row in included_rows
            if not pd.isna(row["li_salt_ratio"])
        ]
        nonactive_values = [
            self._ratio_to_percent(float(row["nonactive_ratio"]))
            for row in included_rows
            if not pd.isna(row["nonactive_ratio"])
        ]
        if not li_values or not nonactive_values:
            return "Li - / Nonactive -"

        li_avg = sum(li_values) / len(li_values)
        nonactive_avg = sum(nonactive_values) / len(nonactive_values)
        return f"Li {li_avg:.2f}% / Nonactive {nonactive_avg:.2f}%"

    def _find_result_csv(self, analysis_path: Path, analysis_id: str) -> Path | None:
        candidates = []
        old_path = analysis_path / _RESULT_PATTERN_OLD.format(analysis_id=analysis_id)
        if old_path.exists():
            candidates.append(old_path)
        candidates.extend(
            analysis_path.glob(_RESULT_PATTERN_NEW.format(analysis_id=analysis_id))
        )
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _on_result_click(self, event) -> str:
        item_id = self._result_tree.identify_row(event.y)
        if not item_id:
            return "break"
        if item_id in self._row_by_item:
            self._select_result_item(item_id)
            return "break"
        if item_id in self._option_rows_by_item:
            self._select_result_option(item_id)
            try:
                self._result_tree.item(item_id, open=not bool(self._result_tree.item(item_id, "open")))
            except tk.TclError:
                pass
            return "break"
        option = self._option_by_item.get(item_id)
        if option is None:
            return "break"

        if option.key in self._selected_options:
            self._remove_option_series(option.key)
        else:
            self._selected_options[option.key] = option
            self._heatmap_option_key = option.key
            self._add_option_series(option)

        self._sync_result_tree_styles()
        self._redraw_graph()
        self._redraw_heatmap()
        return "break"

    def _sync_result_tree_styles(self) -> None:
        for item_id, option in self._option_by_item.items():
            tags = ("selected",) if option.key in self._selected_options else ()
            self._result_tree.item(item_id, tags=tags)
        for item_id in self._row_by_item:
            tags = ("selected",) if item_id == self._active_result_item else ()
            self._result_tree.item(item_id, tags=tags)

    def _sort_results(self, column: str) -> None:
        rows = list(self._row_by_item.values())
        if column == "li_salt":
            rows.sort(key=lambda row: row.li_salt_percent)
        elif column == "face":
            rows.sort(key=lambda row: (row.face, row.date_name, row.work_label, row.image_num))
        else:
            rows.sort(key=lambda row: (row.date_name, row.work_label, row.display_path))
        self._set_result_rows(rows)

    def _select_result_item(self, item_id: str) -> None:
        row = self._row_by_item[item_id]
        self._active_result_item = item_id
        self._selected_options.clear()
        self._all_series_points = [self._series_point_from_row(row)]
        self._shown_point = self._all_series_points[0]
        self._heatmap_option_key = row.key
        self._sync_result_tree_styles()
        self._redraw_graph()
        self._redraw_heatmap()

    def _select_result_option(self, item_id: str) -> None:
        rows = self._option_rows_by_item[item_id]
        self._active_result_item = item_id
        self._selected_options.clear()
        self._all_series_points = [self._series_point_from_row(row) for row in rows]
        self._shown_point = self._all_series_points[0] if self._all_series_points else None
        self._heatmap_option_key = rows[0].key if rows else None
        self._sync_result_tree_styles()
        self._redraw_graph()
        self._redraw_heatmap()

    def _series_point_from_row(self, row: _ResultRow) -> _SeriesPoint:
        return _SeriesPoint(
            option_key=row.key,
            label=row.work_label,
            face=row.face,
            image_num=row.image_num,
            percent=row.li_salt_percent,
            is_issue=self._coerce_bool(row.row.get("isIssue", False)),
            row=dict(row.row),
            pass_image=row.pass_image,
            mask_image=row.mask_image if row.mask_image and row.mask_image.exists() else None,
        )

    def _remove_option_series(self, option_key: str) -> None:
        self._selected_options.pop(option_key, None)
        self._all_series_points = [
            point for point in self._all_series_points if point.option_key != option_key
        ]
        self._overlap_points = [
            point for point in self._overlap_points if point.option_key != option_key
        ]
        if self._shown_point is not None and self._shown_point.option_key == option_key:
            self._clear_images()
        if self._heatmap_option_key == option_key:
            self._heatmap_option_key = next(iter(self._selected_options), None)

    def _add_option_series(self, option: _ResultOption) -> None:
        try:
            frame = pd.read_csv(option.csv_path)
        except Exception as exc:
            self._status_label.config(text=f"Failed to read {option.csv_path.name}: {exc}")
            return

        required = {"Cell", "Face", "ImageNum", "li_salt_ratio"}
        if not required.issubset(frame.columns):
            self._status_label.config(text=f"Missing columns in {option.csv_path.name}")
            return

        cell_frame = frame[frame["Cell"].astype(int) == option.cell].copy()
        cell_frame["ImageNum"] = cell_frame["ImageNum"].astype(int)
        cell_frame = cell_frame.sort_values(["Face", "ImageNum"])

        for _, row in cell_frame.iterrows():
            face = str(row["Face"])
            image_num = int(row["ImageNum"])
            percent = self._ratio_to_percent(float(row["li_salt_ratio"]))
            pass_image = self._find_pass_image(option.analysis_path, option.cell, face, image_num)
            mask_image = self._find_mask_image(pass_image, face)
            row_dict = row.to_dict()
            self._all_series_points.append(
                _SeriesPoint(
                    option_key=option.key,
                    label=option.label,
                    face=face,
                    image_num=image_num,
                    percent=percent,
                    is_issue=self._coerce_bool(row_dict.get("isIssue", False)),
                    row=row_dict,
                    pass_image=pass_image,
                    mask_image=mask_image if mask_image and mask_image.exists() else None,
                )
            )

    def _coerce_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or pd.isna(value):
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    def _clean_text(self, value) -> str:
        if value is None or pd.isna(value):
            return ""
        return str(value)

    def _ratio_to_percent(self, value: float) -> float:
        if value <= 1.0:
            return value * 100.0
        return value

    def _find_pass_image(
        self,
        analysis_path: Path,
        cell: int,
        face: str,
        image_num: int,
    ) -> Path | None:
        if face.upper() == "TEST":
            source_folders = dict(find_test_source_folders(analysis_path))
            source_folder = source_folders.get(cell)
            if source_folder is None:
                return None
            pass_path, _ = get_test_output_folders(analysis_path, source_folder)
        else:
            cell_path = analysis_path / f"Cell #{cell}"
            if not cell_path.is_dir():
                cell_path = analysis_path
            pass_path = cell_path / f"{face}_pass"
        if not pass_path.is_dir():
            return None
        images = sorted(
            [
                p for p in pass_path.iterdir()
                if p.is_file()
                and p.suffix.lower() in _IMAGE_EXTENSIONS
                and not p.name.endswith("_mask.png")
            ],
            key=lambda p: p.name,
        )
        index = image_num - 1
        if index < 0 or index >= len(images):
            return None
        return images[index]

    def _find_mask_image(self, pass_image: Path | None, face: str) -> Path | None:
        if pass_image is None:
            return None

        mask_name = f"{pass_image.stem}_mask.png"
        if face.upper() == "TEST":
            mask_path = get_test_mask_folder_for_pass(pass_image.parent) / mask_name
            if mask_path.exists():
                return mask_path
        mask_path = pass_image.parent.with_name(f"{face}_mask") / mask_name
        if mask_path.exists():
            return mask_path

        legacy_path = pass_image.with_name(mask_name)
        if legacy_path.exists():
            return legacy_path
        return None

    def _visible_points(self) -> list[_SeriesPoint]:
        mode = self._face_mode_var.get()
        points = list(self._all_series_points)
        if mode == "FRONT":
            return [point for point in points if point.face.upper() == "FRONT"]
        if mode == "BACK":
            return [point for point in points if point.face.upper() == "BACK"]
        if mode == "TEST":
            return [point for point in points if point.face.upper() == "TEST"]
        if mode == "Average":
            return self._average_points(points)
        return points

    def _average_points(self, points: list[_SeriesPoint]) -> list[_SeriesPoint]:
        grouped: dict[tuple[str, int], list[_SeriesPoint]] = {}
        for point in points:
            grouped.setdefault((point.option_key, point.image_num), []).append(point)

        averaged: list[_SeriesPoint] = []
        for (option_key, image_num), group_points in grouped.items():
            if not group_points:
                continue
            front_point = next(
                (point for point in group_points if point.face.upper() == "FRONT"),
                None,
            )
            representative = front_point or group_points[0]
            percent = float(sum(point.percent for point in group_points) / len(group_points))
            is_issue = any(point.is_issue for point in group_points)
            row = dict(representative.row)
            row["li_salt_ratio"] = percent / 100.0
            row["isIssue"] = is_issue
            if is_issue and not self._clean_text(row.get("issueCategory", "")):
                issue_point = next((point for point in group_points if point.is_issue), None)
                if issue_point is not None:
                    row["issueCategory"] = issue_point.row.get("issueCategory", "")
                    row["issueNote"] = issue_point.row.get("issueNote", "")
            averaged.append(
                _SeriesPoint(
                    option_key=option_key,
                    label=representative.label,
                    face="AVG",
                    image_num=image_num,
                    percent=percent,
                    is_issue=is_issue,
                    row=row,
                    pass_image=representative.pass_image,
                    mask_image=representative.mask_image,
                )
            )

        return sorted(averaged, key=lambda point: (point.option_key, point.image_num))

    def _clear_graph(self) -> None:
        self._selected_options.clear()
        self._all_series_points.clear()
        self._overlap_points.clear()
        self._shown_point = None
        self._heatmap_option_key = None
        self._clear_images()
        self._sync_result_tree_styles()
        self._redraw_graph()
        self._redraw_heatmap()

    def _export_data(self) -> None:
        options = list(self._selected_options.values()) or list(self._result_options)
        if not options:
            messagebox.showinfo("Analysis Export", "No analysis result data to export.")
            return

        default_dir = self._default_export_dir()
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export analysis data",
            initialdir=str(default_dir),
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=(
                ("CSV", "*.csv"),
                ("Excel workbook", "*.xlsx"),
            ),
        )
        if not file_path:
            return

        try:
            frame = self._export_frame(options)
            target = Path(file_path)
            if target.suffix.lower() == ".xlsx":
                frame.to_excel(target, index=False)
            else:
                frame.to_csv(target, index=False, float_format="%.4f")
        except ImportError as exc:
            messagebox.showerror(
                "Analysis Export",
                f"Excel export requires the openpyxl package:\n{exc}",
            )
            return
        except Exception as exc:
            logger.exception("Analysis export failed")
            messagebox.showerror("Analysis Export", f"Failed to export data:\n{exc}")
            return

        self._status_label.config(text=f"Exported {len(frame)} row(s)")

    def _export_frame(self, options: list[_ResultOption]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for option in options:
            frame = pd.read_csv(option.csv_path)
            if "Cell" in frame.columns:
                numeric_cells = pd.to_numeric(frame["Cell"], errors="coerce")
                frame = frame[numeric_cells == option.cell].copy()
            if frame.empty:
                continue
            frame.insert(0, "date_name", option.date_name)
            frame.insert(1, "analysis_id_folder", option.analysis_id)
            frame.insert(2, "result_csv", str(option.csv_path))
            frames.append(frame)
        if not frames:
            raise RuntimeError("No rows matched the selected result(s).")
        return pd.concat(frames, ignore_index=True)

    def _capture_screen(self) -> None:
        if ImageGrab is None:
            messagebox.showerror(
                "Analysis Capture",
                "Screen capture requires the Pillow package.",
            )
            return

        default_dir = self._default_export_dir()
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = f"analysis_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = filedialog.asksaveasfilename(
            parent=self,
            title="Capture analysis screen",
            initialdir=str(default_dir),
            initialfile=default_name,
            defaultextension=".png",
            filetypes=(("PNG image", "*.png"),),
        )
        if not file_path:
            return

        try:
            self.update_idletasks()
            x0 = self.winfo_rootx()
            y0 = self.winfo_rooty()
            x1 = x0 + self.winfo_width()
            y1 = y0 + self.winfo_height()
            image = ImageGrab.grab(bbox=(x0, y0, x1, y1))
            image.save(file_path)
        except Exception as exc:
            logger.exception("Analysis screen capture failed")
            messagebox.showerror("Analysis Capture", f"Failed to capture screen:\n{exc}")
            return

        self._status_label.config(text=f"Captured {Path(file_path).name}")

    def _default_export_dir(self) -> Path:
        return self._root_path / ".qcml" / "exports"

    def _redraw_graph(self) -> None:
        self._graph.delete("all")
        self._canvas_points.clear()

        width = max(self._graph.winfo_width(), 200)
        height = max(self._graph.winfo_height(), 200)
        pad_l, pad_r, pad_t, pad_b = 56, 24, 24, 44
        plot_w = max(1, width - pad_l - pad_r)
        plot_h = max(1, height - pad_t - pad_b)

        self._graph.create_rectangle(0, 0, width, height, fill=_C_BG, outline="")
        self._graph.create_line(pad_l, pad_t, pad_l, pad_t + plot_h, fill=_C_AXIS)
        self._graph.create_line(pad_l, pad_t + plot_h, pad_l + plot_w, pad_t + plot_h, fill=_C_AXIS)
        self._graph.create_text(12, pad_t + 4, text="Li-salt %", anchor=tk.W, fill=_C_AXIS)
        self._graph.create_text(
            pad_l + plot_w,
            height - 10,
            text="Layer",
            anchor=tk.E,
            fill=_C_AXIS,
        )

        visible_points = self._visible_points()
        if not visible_points:
            self._graph.create_text(
                width / 2,
                height / 2,
                text="Click a cell result on the left, or change the face view mode.",
                fill=_C_MUTED,
            )
            return

        max_layer = max(point.image_num for point in visible_points)
        max_percent = max(point.percent for point in visible_points)
        y_max = max(5.0, min(100.0, float(np.ceil(max_percent / 5.0) * 5.0)))
        if max_percent > y_max:
            y_max = float(np.ceil(max_percent / 10.0) * 10.0)

        def x_for(layer: int) -> float:
            if max_layer <= 1:
                return pad_l + plot_w / 2
            return pad_l + ((layer - 1) / (max_layer - 1)) * plot_w

        def y_for(percent: float) -> float:
            return pad_t + plot_h - (percent / y_max) * plot_h

        for tick in range(0, int(y_max) + 1, max(1, int(y_max // 5))):
            y = y_for(float(tick))
            self._graph.create_line(pad_l, y, pad_l + plot_w, y, fill=_C_GRID)
            self._graph.create_text(pad_l - 8, y, text=str(tick), anchor=tk.E, fill=_C_AXIS)

        x_label_step = max(1, int(np.ceil(max_layer / 12)))
        for layer in range(1, max_layer + 1, x_label_step):
            x = x_for(layer)
            self._graph.create_line(x, pad_t + plot_h, x, pad_t + plot_h + 4, fill=_C_AXIS)
            self._graph.create_text(x, pad_t + plot_h + 16, text=str(layer), fill=_C_AXIS)

        grouped: dict[tuple[str, str], list[_SeriesPoint]] = {}
        for point in visible_points:
            grouped.setdefault((point.option_key, point.face), []).append(point)

        option_order = list(self._selected_options.keys())
        option_color = {
            key: _SERIES_COLORS[index % len(_SERIES_COLORS)]
            for index, key in enumerate(option_order)
        }
        face_order = {"FRONT": 0, "BACK": 1, "TEST": 2, "AVG": 3}

        legend_y = pad_t + 2
        for option_key, face in sorted(
            grouped,
            key=lambda item: (
                option_order.index(item[0]) if item[0] in option_order else 999,
                face_order.get(item[1].upper(), 99),
            ),
        ):
            points = grouped[(option_key, face)]
            points = sorted(points, key=lambda point: point.image_num)
            color = option_color.get(option_key, _SERIES_COLORS[0])
            dash = (4, 3) if face.upper() in {"BACK", "AVG"} else None
            xy: list[float] = []
            for point in points:
                x = x_for(point.image_num)
                y = y_for(point.percent)
                xy.extend([x, y])
            if len(xy) >= 4:
                if dash:
                    self._graph.create_line(*xy, fill=color, width=2, dash=dash)
                else:
                    self._graph.create_line(*xy, fill=color, width=2)
            for point in points:
                x = x_for(point.image_num)
                y = y_for(point.percent)
                radius = 5 if point.is_issue else 4
                outline = _C_ISSUE if point.is_issue else color
                self._graph.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    fill=color,
                    outline=outline,
                    width=2 if point.is_issue else 1,
                )
                self._canvas_points.append((x, y, point))

            label = self._selected_options[option_key].label if option_key in self._selected_options else points[0].label
            legend_text = f"{label} {face}"
            if dash:
                self._graph.create_line(
                    width - 240,
                    legend_y,
                    width - 220,
                    legend_y,
                    fill=color,
                    width=2,
                    dash=dash,
                )
            else:
                self._graph.create_line(
                    width - 240,
                    legend_y,
                    width - 220,
                    legend_y,
                    fill=color,
                    width=2,
                )
            self._graph.create_text(
                width - 214,
                legend_y,
                text=legend_text,
                anchor=tk.W,
                fill=_C_NORMAL,
            )
            legend_y += 18

    def _on_graph_click(self, event) -> None:
        if not self._canvas_points:
            return
        nearest = min(
            self._canvas_points,
            key=lambda item: (item[0] - event.x) ** 2 + (item[1] - event.y) ** 2,
        )
        nearest_dist_sq = (nearest[0] - event.x) ** 2 + (nearest[1] - event.y) ** 2
        if nearest_dist_sq > 20 ** 2:
            return

        overlap_dist_sq = 10 ** 2
        nearby = [
            item[2]
            for item in self._canvas_points
            if (item[0] - nearest[0]) ** 2 + (item[1] - nearest[1]) ** 2 <= overlap_dist_sq
        ]
        self._set_overlap_points(nearby, nearest[2])

    def _set_overlap_points(self, points: list[_SeriesPoint], selected: _SeriesPoint) -> None:
        self._overlap_points = sorted(
            points,
            key=lambda point: (point.label, point.face, point.image_num, point.percent),
        )
        if len(self._overlap_points) > 1:
            values = [self._point_choice_label(point) for point in self._overlap_points]
            self._overlap_combo["values"] = values
            try:
                selected_index = self._overlap_points.index(selected)
            except ValueError:
                selected_index = 0
            self._overlap_var.set(values[selected_index])
            self._overlap_combo.grid()
        else:
            self._overlap_combo.grid_remove()
            self._overlap_combo["values"] = ()
            self._overlap_var.set("")
        self._show_point_images(selected)

    def _point_choice_label(self, point: _SeriesPoint) -> str:
        face_label = "AVG" if point.face.upper() == "AVG" else point.face.upper()
        return f"{point.label} / {face_label} / Layer {point.image_num} / {point.percent:.2f}%"

    def _on_overlap_select(self, _event=None) -> None:
        selected = self._overlap_var.get()
        for point in self._overlap_points:
            if self._point_choice_label(point) == selected:
                self._show_point_images(point)
                return

    def _show_point_images(self, point: _SeriesPoint) -> None:
        self._shown_point = point
        issue_text = ""
        if point.is_issue:
            category = self._clean_text(point.row.get("issueCategory", ""))
            note = self._clean_text(point.row.get("issueNote", ""))
            issue_text = f"  ISSUE {category} {note}".strip()
        face_label = "FRONT/BACK AVG" if point.face.upper() == "AVG" else point.face
        self._selection_label.config(
            text=(
                f"{point.label} / {face_label} / Layer {point.image_num} / "
                f"{point.percent:.2f}%{issue_text}"
            )
        )
        image_title = "Cropped image" if point.face.upper() != "AVG" else "Cropped image (FRONT)"
        mask_title = "Mask image" if point.face.upper() != "AVG" else "Mask image (FRONT)"
        self._draw_image(self._crop_canvas, point.pass_image, image_title)
        self._draw_image(self._mask_canvas, point.mask_image, mask_title)

    def _redraw_heatmap(self) -> None:
        if not self._is_heatmap_tab_active():
            return

        self._heatmap_canvas.delete("all")
        width = max(self._heatmap_canvas.winfo_width(), 200)
        height = max(self._heatmap_canvas.winfo_height(), 200)
        self._heatmap_canvas.create_rectangle(0, 0, width, height, fill=_C_BG, outline="")

        option = self._selected_options.get(self._heatmap_option_key or "")
        if option is None:
            self._heatmap_label.config(
                text="Click a cell result on the left to build a heatmap.",
                foreground=_C_MUTED,
            )
            self._heatmap_canvas.create_text(
                width / 2,
                height / 2,
                text="No selected cell result.",
                fill=_C_MUTED,
            )
            return

        points = self._heatmap_source_points(option.key)
        if not points:
            self._heatmap_label.config(
                text=f"{option.label}: no middle-layer mask images for current view.",
                foreground=_C_MUTED,
            )
            self._heatmap_canvas.create_text(
                width / 2,
                height / 2,
                text="No mask images after excluding first and last layers.",
                fill=_C_MUTED,
            )
            return

        mode = self._heatmap_mode_var.get()
        heatmap = self._build_heatmap_image(points, mode)
        if heatmap is None:
            self._heatmap_label.config(
                text=f"{option.label}: failed to read mask images.",
                foreground=_C_MUTED,
            )
            self._heatmap_canvas.create_text(
                width / 2,
                height / 2,
                text="Failed to load mask images.",
                fill=_C_MUTED,
            )
            return

        suffix = "all TEST images included" if mode == "TEST" else "first/last layers excluded"
        self._heatmap_label.config(
            text=f"{option.label}: {mode}, {len(points)} mask(s), {suffix}.",
            foreground=_C_MUTED,
        )
        self._draw_array_on_canvas(self._heatmap_canvas, heatmap, "Accumulated heatmap")

    def _heatmap_source_points(self, option_key: str) -> list[_SeriesPoint]:
        mode = self._face_mode_var.get()
        points = [
            point for point in self._all_series_points
            if point.option_key == option_key and point.mask_image is not None
        ]
        if mode == "FRONT":
            points = [point for point in points if point.face.upper() == "FRONT"]
        elif mode == "BACK":
            points = [point for point in points if point.face.upper() == "BACK"]
        elif mode == "TEST":
            points = [point for point in points if point.face.upper() == "TEST"]
        elif mode == "Average":
            points = [point for point in points if point.face.upper() in {"FRONT", "BACK"}]

        if not points:
            return []
        if mode == "TEST":
            return sorted(points, key=lambda point: point.image_num)
        layer_nums = sorted({point.image_num for point in points})
        if len(layer_nums) <= 2:
            return []
        first_layer = layer_nums[0]
        last_layer = layer_nums[-1]
        middle_points = [
            point for point in points
            if first_layer < point.image_num < last_layer
        ]
        return sorted(middle_points, key=lambda point: (point.face, point.image_num))

    def _build_heatmap_image(self, points: list[_SeriesPoint], mode: str) -> np.ndarray | None:
        counts = self._heatmap_counts(points)
        if counts is None:
            return None

        li_counts, nonactive_counts, _count = counts
        if mode == "Nonactive":
            return self._single_channel_heatmap(nonactive_counts, (37, 99, 235))
        if mode == "Li-salt + Nonactive":
            return self._combined_heatmap(li_counts, nonactive_counts)
        return self._single_channel_heatmap(li_counts, (220, 38, 38))

    def _heatmap_counts(
        self,
        points: list[_SeriesPoint],
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        cache_key = self._heatmap_cache_key(points)
        if cache_key is not None and cache_key in self._heatmap_counts_cache:
            return self._heatmap_counts_cache[cache_key]

        base_shape: tuple[int, int] | None = None
        li_counts: np.ndarray | None = None
        nonactive_counts: np.ndarray | None = None
        count = 0

        for point in points:
            if point.mask_image is None:
                continue
            mask_image = cv2.imread(str(point.mask_image), cv2.IMREAD_COLOR)
            if mask_image is None:
                continue

            if base_shape is None:
                base_shape = mask_image.shape[:2]
                li_counts = np.zeros(base_shape, dtype=np.uint16)
                nonactive_counts = np.zeros(base_shape, dtype=np.uint16)
            elif mask_image.shape[:2] != base_shape:
                mask_image = cv2.resize(
                    mask_image,
                    (base_shape[1], base_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            li_mask = np.all(mask_image == _MASK_COLOR_LI_SALT, axis=2)
            nonactive_mask = np.all(mask_image == _MASK_COLOR_NONACTIVE, axis=2)
            li_counts += li_mask
            nonactive_counts += nonactive_mask
            count += 1

        if base_shape is None or li_counts is None or nonactive_counts is None:
            return None

        result = (li_counts, nonactive_counts, count)
        if cache_key is not None:
            self._heatmap_counts_cache[cache_key] = result
            while len(self._heatmap_counts_cache) > 8:
                self._heatmap_counts_cache.pop(next(iter(self._heatmap_counts_cache)))
        return result

    def _heatmap_cache_key(
        self,
        points: list[_SeriesPoint],
    ) -> tuple[str, str, tuple[tuple[str, int, int], ...]] | None:
        option_key = points[0].option_key if points else ""
        signature: list[tuple[str, int, int]] = []
        for point in points:
            if point.mask_image is None:
                return None
            try:
                stat = point.mask_image.stat()
            except OSError:
                return None
            signature.append((str(point.mask_image), stat.st_mtime_ns, stat.st_size))
        return (option_key, self._face_mode_var.get(), tuple(signature))

    def _single_channel_heatmap(
        self,
        counts: np.ndarray,
        color_rgb: tuple[int, int, int],
    ) -> np.ndarray:
        max_count = float(counts.max())
        norm = counts / max_count if max_count > 0 else counts
        background = np.full((*counts.shape, 3), 255, dtype=np.float32)
        color = np.array(color_rgb, dtype=np.float32)
        alpha = np.clip(norm, 0.0, 1.0) ** 0.7
        image = background * (1.0 - alpha[..., None]) + color * alpha[..., None]
        return image.astype(np.uint8)

    def _combined_heatmap(self, li_counts: np.ndarray, nonactive_counts: np.ndarray) -> np.ndarray:
        li_max = float(li_counts.max())
        nonactive_max = float(nonactive_counts.max())
        li_norm = li_counts / li_max if li_max > 0 else li_counts
        nonactive_norm = nonactive_counts / nonactive_max if nonactive_max > 0 else nonactive_counts

        image = np.full((*li_counts.shape, 3), 255, dtype=np.float32)
        red = np.array([220, 38, 38], dtype=np.float32)
        blue = np.array([37, 99, 235], dtype=np.float32)
        red_alpha = np.clip(li_norm, 0.0, 1.0) ** 0.7
        blue_alpha = np.clip(nonactive_norm, 0.0, 1.0) ** 0.7
        image = image * (1.0 - red_alpha[..., None]) + red * red_alpha[..., None]
        image = image * (1.0 - blue_alpha[..., None]) + blue * blue_alpha[..., None]
        return image.astype(np.uint8)

    def _draw_array_on_canvas(
        self,
        canvas: tk.Canvas,
        image_rgb: np.ndarray,
        title: str,
    ) -> None:
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 160)
        canvas.create_text(8, 8, text=title, anchor=tk.NW, fill=_C_AXIS)

        image_h, image_w = image_rgb.shape[:2]
        max_w = max(1, width - 16)
        max_h = max(1, height - 34)
        scale = min(max_w / image_w, max_h / image_h, 1.0)
        if scale < 1.0:
            resized = cv2.resize(
                image_rgb,
                (max(1, int(image_w * scale)), max(1, int(image_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image_rgb

        ok, buffer = cv2.imencode(".png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        if not ok:
            canvas.create_text(width / 2, height / 2, text="Failed to render image", fill=_C_MUTED)
            return

        photo = tk.PhotoImage(data=buffer.tobytes())
        self._photo_refs.append(photo)
        if len(self._photo_refs) > 12:
            self._photo_refs = self._photo_refs[-12:]
        canvas.create_image(width / 2, 28, image=photo, anchor=tk.N)

    def _draw_image(self, canvas: tk.Canvas, image_path: Path | None, title: str) -> None:
        canvas.delete("all")
        canvas.update_idletasks()
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 160)
        canvas.create_text(8, 8, text=title, anchor=tk.NW, fill=_C_AXIS)

        if image_path is None or not image_path.exists():
            canvas.create_text(width / 2, height / 2, text="Image not found", fill=_C_MUTED)
            return

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            canvas.create_text(width / 2, height / 2, text="Failed to load image", fill=_C_MUTED)
            return

        image_h, image_w = image.shape[:2]
        max_w = max(1, width - 16)
        max_h = max(1, height - 34)
        scale = min(max_w / image_w, max_h / image_h, 1.0)
        if scale < 1.0:
            resized = cv2.resize(
                image,
                (max(1, int(image_w * scale)), max(1, int(image_h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = image

        ok, buffer = cv2.imencode(".png", resized)
        if not ok:
            canvas.create_text(width / 2, height / 2, text="Failed to render image", fill=_C_MUTED)
            return

        photo = tk.PhotoImage(data=buffer.tobytes())
        self._photo_refs.append(photo)
        if len(self._photo_refs) > 8:
            self._photo_refs = self._photo_refs[-8:]
        canvas.create_image(width / 2, 28, image=photo, anchor=tk.N)

    def _clear_images(self) -> None:
        self._shown_point = None
        self._overlap_points.clear()
        self._overlap_combo.grid_remove()
        self._overlap_combo["values"] = ()
        self._overlap_var.set("")
        self._selection_label.config(text="Click a graph point to inspect images.")
        self._crop_canvas.delete("all")
        self._mask_canvas.delete("all")
        self._photo_refs.clear()
