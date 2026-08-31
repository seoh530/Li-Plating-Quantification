import logging
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import yaml

from modules.utils import path_resolver
from modules.utils.analysis_id import (
    find_test_source_folders,
    get_test_mask_folder_for_pass,
    get_test_output_folders,
    is_test_analysis_id,
)

logger = logging.getLogger(__name__)

_RESULT_PATTERN_OLD = "{analysis_id}_result.csv"
_RESULT_PATTERN_NEW = "{analysis_id}_LI-SALT_*.csv"
_IMAGE_EXTENSIONS = {".png"}

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
    analysis_id: str
    analysis_path: Path
    csv_path: Path
    cell: int
    label: str
    result_summary: str


@dataclass(frozen=True)
class _ResultRow:
    key: str
    option_key: str
    option: _ResultOption
    point: "_SeriesPoint"
    display_path: str
    face: str
    li_salt_percent: float
    modified_text: str
    modified_ts: float
    csv_display: str


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


class AnalysisTab(ttk.Frame):
    """Visual analysis view for ML result CSVs."""

    def __init__(self, parent, config_path: str | Path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._load_config()

        self._analysis_root_path = self._root_path
        self._root_path_var = tk.StringVar(value=str(self._root_path))
        self._folder_listing_cache = path_resolver.FolderListingCache()
        self._result_options: list[_ResultOption] = []
        self._option_by_item: dict[str, _ResultOption] = {}
        self._option_by_key: dict[str, _ResultOption] = {}
        self._result_rows: list[_ResultRow] = []
        self._row_by_item: dict[str, _ResultRow] = {}
        self._points_by_option: dict[str, list[_SeriesPoint]] = {}
        self._selected_options: dict[str, _ResultOption] = {}
        self._active_result_item: str | None = None
        self._result_sort_column = "csv"
        self._result_sort_desc = False
        self._all_series_points: list[_SeriesPoint] = []
        self._canvas_points: list[tuple[float, float, _SeriesPoint]] = []
        self._overlap_points: list[_SeriesPoint] = []
        self._shown_point: _SeriesPoint | None = None
        self._heatmap_option_key: str | None = None
        self._heatmap_counts_cache: dict[
            tuple[str, str, tuple[tuple[str, int, int], ...]],
            tuple[np.ndarray, np.ndarray, int],
        ] = {}
        self._photo_refs: list[tk.PhotoImage] = []

        self._build_ui()
        self._reset_results("Load a root path to browse completed ML results.")

    def reload_config(self) -> None:
        self._load_config()
        self._root_path_var.set(str(self._root_path))
        self._analysis_root_path = self._root_path
        self._folder_listing_cache.clear()
        self._reset_results("Settings reloaded. Load a root path to browse results.")

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._root_path = Path(cfg["app"]["root_path"])
        self._analysis_id_re = re.compile(cfg["folder_qc"]["analysis_id_pattern"])

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=0)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=(10, 6))
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="Root Path").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self._root_path_entry = ttk.Entry(header, textvariable=self._root_path_var)
        self._root_path_entry.grid(row=0, column=1, sticky=tk.EW)
        self._root_path_entry.bind("<Return>", lambda _event: self._load_root_path())
        self._root_path_entry.bind("<KP_Enter>", lambda _event: self._load_root_path())
        ttk.Button(header, text="Load Path", command=self._load_root_path).grid(
            row=0, column=2, sticky=tk.W, padx=(8, 0)
        )
        ttk.Button(header, text="Refresh", command=self._refresh).grid(
            row=0, column=3, sticky=tk.W, padx=(8, 0)
        )

        ttk.Label(header, text="View").grid(row=0, column=4, sticky=tk.W, padx=(12, 6))
        self._face_mode_var = tk.StringVar(value="FRONT")
        self._face_mode_combo = ttk.Combobox(
            header,
            textvariable=self._face_mode_var,
            values=_FACE_MODES,
            state="readonly",
            width=14,
        )
        self._face_mode_combo.grid(row=0, column=5, sticky=tk.W)
        self._face_mode_combo.bind("<<ComboboxSelected>>", self._on_face_mode_change)

        ttk.Button(header, text="Clear graph", command=self._clear_graph).grid(
            row=0, column=6, sticky=tk.W, padx=(8, 0)
        )
        self._status_label = ttk.Label(header, text="", foreground=_C_MUTED)
        self._status_label.grid(row=0, column=7, sticky=tk.W, padx=(12, 0))

        left = ttk.Frame(self)
        left.grid(row=1, column=0, sticky=tk.NS, padx=(10, 6), pady=(0, 8))
        left.rowconfigure(0, weight=1)

        results_frame = ttk.LabelFrame(left, text="ML Results", padding=8)
        results_frame.grid(row=0, column=0, sticky=tk.NSEW)
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        columns = ("face", "li_salt", "modified", "csv")
        self._result_tree = ttk.Treeview(
            results_frame,
            columns=columns,
            show="tree headings",
            height=22,
            selectmode="browse",
        )
        self._result_tree.heading("#0", text="Analysis ID / Image", command=lambda: self._sort_results("image"))
        self._result_tree.heading("face", text="FRONT/BACK", command=lambda: self._sort_results("face"))
        self._result_tree.heading("li_salt", text="LI_SALT %", command=lambda: self._sort_results("li_salt"))
        self._result_tree.heading("modified", text="Modified", command=lambda: self._sort_results("modified"))
        self._result_tree.heading("csv", text="CSV", command=lambda: self._sort_results("csv"))
        self._result_tree.column("#0", width=250, anchor=tk.W, stretch=True)
        self._result_tree.column("face", width=82, anchor=tk.W)
        self._result_tree.column("li_salt", width=82, anchor=tk.E)
        self._result_tree.column("modified", width=126, anchor=tk.W)
        self._result_tree.column("csv", width=180, anchor=tk.W)
        self._result_tree.grid(row=0, column=0, sticky=tk.NSEW)
        self._result_tree.tag_configure("selected", background=_C_SELECTED)
        self._result_tree.bind("<Button-1>", self._on_result_click)

        tree_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self._result_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._result_tree.configure(yscrollcommand=tree_scroll.set)
        x_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self._result_tree.xview)
        x_scroll.grid(row=1, column=0, sticky=tk.EW)
        self._result_tree.configure(xscrollcommand=x_scroll.set)
        self._setup_root_drag_drop()

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
            text="Click an ML result on the left to build a heatmap.",
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

        self._selection_label = ttk.Label(images, text="Click an ML result to inspect images.")
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

    def _setup_root_drag_drop(self) -> None:
        try:
            self.tk.call("package", "require", "tkdnd")
            for widget in (self._root_path_entry, self._result_tree):
                self.tk.call("tkdnd::drop_target", "register", widget, "DND_Files")
                widget.bind("<<Drop>>", self._on_root_drop, add="+")
        except tk.TclError:
            pass

    def _on_root_drop(self, event) -> str:
        paths = self._parse_dropped_paths(getattr(event, "data", ""))
        if paths:
            self._root_path_var.set(paths[0])
            self._load_root_path()
        return "break"

    def _parse_dropped_paths(self, data: str) -> list[str]:
        text = str(data or "").strip()
        if not text:
            return []
        if "\n" in text or "\r" in text or ";" in text:
            paths: list[str] = []
            for chunk in re.split(r"[\r\n;]+", text):
                paths.extend(self._parse_dropped_paths(chunk))
            return paths
        if re.match(r"^[A-Za-z]:(?:[\\/]|[^\\/])", text):
            return [path_resolver.clean_path(text)]
        try:
            parts = list(self.tk.splitlist(text))
        except tk.TclError:
            parts = [text]
        return [path_resolver.clean_path(part) for part in parts if str(part).strip()]

    def _refresh(self) -> None:
        self._load_root_path(clear_cache=True)

    def _load_root_path(self, clear_cache: bool = False) -> None:
        root_text = self._root_path_var.get().strip()
        if not root_text:
            self._reset_results("Enter a Root Path.")
            return
        if clear_cache:
            self._folder_listing_cache.clear()
        root_path = path_resolver.resolve_path(root_text)
        self._analysis_root_path = root_path
        self._root_path_var.set(str(root_path))
        self._reset_results("Loading ML results...")
        self.update_idletasks()
        try:
            rows = self._discover_result_rows(root_path)
        except OSError as exc:
            self._reset_results(f"Failed to load results: {exc}")
            return
        self._set_result_rows(rows)
        self._status_label.config(text=f"Loaded {len(rows)} ML result(s) from {root_path}")

    def _relative_path_label(self, path: Path | None) -> str:
        if path is None:
            return "-"
        try:
            relative = path.relative_to(self._analysis_root_path)
            return str(relative) if str(relative) != "." else "."
        except ValueError:
            return path.name or str(path)

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

    def _reset_results(self, status: str = "") -> None:
        for item in self._result_tree.get_children():
            self._result_tree.delete(item)
        self._option_by_item.clear()
        self._option_by_key.clear()
        self._row_by_item.clear()
        self._points_by_option.clear()
        self._result_rows.clear()
        self._result_options.clear()
        self._selected_options.clear()
        self._active_result_item = None
        self._all_series_points.clear()
        self._overlap_points.clear()
        self._shown_point = None
        self._heatmap_option_key = None
        self._heatmap_counts_cache.clear()
        self._clear_images()
        if status:
            self._status_label.config(text=status)
        self._redraw_graph()
        self._redraw_heatmap()

    def _set_result_rows(self, rows: list[_ResultRow]) -> None:
        self._result_rows = list(rows)
        self._sort_result_rows()
        group_items: dict[str, str] = {}
        for row in self._result_rows:
            group_key = f"{row.option.analysis_path}|{row.option.analysis_id}"
            group_id = group_items.get(group_key)
            if group_id is None:
                group_id = self._result_tree.insert(
                    "",
                    tk.END,
                    text=row.option.analysis_id,
                    values=("", "", "", self._relative_path_label(row.option.csv_path)),
                    open=True,
                )
                group_items[group_key] = group_id
            item_id = self._result_tree.insert(
                group_id,
                tk.END,
                text=row.display_path,
                values=(
                    row.face,
                    f"{row.li_salt_percent:.2f}",
                    row.modified_text,
                    row.csv_display,
                ),
            )
            self._row_by_item[item_id] = row
            self._option_by_item[item_id] = row.option
        self._sync_result_tree_styles()
        self._redraw_graph()
        self._redraw_heatmap()

    def _sort_results(self, column: str) -> None:
        if self._result_sort_column == column:
            self._result_sort_desc = not self._result_sort_desc
        else:
            self._result_sort_column = column
            self._result_sort_desc = column == "modified"
        for item in self._result_tree.get_children():
            self._result_tree.delete(item)
        self._row_by_item.clear()
        self._option_by_item.clear()
        self._set_result_rows(self._result_rows)

    def _sort_result_rows(self) -> None:
        column = self._result_sort_column
        reverse = self._result_sort_desc
        if column == "li_salt":
            key = lambda row: (row.li_salt_percent, row.display_path.lower())
        elif column == "face":
            key = lambda row: (row.face.upper(), row.display_path.lower())
        elif column == "image":
            key = lambda row: row.display_path.lower()
        elif column == "csv":
            key = lambda row: (row.csv_display.lower(), row.display_path.lower())
        else:
            key = lambda row: (row.modified_ts, row.display_path.lower())
        self._result_rows.sort(key=key, reverse=reverse)

    def _discover_result_rows(self, root_path: Path) -> list[_ResultRow]:
        options_by_key: dict[str, _ResultOption] = {}
        points_by_option: dict[str, list[_SeriesPoint]] = {}
        rows_by_identity: dict[str, _ResultRow] = {}
        for csv_path in self._iter_result_csvs(root_path):
            for row in self._rows_from_csv(csv_path, options_by_key, points_by_option):
                existing = rows_by_identity.get(row.key)
                if existing is None or row.modified_ts >= existing.modified_ts:
                    rows_by_identity[row.key] = row

        self._option_by_key = options_by_key
        self._result_options = list(options_by_key.values())
        self._points_by_option = {
            key: sorted(points, key=lambda point: (point.face.upper(), point.image_num))
            for key, points in points_by_option.items()
        }
        return list(rows_by_identity.values())

    def _iter_result_csvs(self, root_path: Path) -> list[Path]:
        csvs: list[Path] = []
        stack = [root_path]
        seen_dirs: set[str] = set()
        while stack:
            folder = stack.pop()
            folder_key = str(folder).lower()
            if folder_key in seen_dirs:
                continue
            seen_dirs.add(folder_key)
            try:
                children = self._folder_listing_cache.children(folder, allow_fallback=True)
            except OSError:
                continue
            for child in children:
                name = child.path.name
                lower = name.lower()
                if child.is_dir:
                    if lower.endswith("_pass"):
                        continue
                    stack.append(child.path)
                elif child.is_file and lower.endswith(".csv") and self._is_result_csv_name(name):
                    csvs.append(child.path)
        return sorted(csvs, key=lambda path: str(path).lower())

    def _is_result_csv_name(self, name: str) -> bool:
        lower = name.lower()
        return (
            lower.endswith("_result.csv")
            or "_li-salt_" in lower
            or "_li_salt_" in lower
            or lower.endswith("_li_salt.csv")
        )

    def _rows_from_csv(
        self,
        csv_path: Path,
        options_by_key: dict[str, _ResultOption],
        points_by_option: dict[str, list[_SeriesPoint]],
    ) -> list[_ResultRow]:
        try:
            frame = pd.read_csv(csv_path)
        except Exception:
            logger.exception("Failed to read result CSV: %s", csv_path)
            return []
        if frame.empty:
            frame = pd.DataFrame([{}])

        analysis_path = self._analysis_path_for_csv(csv_path)
        filename_percent = self._percent_from_filename(csv_path)
        rows: list[_ResultRow] = []
        for index, series in frame.iterrows():
            row_dict = series.to_dict()
            percent = self._li_salt_percent_from_row(row_dict, filename_percent)
            if percent is None:
                continue
            cell = self._int_from_row(row_dict, "Cell", "cell", default=1)
            image_num = self._int_from_row(row_dict, "ImageNum", "image_num", default=index + 1)
            analysis_id = self._analysis_id_for_row(row_dict, analysis_path)
            face = self._face_for_row(row_dict, csv_path)
            pass_image = self._path_from_row(
                row_dict,
                csv_path,
                "crop_image",
                "pass_image_path",
                "pass_image",
            )
            mask_image = self._path_from_row(row_dict, csv_path, "mask_image", "mask_path")
            if pass_image is None and mask_image is not None:
                pass_image = self._pass_image_for_mask(mask_image)
            if pass_image is None:
                pass_image = self._find_pass_image(analysis_path, cell, face, image_num)
            if mask_image is None:
                mask_image = self._find_mask_image(pass_image, face)
            face = self._infer_face(face, csv_path, pass_image, mask_image)
            row_dict.setdefault("li_salt_ratio", percent / 100.0)

            option_key = f"{analysis_path}|{cell}|{csv_path.name}"
            option = options_by_key.get(option_key)
            if option is None:
                option = _ResultOption(
                    key=option_key,
                    analysis_id=analysis_id,
                    analysis_path=analysis_path,
                    csv_path=csv_path,
                    cell=cell,
                    label=f"{analysis_id} Cell #{cell}",
                    result_summary=self._cell_result_summary(frame, cell),
                )
                options_by_key[option_key] = option

            point = _SeriesPoint(
                option_key=option_key,
                label=option.label,
                face=face,
                image_num=image_num,
                percent=percent,
                is_issue=self._coerce_bool(row_dict.get("isIssue", False)),
                row=row_dict,
                pass_image=pass_image if pass_image and pass_image.exists() else pass_image,
                mask_image=mask_image if mask_image and mask_image.exists() else None,
            )
            points_by_option.setdefault(option_key, []).append(point)
            display_path = self._display_path_for_result(csv_path, pass_image, mask_image)
            modified_ts = self._modified_ts(csv_path, pass_image, mask_image)
            result_row = _ResultRow(
                key=self._result_identity(csv_path, pass_image, mask_image, row_dict, cell, face, image_num),
                option_key=option_key,
                option=option,
                point=point,
                display_path=display_path,
                face=face,
                li_salt_percent=percent,
                modified_text=self._format_modified(modified_ts),
                modified_ts=modified_ts,
                csv_display=self._relative_path_label(csv_path),
            )
            rows.append(result_row)
        return rows

    def _is_analysis_folder(self, folder_name: str) -> bool:
        return bool(self._analysis_id_re.match(folder_name)) or is_test_analysis_id(folder_name)

    def _analysis_path_for_csv(self, csv_path: Path) -> Path:
        parent = csv_path.parent
        lower = parent.name.lower()
        if lower.endswith("_mask") or lower.endswith("_masks"):
            return parent.parent
        return parent

    def _analysis_id_for_row(self, row: dict, analysis_path: Path) -> str:
        for key in ("AnalysisID", "analysis_id", "analysisId"):
            value = self._clean_text(row.get(key, ""))
            if value:
                return value
        return analysis_path.name

    def _face_for_row(self, row: dict, csv_path: Path) -> str:
        value = self._clean_text(row.get("Face", row.get("face", ""))).upper()
        if value:
            return value
        return self._infer_face("", csv_path, None, None)

    def _infer_face(
        self,
        face: str,
        csv_path: Path,
        pass_image: Path | None,
        mask_image: Path | None,
    ) -> str:
        candidates = [
            face,
            pass_image.parent.name if pass_image is not None else "",
            mask_image.parent.name if mask_image is not None else "",
            csv_path.parent.name,
        ]
        for candidate in candidates:
            upper = str(candidate).upper()
            if "FRONT" in upper:
                return "FRONT"
            if "BACK" in upper:
                return "BACK"
            if "TEST" in upper:
                return "TEST"
        return str(face or "UNKNOWN").upper()

    def _int_from_row(self, row: dict, *keys: str, default: int) -> int:
        for key in keys:
            value = row.get(key)
            try:
                if value is not None and not pd.isna(value):
                    return int(float(value))
            except (TypeError, ValueError):
                continue
        return default

    def _li_salt_percent_from_row(self, row: dict, fallback: float | None) -> float | None:
        for key in ("li_salt_ratio", "li_salt_percent", "LI_SALT %", "LI_SALT"):
            value = row.get(key)
            try:
                if value is None or pd.isna(value):
                    continue
                number = float(value)
            except (TypeError, ValueError):
                continue
            if key == "li_salt_ratio":
                return self._ratio_to_percent(number)
            return number
        return fallback

    def _percent_from_filename(self, path: Path) -> float | None:
        match = re.search(r"_LI[-_]SALT_([-+]?[0-9]*\.?[0-9]+)%", path.name, re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _path_from_row(self, row: dict, csv_path: Path, *keys: str) -> Path | None:
        for key in keys:
            text = self._clean_text(row.get(key, ""))
            if not text:
                continue
            path = path_resolver.resolve_path(text)
            if not path.is_absolute():
                path = csv_path.parent / path
            return path
        return None

    def _pass_image_for_mask(self, mask_image: Path) -> Path | None:
        stem = mask_image.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        parent_name = mask_image.parent.name
        candidates = []
        for suffix in ("_MASK", "_mask", "_MASKS", "_masks"):
            if parent_name.endswith(suffix):
                candidates.append(mask_image.parent.with_name(parent_name[: -len(suffix)] + "_PASS"))
                candidates.append(mask_image.parent.with_name(parent_name[: -len(suffix)] + "_pass"))
        candidates.append(mask_image.parent.with_name(mask_image.parent.name.replace("MASK", "PASS")))
        for folder in candidates:
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
                path = folder / f"{stem}{ext}"
                if path.exists():
                    return path
        return None

    def _display_path_for_result(
        self,
        csv_path: Path,
        pass_image: Path | None,
        mask_image: Path | None,
    ) -> str:
        if pass_image is not None:
            return self._relative_path_label(pass_image)
        if mask_image is not None:
            return self._display_label_from_mask(mask_image)
        if csv_path.parent.name.lower().endswith(("_mask", "_masks")):
            return self._display_label_from_mask(csv_path)
        return self._relative_path_label(csv_path)

    def _display_label_from_mask(self, path: Path) -> str:
        parent_name = path.parent.name
        display_parent = path.parent
        for suffix in ("_MASKS", "_masks", "_MASK", "_mask"):
            if parent_name.endswith(suffix):
                display_parent = path.parent.with_name(parent_name[: -len(suffix)])
                break
        stem = path.stem
        if stem.endswith("_mask"):
            stem = stem[:-5]
        stem = re.sub(r"_LI[-_]SALT(?:_[-+]?[0-9]*\.?[0-9]+%)?$", "", stem, flags=re.IGNORECASE)
        return self._relative_path_label(display_parent / f"{stem}.png")

    def _modified_ts(
        self,
        csv_path: Path,
        pass_image: Path | None,
        mask_image: Path | None,
    ) -> float:
        paths = [path for path in (csv_path, mask_image, pass_image) if path is not None]
        values = []
        for path in paths:
            try:
                values.append(path.stat().st_mtime)
            except OSError:
                continue
        return max(values) if values else 0.0

    def _format_modified(self, timestamp: float) -> str:
        if timestamp <= 0:
            return "-"
        return pd.Timestamp.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")

    def _result_identity(
        self,
        csv_path: Path,
        pass_image: Path | None,
        mask_image: Path | None,
        row: dict,
        cell: int,
        face: str,
        image_num: int,
    ) -> str:
        for path in (mask_image, pass_image):
            if path is not None:
                return str(path).lower()
        source = self._clean_text(row.get("source_image", row.get("image_name", "")))
        if source:
            return source.lower()
        return f"{csv_path}|{cell}|{face}|{image_num}".lower()

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
        if item_id not in self._row_by_item:
            self._result_tree.item(item_id, open=not bool(self._result_tree.item(item_id, "open")))
            return "break"
        self._select_result_item(item_id)
        return "break"

    def _select_result_item(self, item_id: str) -> None:
        row = self._row_by_item.get(item_id)
        if row is None:
            return
        option = row.option
        if option.key not in self._selected_options:
            self._selected_options[option.key] = option
            self._add_option_points(option)
        self._active_result_item = item_id
        self._heatmap_option_key = option.key
        if row.face in _FACE_MODES:
            self._face_mode_var.set(row.face)
        self._sync_result_tree_styles()
        self._redraw_graph()
        self._redraw_heatmap()
        self._show_point_images(row.point)

    def _sync_result_tree_styles(self) -> None:
        for item_id, row in self._row_by_item.items():
            tags = ("selected",) if item_id == self._active_result_item else ()
            self._result_tree.item(item_id, tags=tags)

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

    def _add_option_points(self, option: _ResultOption) -> None:
        points = self._points_by_option.get(option.key)
        if points:
            existing = {id(point) for point in self._all_series_points}
            self._all_series_points.extend(point for point in points if id(point) not in existing)
            return
        self._add_option_series(option)

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
            pass_path = self._first_existing_dir(
                cell_path / f"{face}_pass",
                cell_path / f"{face}_PASS",
                cell_path / f"{face.upper()}_pass",
                cell_path / f"{face.upper()}_PASS",
            )
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
        for folder in (
            pass_image.parent.with_name(f"{face}_mask"),
            pass_image.parent.with_name(f"{face}_MASK"),
            pass_image.parent.with_name(f"{face.upper()}_mask"),
            pass_image.parent.with_name(f"{face.upper()}_MASK"),
            pass_image.parent.with_name(pass_image.parent.name.replace("_PASS", "_MASK")),
            pass_image.parent.with_name(pass_image.parent.name.replace("_pass", "_mask")),
        ):
            mask_path = folder / mask_name
            if mask_path.exists():
                return mask_path

        legacy_path = pass_image.with_name(mask_name)
        if legacy_path.exists():
            return legacy_path
        return None

    def _first_existing_dir(self, *paths: Path) -> Path:
        for path in paths:
            if path.is_dir():
                return path
        return paths[0]

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
        self._active_result_item = None
        self._clear_images()
        self._sync_result_tree_styles()
        self._redraw_graph()
        self._redraw_heatmap()

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
                text="Click an ML result on the left, or change the face view mode.",
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

            label = self._selected_options[option_key].label
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
                text="Click an ML result on the left to build a heatmap.",
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
        self._selection_label.config(text="Click an ML result to inspect images.")
        self._crop_canvas.delete("all")
        self._mask_canvas.delete("all")
        self._photo_refs.clear()
