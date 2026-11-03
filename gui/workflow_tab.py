import json
import logging
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import yaml

from modules.ml.background_normalizer import BackgroundLABNormalizer
from modules.ml.feature_extractor import FeatureExtractor
from modules.ml.drift_detector import BackgroundDrift, DriftDetector
from modules.ml.inference import InferenceEngine, LABEL_BACKGROUND
from modules.ml.model_registry import resolve_model_path
from modules.ml.quantifier import Quantifier
from modules.qc.crop_engine import CropEngine
from modules.qc.folder_qc import FolderQC
from modules.storage.sqlite_logger import SQLitePredictionLogger
from modules.utils.analysis_id import (
    build_test_output_name,
    ensure_test_output_folders,
    find_test_images,
    find_test_source_folders,
    get_test_mask_folder_for_pass,
    get_test_output_folders,
    is_test_analysis_id,
)
from modules.utils.image_io import quiet_imread

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_RESULT_PATTERN_NEW = "{analysis_id}_LI-SALT_*.csv"
_RESULT_PATTERN_OLD = "{analysis_id}_result.csv"
_PREVIEW_BG = "#2b3036"
_PREVIEW_TEXT = "#e5e7eb"
_PREVIEW_MUTED = "#aeb6c2"
_HOLD_STATE_FILENAME = "_qc_hold.json"
_DEFAULT_WORKFLOW_MODEL = "models/V2F"


@dataclass
class _WorkflowItem:
    key: str
    date_name: str
    date_path: Path
    analysis_id: str
    analysis_path: Path
    qc_status: str = "UNKNOWN"
    ml_status: str = "UNKNOWN"
    li_salt: str = "-"
    correction: str = "-"
    summary: str = ""


@dataclass(frozen=True)
class _ImageTask:
    analysis_id: str
    analysis_path: Path
    cell: int
    face: str
    image_num: int
    image_path: Path


class WorkflowTab(ttk.Frame):
    """Integrated QC -> ML workflow dashboard."""

    def __init__(self, parent, config_path: str | Path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._load_config()

        self._folder_qc = FolderQC(config_path)
        self._crop_engine = CropEngine(config_path)
        self._feature_extractor = FeatureExtractor(config_path)
        self._inference: InferenceEngine | None = None
        self._quantifier = Quantifier(config_path)
        self._sqlite_logger = SQLitePredictionLogger(config_path)

        self._color_normalizer = None
        try:
            from modules.qc.color_normalizer import ColorNormalizer
            self._color_normalizer = ColorNormalizer(config_path)
        except RuntimeError:
            logger.debug("WorkflowTab: ColorNormalizer skipped")

        self._queue: queue.Queue = queue.Queue()
        self._items: list[_WorkflowItem] = []
        self._item_by_tree_id: dict[str, _WorkflowItem] = {}
        self._held_qc_failures: dict[str, dict[str, object]] = {}
        self._hold_row_meta: dict[str, tuple[str, str]] = {}
        self._active_run_ml = False
        self._pending_hold_ml: list[dict[str, object]] = []
        self._running = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._photo_refs: list[tk.PhotoImage] = []
        self._photo_cache: dict[tuple[str, int, int, int], tk.PhotoImage] = {}
        self._photo_cache_order: list[tuple[str, int, int, int]] = []
        self._save_mask_images = True
        self._sort_column: str | None = None
        self._sort_reverse = False
        self._preview_enabled = True
        self._review_entries: list[dict[str, object]] = []
        self._review_index = 0
        self._normalization_enabled = False
        self._normalization_cfg: dict = {}
        self._low_confidence_threshold = 0.60
        self._drift_detector: DriftDetector | None = None
        self._background_normalizer: BackgroundLABNormalizer | None = None
        self._selected_customer = "HKMC"
        self._external_work_paths: list[Path] = []
        self._external_original_roots: list[Path] = []
        self._external_stage_root: Path | None = None
        self._external_stage_map: list[tuple[Path, Path]] = []
        self._external_direct_crop_ml = False
        self._external_requested_faces_by_stage: dict[str, set[str]] = {}

        self._build_ui()
        self._refresh_items()
        self._poll_queue()

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._root_path = Path(cfg["app"]["root_path"])
        self._cell_folder_re = re.compile(
            cfg["folder_qc"].get("cell_folder_pattern", r"^cell\s*#\d+$"),
            re.IGNORECASE,
        )
        self._date_formats = list(cfg["folder_qc"]["date_formats"])
        self._customer_codes = dict(cfg["folder_qc"].get("customer_codes", {}))

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=(10, 6))
        toolbar.columnconfigure(12, weight=1)

        ttk.Label(toolbar, text="View").grid(row=0, column=0, sticky=tk.W)
        self._mode_var = tk.StringVar(value="By Date")
        self._mode_combo = ttk.Combobox(
            toolbar,
            textvariable=self._mode_var,
            values=("By Date", "By Folder ID"),
            state="readonly",
            width=16,
        )
        self._mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(6, 12))
        self._mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_items())

        ttk.Label(toolbar, text="Date").grid(row=0, column=2, sticky=tk.W)
        self._date_var = tk.StringVar()
        self._date_combo = ttk.Combobox(
            toolbar,
            textvariable=self._date_var,
            state="readonly",
            width=16,
        )
        self._date_combo.grid(row=0, column=3, sticky=tk.W, padx=(6, 12))
        self._date_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_items())

        ttk.Label(toolbar, text="Customer").grid(row=0, column=4, sticky=tk.W)
        self._customer_var = tk.StringVar(value=self._selected_customer)
        self._customer_combo = ttk.Combobox(
            toolbar,
            textvariable=self._customer_var,
            values=("HKMC", "FORD", "VW", "ESS", "ETC"),
            state="readonly",
            width=10,
        )
        self._customer_combo.grid(row=0, column=5, sticky=tk.W, padx=(6, 12))
        self._customer_combo.bind("<<ComboboxSelected>>", self._on_customer_change)

        self._refresh_btn = ttk.Button(toolbar, text="Refresh", command=self._refresh_items)
        self._refresh_btn.grid(row=0, column=6, sticky=tk.W)
        self._run_qc_var = tk.BooleanVar(value=True)
        self._run_qc_check = ttk.Checkbutton(
            toolbar,
            text="QC",
            variable=self._run_qc_var,
        )
        self._run_qc_check.grid(row=0, column=7, sticky=tk.W, padx=(10, 0))
        self._run_ml_var = tk.BooleanVar(value=True)
        self._run_ml_check = ttk.Checkbutton(
            toolbar,
            text="ML",
            variable=self._run_ml_var,
        )
        self._run_ml_check.grid(row=0, column=8, sticky=tk.W, padx=(4, 0))
        self._run_btn = ttk.Button(
            toolbar,
            text="Run",
            command=self._run_requested,
        )
        self._run_btn.grid(row=0, column=9, sticky=tk.W, padx=(10, 0))
        self._stop_btn = ttk.Button(
            toolbar,
            text="Stop",
            command=self._request_stop,
            state=tk.DISABLED,
        )
        self._stop_btn.grid(row=0, column=10, sticky=tk.W, padx=(6, 0))
        self._force_run_var = tk.BooleanVar(value=False)
        self._force_run_check = ttk.Checkbutton(
            toolbar,
            text="Force Run",
            variable=self._force_run_var,
        )
        self._force_run_check.grid(row=0, column=11, sticky=tk.W, padx=(10, 0))
        self._status_label = ttk.Label(toolbar, text="", foreground="#555555")
        self._status_label.grid(row=0, column=12, sticky=tk.W, padx=(12, 0))

        # Workflow uses one deployment model; keep the existing ML loader path
        # while removing model selection from the operator-facing toolbar.
        self._model_var = tk.StringVar(value=_DEFAULT_WORKFLOW_MODEL)

        external_frame = ttk.Frame(self)
        external_frame.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=(0, 6))
        external_frame.columnconfigure(0, weight=1)

        self._drop_label = tk.Label(
            external_frame,
            text="Drag & Drop Here",
            fg="#1f2937",
            bg="#eef6ff",
            relief=tk.GROOVE,
            borderwidth=2,
            padx=14,
            pady=18,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            font=("TkDefaultFont", 11, "bold"),
            cursor="hand2",
        )
        self._drop_label.grid(row=0, column=0, sticky=tk.EW)

        self._clear_external_btn = ttk.Button(
            external_frame,
            text="Clear",
            command=self._clear_external_work_paths,
        )
        self._clear_external_btn.grid(row=0, column=1, sticky=tk.E, padx=(8, 0))

        self._enable_drop_target()

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=2, column=0, sticky=tk.NSEW, padx=10, pady=(0, 10))

        left = ttk.Frame(paned)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        paned.add(left, weight=3)

        self._left_notebook = ttk.Notebook(left)
        self._left_notebook.grid(row=0, column=0, sticky=tk.NSEW)

        all_items_tab = ttk.Frame(self._left_notebook)
        all_items_tab.columnconfigure(0, weight=1)
        all_items_tab.rowconfigure(0, weight=1)
        self._left_notebook.add(all_items_tab, text="All Items")

        hold_items_tab = ttk.Frame(self._left_notebook)
        hold_items_tab.columnconfigure(0, weight=1)
        hold_items_tab.rowconfigure(0, weight=1)
        self._left_notebook.add(hold_items_tab, text="QC Hold Images (0)")
        self._hold_tab = hold_items_tab

        columns = ("source", "date", "analysis_id", "qc", "ml", "li_salt", "correction", "summary")
        self._tree = ttk.Treeview(all_items_tab, columns=columns, show="headings", selectmode="extended")
        headings = {
            "source": "Source",
            "date": "Date",
            "analysis_id": "Folder ID",
            "qc": "QC",
            "ml": "ML / DB",
            "li_salt": "Li-salt %",
            "correction": "Correction",
            "summary": "Summary",
        }
        widths = {
            "source": 105,
            "date": 95,
            "analysis_id": 110,
            "qc": 95,
            "ml": 100,
            "li_salt": 80,
            "correction": 85,
            "summary": 265,
        }
        for col in columns:
            self._tree.heading(
                col,
                text=headings[col],
                command=lambda column=col: self._sort_by_column(column),
            )
            self._tree.column(col, width=widths[col], anchor=tk.W)
        self._tree.grid(row=0, column=0, sticky=tk.NSEW)
        self._tree.bind("<Button-1>", self._on_tree_click)
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected_preview())
        self._tree.tag_configure("red", background="#fee2e2")
        self._tree.tag_configure("yellow", background="#fef9c3")
        self._tree.tag_configure("blue", background="#dbeafe")

        tree_scroll = ttk.Scrollbar(all_items_tab, orient=tk.VERTICAL, command=self._tree.yview)
        tree_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._tree.configure(yscrollcommand=tree_scroll.set)

        hold_columns = ("folder_id", "cell", "face", "image", "reason")
        self._hold_tree = ttk.Treeview(
            hold_items_tab,
            columns=hold_columns,
            show="headings",
            selectmode="extended",
        )
        for column, title, width in (
            ("folder_id", "Folder ID", 115),
            ("cell", "Cell", 70),
            ("face", "Face", 70),
            ("image", "Image", 190),
            ("reason", "Fail Reason", 300),
        ):
            self._hold_tree.heading(column, text=title)
            self._hold_tree.column(column, width=width, anchor=tk.W, stretch=(column == "reason"))
        self._hold_tree.grid(row=0, column=0, sticky=tk.NSEW)
        hold_scroll = ttk.Scrollbar(hold_items_tab, orient=tk.VERTICAL, command=self._hold_tree.yview)
        hold_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._hold_tree.configure(yscrollcommand=hold_scroll.set)
        self._hold_tree.bind("<<TreeviewSelect>>", self._on_hold_tree_select)

        right = ttk.Frame(paned)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(4, weight=1)
        paned.add(right, weight=2)

        self._progress = ttk.Progressbar(right, mode="determinate", maximum=1, value=0)
        self._progress.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
        self._progress_label = ttk.Label(right, text="Progress: 0/0 images", foreground="#555555")
        self._progress_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 3))
        self._current_label = ttk.Label(right, text="Current: -")
        self._current_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 6))

        legend = ttk.Frame(right)
        legend.grid(row=3, column=0, sticky=tk.W, pady=(0, 6))
        self._preview_var = tk.BooleanVar(value=True)
        self._preview_check = ttk.Checkbutton(
            legend,
            text="Preview",
            variable=self._preview_var,
            command=self._toggle_preview,
        )
        self._preview_check.pack(side=tk.LEFT, padx=(0, 14))
        for text, color in (
            ("QC missing/fail", "#fee2e2"),
            ("ML/DB missing", "#fef9c3"),
            ("Complete", "#dbeafe"),
        ):
            tk.Label(legend, width=2, bg=color, relief=tk.GROOVE).pack(side=tk.LEFT)
            ttk.Label(legend, text=text).pack(side=tk.LEFT, padx=(4, 12))

        review = ttk.LabelFrame(right, text="Before / After Review", padding=8)
        review.grid(row=4, column=0, sticky=tk.NSEW)
        review.columnconfigure(0, weight=1)
        review.rowconfigure(3, weight=1)
        review.rowconfigure(4, weight=1)

        self._review_label = ttk.Label(review, text="Select a Folder ID to preview.")
        self._review_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 6))
        review_nav = ttk.Frame(review)
        review_nav.grid(row=1, column=0, sticky=tk.EW, pady=(0, 6))
        review_nav.columnconfigure(0, weight=1)
        self._review_count_label = ttk.Label(review_nav, text="", foreground="#555555")
        self._review_count_label.grid(row=0, column=0, sticky=tk.W)
        self._prev_review_btn = ttk.Button(
            review_nav,
            text="Previous",
            command=self._show_previous_review,
            state=tk.DISABLED,
        )
        self._prev_review_btn.grid(row=0, column=1, sticky=tk.E, padx=(0, 6))
        self._next_review_btn = ttk.Button(
            review_nav,
            text="Next",
            command=self._show_next_review,
            state=tk.DISABLED,
        )
        self._next_review_btn.grid(row=0, column=2, sticky=tk.E)
        self._retake_review_btn = ttk.Button(
            review_nav,
            text="Retake done",
            command=lambda: self._process_current_qc_ng(force=False),
            state=tk.DISABLED,
        )
        self._retake_review_btn.grid(row=0, column=3, sticky=tk.E, padx=(10, 6))
        self._crop_review_btn = ttk.Button(
            review_nav,
            text="Crop anyway",
            command=lambda: self._process_current_qc_ng(force=True),
            state=tk.DISABLED,
        )
        self._crop_review_btn.grid(row=0, column=4, sticky=tk.E)

        bulk_actions = ttk.Frame(review)
        bulk_actions.grid(row=2, column=0, sticky=tk.EW, pady=(0, 6))
        bulk_actions.columnconfigure(0, weight=1)
        self._crop_selected_btn = ttk.Button(
            bulk_actions,
            text="Crop Selected Anyway",
            command=self._crop_selected_hold_images,
            state=tk.DISABLED,
        )
        self._crop_selected_btn.grid(row=0, column=1, sticky=tk.E, padx=(0, 6))
        self._crop_all_btn = ttk.Button(
            bulk_actions,
            text="Crop All Anyway",
            command=self._crop_all_hold_images,
            state=tk.DISABLED,
        )
        self._crop_all_btn.grid(row=0, column=2, sticky=tk.E)

        self._before_canvas = tk.Canvas(review, bg=_PREVIEW_BG, height=165, highlightthickness=1)
        self._before_canvas.grid(row=3, column=0, sticky=tk.NSEW, pady=(0, 5))
        self._after_canvas = tk.Canvas(review, bg=_PREVIEW_BG, height=165, highlightthickness=1)
        self._after_canvas.grid(row=4, column=0, sticky=tk.NSEW, pady=(5, 0))
        self.bind_all("<Tab>", self._on_review_tab_key, add="+")

    def reload_settings(self) -> None:
        if self._running:
            messagebox.showwarning(
                "Settings",
                "Workflow is running. Stop or wait for completion before reloading settings.",
            )
            return

        self._load_config()
        self._folder_qc = FolderQC(self._config_path)
        self._crop_engine = CropEngine(self._config_path)
        self._feature_extractor = FeatureExtractor(self._config_path)
        self._quantifier = Quantifier(self._config_path)
        self._sqlite_logger = SQLitePredictionLogger(self._config_path)
        self._inference = None
        self._held_qc_failures.clear()
        self._active_run_ml = False
        self._photo_cache.clear()
        self._photo_cache_order.clear()
        try:
            from modules.qc.color_normalizer import ColorNormalizer
            self._color_normalizer = ColorNormalizer(self._config_path)
        except RuntimeError:
            logger.debug("WorkflowTab: ColorNormalizer skipped")
            self._color_normalizer = None

        self._model_var.set(_DEFAULT_WORKFLOW_MODEL)

        self._refresh_items()
        self._status_label.config(text=f"Settings reloaded: {self._root_path}")

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

    def _is_work_folder(self, path: Path) -> bool:
        """Return True when a DATE child contains at least one usable source set."""
        if not path.is_dir() or path.name.lower().endswith(("_pass", "_mask")):
            return False
        try:
            children = list(path.iterdir())
        except OSError:
            return False

        if any(
            child.is_file() and child.suffix.lower() in _IMAGE_EXTENSIONS
            for child in children
        ):
            return True

        for child in children:
            if not child.is_dir():
                continue
            if child.name.upper() in {"FRONT", "BACK"}:
                return True
            if not self._cell_folder_re.match(child.name):
                continue
            try:
                nested_items = list(child.iterdir())
            except OSError:
                continue
            if any(
                nested.is_file() and nested.suffix.lower() in _IMAGE_EXTENSIONS
                for nested in nested_items
            ):
                return True
            if any(
                nested.is_dir() and nested.name.upper() in {"FRONT", "BACK"}
                for nested in nested_items
            ):
                return True
        return False

    def _clear_external_work_paths(self) -> None:
        if self._running:
            return
        self._cleanup_external_stage()
        self._external_work_paths = []
        self._external_original_roots = []
        self._external_direct_crop_ml = False
        self._external_requested_faces_by_stage.clear()
        self._drop_label.config(
            text="Drag & Drop Here",
            bg="#eef6ff",
        )
        self._refresh_items()

    def _enable_drop_target(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES
        except Exception:
            self._drop_label.config(
                text="Drag & Drop Here",
                bg="#fff7ed",
            )
            return
        registered = False
        for widget in (self, self._drop_label):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_workflow_drop)
                registered = True
            except Exception:
                logger.debug("Workflow drag/drop target unavailable on %s", widget, exc_info=True)
        if not registered:
            self._drop_label.config(
                text="Drag & Drop Here",
                bg="#fff7ed",
            )

    def _on_workflow_drop(self, event) -> str:
        if self._running:
            return "break"
        paths = [
            self._normalize_dropped_path(value)
            for value in self.tk.splitlist(getattr(event, "data", ""))
        ]
        self._set_external_work_paths(
            [path for path in paths if path is not None],
            direct_crop_ml=True,
        )
        return "break"

    def _normalize_dropped_path(self, value: str) -> Path | None:
        value = str(value).strip().strip("{}")
        if not value:
            return None
        if value.lower().startswith("file:"):
            parsed = urlparse(value)
            if parsed.netloc:
                return Path(f"//{parsed.netloc}{unquote(parsed.path)}")
            return Path(unquote(parsed.path.lstrip("/")))
        return Path(value)

    def _set_external_work_paths(
        self,
        paths: list[Path],
        *,
        direct_crop_ml: bool = False,
    ) -> None:
        # Match ReportTab secure-path behavior: do not probe the external source
        # with pathlib. Resolve the DATE/work root lexically, then stage with cmd.
        original_roots: list[Path] = []
        seen: set[str] = set()
        requested_faces: dict[str, set[str]] = {}
        for raw_path in paths:
            raw_path = Path(raw_path)
            raw_face = raw_path.name.upper() if raw_path.name.upper() in {"FRONT", "BACK"} else ""

            # FRONT/BACK is a valid direct Drag & Drop source by itself.
            # Do not require it to live under a recognized DATE/work-folder tree.
            # Stage its parent so sibling *_pass/_mask outputs keep the normal layout.
            target = raw_path.parent if raw_face else self._workflow_root_from_external_path(raw_path)
            if target is None:
                continue
            key = str(target).rstrip("\\/").lower()
            if raw_face:
                requested_faces.setdefault(key, set()).add(raw_face)
            if key in seen:
                continue
            seen.add(key)
            original_roots.append(target)

        if not original_roots:
            messagebox.showwarning(
                "Workflow",
                "No valid Drag & Drop source found.\n\n"
                "Drop a FRONT/BACK folder directly, or a DATE/work folder that contains images.",
            )
            return

        try:
            staged_roots = self._stage_external_workflow_roots(original_roots)
        except Exception as exc:
            logger.exception("Workflow external staging failed")
            messagebox.showerror("Workflow", str(exc))
            return

        self._external_original_roots = original_roots
        self._external_work_paths = staged_roots
        self._external_direct_crop_ml = bool(direct_crop_ml)
        self._external_requested_faces_by_stage = {}
        for staged_root, original_root in self._external_stage_map:
            original_key = str(original_root).rstrip("\\/").lower()
            faces = requested_faces.get(original_key)
            if faces:
                self._external_requested_faces_by_stage[str(staged_root).lower()] = set(faces)
        self._drop_label.config(
            text="Drag & Drop Here",
            bg="#dcfce7",
        )
        self._refresh_items()

    def _workflow_root_from_external_path(self, path: Path) -> Path | None:
        # No exists/is_dir/iterdir here: secure Team Room paths can reject Python stat.
        if self._parse_date_folder(path.name) is not None:
            return path
        for candidate in [path, *path.parents]:
            if self._parse_date_folder(candidate.parent.name) is not None:
                return candidate
        return None

    def _stage_external_workflow_roots(self, original_roots: list[Path]) -> list[Path]:
        self._cleanup_external_stage()
        stage_root = Path(tempfile.mkdtemp(prefix="photojig_workflow_"))
        staged_roots: list[Path] = []
        stage_map: list[tuple[Path, Path]] = []
        try:
            for index, source in enumerate(original_roots, start=1):
                destination = stage_root / f"source_{index}" / source.name
                self._copy_external_to_local_subprocess(source, destination)
                if not self._local_path_has_content(destination):
                    raise RuntimeError(
                        "No files were copied from the selected workflow path.\n\n"
                        f"Source: {source}\n"
                        "Check folder access or select a lower DATE/work folder."
                    )
                staged_roots.append(destination)
                stage_map.append((destination, source))
        except Exception:
            shutil.rmtree(stage_root, ignore_errors=True)
            raise
        self._external_stage_root = stage_root
        self._external_stage_map = stage_map
        return staged_roots

    def _external_targets(self) -> list[tuple[str, Path, str]]:
        targets: list[tuple[str, Path, str]] = []
        seen: set[tuple[str, str]] = set()

        for path in self._external_work_paths:
            if self._parse_date_folder(path.name) is not None:
                date_name = path.name
                date_path = path
                try:
                    children = sorted(path.iterdir(), key=lambda child: child.name.lower())
                except OSError:
                    continue
                work_paths = [child for child in children if child.is_dir() and self._is_work_folder(child)]
            else:
                date_path = path.parent
                date_name = date_path.name
                work_paths = [path]

            for work_path in work_paths:
                key = (date_name, work_path.name)
                if key in seen:
                    continue
                seen.add(key)
                targets.append((date_name, date_path, work_path.name))

        return targets

    def _copy_external_to_local_subprocess(self, source: Path, destination: Path) -> None:
        # Same strategy as ReportTab: avoid pathlib probes on the external path.
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(parents=True, exist_ok=True)
        source_text = str(source).rstrip("\\/")
        list_files = self._run_cmd_args("dir", "/b", "/s", "/a:-d", source_text)
        if list_files.returncode == 0:
            copied_any = False
            for raw_line in list_files.stdout.splitlines():
                source_file_text = raw_line.strip()
                if not source_file_text:
                    continue
                relative_text = self._relative_text(source_file_text, source_text)
                if not relative_text:
                    continue
                destination_file = destination / relative_text
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                copy_result = self._run_cmd_args("copy", "/Y", source_file_text, str(destination_file))
                if copy_result.returncode != 0:
                    raise RuntimeError(
                        "Source path not found or access denied.\n\n"
                        f"Source: {source_file_text}\n"
                        f"copy: {self._short_process_output(copy_result)}"
                    )
                copied_any = True
            if copied_any:
                return

        file_destination = destination / Path(str(source)).name
        file_copy = self._run_cmd_args("copy", "/Y", str(source), str(file_destination))
        if file_copy.returncode != 0:
            raise RuntimeError(
                "Source path not found or access denied.\n\n"
                f"Source: {source}\n"
                f"dir: {self._short_process_output(list_files)}\n"
                f"copy: {self._short_process_output(file_copy)}"
            )

    def _copy_local_file_to_external_subprocess(self, source: Path, destination: Path) -> None:
        source = Path(source)
        destination = Path(destination)
        self._run_cmd_args("mkdir", str(destination.parent))
        completed = self._run_cmd_args("copy", "/Y", str(source), str(destination))
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to copy Workflow output back to the selected folder.\n\n"
                f"Destination: {destination}\n"
                f"copy: {self._short_process_output(completed)}"
            )

    def _copy_external_workflow_outputs_back(self) -> None:
        if not self._external_stage_map:
            return
        for staged_root, original_root in self._external_stage_map:
            if not staged_root.exists():
                continue
            for path in staged_root.rglob("*"):
                if not path.is_file():
                    continue
                relative = path.relative_to(staged_root)
                parent_names = [part.lower() for part in relative.parts[:-1]]
                in_output_folder = any(
                    name == "_pass"
                    or name == "_mask"
                    or name.endswith("_pass")
                    or name.endswith("_mask")
                    or name.endswith("_masks")
                    for name in parent_names
                )
                name_lower = path.name.lower()
                generated_file = (
                    in_output_folder
                    or name_lower == _HOLD_STATE_FILENAME.lower()
                    or (path.suffix.lower() == ".csv" and ("_li-salt_" in name_lower or name_lower.endswith("_result.csv")))
                )
                if not generated_file:
                    continue
                self._copy_local_file_to_external_subprocess(path, original_root / relative)

    def _cleanup_external_stage(self) -> None:
        stage_root = getattr(self, "_external_stage_root", None)
        if stage_root is not None:
            shutil.rmtree(stage_root, ignore_errors=True)
        self._external_stage_root = None
        self._external_stage_map = []
        self._external_requested_faces_by_stage.clear()

    def _local_path_has_content(self, path: Path) -> bool:
        if path.is_file():
            return True
        if not path.is_dir():
            return False
        try:
            return any(path.rglob("*"))
        except OSError:
            return False

    def _run_cmd_args(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["cmd.exe", "/d", "/c", *args],
            capture_output=True,
            text=True,
        )

    def _relative_text(self, child: str, parent: str) -> str:
        normalized_child = child.replace("/", "\\")
        normalized_parent = parent.replace("/", "\\").rstrip("\\")
        if normalized_child.lower().startswith(normalized_parent.lower() + "\\"):
            return normalized_child[len(normalized_parent) + 1 :]
        return Path(normalized_child).name

    def _short_process_output(self, completed: subprocess.CompletedProcess) -> str:
        output = (completed.stderr or completed.stdout or "").strip()
        output = " ".join(output.split())
        if len(output) > 500:
            output = output[:500] + "..."
        return output or f"exit code {completed.returncode}"

    def _refresh_items(self) -> None:
        date_names = self._list_date_folders()
        previous = self._date_var.get()
        self._date_combo["values"] = date_names
        if previous in date_names:
            selected_date = previous
        else:
            today = date.today()
            selected_date = date_names[0] if date_names else ""
            for name in date_names:
                parsed = self._parse_date_folder(name)
                if parsed and parsed.date() == today:
                    selected_date = name
                    break
        self._date_var.set(selected_date)
        self._date_combo.config(
            state=tk.DISABLED
            if self._running or self._mode_var.get() == "By Folder ID"
            else "readonly"
        )

        targets = self._external_targets() if self._external_work_paths else self._scan_targets(selected_date)
        self._items = [
            self._build_item(date_name, date_path, analysis_id)
            for date_name, date_path, analysis_id in targets
        ]
        self._render_tree()
        source_label = "Drag & Drop" if self._external_work_paths else "Date Search"
        self._status_label.config(text=f"{source_label}: {len(self._items)} work item(s)")

    def _refresh_item_status(self, date_name: str, date_path: Path, analysis_id: str) -> None:
        key = f"{date_name}|{analysis_id}"
        updated = self._build_item(date_name, date_path, analysis_id)
        found = False
        for index, item in enumerate(self._items):
            if item.key == key:
                self._items[index] = updated
                found = True
                break
        if not found:
            self._items.append(updated)
        if self._sort_column is not None:
            self._render_tree()
        else:
            self._update_tree_item(updated)
        self._status_label.config(text=f"{len(self._items)} Folder ID(s)")

    def _scan_targets(self, selected_date: str) -> list[tuple[str, Path, str]]:
        targets: list[tuple[str, Path, str]] = []
        date_names = self._list_date_folders()
        if self._mode_var.get() == "By Date":
            date_names = [selected_date] if selected_date else []
        for date_name in date_names:
            date_path = self._root_path / date_name
            if not date_path.is_dir():
                continue
            for path in sorted(date_path.iterdir(), key=lambda p: p.name):
                if self._is_work_folder(path):
                    targets.append((date_name, date_path, path.name))
        if self._mode_var.get() == "By Folder ID":
            targets.sort(key=lambda item: (item[2], item[0]))
        return targets

    def _build_item(self, date_name: str, date_path: Path, analysis_id: str) -> _WorkflowItem:
        analysis_path = date_path / analysis_id
        key = f"{date_name}|{analysis_id}"
        qc_ok, qc_summary = self._qc_complete(date_name, date_path, analysis_id)
        held = self._held_qc_failures.get(key) or self._load_qc_hold_state(
            date_name,
            date_path,
            analysis_id,
        )
        csv_path = self._find_result_csv(analysis_path, analysis_id)
        masks_ok = self._mask_outputs_complete(analysis_id, analysis_path)
        cells = self._analysis_cells(date_name, analysis_id, analysis_path)
        db_ok = self._sqlite_logger.existing_prediction_cells({(analysis_id, str(cell)) for cell in cells})
        db_complete = bool(cells) and len(db_ok) == len(set(cells))
        li_salt = self._li_salt_summary(csv_path)
        correction = self._correction_summary(analysis_id)

        if held:
            failed_count = len(held.get("failures", []))
            ml_status = "WAIT"
            summary = f"QC hold: {failed_count} NG image(s)"
        elif not qc_ok:
            ml_status = "WAIT"
            summary = qc_summary
        elif csv_path is None:
            ml_status = "ML missing"
            summary = "QC ready, ML result missing"
        elif not masks_ok:
            ml_status = "Mask missing"
            summary = "CSV exists, mask output missing"
        elif not db_complete:
            ml_status = "DB missing"
            summary = "CSV exists, SQLite rows missing"
        else:
            ml_status = "DONE"
            summary = "PASS + mask + CSV + DB complete"

        return _WorkflowItem(
            key=key,
            date_name=date_name,
            date_path=date_path,
            analysis_id=analysis_id,
            analysis_path=analysis_path,
            qc_status="HOLD" if held else ("DONE" if qc_ok else "NEEDED"),
            ml_status=ml_status,
            li_salt=li_salt,
            correction=correction,
            summary=summary,
        )

    def _render_tree(self) -> None:
        selected_keys = {
            self._item_by_tree_id[item_id].key
            for item_id in self._tree.selection()
            if item_id in self._item_by_tree_id
        }
        restored_selection: list[str] = []
        for item_id in self._tree.get_children():
            self._tree.delete(item_id)
        self._item_by_tree_id.clear()
        self._sync_tree_headings()
        items = self._sorted_items()
        for item in items:
            tree_id = self._tree.insert(
                "",
                tk.END,
                values=self._tree_values(item),
                tags=(self._tree_tag(item),),
            )
            self._item_by_tree_id[tree_id] = item
            if item.key in selected_keys:
                restored_selection.append(tree_id)
        if restored_selection:
            self._tree.selection_set(restored_selection)
        self._refresh_qc_hold_tree()

    def _update_tree_item(self, item: _WorkflowItem) -> None:
        tree_id = self._tree_id_for_key(item.key)
        if tree_id is None:
            tree_id = self._tree.insert(
                "",
                tk.END,
                values=self._tree_values(item),
                tags=(self._tree_tag(item),),
            )
        else:
            self._tree.item(
                tree_id,
                values=self._tree_values(item),
                tags=(self._tree_tag(item),),
            )
        self._item_by_tree_id[tree_id] = item

    def _on_tree_click(self, event) -> str | None:
        region = self._tree.identify_region(event.x, event.y)
        if region not in {"cell", "tree"}:
            return None
        item_id = self._tree.identify_row(event.y)
        if not item_id:
            return None
        if item_id in self._tree.selection():
            self._tree.selection_remove(item_id)
            self.after_idle(self._show_selected_preview)
            return "break"
        return None
    def _tree_id_for_key(self, key: str) -> str | None:
        for tree_id, item in self._item_by_tree_id.items():
            if item.key == key:
                return tree_id
        return None

    def _tree_values(self, item: _WorkflowItem) -> tuple[str, str, str, str, str, str, str, str]:
        return (
            "Drag & Drop" if self._external_work_paths else "Date Search",
            item.date_name,
            item.analysis_id,
            item.qc_status,
            item.ml_status,
            item.li_salt,
            item.correction,
            item.summary,
        )

    def _tree_tag(self, item: _WorkflowItem) -> str:
        if item.qc_status != "DONE":
            return "red"
        if item.ml_status != "DONE":
            return "yellow"
        return "blue"

    def _sort_by_column(self, column: str) -> None:
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False
        self._render_tree()

    def _sync_tree_headings(self) -> None:
        headings = {
            "source": "Source",
            "date": "Date",
            "analysis_id": "Folder ID",
            "qc": "QC",
            "ml": "ML / DB",
            "li_salt": "Li-salt %",
            "correction": "Correction",
            "summary": "Summary",
        }
        for column, label in headings.items():
            if self._sort_column == column:
                label = f"{label} {'v' if self._sort_reverse else '^'}"
            self._tree.heading(
                column,
                text=label,
                command=lambda col=column: self._sort_by_column(col),
            )

    def _sorted_items(self) -> list[_WorkflowItem]:
        if self._sort_column is None:
            return list(self._items)
        return sorted(
            self._items,
            key=lambda item: self._sort_value(item, self._sort_column or ""),
            reverse=self._sort_reverse,
        )

    def _sort_value(self, item: _WorkflowItem, column: str):
        if column == "source":
            return "Drag & Drop" if self._external_work_paths else "Date Search"
        if column == "date":
            parsed = self._parse_date_folder(item.date_name)
            return parsed or datetime.min
        if column == "analysis_id":
            return item.analysis_id
        if column == "qc":
            return item.qc_status
        if column == "ml":
            return item.ml_status
        if column == "li_salt":
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", item.li_salt)
            return float(match.group(0)) if match else -1.0
        if column == "correction":
            order = {"Mixed": 2, "NO": 1, "-": 0}
            return order.get(item.correction, 0)
        if column == "summary":
            return item.summary
        return ""

    def _correction_summary(self, analysis_id: str) -> str:
        rows = self._sqlite_logger.correction_details(analysis_id)
        if not rows:
            return "-"
        corrected_rows = [
            row for row in rows if int(row.get("correction_applied") or 0) == 1
        ]
        return "Mixed" if corrected_rows else "NO"

    def _qc_complete(self, date_name: str, date_path: Path, analysis_id: str) -> tuple[bool, str]:
        if self._uses_legacy_test_layout(analysis_id):
            analysis_path = date_path / analysis_id
            missing = [
                source_path.name
                for _, source_path in find_test_source_folders(analysis_path)
                if not self._has_pass_png(
                    get_test_output_folders(analysis_path, source_path)[0]
                )
            ]
            return (
                not missing,
                f"TEST PASS missing: {', '.join(missing)}" if missing else "QC ready",
            )
        analysis_path = date_path / analysis_id
        source_folders = self._forced_source_folders(analysis_path)
        if not source_folders:
            return False, "Source images missing"
        missing = []
        for cell_name, face_name, source_path in source_folders:
            pass_path = self._existing_pass_folder_for_source(face_name, source_path)
            if not self._has_pass_png(pass_path):
                missing.append(f"{cell_name} {pass_path.name}")
        if missing:
            return False, f"PASS missing: {', '.join(missing)}"
        return True, "QC ready"


    def _has_pass_png(self, folder_path: Path) -> bool:
        if not folder_path.is_dir():
            return False
        return any(
            path.is_file()
            and path.suffix.lower() == ".png"
            and not path.name.endswith("_mask.png")
            for path in folder_path.iterdir()
        )

    def _analysis_cells(self, date_name: str, analysis_id: str, analysis_path: Path) -> list[int]:
        if self._uses_legacy_test_layout(analysis_id):
            return [
                cell_number
                for cell_number, _ in find_test_source_folders(analysis_path)
            ]
        folder_result = self._folder_qc.validate(
            self._root_path,
            date_name,
            analysis_id,
            customer=self._selected_customer,
        )
        cells: list[int] = []
        if folder_result.status != "PASS":
            return sorted(
                {
                    self._cell_number_from_name(cell_name)
                    for cell_name, _face_name, _source_path in self._forced_source_folders(analysis_path)
                }
            )
        for cell in folder_result.cells:
            cells.append(self._cell_number_from_name(cell.cell_name))
        return cells or [1]

    def _forced_cell_paths(self, analysis_path: Path) -> list[tuple[str, Path]]:
        if not analysis_path.is_dir():
            return []
        cell_dirs = sorted(
            [
                path
                for path in analysis_path.iterdir()
                if path.is_dir() and self._cell_folder_re.match(path.name)
            ],
            key=lambda path: path.name.lower(),
        )
        if cell_dirs:
            return [(path.name, path) for path in cell_dirs]
        return [("single", analysis_path)]

    def _forced_source_folders(self, analysis_path: Path) -> list[tuple[str, str, Path]]:
        sources: list[tuple[str, str, Path]] = []
        for cell_name, cell_path in self._forced_cell_paths(analysis_path):
            if not cell_path.is_dir():
                continue
            face_paths: dict[str, Path] = {}
            for path in cell_path.iterdir():
                if path.is_dir() and path.name.upper() in {"FRONT", "BACK"}:
                    face_paths[path.name.upper()] = path
            for face_name in ("FRONT", "BACK"):
                folder_path = face_paths.get(face_name)
                if folder_path is not None:
                    sources.append((cell_name, face_name, folder_path))
            if not face_paths and self._source_images(cell_path):
                sources.append((cell_name, "SOURCE", cell_path))
        return sources

    def _cell_source_folders(self, cell) -> list[tuple[str, Path]]:
        sources: list[tuple[str, Path]] = []
        if getattr(cell, "source_path", None) is not None:
            sources.append(("SOURCE", cell.source_path))
        if cell.front_path is not None:
            sources.append(("FRONT", cell.front_path))
        if cell.back_path is not None:
            sources.append(("BACK", cell.back_path))
        return sources

    def _pass_folder_for_source(self, face_name: str, source_path: Path) -> Path:
        if face_name == "SOURCE":
            return source_path / "_pass"
        return source_path.parent / f"{face_name}_pass"

    def _pass_folder_candidates(self, face_name: str, source_path: Path) -> list[Path]:
        primary = self._pass_folder_for_source(face_name, source_path)
        if face_name != "SOURCE":
            return [primary]
        return [primary, source_path.with_name(f"{source_path.name}_pass")]

    def _existing_pass_folder_for_source(self, face_name: str, source_path: Path) -> Path:
        candidates = self._pass_folder_candidates(face_name, source_path)
        for candidate in candidates:
            if self._has_pass_png(candidate):
                return candidate
        return candidates[0]

    def _mask_folder_for_source(self, face_name: str, pass_path: Path) -> Path:
        if face_name == "SOURCE":
            if pass_path.name.lower() == "_pass":
                return pass_path.parent / "_mask"
            if pass_path.name.lower().endswith("_pass"):
                return pass_path.with_name(f"{pass_path.name[:-5]}_mask")
            return pass_path.parent / "_mask"
        return pass_path.with_name(f"{face_name}_mask")

    def _mask_outputs_complete(self, _analysis_id: str, analysis_path: Path) -> bool:
        source_folders = self._forced_source_folders(analysis_path)
        pass_images: list[tuple[str, Path]] = []
        for _cell_name, face_name, source_path in source_folders:
            pass_path = self._existing_pass_folder_for_source(face_name, source_path)
            pass_images.extend((face_name, path) for path in self._pass_pngs(pass_path))
        if not pass_images:
            return False
        for face_name, pass_image in pass_images:
            mask_dir = self._mask_folder_for_source(face_name, pass_image.parent)
            preferred = mask_dir / f"{pass_image.stem}_mask.png"
            if preferred.exists():
                continue
            if not mask_dir.is_dir() or not any(mask_dir.glob(f"{pass_image.stem}*_mask.png")):
                return False
        return True

    def _find_result_csv(self, analysis_path: Path, analysis_id: str) -> Path | None:
        old_path = analysis_path / _RESULT_PATTERN_OLD.format(analysis_id=analysis_id)
        candidates = [old_path] if old_path.exists() else []
        candidates.extend(analysis_path.glob(_RESULT_PATTERN_NEW.format(analysis_id=analysis_id)))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _li_salt_summary(self, csv_path: Path | None) -> str:
        if csv_path is None:
            return "-"
        match = re.search(r"_LI-SALT_([0-9.]+)%", csv_path.name)
        if match:
            return f"{match.group(1)}%"
        try:
            frame = pd.read_csv(csv_path)
            if "li_salt_ratio" not in frame.columns:
                return "-"
            value = pd.to_numeric(frame["li_salt_ratio"], errors="coerce").dropna().mean()
            if pd.isna(value):
                return "-"
            if value <= 1.0:
                value *= 100.0
            return f"{value:.2f}%"
        except Exception:
            logger.exception("Failed to summarize result CSV: %s", csv_path)
            return "-"

    def _explicit_selected_items(self) -> list[_WorkflowItem]:
        return [
            self._item_by_tree_id[item_id]
            for item_id in self._tree.selection()
            if item_id in self._item_by_tree_id
        ]

    def _selected_items(self) -> list[_WorkflowItem]:
        return self._explicit_selected_items() or list(self._items)

    def _auto_items(self) -> list[_WorkflowItem]:
        return [
            item
            for item in self._selected_items()
            if item.qc_status != "DONE" or item.ml_status != "DONE"
        ]

    def _run_requested(self) -> None:
        run_qc = bool(self._run_qc_var.get())
        run_ml = bool(self._run_ml_var.get())
        if getattr(self, "_external_direct_crop_ml", False):
            # Drag & Drop is an operator-directed quick path: bypass folder/QC
            # gating and always run Crop -> ML on the dropped source.
            run_qc = True
            run_ml = True
        if not run_qc and not run_ml:
            messagebox.showinfo("Workflow", "Select QC and/or ML before running.")
            self._status_label.config(text="Select QC and/or ML")
            return
        items = self._auto_items() if run_qc and run_ml else self._selected_items()
        if not items:
            self._status_label.config(text="Selected/all work items are already complete")
            return
        self._start_run(
            items,
            force_qc=bool(self._force_run_var.get()),
            run_qc=run_qc,
            run_ml=run_ml,
        )

    def _on_customer_change(self, _event=None) -> None:
        selected = self._customer_var.get().strip().upper()
        self._selected_customer = selected or "HKMC"
        self._refresh_items()

    def _start_run(
        self,
        items: list[_WorkflowItem],
        force_qc: bool,
        *,
        run_qc: bool,
        run_ml: bool,
    ) -> None:
        if self._running:
            return
        if not run_qc and not run_ml:
            messagebox.showinfo("Workflow", "Select QC and/or ML before running.")
            return
        if not items:
            messagebox.showinfo("Workflow", "No work items to process.")
            return
        if run_ml and not self._model_var.get().strip():
            messagebox.showwarning("Workflow", "Select a valid model before running ML.")
            return

        duplicate_items = [
            item
            for item in items
            if run_ml and (run_qc or force_qc or item.qc_status == "DONE")
        ]
        duplicate_pairs = self._existing_pairs_for_items(duplicate_items) if duplicate_items else set()
        skip_pairs: set[tuple[str, str]] = set()
        if duplicate_pairs:
            preview = ", ".join(f"{a} / Cell {c}" for a, c in sorted(duplicate_pairs)[:8])
            if len(duplicate_pairs) > 8:
                preview += ", ..."
            replace = messagebox.askyesno(
                "SQLite Monitoring Duplicate",
                "Monitoring DB already has rows for these Folder ID + Cell pairs:\n\n"
                f"{preview}\n\nReplace existing SQLite rows?",
            )
            if replace:
                self._sqlite_logger.delete_prediction_cells(duplicate_pairs)
            else:
                skip_pairs = duplicate_pairs

        self._running = True
        self._active_run_ml = bool(run_ml and run_qc and not self._external_direct_crop_ml)
        self._preview_enabled = bool(self._preview_var.get())
        self._stop_event.clear()
        self._set_controls_enabled(False)
        self._progress.configure(maximum=max(len(items), 1), value=0)
        self._progress_label.config(
            text=("Progress: 0/0 images" if run_ml else f"Progress: 0/{len(items)} Folder ID(s)")
        )
        self._progress_mode = "image" if run_ml else "item"
        self._worker = threading.Thread(
            target=self._workflow_worker,
            args=(items, skip_pairs, self._preview_enabled, force_qc, run_qc, run_ml),
            daemon=True,
        )
        self._worker.start()

    def _existing_pairs_for_items(self, items: list[_WorkflowItem]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for item in items:
            for cell in self._analysis_cells(item.date_name, item.analysis_id, item.analysis_path):
                pairs.add((item.analysis_id, str(cell)))
        return self._sqlite_logger.existing_prediction_cells(pairs)

    def _request_stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._status_label.config(text="Stopping after current Folder ID...")

    def _workflow_worker(
        self,
        items: list[_WorkflowItem],
        skip_pairs: set[tuple[str, str]],
        preview_enabled: bool,
        force_qc: bool,
        run_qc: bool,
        run_ml: bool,
    ) -> None:
        try:
            progress_state = {"done": 0, "total": 0, "precounted": False}
            if run_ml:
                self._load_ml_config()
                if all(force_qc or item.qc_status == "DONE" for item in items):
                    progress_state["total"] = sum(
                        len(self._build_ml_tasks(item, force_qc=force_qc))
                        for item in items
                    )
                    progress_state["precounted"] = True
                    self._queue.put(("image_total", progress_state["total"]))
            processed = 0
            for item in items:
                if self._stop_event.is_set():
                    break
                self._cleanup_runtime_artifacts(item.analysis_path)
                refreshed = item
                qc_run_succeeded = True
                direct_external = self._external_direct_crop_ml and bool(self._external_stage_map)
                if direct_external:
                    self._queue.put(("current", f"{item.analysis_id}: Direct Crop"))
                    self._reset_qc_hold_for_rerun(item)
                    qc_run_succeeded = self._run_external_direct_crop_for_item(item)
                    refreshed = item
                else:
                    should_run_qc = run_qc and (
                        not run_ml or force_qc or item.qc_status != "DONE"
                    )
                    if should_run_qc:
                        self._queue.put(("current", f"{item.analysis_id}: QC"))
                        self._reset_qc_hold_for_rerun(item)
                        qc_run_succeeded = self._run_qc_for_item(
                            item,
                            force_folder_qc=force_qc,
                        )
                        refreshed = self._build_item(item.date_name, item.date_path, item.analysis_id)
                        self._queue.put(("refresh", (item.date_name, item.date_path, item.analysis_id)))
                if run_ml and qc_run_succeeded and (direct_external or force_qc or refreshed.qc_status == "DONE"):
                    self._queue.put(("current", f"{item.analysis_id}: ML"))
                    self._run_ml_for_item(
                        refreshed,
                        skip_pairs,
                        preview_enabled,
                        force_qc=(force_qc or direct_external),
                        progress_state=progress_state,
                    )
                elif run_ml:
                    logger.warning(
                        "ML skipped because QC is not ready: %s / %s",
                        item.analysis_id,
                        refreshed.summary,
                    )
                processed += 1
                self._queue.put(("progress", (processed, item.date_name, item.date_path, item.analysis_id)))
            if self._external_stage_map:
                self._copy_external_workflow_outputs_back()
            self._queue.put(("done", None))
        except Exception as exc:
            logger.exception("Workflow worker error")
            if self._external_stage_map:
                try:
                    self._copy_external_workflow_outputs_back()
                except Exception:
                    logger.exception("Failed to copy partial Workflow outputs back")
            self._queue.put(("error", str(exc)))

    def _run_external_direct_crop_for_item(self, item: _WorkflowItem) -> bool:
        """Crop staged Drag & Drop sources without FolderQC/QC-hold gating."""
        image_postprocess = self._color_normalizer.normalize if self._color_normalizer is not None else None
        customer_code = self._customer_code_for_analysis(item.analysis_id)
        requested_faces = self._external_requested_faces_by_stage.get(
            str(item.analysis_path).lower(),
            set(),
        )
        source_folders = self._forced_source_folders(item.analysis_path)
        if requested_faces:
            source_folders = [
                source
                for source in source_folders
                if source[1].upper() in requested_faces
            ]
        if not source_folders:
            raise RuntimeError(
                f"No FRONT/BACK images found in dropped item: {item.analysis_id}"
            )

        cropped_count = 0
        for _cell_name, face_name, folder_path in source_folders:
            output_folder = self._pass_folder_for_source(face_name, folder_path)
            self._clear_pass_outputs(output_folder)
            for image_path in self._source_images(folder_path):
                output_path = output_folder / f"{image_path.stem}.png"
                self._crop_engine.process_image(
                    image_path=image_path,
                    customer_code=customer_code,
                    output_path=output_path,
                    skip_size_validation=True,
                    image_postprocess=image_postprocess,
                    force_save_failed=True,
                )
                if output_path.exists():
                    cropped_count += 1

        if cropped_count <= 0:
            raise RuntimeError(
                f"Crop produced no PASS images for dropped item: {item.analysis_id}"
            )
        self._clear_qc_hold(item)
        return True

    def _run_qc_for_item(self, item: _WorkflowItem, force_folder_qc: bool = False) -> bool:
        if self._uses_legacy_test_layout(item.analysis_id):
            ensure_test_output_folders(item.analysis_path)
        folder_result = self._folder_qc.validate(
            self._root_path,
            item.date_name,
            item.analysis_id,
            force=force_folder_qc,
            customer=self._selected_customer,
        )
        if force_folder_qc and folder_result.warnings:
            logger.warning(
                "FolderQC force warnings for %s: %s",
                item.analysis_id,
                " | ".join(folder_result.warnings),
            )
        if folder_result.status != "PASS":
            return False
        image_postprocess = self._color_normalizer.normalize if self._color_normalizer is not None else None
        customer_code = self._customer_code_for_analysis(item.analysis_id)
        failed_items: list[dict[str, object]] = []
        if self._uses_legacy_test_layout(item.analysis_id):
            test_sources = [
                (index, cell.front_path)
                for index, cell in enumerate(folder_result.cells, start=1)
                if cell.front_path is not None
            ]
            for _cell, source_folder in test_sources:
                if source_folder is None:
                    continue
                output_folder, mask_folder = get_test_output_folders(
                    item.analysis_path,
                    source_folder,
                )
                output_folder.mkdir(parents=True, exist_ok=True)
                mask_folder.mkdir(parents=True, exist_ok=True)
                self._clear_pass_outputs(output_folder)
                for image_path in find_test_images(source_folder, _IMAGE_EXTENSIONS):
                    output_path = output_folder / build_test_output_name(
                        source_folder,
                        image_path,
                    )
                    result = self._crop_engine.process_image(
                        image_path,
                        customer_code,
                        output_path,
                        skip_size_validation=True,
                        image_postprocess=image_postprocess,
                    )
                    if result.status == "FAIL":
                        failed_items.append(
                            self._qc_failure_item(
                                result,
                                customer_code=customer_code,
                                output_path=output_path,
                                skip_size_validation=True,
                                image_postprocess=image_postprocess,
                            )
                        )
            self._handle_qc_failures(item, failed_items)
            return not failed_items

        source_folders = [
            (face_name, folder_path)
            for cell in folder_result.cells
            for face_name, folder_path in self._cell_source_folders(cell)
        ]

        for face_name, folder_path in source_folders:
            output_folder = self._pass_folder_for_source(face_name, folder_path)
            self._clear_pass_outputs(output_folder)
            crop_results = self._crop_engine.process_folder(
                folder_path,
                customer_code,
                output_folder,
                image_postprocess=image_postprocess,
            )
            for result in crop_results:
                if result.status == "FAIL":
                    failed_items.append(
                        self._qc_failure_item(
                            result,
                            customer_code=customer_code,
                            output_path=output_folder / f"{result.image_path.stem}.png",
                            skip_size_validation=False,
                            image_postprocess=image_postprocess,
                        )
                    )
        self._handle_qc_failures(item, failed_items)
        return not failed_items

    def _qc_failure_item(
        self,
        result,
        *,
        customer_code: str,
        output_path: Path,
        skip_size_validation: bool,
        image_postprocess,
    ) -> dict[str, object]:
        return {
            "id": str(result.image_path),
            "image_path": result.image_path,
            "output_path": output_path,
            "fail_reasons": list(result.fail_reasons),
            "customer_code": customer_code,
            "skip_size_validation": skip_size_validation,
            "image_postprocess": image_postprocess,
        }

    def _handle_qc_failures(self, item: _WorkflowItem, failed_items: list[dict[str, object]]) -> None:
        if not failed_items:
            self._clear_qc_hold(item)
            return
        if self._stop_event.is_set():
            return
        held = {
            "key": item.key,
            "date_name": item.date_name,
            "date_path": item.date_path,
            "analysis_id": item.analysis_id,
            "analysis_path": item.analysis_path,
            "failures": failed_items,
            "resume_ml": bool(self._active_run_ml),
        }
        self._held_qc_failures[item.key] = held
        self._persist_qc_hold_state(held)
        self._queue.put(
            (
                "qc_hold",
                held,
            )
        )

    def _load_qc_hold_state(
        self,
        date_name: str,
        date_path: Path,
        analysis_id: str,
    ) -> dict[str, object] | None:
        analysis_path = date_path / analysis_id
        hold_path = self._hold_state_path(analysis_path)
        if not hold_path.exists():
            return None
        try:
            raw = json.loads(hold_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load QC hold state: %s", hold_path, exc_info=True)
            return None

        key = f"{date_name}|{analysis_id}"
        failures: list[dict[str, object]] = []
        for failure in raw.get("failures", []):
            image_path = Path(failure.get("image_path", ""))
            output_path = Path(failure.get("output_path", ""))
            if not image_path.exists():
                continue
            failures.append(
                {
                    "id": str(failure.get("id") or image_path),
                    "image_path": image_path,
                    "output_path": output_path,
                    "fail_reasons": list(failure.get("fail_reasons", [])),
                    "customer_code": str(
                        failure.get("customer_code")
                        or self._customer_code_for_analysis(analysis_id)
                    ),
                    "skip_size_validation": bool(failure.get("skip_size_validation", False)),
                    "image_postprocess": self._current_image_postprocess(),
                }
            )
        if not failures:
            self._delete_qc_hold_state({"analysis_path": analysis_path})
            return None

        held = {
            "key": key,
            "date_name": date_name,
            "date_path": date_path,
            "analysis_id": analysis_id,
            "analysis_path": analysis_path,
            "failures": failures,
            "resume_ml": bool(raw.get("resume_ml", False)),
        }
        self._held_qc_failures[key] = held
        return held

    def _persist_qc_hold_state(self, held: dict[str, object]) -> None:
        analysis_path = Path(held.get("analysis_path"))
        failures = []
        for failure in held.get("failures", []):
            failures.append(
                {
                    "id": str(failure.get("id", "")),
                    "image_path": str(failure.get("image_path", "")),
                    "output_path": str(failure.get("output_path", "")),
                    "fail_reasons": list(failure.get("fail_reasons", [])),
                    "customer_code": str(failure.get("customer_code", "")),
                    "skip_size_validation": bool(failure.get("skip_size_validation", False)),
                }
            )
        payload = {
            "date_name": str(held.get("date_name", "")),
            "analysis_id": str(held.get("analysis_id", "")),
            "resume_ml": bool(held.get("resume_ml", False)),
            "failures": failures,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            analysis_path.mkdir(parents=True, exist_ok=True)
            self._hold_state_path(analysis_path).write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        except Exception:
            logger.warning("Failed to persist QC hold state: %s", analysis_path, exc_info=True)

    def _delete_qc_hold_state(self, held: dict[str, object]) -> None:
        try:
            hold_path = self._hold_state_path(Path(held.get("analysis_path")))
            if hold_path.exists():
                hold_path.unlink()
        except Exception:
            logger.warning("Failed to delete QC hold state", exc_info=True)

    def _clear_qc_hold(self, item: _WorkflowItem) -> None:
        self._held_qc_failures.pop(item.key, None)
        self._delete_qc_hold_state({"analysis_path": item.analysis_path})

    def _reset_qc_hold_for_rerun(self, item: _WorkflowItem) -> None:
        self._clear_qc_hold(item)

    def _hold_state_path(self, analysis_path: Path) -> Path:
        return analysis_path / _HOLD_STATE_FILENAME

    def _current_image_postprocess(self):
        return self._color_normalizer.normalize if self._color_normalizer is not None else None

    def _process_failed_qc_item(self, failed_item: dict[str, object], *, force: bool):
        image_path = Path(failed_item["image_path"])
        output_path = Path(failed_item["output_path"])
        result = self._crop_engine.process_image(
            image_path=image_path,
            customer_code=str(failed_item["customer_code"]),
            output_path=output_path,
            skip_size_validation=bool(failed_item["skip_size_validation"]),
            image_postprocess=failed_item.get("image_postprocess"),
            force_save_failed=force,
        )
        if result.status != "PASS":
            logger.warning(
                "QC NG action failed: %s / %s",
                image_path,
                " | ".join(result.fail_reasons),
            )
        return result

    def _clear_pass_outputs(self, output_folder: Path) -> None:
        if not output_folder.is_dir():
            return
        for path in output_folder.iterdir():
            if path.is_file() and path.suffix.lower() == ".png":
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Failed to remove stale pass output: %s", path, exc_info=True)

    def _cleanup_runtime_artifacts(self, analysis_path: Path) -> None:
        if not analysis_path.is_dir():
            return
        for temp_path in analysis_path.rglob("_CROPPED_TEMP"):
            if temp_path.is_dir():
                try:
                    shutil.rmtree(temp_path)
                    logger.info("Removed runtime temp folder: %s", temp_path)
                except OSError:
                    logger.warning("Failed to remove runtime temp folder: %s", temp_path, exc_info=True)

    def _load_ml_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        selected_model = self._model_var.get().strip()
        model_dir = resolve_model_path(selected_model or cfg["ml"]["model_path"])
        self._save_mask_images = bool(cfg["ml"].get("save_mask_images", True))
        normalization_cfg = dict(cfg.get("normalization", {}) or {})
        self._normalization_cfg = normalization_cfg
        self._normalization_enabled = bool(normalization_cfg.get("enabled", False))
        self._low_confidence_threshold = float(
            normalization_cfg.get("low_confidence_threshold", 0.60)
        )
        self._drift_detector = DriftDetector(
            model_dir / "baseline_stats.json",
            min_high_conf_bg_ratio=float(
                normalization_cfg.get("min_high_conf_bg_ratio", 0.05)
            ),
            drift_threshold=float(normalization_cfg.get("drift_threshold", 2.0)),
        )
        self._background_normalizer = BackgroundLABNormalizer(
            max_l_shift=float(normalization_cfg.get("max_l_shift", 25)),
            max_a_shift=float(normalization_cfg.get("max_a_shift", 5)),
            max_b_shift=float(normalization_cfg.get("max_b_shift", 5)),
        )
        if self._inference is None or self._inference.model_dir != model_dir:
            self._inference = InferenceEngine(self._config_path, model_path=model_dir)

    def _run_ml_for_item(
        self,
        item: _WorkflowItem,
        skip_pairs: set[tuple[str, str]],
        preview_enabled: bool,
        force_qc: bool = False,
        progress_state: dict[str, int | bool] | None = None,
    ) -> None:
        qc_ok, qc_summary = self._qc_complete(item.date_name, item.date_path, item.analysis_id)
        if not qc_ok and not force_qc:
            logger.warning("ML skipped because QC pass outputs are incomplete: %s / %s", item.analysis_id, qc_summary)
            self._queue.put(("refresh", (item.date_name, item.date_path, item.analysis_id)))
            return
        tasks = self._build_ml_tasks(item, force_qc=force_qc)
        if progress_state is not None and not progress_state.get("precounted"):
            progress_state["total"] = int(progress_state.get("total", 0)) + len(tasks)
            self._queue.put(("image_total", int(progress_state["total"])))
        if self._save_mask_images:
            self._clear_mask_outputs_for_tasks(tasks)
        rows: list[dict] = []
        monitoring_rows: list[dict] = []
        for task in tasks:
            if self._stop_event.is_set():
                return
            task_label = self._task_label(task)
            self._queue.put(("current", task_label))
            image_started = time.perf_counter()
            source_image = quiet_imread(task.image_path, cv2.IMREAD_UNCHANGED)
            if source_image is None:
                self._advance_image_progress(progress_state, task_label)
                continue
            read_elapsed = time.perf_counter() - image_started
            image_stats = self._image_statistics(source_image)
            stats_elapsed = time.perf_counter() - image_started - read_elapsed
            label_mask, confidence_mask = self._predict_image(source_image)
            predict_total_elapsed = time.perf_counter() - image_started - read_elapsed - stats_elapsed
            monitoring_info = self._monitoring_info(
                source_image=source_image,
                label_mask=label_mask,
                confidence_mask=confidence_mask,
            )
            corrected_preview_image: np.ndarray | None = None
            if self._should_apply_normalization(monitoring_info):
                normalized = self._background_normalizer.normalize(
                    source_image,
                    l_shift=monitoring_info["l_shift"],
                    a_shift=monitoring_info["a_shift"],
                    b_shift=monitoring_info["b_shift"],
                )
                if normalized.correction_applied:
                    del label_mask
                    del confidence_mask
                    label_mask, confidence_mask = self._predict_image(normalized.image)
                    monitoring_info["correction_applied"] = 1
                    monitoring_info["normalization_skipped_reason"] = None
                    monitoring_info["l_shift"] = normalized.l_shift
                    monitoring_info["a_shift"] = normalized.a_shift
                    monitoring_info["b_shift"] = normalized.b_shift
                    monitoring_info["drift_score_after"] = self._drift_score_after(
                        normalized.image,
                        label_mask,
                        confidence_mask,
                    )
                    corrected_preview_image = normalized.image
            result_row = self._quantifier.quantify(
                label_mask,
                analysis_id=task.analysis_id,
                cell=task.cell,
                face=task.face,
                image_num=task.image_num,
            )
            result_row["correction_applied"] = int(monitoring_info.get("correction_applied") or 0)
            rows.append(result_row)
            mask_path = None
            mask_elapsed = 0.0
            if self._save_mask_images:
                mask_dir = self._mask_dir_for_task(task)
                mask_path = mask_dir / f"{task.image_path.stem}_mask.png"
                mask_started = time.perf_counter()
                self._quantifier.save_mask_image(label_mask, mask_path)
                mask_elapsed = time.perf_counter() - mask_started
                if mask_elapsed >= 0.25 or logger.isEnabledFor(logging.DEBUG):
                    logger.info(
                        "Workflow mask write timing: %s path=%s %.3fs",
                        task_label,
                        mask_path,
                        mask_elapsed,
                    )
            total_elapsed = time.perf_counter() - image_started
            if total_elapsed >= 5.0 or mask_elapsed >= 0.25 or logger.isEnabledFor(logging.DEBUG):
                logger.info(
                    "Workflow image timing: %s read=%.2fs stats=%.2fs predict_total=%.2fs "
                    "mask_write=%.2fs total=%.2fs save_mask=%s",
                    task_label,
                    read_elapsed,
                    stats_elapsed,
                    predict_total_elapsed,
                    mask_elapsed,
                    total_elapsed,
                    self._save_mask_images,
                )
            monitoring_rows.append(
                self._prediction_record(
                    task=task,
                    result_row=result_row,
                    image_stats=image_stats,
                    confidence_stats=self._confidence_statistics(confidence_mask),
                    monitoring_info=monitoring_info,
                    mask_path=mask_path,
                )
            )
            if preview_enabled:
                if corrected_preview_image is not None:
                    self._put_preview(
                        (
                            source_image,
                            corrected_preview_image,
                            f"{task_label} / {self._correction_delta_text(monitoring_info)}",
                            "Before color",
                            "After color correction",
                        )
                    )
                else:
                    self._put_preview(
                        (
                            task.image_path,
                            mask_path or task.image_path,
                            task_label,
                            "Before",
                            "After / Mask",
                        )
                    )
            del source_image
            del label_mask
            del confidence_mask
            self._advance_image_progress(progress_state, task_label)
        if not rows:
            return
        csv_path = self._write_csv(item.analysis_id, tasks, rows)
        for record in monitoring_rows:
            pair = (str(record["analysis_id"]), str(record["cell"]))
            if pair in skip_pairs:
                continue
            record["csv_path"] = str(csv_path)
            self._sqlite_logger.insert_prediction_run(record)

    def _advance_image_progress(
        self,
        progress_state: dict[str, int | bool] | None,
        task_label: str,
    ) -> None:
        if progress_state is None:
            return
        progress_state["done"] = int(progress_state.get("done", 0)) + 1
        total = int(progress_state.get("total", 0))
        self._queue.put(("image_progress", (int(progress_state["done"]), total, task_label)))

    def _build_ml_tasks(self, item: _WorkflowItem, force_qc: bool = False) -> list[_ImageTask]:
        if self._uses_legacy_test_layout(item.analysis_id):
            tasks: list[_ImageTask] = []
            for cell, source_folder in find_test_source_folders(item.analysis_path):
                pass_path, _ = get_test_output_folders(item.analysis_path, source_folder)
                tasks.extend(
                    _ImageTask(
                        item.analysis_id,
                        item.analysis_path,
                        cell,
                        "TEST",
                        index,
                        path,
                    )
                    for index, path in enumerate(
                        self._pass_pngs(pass_path),
                        start=1,
                    )
                )
            return tasks
        tasks: list[_ImageTask] = []
        source_folders = self._forced_source_folders(item.analysis_path)

        for cell_name, face_name, source_path in source_folders:
            cell_number = self._cell_number_from_name(cell_name)
            pass_path = self._existing_pass_folder_for_source(face_name, source_path)
            source_index = self._source_layer_index(source_path)
            for fallback_index, path in enumerate(
                self._ml_pass_pngs(source_path, pass_path, item.analysis_id),
                start=1,
            ):
                image_num = source_index.get(path.stem, fallback_index)
                tasks.append(_ImageTask(item.analysis_id, item.analysis_path, cell_number, face_name, image_num, path))
        return tasks

    def _task_label(self, task: _ImageTask) -> str:
        return f"{task.analysis_id} / Cell {task.cell} / {task.face} / Layer {task.image_num}"

    def _source_layer_index(self, folder_path: Path) -> dict[str, int]:
        return {
            path.stem: index
            for index, path in enumerate(self._source_images(folder_path), start=1)
        }

    def _source_images(self, folder_path: Path) -> list[Path]:
        if not folder_path.is_dir():
            return []
        return sorted(
            [
                path for path in folder_path.iterdir()
                if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
            ],
            key=lambda path: path.name,
        )

    def _pass_pngs(self, folder_path: Path) -> list[Path]:
        if not folder_path.is_dir():
            return []
        return sorted(
            [
                path for path in folder_path.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".png"
                and not path.name.endswith("_mask.png")
            ],
            key=lambda path: path.name,
        )

    def _ml_pass_pngs(
        self,
        source_folder: Path,
        pass_folder: Path,
        analysis_id: str,
    ) -> list[Path]:
        pass_paths = self._pass_pngs(pass_folder)
        source_images = self._source_images(source_folder)
        if len(source_images) <= 1:
            return pass_paths
        excluded_stems = {source_images[0].stem, source_images[-1].stem}
        return [path for path in pass_paths if path.stem not in excluded_stems]

    def _predict_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        started = time.perf_counter()
        feature_batch = self._feature_extractor.extract(
            image,
            downsample_factor=self._inference.downsample_factor,
            feature_columns=self._inference.feature_columns,
        )
        extract_elapsed = time.perf_counter() - started
        predict_started = time.perf_counter()
        try:
            label_mask, confidence_mask = self._inference.predict_mask_with_confidence(feature_batch)
            predict_elapsed = time.perf_counter() - predict_started
            total_elapsed = time.perf_counter() - started
            if total_elapsed >= 5.0 or logger.isEnabledFor(logging.DEBUG):
                logger.info(
                    "Workflow ML predict timing: model=%s type=%s family=%s feature_set=%s "
                    "image_shape=%s downsample=%.2f valid_pixels=%d features=%.2fs inference=%.2fs total=%.2fs",
                    self._inference.model_dir,
                    self._inference.model_type,
                    self._inference.model_family,
                    self._inference.feature_set,
                    image.shape[:2],
                    self._inference.downsample_factor,
                    int(feature_batch.features.shape[0]),
                    extract_elapsed,
                    predict_elapsed,
                    total_elapsed,
                )
            return label_mask, confidence_mask
        finally:
            del feature_batch

    def _valid_mask_for_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, 3] > 0
        return np.ones(image.shape[:2], dtype=bool)

    def _bgr_from_image(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3]
        return image

    def _mask_dir_for_task(self, task: _ImageTask) -> Path:
        if self._uses_legacy_test_layout(task.analysis_id) and task.face.upper() == "TEST":
            return get_test_mask_folder_for_pass(task.image_path.parent)
        return self._mask_folder_for_source(task.face, task.image_path.parent)

    def _clear_mask_outputs_for_tasks(self, tasks: list[_ImageTask]) -> None:
        mask_dirs = {self._mask_dir_for_task(task) for task in tasks}
        for mask_dir in mask_dirs:
            if not mask_dir.is_dir():
                continue
            for path in mask_dir.iterdir():
                if path.is_file() and path.suffix.lower() == ".png":
                    try:
                        path.unlink()
                    except OSError:
                        logger.warning("Failed to remove stale mask output: %s", path, exc_info=True)

    def _prediction_record(
        self,
        *,
        task: _ImageTask,
        result_row: dict,
        image_stats: dict[str, float | None],
        confidence_stats: dict[str, float | None],
        monitoring_info: dict[str, float | int | str | None],
        mask_path: Path | None,
    ) -> dict:
        record = {
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
            "analysis_id": task.analysis_id,
            "cell": str(task.cell),
            "face": task.face,
            "image_num": task.image_num,
            "image_name": task.image_path.name,
            "customer": self._customer_for_analysis(task.analysis_id),
            "model_version": self._inference.model_version if self._inference else None,
            "total_pixel": int(result_row["total_pixel"]),
            "background_ratio": float(result_row["background_ratio"]),
            "nonactive_ratio": float(result_row["nonactive_ratio"]),
            "li_salt_ratio": float(result_row["li_salt_ratio"]),
            "mask_path": str(mask_path) if mask_path is not None else None,
            "pass_image_path": str(task.image_path),
        }
        record.update(image_stats)
        record.update(confidence_stats)
        record.update(monitoring_info)
        low_conf = record.get("low_confidence_ratio")
        drift_after = record.get("drift_score_after")
        drift_before = record.get("drift_score_before")
        drift_score = drift_after if drift_after is not None else drift_before
        detector = self._drift_detector
        retraining_low_conf = float(
            self._normalization_cfg.get("retraining_low_confidence_threshold", 0.20)
        )
        if (
            isinstance(drift_score, (int, float))
            and isinstance(low_conf, (int, float))
            and detector is not None
            and drift_score > detector.drift_threshold
            and low_conf > retraining_low_conf
        ):
            record["retraining_candidate"] = 1
        return record

    def _image_statistics(self, image: np.ndarray) -> dict[str, float | None]:
        valid_mask = self._valid_mask_for_image(image)
        if not np.any(valid_mask):
            return {
                "mean_r": None,
                "mean_g": None,
                "mean_b": None,
                "mean_h": None,
                "mean_s": None,
                "mean_v": None,
                "mean_l": None,
                "mean_a": None,
                "mean_b_lab": None,
            }
        bgr = self._bgr_from_image(image)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        bgr_values = bgr[valid_mask].astype(np.float64)
        hsv_values = hsv[valid_mask].astype(np.float64)
        lab_values = lab[valid_mask].astype(np.float64)
        return {
            "mean_r": float(bgr_values[:, 2].mean()),
            "mean_g": float(bgr_values[:, 1].mean()),
            "mean_b": float(bgr_values[:, 0].mean()),
            "mean_h": float(hsv_values[:, 0].mean()),
            "mean_s": float(hsv_values[:, 1].mean()),
            "mean_v": float(hsv_values[:, 2].mean()),
            "mean_l": float(lab_values[:, 0].mean()),
            "mean_a": float(lab_values[:, 1].mean()),
            "mean_b_lab": float(lab_values[:, 2].mean()),
        }

    def _confidence_statistics(self, confidence_mask: np.ndarray | None) -> dict[str, float | None]:
        if confidence_mask is None:
            return {"mean_confidence": None, "low_confidence_ratio": None}
        valid = confidence_mask[np.isfinite(confidence_mask)]
        if valid.size == 0:
            return {"mean_confidence": None, "low_confidence_ratio": None}
        return {
            "mean_confidence": float(valid.mean()),
            "low_confidence_ratio": float(
                np.count_nonzero(valid < self._low_confidence_threshold) / valid.size
            ),
        }

    def _monitoring_info(
        self,
        *,
        source_image: np.ndarray,
        label_mask: np.ndarray,
        confidence_mask: np.ndarray | None,
    ) -> dict[str, float | int | str | None]:
        info: dict[str, float | int | str | None] = {
            "high_conf_bg_ratio": None,
            "raw_bg_l": None,
            "raw_bg_a": None,
            "raw_bg_b_lab": None,
            "baseline_bg_l": None,
            "baseline_bg_a": None,
            "baseline_bg_b_lab": None,
            "l_shift": None,
            "a_shift": None,
            "b_shift": None,
            "drift_score_before": None,
            "drift_score_after": None,
            "correction_applied": 0,
            "normalization_skipped_reason": None,
            "retraining_candidate": 0,
            "qc_warning": 0,
        }
        detector = self._drift_detector
        if detector is None:
            info["normalization_skipped_reason"] = "missing_baseline_stats"
            return info

        valid_mask = self._valid_mask_for_image(source_image)
        lab = cv2.cvtColor(self._bgr_from_image(source_image), cv2.COLOR_BGR2LAB)
        background_mask = self._high_confidence_background_mask(label_mask, confidence_mask)
        drift = detector.evaluate(lab, valid_mask, background_mask)
        self._apply_drift_info(info, drift)
        info["normalization_skipped_reason"] = self._normalization_skip_reason(drift)
        return info

    def _high_confidence_background_mask(
        self,
        label_mask: np.ndarray,
        confidence_mask: np.ndarray | None,
    ) -> np.ndarray | None:
        if confidence_mask is None:
            return None
        threshold = float(self._normalization_cfg.get("bg_confidence_threshold", 0.90))
        return (
            (label_mask == LABEL_BACKGROUND)
            & np.isfinite(confidence_mask)
            & (confidence_mask >= threshold)
        )

    def _apply_drift_info(
        self,
        info: dict[str, float | int | str | None],
        drift: BackgroundDrift,
    ) -> None:
        info.update(
            {
                "high_conf_bg_ratio": drift.high_conf_bg_ratio,
                "raw_bg_l": drift.raw_bg_l,
                "raw_bg_a": drift.raw_bg_a,
                "raw_bg_b_lab": drift.raw_bg_b_lab,
                "baseline_bg_l": drift.baseline_bg_l,
                "baseline_bg_a": drift.baseline_bg_a,
                "baseline_bg_b_lab": drift.baseline_bg_b_lab,
                "l_shift": drift.l_shift,
                "a_shift": drift.a_shift,
                "b_shift": drift.b_shift,
                "drift_score_before": drift.drift_score_before,
            }
        )

    def _normalization_skip_reason(self, drift: BackgroundDrift) -> str | None:
        if drift.skipped_reason is not None:
            return drift.skipped_reason
        score = drift.drift_score_before
        detector = self._drift_detector
        if score is None or detector is None:
            return None
        if score <= detector.drift_threshold:
            return "drift_below_threshold"
        if not self._normalization_enabled:
            return "normalization_disabled"
        return None

    def _should_apply_normalization(self, info: dict[str, float | int | str | None]) -> bool:
        if not self._normalization_enabled or self._background_normalizer is None:
            return False
        return self._would_apply_normalization(info)

    def _would_apply_normalization(self, info: dict[str, float | int | str | None]) -> bool:
        score = info.get("drift_score_before")
        detector = self._drift_detector
        return (
            isinstance(score, (int, float))
            and detector is not None
            and score > detector.drift_threshold
            and info.get("l_shift") is not None
            and info.get("a_shift") is not None
            and info.get("b_shift") is not None
        )

    def _correction_delta_text(self, info: dict[str, float | int | str | None]) -> str:
        l_shift = self._format_signed_float(info.get("l_shift"))
        a_shift = self._format_signed_float(info.get("a_shift"))
        b_shift = self._format_signed_float(info.get("b_shift"))
        before = self._format_optional_float(info.get("drift_score_before"))
        after = self._format_optional_float(info.get("drift_score_after"))
        return f"Color correction LAB dL {l_shift}, da {a_shift}, db {b_shift}, drift {before}->{after}"

    @staticmethod
    def _format_signed_float(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{value:+.2f}"
        return "-"

    @staticmethod
    def _format_optional_float(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return "-"

    def _drift_score_after(
        self,
        image: np.ndarray,
        label_mask: np.ndarray,
        confidence_mask: np.ndarray | None,
    ) -> float | None:
        detector = self._drift_detector
        if detector is None:
            return None
        valid_mask = self._valid_mask_for_image(image)
        lab = cv2.cvtColor(self._bgr_from_image(image), cv2.COLOR_BGR2LAB)
        bg_mask = self._high_confidence_background_mask(label_mask, confidence_mask)
        drift = detector.evaluate(lab, valid_mask, bg_mask)
        return drift.drift_score_before

    def _write_csv(self, analysis_id: str, tasks: list[_ImageTask], rows: list[dict]) -> Path:
        average_percent = self._compute_average_li_salt_percent(tasks, rows)
        csv_path = tasks[0].analysis_path / f"{analysis_id}_LI-SALT_{average_percent:.2f}%.csv"
        pd.DataFrame(
            rows,
            columns=[
                "AnalysisID",
                "Cell",
                "Face",
                "ImageNum",
                "total_pixel",
                "background_ratio",
                "nonactive_ratio",
                "li_salt_ratio",
                "correction_applied",
            ],
        ).to_csv(csv_path, index=False, float_format="%.3f")
        return csv_path

    def _compute_average_li_salt_percent(self, tasks: list[_ImageTask], rows: list[dict]) -> float:
        row_map = {(row["Cell"], row["Face"], row["ImageNum"]): row for row in rows}
        included: list[float] = []
        grouped: dict[tuple[int, str], list[_ImageTask]] = {}
        for task in tasks:
            grouped.setdefault((task.cell, task.face), []).append(task)
        for group_tasks in grouped.values():
            ordered = sorted(group_tasks, key=lambda task: task.image_num)
            for task in ordered:
                row = row_map[(task.cell, task.face, task.image_num)]
                included.append(float(row["li_salt_ratio"]) * 100.0)
        return float(sum(included) / len(included)) if included else 0.0

    def _cell_number_from_name(self, cell_name: str) -> int:
        if cell_name.lower() == "single":
            return 1
        match = re.search(r"(\d+)", cell_name)
        return int(match.group(1)) if match else 1

    def _customer_code_for_analysis(self, analysis_id: str) -> str:
        code = self._folder_qc.customer_code_for_name(self._selected_customer)
        if code is not None:
            return code
        if self._selected_customer == "ETC":
            return "ETC"
        return "T" if is_test_analysis_id(analysis_id) else (analysis_id[1] if len(analysis_id) > 1 else "")

    def _customer_for_analysis(self, analysis_id: str) -> str:
        if self._selected_customer:
            return self._selected_customer
        if is_test_analysis_id(analysis_id):
            return "TEST"
        code = analysis_id[1] if len(analysis_id) > 1 else ""
        return str(self._customer_codes.get(code, code))

    def _uses_legacy_test_layout(self, analysis_id: str) -> bool:
        return is_test_analysis_id(analysis_id) and not self._selected_customer

    def _toggle_preview(self) -> None:
        self._preview_enabled = bool(self._preview_var.get())
        if self._preview_enabled:
            self._review_label.config(text="Select a Folder ID to preview.")
        else:
            self._review_entries = []
            self._review_index = 0
            self._review_label.config(text="Preview disabled.")
            self._review_count_label.config(text="")
            self._before_canvas.delete("all")
            self._after_canvas.delete("all")
        self._sync_review_nav_state()

    def _put_preview(self, payload) -> None:
        if self._queue.qsize() > 4:
            return
        self._queue.put(("preview", payload))

    def _poll_queue(self) -> None:
        pending_preview = None
        try:
            while True:
                msg_type, payload = self._queue.get_nowait()
                if msg_type == "preview":
                    pending_preview = payload
                else:
                    self._dispatch(msg_type, payload)
        except queue.Empty:
            pass
        if pending_preview is not None:
            self._dispatch("preview", pending_preview)
        self.after(100, self._poll_queue)

    def _dispatch(self, msg_type: str, payload) -> None:
        if msg_type == "current":
            self._current_label.config(text=f"Current: {payload}")
        elif msg_type == "image_total":
            total = int(payload)
            try:
                done = int(float(self._progress.cget("value")))
            except (tk.TclError, ValueError):
                done = 0
            self._progress.configure(maximum=max(total, 1), value=min(done, max(total, 1)))
            self._progress_label.config(text=f"Progress: {done}/{total} images")
        elif msg_type == "image_progress":
            done, total, label = payload
            done = int(done)
            total = int(total)
            self._progress.configure(maximum=max(total, 1), value=min(done, max(total, 1)))
            self._progress_label.config(text=f"Progress: {done}/{total} images")
            self._current_label.config(text=f"Current: {label}")
        elif msg_type == "progress":
            if isinstance(payload, tuple):
                processed, date_name, date_path, analysis_id = payload
                if getattr(self, "_progress_mode", "item") == "item":
                    self._progress.configure(value=processed)
                    try:
                        maximum = int(float(self._progress.cget("maximum")))
                    except (tk.TclError, ValueError):
                        maximum = processed
                    self._progress_label.config(text=f"Progress: {processed}/{maximum} Folder ID(s)")
                self._refresh_item_status(str(date_name), Path(date_path), str(analysis_id))
            else:
                self._progress.configure(value=payload)
                self._refresh_items()
        elif msg_type == "refresh":
            if isinstance(payload, tuple):
                date_name, date_path, analysis_id = payload
                self._refresh_item_status(str(date_name), Path(date_path), str(analysis_id))
            else:
                self._refresh_items()
        elif msg_type == "qc_hold":
            self._handle_qc_hold(payload)
        elif msg_type == "qc_ng_action_done":
            self._handle_qc_ng_action_done(payload)
        elif msg_type == "qc_ng_bulk_done":
            self._handle_qc_ng_bulk_done(payload)
        elif msg_type == "preview":
            if not self._preview_enabled:
                return
            self._review_entries = []
            self._review_index = 0
            self._review_count_label.config(text="")
            self._sync_review_nav_state()
            before, after, label, before_title, after_title = self._normalize_preview_payload(payload)
            self._review_label.config(text=label)
            self._draw_image(self._before_canvas, before, before_title)
            self._draw_image(self._after_canvas, after, after_title)
        elif msg_type == "done":
            self._running = False
            self._active_run_ml = False
            self._set_controls_enabled(True)
            self._current_label.config(text="Current: -")
            self._refresh_items()
            self._show_selected_preview()
            self._status_label.config(text="Workflow complete")
            if self._pending_hold_ml:
                self.after(50, self._resume_next_pending_hold_ml)
        elif msg_type == "error":
            self._running = False
            self._active_run_ml = False
            self._set_controls_enabled(True)
            messagebox.showerror("Workflow Error", payload)
            self._refresh_items()
            self._status_label.config(text=f"Error: {payload}")

    def _handle_qc_hold(self, payload: dict) -> None:
        key = str(payload.get("key", ""))
        if not key:
            return
        failures = list(payload.get("failures", []))
        held = {
            "key": key,
            "date_name": payload.get("date_name"),
            "date_path": payload.get("date_path"),
            "analysis_id": payload.get("analysis_id"),
            "analysis_path": payload.get("analysis_path"),
            "failures": failures,
            "resume_ml": bool(payload.get("resume_ml", False)),
        }
        self._held_qc_failures[key] = held
        self._persist_qc_hold_state(held)
        analysis_id = payload.get("analysis_id", "-")
        self._status_label.config(
            text=f"{analysis_id}: QC hold ({len(failures)} NG image(s))"
        )
        self._refresh_item_status(
            str(payload.get("date_name")),
            Path(payload.get("date_path")),
            str(payload.get("analysis_id")),
        )
        self._refresh_qc_hold_tree()
        if not self._running:
            self._show_held_qc_preview(key)

    def _handle_qc_ng_action_done(self, payload: dict) -> None:
        held_key = str(payload.get("held_key", ""))
        failure_id = str(payload.get("failure_id", ""))
        status = str(payload.get("status", "FAIL"))
        fail_reasons = list(payload.get("fail_reasons", []))
        held = self._held_qc_failures.get(held_key)
        if not held:
            return
        failures = list(held.get("failures", []))
        if status == "PASS":
            failures = [failure for failure in failures if str(failure.get("id")) != failure_id]
            held["failures"] = failures
            self._status_label.config(text=f"{held.get('analysis_id', '-')}: NG resolved")
        else:
            for failure in failures:
                if str(failure.get("id")) == failure_id:
                    failure["fail_reasons"] = fail_reasons
                    break
            held["failures"] = failures
            self._status_label.config(text=f"{held.get('analysis_id', '-')}: NG still failed")

        if failures:
            self._held_qc_failures[held_key] = held
            self._persist_qc_hold_state(held)
            self._refresh_item_status(
                str(held.get("date_name")),
                Path(held.get("date_path")),
                str(held.get("analysis_id")),
            )
            self._refresh_qc_hold_tree()
            self._show_held_qc_preview(held_key)
        else:
            self._held_qc_failures.pop(held_key, None)
            self._delete_qc_hold_state(held)
            self._refresh_item_status(
                str(held.get("date_name")),
                Path(held.get("date_path")),
                str(held.get("analysis_id")),
            )
            self._refresh_qc_hold_tree()
            self._show_selected_preview()
            if bool(held.get("resume_ml", False)):
                self._pending_hold_ml.append(dict(held))
                self._resume_next_pending_hold_ml()

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        combo_state = "readonly" if enabled else tk.DISABLED
        self._refresh_btn.config(state=state)
        self._clear_external_btn.config(state=state)
        self._run_qc_check.config(state=state)
        self._run_ml_check.config(state=state)
        self._run_btn.config(state=state)
        self._stop_btn.config(state=tk.DISABLED if enabled else tk.NORMAL)
        self._preview_check.config(state=state)
        self._force_run_check.config(state=state)
        self._mode_combo.config(state=combo_state)
        self._customer_combo.config(state=combo_state)
        self._date_combo.config(
            state=tk.DISABLED
            if self._mode_var.get() == "By Folder ID" or not enabled
            else "readonly"
        )
        self._tree.config(selectmode="extended")
        self._sync_review_nav_state()

    def _show_selected_preview(self) -> None:
        selected = self._explicit_selected_items()
        if not selected:
            self._review_entries = []
            self._review_index = 0
            self._review_label.config(text="Select a Folder ID to preview.")
            self._review_count_label.config(text="")
            self._before_canvas.delete("all")
            self._after_canvas.delete("all")
            self._sync_review_nav_state()
            return
        item = selected[0]
        if item.key in self._held_qc_failures:
            self._show_held_qc_preview(item.key)
            return
        if not self._preview_enabled:
            return
        if self._running:
            self._review_entries = []
            self._review_index = 0
            self._review_label.config(text="Review is available after workflow stops.")
            self._review_count_label.config(text="")
            self._sync_review_nav_state()
            return
        if item.qc_status != "DONE" or item.ml_status != "DONE":
            self._review_entries = []
            self._review_index = 0
            self._review_label.config(text=f"{item.analysis_id} / Complete QC + ML first.")
            self._review_count_label.config(text="")
            self._before_canvas.delete("all")
            self._after_canvas.delete("all")
            self._sync_review_nav_state()
            return

        self._review_entries = self._build_review_entries(item)
        self._review_index = 0
        if not self._review_entries:
            self._review_label.config(text=f"{item.analysis_id} / No middle-layer pass+mask pair found.")
            self._review_count_label.config(text="")
            self._before_canvas.delete("all")
            self._after_canvas.delete("all")
            self._sync_review_nav_state()
            return
        self._show_review_entry()

    def _show_held_qc_preview(self, key: str) -> None:
        held = self._held_qc_failures.get(key)
        if not held:
            return
        self._review_entries = self._build_held_qc_entries(held)
        self._review_index = 0
        if not self._review_entries:
            self._review_label.config(text=f"{held.get('analysis_id', '-')} / No NG image found.")
            self._review_count_label.config(text="")
            self._before_canvas.delete("all")
            self._after_canvas.delete("all")
            self._sync_review_nav_state()
            return
        self._show_review_entry()

    def _refresh_qc_hold_tree(self) -> None:
        if not hasattr(self, "_hold_tree"):
            return
        selected_meta = {
            self._hold_row_meta[item_id]
            for item_id in self._hold_tree.selection()
            if item_id in self._hold_row_meta
        }
        self._hold_tree.delete(*self._hold_tree.get_children())
        self._hold_row_meta.clear()
        restore: list[str] = []
        total = 0
        for held_key, held in sorted(
            self._held_qc_failures.items(),
            key=lambda item: (str(item[1].get("analysis_id", "")), item[0]),
        ):
            folder_id = str(held.get("analysis_id", "-"))
            analysis_path = Path(held.get("analysis_path"))
            for failure in held.get("failures", []):
                failure_id = str(failure.get("id", ""))
                image_path = Path(failure.get("image_path", ""))
                cell, face = self._failure_cell_face(image_path, analysis_path)
                reason = " | ".join(str(value) for value in failure.get("fail_reasons", [])) or "QC failed"
                row_id = self._hold_tree.insert(
                    "",
                    tk.END,
                    values=(folder_id, cell, face, image_path.name, reason),
                )
                meta = (held_key, failure_id)
                self._hold_row_meta[row_id] = meta
                if meta in selected_meta:
                    restore.append(row_id)
                total += 1
        if restore:
            self._hold_tree.selection_set(restore)
        try:
            tab_index = self._left_notebook.index(self._hold_tab)
            self._left_notebook.tab(tab_index, text=f"QC Hold Images ({total})")
        except tk.TclError:
            pass
        self._sync_review_nav_state()

    def _failure_cell_face(self, image_path: Path, analysis_path: Path) -> tuple[str, str]:
        cell = "-"
        face = "-"
        try:
            relative_parts = image_path.relative_to(analysis_path).parts
        except ValueError:
            relative_parts = image_path.parts
        for part in relative_parts:
            if self._cell_folder_re.fullmatch(part):
                match = re.search(r"(\d+)", part)
                cell = f"#{match.group(1)}" if match else part
            upper = part.upper()
            if upper in {"FRONT", "BACK"}:
                face = upper
        return cell, face

    def _on_hold_tree_select(self, _event=None) -> None:
        selection = self._hold_tree.selection()
        self._sync_review_nav_state()
        if not selection:
            return
        meta = self._hold_row_meta.get(selection[0])
        if meta is None:
            return
        held_key, failure_id = meta
        held = self._held_qc_failures.get(held_key)
        if not held:
            return
        entries = self._build_held_qc_entries(held)
        for index, entry in enumerate(entries):
            if str(entry.get("failure_id")) == failure_id:
                self._review_entries = entries
                self._review_index = index
                self._show_review_entry()
                return

    def _selected_hold_failures(self, *, all_rows: bool = False) -> list[tuple[str, str, dict[str, object]]]:
        row_ids = self._hold_tree.get_children() if all_rows else self._hold_tree.selection()
        selected: list[tuple[str, str, dict[str, object]]] = []
        seen: set[tuple[str, str]] = set()
        for row_id in row_ids:
            meta = self._hold_row_meta.get(row_id)
            if meta is None or meta in seen:
                continue
            held_key, failure_id = meta
            held = self._held_qc_failures.get(held_key)
            if not held:
                continue
            failure = next(
                (item for item in held.get("failures", []) if str(item.get("id")) == failure_id),
                None,
            )
            if failure is None:
                continue
            selected.append((held_key, failure_id, dict(failure)))
            seen.add(meta)
        return selected

    def _crop_selected_hold_images(self) -> None:
        self._start_bulk_hold_crop(self._selected_hold_failures(), label="selected")

    def _crop_all_hold_images(self) -> None:
        items = self._selected_hold_failures(all_rows=True)
        if not items:
            return
        if not messagebox.askyesno(
            "Crop All Anyway",
            f"Force-crop all {len(items)} QC Hold image(s)?\n\nThis will accept every currently held image.",
        ):
            return
        self._start_bulk_hold_crop(items, label="all")

    def _start_bulk_hold_crop(
        self,
        items: list[tuple[str, str, dict[str, object]]],
        *,
        label: str,
    ) -> None:
        if self._running or not items:
            return
        self._crop_selected_btn.config(state=tk.DISABLED)
        self._crop_all_btn.config(state=tk.DISABLED)
        self._retake_review_btn.config(state=tk.DISABLED)
        self._crop_review_btn.config(state=tk.DISABLED)
        self._status_label.config(text=f"Crop {label} anyway: 0/{len(items)}")

        def worker() -> None:
            results: list[dict[str, object]] = []
            for index, (held_key, failure_id, failure) in enumerate(items, start=1):
                try:
                    result = self._process_failed_qc_item(failure, force=True)
                    forced_saved = Path(failure.get("output_path", "")).is_file()
                    results.append(
                        {
                            "held_key": held_key,
                            "failure_id": failure_id,
                            "status": "PASS" if result.status == "PASS" or forced_saved else result.status,
                            "fail_reasons": list(result.fail_reasons),
                        }
                    )
                except Exception as exc:
                    logger.exception("Bulk QC Hold crop error")
                    results.append(
                        {
                            "held_key": held_key,
                            "failure_id": failure_id,
                            "status": "ERROR",
                            "fail_reasons": [str(exc)],
                        }
                    )
                self._queue.put(("current", f"Crop hold {index}/{len(items)}"))
            self._queue.put(("qc_ng_bulk_done", results))

        threading.Thread(target=worker, daemon=True).start()

    def _handle_qc_ng_bulk_done(self, results: list[dict[str, object]]) -> None:
        resolved_holds: dict[str, dict[str, object]] = {}
        for payload in results:
            held_key = str(payload.get("held_key", ""))
            failure_id = str(payload.get("failure_id", ""))
            held = self._held_qc_failures.get(held_key)
            if not held:
                continue
            failures = list(held.get("failures", []))
            if str(payload.get("status", "FAIL")) == "PASS":
                failures = [item for item in failures if str(item.get("id")) != failure_id]
            else:
                for failure in failures:
                    if str(failure.get("id")) == failure_id:
                        failure["fail_reasons"] = list(payload.get("fail_reasons", []))
                        break
            held["failures"] = failures
            if failures:
                self._held_qc_failures[held_key] = held
                self._persist_qc_hold_state(held)
            else:
                resolved_holds[held_key] = held
                self._held_qc_failures.pop(held_key, None)
                self._delete_qc_hold_state(held)
            self._refresh_item_status(
                str(held.get("date_name")),
                Path(held.get("date_path")),
                str(held.get("analysis_id")),
            )
        self._refresh_qc_hold_tree()
        self._status_label.config(text="QC Hold bulk crop complete")
        self._show_selected_preview()
        for held in resolved_holds.values():
            if bool(held.get("resume_ml", False)):
                self._pending_hold_ml.append(dict(held))
        if not self._running:
            self._resume_next_pending_hold_ml()

    def _resume_next_pending_hold_ml(self) -> None:
        if self._running or not self._pending_hold_ml:
            return
        held = self._pending_hold_ml.pop(0)
        self._resume_ml_for_resolved_hold(held)
        if not self._running and self._pending_hold_ml:
            self.after(50, self._resume_next_pending_hold_ml)

    def _resume_ml_for_resolved_hold(self, held: dict[str, object]) -> None:
        if self._running or not bool(held.get("resume_ml", False)):
            return
        date_name = str(held.get("date_name", ""))
        date_path = Path(held.get("date_path"))
        analysis_id = str(held.get("analysis_id", ""))
        if not date_name or not analysis_id:
            return
        item = self._build_item(date_name, date_path, analysis_id)
        if item.qc_status != "DONE":
            return
        self._status_label.config(text=f"{analysis_id}: QC resolved, resuming ML")
        self._start_run([item], force_qc=False, run_qc=False, run_ml=True)

    def _normalize_preview_payload(self, payload) -> tuple[object, object, str, str, str]:
        if isinstance(payload, tuple) and len(payload) == 5:
            before, after, label, before_title, after_title = payload
            return before, after, str(label), str(before_title), str(after_title)
        before, after, label = payload
        return before, after, str(label), "Before", "After / Mask"

    def _build_review_entries(self, item: _WorkflowItem) -> list[dict[str, object]]:
        tasks = self._build_ml_tasks(item)
        entries: list[dict[str, object]] = []
        for task in tasks:
            mask_path = self._mask_path_for_task(task)
            if not task.image_path.exists() or mask_path is None or not mask_path.exists():
                continue
            entries.append(
                {
                    "before": task.image_path,
                    "after": mask_path,
                    "label": self._task_label(task),
                }
            )
        return entries

    def _build_held_qc_entries(self, held: dict[str, object]) -> list[dict[str, object]]:
        analysis_id = str(held.get("analysis_id", "-"))
        analysis_path = Path(held.get("analysis_path"))
        entries: list[dict[str, object]] = []
        for failure in held.get("failures", []):
            image_path = Path(failure["image_path"])
            try:
                rel_name = str(image_path.relative_to(analysis_path))
            except ValueError:
                rel_name = image_path.name
            reason = " | ".join(str(reason) for reason in failure.get("fail_reasons", []))
            entries.append(
                {
                    "type": "qc_ng",
                    "held_key": held.get("key"),
                    "failure_id": failure.get("id"),
                    "failure": failure,
                    "before": image_path,
                    "after": None,
                    "before_title": "NG image",
                    "after_title": "No pass output",
                    "label": f"{analysis_id} / {rel_name} / {reason or 'QC failed'}",
                }
            )
        return entries

    def _mask_path_for_task(self, task: _ImageTask) -> Path | None:
        mask_dir = self._mask_dir_for_task(task)
        preferred = mask_dir / f"{task.image_path.stem}_mask.png"
        if preferred.exists():
            return preferred
        candidates = sorted(mask_dir.glob(f"{task.image_path.stem}*_mask.png")) if mask_dir.is_dir() else []
        return candidates[0] if candidates else None

    def _show_review_entry(self) -> None:
        if not self._review_entries:
            self._sync_review_nav_state()
            return
        self._review_index = max(0, min(self._review_index, len(self._review_entries) - 1))
        entry = self._review_entries[self._review_index]
        label = str(entry["label"])
        self._review_label.config(text=label)
        self._review_count_label.config(
            text=f"{self._review_index + 1} / {len(self._review_entries)}"
        )
        self._draw_image(self._before_canvas, entry["before"], str(entry.get("before_title", "Before")))
        self._draw_image(self._after_canvas, entry.get("after"), str(entry.get("after_title", "After / Mask")))
        self._sync_review_nav_state()

    def _show_next_review(self) -> None:
        if not self._review_can_navigate():
            return
        if self._review_index < len(self._review_entries) - 1:
            self._review_index += 1
            self._show_review_entry()

    def _show_previous_review(self) -> None:
        if not self._review_can_navigate():
            return
        if self._review_index > 0:
            self._review_index -= 1
            self._show_review_entry()

    def _on_review_tab_key(self, event) -> str | None:
        if not self._review_can_navigate() or not self._is_active_tab():
            return None
        self._show_next_review()
        return "break"

    def _is_active_tab(self) -> bool:
        parent = self.master
        try:
            return parent.select() == str(self)
        except Exception:
            return True

    def _sync_review_nav_state(self) -> None:
        enabled = self._review_can_navigate()
        self._prev_review_btn.config(
            state=tk.NORMAL if enabled and self._review_index > 0 else tk.DISABLED
        )
        self._next_review_btn.config(
            state=tk.NORMAL
            if enabled and self._review_index < len(self._review_entries) - 1
            else tk.DISABLED
        )
        current_entry = self._current_review_entry()
        ng_enabled = (
            not self._running
            and current_entry is not None
            and current_entry.get("type") == "qc_ng"
        )
        self._retake_review_btn.config(state=tk.NORMAL if ng_enabled else tk.DISABLED)
        self._crop_review_btn.config(state=tk.NORMAL if ng_enabled else tk.DISABLED)
        selected_hold_count = len(self._hold_tree.selection()) if hasattr(self, "_hold_tree") else 0
        total_hold_count = len(self._hold_row_meta) if hasattr(self, "_hold_row_meta") else 0
        self._crop_selected_btn.config(
            state=tk.NORMAL if not self._running and selected_hold_count > 0 else tk.DISABLED
        )
        self._crop_all_btn.config(
            state=tk.NORMAL if not self._running and total_hold_count > 0 else tk.DISABLED
        )

    def _review_can_navigate(self) -> bool:
        if not self._review_entries:
            return False
        current_entry = self._current_review_entry()
        if current_entry is not None and current_entry.get("type") == "qc_ng":
            return True
        return not self._running and self._preview_enabled

    def _current_review_entry(self) -> dict[str, object] | None:
        if not self._review_entries:
            return None
        index = max(0, min(self._review_index, len(self._review_entries) - 1))
        return self._review_entries[index]

    def _process_current_qc_ng(self, *, force: bool) -> None:
        if self._running:
            return
        entry = self._current_review_entry()
        if entry is None or entry.get("type") != "qc_ng":
            return
        failure = dict(entry.get("failure", {}))
        held_key = str(entry.get("held_key", ""))
        failure_id = str(entry.get("failure_id", ""))
        if not failure or not held_key or not failure_id:
            return
        self._retake_review_btn.config(state=tk.DISABLED)
        self._crop_review_btn.config(state=tk.DISABLED)
        action = "Crop anyway" if force else "Retake done"
        self._status_label.config(text=f"{action}: {Path(failure['image_path']).name}")

        def worker() -> None:
            try:
                result = self._process_failed_qc_item(failure, force=force)
                forced_saved = force and Path(failure.get("output_path", "")).is_file()
                self._queue.put(
                    (
                        "qc_ng_action_done",
                        {
                            "held_key": held_key,
                            "failure_id": failure_id,
                            "status": "PASS" if result.status == "PASS" or forced_saved else result.status,
                            "fail_reasons": list(result.fail_reasons),
                        },
                    )
                )
            except Exception as exc:
                logger.exception("QC NG action error")
                self._queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _draw_image(self, canvas: tk.Canvas, image_source: Path | np.ndarray | None, title: str) -> None:
        canvas.delete("all")
        width = max(canvas.winfo_width(), 220)
        height = max(canvas.winfo_height(), 180)
        canvas.create_text(8, 8, text=title, anchor=tk.NW, fill=_PREVIEW_TEXT)
        if image_source is None:
            canvas.create_text(width / 2, height / 2, text="Image not found", fill=_PREVIEW_MUTED)
            return
        if isinstance(image_source, np.ndarray):
            image = image_source
        else:
            image_path = Path(image_source)
            if not image_path.exists():
                canvas.create_text(width / 2, height / 2, text="Image not found", fill=_PREVIEW_MUTED)
                return
            cached = self._cached_photo(image_path, width, height)
            if cached is not None:
                canvas.create_image(width / 2, 30, image=cached, anchor=tk.N)
                return
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            canvas.create_text(width / 2, height / 2, text="Failed to load image", fill=_PREVIEW_MUTED)
            return
        image_h, image_w = image.shape[:2]
        scale = min((width - 16) / image_w, (height - 34) / image_h, 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(image_w * scale)), max(1, int(image_h * scale))),
                interpolation=cv2.INTER_AREA,
        )
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            canvas.create_text(width / 2, height / 2, text="Failed to render image", fill=_PREVIEW_MUTED)
            return
        photo = tk.PhotoImage(data=buffer.tobytes())
        self._photo_refs.append(photo)
        self._photo_refs = self._photo_refs[-8:]
        if not isinstance(image_source, np.ndarray):
            self._remember_photo(Path(image_source), width, height, photo)
        canvas.create_image(width / 2, 30, image=photo, anchor=tk.N)

    def _cached_photo(self, image_path: Path, width: int, height: int) -> tk.PhotoImage | None:
        try:
            mtime_ns = image_path.stat().st_mtime_ns
        except OSError:
            return None
        key = (str(image_path), int(mtime_ns), int(width), int(height))
        return self._photo_cache.get(key)

    def _remember_photo(self, image_path: Path, width: int, height: int, photo: tk.PhotoImage) -> None:
        try:
            mtime_ns = image_path.stat().st_mtime_ns
        except OSError:
            return
        key = (str(image_path), int(mtime_ns), int(width), int(height))
        self._photo_cache[key] = photo
        self._photo_cache_order.append(key)
        while len(self._photo_cache_order) > 32:
            old_key = self._photo_cache_order.pop(0)
            self._photo_cache.pop(old_key, None)
