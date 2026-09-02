import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import base64
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

from modules.ml.model_registry import list_model_options, resolve_model_path
from modules.pipelines.inference_pipeline import InferencePipeline, InferenceTask
from modules.pipelines.qc_pipeline import QCPipeline
from modules.qc.crop_engine import CropEngine
from modules.storage.sqlite_logger import SQLitePredictionLogger
from modules.utils import path_resolver
from modules.utils.analysis_id import (
    find_test_source_folders,
    get_test_mask_folder_for_pass,
    get_test_output_folders,
    is_test_analysis_id,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_RESULT_PATTERN_NEW = "{analysis_id}_LI-SALT_*.csv"
_RESULT_PATTERN_OLD = "{analysis_id}_result.csv"
_PREVIEW_BG = "#2b3036"
_PREVIEW_TEXT = "#e5e7eb"
_PREVIEW_MUTED = "#aeb6c2"
_HOLD_STATE_FILENAME = "_qc_hold.json"


@dataclass
class _WorkflowItem:
    key: str
    date_name: str
    date_path: Path
    analysis_id: str
    analysis_path: Path
    work_label: str = ""
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
    customer: str | None = None


@dataclass(frozen=True)
class _StageSource:
    raw_path: str
    name: str
    is_dir: bool | None


@dataclass(frozen=True)
class _FolderChild:
    path: Path
    is_dir: bool
    is_file: bool


class _QuickQCHoldError(RuntimeError):
    def __init__(self, message: str, failed_item: dict[str, object]) -> None:
        super().__init__(message)
        self.failed_item = failed_item


class WorkflowTab(ttk.Frame):
    """Integrated QC -> ML workflow dashboard."""

    def __init__(self, parent, config_path: str | Path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._load_config()

        self._qc_pipeline = QCPipeline(config_path)
        self._quick_crop_engine = CropEngine(config_path)
        self._inference_pipeline: InferencePipeline | None = None
        self._inference_save_mask_images: bool | None = None
        self._sqlite_logger = SQLitePredictionLogger(config_path)

        self._queue: queue.Queue = queue.Queue()
        self._items: list[_WorkflowItem] = []
        self._item_by_tree_id: dict[str, _WorkflowItem] = {}
        self._held_qc_failures: dict[str, dict[str, object]] = {}
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
        self._qc_fail_entries: list[dict[str, object]] = []
        self._qc_fail_index = 0
        self._qc_fail_entry_by_tree_id: dict[str, dict[str, object]] = {}
        self._quick_files: list[Path] = []
        self._quick_tree_by_path: dict[Path, str] = {}
        self._quick_display_by_path: dict[Path, str] = {}
        self._root_tree_paths: dict[str, Path] = {}
        self._folder_listing_cache = path_resolver.FolderListingCache()
        self._folder_root_override: Path | None = None
        self._path_quick_mode = False
        self._quick_path_var = tk.StringVar()
        self._run_qc_var = tk.BooleanVar(value=True)
        self._only_crop_var = tk.BooleanVar(value=False)
        self._run_ml_var = tk.BooleanVar(value=True)
        self._customer_var = tk.StringVar(value="ETC")
        self._default_view_mode = "By Folder"
        self._view_modes = ("By Date", "By Work Folder", "By Folder")

        self._build_ui()
        self._refresh_items()
        self._poll_queue()

    def reload_config(self) -> None:
        if self._running:
            self._status_label.config(text="Settings saved; reload after current run.")
            return
        self._load_config()
        self._qc_pipeline = QCPipeline(self._config_path)
        self._quick_crop_engine = CropEngine(self._config_path)
        self._inference_pipeline = None
        self._inference_save_mask_images = None
        self._folder_root_override = None
        self._path_quick_mode = False
        model_options = list_model_options(self._config_path)
        self._model_combo["values"] = model_options
        if model_options:
            self._model_var.set(model_options[0])
        self._customer_combo["values"] = self._customer_options()
        if self._customer_var.get() not in self._customer_combo["values"]:
            self._customer_var.set("ETC")
        if self._mode_var.get() in {"By Tree", "Root Tree"}:
            self._mode_var.set("By Folder")
        self._sync_view_mode_options()
        self._refresh_items()

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._root_path = Path(cfg["app"]["root_path"])
        self._analysis_id_re = re.compile(cfg["folder_qc"]["analysis_id_pattern"])
        self._cell_folder_re = re.compile(
            cfg["folder_qc"].get("cell_folder_pattern", r"^cell\s*#\d+$"),
            re.IGNORECASE,
        )
        self._date_formats = list(cfg["folder_qc"]["date_formats"])
        self._customer_codes = dict(cfg["folder_qc"].get("customer_codes", {}))
        self._image_count_rules = dict(cfg["folder_qc"].get("image_count_rules", {}))

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=(10, 6))
        toolbar.columnconfigure(10, weight=1)

        ttk.Label(toolbar, text="View").grid(row=0, column=0, sticky=tk.W)
        self._mode_var = tk.StringVar(value=self._default_view_mode)
        self._mode_combo = ttk.Combobox(
            toolbar,
            textvariable=self._mode_var,
            values=list(self._view_modes),
            state="readonly",
            width=16,
            postcommand=self._sync_view_mode_options,
        )
        self._mode_combo.grid(row=0, column=1, sticky=tk.W, padx=(6, 12))
        self._mode_combo.current(self._view_modes.index(self._default_view_mode))
        self._mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

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

        self._refresh_btn = ttk.Button(
            toolbar,
            text="Refresh",
            command=lambda: self._refresh_items(clear_cache=True),
        )
        self._refresh_btn.grid(row=0, column=4, sticky=tk.W)
        self._run_qc_check = ttk.Checkbutton(
            toolbar,
            text="QC (format/structure + crop/tilt)",
            variable=self._run_qc_var,
            command=self._on_run_qc_toggle,
        )
        self._run_qc_check.grid(row=0, column=5, sticky=tk.W, padx=(10, 0))
        self._only_crop_check = ttk.Checkbutton(
            toolbar,
            text="Only Crop",
            variable=self._only_crop_var,
            command=self._on_only_crop_toggle,
        )
        self._only_crop_check.grid(row=0, column=6, sticky=tk.W, padx=(8, 0))
        self._run_ml_check = ttk.Checkbutton(
            toolbar,
            text="ML",
            variable=self._run_ml_var,
            command=self._on_run_ml_toggle,
        )
        self._run_ml_check.grid(row=0, column=7, sticky=tk.W, padx=(8, 0))
        self._run_btn = ttk.Button(
            toolbar,
            text="Run",
            command=self._run_workflow,
        )
        self._run_btn.grid(row=0, column=8, sticky=tk.W, padx=(8, 0))
        self._stop_btn = ttk.Button(
            toolbar,
            text="Stop",
            command=self._request_stop,
            state=tk.DISABLED,
        )
        self._stop_btn.grid(row=0, column=9, sticky=tk.W, padx=(8, 0))
        self._status_label = ttk.Label(toolbar, text="", foreground="#555555")
        self._status_label.grid(row=0, column=10, sticky=tk.W, padx=(12, 0))

        ttk.Label(toolbar, text="Model").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self._model_var = tk.StringVar()
        model_options = list_model_options(self._config_path)
        if model_options:
            self._model_var.set(model_options[0])
        self._model_combo = ttk.Combobox(
            toolbar,
            textvariable=self._model_var,
            values=model_options,
            state="readonly",
            width=34,
        )
        self._model_combo.grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=(6, 12), pady=(6, 0))
        self._model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        ttk.Label(toolbar, text="Customer").grid(row=1, column=4, sticky=tk.W, pady=(6, 0))
        self._customer_combo = ttk.Combobox(
            toolbar,
            textvariable=self._customer_var,
            values=self._customer_options(),
            state="readonly",
            width=18,
        )
        self._customer_combo.grid(row=1, column=5, columnspan=3, sticky=tk.W, padx=(6, 12), pady=(6, 0))
        self._customer_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_items())
        if not self._customer_var.get():
            self._customer_var.set("ETC")
        self._build_quick_run_ui()

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=2, column=0, sticky=tk.NSEW, padx=10, pady=(0, 10))

        left = ttk.Frame(paned)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        paned.add(left, weight=3)

        columns = ("date", "analysis_id", "qc", "ml", "li_salt", "correction", "summary")
        self._tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="extended")
        headings = {
            "date": "Date",
            "analysis_id": "Work Folder",
            "qc": "QC",
            "ml": "ML / DB",
            "li_salt": "Li-salt %",
            "correction": "Correction",
            "summary": "Summary",
        }
        widths = {
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
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._show_selected_preview())
        self._tree.tag_configure("red", background="#fee2e2")
        self._tree.tag_configure("yellow", background="#fef9c3")
        self._tree.tag_configure("blue", background="#dbeafe")

        self._tree_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._tree.configure(yscrollcommand=self._tree_scroll.set)

        self._root_tree = ttk.Treeview(
            left,
            columns=("images", "folders", "selected", "status"),
            show="tree headings",
            selectmode="extended",
        )
        self._root_tree.heading("#0", text="Name")
        self._root_tree.heading("images", text="Images")
        self._root_tree.heading("folders", text="Folders")
        self._root_tree.heading("selected", text="Selected")
        self._root_tree.heading("status", text="Status")
        self._root_tree.column("#0", width=460, anchor=tk.W)
        self._root_tree.column("images", width=80, anchor=tk.W)
        self._root_tree.column("folders", width=80, anchor=tk.W)
        self._root_tree.column("selected", width=80, anchor=tk.W)
        self._root_tree.column("status", width=260, anchor=tk.W)
        self._root_tree.tag_configure("folder_done", background="#dbeafe")
        self._root_tree.tag_configure("folder_crop", background="#fef9c3")
        self._root_tree.tag_configure("folder_partial", background="#fef9c3")
        self._root_tree.tag_configure("folder_ready", background="#ffffff")
        self._root_tree.tag_configure("folder_output", background="#e5e7eb")
        self._root_tree.tag_configure("folder_fail", background="#fee2e2")
        self._root_tree.bind("<<TreeviewOpen>>", self._on_root_tree_open)
        self._root_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_root_tree_selection())
        self._root_tree.bind("<Double-1>", self._on_root_tree_double_click)
        self._root_tree_scroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._root_tree.yview)
        self._root_tree.configure(yscrollcommand=self._root_tree_scroll.set)
        self._setup_folder_drag_drop()

        right = ttk.Frame(paned)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        paned.add(right, weight=2)

        self._progress = ttk.Progressbar(right, mode="determinate", maximum=1, value=0)
        self._progress.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        self._current_label = ttk.Label(right, text="Current: -")
        self._current_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 6))

        legend = ttk.Frame(right)
        legend.grid(row=2, column=0, sticky=tk.W, pady=(0, 6))
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

        self._review_notebook = ttk.Notebook(right)
        self._review_notebook.grid(row=3, column=0, sticky=tk.NSEW)

        review = ttk.Frame(self._review_notebook, padding=8)
        self._review_notebook.add(review, text="Review")
        review.columnconfigure(0, weight=1)
        review.rowconfigure(2, weight=1)
        review.rowconfigure(3, weight=1)

        self._review_label = ttk.Label(review, text="Select a work folder to preview.")
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
        self._before_canvas = tk.Canvas(review, bg=_PREVIEW_BG, height=165, highlightthickness=1)
        self._before_canvas.grid(row=2, column=0, sticky=tk.NSEW, pady=(0, 5))
        self._after_canvas = tk.Canvas(review, bg=_PREVIEW_BG, height=165, highlightthickness=1)
        self._after_canvas.grid(row=3, column=0, sticky=tk.NSEW, pady=(5, 0))

        qc_fail = ttk.Frame(self._review_notebook, padding=8)
        self._qc_fail_tab = qc_fail
        self._review_notebook.add(qc_fail, text="QC Failures")
        qc_fail.columnconfigure(0, weight=1)
        qc_fail.rowconfigure(2, weight=1)
        qc_fail.rowconfigure(3, weight=1)
        self._qc_fail_tree = ttk.Treeview(
            qc_fail,
            columns=("image", "reason"),
            show="headings",
            height=4,
            selectmode="browse",
        )
        self._qc_fail_tree.heading("image", text="Image")
        self._qc_fail_tree.heading("reason", text="Reason")
        self._qc_fail_tree.column("image", width=320, anchor=tk.W)
        self._qc_fail_tree.column("reason", width=440, anchor=tk.W)
        self._qc_fail_tree.tag_configure("qc_fail", background="#fee2e2")
        self._qc_fail_tree.grid(row=0, column=0, sticky=tk.EW)
        self._qc_fail_tree.bind("<<TreeviewSelect>>", self._on_qc_fail_select)
        qc_fail_nav = ttk.Frame(qc_fail)
        qc_fail_nav.grid(row=1, column=0, sticky=tk.EW, pady=(6, 6))
        qc_fail_nav.columnconfigure(0, weight=1)
        self._qc_fail_count_label = ttk.Label(qc_fail_nav, text="", foreground="#555555")
        self._qc_fail_count_label.grid(row=0, column=0, sticky=tk.W)
        self._prev_qc_fail_btn = ttk.Button(
            qc_fail_nav,
            text="Previous",
            command=self._show_previous_qc_fail,
            state=tk.DISABLED,
        )
        self._prev_qc_fail_btn.grid(row=0, column=1, sticky=tk.E, padx=(0, 6))
        self._next_qc_fail_btn = ttk.Button(
            qc_fail_nav,
            text="Next",
            command=self._show_next_qc_fail,
            state=tk.DISABLED,
        )
        self._next_qc_fail_btn.grid(row=0, column=2, sticky=tk.E)
        self._retake_qc_fail_btn = ttk.Button(
            qc_fail_nav,
            text="Retake done",
            command=lambda: self._process_current_qc_fail_ng(force=False),
            state=tk.DISABLED,
        )
        self._retake_qc_fail_btn.grid(row=0, column=3, sticky=tk.E, padx=(10, 6))
        self._crop_qc_fail_btn = ttk.Button(
            qc_fail_nav,
            text="Crop anyway",
            command=lambda: self._process_current_qc_fail_ng(force=True),
            state=tk.DISABLED,
        )
        self._crop_qc_fail_btn.grid(row=0, column=4, sticky=tk.E)
        self._qc_fail_before_canvas = tk.Canvas(qc_fail, bg=_PREVIEW_BG, height=150, highlightthickness=1)
        self._qc_fail_before_canvas.grid(row=2, column=0, sticky=tk.NSEW, pady=(0, 5))
        self._qc_fail_after_canvas = tk.Canvas(qc_fail, bg=_PREVIEW_BG, height=150, highlightthickness=1)
        self._qc_fail_after_canvas.grid(row=3, column=0, sticky=tk.NSEW, pady=(5, 0))
        self.bind_all("<Tab>", self._on_review_tab_key, add="+")

    def _build_quick_run_ui(self) -> None:
        quick = ttk.LabelFrame(self, text="Quick Crop + ML", padding=8)
        quick.grid(row=1, column=0, sticky=tk.EW, padx=10, pady=(0, 8))
        quick.columnconfigure(2, weight=1)

        self._quick_run_btn = ttk.Button(
            quick,
            text="Run Quick Crop + ML",
            command=self._start_quick_run,
        )
        self._quick_run_btn.grid(row=0, column=0, sticky=tk.W)
        self._quick_clear_btn = ttk.Button(
            quick,
            text="Clear",
            command=self._clear_quick_files,
        )
        self._quick_clear_btn.grid(row=0, column=1, sticky=tk.W, padx=(8, 0))

        self._quick_drop_label = ttk.Label(
            quick,
            text="Paste a secure image/folder path below; files are listed here and copied locally when run.",
            foreground="#555555",
        )
        self._quick_drop_label.grid(row=0, column=2, columnspan=4, sticky=tk.W, padx=(12, 0))

        ttk.Label(quick, text="Paths").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self._quick_path_entry = ttk.Entry(quick, textvariable=self._quick_path_var)
        self._quick_path_entry.grid(row=1, column=1, columnspan=4, sticky=tk.EW, padx=(8, 8), pady=(6, 0))
        self._quick_path_entry.bind("<Return>", lambda _event: self._add_quick_paths_from_entry())
        self._quick_path_entry.bind("<KP_Enter>", lambda _event: self._add_quick_paths_from_entry())
        self._quick_path_add_btn = ttk.Button(
            quick,
            text="Load Path",
            command=self._add_quick_paths_from_entry,
        )
        self._quick_path_add_btn.grid(row=1, column=5, sticky=tk.E, pady=(6, 0))

        columns = ("file", "status", "output")
        self._quick_tree = ttk.Treeview(
            quick,
            columns=columns,
            show="headings",
            height=4,
            selectmode="extended",
        )
        self._quick_tree.heading("file", text="Image")
        self._quick_tree.heading("status", text="Status")
        self._quick_tree.heading("output", text="Output")
        self._quick_tree.column("file", width=360, anchor=tk.W)
        self._quick_tree.column("status", width=160, anchor=tk.W)
        self._quick_tree.column("output", width=480, anchor=tk.W)
        self._quick_tree.tag_configure("quick_fail", background="#fee2e2")
        self._quick_tree.tag_configure("quick_running", background="#fef9c3")
        self._quick_tree.tag_configure("quick_pass", background="#dbeafe")
        self._quick_tree.grid(row=2, column=0, columnspan=5, sticky=tk.EW, pady=(6, 0))

        quick_scroll = ttk.Scrollbar(quick, orient=tk.VERTICAL, command=self._quick_tree.yview)
        quick_scroll.grid(row=2, column=5, sticky=tk.NS, pady=(6, 0))
        self._quick_tree.configure(yscrollcommand=quick_scroll.set)
        self._setup_quick_drag_drop()

    def _setup_quick_drag_drop(self) -> None:
        try:
            self.tk.call("package", "require", "tkdnd")
            for widget in (self._quick_drop_label, self._quick_tree, self._quick_path_entry):
                self.tk.call("tkdnd::drop_target", "register", widget, "DND_Files")
                widget.bind("<<Drop>>", self._on_quick_drop, add="+")
        except tk.TclError:
            self._quick_drop_label.config(
                text="Paste a secure image/folder path below; files are listed here and copied locally when run.",
            )

    def _setup_folder_drag_drop(self) -> None:
        try:
            self.tk.call("package", "require", "tkdnd")
            self.tk.call("tkdnd::drop_target", "register", self._root_tree, "DND_Files")
            self._root_tree.bind("<<Drop>>", self._on_quick_drop, add="+")
        except tk.TclError:
            pass

    def _sync_view_mode_options(self) -> None:
        self._mode_combo["values"] = list(self._view_modes)

    def _on_mode_change(self, _event=None) -> None:
        if self._is_folder_mode():
            self._run_qc_var.set(True)
            self._only_crop_var.set(False)
            self._run_ml_var.set(True)
        if not self._is_folder_mode():
            self._path_quick_mode = False
        self._sync_run_mode_controls(not self._running)
        self._refresh_items()

    def _is_folder_mode(self) -> bool:
        return self._mode_var.get() in {"By Folder", "By Tree", "Root Tree"}

    def _sync_mode_view(self) -> None:
        if self._is_folder_mode():
            self._tree.grid_remove()
            self._tree_scroll.grid_remove()
            self._root_tree.grid(row=0, column=0, sticky=tk.NSEW)
            self._root_tree_scroll.grid(row=0, column=1, sticky=tk.NS)
        else:
            self._root_tree.grid_remove()
            self._root_tree_scroll.grid_remove()
            self._tree.grid(row=0, column=0, sticky=tk.NSEW)
            self._tree_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._sync_run_button_text()

    def _current_folder_root(self) -> Path:
        return self._folder_root_override or self._root_path

    def _folder_cache_key(self, folder_path: Path) -> str:
        return str(Path(folder_path)).rstrip("\\/").lower()

    def _ensure_folder_scan_cache(self) -> None:
        if not hasattr(self, "_folder_listing_cache"):
            self._folder_listing_cache = path_resolver.FolderListingCache()

    def _clear_folder_scan_cache(self) -> None:
        self._ensure_folder_scan_cache()
        self._folder_listing_cache.clear()

    def _path_is_dir(self, folder_path: Path) -> bool:
        self._ensure_folder_scan_cache()
        folder_path = Path(folder_path)
        if folder_path.suffix.lower() in _IMAGE_EXTENSIONS:
            return False
        try:
            if folder_path.is_dir():
                return True
        except OSError:
            pass
        try:
            self._list_folder_children(folder_path)
            return True
        except OSError:
            return False

    def _list_folder_children(self, folder_path: Path) -> list[_FolderChild]:
        self._ensure_folder_scan_cache()
        return [
            _FolderChild(
                path=child.path,
                is_dir=child.is_dir,
                is_file=child.is_file,
            )
            for child in self._folder_listing_cache.children(
                folder_path,
                allow_fallback=False,
            )
        ]

    def _list_folder_children_with_subprocess(self, folder_path: Path) -> list[_FolderChild]:
        result = self._run_powershell_env(
            (
                "$enc=[Text.Encoding]::UTF8; "
                "Get-ChildItem -LiteralPath $env:QCML_SOURCE_PATH -Force | ForEach-Object { "
                "$kind = if ($_.PSIsContainer) { 'dir' } else { 'file' }; "
                "[Convert]::ToBase64String($enc.GetBytes($_.FullName)) + \"`t\" + $kind "
                "}"
            ),
            {"QCML_SOURCE_PATH": str(folder_path)},
        )
        if result.returncode != 0:
            detail = self._short_copy_output(result.stderr or result.stdout or "")
            raise OSError(detail or f"Failed to list folder: {folder_path}")

        children: list[_FolderChild] = []
        for line in (result.stdout or "").splitlines():
            encoded, sep, kind = line.partition("\t")
            if not sep:
                continue
            try:
                child_path = Path(base64.b64decode(encoded).decode("utf-8"))
            except Exception:
                continue
            children.append(
                _FolderChild(
                    path=child_path,
                    is_dir=kind.strip().lower() == "dir",
                    is_file=kind.strip().lower() == "file",
                )
            )
        return children

    def _refresh_root_tree(self, *, clear_cache: bool = False) -> None:
        if clear_cache:
            self._clear_folder_scan_cache()
        folder_root = self._current_folder_root()
        self._root_tree_paths.clear()
        for item_id in self._root_tree.get_children():
            self._root_tree.delete(item_id)
        if not self._path_is_dir(folder_root):
            self._status_label.config(text=f"Root path not found: {folder_root}")
            return
        root_label = folder_root.name or str(folder_root)
        root_values, root_tag = self._folder_tree_values(folder_root, selected=False)
        root_id = self._root_tree.insert(
            "",
            tk.END,
            text=root_label,
            values=root_values,
            tags=(root_tag,),
            open=True,
        )
        self._root_tree_paths[root_id] = folder_root
        self._insert_root_tree_children(root_id, folder_root)
        self._auto_expand_loaded_folder(root_id)
        self._sync_path_current_label()
        self._status_label.config(text=f"By Folder: {folder_root}")

    def _insert_root_tree_children(self, parent_id: str, folder_path: Path) -> None:
        for child_id in self._root_tree.get_children(parent_id):
            if child_id not in self._root_tree_paths:
                self._root_tree.delete(child_id)
        try:
            children = sorted(
                self._list_folder_children(folder_path),
                key=lambda child: child.path.name.lower(),
            )
        except OSError as exc:
            self._root_tree.item(parent_id, values=("", "", "", f"FAIL: {exc}"), tags=("folder_fail",))
            return
        for child in children:
            child_path = child.path
            if self._is_output_folder(child_path):
                continue
            if child.is_dir:
                values, tag = self._folder_tree_values(child_path, selected=False)
                child_id = self._root_tree.insert(
                    parent_id,
                    tk.END,
                    text=child_path.name,
                    values=values,
                    tags=(tag,),
                )
                self._root_tree_paths[child_id] = child_path
                if self._has_root_tree_child(child_path):
                    self._root_tree.insert(child_id, tk.END, text="...", values=("", "", "", ""))

    def _folder_tree_values(self, folder_path: Path, *, selected: bool) -> tuple[tuple[str, str, str, str], str]:
        image_count, folder_count = self._folder_child_counts(folder_path)
        status, tag = self._folder_processing_status(folder_path)
        return (str(image_count), str(folder_count), "Yes" if selected else "", status), tag

    def _folder_child_counts(self, folder_path: Path) -> tuple[int, int]:
        image_count = 0
        folder_count = 0
        try:
            children = self._list_folder_children(folder_path)
        except OSError:
            return 0, 0
        for child in children:
            if self._is_output_folder(child.path):
                continue
            if child.is_dir:
                folder_count += 1
            elif child.is_file and child.path.suffix.lower() in _IMAGE_EXTENSIONS:
                image_count += 1
        return image_count, folder_count

    def _folder_processing_status(self, folder_path: Path) -> tuple[str, str]:
        if self._is_output_folder(folder_path):
            return "Output folder", "folder_output"
        image_paths = self._direct_quick_image_paths(folder_path)
        conventional_sources = self._conventional_source_folders_for_status(folder_path)
        if conventional_sources:
            total, pass_count, mask_count = self._conventional_status_counts(conventional_sources)
            return self._pass_mask_status(
                total=total,
                pass_count=pass_count,
                mask_count=mask_count,
                li_salt=self._li_salt_summary(self._result_csv_for_folder(folder_path)),
            )

        conventional_crop, conventional_ml = self._conventional_output_counts(folder_path)
        if not image_paths:
            if conventional_ml:
                status = f"MASK done {conventional_ml}"
                return self._append_li_salt_status(
                    status,
                    self._li_salt_summary(self._result_csv_for_folder(folder_path)),
                ), "folder_done"
            if conventional_crop:
                return f"MASK missing 0/{conventional_crop}", "folder_partial"
            if self._has_root_tree_child(folder_path):
                return "Open to inspect", "folder_ready"
            return "No images", "folder_ready"

        crop_done = 0
        ml_done = 0
        for image_path in image_paths:
            crop_path, mask_path, csv_path = self._quick_output_paths(image_path)
            conventional_crop_paths, conventional_mask_paths = self._conventional_outputs_for_image(image_path)
            if crop_path.exists() or any(path.exists() for path in conventional_crop_paths):
                crop_done += 1
            if mask_path.exists() or csv_path.exists() or any(path.exists() for path in conventional_mask_paths):
                ml_done += 1
        return self._pass_mask_status(
            total=len(image_paths),
            pass_count=max(crop_done, conventional_crop),
            mask_count=max(ml_done, conventional_ml),
            li_salt=self._li_salt_for_image_folder(folder_path, image_paths),
        )

    def _pass_mask_status(
        self,
        *,
        total: int,
        pass_count: int,
        mask_count: int,
        li_salt: str = "-",
    ) -> tuple[str, str]:
        if mask_count > 0:
            if pass_count <= 0:
                return self._append_li_salt_status(f"MASK done {mask_count}", li_salt), "folder_done"
            if mask_count >= pass_count:
                return self._append_li_salt_status(f"MASK done {mask_count}/{pass_count}", li_salt), "folder_done"
            return f"MASK missing {mask_count}/{pass_count}", "folder_partial"
        if pass_count > 0:
            return f"MASK missing 0/{pass_count}", "folder_partial"
        if total > 0:
            return f"PASS missing 0/{total}", "folder_fail"
        return "No images", "folder_ready"

    def _append_li_salt_status(self, status: str, li_salt: str) -> str:
        return f"{status} / Li-salt {li_salt}" if li_salt and li_salt != "-" else status

    def _direct_quick_image_paths(self, folder_path: Path) -> list[Path]:
        if not self._path_is_dir(folder_path):
            return []
        try:
            return sorted(
                [
                    child.path
                    for child in self._list_folder_children(folder_path)
                    if child.is_file
                    and child.path.suffix.lower() in _IMAGE_EXTENSIONS
                    and not self._is_ignored_quick_source(child.path)
                ],
                key=lambda path: path.name.lower(),
            )
        except OSError:
            return []

    def _image_processing_status(self, image_path: Path) -> tuple[str, str]:
        crop_path, mask_path, csv_path = self._quick_output_paths(image_path)
        conventional_crops, conventional_masks = self._conventional_outputs_for_image(image_path)
        if mask_path.exists() or csv_path.exists() or any(path.exists() for path in conventional_masks):
            return self._append_li_salt_status(
                "MASK done",
                self._li_salt_for_image(image_path, csv_path),
            ), "folder_done"
        if crop_path.exists() or any(path.exists() for path in conventional_crops):
            return "MASK missing", "folder_partial"
        return "PASS missing", "folder_fail"

    def _quick_output_paths(self, image_path: Path) -> tuple[Path, Path, Path]:
        output_dir = image_path.parent / "_quick_crop_ml"
        return (
            output_dir / f"{image_path.stem}_crop.png",
            output_dir / f"{image_path.stem}_mask.png",
            output_dir / f"{image_path.stem}_quant.csv",
        )

    def _is_output_folder(self, path: Path) -> bool:
        name = path.name.lower()
        return (
            name in {"_quick_crop_ml", "__pycache__"}
            or name.endswith("_pass")
            or name.endswith("_mask")
            or name.endswith("_masks")
        )

    def _conventional_output_counts(self, folder_path: Path) -> tuple[int, int]:
        crop_count = 0
        mask_count = 0
        try:
            output_dirs = [
                child.path
                for child in self._list_folder_children(folder_path)
                if child.is_dir and self._is_output_folder(child.path)
            ]
        except OSError:
            return 0, 0
        for output_dir in output_dirs:
            name = output_dir.name.lower()
            if name.endswith("_mask") or name.endswith("_masks"):
                mask_count += len(self._mask_pngs(output_dir))
            elif name.endswith("_pass"):
                crop_count += len(self._pass_pngs(output_dir))
        return crop_count, mask_count

    def _conventional_source_folders_for_status(self, folder_path: Path) -> list[tuple[str, Path]]:
        if not self._path_is_dir(folder_path):
            return []
        sources: list[tuple[str, Path]] = []
        if folder_path.name.upper() in {"FRONT", "BACK"}:
            sources.append((folder_path.name.upper(), folder_path))
        try:
            children = [
                child.path
                for child in self._list_folder_children(folder_path)
                if child.is_dir
            ]
        except OSError:
            return sources
        for child in children:
            if child.name.upper() in {"FRONT", "BACK"}:
                sources.append((child.name.upper(), child))
                continue
            cell_folder_re = getattr(self, "_cell_folder_re", None)
            is_cell_folder = bool(cell_folder_re.match(child.name)) if cell_folder_re else False
            if not is_cell_folder:
                continue
            try:
                for grandchild in self._list_folder_children(child):
                    if grandchild.is_dir and grandchild.path.name.upper() in {"FRONT", "BACK"}:
                        sources.append((grandchild.path.name.upper(), grandchild.path))
            except OSError:
                continue

        unique: list[tuple[str, Path]] = []
        seen: set[Path] = set()
        for face_name, source_path in sources:
            try:
                key = source_path.resolve(strict=False)
            except OSError:
                key = source_path
            if key in seen:
                continue
            seen.add(key)
            unique.append((face_name, source_path))
        return unique

    def _conventional_status_counts(self, sources: list[tuple[str, Path]]) -> tuple[int, int, int]:
        total = 0
        pass_count = 0
        mask_count = 0
        for face_name, source_path in sources:
            images = self._source_images(source_path)
            total += len(images)
            pass_count += len(self._pass_pngs(self._pass_output_folder(face_name, source_path)))
            mask_count += len(self._mask_pngs(source_path.parent / f"{face_name}_mask"))
            mask_count += len(self._mask_pngs(source_path.parent / f"{face_name}_masks"))
        return total, pass_count, mask_count

    def _result_csv_for_folder(self, folder_path: Path) -> Path | None:
        for candidate in [folder_path, *folder_path.parents]:
            if self._is_analysis_folder_name(candidate.name):
                return self._find_result_csv(candidate, candidate.name)
        return None

    def _is_analysis_folder_name(self, folder_name: str) -> bool:
        analysis_id_re = getattr(self, "_analysis_id_re", None)
        if analysis_id_re is not None and analysis_id_re.match(folder_name):
            return True
        return bool(re.match(r"^[A-Z]{3}[0-9]{5}$", folder_name)) or is_test_analysis_id(folder_name)

    def _li_salt_for_image_folder(self, folder_path: Path, image_paths: list[Path]) -> str:
        values = []
        for image_path in image_paths:
            value = self._li_salt_percent_from_summary(
                self._li_salt_summary(self._quick_result_csv_for_source(image_path))
            )
            if value is not None:
                values.append(value)
        if values:
            return f"{self._format_li_salt_percent(sum(values) / len(values))}%"
        return self._li_salt_summary(self._result_csv_for_folder(folder_path))

    def _li_salt_for_image(self, image_path: Path, quick_csv_path: Path) -> str:
        li_salt = self._li_salt_summary(self._quick_result_csv_for_source(image_path) or quick_csv_path)
        if li_salt != "-":
            return li_salt
        return self._li_salt_summary(self._result_csv_for_folder(image_path.parent))

    def _conventional_outputs_for_image(self, image_path: Path) -> tuple[list[Path], list[Path]]:
        source_folder = image_path.parent
        face_name = source_folder.name if source_folder.name.upper() in {"FRONT", "BACK"} else "FRONT"
        crop_paths = [
            self._pass_output_folder(face_name, source_folder) / f"{image_path.stem}.png",
            *[
                folder / f"{image_path.stem}.png"
                for folder in self._work_pass_folder_candidates(source_folder)
            ],
        ]
        mask_paths = [
            source_folder.parent / f"{face_name}_mask" / f"{image_path.stem}_mask.png",
            source_folder.parent / f"{face_name}_MASK" / f"{image_path.stem}_mask.png",
            *[
                folder / f"{image_path.stem}_mask.png"
                for folder in self._work_mask_folder_candidates(source_folder)
            ],
        ]
        return self._unique_paths(crop_paths), self._unique_paths(mask_paths)

    def _has_root_tree_child(self, folder_path: Path) -> bool:
        try:
            for child in self._list_folder_children(folder_path):
                if self._is_output_folder(child.path):
                    continue
                if child.is_dir:
                    return True
        except OSError:
            return False
        return False

    def _auto_expand_loaded_folder(self, root_id: str) -> None:
        face_parent_id = root_id
        face_parent_path = self._root_tree_paths.get(face_parent_id)
        if face_parent_path is None:
            return

        for _depth in range(3):
            if self._folder_has_face_children(face_parent_path):
                self._expand_face_folder_children(face_parent_id)
                return

            child_ids = [
                child_id
                for child_id in self._root_tree.get_children(face_parent_id)
                if child_id in self._root_tree_paths
            ]
            if len(child_ids) != 1:
                return
            child_id = child_ids[0]
            child_path = self._root_tree_paths.get(child_id)
            if child_path is None or not self._path_is_dir(child_path):
                return

            self._expand_tree_folder_node(child_id, child_path)
            face_parent_id = child_id
            face_parent_path = child_path

    def _expand_face_folder_children(self, parent_id: str) -> None:
        for child_id in self._root_tree.get_children(parent_id):
            child_path = self._root_tree_paths.get(child_id)
            if child_path is None or child_path.name.upper() not in {"FRONT", "BACK"}:
                continue
            self._expand_tree_folder_node(child_id, child_path)

    def _expand_tree_folder_node(self, item_id: str, folder_path: Path) -> None:
        self._root_tree.item(item_id, open=True)
        for placeholder_id in self._root_tree.get_children(item_id):
            if placeholder_id not in self._root_tree_paths:
                self._root_tree.delete(placeholder_id)
        self._insert_root_tree_children(item_id, folder_path)

    def _auto_expand_single_face_parent(self, parent_id: str) -> None:
        child_ids = [
            child_id
            for child_id in self._root_tree.get_children(parent_id)
            if child_id in self._root_tree_paths
        ]
        if len(child_ids) != 1:
            return
        child_id = child_ids[0]
        child_path = self._root_tree_paths.get(child_id)
        if child_path is None or not self._path_is_dir(child_path):
            return
        if not self._folder_has_face_children(child_path):
            return
        self._expand_tree_folder_node(child_id, child_path)
        self._expand_face_folder_children(child_id)

    def _folder_has_face_children(self, folder_path: Path) -> bool:
        try:
            child_names = {
                child.path.name.upper()
                for child in self._list_folder_children(folder_path)
                if child.is_dir
            }
        except OSError:
            return False
        return bool({"FRONT", "BACK"} & child_names)

    def _on_root_tree_open(self, _event=None) -> None:
        selected_id = self._root_tree.focus()
        if not selected_id or selected_id not in self._root_tree_paths:
            selected_id = self._open_placeholder_parent()
        path = self._root_tree_paths.get(selected_id)
        if path is None or not self._path_is_dir(path):
            return
        children = self._root_tree.get_children(selected_id)
        if children and all(child not in self._root_tree_paths for child in children):
            for child in children:
                self._root_tree.delete(child)
            self._insert_root_tree_children(selected_id, path)

    def _on_root_tree_double_click(self, _event=None) -> str:
        item_id = self._root_tree.focus()
        if not item_id:
            selection = self._root_tree.selection()
            item_id = selection[0] if selection else ""
        folder_path = self._root_tree_paths.get(item_id)
        if folder_path is not None and self._path_is_dir(folder_path):
            self._set_folder_root(folder_path, quick_mode=self._path_quick_mode)
        return "break"

    def _go_to_parent_folder(self) -> None:
        if self._running:
            return
        current = self._current_folder_root()
        parent = current.parent
        if parent == current:
            return
        self._set_folder_root(parent, quick_mode=self._path_quick_mode)

    def _set_folder_root(self, folder_path: Path, *, quick_mode: bool) -> None:
        self._folder_root_override = folder_path
        self._path_quick_mode = quick_mode
        self._mode_var.set("By Folder")
        self._sync_mode_view()
        self._refresh_root_tree()
        self._sync_run_button_text()
        self._sync_path_current_label()

    def _open_placeholder_parent(self) -> str | None:
        for item_id, path in self._root_tree_paths.items():
            try:
                if not self._root_tree.item(item_id, "open"):
                    continue
                children = self._root_tree.get_children(item_id)
            except tk.TclError:
                continue
            if children and all(child not in self._root_tree_paths for child in children):
                return item_id
        return None

    def _selected_root_paths(self) -> list[Path]:
        return [
            self._root_tree_paths[item_id]
            for item_id in self._root_tree.selection()
            if item_id in self._root_tree_paths
        ]

    def _show_root_tree_selection(self) -> None:
        self._update_root_tree_selection_marks()
        selected = self._selected_root_paths()
        if not selected:
            self._status_label.config(text=f"By Folder: {self._current_folder_root()}")
            return
        if self._path_quick_mode:
            self._status_label.config(
                text=f"Quick folder selection: {len(selected)} selected"
            )
        else:
            self._status_label.config(text=f"By Folder: {len(selected)} selected")
        return

    def _update_root_tree_selection_marks(self) -> None:
        selected_ids = set(self._root_tree.selection())
        for item_id, folder_path in list(self._root_tree_paths.items()):
            try:
                values = list(self._root_tree.item(item_id, "values"))
            except tk.TclError:
                continue
            while len(values) < 4:
                values.append("")
            values[2] = "Yes" if item_id in selected_ids else ""
            self._root_tree.item(item_id, values=values)

    def _show_image_preview_from_selection(self) -> None:
        selected = self._selected_root_paths()
        if len(selected) != 1 or not selected[0].is_file():
            return
        image_path = selected[0]
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            return
        crop_path, mask_path, csv_path = self._quick_output_paths(image_path)
        after_path: Path | None = None
        after_title = "Crop / Mask"
        if mask_path.exists():
            after_path = mask_path
            after_title = "ML mask"
        elif crop_path.exists():
            after_path = crop_path
            after_title = "Crop"
        status = "ML done" if mask_path.exists() or csv_path.exists() else ("Crop done" if crop_path.exists() else "Ready")
        self._review_entries = [
            {
                "before": image_path,
                "after": after_path,
                "before_title": "Source",
                "after_title": after_title,
                "label": f"By Folder / {image_path.name} / {status}",
            }
        ]
        self._review_index = 0
        self._show_review_entry()

    def _list_date_folders(self) -> list[str]:
        if not self._root_path.exists():
            return []
        folders: list[tuple[bool, datetime, str]] = []
        for path in self._root_path.iterdir():
            if not path.is_dir():
                continue
            parsed = self._parse_date_folder(path.name)
            folders.append((parsed is not None, parsed or datetime.min, path.name))
        folders.sort(key=lambda item: (item[0], item[1], item[2].lower()), reverse=True)
        return [name for _valid, _parsed, name in folders]

    def _parse_date_folder(self, folder_name: str) -> datetime | None:
        for fmt in self._date_formats:
            try:
                return datetime.strptime(folder_name, fmt)
            except ValueError:
                continue
        return None

    def _is_analysis_folder(self, folder_name: str) -> bool:
        return bool(self._analysis_id_re.match(folder_name)) or is_test_analysis_id(folder_name)

    def _customer_options(self) -> list[str]:
        options = ["ETC"]
        for code, name in sorted(self._customer_codes.items()):
            if code.upper() in {"T"}:
                continue
            options.append(f"{code} - {name}")
        return options

    def _selected_customer_code(self) -> str:
        value = getattr(self, "_customer_var", None)
        text = value.get().strip() if value is not None else "ETC"
        if not text:
            return "ETC"
        return text.split(" - ", 1)[0].strip().upper()

    def _refresh_items(self, *, clear_cache: bool = False) -> None:
        self._sync_mode_view()
        if self._is_folder_mode():
            self._date_combo.config(state=tk.DISABLED)
            self._refresh_root_tree(clear_cache=clear_cache)
            self._sync_run_mode_controls(not self._running)
            return

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
            if self._running or self._mode_var.get() == "By Work Folder"
            else "readonly"
        )

        targets = self._scan_targets(selected_date)
        self._items = [
            self._build_item(date_name, date_path, work_id, work_path)
            for date_name, date_path, work_id, work_path in targets
        ]
        self._render_tree()
        if selected_date and self._parse_date_folder(selected_date) is None:
            self._status_label.config(text=f"Fail: invalid date folder ({selected_date})")
        elif selected_date and not self._items:
            self._status_label.config(text="Fail: no image work folder found")
        else:
            self._status_label.config(text=f"{len(self._items)} work folder(s)")
        self._sync_run_mode_controls(not self._running)

    def _refresh_item_status(
        self,
        date_name: str,
        date_path: Path,
        analysis_id: str,
        analysis_path: Path | None = None,
    ) -> None:
        if analysis_path is None:
            existing = next(
                (
                    item
                    for item in self._items
                    if item.date_name == date_name and item.analysis_id == analysis_id
                ),
                None,
            )
            analysis_path = existing.analysis_path if existing else date_path / analysis_id
        key = self._work_item_key(date_name, analysis_path)
        updated = self._build_item(date_name, date_path, analysis_id, analysis_path)
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
        self._status_label.config(text=f"{len(self._items)} work folder(s)")

    def _scan_targets(self, selected_date: str) -> list[tuple[str, Path, str, Path]]:
        targets: list[tuple[str, Path, str, Path]] = []
        date_names = self._list_date_folders()
        if self._mode_var.get() == "By Date":
            date_names = [selected_date] if selected_date else []
        for date_name in date_names:
            date_path = self._root_path / date_name
            if not date_path.is_dir():
                continue
            for work_path in self._work_folders_for_date(date_path):
                targets.append(
                    (
                        date_name,
                        date_path,
                        self._work_analysis_id(date_path, work_path),
                        work_path,
                    )
                )
        if self._mode_var.get() == "By Work Folder":
            targets.sort(key=lambda item: (item[2], item[0]))
        return targets

    def _work_folders_for_date(self, date_path: Path) -> list[Path]:
        work_folders: list[Path] = []
        stack = [
            child
            for child in sorted(date_path.iterdir(), key=lambda path: path.name.lower())
            if child.is_dir() and not self._is_output_folder(child)
        ]
        while stack:
            folder_path = stack.pop(0)
            if self._is_output_folder(folder_path):
                continue
            if self._source_images(folder_path):
                work_folders.append(folder_path)
            try:
                children = [
                    child
                    for child in sorted(folder_path.iterdir(), key=lambda path: path.name.lower())
                    if child.is_dir() and not self._is_output_folder(child)
                ]
            except OSError:
                children = []
            stack[0:0] = children
        return work_folders

    def _work_label(self, date_path: Path, work_path: Path) -> str:
        try:
            return work_path.relative_to(date_path).as_posix()
        except ValueError:
            return work_path.name

    def _work_analysis_id(self, date_path: Path, work_path: Path) -> str:
        raw = f"{date_path.name}_{self._work_label(date_path, work_path)}"
        value = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
        return value[:120] or "WORK"

    def _work_item_key(self, date_name: str, work_path: Path) -> str:
        return f"{date_name}|{work_path.resolve(strict=False)}"

    def _build_item(
        self,
        date_name: str,
        date_path: Path,
        analysis_id: str,
        analysis_path: Path,
    ) -> _WorkflowItem:
        key = self._work_item_key(date_name, analysis_path)
        work_label = self._work_label(date_path, analysis_path)
        qc_status, qc_summary = self._qc_output_status(date_name, date_path, analysis_path)
        held = self._held_qc_failures.get(key) or self._load_qc_hold_state(
            date_name,
            date_path,
            analysis_id,
            analysis_path,
        )
        csv_path = self._find_result_csv(analysis_path, analysis_id)
        cells = self._analysis_cells_from_files(analysis_id, analysis_path)
        db_ok = self._sqlite_logger.existing_prediction_cells({(analysis_id, str(cell)) for cell in cells})
        db_complete = bool(cells) and len(db_ok) == len(set(cells))
        li_salt = self._li_salt_summary(csv_path)
        correction = self._correction_summary(analysis_id)

        if held:
            failed_count = len(held.get("failures", []))
            ml_status = "WAIT"
            summary = f"QC hold: {failed_count} NG image(s)"
        elif not self._qc_allows_ml(qc_status):
            ml_status = "WAIT"
            summary = qc_summary
        elif csv_path is None:
            ml_status = "ML missing"
            summary = qc_summary if qc_status == "Warning" else "QC ready, ML result missing"
        elif not db_complete:
            ml_status = "DB missing"
            summary = "CSV exists, SQLite rows missing"
        else:
            ml_status = "DONE"
            summary = qc_summary if qc_status == "Warning" else "QC + ML + DB complete"

        return _WorkflowItem(
            key=key,
            date_name=date_name,
            date_path=date_path,
            analysis_id=analysis_id,
            analysis_path=analysis_path,
            work_label=work_label,
            qc_status="Hold" if held else qc_status,
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

    def _tree_id_for_key(self, key: str) -> str | None:
        for tree_id, item in self._item_by_tree_id.items():
            if item.key == key:
                return tree_id
        return None

    def _tree_values(self, item: _WorkflowItem) -> tuple[str, str, str, str, str, str, str]:
        return (
            item.date_name,
            item.work_label or item.analysis_id,
            item.qc_status,
            item.ml_status,
            item.li_salt,
            item.correction,
            item.summary,
        )

    def _tree_tag(self, item: _WorkflowItem) -> str:
        if item.qc_status in {"Fail", "Needed", "Hold"}:
            return "red"
        if item.qc_status == "Warning" or item.ml_status != "DONE":
            return "yellow"
        return "blue"

    @staticmethod
    def _qc_allows_ml(qc_status: str) -> bool:
        return qc_status in {"Pass", "Warning"}

    def _sort_by_column(self, column: str) -> None:
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False
        self._render_tree()

    def _sync_tree_headings(self) -> None:
        headings = {
            "date": "Date",
            "analysis_id": "Work Folder",
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
        if column == "date":
            parsed = self._parse_date_folder(item.date_name)
            return parsed or datetime.min
        if column == "analysis_id":
            return item.work_label or item.analysis_id
        if column == "qc":
            return item.qc_status
        if column == "ml":
            return item.ml_status
        if column == "li_salt":
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", item.li_salt)
            return float(match.group(0)) if match else -1.0
        if column == "correction":
            order = {"Review": 3, "Mixed": 2, "NO": 1, "-": 0}
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
        review_rows = [
            row for row in rows if int(row.get("qc_warning") or 0) == 1
        ]
        if review_rows:
            return "Review"
        return "Mixed" if corrected_rows else "NO"

    def _qc_output_status(self, date_name: str, date_path: Path, work_path: Path) -> tuple[str, str]:
        if self._parse_date_folder(date_name) is None:
            return "Fail", f"Invalid date folder: {date_name}"
        if not work_path.is_dir():
            return "Fail", f"Work folder not found: {work_path}"

        source_images = self._source_images(work_path)
        if not source_images:
            return "Fail", "No image work folder found"

        warning = self._image_count_warning(len(source_images))
        pass_count = len(self._work_pass_pngs(work_path))
        if pass_count <= 0:
            summary = "QC not run"
            if warning:
                summary = f"{summary}; {warning}"
            return "Needed", summary
        if pass_count < len(source_images):
            summary = f"_PASS count {pass_count}/{len(source_images)}"
            if warning:
                summary = f"{summary}; {warning}"
            return "Needed", summary
        if warning:
            return "Warning", warning
        return "Pass", "QC ready"

    def _qc_complete(self, date_name: str, date_path: Path, analysis_id: str) -> tuple[bool, str]:
        item = next(
            (
                candidate
                for candidate in self._items
                if candidate.date_name == date_name and candidate.analysis_id == analysis_id
            ),
            None,
        )
        work_path = item.analysis_path if item is not None else date_path / analysis_id
        qc_status, summary = self._qc_output_status(date_name, date_path, work_path)
        return self._qc_allows_ml(qc_status), summary

    def _pass_outputs_complete(self, folder_result, analysis_id: str) -> tuple[bool, list[str]]:
        mismatches: list[tuple[str, str, int, int | None]] = []
        expected_pass_count = (
            None if folder_result.status == "WARNING" else self._expected_pass_count(analysis_id)
        )
        for cell in folder_result.cells:
            for face, source_path in (("FRONT", cell.front_path), ("BACK", cell.back_path)):
                if source_path is None:
                    continue
                pass_path = self._pass_output_folder(face, source_path)
                pass_count = len(self._pass_pngs(pass_path))
                if expected_pass_count is None:
                    if pass_count <= 0:
                        mismatches.append((cell.cell_name, pass_path.name, pass_count, None))
                elif pass_count != expected_pass_count:
                    mismatches.append((cell.cell_name, pass_path.name, pass_count, expected_pass_count))
        if not mismatches:
            return True, []
        if all(pass_count <= 0 for _cell, _folder, pass_count, _expected in mismatches):
            return False, ["QC not run"]
        reasons = []
        for cell_name, folder_name, pass_count, expected in mismatches:
            if expected is None:
                reasons.append(f"{cell_name} {folder_name} not generated")
            elif pass_count <= 0:
                reasons.append(f"{cell_name} {folder_name} not generated")
            else:
                reasons.append(f"{cell_name} {folder_name} count {pass_count}/{expected}")
        return False, reasons

    def _expected_pass_count(self, analysis_id: str) -> int | None:
        expected_source = self._expected_source_count()
        if expected_source is None:
            return None
        return max(int(expected_source) - 2, 0)

    def _expected_source_count(self) -> int | None:
        customer_code = self._selected_customer_code()
        if customer_code == "ETC":
            return None
        expected_source = self._image_count_rules.get(customer_code)
        if expected_source is None:
            return None
        return int(expected_source)

    def _image_count_warning(self, actual_count: int) -> str:
        expected_count = self._expected_source_count()
        if expected_count is None or actual_count == expected_count:
            return ""
        return f"Warning: Expected {expected_count} electrodes, found {actual_count}"

    def _has_pass_png(self, folder_path: Path) -> bool:
        if not folder_path.is_dir():
            return False
        return any(
            path.is_file()
            and path.suffix.lower() == ".png"
            and not path.name.endswith("_mask.png")
            for path in folder_path.iterdir()
        )

    def _analysis_cells_from_files(self, analysis_id: str, analysis_path: Path) -> list[int]:
        if is_test_analysis_id(analysis_id):
            cells = [cell_number for cell_number, _ in find_test_source_folders(analysis_path)]
            return cells or [1]
        cells = sorted(
            {
                self._cell_number_from_name(cell_name)
                for cell_name, _face_name, _source_path in self._forced_source_folders(analysis_path)
            }
        )
        return cells or [1]

    def _analysis_cells(self, date_name: str, analysis_id: str, analysis_path: Path) -> list[int]:
        if is_test_analysis_id(analysis_id):
            return [
                cell_number
                for cell_number, _ in find_test_source_folders(analysis_path)
            ]
        folder_result = self._qc_pipeline.validate_folder(
            root_path=self._root_path,
            date_name=date_name,
            analysis_id=analysis_id,
        )
        cells: list[int] = []
        if folder_result.status == "FAIL":
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
            if not face_paths and any(
                child.path.is_file() and child.path.suffix.lower() in _IMAGE_EXTENSIONS
                for child in self._list_folder_children(cell_path)
            ):
                sources.append((cell_name, "FRONT", cell_path))
        return sources

    @staticmethod
    def _pass_output_folder(face_name: str, source_path: Path) -> Path:
        if face_name.upper() == "WORK":
            return source_path / "_PASS"
        if source_path.name.upper() == face_name.upper():
            return source_path.parent / f"{face_name}_pass"
        return source_path / f"{face_name}_pass"

    def _work_pass_folder_candidates(self, work_path: Path) -> list[Path]:
        return self._unique_paths(
            [
                work_path / "_PASS",
                work_path / "_pass",
                work_path.parent / f"{work_path.name}_PASS",
                work_path.parent / f"{work_path.name}_pass",
            ]
        )

    def _work_mask_folder_candidates(self, work_path: Path) -> list[Path]:
        return self._unique_paths(
            [
                work_path / "_MASK",
                work_path / "_mask",
                work_path.parent / f"{work_path.name}_MASK",
                work_path.parent / f"{work_path.name}_mask",
            ]
        )

    @staticmethod
    def _unique_paths(paths: list[Path]) -> list[Path]:
        unique: list[Path] = []
        seen: set[str] = set()
        for path in paths:
            key = os.path.normcase(str(path.resolve(strict=False)))
            if key in seen:
                continue
            seen.add(key)
            unique.append(path)
        return unique

    def _work_pass_pngs(self, work_path: Path) -> list[Path]:
        return self._unique_paths(
            [
                path
                for folder in self._work_pass_folder_candidates(work_path)
                for path in self._pass_pngs(folder)
            ]
        )

    def _work_mask_pngs(self, work_path: Path) -> list[Path]:
        return self._unique_paths(
            [
                path
                for folder in self._work_mask_folder_candidates(work_path)
                for path in self._mask_pngs(folder)
            ]
        )

    def _preferred_work_pass_folder(self, work_path: Path) -> Path:
        primary = self._pass_output_folder("WORK", work_path)
        if self._pass_pngs(primary):
            return primary
        for candidate in self._work_pass_folder_candidates(work_path):
            if self._pass_pngs(candidate):
                return candidate
        return primary

    def _find_result_csv(self, analysis_path: Path, analysis_id: str) -> Path | None:
        search_dirs = self._unique_paths([analysis_path, *self._work_mask_folder_candidates(analysis_path)])
        candidates = []
        for search_dir in search_dirs:
            old_path = search_dir / _RESULT_PATTERN_OLD.format(analysis_id=analysis_id)
            if old_path.exists():
                candidates.append(old_path)
            candidates.extend(search_dir.glob(_RESULT_PATTERN_NEW.format(analysis_id=analysis_id)))
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
            return f"{self._format_li_salt_percent(value)}%"
        except Exception:
            logger.exception("Failed to summarize result CSV: %s", csv_path)
            return "-"

    @staticmethod
    def _li_salt_percent_from_summary(summary: str) -> float | None:
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", summary)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    @staticmethod
    def _format_li_salt_percent(value: float) -> str:
        if value == 0:
            return "0.00"
        if abs(value) < 0.01:
            return f"{value:.4f}"
        return f"{value:.2f}"

    def _selected_items(self) -> list[_WorkflowItem]:
        selected = [self._item_by_tree_id[item_id] for item_id in self._tree.selection() if item_id in self._item_by_tree_id]
        return selected or list(self._items)

    def _selected_preview_items(self) -> list[_WorkflowItem]:
        return [
            self._item_by_tree_id[item_id]
            for item_id in self._tree.selection()
            if item_id in self._item_by_tree_id
        ]

    def _on_run_qc_toggle(self) -> None:
        if self._run_qc_var.get():
            self._only_crop_var.set(False)
        self._sync_run_mode_controls(not self._running)

    def _on_only_crop_toggle(self) -> None:
        if self._only_crop_var.get():
            self._run_qc_var.set(False)
        self._sync_run_mode_controls(not self._running)

    def _on_run_ml_toggle(self) -> None:
        self._sync_run_mode_controls(not self._running)

    def _run_workflow(self) -> None:
        if self._is_folder_mode():
            if self._path_quick_mode:
                self._run_path_folder_selection()
                return
            self._run_folder_selection()
            return
        run_ml = bool(self._run_ml_var.get())
        run_qc = bool(self._run_qc_var.get())
        only_crop = bool(self._only_crop_var.get())
        if not run_qc and not only_crop and not run_ml:
            messagebox.showinfo("Workflow", "Select QC, Only Crop, or ML before running.")
            return
        self._start_run(
            self._selected_items(),
            force_qc=False,
            run_qc=run_qc,
            only_crop=only_crop,
            run_ml=run_ml,
        )

    def _run_folder_selection(self) -> None:
        run_ml = bool(self._run_ml_var.get())
        run_crop = bool(self._run_qc_var.get()) or bool(self._only_crop_var.get())
        if not run_crop:
            if run_ml:
                messagebox.showinfo("By Folder", "Select QC or Only Crop before running ML in By Folder mode.")
            else:
                messagebox.showinfo("By Folder", "Select QC, Only Crop, or ML before running.")
            return
        selected_paths = self._selected_root_paths()
        if not selected_paths:
            messagebox.showinfo("By Folder", "Select one or more folders or image files first.")
            return
        self._start_quick_paths(
            selected_paths,
            empty_message="Select one or more folders or image files first.",
            run_ml=run_ml,
            label_prefix="Folder",
        )

    def _run_path_folder_selection(self) -> None:
        selected_paths = self._selected_root_paths()
        if not selected_paths:
            messagebox.showinfo("Quick Crop + ML", "Select one or more target folders first.")
            return
        self._start_quick_paths(
            selected_paths,
            empty_message="Select one or more target folders first.",
            run_ml=True,
            label_prefix="Quick",
        )

    def _on_model_change(self, _event=None) -> None:
        self._inference_pipeline = None
        selected = self._model_var.get().strip()
        if selected:
            self._status_label.config(text=f"Model: {selected}")

    def _start_run(
        self,
        items: list[_WorkflowItem],
        force_qc: bool,
        *,
        run_qc: bool,
        only_crop: bool,
        run_ml: bool,
    ) -> None:
        if self._running:
            return
        if not run_qc and not only_crop and not run_ml:
            return
        if not items:
            messagebox.showinfo("Workflow", "No work folders to process.")
            return

        duplicate_pairs = self._existing_pairs_for_items(items) if run_ml else set()
        skip_pairs: set[tuple[str, str]] = set()
        if duplicate_pairs:
            preview = ", ".join(f"{a} / Cell {c}" for a, c in sorted(duplicate_pairs)[:8])
            if len(duplicate_pairs) > 8:
                preview += ", ..."
            replace = messagebox.askyesno(
                "SQLite Monitoring Duplicate",
                "Monitoring DB already has rows for these work folder + cell pairs:\n\n"
                f"{preview}\n\nReplace existing SQLite rows?",
            )
            if replace:
                self._sqlite_logger.delete_prediction_cells(duplicate_pairs)
            else:
                skip_pairs = duplicate_pairs

        self._running = True
        self._preview_enabled = bool(self._preview_var.get())
        self._stop_event.clear()
        self._set_controls_enabled(False)
        self._progress.configure(maximum=max(len(items), 1), value=0)
        self._worker = threading.Thread(
            target=self._workflow_worker,
            args=(items, skip_pairs, self._preview_enabled, force_qc, run_qc, only_crop, run_ml),
            daemon=True,
        )
        self._worker.start()

    def _existing_pairs_for_items(self, items: list[_WorkflowItem]) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for item in items:
            for cell in self._analysis_cells(item.date_name, item.analysis_id, item.analysis_path):
                pairs.add((item.analysis_id, str(cell)))
        return self._sqlite_logger.existing_prediction_cells(pairs)

    def _add_quick_paths_from_entry(self) -> None:
        raw_text = self._quick_path_var.get().strip()
        if not raw_text:
            return
        paths = self._parse_quick_path_entry(raw_text)
        self._start_folder_path_load(paths)

    def _on_quick_drop(self, event) -> str:
        paths = self._parse_dropped_files(getattr(event, "data", ""))
        if paths:
            self._quick_path_var.set("; ".join(paths))
        self._start_folder_path_load(paths)
        return "break"

    def _start_folder_path_load(self, paths: list[str]) -> None:
        if self._running:
            return
        if not paths:
            messagebox.showinfo("By Folder", "Enter one or more image/folder paths first.")
            return
        self._running = True
        self._stop_event.clear()
        self._set_controls_enabled(False)
        self._current_label.config(text="Current: Loading path...")
        self._status_label.config(text="Loading path...")
        self._worker = threading.Thread(
            target=self._folder_path_load_worker,
            args=(paths,),
            daemon=True,
        )
        self._worker.start()

    def _folder_path_load_worker(self, paths: list[str]) -> None:
        message = None
        try:
            payload = self._discover_shallow_path_payload(paths)
            message = ("quick_paths_ready", payload)
        except Exception as exc:
            logger.debug("Failed to load folder path", exc_info=True)
            message = ("error", str(exc))
        finally:
            if message is not None:
                self._queue.put(message)

    def _discover_shallow_path_payload(self, paths: list[str]) -> dict[str, object]:
        resolved_paths = [path_resolver.resolve_path(path) for path in paths]
        direct_images = [
            path
            for path in resolved_paths
            if path.suffix.lower() in _IMAGE_EXTENSIONS and path.exists()
        ]
        folders = [path for path in resolved_paths if path not in direct_images]
        if len(resolved_paths) == 1 and not direct_images:
            folder = resolved_paths[0]
            self._list_folder_children(folder)
            return {"folder_root": str(folder), "images": [], "display": {}}
        display = {str(path): path.name for path in direct_images}
        if direct_images:
            return {
                "folder_root": "",
                "images": [str(path) for path in direct_images],
                "display": display,
            }
        if folders:
            return {"folder_root": str(folders[0]), "images": [], "display": {}}
        return {"folder_root": "", "images": [], "display": {}}

    def _discover_quick_image_paths(self, paths: list[str]) -> list[Path]:
        return [path for path, _display in self._discover_quick_image_entries(paths)]

    def _discover_quick_image_entries(self, paths: list[str]) -> list[tuple[Path, str]]:
        images: list[Path] = []
        display_by_path: dict[Path, str] = {}
        summaries: list[str] = []
        for raw_path in paths:
            scan = path_resolver.scan_image_files(raw_path, _IMAGE_EXTENSIONS)
            logger.debug("Quick path scan summary:\n%s", scan.summary())
            summaries.append(scan.summary())
            base_path = self._quick_display_base_for_input(raw_path)
            for image_path in scan.files:
                display_by_path[image_path] = self._quick_relative_display(image_path, base_path)
            images.extend(scan.files)
        unique_images = self._dedupe_paths(images)
        if not unique_images:
            raise RuntimeError("No supported image files found.")
        return [(image_path, display_by_path.get(image_path, image_path.name)) for image_path in unique_images]

    def _quick_display_base_for_input(self, raw_path: str) -> Path:
        base_path = path_resolver.resolve_path(raw_path)
        if base_path.suffix.lower() in _IMAGE_EXTENSIONS:
            return base_path.parent
        return base_path

    def _quick_relative_display(self, image_path: Path, base_path: Path) -> str:
        try:
            return str(image_path.relative_to(base_path))
        except ValueError:
            pass
        image_text = str(image_path).replace("/", "\\")
        base_text = str(base_path).rstrip("\\/").replace("/", "\\")
        if image_text.lower().startswith(base_text.lower() + "\\"):
            return image_text[len(base_text) + 1 :]
        return image_path.name

    def _folder_root_from_loaded_paths(self, paths: list[str], image_paths: list[Path] | None = None) -> Path | None:
        if len(paths) != 1:
            return None
        source_path = path_resolver.resolve_path(paths[0])
        if image_paths and source_path.suffix.lower() in _IMAGE_EXTENSIONS:
            return source_path.parent
        if source_path.suffix.lower() in _IMAGE_EXTENSIONS:
            return source_path.parent
        return source_path

    def _parse_dropped_files(self, data: str) -> list[str]:
        if not data:
            return []
        text = str(data).strip()
        if not text:
            return []
        if "\n" in text or "\r" in text or ";" in text:
            paths: list[str] = []
            for chunk in re.split(r"[\r\n;]+", text):
                paths.extend(self._parse_dropped_files(chunk))
            return paths
        if re.match(r"^[A-Za-z]:(?:[\\/]|[^\\/])", text):
            return [self._clean_drop_path(text)]

        raw_parts = re.findall(r"\{([^}]*)\}|([^\s]+)", text)
        if raw_parts:
            parts = [left or right for left, right in raw_parts]
        else:
            try:
                parts = list(self.tk.splitlist(text))
            except tk.TclError:
                parts = [text]

        paths: list[str] = []
        for raw in parts:
            cleaned = self._clean_drop_path(str(raw))
            if not cleaned:
                continue
            paths.append(cleaned)
        return paths

    def _parse_quick_path_entry(self, data: str) -> list[str]:
        text = data.strip()
        if not text:
            return []
        single_path = self._normalize_quick_path_text(text)
        if self._is_single_path_text(single_path):
            return [single_path]

        paths: list[str] = []
        chunks = [chunk.strip() for chunk in re.split(r"[\r\n]+", text) if chunk.strip()]
        if len(chunks) == 1 and ";" in chunks[0]:
            chunks = [chunk.strip() for chunk in chunks[0].split(";") if chunk.strip()]

        for chunk in chunks:
            normalized = self._normalize_quick_path_text(chunk)
            if self._is_single_path_text(normalized):
                paths.append(normalized)
                continue
            paths.extend(self._parse_dropped_files(chunk))
        return paths or [single_path]

    def _is_single_path_text(self, text: str) -> bool:
        return bool(
            re.match(r"^[A-Za-z]:[\\/]", text)
            or text.startswith("\\\\")
            or text.startswith("/")
        )

    def _normalize_quick_path_text(self, text: str) -> str:
        return path_resolver.clean_path(text)

    def _clean_drop_path(self, text: str) -> str:
        return path_resolver.clean_path(text)

    def _quick_path_exists(self, text: str) -> bool:
        try:
            if Path(text).exists():
                return True
        except OSError:
            pass
        if self._subprocess_path_type(text) is not None:
            return True
        return False

    def _stage_paths_to_local(self, paths: list[str]) -> Path:
        sources: list[_StageSource] = []
        for raw_path in paths:
            source_text = self._normalize_quick_path_text(str(raw_path))
            source_type = self._source_path_type(source_text)
            sources.append(
                _StageSource(
                    raw_path=source_text,
                    name=self._source_display_name(source_text),
                    is_dir=None if source_type is None else source_type == "dir",
                )
            )
        if not sources:
            raise RuntimeError("No paths to load.")

        staging_root = self._make_unique_work_dir(
            self._config_path.parent.parent / "_local_folder_inputs"
        )

        for source in sources:
            destination = self._make_unique_stage_path(staging_root, source.name)
            self._copy_path_with_subprocess(source, destination)
            self._verify_stage_copy(source, destination)
        if not self._expand_quick_paths([staging_root]):
            raise RuntimeError(
                "No supported image files were copied into the staging folder.\n"
                "Check team-room access permission or select the FRONT/BACK folder directly."
            )
        return staging_root

    def _make_unique_work_dir(self, parent: Path) -> Path:
        parent.mkdir(parents=True, exist_ok=True)
        base_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = parent / base_name
        suffix = 2
        while candidate.exists():
            candidate = parent / f"{base_name}_{suffix}"
            suffix += 1
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    def _source_path_type(self, source_text: str) -> str | None:
        try:
            path = Path(source_text)
            if path.is_dir():
                return "dir"
            if path.is_file():
                return "file"
        except OSError:
            pass
        return self._subprocess_path_type(source_text)

    def _subprocess_path_type(self, source_text: str) -> str | None:
        result = self._run_powershell_env(
            (
                "$p=$env:QCML_SOURCE_PATH; "
                "if (Test-Path -LiteralPath $p -PathType Container) { 'dir' } "
                "elseif (Test-Path -LiteralPath $p -PathType Leaf) { 'file' }"
            ),
            {"QCML_SOURCE_PATH": source_text},
        )
        if result.returncode != 0:
            return None
        value = (result.stdout or "").strip().lower()
        return value if value in {"dir", "file"} else None

    def _run_powershell_env(self, script: str, env_vars: dict[str, str]) -> subprocess.CompletedProcess:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
        env = os.environ.copy()
        env.update(env_vars)
        return subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            capture_output=True,
            text=True,
            creationflags=creationflags,
            env=env,
        )

    def _source_display_name(self, source_text: str) -> str:
        parts = [part for part in re.split(r"[\\/]+", source_text.rstrip("\\/")) if part]
        if parts and not re.fullmatch(r"[A-Za-z]:", parts[-1]):
            return parts[-1]
        drive_match = re.match(r"^([A-Za-z]):", source_text)
        if drive_match:
            return f"{drive_match.group(1)}_drive"
        return "input"

    def _unique_stage_destination(self, parent: Path, name: str) -> Path:
        return self._make_unique_stage_path(parent, name)

    def _make_unique_stage_path(self, parent: Path, name: str) -> Path:
        safe_name = re.sub(r'[<>:"/\\|?*]+', "_", name).strip() or "input"
        candidate = parent / safe_name
        suffix = 2
        while candidate.exists():
            candidate = parent / f"{safe_name}_{suffix}"
            suffix += 1
        return candidate

    def _copy_path_with_subprocess(self, source: _StageSource, destination: Path) -> None:
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        prefer_file = source.is_dir is False or source.raw_path.lower().endswith(
            tuple(_IMAGE_EXTENSIONS)
        )
        copy_errors: list[str] = []
        if not prefer_file:
            result = subprocess.run(
                [
                    "robocopy",
                    source.raw_path,
                    str(destination),
                    "/E",
                    "/NFL",
                    "/NDL",
                    "/NJH",
                    "/NJS",
                    "/NP",
                ],
                capture_output=True,
                text=True,
                creationflags=creationflags,
            )
            if result.returncode < 8:
                if self._staged_path_has_content(destination):
                    return
                copy_errors.append("robocopy completed but the staged destination is empty.")
            else:
                copy_errors.append(self._short_copy_output(result.stderr or result.stdout or ""))

        destination.parent.mkdir(parents=True, exist_ok=True)
        result = self._copy_path_with_powershell(source, destination)
        if result.returncode != 0:
            copy_errors.append(self._short_copy_output(result.stderr or result.stdout or ""))
            detail = "\n".join(error for error in copy_errors if error)
            raise RuntimeError(f"Failed to copy path: {source.raw_path}\n{detail}")

    def _copy_path_with_powershell(
        self,
        source: _StageSource,
        destination: Path,
    ) -> subprocess.CompletedProcess:
        return self._run_powershell_env(
            (
                "$src=$env:QCML_SOURCE_PATH; "
                "$dst=$env:QCML_DESTINATION_PATH; "
                "$parent=Split-Path -Parent $dst; "
                "if ($parent) { [System.IO.Directory]::CreateDirectory($parent) | Out-Null }; "
                "$folderError=$null; "
                "try { "
                "  [System.IO.Directory]::CreateDirectory($dst) | Out-Null; "
                "  $items=Get-ChildItem -LiteralPath $src -Force -ErrorAction Stop; "
                "  foreach ($item in $items) { "
                "    Copy-Item -LiteralPath $item.FullName -Destination $dst -Recurse -Force -ErrorAction Stop "
                "  }; "
                "  exit 0 "
                "} catch { $folderError=$_.Exception.Message }; "
                "try { "
                "  Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force -ErrorAction Stop; "
                "  exit 0 "
                "} catch { "
                "  Write-Error (\"Folder copy failed: \" + $folderError + \"`nPath copy failed: \" + $_.Exception.Message); "
                "  exit 1 "
                "}"
            ),
            {
                "QCML_SOURCE_PATH": source.raw_path,
                "QCML_DESTINATION_PATH": str(destination),
            },
        )

    def _staged_path_has_content(self, destination: Path) -> bool:
        if destination.is_file():
            return True
        if not destination.is_dir():
            return False
        try:
            return any(destination.iterdir())
        except OSError:
            return False

    def _verify_stage_copy(self, source: _StageSource, destination: Path) -> None:
        if destination.is_file():
            return
        if destination.is_dir():
            try:
                if any(destination.iterdir()):
                    return
            except OSError:
                return
            if source.is_dir is not False:
                raise RuntimeError(
                    f"Copied folder is empty: {source.raw_path}\n"
                    "Check access permission or select a lower folder that contains the images."
                )
            return
        raise RuntimeError(f"Copy completed but staged path was not created: {destination}")

    def _short_copy_output(self, text: str, limit: int = 1200) -> str:
        detail = str(text or "").strip()
        if len(detail) <= limit:
            return detail
        return detail[:limit].rstrip() + "..."

    def _expand_quick_paths(self, file_paths) -> list[Path]:
        images: list[Path] = []
        for raw_path in file_paths:
            images.extend(path_resolver.iter_image_files(raw_path, _IMAGE_EXTENSIONS))
        return images

    def _is_ignored_quick_source(self, path: Path) -> bool:
        return any(
            part.lower() in {"_quick_crop_ml", "__pycache__"}
            or part.lower().endswith("_pass")
            or part.lower().endswith("_mask")
            or part.lower().endswith("_masks")
            for part in path.parts
        )

    def _dedupe_paths(self, paths: list[Path]) -> list[Path]:
        unique: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            try:
                key = path.resolve(strict=False)
            except OSError:
                key = path
            if key in seen:
                continue
            seen.add(key)
            unique.append(key)
        return unique

    def _add_quick_files(self, file_paths, display_by_path: dict[Path, str] | None = None) -> None:
        display_by_path = display_by_path or {}
        added = 0
        for path in self._expand_quick_paths(file_paths):
            if path in self._quick_tree_by_path:
                continue
            display = display_by_path.get(path) or self._quick_display_by_path.get(path) or str(path)
            tree_id = self._quick_tree.insert(
                "",
                tk.END,
                values=(display, "Ready", ""),
            )
            self._quick_files.append(path)
            self._quick_tree_by_path[path] = tree_id
            self._quick_display_by_path[path] = display
            added += 1
        if added:
            self._status_label.config(text=f"Quick: {len(self._quick_files)} image(s) ready")
        elif file_paths:
            self._status_label.config(text="Quick: no supported image files added")

    def _clear_quick_files(self) -> None:
        if self._running:
            return
        for tree_id in self._quick_tree.get_children():
            self._quick_tree.delete(tree_id)
        self._quick_files = []
        self._quick_tree_by_path.clear()
        self._quick_display_by_path.clear()
        self._folder_root_override = None
        self._path_quick_mode = False
        if self._is_folder_mode():
            self._refresh_root_tree(clear_cache=True)
        self._status_label.config(text="Quick list cleared; By Folder reset to Settings root")

    def _start_quick_run(self) -> None:
        self._start_quick_files(
            self._selected_quick_files(),
            empty_message="Add one or more image files first.",
            run_ml=True,
            label_prefix="Quick",
        )

    def _selected_quick_files(self) -> list[Path]:
        selected = [
            path
            for path, tree_id in self._quick_tree_by_path.items()
            if tree_id in self._quick_tree.selection()
        ]
        return selected or list(self._quick_files)

    def _start_quick_paths(
        self,
        paths: list[Path],
        *,
        empty_message: str,
        run_ml: bool,
        label_prefix: str,
    ) -> None:
        if self._running:
            return
        if not paths:
            messagebox.showinfo("Quick Crop + ML", empty_message)
            return
        self._running = True
        self._preview_enabled = bool(self._preview_var.get())
        self._stop_event.clear()
        self._set_controls_enabled(False)
        self._progress.configure(maximum=1, value=0)
        self._current_label.config(text=f"Current: {label_prefix}: finding images...")
        self._status_label.config(text=f"{label_prefix}: finding images...")
        self._worker = threading.Thread(
            target=self._quick_paths_worker,
            args=(list(paths), self._preview_enabled, run_ml, label_prefix),
            daemon=True,
        )
        self._worker.start()

    def _quick_paths_worker(
        self,
        paths: list[Path],
        preview_enabled: bool,
        run_ml: bool,
        label_prefix: str,
    ) -> None:
        try:
            files = self._dedupe_paths(self._expand_quick_paths(paths))
            if not files:
                raise RuntimeError("No supported image files found in the selection.")
            self._quick_worker(files, preview_enabled, run_ml, label_prefix)
        except Exception as exc:
            logger.debug("Quick path expansion failed", exc_info=True)
            self._queue.put(("error", str(exc)))

    def _start_quick_files(
        self,
        files: list[Path],
        *,
        empty_message: str,
        run_ml: bool,
        label_prefix: str,
    ) -> None:
        if self._running:
            return
        files = self._dedupe_paths(files)
        if not files:
            messagebox.showinfo("Quick Crop + ML", empty_message)
            return
        self._running = True
        self._preview_enabled = bool(self._preview_var.get())
        self._stop_event.clear()
        self._set_controls_enabled(False)
        self._progress.configure(maximum=max(len(files), 1), value=0)
        self._worker = threading.Thread(
            target=self._quick_worker,
            args=(files, self._preview_enabled, run_ml, label_prefix),
            daemon=True,
        )
        self._worker.start()

    def _quick_worker(
        self,
        files: list[Path],
        preview_enabled: bool,
        run_ml: bool,
        label_prefix: str,
    ) -> None:
        success_count = 0
        fail_count = 0
        qc_failures: list[dict[str, object]] = []
        folder_rows: dict[Path, list[dict]] = {}
        try:
            if run_ml:
                self._load_quick_ml_config()
                self._clear_quick_generated_csvs(files)
            for index, image_path in enumerate(files, start=1):
                if self._stop_event.is_set():
                    break
                self._queue.put(("current", f"{label_prefix}: {image_path.name}"))
                self._queue.put(
                    (
                        "quick_status",
                        {"path": image_path, "status": "Running", "output": ""},
                    )
                )
                try:
                    result = self._run_quick_file(image_path, image_num=index, run_ml=run_ml)
                    if run_ml and "result_row" in result:
                        folder_rows.setdefault(image_path.parent, []).append(dict(result["result_row"]))
                    success_count += 1
                    self._queue.put(
                        (
                            "quick_status",
                            {
                                "path": image_path,
                                "status": "PASS",
                                "output": str(result["output_dir"]),
                            },
                        )
                    )
                    if preview_enabled:
                        if run_ml:
                            self._put_preview(
                                (
                                    result["crop_path"],
                                    result["mask_path"],
                                    f"{label_prefix} / {image_path.name}",
                                    "Crop",
                                    "ML mask",
                                )
                            )
                        else:
                            self._put_preview(
                                (
                                    image_path,
                                    result["crop_path"],
                                    f"{label_prefix} / {image_path.name}",
                                    "Source",
                                    "Crop",
                                )
                            )
                except _QuickQCHoldError as exc:
                    fail_count += 1
                    qc_failures.append(exc.failed_item)
                    logger.debug("Quick/Folder QC hold: %s", image_path, exc_info=True)
                    self._queue.put(
                        (
                            "quick_status",
                            {
                                "path": image_path,
                                "status": "FAIL - QC hold",
                                "output": str(exc),
                            },
                        )
                    )
                except Exception as exc:
                    fail_count += 1
                    logger.debug("Quick Crop + ML failed: %s", image_path, exc_info=True)
                    self._queue.put(
                        (
                            "quick_status",
                            {
                                "path": image_path,
                                "status": "FAIL",
                                "output": str(exc),
                            },
                        )
                    )
                self._queue.put(("quick_progress", index))
            if run_ml and folder_rows:
                self._write_quick_folder_csvs(folder_rows)
            if qc_failures:
                self._queue.put(
                    (
                        "quick_qc_hold",
                        {
                            "key": f"quick|{datetime.now().isoformat(timespec='microseconds')}",
                            "analysis_id": label_prefix,
                            "analysis_path": files[0].parent if files else self._root_path,
                            "batch_hold": True,
                            "failures": qc_failures,
                        },
                    )
                )
                return
            self._queue.put(
                (
                    "quick_done",
                    {
                        "success": success_count,
                        "fail": fail_count,
                        "label": label_prefix,
                        "run_ml": run_ml,
                    },
                )
            )
        except Exception as exc:
            logger.debug("Quick worker error", exc_info=True)
            self._queue.put(("error", str(exc)))

    def _run_quick_file(self, image_path: Path, *, image_num: int, run_ml: bool = True) -> dict[str, object]:
        work_dir = path_resolver.make_unique_work_dir(
            self._config_path.parent.parent / "_quick_temp",
            prefix="image_",
        )
        keep_work_dir = False
        try:
            return self._run_quick_file_from_local_copy(
                image_path,
                image_num=image_num,
                run_ml=run_ml,
                work_dir=work_dir,
            )
        except _QuickQCHoldError:
            raise
        except Exception:
            keep_work_dir = True
            raise
        finally:
            if not keep_work_dir:
                try:
                    shutil.rmtree(work_dir)
                except OSError:
                    logger.warning("Failed to remove quick temp folder: %s", work_dir, exc_info=True)

    def _run_quick_file_from_local_copy(
        self,
        image_path: Path,
        *,
        image_num: int,
        run_ml: bool,
        work_dir: Path,
    ) -> dict[str, object]:
        local_source = path_resolver.make_unique_stage_path(work_dir, image_path.name)
        copy_result = path_resolver.safe_copy_file(image_path, local_source)
        if not copy_result.success:
            raise RuntimeError(
                f"Failed to copy source image into local temp: {image_path}\n"
                f"Fallback temp folder: {work_dir}\n"
                f"{copy_result.error}"
            )

        output_dir = work_dir / "_quick_crop_ml"
        output_dir.mkdir(parents=True, exist_ok=True)
        crop_path, mask_path, csv_path = self._quick_output_paths(local_source)

        customer_code, skip_size_validation = self._quick_crop_options(image_path)
        crop_result = self._quick_crop_engine.process_image(
            image_path=local_source,
            customer_code=customer_code,
            output_path=crop_path,
            skip_size_validation=skip_size_validation,
            image_postprocess=None,
        )
        if crop_result.status != "PASS" or crop_result.output_path is None:
            reason = " | ".join(crop_result.fail_reasons) or "Crop failed"
            raise _QuickQCHoldError(
                reason,
                {
                    "id": str(image_path),
                    "image_path": image_path,
                    "output_path": self._quick_destination_for_output(image_path, crop_path),
                    "fail_reasons": list(crop_result.fail_reasons),
                    "customer_code": customer_code,
                    "skip_size_validation": skip_size_validation,
                    "image_postprocess": None,
                },
            )

        if not run_ml:
            saved_paths = self._copy_quick_outputs_to_source(
                image_path,
                [crop_result.output_path],
                fallback_dir=output_dir,
            )
            return {
                "output_dir": saved_paths[0].parent,
                "crop_path": saved_paths[0],
            }

        pipeline = self._inference_pipeline
        if pipeline is None:
            raise RuntimeError("Inference pipeline is not loaded.")
        task = InferenceTask(
            analysis_id=self._quick_analysis_id(image_path),
            analysis_path=output_dir,
            cell=1,
            face="QUICK",
            image_num=image_num,
            image_path=crop_result.output_path,
        )
        inference_result = pipeline.run_image(task, mask_path=mask_path)
        result_row = dict(inference_result.result_row)
        result_row["correction_applied"] = int(
            inference_result.monitoring_record.get("correction_applied") or 0
        )
        li_salt_percent = self._li_salt_percent_from_result_row(result_row)
        saved_crop, saved_mask = self._copy_quick_outputs_to_source(
            image_path,
            [crop_result.output_path, inference_result.mask_path or mask_path],
            fallback_dir=output_dir,
        )
        return {
            "output_dir": saved_mask.parent,
            "crop_path": saved_crop,
            "mask_path": saved_mask,
            "result_row": self._quick_csv_row(
                result_row,
                source_path=image_path,
                crop_path=saved_crop,
                mask_path=saved_mask,
            ),
        }

    def _copy_quick_outputs_to_source(
        self,
        source_image_path: Path,
        local_outputs: list[Path],
        *,
        fallback_dir: Path,
        li_salt_percent: float | None = None,
    ) -> list[Path]:
        saved_paths: list[Path] = []
        errors: list[str] = []
        for local_path in local_outputs:
            destination = self._quick_destination_for_output(
                source_image_path,
                local_path,
                li_salt_percent=li_salt_percent,
            )
            copy_result = path_resolver.safe_copy_file(local_path, destination)
            if not copy_result.success:
                errors.append(f"{local_path.name}: {copy_result.error}")
                continue
            saved_paths.append(destination)
        if errors:
            pass_dir, mask_dir = self._quick_result_dirs_for_source(source_image_path)
            raise RuntimeError(
                f"Failed to save result back to source folders: {pass_dir}, {mask_dir}\n"
                f"Fallback saved at: {fallback_dir}\n"
                + "\n".join(errors)
            )
        return saved_paths

    def _quick_destination_for_output(
        self,
        source_image_path: Path,
        local_path: Path,
        *,
        li_salt_percent: float | None = None,
    ) -> Path:
        pass_dir, mask_dir = self._quick_result_dirs_for_source(source_image_path)
        name = local_path.name.lower()
        if local_path.suffix.lower() == ".csv":
            return mask_dir / self._quick_csv_filename(source_image_path, li_salt_percent)
        if name.endswith("_mask.png"):
            return mask_dir / f"{source_image_path.stem}_mask.png"
        return pass_dir / f"{source_image_path.stem}.png"

    def _quick_result_dirs_for_source(self, source_image_path: Path) -> tuple[Path, Path]:
        return self._quick_result_dirs_for_folder(source_image_path.parent)

    def _quick_result_dirs_for_folder(self, source_folder: Path) -> tuple[Path, Path]:
        folder_name = source_folder.name
        return (
            source_folder.parent / f"{folder_name}_PASS",
            source_folder.parent / f"{folder_name}_MASK",
        )

    def _quick_result_csv_for_source(self, source_image_path: Path) -> Path | None:
        folder_csv = self._quick_folder_result_csv(source_image_path.parent)
        if folder_csv is not None:
            return folder_csv
        _pass_dir, mask_dir = self._quick_result_dirs_for_source(source_image_path)
        candidates = sorted(
            mask_dir.glob(f"{source_image_path.stem}_LI_SALT_*.csv"),
            key=lambda path: path.name.lower(),
            reverse=True,
        )
        if candidates:
            return candidates[0]
        fallback = mask_dir / f"{source_image_path.stem}_LI_SALT.csv"
        if fallback.exists():
            return fallback
        return self._quick_output_paths(source_image_path)[2]

    def _quick_result_dir_for_source(self, source_image_path: Path) -> Path:
        try:
            with open(self._config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            cfg = {}
        configured = (
            (cfg.get("quick", {}) or {}).get("result_path")
            or (cfg.get("app", {}) or {}).get("quick_result_path")
        )
        if configured:
            return path_resolver.resolve_path(configured)
        return self._quick_result_dirs_for_source(source_image_path)[0]

    def _quick_csv_filename(self, image_path: Path, li_salt_percent: float | None) -> str:
        if li_salt_percent is None:
            return f"{image_path.stem}_LI_SALT.csv"
        return f"{image_path.stem}_LI_SALT_{self._format_li_salt_percent(li_salt_percent)}%.csv"

    def _quick_folder_csv_filename(self, source_folder: Path, li_salt_percent: float | None) -> str:
        folder_name = re.sub(r"[^A-Za-z0-9_]+", "_", source_folder.name).strip("_") or "WORK"
        if li_salt_percent is None:
            return f"{folder_name}_LI-SALT.csv"
        return f"{folder_name}_LI-SALT_{self._format_li_salt_percent(li_salt_percent)}%.csv"

    def _quick_folder_result_csv(self, source_folder: Path) -> Path | None:
        _pass_dir, mask_dir = self._quick_result_dirs_for_folder(source_folder)
        folder_name = re.sub(r"[^A-Za-z0-9_]+", "_", source_folder.name).strip("_") or "WORK"
        candidates = []
        if mask_dir.is_dir():
            candidates.extend(mask_dir.glob(f"{folder_name}_LI-SALT_*.csv"))
            candidates.extend(mask_dir.glob(f"{folder_name}_LI-SALT.csv"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _li_salt_percent_from_result_row(self, result_row: dict) -> float | None:
        try:
            return float(result_row["li_salt_ratio"]) * 100.0
        except (KeyError, TypeError, ValueError):
            return None


    def _quick_crop_options(self, image_path: Path) -> tuple[str, bool]:
        for parent in image_path.parents:
            name = parent.name
            if self._analysis_id_re.match(name) and len(name) >= 2:
                return self._customer_code_for_analysis(name), False
            if is_test_analysis_id(name):
                return "T", True
        return "T", True

    def _quick_analysis_id(self, image_path: Path) -> str:
        stem = re.sub(r"[^A-Za-z0-9_]+", "_", image_path.stem).strip("_")
        return f"QUICK_{stem or 'IMAGE'}"

    def _quick_csv_row(
        self,
        result_row: dict,
        *,
        source_path: Path,
        crop_path: Path,
        mask_path: Path,
    ) -> dict:
        row = {
            "source_image": str(source_path),
            "crop_image": str(crop_path),
            "mask_image": str(mask_path),
        }
        row.update(result_row)
        return row

    def _write_quick_folder_csvs(self, folder_rows: dict[Path, list[dict]]) -> list[Path]:
        csv_paths: list[Path] = []
        for source_folder, rows in folder_rows.items():
            if not rows:
                continue
            _pass_dir, mask_dir = self._quick_result_dirs_for_folder(source_folder)
            mask_dir.mkdir(parents=True, exist_ok=True)
            average_percent = self._average_li_salt_percent_from_rows(rows)
            csv_path = mask_dir / self._quick_folder_csv_filename(source_folder, average_percent)
            pd.DataFrame(
                rows,
                columns=[
                    "source_image",
                    "crop_image",
                    "mask_image",
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
            ).to_csv(csv_path, index=False, float_format="%.6f")
            csv_paths.append(csv_path)
        return csv_paths

    def _clear_quick_generated_csvs(self, source_images: list[Path]) -> None:
        mask_dirs = {self._quick_result_dirs_for_source(path)[1] for path in source_images}
        for mask_dir in mask_dirs:
            if not mask_dir.is_dir():
                continue
            for path in mask_dir.iterdir():
                if path.is_file() and self._is_generated_li_salt_csv(path):
                    try:
                        path.unlink()
                    except OSError:
                        logger.warning("Failed to remove stale Li-salt CSV: %s", path, exc_info=True)

    def _write_quick_csv(
        self,
        csv_path: Path,
        result_row: dict,
        *,
        source_path: Path,
        crop_path: Path,
        mask_path: Path,
    ) -> None:
        row = self._quick_csv_row(
            result_row,
            source_path=source_path,
            crop_path=crop_path,
            mask_path=mask_path,
        )
        pd.DataFrame(
            [row],
            columns=[
                "source_image",
                "crop_image",
                "mask_image",
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
        ).to_csv(csv_path, index=False, float_format="%.6f")

    def _update_quick_tree_status(self, payload: dict) -> None:
        path = Path(payload.get("path", ""))
        tree_id = self._quick_tree_by_path.get(path)
        if tree_id is None:
            display = self._quick_display_by_path.get(path) or str(path)
            tree_id = self._quick_tree.insert(
                "",
                tk.END,
                values=(display, "", ""),
            )
            self._quick_tree_by_path[path] = tree_id
            if path not in self._quick_files:
                self._quick_files.append(path)
            self._quick_display_by_path[path] = display
        values = list(self._quick_tree.item(tree_id, "values"))
        if len(values) < 3:
            values = [str(path), "", ""]
        values[1] = str(payload.get("status", ""))
        values[2] = str(payload.get("output", ""))
        status = values[1].upper()
        if "FAIL" in status or "HOLD" in status:
            tags = ("quick_fail",)
        elif "RUN" in status:
            tags = ("quick_running",)
        elif "PASS" in status:
            tags = ("quick_pass",)
        else:
            tags = ()
        self._quick_tree.item(tree_id, values=values, tags=tags)
        self._quick_tree.see(tree_id)

    def _request_stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._status_label.config(text="Stopping after current work folder...")

    def _workflow_worker(
        self,
        items: list[_WorkflowItem],
        skip_pairs: set[tuple[str, str]],
        preview_enabled: bool,
        force_qc: bool,
        run_qc: bool,
        only_crop: bool,
        run_ml: bool,
    ) -> None:
        try:
            if run_ml:
                self._load_ml_config()
            processed = 0
            if run_qc:
                held_failures: list[dict[str, object]] = []
                for item in items:
                    if self._stop_event.is_set():
                        break
                    self._cleanup_runtime_artifacts(item.analysis_path)
                    self._queue.put(("current", f"{item.analysis_id}: QC"))
                    self._reset_qc_hold_for_rerun(item)
                    failures = self._run_work_folder_crop(
                        item,
                        skip_size_validation=False,
                        force_save_failed=False,
                    )
                    if failures:
                        self._store_qc_hold(item, failures)
                        held_failures.extend(failures)
                    else:
                        self._clear_qc_hold(item)
                    processed += 1
                    self._queue.put(("progress", (processed, item.date_name, item.date_path, item.analysis_id, item.analysis_path)))
                if held_failures:
                    self._queue.put(
                        (
                            "batch_qc_hold",
                            {
                                "key": f"batch|{datetime.now().isoformat(timespec='microseconds')}",
                                "analysis_id": "QC Failures",
                                "analysis_path": items[0].analysis_path if items else self._root_path,
                                "failures": held_failures,
                            },
                        )
                    )
                    return
                if not run_ml:
                    self._queue.put(("done", None))
                    return
                processed = 0
            for item in items:
                if self._stop_event.is_set():
                    break
                self._cleanup_runtime_artifacts(item.analysis_path)
                refreshed = item
                should_run_qc = run_qc
                if should_run_qc:
                    refreshed = self._build_item(item.date_name, item.date_path, item.analysis_id, item.analysis_path)
                elif only_crop:
                    self._queue.put(("current", f"{item.analysis_id}: crop"))
                    self._reset_qc_hold_for_rerun(item)
                    qc_held = self._run_only_crop_for_item(item)
                    if qc_held:
                        return
                    refreshed = self._build_item(item.date_name, item.date_path, item.analysis_id, item.analysis_path)
                    self._queue.put(("refresh", (item.date_name, item.date_path, item.analysis_id, item.analysis_path)))
                if run_ml and (force_qc or self._qc_allows_ml(refreshed.qc_status)):
                    self._queue.put(("current", f"{item.analysis_id}: ML"))
                    self._run_ml_for_item(refreshed, skip_pairs, preview_enabled, force_qc=force_qc)
                elif run_ml:
                    logger.warning(
                        "ML skipped because QC is not ready: %s / %s",
                        item.analysis_id,
                        refreshed.summary,
                    )
                processed += 1
                self._queue.put(("progress", (processed, item.date_name, item.date_path, item.analysis_id, item.analysis_path)))
            self._queue.put(("done", None))
        except Exception as exc:
            logger.exception("Workflow worker error")
            self._queue.put(("error", str(exc)))

    def _run_qc_for_item(self, item: _WorkflowItem, force_folder_qc: bool = False) -> bool:
        return self._handle_qc_failures(
            item,
            self._run_work_folder_crop(
                item,
                skip_size_validation=False,
                force_save_failed=False,
            ),
        )

    def _run_only_crop_for_item(self, item: _WorkflowItem) -> bool:
        return self._handle_qc_failures(
            item,
            self._run_work_folder_crop(
                item,
                skip_size_validation=True,
                force_save_failed=True,
            ),
        )

    def _run_work_folder_crop(
        self,
        item: _WorkflowItem,
        *,
        skip_size_validation: bool,
        force_save_failed: bool,
    ) -> list[dict[str, object]]:
        output_folder = self._pass_output_folder("WORK", item.analysis_path)
        self._qc_pipeline.clear_pass_outputs(output_folder)
        customer_code = self._selected_customer_code()
        failed_items: list[dict[str, object]] = []
        for image_path in self._source_images(item.analysis_path):
            output_path = output_folder / f"{image_path.stem}.png"
            result = self._quick_crop_engine.process_image(
                image_path=image_path,
                customer_code=customer_code,
                output_path=output_path,
                skip_size_validation=skip_size_validation,
                image_postprocess=None,
                force_save_failed=force_save_failed,
            )
            if result.status == "FAIL":
                failed_items.append(
                    {
                        "id": str(image_path),
                        "image_path": image_path,
                        "output_path": output_path,
                        "fail_reasons": list(result.fail_reasons),
                        "customer_code": customer_code,
                        "skip_size_validation": skip_size_validation,
                        "image_postprocess": None,
                    }
                )
        return failed_items

    def _handle_qc_failures(self, item: _WorkflowItem, failed_items: list[dict[str, object]]) -> bool:
        if not failed_items:
            self._clear_qc_hold(item)
            return False
        if self._stop_event.is_set():
            return True
        held = self._store_qc_hold(item, failed_items)
        self._queue.put(("qc_hold", held))
        return True

    def _store_qc_hold(self, item: _WorkflowItem, failed_items: list[dict[str, object]]) -> dict[str, object]:
        held = {
            "key": item.key,
            "date_name": item.date_name,
            "date_path": item.date_path,
            "analysis_id": item.analysis_id,
            "analysis_path": item.analysis_path,
            "failures": failed_items,
        }
        self._held_qc_failures[item.key] = held
        self._persist_qc_hold_state(held)
        return held

    def _load_qc_hold_state(
        self,
        date_name: str,
        date_path: Path,
        analysis_id: str,
        analysis_path: Path | None = None,
    ) -> dict[str, object] | None:
        analysis_path = analysis_path or date_path / analysis_id
        hold_path = self._hold_state_path(analysis_path)
        if not hold_path.exists():
            return None
        try:
            raw = json.loads(hold_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to load QC hold state: %s", hold_path, exc_info=True)
            return None

        key = self._work_item_key(date_name, analysis_path)
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
                        or self._selected_customer_code()
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
        }
        self._held_qc_failures[key] = held
        return held

    def _recover_qc_hold_from_incomplete_pass_outputs(
        self,
        date_name: str,
        date_path: Path,
        analysis_id: str,
    ) -> dict[str, object] | None:
        if is_test_analysis_id(analysis_id):
            return None
        folder_result = self._qc_pipeline.validate_folder(
            root_path=self._root_path,
            date_name=date_name,
            analysis_id=analysis_id,
        )
        if folder_result.status == "FAIL":
            return None
        expected_pass_count = self._expected_pass_count(analysis_id)
        if expected_pass_count is None:
            return None

        failures: list[dict[str, object]] = []
        customer_code = self._selected_customer_code()
        image_postprocess = self._current_image_postprocess()
        for cell in folder_result.cells:
            for face_name, source_path in (("FRONT", cell.front_path), ("BACK", cell.back_path)):
                if source_path is None:
                    continue
                output_folder = self._pass_output_folder(face_name, source_path)
                pass_count = len(self._pass_pngs(output_folder))
                if pass_count <= 0 or pass_count >= expected_pass_count:
                    continue
                for image_path in self._source_images(source_path)[1:-1]:
                    output_path = output_folder / f"{image_path.stem}.png"
                    if output_path.exists():
                        continue
                    failures.append(
                        {
                            "id": str(image_path),
                            "image_path": image_path,
                            "output_path": output_path,
                            "fail_reasons": [
                                "Recovered QC hold: pass output missing after previous QC run"
                            ],
                            "customer_code": customer_code,
                            "skip_size_validation": False,
                            "image_postprocess": image_postprocess,
                        }
                    )
        if not failures:
            return None

        held = {
            "key": f"{date_name}|{analysis_id}",
            "date_name": date_name,
            "date_path": date_path,
            "analysis_id": analysis_id,
            "analysis_path": date_path / analysis_id,
            "failures": failures,
        }
        self._held_qc_failures[str(held["key"])] = held
        self._persist_qc_hold_state(held)
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
        self._held_qc_failures.pop(item.key, None)
        self._delete_qc_hold_state({"analysis_path": item.analysis_path})

    def _hold_state_path(self, analysis_path: Path) -> Path:
        return analysis_path / _HOLD_STATE_FILENAME

    def _current_image_postprocess(self):
        return None

    def _force_crop_failed_item(self, failed_item: dict[str, object]):
        return self._process_failed_qc_item(failed_item, force=True)

    def _process_failed_qc_item(self, failed_item: dict[str, object], *, force: bool):
        return self._qc_pipeline.force_crop_failed_item(failed_item, force=force)

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
        model_dir = resolve_model_path(selected_model) if selected_model else Path(cfg["ml"]["model_path"])
        self._save_mask_images = bool(cfg["ml"].get("save_mask_images", True))
        if (
            self._inference_pipeline is None
            or self._inference_pipeline.model_dir != model_dir
            or self._inference_save_mask_images != self._save_mask_images
        ):
            self._inference_pipeline = InferencePipeline(
                self._config_path,
                model_dir=model_dir,
                save_mask_images=self._save_mask_images,
                customer_codes=self._customer_codes,
            )
            self._inference_save_mask_images = self._save_mask_images

    def _load_quick_ml_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        selected_model = self._model_var.get().strip()
        model_dir = resolve_model_path(selected_model) if selected_model else Path(cfg["ml"]["model_path"])
        self._save_mask_images = True
        if (
            self._inference_pipeline is None
            or self._inference_pipeline.model_dir != model_dir
            or self._inference_save_mask_images is not True
        ):
            self._inference_pipeline = InferencePipeline(
                self._config_path,
                model_dir=model_dir,
                save_mask_images=True,
                customer_codes=self._customer_codes,
            )
            self._inference_save_mask_images = True

    def _run_ml_for_item(
        self,
        item: _WorkflowItem,
        skip_pairs: set[tuple[str, str]],
        preview_enabled: bool,
        force_qc: bool = False,
    ) -> None:
        qc_ok, qc_summary = self._qc_complete(item.date_name, item.date_path, item.analysis_id)
        if not qc_ok and not force_qc:
            logger.warning("ML skipped because QC pass outputs are incomplete: %s / %s", item.analysis_id, qc_summary)
            self._queue.put(("refresh", (item.date_name, item.date_path, item.analysis_id, item.analysis_path)))
            return
        tasks = self._build_ml_tasks(item, force_qc=force_qc)
        self._clear_mask_outputs_for_tasks(tasks, clear_masks=self._save_mask_images)
        pipeline = self._inference_pipeline
        if pipeline is None:
            raise RuntimeError("Inference pipeline is not loaded.")
        rows: list[dict] = []
        monitoring_rows: list[dict] = []
        for task in tasks:
            if self._stop_event.is_set():
                return
            task_label = self._task_label(task)
            self._queue.put(("current", task_label))
            pipeline_task = InferenceTask(
                analysis_id=task.analysis_id,
                analysis_path=task.analysis_path,
                cell=task.cell,
                face=task.face,
                image_num=task.image_num,
                image_path=task.image_path,
                customer=task.customer,
            )
            mask_path = self._mask_dir_for_task(task) / f"{task.image_path.stem}_mask.png"
            result = pipeline.run_image(pipeline_task, mask_path=mask_path)
            result_row = result.result_row
            result_row["correction_applied"] = int(
                result.monitoring_record.get("correction_applied") or 0
            )
            rows.append(result_row)
            monitoring_rows.append(result.monitoring_record)
            if preview_enabled:
                self._put_preview(
                    (
                        task.image_path,
                        result.mask_path or task.image_path,
                        task_label,
                        "Before",
                        "After / Mask",
                    )
                )
        if not rows:
            return
        csv_path = self._write_csv(item.analysis_id, tasks, rows)
        for record in monitoring_rows:
            pair = (str(record["analysis_id"]), str(record["cell"]))
            if pair in skip_pairs:
                continue
            record["csv_path"] = str(csv_path)
            self._sqlite_logger.insert_prediction_run(record)

    def _build_ml_tasks(self, item: _WorkflowItem, force_qc: bool = False) -> list[_ImageTask]:
        tasks: list[_ImageTask] = []
        pass_path = self._preferred_work_pass_folder(item.analysis_path)
        source_index = self._source_layer_index(item.analysis_path)
        customer_code = self._selected_customer_code()
        for fallback_index, path in enumerate(self._pass_pngs(pass_path), start=1):
            image_num = source_index.get(path.stem, fallback_index)
            tasks.append(_ImageTask(item.analysis_id, item.analysis_path, 1, "WORK", image_num, path, customer_code))
        return tasks

    def _task_label(self, task: _ImageTask) -> str:
        return f"{task.analysis_id} / Cell {task.cell} / {task.face} / Layer {task.image_num}"

    def _task_key(self, task: _ImageTask) -> tuple[int, str, int]:
        return (task.cell, task.face.upper(), task.image_num)

    def _source_layer_index(self, folder_path: Path) -> dict[str, int]:
        return {
            path.stem: index
            for index, path in enumerate(self._source_images(folder_path), start=1)
        }

    def _source_images(self, folder_path: Path) -> list[Path]:
        if not self._path_is_dir(folder_path):
            return []
        try:
            return sorted(
                [
                    child.path
                    for child in self._list_folder_children(folder_path)
                    if child.is_file and child.path.suffix.lower() in _IMAGE_EXTENSIONS
                ],
                key=lambda path: path.name,
            )
        except OSError:
            return []

    def _pass_pngs(self, folder_path: Path) -> list[Path]:
        if not self._path_is_dir(folder_path):
            return []
        try:
            return sorted(
                [
                    child.path
                    for child in self._list_folder_children(folder_path)
                    if child.is_file
                    and child.path.suffix.lower() in _IMAGE_EXTENSIONS
                    and not child.path.name.lower().endswith("_mask.png")
                ],
                key=lambda path: path.name,
            )
        except OSError:
            return []

    def _mask_pngs(self, folder_path: Path) -> list[Path]:
        if not self._path_is_dir(folder_path):
            return []
        try:
            return sorted(
                [
                    child.path
                    for child in self._list_folder_children(folder_path)
                    if child.is_file
                    and child.path.suffix.lower() == ".png"
                    and child.path.name.endswith("_mask.png")
                ],
                key=lambda path: path.name,
            )
        except OSError:
            return []

    def _mask_dir_for_task(self, task: _ImageTask) -> Path:
        if is_test_analysis_id(task.analysis_id) and task.face.upper() == "TEST":
            return get_test_mask_folder_for_pass(task.image_path.parent)
        pass_folder = task.image_path.parent
        pass_folder_name = pass_folder.name
        if pass_folder_name.upper() == "_PASS":
            return pass_folder.with_name("_MASK")
        if pass_folder_name.lower().endswith("_pass"):
            return pass_folder.with_name(f"{pass_folder_name[:-5]}_MASK")
        return task.image_path.parent.with_name(f"{task.face}_mask")

    def _clear_mask_outputs_for_tasks(self, tasks: list[_ImageTask], *, clear_masks: bool) -> None:
        mask_dirs = {self._mask_dir_for_task(task) for task in tasks}
        for mask_dir in mask_dirs:
            if not mask_dir.is_dir():
                continue
            for path in mask_dir.iterdir():
                if not path.is_file():
                    continue
                if (clear_masks and path.suffix.lower() == ".png") or self._is_generated_li_salt_csv(path):
                    try:
                        path.unlink()
                    except OSError:
                        logger.warning("Failed to remove stale ML output: %s", path, exc_info=True)

    def _write_csv(self, analysis_id: str, tasks: list[_ImageTask], rows: list[dict]) -> Path:
        average_percent = self._compute_average_li_salt_percent(tasks, rows)
        csv_dir = self._mask_dir_for_task(tasks[0])
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / (
            f"{analysis_id}_LI-SALT_{self._format_li_salt_percent(average_percent)}%.csv"
        )
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
        ).to_csv(csv_path, index=False, float_format="%.6f")
        return csv_path

    @staticmethod
    def _is_generated_li_salt_csv(path: Path) -> bool:
        name = path.name.upper()
        return path.suffix.lower() == ".csv" and ("LI-SALT" in name or "LI_SALT" in name)

    def _average_li_salt_percent_from_rows(self, rows: list[dict]) -> float | None:
        values = []
        for row in rows:
            try:
                values.append(float(row["li_salt_ratio"]) * 100.0)
            except (KeyError, TypeError, ValueError):
                continue
        if not values:
            return None
        return sum(values) / len(values)

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
        return "T" if is_test_analysis_id(analysis_id) else analysis_id[1]

    def _toggle_preview(self) -> None:
        self._preview_enabled = bool(self._preview_var.get())
        if self._preview_enabled:
            self._review_label.config(text="Select a work folder to preview.")
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
        elif msg_type == "progress":
            if isinstance(payload, tuple):
                processed, date_name, date_path, analysis_id, *rest = payload
                analysis_path = Path(rest[0]) if rest else None
                self._progress.configure(value=processed)
                self._refresh_item_status(str(date_name), Path(date_path), str(analysis_id), analysis_path)
            else:
                self._progress.configure(value=payload)
                self._refresh_items()
        elif msg_type == "quick_progress":
            self._progress.configure(value=payload)
        elif msg_type == "quick_status":
            self._update_quick_tree_status(payload)
        elif msg_type == "folder_root_ready":
            self._running = False
            self._set_controls_enabled(True)
            self._current_label.config(text="Current: -")
            self._set_folder_root(Path(payload), quick_mode=True)
            self._status_label.config(text=f"By Folder loaded from local copy: {payload}")
        elif msg_type == "quick_paths_ready":
            self._running = False
            self._path_quick_mode = False
            self._set_controls_enabled(True)
            self._current_label.config(text="Current: -")
            display_by_path: dict[Path, str] = {}
            if isinstance(payload, dict):
                image_paths = [Path(path) for path in payload.get("images", [])]
                display_by_path = {
                    Path(path): str(display)
                    for path, display in payload.get("display", {}).items()
                }
                folder_root = payload.get("folder_root") or ""
            elif payload and all(isinstance(item, tuple) and len(item) == 2 for item in payload):
                image_paths = [Path(path) for path, _display in payload]
                display_by_path = {Path(path): str(display) for path, display in payload}
                folder_root = ""
            else:
                image_paths = [Path(path) for path in payload]
                folder_root = ""
            if folder_root:
                self._quick_files = []
                self._set_folder_root(Path(folder_root), quick_mode=True)
                self._status_label.config(
                    text=f"By Folder: {folder_root}"
                )
                return
            self._add_quick_files(image_paths, display_by_path=display_by_path)
            if image_paths:
                self._status_label.config(text=f"Quick: {len(image_paths)} image(s) loaded")
            else:
                self._status_label.config(text="Quick: no supported image files found")
        elif msg_type == "refresh":
            if isinstance(payload, tuple):
                date_name, date_path, analysis_id, *rest = payload
                analysis_path = Path(rest[0]) if rest else None
                self._refresh_item_status(str(date_name), Path(date_path), str(analysis_id), analysis_path)
            else:
                self._refresh_items(clear_cache=True)
        elif msg_type == "qc_hold":
            self._handle_qc_hold(payload)
        elif msg_type == "batch_qc_hold":
            self._handle_batch_qc_hold(payload)
        elif msg_type == "quick_qc_hold":
            self._handle_quick_qc_hold(payload)
        elif msg_type == "qc_ng_action_done":
            self._handle_qc_ng_action_done(payload)
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
            self._set_controls_enabled(True)
            self._current_label.config(text="Current: -")
            self._clear_folder_scan_cache()
            self._refresh_items(clear_cache=True)
            self._show_selected_preview()
            self._status_label.config(text="Workflow complete")
        elif msg_type == "quick_done":
            self._running = False
            self._set_controls_enabled(True)
            self._current_label.config(text="Current: -")
            self._clear_folder_scan_cache()
            success = int(payload.get("success", 0))
            fail = int(payload.get("fail", 0))
            label = str(payload.get("label") or "Quick")
            mode = "crop + ML" if bool(payload.get("run_ml", True)) else "crop"
            if self._is_folder_mode():
                self._refresh_root_tree(clear_cache=True)
            self._status_label.config(text=f"{label} {mode} complete: {success} pass, {fail} fail")
        elif msg_type == "error":
            self._running = False
            self._set_controls_enabled(True)
            messagebox.showerror("Workflow Error", payload)
            self._refresh_items()
            self._status_label.config(text=f"Error: {payload}")

    def _handle_qc_hold(self, payload: dict) -> None:
        key = str(payload.get("key", ""))
        if not key:
            return
        self._running = False
        self._set_controls_enabled(True)
        self._current_label.config(text="Current: -")
        failures = list(payload.get("failures", []))
        held = {
            "key": key,
            "date_name": payload.get("date_name"),
            "date_path": payload.get("date_path"),
            "analysis_id": payload.get("analysis_id"),
            "analysis_path": payload.get("analysis_path"),
            "failures": failures,
        }
        self._held_qc_failures[key] = held
        self._persist_qc_hold_state(held)
        analysis_id = payload.get("analysis_id", "-")
        self._status_label.config(
            text=(
                f"{analysis_id}: QC hold ({len(failures)} NG image(s)) - "
                "choose Retake done or Crop anyway"
            )
        )
        self._refresh_item_status(
            str(payload.get("date_name")),
            Path(payload.get("date_path")),
            str(payload.get("analysis_id")),
            Path(payload.get("analysis_path")),
        )
        self._show_held_qc_preview(key)

    def _handle_batch_qc_hold(self, payload: dict) -> None:
        key = str(payload.get("key", ""))
        if not key:
            return
        self._running = False
        self._set_controls_enabled(True)
        self._current_label.config(text="Current: -")
        failures = list(payload.get("failures", []))
        held = {
            "key": key,
            "analysis_id": payload.get("analysis_id", "QC Failures"),
            "analysis_path": payload.get("analysis_path") or self._root_path,
            "batch_hold": True,
            "failures": failures,
        }
        self._held_qc_failures[key] = held
        self._status_label.config(
            text=(
                f"QC hold ({len(failures)} NG image(s)) - "
                "review QC Failures"
            )
        )
        self._refresh_items(clear_cache=True)
        self._show_held_qc_preview(key)

    def _handle_quick_qc_hold(self, payload: dict) -> None:
        key = str(payload.get("key", ""))
        if not key:
            return
        self._running = False
        self._set_controls_enabled(True)
        self._current_label.config(text="Current: -")
        quick_path_value = payload.get("quick_path")
        quick_path = Path(quick_path_value) if quick_path_value else None
        failures = list(payload.get("failures", []))
        held = {
            "key": key,
            "analysis_id": payload.get("analysis_id", "Folder"),
            "analysis_path": payload.get("analysis_path")
            or (quick_path.parent if quick_path is not None else self._root_path),
            "quick_path": quick_path,
            "batch_hold": bool(payload.get("batch_hold")),
            "failures": failures,
        }
        self._held_qc_failures[key] = held
        self._status_label.config(
            text=(
                f"{held['analysis_id']}: QC hold ({len(failures)} NG image(s)) - "
                "choose Retake done or Crop anyway"
            )
        )
        if quick_path is not None:
            self._update_quick_tree_status(
                {
                    "path": quick_path,
                    "status": "FAIL - QC hold",
                    "output": "choose Retake done or Crop anyway",
                }
            )
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
        quick_path = Path(held.get("quick_path", "")) if held.get("quick_path") else None
        batch_hold = bool(held.get("batch_hold"))
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

        if quick_path is not None:
            if failures:
                self._held_qc_failures[held_key] = held
                self._update_quick_tree_status(
                    {
                        "path": quick_path,
                        "status": "FAIL - QC hold",
                        "output": " | ".join(fail_reasons) or "QC failed",
                    }
                )
                self._show_held_qc_preview(held_key)
            else:
                self._held_qc_failures.pop(held_key, None)
                output_path = Path(payload.get("output_path", ""))
                self._update_quick_tree_status(
                    {
                        "path": quick_path,
                        "status": "PASS",
                        "output": str(output_path.parent if output_path else ""),
                    }
                )
                self._show_selected_preview()
            return

        if batch_hold:
            self._sync_related_qc_holds(failure_id, status, fail_reasons)
            failure_path = Path(failure_id)
            if failure_path in self._quick_tree_by_path:
                if status == "PASS":
                    output_path = Path(payload.get("output_path", ""))
                    self._update_quick_tree_status(
                        {
                            "path": failure_path,
                            "status": "PASS",
                            "output": str(output_path.parent if output_path else ""),
                        }
                    )
                else:
                    self._update_quick_tree_status(
                        {
                            "path": failure_path,
                            "status": "FAIL - QC hold",
                            "output": " | ".join(fail_reasons) or "QC failed",
                        }
                    )
            if failures:
                self._held_qc_failures[held_key] = held
                self._show_held_qc_preview(held_key)
            else:
                self._held_qc_failures.pop(held_key, None)
                self._set_qc_fail_entries([])
                self._show_selected_preview()
            if self._items:
                self._refresh_items(clear_cache=True)
            return

        if failures:
            self._held_qc_failures[held_key] = held
            self._persist_qc_hold_state(held)
            self._refresh_item_status(
                str(held.get("date_name")),
                Path(held.get("date_path")),
                str(held.get("analysis_id")),
                Path(held.get("analysis_path")),
            )
            self._show_held_qc_preview(held_key)
        else:
            self._held_qc_failures.pop(held_key, None)
            self._delete_qc_hold_state(held)
            self._refresh_item_status(
                str(held.get("date_name")),
                Path(held.get("date_path")),
                str(held.get("analysis_id")),
                Path(held.get("analysis_path")),
            )
            self._show_selected_preview()

    def _sync_related_qc_holds(
        self,
        failure_id: str,
        status: str,
        fail_reasons: list[str],
    ) -> None:
        for key, held in list(self._held_qc_failures.items()):
            if held.get("batch_hold") or held.get("quick_path"):
                continue
            failures = list(held.get("failures", []))
            if not any(str(failure.get("id")) == failure_id for failure in failures):
                continue
            if status == "PASS":
                failures = [
                    failure
                    for failure in failures
                    if str(failure.get("id")) != failure_id
                ]
            else:
                for failure in failures:
                    if str(failure.get("id")) == failure_id:
                        failure["fail_reasons"] = fail_reasons
                        break
            if failures:
                held["failures"] = failures
                self._held_qc_failures[key] = held
                self._persist_qc_hold_state(held)
            else:
                self._held_qc_failures.pop(key, None)
                self._delete_qc_hold_state(held)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        combo_state = "readonly" if enabled else tk.DISABLED
        self._refresh_btn.config(state=state)
        self._sync_run_mode_controls(enabled)
        self._quick_run_btn.config(state=state)
        self._quick_clear_btn.config(state=state)
        self._quick_path_add_btn.config(state=state)
        self._quick_path_entry.config(state=state)
        self._stop_btn.config(state=tk.DISABLED if enabled else tk.NORMAL)
        self._preview_check.config(state=state)
        self._sync_view_mode_options()
        self._mode_combo.config(state=combo_state)
        self._model_combo.config(state=combo_state)
        self._customer_combo.config(state=combo_state)
        self._date_combo.config(
            state=tk.DISABLED
            if self._mode_var.get() == "By Work Folder" or self._is_folder_mode() or not enabled
            else "readonly"
        )
        self._tree.config(selectmode="extended")
        self._root_tree.config(selectmode="extended" if enabled else "none")
        self._sync_review_nav_state()
        self._sync_run_button_text()

    def _sync_run_mode_controls(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        if self._run_qc_var.get() and self._only_crop_var.get():
            self._only_crop_var.set(False)
        self._run_qc_check.config(state=state)
        only_crop_state = tk.DISABLED if (not enabled or self._run_qc_var.get()) else tk.NORMAL
        self._only_crop_check.config(state=only_crop_state)
        self._run_ml_check.config(state=state)
        has_run_option = (
            bool(self._run_qc_var.get())
            or bool(self._only_crop_var.get())
            or bool(self._run_ml_var.get())
        )
        self._run_btn.config(state=tk.NORMAL if enabled and has_run_option else tk.DISABLED)
        self._sync_run_button_text()

    def _sync_run_button_text(self) -> None:
        if self._is_folder_mode() and self._path_quick_mode:
            self._run_btn.config(text="Run Quick Crop + ML")
            return
        self._run_btn.config(text="Run")

    def _sync_path_current_label(self) -> None:
        if not hasattr(self, "_path_current_label"):
            return
        label = f"Current: {self._current_folder_root()}"
        if self._path_quick_mode:
            label += " (Quick Crop + ML)"
        self._path_current_label.config(text=label)

    def _show_selected_preview(self) -> None:
        selected = self._selected_preview_items()
        if not selected:
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
        if not self._qc_allows_ml(item.qc_status) or item.ml_status != "DONE":
            self._review_entries = []
            self._review_index = 0
            self._review_label.config(text=f"{item.work_label or item.analysis_id} / Complete QC + ML first.")
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
        self._set_qc_fail_entries(self._review_entries)
        if not self._review_entries:
            self._review_label.config(text=f"{held.get('analysis_id', '-')} / No NG image found.")
            self._review_count_label.config(text="")
            self._before_canvas.delete("all")
            self._after_canvas.delete("all")
            self._sync_review_nav_state()
            return
        self._show_review_entry()

    def _set_qc_fail_entries(self, entries: list[dict[str, object]]) -> None:
        self._qc_fail_entries = list(entries)
        self._qc_fail_index = 0
        if not hasattr(self, "_qc_fail_tree"):
            return
        for item_id in self._qc_fail_tree.get_children():
            self._qc_fail_tree.delete(item_id)
        self._qc_fail_entry_by_tree_id.clear()
        for entry in self._qc_fail_entries:
            failure = dict(entry.get("failure", {}))
            image_path = Path(failure.get("image_path", ""))
            reason = " | ".join(str(reason) for reason in failure.get("fail_reasons", []))
            item_id = self._qc_fail_tree.insert(
                "",
                tk.END,
                values=(image_path.name, reason or "QC failed"),
                tags=("qc_fail",),
            )
            self._qc_fail_entry_by_tree_id[item_id] = entry
        if self._qc_fail_entries:
            first_id = self._qc_fail_tree.get_children()[0]
            self._qc_fail_tree.selection_set(first_id)
            self._qc_fail_tree.see(first_id)
            if hasattr(self, "_review_notebook") and hasattr(self, "_qc_fail_tab"):
                self._review_notebook.select(self._qc_fail_tab)
            self._show_qc_fail_entry()
        else:
            self._qc_fail_count_label.config(text="")
            self._qc_fail_before_canvas.delete("all")
            self._qc_fail_after_canvas.delete("all")
            self._sync_qc_fail_nav_state()

    def _on_qc_fail_select(self, _event=None) -> None:
        if not hasattr(self, "_qc_fail_tree"):
            return
        selected = self._qc_fail_tree.selection()
        if not selected:
            return
        entry = self._qc_fail_entry_by_tree_id.get(selected[0])
        if entry not in self._qc_fail_entries:
            return
        self._qc_fail_index = self._qc_fail_entries.index(entry)
        self._show_qc_fail_entry()

    def _show_qc_fail_entry(self) -> None:
        if not self._qc_fail_entries or not hasattr(self, "_qc_fail_before_canvas"):
            return
        self._qc_fail_index = max(0, min(self._qc_fail_index, len(self._qc_fail_entries) - 1))
        entry = self._qc_fail_entries[self._qc_fail_index]
        self._qc_fail_count_label.config(
            text=f"{self._qc_fail_index + 1} / {len(self._qc_fail_entries)}"
        )
        tree_ids = self._qc_fail_tree.get_children()
        if tree_ids:
            self._qc_fail_tree.selection_set(tree_ids[self._qc_fail_index])
            self._qc_fail_tree.see(tree_ids[self._qc_fail_index])
        self._draw_image(
            self._qc_fail_before_canvas,
            entry.get("before"),
            str(entry.get("before_title", "NG image")),
        )
        self._draw_image(
            self._qc_fail_after_canvas,
            entry.get("after"),
            str(entry.get("after_title", "No pass output")),
        )
        self._sync_qc_fail_nav_state()

    def _show_next_qc_fail(self) -> None:
        if self._qc_fail_index < len(self._qc_fail_entries) - 1:
            self._qc_fail_index += 1
            self._show_qc_fail_entry()

    def _show_previous_qc_fail(self) -> None:
        if self._qc_fail_index > 0:
            self._qc_fail_index -= 1
            self._show_qc_fail_entry()

    def _normalize_preview_payload(self, payload) -> tuple[object, object, str, str, str]:
        if isinstance(payload, tuple) and len(payload) == 5:
            before, after, label, before_title, after_title = payload
            return before, after, str(label), str(before_title), str(after_title)
        before, after, label = payload
        return before, after, str(label), "Before", "After / Mask"

    def _build_review_entries(self, item: _WorkflowItem) -> list[dict[str, object]]:
        tasks = self._build_existing_ml_tasks(item)
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

    def _build_existing_ml_tasks(self, item: _WorkflowItem) -> list[_ImageTask]:
        tasks: list[_ImageTask] = []
        pass_path = self._preferred_work_pass_folder(item.analysis_path)
        source_index = self._source_layer_index(item.analysis_path)
        customer_code = self._selected_customer_code()
        for fallback_index, path in enumerate(self._pass_pngs(pass_path), start=1):
            image_num = source_index.get(path.stem, fallback_index)
            tasks.append(
                _ImageTask(
                    item.analysis_id,
                    item.analysis_path,
                    1,
                    "WORK",
                    image_num,
                    path,
                    customer_code,
                )
            )
        return tasks

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
        ng_enabled = current_entry is not None and current_entry.get("type") == "qc_ng"
        self._retake_review_btn.config(state=tk.NORMAL if ng_enabled else tk.DISABLED)
        self._crop_review_btn.config(state=tk.NORMAL if ng_enabled else tk.DISABLED)
        self._sync_qc_fail_nav_state()

    def _sync_qc_fail_nav_state(self) -> None:
        if not hasattr(self, "_retake_qc_fail_btn"):
            return
        has_entries = bool(self._qc_fail_entries)
        self._prev_qc_fail_btn.config(
            state=tk.NORMAL if has_entries and self._qc_fail_index > 0 else tk.DISABLED
        )
        self._next_qc_fail_btn.config(
            state=tk.NORMAL
            if has_entries and self._qc_fail_index < len(self._qc_fail_entries) - 1
            else tk.DISABLED
        )
        self._retake_qc_fail_btn.config(state=tk.NORMAL if has_entries else tk.DISABLED)
        self._crop_qc_fail_btn.config(state=tk.NORMAL if has_entries else tk.DISABLED)

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
        entry = self._current_review_entry()
        self._process_qc_ng_entry(entry, force=force)

    def _process_current_qc_fail_ng(self, *, force: bool) -> None:
        entry = None
        if self._qc_fail_entries:
            index = max(0, min(self._qc_fail_index, len(self._qc_fail_entries) - 1))
            entry = self._qc_fail_entries[index]
        self._process_qc_ng_entry(entry, force=force)

    def _process_qc_ng_entry(self, entry: dict[str, object] | None, *, force: bool) -> None:
        if entry is None or entry.get("type") != "qc_ng":
            return
        failure = dict(entry.get("failure", {}))
        held_key = str(entry.get("held_key", ""))
        failure_id = str(entry.get("failure_id", ""))
        if not failure or not held_key or not failure_id:
            return
        self._retake_review_btn.config(state=tk.DISABLED)
        self._crop_review_btn.config(state=tk.DISABLED)
        if hasattr(self, "_retake_qc_fail_btn"):
            self._retake_qc_fail_btn.config(state=tk.DISABLED)
            self._crop_qc_fail_btn.config(state=tk.DISABLED)
        action = "Crop anyway" if force else "Retake done"
        self._status_label.config(text=f"{action}: {Path(failure['image_path']).name}")

        def worker() -> None:
            try:
                result = self._process_failed_qc_item(failure, force=force)
                self._queue.put(
                    (
                        "qc_ng_action_done",
                        {
                            "held_key": held_key,
                            "failure_id": failure_id,
                            "status": result.status,
                            "fail_reasons": list(result.fail_reasons),
                            "output_path": str(result.output_path or failure.get("output_path", "")),
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
