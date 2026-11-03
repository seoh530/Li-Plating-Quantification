import logging
import json
import queue
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
import yaml

from modules.ml.feature_extractor import FeatureExtractor
from modules.ml.background_normalizer import BackgroundLABNormalizer
from modules.ml.drift_detector import BackgroundDrift, DriftDetector
from modules.ml.inference import InferenceEngine, LABEL_BACKGROUND
from modules.ml.model_registry import resolve_model_path
from modules.ml.quantifier import Quantifier
from modules.qc.folder_qc import FolderQC
from modules.storage.sqlite_logger import SQLitePredictionLogger
from modules.utils.analysis_id import (
    find_test_source_folders,
    get_test_mask_folder_for_pass,
    get_test_output_folders,
    is_test_analysis_id,
)
from modules.utils.image_io import quiet_imread

logger = logging.getLogger(__name__)

_C_FAIL = "#cc0000"
_C_MUTED = "#777777"
_C_NORMAL = "#000000"

_RESULT_PATTERN_OLD = "{analysis_id}_result.csv"
_RESULT_PATTERN_NEW = "{analysis_id}_LI-SALT_*.csv"
_ISSUE_REPORT_FILE = "issue_report.json"


@dataclass(frozen=True)
class _DateOption:
    label: str
    folder_name: str
    path: Path
    parsed: datetime


@dataclass(frozen=True)
class _AnalysisOption:
    run_key: str
    date_name: str
    date_label: str
    analysis_id: str
    analysis_path: Path
    enabled: bool
    existing_result: Path | None
    disabled_reason: str | None


@dataclass(frozen=True)
class _ImageTask:
    run_key: str
    date_name: str
    analysis_id: str
    analysis_path: Path
    cell: int
    face: str
    image_num: int
    image_path: Path


class MLTab(ttk.Frame):
    """Manual ML batch runner for QC-ready PhotoJig images."""

    def __init__(self, parent, config_path: str | Path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._load_config()

        self._folder_qc = FolderQC(config_path)
        self._feature_extractor = FeatureExtractor(config_path)
        self._inference: InferenceEngine | None = None
        self._quantifier = Quantifier(config_path)
        self._sqlite_logger = SQLitePredictionLogger(config_path)

        self._queue: queue.Queue = queue.Queue()
        self._analysis_vars: dict[str, tk.BooleanVar] = {}
        self._analysis_checks: list[ttk.Checkbutton] = []
        self._analysis_options: list[_AnalysisOption] = []
        self._option_by_key: dict[str, _AnalysisOption] = {}
        self._date_values: list[_DateOption] = []
        self._progress_rows: dict[tuple[str, int], str] = {}
        self._cell_progress: dict[tuple[str, int], dict[str, int | str]] = {}
        self._running = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._total_elapsed = 0.0
        self._processed_images = 0
        self._save_mask_images = True
        self._normalization_skipped_reason: str | None = None
        self._normalization_enabled = False
        self._normalization_cfg: dict = {}
        self._drift_detector: DriftDetector | None = None
        self._background_normalizer: BackgroundLABNormalizer | None = None
        self._issue_report_cache: dict[Path, dict[str, dict]] = {}

        self._build_ui()
        self._load_dates()
        self._poll_queue()

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        self._root_path = Path(cfg["app"]["root_path"])
        self._analysis_id_re = re.compile(cfg["folder_qc"]["analysis_id_pattern"])
        self._date_formats = list(cfg["folder_qc"]["date_formats"])
        self._customer_codes = dict(cfg["folder_qc"].get("customer_codes", {}))
        self._image_count_rules = dict(cfg["folder_qc"].get("image_count_rules", {}))
        self._valid_image_extensions = set(
            cfg["folder_qc"].get(
                "valid_image_extensions",
                [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
            )
        )
        self._normalization_cfg = dict(cfg.get("normalization", {}) or {})
        self._low_confidence_threshold = float(
            self._normalization_cfg.get("low_confidence_threshold", 0.60)
        )

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky=tk.EW, padx=10, pady=(10, 6))
        header.columnconfigure(4, weight=1)

        ttk.Label(header, text="Date").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        self._run_btn = ttk.Button(header, text="Run ML", command=self._run_ml)
        self._run_btn.grid(row=0, column=1, sticky=tk.W)
        self._stop_btn = ttk.Button(
            header,
            text="Stop",
            command=self._request_stop,
            state=tk.DISABLED,
        )
        self._stop_btn.grid(row=0, column=2, sticky=tk.W, padx=(6, 0))
        self._force_run_var = tk.BooleanVar(value=False)
        self._force_run_check = ttk.Checkbutton(
            header,
            text="Force run",
            variable=self._force_run_var,
            command=self._on_force_run_change,
        )
        self._force_run_check.grid(row=0, column=3, sticky=tk.W, padx=(8, 0))
        self._date_summary_label = ttk.Label(header, text="", foreground=_C_MUTED)
        self._date_summary_label.grid(row=0, column=4, sticky=tk.W, padx=(12, 0))
        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky=tk.NSEW, padx=10)
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=0)
        body.rowconfigure(1, weight=1)

        selectors = ttk.Frame(body)
        selectors.grid(row=0, column=0, rowspan=2, sticky=tk.NS, padx=(0, 12))
        selectors.columnconfigure(0, weight=1)
        selectors.rowconfigure(1, weight=1)
        selectors.rowconfigure(3, weight=1)

        date_frame = ttk.LabelFrame(selectors, text="Dates", padding=8)
        date_frame.grid(row=0, column=0, sticky=tk.NSEW)
        date_frame.columnconfigure(0, weight=1)
        date_frame.rowconfigure(0, weight=1)
        self._date_listbox = tk.Listbox(
            date_frame,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=8,
            width=18,
        )
        self._date_listbox.grid(row=0, column=0, sticky=tk.NSEW)
        date_scroll = ttk.Scrollbar(
            date_frame,
            orient=tk.VERTICAL,
            command=self._date_listbox.yview,
        )
        date_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._date_listbox.configure(yscrollcommand=date_scroll.set)
        self._date_listbox.bind("<<ListboxSelect>>", self._on_date_change)

        analysis_frame = ttk.LabelFrame(selectors, text="Analysis IDs", padding=8)
        analysis_frame.grid(row=1, column=0, sticky=tk.NSEW, pady=(10, 0))
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(analysis_frame, highlightthickness=0, height=240)
        self._canvas.grid(row=0, column=0, sticky=tk.NSEW)
        analysis_scroll = ttk.Scrollbar(
            analysis_frame,
            orient=tk.VERTICAL,
            command=self._canvas.yview,
        )
        analysis_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._canvas.configure(yscrollcommand=analysis_scroll.set)

        self._check_frame = ttk.Frame(self._canvas)
        self._canvas_window = self._canvas.create_window(
            (0, 0),
            window=self._check_frame,
            anchor=tk.NW,
        )
        self._check_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        progress_frame = ttk.LabelFrame(body, text="Run Progress", padding=8)
        progress_frame.grid(row=0, column=1, rowspan=2, sticky=tk.NSEW)
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.rowconfigure(0, weight=1)

        columns = ("analysis_id", "cell_num", "status")
        self._progress_tree = ttk.Treeview(
            progress_frame,
            columns=columns,
            show="headings",
            height=14,
        )
        self._progress_tree.heading("analysis_id", text="Analysis ID")
        self._progress_tree.heading("cell_num", text="Cell Num")
        self._progress_tree.heading("status", text="Status")
        self._progress_tree.column("analysis_id", width=220, anchor=tk.W)
        self._progress_tree.column("cell_num", width=90, anchor=tk.W)
        self._progress_tree.column("status", width=220, anchor=tk.W)
        self._progress_tree.grid(row=0, column=0, sticky=tk.NSEW)

        tree_scroll = ttk.Scrollbar(
            progress_frame,
            orient=tk.VERTICAL,
            command=self._progress_tree.yview,
        )
        tree_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._progress_tree.configure(yscrollcommand=tree_scroll.set)

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky=tk.EW, padx=10, pady=(8, 10))
        footer.columnconfigure(0, weight=1)

        self._progress = ttk.Progressbar(footer, mode="determinate", maximum=1, value=0)
        self._progress.grid(row=0, column=0, sticky=tk.EW)

        self._progress_label = ttk.Label(footer, text="Progress: 0/0 images")
        self._progress_label.grid(row=1, column=0, sticky=tk.W, pady=(4, 0))

        self._current_label = ttk.Label(footer, text="Current: -")
        self._current_label.grid(row=2, column=0, sticky=tk.W, pady=(2, 0))

        self._avg_time_label = ttk.Label(footer, text="Avg / image: -")
        self._avg_time_label.grid(row=3, column=0, sticky=tk.W, pady=(2, 0))

        self._status_label = ttk.Label(footer, text="", foreground=_C_NORMAL)
        self._status_label.grid(row=4, column=0, sticky=tk.W, pady=(4, 0))

    def _on_canvas_resize(self, event) -> None:
        self._canvas.itemconfigure(self._canvas_window, width=event.width)

    def reload_settings(self) -> None:
        if self._running:
            messagebox.showwarning(
                "Settings",
                "ML is running. Stop or wait for completion before reloading settings.",
            )
            return

        self._load_config()
        self._folder_qc = FolderQC(self._config_path)
        self._feature_extractor = FeatureExtractor(self._config_path)
        self._quantifier = Quantifier(self._config_path)
        self._sqlite_logger = SQLitePredictionLogger(self._config_path)
        self._inference = None
        self._issue_report_cache.clear()
        self._load_dates()
        self._status_label.config(text=f"Settings reloaded: {self._root_path}", foreground=_C_NORMAL)

    def _load_dates(self) -> None:
        self._date_values = self._list_date_folders()
        self._date_listbox.delete(0, tk.END)
        for date_option in self._date_values:
            self._date_listbox.insert(tk.END, date_option.label)

        default_index = self._pick_default_date_index()
        if default_index is not None:
            self._date_listbox.selection_set(default_index)
            self._date_listbox.see(default_index)

        self._refresh_date_summary()
        self._refresh_analysis_list()

    def _pick_default_date_index(self) -> int | None:
        if not self._date_values:
            return None
        today_str = date.today().strftime("%Y-%m-%d")
        for idx, value in enumerate(self._date_values):
            if value.label == today_str:
                return idx
        return 0

    def _list_date_folders(self) -> list[_DateOption]:
        if not self._root_path.exists():
            return []

        dated: list[_DateOption] = []
        for path in self._root_path.iterdir():
            if not path.is_dir():
                continue
            parsed = self._parse_date_folder(path.name)
            if parsed is not None:
                canonical = parsed.strftime("%Y-%m-%d")
                actual_path = self._normalize_date_folder(path, canonical)
                dated.append(
                    _DateOption(
                        label=canonical,
                        folder_name=actual_path.name,
                        path=actual_path,
                        parsed=parsed,
                    )
                )

        dated.sort(key=lambda item: (item.parsed, item.folder_name), reverse=True)
        return dated

    def _normalize_date_folder(self, path: Path, canonical_name: str) -> Path:
        if path.name == canonical_name:
            return path

        target = path.with_name(canonical_name)
        if target.exists():
            logger.warning(
                "Date folder normalization skipped because target exists: %s -> %s",
                path,
                target,
            )
            return path

        try:
            path.rename(target)
            logger.info("Date folder normalized: %s -> %s", path, target)
            return target
        except OSError:
            logger.exception("Date folder normalization failed: %s -> %s", path, target)
            return path

    def _parse_date_folder(self, folder_name: str) -> datetime | None:
        for fmt in self._date_formats:
            try:
                return datetime.strptime(folder_name, fmt)
            except ValueError:
                continue
        return None

    def _get_selected_dates(self) -> list[_DateOption]:
        return [self._date_values[i] for i in self._date_listbox.curselection()]

    def _refresh_date_summary(self) -> None:
        selected = self._get_selected_dates()
        if not selected:
            self._date_summary_label.config(text="No dates selected")
            return
        preview = ", ".join(date_option.label for date_option in selected[:3])
        if len(selected) > 3:
            preview += ", ..."
        self._date_summary_label.config(
            text=f"Selected dates: {len(selected)}  ({preview})"
        )

    def _on_date_change(self, _event=None) -> None:
        if self._running:
            return
        self._refresh_date_summary()
        self._refresh_analysis_list()

    def _on_force_run_change(self) -> None:
        if self._running:
            return
        self._refresh_analysis_list()

    def _refresh_analysis_list(self) -> None:
        for check in self._analysis_checks:
            check.destroy()
        self._analysis_checks.clear()
        self._analysis_vars.clear()
        self._analysis_options.clear()
        self._option_by_key.clear()

        selected_dates = self._get_selected_dates()
        if not selected_dates:
            self._status_label.config(text="Select at least one date.", foreground=_C_MUTED)
            return

        options: list[_AnalysisOption] = []
        multi_date = len(selected_dates) > 1
        force_run = bool(self._force_run_var.get())
        for date_option in selected_dates:
            date_path = date_option.path
            if not date_path.is_dir():
                continue
            for path in sorted(
                [
                    p for p in date_path.iterdir()
                    if p.is_dir() and self._is_analysis_folder(p.name)
                ],
                key=lambda p: p.name,
            ):
                existing_result = self._find_existing_result_csv(path, path.name)
                disabled_reason = self._get_qc_ready_disabled_reason(
                    date_option.folder_name,
                    path.name,
                )
                if disabled_reason is None and existing_result is not None and not force_run:
                    disabled_reason = existing_result.name
                run_key = f"{date_option.folder_name}|{path.name}"
                options.append(
                    _AnalysisOption(
                        run_key=run_key,
                        date_name=date_option.folder_name,
                        date_label=date_option.label,
                        analysis_id=path.name,
                        analysis_path=path,
                        enabled=disabled_reason is None,
                        existing_result=existing_result,
                        disabled_reason=disabled_reason,
                    )
                )

        self._analysis_options = options
        self._option_by_key = {option.run_key: option for option in options}

        for row, option in enumerate(options):
            var = tk.BooleanVar(value=option.enabled)
            self._analysis_vars[option.run_key] = var

            label = (
                f"{option.date_label} / {option.analysis_id}"
                if multi_date else option.analysis_id
            )
            if option.disabled_reason is not None:
                label = f"{label}   ({option.disabled_reason})"
            elif option.existing_result is not None and force_run:
                label = f"{label}   (overwrite {option.existing_result.name})"

            check = ttk.Checkbutton(
                self._check_frame,
                text=label,
                variable=var,
            )
            if not option.enabled:
                check.state(["disabled"])
            check.grid(row=row, column=0, sticky=tk.W, pady=2)
            self._analysis_checks.append(check)

        self._status_label.config(
            text=(
                f"Ready: {sum(1 for option in options if option.enabled)}/"
                f"{len(options)} Analysis ID(s)"
            ),
            foreground=_C_NORMAL,
        )

    def _is_analysis_folder(self, folder_name: str) -> bool:
        return bool(self._analysis_id_re.match(folder_name)) or is_test_analysis_id(folder_name)

    def _get_qc_ready_disabled_reason(self, date_name: str, analysis_id: str) -> str | None:
        if is_test_analysis_id(analysis_id):
            analysis_path = self._root_path / date_name / analysis_id
            for _, source_folder in find_test_source_folders(analysis_path):
                pass_path, _ = get_test_output_folders(analysis_path, source_folder)
                if not pass_path.is_dir():
                    return f"{source_folder.name}_pass missing"
                if not self._has_qc_ready_png(pass_path):
                    return f"{source_folder.name}_pass empty"
            return None

        folder_result = self._folder_qc.validate(
            root_path=self._root_path,
            date_str=date_name,
            analysis_id=analysis_id,
        )
        if folder_result.status != "PASS":
            return "FolderQC FAIL"

        for cell_result in folder_result.cells:
            for face_name, source_path in [
                ("FRONT", cell_result.front_path),
                ("BACK", cell_result.back_path),
            ]:
                if source_path is None:
                    return f"{face_name} folder missing"
                pass_path = source_path.parent / f"{face_name}_pass"
                if not pass_path.is_dir():
                    return f"{face_name}_pass missing"
                if not self._has_qc_ready_png(pass_path):
                    return f"{face_name}_pass empty"
        return None

    def _has_qc_ready_png(self, folder_path: Path) -> bool:
        return any(
            p.is_file()
            and p.suffix.lower() == ".png"
            and not p.name.endswith("_mask.png")
            for p in folder_path.iterdir()
        )

    def _find_existing_result_csv(self, analysis_path: Path, analysis_id: str) -> Path | None:
        old_path = analysis_path / _RESULT_PATTERN_OLD.format(analysis_id=analysis_id)
        if old_path.exists():
            return old_path

        matches = sorted(
            analysis_path.glob(_RESULT_PATTERN_NEW.format(analysis_id=analysis_id))
        )
        return matches[0] if matches else None

    def _run_ml(self) -> None:
        if self._running:
            return

        selected_options = [
            option
            for option in self._analysis_options
            if option.enabled and self._analysis_vars[option.run_key].get()
        ]
        if not selected_options:
            messagebox.showwarning("ML", "Select at least one Analysis ID to process.")
            return

        try:
            with open(self._config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
            model_dir = resolve_model_path(cfg["ml"]["model_path"])
            self._save_mask_images = bool(cfg["ml"].get("save_mask_images", True))
            baseline_file = model_dir / "baseline_stats.json"
            normalization_cfg = dict(cfg.get("normalization", {}) or {})
            self._normalization_cfg = normalization_cfg
            self._normalization_enabled = bool(normalization_cfg.get("enabled", False))
            self._low_confidence_threshold = float(
                normalization_cfg.get("low_confidence_threshold", 0.60)
            )
            self._drift_detector = DriftDetector(
                baseline_file,
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
            self._normalization_skipped_reason = None
        except Exception as exc:
            messagebox.showerror("ML Error", f"Failed to read ML config: {exc}")
            return

        model_file = model_dir / "model.txt"
        joblib_file = model_dir / "model.joblib"
        metadata_file = model_dir / "metadata.json"
        if not (model_file.exists() or joblib_file.exists()) or not metadata_file.exists():
            messagebox.showerror(
                "ML Error",
                "Model files are missing.\n"
                f"Expected:\n- {metadata_file}\n- {model_file} or {joblib_file}",
            )
            return

        try:
            tasks_by_key = {
                option.run_key: self._build_analysis_tasks(option)
                for option in selected_options
            }
        except Exception as exc:
            messagebox.showerror("ML Error", str(exc))
            return

        total_images = sum(len(tasks) for tasks in tasks_by_key.values())
        if total_images == 0:
            messagebox.showerror(
                "ML Error",
                "No QC-ready images were found in FRONT_pass / BACK_pass / PASS.",
            )
            return

        force_run = bool(self._force_run_var.get())
        db_skip_pairs = self._confirm_sqlite_duplicate_policy(tasks_by_key)

        try:
            if self._inference is None or self._inference.model_dir != model_dir:
                self._inference = InferenceEngine(self._config_path, model_path=model_dir)
        except Exception as exc:
            messagebox.showerror("ML Error", f"Model load failed: {exc}")
            return

        self._running = True
        self._stop_event.clear()
        self._total_elapsed = 0.0
        self._processed_images = 0
        self._issue_report_cache.clear()
        self._set_controls_enabled(False)
        self._progress.configure(maximum=max(total_images, 1), value=0)
        self._progress_label.config(text=f"Progress: 0/{total_images} images")
        self._current_label.config(text="Current: -")
        self._avg_time_label.config(text="Avg / image: -")
        self._status_label.config(text="Running ML...", foreground=_C_NORMAL)
        self._init_progress_rows(selected_options, tasks_by_key)

        self._worker = threading.Thread(
            target=self._ml_worker,
            args=(selected_options, tasks_by_key, total_images, force_run, db_skip_pairs),
            daemon=True,
        )
        self._worker.start()

    def _confirm_sqlite_duplicate_policy(
        self,
        tasks_by_key: dict[str, list[_ImageTask]],
    ) -> set[tuple[str, str]]:
        pairs = {
            (task.analysis_id, str(task.cell))
            for tasks in tasks_by_key.values()
            for task in tasks
        }
        existing = self._sqlite_logger.existing_prediction_cells(pairs)
        if not existing:
            return set()

        preview = ", ".join(
            f"{analysis_id} / Cell {cell}"
            for analysis_id, cell in sorted(existing)[:8]
        )
        if len(existing) > 8:
            preview += ", ..."
        replace = messagebox.askyesno(
            "SQLite Monitoring Duplicate",
            "Monitoring DB already has rows for these AnalysisID + Cell pairs:\n\n"
            f"{preview}\n\n"
            "Replace existing SQLite monitoring rows for these pairs?\n\n"
            "Yes: delete old DB rows and insert new monitoring rows.\n"
            "No: keep old DB rows and skip new DB inserts for these pairs.\n\n"
            "CSV and mask outputs are not affected.",
        )
        if replace:
            self._sqlite_logger.delete_prediction_cells(existing)
            return set()
        return existing

    def _init_progress_rows(
        self,
        selected_options: list[_AnalysisOption],
        tasks_by_key: dict[str, list[_ImageTask]],
    ) -> None:
        for item_id in self._progress_tree.get_children():
            self._progress_tree.delete(item_id)
        self._progress_rows.clear()
        self._cell_progress.clear()

        multi_date = len({option.date_name for option in selected_options}) > 1
        for option in selected_options:
            grouped: dict[int, int] = {}
            for task in tasks_by_key[option.run_key]:
                grouped[task.cell] = grouped.get(task.cell, 0) + 1

            for cell in sorted(grouped):
                analysis_label = (
                    f"{option.date_label} / {option.analysis_id}"
                    if multi_date else option.analysis_id
                )
                item_id = self._progress_tree.insert(
                    "",
                    tk.END,
                    values=(analysis_label, f"Cell #{cell}", f"0/{grouped[cell]} [queued]"),
                )
                key = (option.run_key, cell)
                self._progress_rows[key] = item_id
                self._cell_progress[key] = {
                    "processed": 0,
                    "total": grouped[cell],
                    "analysis_label": analysis_label,
                }

    def _request_stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._status_label.config(text="Stopping after current image...", foreground=_C_FAIL)

    def _build_analysis_tasks(self, option: _AnalysisOption) -> list[_ImageTask]:
        if is_test_analysis_id(option.analysis_id):
            return self._build_test_analysis_tasks(option)

        folder_result = self._folder_qc.validate(
            root_path=self._root_path,
            date_str=option.date_name,
            analysis_id=option.analysis_id,
        )
        if folder_result.status != "PASS":
            reasons = "; ".join(folder_result.fail_reasons) or "FolderQC FAIL"
            raise RuntimeError(f"{option.analysis_id}: {reasons}")

        tasks: list[_ImageTask] = []
        for cell_result in folder_result.cells:
            cell_number = self._cell_number_from_name(cell_result.cell_name)
            for face_name, source_path in [
                ("FRONT", cell_result.front_path),
                ("BACK", cell_result.back_path),
            ]:
                if source_path is None:
                    continue
                pass_path = source_path.parent / f"{face_name}_pass"
                if not pass_path.is_dir():
                    raise RuntimeError(f"{option.analysis_id}: pass folder missing: {pass_path}")

                image_paths = self._ml_pass_pngs(
                    source_path,
                    pass_path,
                    option.analysis_id,
                )

                for image_index, image_path in enumerate(image_paths, start=1):
                    tasks.append(
                        _ImageTask(
                            run_key=option.run_key,
                            date_name=option.date_name,
                            analysis_id=option.analysis_id,
                            analysis_path=option.analysis_path,
                            cell=cell_number,
                            face=face_name,
                            image_num=image_index,
                            image_path=image_path,
                        )
                    )

        return tasks

    def _expected_pass_count(self, analysis_id: str) -> int | None:
        if is_test_analysis_id(analysis_id) or len(analysis_id) < 2:
            return None
        expected_source = self._image_count_rules.get(analysis_id[1])
        if expected_source is None:
            return None
        return max(int(expected_source) - 2, 0)

    def _source_images(self, folder_path: Path) -> list[Path]:
        if not folder_path.is_dir():
            return []
        return sorted(
            [
                path
                for path in folder_path.iterdir()
                if path.is_file()
                and path.suffix.lower() in self._valid_image_extensions
            ],
            key=lambda path: path.name,
        )

    def _pass_pngs(self, folder_path: Path) -> list[Path]:
        if not folder_path.is_dir():
            return []
        return sorted(
            [
                path
                for path in folder_path.iterdir()
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
        expected_pass_count = self._expected_pass_count(analysis_id)
        if expected_pass_count is None or len(pass_paths) == expected_pass_count:
            return pass_paths

        source_images = self._source_images(source_folder)
        if len(source_images) <= 2:
            return pass_paths

        excluded_stems = {source_images[0].stem, source_images[-1].stem}
        filtered = [path for path in pass_paths if path.stem not in excluded_stems]
        if len(filtered) == expected_pass_count:
            logger.debug(
                "ML excludes report-only first/last crops: %s -> %d/%d images",
                pass_folder,
                len(filtered),
                len(pass_paths),
            )
        return filtered

    def _build_test_analysis_tasks(self, option: _AnalysisOption) -> list[_ImageTask]:
        tasks: list[_ImageTask] = []
        for cell, source_folder in find_test_source_folders(option.analysis_path):
            pass_path, _ = get_test_output_folders(option.analysis_path, source_folder)
            if not pass_path.is_dir():
                raise RuntimeError(
                    f"{option.analysis_id}: PASS folder missing: {pass_path}"
                )
            image_paths = sorted(
                [
                    path
                    for path in pass_path.iterdir()
                    if path.is_file()
                    and path.suffix.lower() == ".png"
                    and not path.name.endswith("_mask.png")
                ],
                key=lambda path: path.name,
            )
            tasks.extend(
                _ImageTask(
                    run_key=option.run_key,
                    date_name=option.date_name,
                    analysis_id=option.analysis_id,
                    analysis_path=option.analysis_path,
                    cell=cell,
                    face="TEST",
                    image_num=image_index,
                    image_path=image_path,
                )
                for image_index, image_path in enumerate(image_paths, start=1)
            )
        return tasks

    def _cell_number_from_name(self, cell_name: str) -> int:
        if cell_name.lower() == "single":
            return 1
        match = re.search(r"(\d+)", cell_name)
        if not match:
            raise RuntimeError(f"Unable to parse cell number from '{cell_name}'")
        return int(match.group(1))

    def _ml_worker(
        self,
        selected_options: list[_AnalysisOption],
        tasks_by_key: dict[str, list[_ImageTask]],
        total_images: int,
        force_run: bool,
        db_skip_pairs: set[tuple[str, str]],
    ) -> None:
        try:
            processed = 0
            for option in selected_options:
                if self._stop_event.is_set():
                    self._queue.put(("stopped", (processed, total_images)))
                    return

                tasks = tasks_by_key[option.run_key]
                rows: list[dict] = []
                monitoring_rows: list[dict] = []
                for task in tasks:
                    if self._stop_event.is_set():
                        self._queue.put(("stopped", (processed, total_images)))
                        return

                    self._queue.put(("current", (processed, total_images, task)))
                    started = time.perf_counter()

                    source_image = quiet_imread(task.image_path, cv2.IMREAD_UNCHANGED)
                    if source_image is None:
                        raise RuntimeError(f"Failed to read image: {task.image_path}")
                    read_elapsed = time.perf_counter() - started
                    image_stats = self._image_statistics(source_image)
                    stats_elapsed = time.perf_counter() - started - read_elapsed
                    label_mask, confidence_mask = self._predict_image(source_image)
                    predict_total_elapsed = time.perf_counter() - started - read_elapsed - stats_elapsed
                    monitoring_info = self._monitoring_info(
                        source_image=source_image,
                        label_mask=label_mask,
                        confidence_mask=confidence_mask,
                    )
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
                    result_row = self._quantifier.quantify(
                        label_mask,
                        analysis_id=task.analysis_id,
                        cell=task.cell,
                        face=task.face,
                        image_num=task.image_num,
                    )
                    result_row.update(self._issue_info_for_task(task))
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
                                "ML tab mask write timing: %s / Cell %s / %s / Layer %s path=%s %.3fs",
                                task.analysis_id,
                                task.cell,
                                task.face,
                                task.image_num,
                                mask_path,
                                mask_elapsed,
                            )
                    total_before_cleanup = time.perf_counter() - started
                    if total_before_cleanup >= 5.0 or mask_elapsed >= 0.25 or logger.isEnabledFor(logging.DEBUG):
                        logger.info(
                            "ML tab image timing: %s / Cell %s / %s / Layer %s "
                            "read=%.2fs stats=%.2fs predict_total=%.2fs mask_write=%.2fs total=%.2fs save_mask=%s",
                            task.analysis_id,
                            task.cell,
                            task.face,
                            task.image_num,
                            read_elapsed,
                            stats_elapsed,
                            predict_total_elapsed,
                            mask_elapsed,
                            total_before_cleanup,
                            self._save_mask_images,
                        )
                    monitoring_rows.append(
                        self._build_prediction_record(
                            task=task,
                            result_row=result_row,
                            image_stats=image_stats,
                            confidence_stats=self._confidence_statistics(confidence_mask),
                            monitoring_info=monitoring_info,
                            mask_path=mask_path,
                        )
                    )
                    del label_mask  # mask ë°°ì—´ ì¦‰ì‹œ í•´ì œ

                    del confidence_mask
                    del source_image

                    elapsed = time.perf_counter() - started
                    processed += 1
                    self._queue.put(("progress", (processed, total_images, task, elapsed)))

                csv_path = self._write_analysis_csv(
                    option.analysis_id,
                    tasks,
                    rows,
                    force_run=force_run,
                )
                for monitoring_row in monitoring_rows:
                    pair = (
                        str(monitoring_row.get("analysis_id", "")),
                        str(monitoring_row.get("cell", "")),
                    )
                    if pair in db_skip_pairs:
                        continue
                    monitoring_row["csv_path"] = str(csv_path)
                    self._sqlite_logger.insert_prediction_run(monitoring_row)
                self._queue.put(("analysis_done", (option.run_key, csv_path)))

            self._queue.put(("done", total_images))
        except Exception as exc:
            logger.exception("ML worker error")
            self._queue.put(("error", str(exc)))

    def _mask_dir_for_task(self, task: _ImageTask) -> Path:
        if is_test_analysis_id(task.analysis_id) and task.face.upper() == "TEST":
            return get_test_mask_folder_for_pass(task.image_path.parent)
        return task.image_path.parent.with_name(f"{task.face}_mask")

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
                    "ML tab predict timing: model=%s type=%s family=%s feature_set=%s "
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

    def _confidence_statistics(
        self,
        confidence_mask: np.ndarray | None,
    ) -> dict[str, float | None]:
        if confidence_mask is None:
            return {"mean_confidence": None, "low_confidence_ratio": None}

        valid_confidence = confidence_mask[np.isfinite(confidence_mask)]
        if valid_confidence.size == 0:
            return {"mean_confidence": None, "low_confidence_ratio": None}

        return {
            "mean_confidence": float(valid_confidence.mean()),
            "low_confidence_ratio": float(
                np.count_nonzero(valid_confidence < self._low_confidence_threshold)
                / valid_confidence.size
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
        bgr = self._bgr_from_image(source_image)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        background_mask = self._high_confidence_background_mask(
            label_mask,
            confidence_mask,
        )
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

    def _build_prediction_record(
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

    def _customer_for_analysis(self, analysis_id: str) -> str:
        if is_test_analysis_id(analysis_id):
            return "TEST"
        if len(analysis_id) < 2:
            return ""
        customer_code = analysis_id[1]
        return str(self._customer_codes.get(customer_code, customer_code))

    def _write_analysis_csv(
        self,
        analysis_id: str,
        tasks: list[_ImageTask],
        rows: list[dict],
        *,
        force_run: bool = False,
    ) -> Path:
        if not tasks or not rows:
            raise RuntimeError(f"{analysis_id}: no ML rows generated.")

        average_percent = self._compute_average_li_salt_percent(tasks, rows)
        csv_name = f"{analysis_id}_LI-SALT_{average_percent:.2f}%.csv"
        csv_path = tasks[0].analysis_path / csv_name
        if force_run:
            self._delete_existing_result_csvs(tasks[0].analysis_path, analysis_id)

        frame = pd.DataFrame(
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
                "isIssue",
                "issueCategory",
                "issueNote",
            ],
        )
        frame.to_csv(csv_path, index=False, float_format="%.3f")
        return csv_path

    def _issue_info_for_task(self, task: _ImageTask) -> dict:
        issue_report = self._load_issue_report(task.analysis_path)
        relative_path = self._relative_or_name(task.image_path, task.analysis_path)
        issue_key = f"Cell #{task.cell}|{task.face}|{task.image_path.stem}"
        record = (
            issue_report.get(relative_path)
            or issue_report.get(task.image_path.name)
            or issue_report.get(issue_key)
        )
        is_issue = record is not None or "_issue" in task.image_path.stem
        return {
            "isIssue": bool(is_issue),
            "issueCategory": record.get("issueCategory", "") if record else "",
            "issueNote": record.get("issueNote", "") if record else "",
        }

    def _load_issue_report(self, analysis_path: Path) -> dict[str, dict]:
        analysis_path = Path(analysis_path)
        cached = self._issue_report_cache.get(analysis_path)
        if cached is not None:
            return cached

        report_path = analysis_path / _ISSUE_REPORT_FILE
        records: dict[str, dict] = {}
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Invalid issue report JSON ignored: %s", report_path)
                data = {}
            for item in data.get("issues", []):
                if not isinstance(item, dict):
                    continue
                for key_name in ("pass_image", "source_image"):
                    value = item.get(key_name)
                    if value:
                        records[str(value)] = item
                        records[Path(str(value)).name] = item
                source_image = item.get("source_image", "")
                source_stem = Path(str(source_image)).stem if source_image else ""
                cell = item.get("cell", "")
                face = item.get("face", "")
                if cell and face and source_stem:
                    records[f"{cell}|{face}|{source_stem}"] = item

        self._issue_report_cache[analysis_path] = records
        return records

    def _relative_or_name(self, path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return path.name

    def _delete_existing_result_csvs(self, analysis_path: Path, analysis_id: str) -> None:
        old_path = analysis_path / _RESULT_PATTERN_OLD.format(analysis_id=analysis_id)
        candidates = [old_path]
        candidates.extend(
            sorted(analysis_path.glob(_RESULT_PATTERN_NEW.format(analysis_id=analysis_id)))
        )
        for path in candidates:
            if not path.exists() or not path.is_file():
                continue
            try:
                path.unlink()
            except OSError as exc:
                raise RuntimeError(f"Failed to overwrite existing result CSV: {path}") from exc

    def _compute_average_li_salt_percent(
        self,
        tasks: list[_ImageTask],
        rows: list[dict],
    ) -> float:
        row_map = {
            (row["Cell"], row["Face"], row["ImageNum"]): row
            for row in rows
        }

        included_values: list[float] = []
        grouped: dict[tuple[int, str], list[_ImageTask]] = {}
        for task in tasks:
            grouped.setdefault((task.cell, task.face), []).append(task)

        for group_tasks in grouped.values():
            ordered = sorted(group_tasks, key=lambda task: task.image_num)
            if all(task.face.upper() == "TEST" for task in ordered):
                for task in ordered:
                    row = row_map[(task.cell, task.face, task.image_num)]
                    included_values.append(float(row["li_salt_ratio"]) * 100.0)
                continue
            if len(ordered) <= 2:
                continue
            for task in ordered[1:-1]:
                row = row_map[(task.cell, task.face, task.image_num)]
                included_values.append(float(row["li_salt_ratio"]) * 100.0)

        if not included_values:
            return 0.0
        return float(sum(included_values) / len(included_values))

    def _poll_queue(self) -> None:
        try:
            while True:
                msg_type, payload = self._queue.get_nowait()
                self._dispatch(msg_type, payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _dispatch(self, msg_type: str, payload) -> None:
        if msg_type == "current":
            processed, total, task = payload
            self._progress.configure(value=processed)
            self._progress_label.config(text=f"Progress: {processed}/{total} images")
            self._current_label.config(
                text=(
                    "Current: "
                    f"{task.analysis_id} / Cell #{task.cell} / {task.face} / {task.image_path.name}"
                )
            )
            self._set_cell_status(task.run_key, task.cell, "processing")

        elif msg_type == "progress":
            processed, total, task, elapsed = payload
            self._progress.configure(value=processed)
            self._progress_label.config(text=f"Progress: {processed}/{total} images")
            self._current_label.config(
                text=(
                    "Current: "
                    f"{task.analysis_id} / Cell #{task.cell} / {task.face} / {task.image_path.name}"
                )
            )
            self._total_elapsed += float(elapsed)
            self._processed_images = processed
            avg = self._total_elapsed / processed if processed else 0.0
            self._avg_time_label.config(text=f"Avg / image: {avg:.2f}s")
            self._advance_cell_progress(task.run_key, task.cell)

        elif msg_type == "analysis_done":
            run_key, csv_path = payload
            option = self._option_by_key.get(run_key)
            label = option.analysis_id if option else run_key
            self._status_label.config(
                text=f"{label}: saved {csv_path.name}",
                foreground=_C_NORMAL,
            )

        elif msg_type == "done":
            total_images = payload
            self._running = False
            self._set_controls_enabled(True)
            self._current_label.config(text="Current: -")
            self._refresh_analysis_list()
            self._status_label.config(
                text=f"ML complete: {total_images} image(s)",
                foreground=_C_NORMAL,
            )

        elif msg_type == "stopped":
            processed, total = payload
            self._running = False
            self._set_controls_enabled(True)
            self._progress.configure(value=processed)
            self._progress_label.config(text=f"Progress: {processed}/{total} images")
            self._current_label.config(text="Current: stopped")
            self._refresh_analysis_list()
            self._status_label.config(
                text=f"ML stopped: {processed}/{total} image(s) processed",
                foreground=_C_FAIL,
            )

        elif msg_type == "error":
            self._running = False
            self._set_controls_enabled(True)
            self._status_label.config(text=f"Error: {payload}", foreground=_C_FAIL)
            messagebox.showerror("ML Error", payload)

    def _set_cell_status(self, run_key: str, cell: int, mode: str) -> None:
        key = (run_key, cell)
        cell_info = self._cell_progress.get(key)
        item_id = self._progress_rows.get(key)
        if cell_info is None or item_id is None:
            return
        processed = int(cell_info["processed"])
        total = int(cell_info["total"])
        suffix = "processing..." if mode == "processing" else str(mode)
        self._progress_tree.item(
            item_id,
            values=(
                cell_info["analysis_label"],
                f"Cell #{cell}",
                f"{processed}/{total} [{suffix}]",
            ),
        )

    def _advance_cell_progress(self, run_key: str, cell: int) -> None:
        key = (run_key, cell)
        cell_info = self._cell_progress.get(key)
        item_id = self._progress_rows.get(key)
        if cell_info is None or item_id is None:
            return

        processed = int(cell_info["processed"]) + 1
        total = int(cell_info["total"])
        cell_info["processed"] = processed
        status = "done" if processed >= total else "processing..."
        self._progress_tree.item(
            item_id,
            values=(
                cell_info["analysis_label"],
                f"Cell #{cell}",
                f"{processed}/{total} [{status}]",
            ),
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._run_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)
        self._stop_btn.config(state=tk.DISABLED if enabled else tk.NORMAL)
        self._date_listbox.config(state=tk.NORMAL if enabled else tk.DISABLED)
        self._force_run_check.config(state=tk.NORMAL if enabled else tk.DISABLED)




