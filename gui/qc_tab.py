# qc_tab.py — tkinter QC tab: PhotoJig QC Dashboard

import logging
import json
import queue
import re
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import yaml

from modules.qc.folder_qc import FolderQC, FolderQCResult
from modules.qc.crop_engine import CropEngine, CropResult
from modules.qc.retake_manager import RetakeManager, RetakeItem
from modules.utils.image_io import quiet_imread
from modules.utils.analysis_id import (
    build_test_output_name,
    ensure_test_output_folders,
    find_test_images,
    find_test_source_folders,
    get_test_output_folders,
    is_test_analysis_id,
)

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

_ANALYSIS_ID_RE = re.compile(r"^[A-Z]{3}[0-9]{5}$")

_COL_DEFS: list[tuple[str, int]] = [
    ("AnalysisID",  110),
    ("FolderQC",     85),
    ("CropEngine",   95),
    ("FAIL Image",  220),
    ("FAIL Reason", 240),
    ("Retake",      220),
]
_CI: dict[str, int] = {name: i for i, (name, _) in enumerate(_COL_DEFS)}

_C_PASS   = "#007700"
_C_FAIL   = "#cc0000"
_C_NORMAL = "#000000"
_C_MUTED  = "#888888"
_BG_HEAD  = "#d0d0d0"
_BG_EVEN  = "#ffffff"
_BG_ODD   = "#f5f5f5"
_BG_CROP_DONE = "#e0f2fe"
_BG_CROP_TODO = "#fef9c3"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_ISSUE_REPORT_FILE = "issue_report.json"
_ISSUE_CATEGORIES = (
    "찢김 (tear)",
    "얼룩 (stain)",
    "기타 (other)",
)


# ── State dataclass ────────────────────────────────────────────────────────────

@dataclass
class _AnalysisState:
    """Full QC state for one AnalysisID."""
    analysis_id: str
    analysis_path: Path
    date_str: str = ""
    customer_code: str = ""
    folder_qc_result: Optional[FolderQCResult] = None
    crop_done: bool = False
    retake_items: list[RetakeItem] = field(default_factory=list)
    run_status: str = "Queued"

    @property
    def folder_qc_pass(self) -> bool:
        return (self.folder_qc_result is not None
                and self.folder_qc_result.status == "PASS")

    @property
    def crop_all_pass(self) -> bool:
        return self.crop_done and not self.retake_items

    @property
    def all_pass(self) -> bool:
        return self.folder_qc_pass and self.crop_all_pass


# ── Utility ────────────────────────────────────────────────────────────────────

def _is_subpath(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


# ── QCTab ──────────────────────────────────────────────────────────────────────

class QCTab(ttk.Frame):
    """
    QC Tab — PhotoJig electrode image quality control dashboard.

    Layout:
      [Header: Root path + Date]
      [Toolbar: Run QC button + status label]
      [Table: AnalysisID × FAIL image rows — scrollable]
    """

    def __init__(self, parent, config_path: str | Path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._load_config()

        self._folder_qc      = FolderQC(config_path)
        self._crop_engine    = CropEngine(config_path)
        self._retake_manager = RetakeManager(config_path)

        self._color_normalizer = None
        try:
            from modules.qc.color_normalizer import ColorNormalizer
            self._color_normalizer = ColorNormalizer(config_path)
        except RuntimeError:
            logger.warning("QCTab: ColorNormalizer skipped — reference not configured")

        self._states: list[_AnalysisState] = []
        self._available_analysis_ids: list[str] = []
        self._available_analysis_targets: list[tuple[str, Path, str]] = []
        self._queue: queue.Queue = queue.Queue()
        self._row_widgets: list[list[tk.Widget]] = []
        self._review_selection_key: tuple[str, str] | None = None
        self._review_photo: tk.PhotoImage | None = None
        self._running = False
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._build_ui()
        self._refresh_date_options()
        self._poll_queue()

    # ── Config ─────────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._root_path = Path(cfg["app"]["root_path"])
        self._analysis_id_re = re.compile(cfg["folder_qc"]["analysis_id_pattern"])
        self._date_formats = list(cfg["folder_qc"]["date_formats"])

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_header(row=0)
        self._build_toolbar(row=1)
        self._build_table_area(row=2)
        self._build_review_area(row=3)

    def _build_header(self, row: int) -> None:
        fr = ttk.LabelFrame(self, text="PhotoJig QC", padding=8)
        fr.grid(row=row, column=0, sticky=tk.EW, padx=10, pady=(8, 2))
        fr.columnconfigure(1, weight=1)
        fr.columnconfigure(3, weight=1)

        ttk.Label(fr, text="Mode:").grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        self._selection_mode_var = tk.StringVar(value="By Date")
        self._selection_mode_combo = ttk.Combobox(
            fr,
            textvariable=self._selection_mode_var,
            state="readonly",
            values=("By Date", "By Analysis ID"),
            width=18,
        )
        self._selection_mode_combo.grid(row=0, column=1, sticky=tk.W)
        self._selection_mode_combo.bind("<<ComboboxSelected>>", self._on_selection_mode_change)

        ttk.Label(fr, text="Date:").grid(row=0, column=2, sticky=tk.W, padx=(18, 6))
        self._date_var = tk.StringVar()
        self._date_combo = ttk.Combobox(
            fr,
            textvariable=self._date_var,
            state="readonly",
            width=18,
        )
        self._date_combo.grid(row=0, column=3, sticky=tk.W)
        self._date_combo.bind("<<ComboboxSelected>>", self._on_date_change)

        legend = ttk.Frame(fr)
        legend.grid(row=0, column=4, sticky=tk.E, padx=(18, 0))
        tk.Label(legend, width=2, bg=_BG_CROP_DONE, relief=tk.GROOVE).pack(side=tk.LEFT)
        ttk.Label(legend, text="Cropped").pack(side=tk.LEFT, padx=(4, 10))
        tk.Label(legend, width=2, bg=_BG_CROP_TODO, relief=tk.GROOVE).pack(side=tk.LEFT)
        ttk.Label(legend, text="Not cropped").pack(side=tk.LEFT, padx=(4, 0))

        ttk.Label(fr, text="AnalysisID:").grid(row=1, column=0, sticky=tk.NW, padx=(0, 6), pady=(8, 0))
        list_fr = ttk.Frame(fr)
        list_fr.grid(row=1, column=1, columnspan=4, sticky=tk.EW, pady=(8, 0))
        list_fr.columnconfigure(0, weight=1)
        self._analysis_listbox = tk.Listbox(
            list_fr,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=5,
        )
        self._analysis_listbox.grid(row=0, column=0, sticky=tk.EW)
        list_scroll = ttk.Scrollbar(
            list_fr,
            orient=tk.VERTICAL,
            command=self._analysis_listbox.yview,
        )
        list_scroll.grid(row=0, column=1, sticky=tk.NS)
        self._analysis_listbox.configure(yscrollcommand=list_scroll.set)

        self._analysis_hint_lbl = ttk.Label(fr, text="", foreground=_C_MUTED)
        self._analysis_hint_lbl.grid(row=2, column=1, columnspan=4, sticky=tk.W, pady=(4, 0))

    def _build_toolbar(self, row: int) -> None:
        fr = ttk.Frame(self)
        fr.grid(row=row, column=0, sticky=tk.EW, padx=10, pady=4)
        fr.columnconfigure(5, weight=1)

        self._run_btn = ttk.Button(fr, text="Run QC", command=self._run_qc)
        self._run_btn.grid(row=0, column=0, sticky=tk.W)

        self._stop_btn = ttk.Button(
            fr,
            text="Stop",
            command=self._request_stop,
            state=tk.DISABLED,
        )
        self._stop_btn.grid(row=0, column=1, sticky=tk.W, padx=(6, 0))

        self._issue_btn = ttk.Button(
            fr,
            text="Report issue cell",
            command=self._open_issue_report_dialog,
        )
        self._issue_btn.grid(row=0, column=2, sticky=tk.W, padx=(6, 0))

        self._force_run_var = tk.BooleanVar(value=False)
        self._force_run_check = ttk.Checkbutton(
            fr,
            text="Force Run",
            variable=self._force_run_var,
        )
        self._force_run_check.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        self._progress = ttk.Progressbar(fr, mode="determinate", maximum=1, value=0)
        self._progress.grid(row=0, column=4, sticky=tk.EW, padx=(12, 12))

        self._status_lbl = ttk.Label(fr, text="")
        self._status_lbl.grid(row=0, column=5, sticky=tk.W)

        self._current_lbl = ttk.Label(fr, text="", foreground=_C_MUTED)
        self._current_lbl.grid(row=1, column=4, columnspan=2, sticky=tk.W, pady=(4, 0))

    def _build_table_area(self, row: int) -> None:
        container = ttk.Frame(self)
        container.grid(row=row, column=0, sticky=tk.NSEW, padx=10, pady=(2, 8))
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        self._canvas = tk.Canvas(container, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient=tk.VERTICAL,   command=self._canvas.yview)
        hscroll = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=self._canvas.xview)
        self._canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        self._canvas.grid(row=0, column=0, sticky=tk.NSEW)
        vscroll.grid(row=0, column=1, sticky=tk.NS)
        hscroll.grid(row=1, column=0, sticky=tk.EW)

        self._tbl = tk.Frame(self._canvas, bg=_BG_EVEN)
        self._tbl_win = self._canvas.create_window((0, 0), window=self._tbl, anchor=tk.NW)

        self._tbl.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas.bind("<Configure>", self._on_canvas_resize)

        self._canvas.bind(
            "<Enter>",
            lambda e: self._canvas.bind_all("<MouseWheel>", self._on_mousewheel),
        )
        self._canvas.bind(
            "<Leave>",
            lambda e: self._canvas.unbind_all("<MouseWheel>"),
        )

        # Column headers (row 0)
        for col, (header, width) in enumerate(_COL_DEFS):
            lbl = tk.Label(
                self._tbl, text=header,
                bg=_BG_HEAD, font=("TkDefaultFont", 9, "bold"),
                anchor=tk.CENTER, relief=tk.GROOVE, padx=4, pady=5,
            )
            lbl.grid(row=0, column=col, sticky=tk.EW, padx=1, pady=1)
            self._tbl.columnconfigure(col, minsize=width, weight=0)

        self._data_row = 1

    def _build_review_area(self, row: int) -> None:
        fr = ttk.LabelFrame(self, text="Fail image review", padding=8)
        fr.grid(row=row, column=0, sticky=tk.EW, padx=10, pady=(0, 8))
        fr.columnconfigure(0, weight=1)
        fr.columnconfigure(1, weight=0)

        self._review_info_lbl = ttk.Label(fr, text="No fail image selected.")
        self._review_info_lbl.grid(row=0, column=0, sticky=tk.W)

        self._review_reason_lbl = ttk.Label(fr, text="", foreground=_C_FAIL)
        self._review_reason_lbl.grid(row=1, column=0, sticky=tk.W, pady=(2, 6))

        controls = ttk.Frame(fr)
        controls.grid(row=0, column=1, rowspan=2, sticky=tk.E, padx=(10, 0))

        self._review_prev_btn = ttk.Button(
            controls, text="Previous", width=9,
            command=lambda: self._select_review_offset(-1),
        )
        self._review_prev_btn.pack(side=tk.LEFT)

        self._review_next_btn = ttk.Button(
            controls, text="Next", width=9,
            command=lambda: self._select_review_offset(1),
        )
        self._review_next_btn.pack(side=tk.LEFT, padx=(6, 0))

        self._review_retake_btn = ttk.Button(
            controls, text="Retake complete", width=16,
            command=self._review_retake_current,
        )
        self._review_retake_btn.pack(side=tk.LEFT, padx=(10, 0))

        self._review_crop_btn = ttk.Button(
            controls, text="Crop anyway", width=12,
            command=self._review_force_crop_current,
        )
        self._review_crop_btn.pack(side=tk.LEFT, padx=(6, 0))

        self._review_canvas = tk.Canvas(fr, height=260, bg="#f2f2f2", highlightthickness=1)
        self._review_canvas.grid(row=2, column=0, columnspan=2, sticky=tk.EW)
        self._review_canvas.bind("<Configure>", lambda _e: self._draw_review_image())
        self._set_review_controls_enabled(False)

    def _on_canvas_resize(self, event) -> None:
        total_w = sum(w for _, w in _COL_DEFS) + 20
        self._canvas.itemconfig(self._tbl_win, width=max(event.width, total_w))

    def _on_mousewheel(self, event) -> None:
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ── Date folder search ─────────────────────────────────────────────────────

    def _parse_date_folder(self, folder_name: str) -> Optional[datetime]:
        for fmt in self._date_formats:
            try:
                return datetime.strptime(folder_name, fmt)
            except ValueError:
                continue
        return None

    def _list_date_folders(self) -> list[str]:
        if not self._root_path.exists():
            return []

        dated: list[tuple[datetime, str]] = []
        for path in self._root_path.iterdir():
            if not path.is_dir():
                continue
            parsed = self._parse_date_folder(path.name)
            if parsed is not None:
                canonical = parsed.strftime("%Y-%m-%d")
                actual_path = self._normalize_date_folder(path, canonical)
                dated.append((parsed, actual_path.name))

        dated.sort(key=lambda item: item[0], reverse=True)
        return [name for _, name in dated]

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

    def _list_analysis_ids(self, date_path: Path) -> list[str]:
        if not date_path.is_dir():
            return []
        return sorted(
            d.name for d in date_path.iterdir()
            if d.is_dir() and self._is_analysis_folder(d.name)
        )

    def _list_analysis_targets_for_date(self, date_str: str) -> list[tuple[str, Path, str]]:
        date_path = self._root_path / date_str if date_str else None
        if date_path is None or not date_path.is_dir():
            return []
        return [
            (date_str, date_path, analysis_id)
            for analysis_id in self._list_analysis_ids(date_path)
        ]

    def _list_all_analysis_targets(self) -> list[tuple[str, Path, str]]:
        targets: list[tuple[str, Path, str]] = []
        for date_name in self._list_date_folders():
            date_path = self._root_path / date_name
            for analysis_id in self._list_analysis_ids(date_path):
                targets.append((date_name, date_path, analysis_id))
        return sorted(targets, key=lambda item: (item[2], item[0]))

    def _analysis_crop_complete(self, date_path: Path, analysis_id: str) -> bool:
        analysis_path = date_path / analysis_id
        if not analysis_path.is_dir():
            return False

        if is_test_analysis_id(analysis_id):
            return all(
                self._has_pass_png(get_test_output_folders(analysis_path, source_path)[0])
                for _, source_path in find_test_source_folders(analysis_path)
            )

        cell_pattern = re.compile(r"^cell\s*#\d+$", re.IGNORECASE)
        cell_dirs = sorted(
            [
                path for path in analysis_path.iterdir()
                if path.is_dir() and cell_pattern.match(path.name)
            ],
            key=lambda path: path.name.lower(),
        )
        cell_paths = cell_dirs if cell_dirs else [analysis_path]
        for cell_path in cell_paths:
            if not self._has_pass_png(cell_path / "FRONT_pass"):
                return False
            if not self._has_pass_png(cell_path / "BACK_pass"):
                return False
        return True

    def _has_pass_png(self, folder_path: Path) -> bool:
        if not folder_path.is_dir():
            return False
        return any(
            path.is_file()
            and path.suffix.lower() == ".png"
            and not path.name.endswith("_mask.png")
            for path in folder_path.iterdir()
        )

    def _is_analysis_folder(self, folder_name: str) -> bool:
        return bool(self._analysis_id_re.match(folder_name)) or is_test_analysis_id(folder_name)

    def _pick_default_date(self, date_names: list[str]) -> str:
        if not date_names:
            return ""

        today_value = date.today()
        for name in date_names:
            parsed = self._parse_date_folder(name)
            if parsed is not None and parsed.date() == today_value:
                return name
        return date_names[0]

    def _refresh_date_options(self) -> None:
        previous = self._date_var.get().strip() if hasattr(self, "_date_var") else ""
        date_names = self._list_date_folders()
        self._date_combo["values"] = date_names

        if previous in date_names:
            selected = previous
        else:
            selected = self._pick_default_date(date_names)

        self._date_var.set(selected)
        self._date_str = selected
        self._refresh_analysis_hint()

    def _refresh_analysis_hint(self) -> None:
        if not hasattr(self, "_analysis_listbox"):
            return

        self._analysis_listbox.delete(0, tk.END)
        mode = self._selection_mode_var.get() if hasattr(self, "_selection_mode_var") else "By Date"
        date_str = self._date_var.get().strip()

        if mode == "By Analysis ID":
            self._date_combo.config(state=tk.DISABLED if not self._running else tk.DISABLED)
            self._available_analysis_targets = self._list_all_analysis_targets()
        else:
            if not self._running:
                self._date_combo.config(state="readonly")
            self._available_analysis_targets = self._list_analysis_targets_for_date(date_str)

        self._available_analysis_ids = [
            analysis_id for _date_str, _date_path, analysis_id in self._available_analysis_targets
        ]

        for index, (date_name, date_path, analysis_id) in enumerate(self._available_analysis_targets):
            label = (
                f"{analysis_id}  ({date_name})"
                if mode == "By Analysis ID"
                else analysis_id
            )
            self._analysis_listbox.insert(tk.END, label)
            bg = (
                _BG_CROP_DONE
                if self._analysis_crop_complete(date_path, analysis_id)
                else _BG_CROP_TODO
            )
            self._analysis_listbox.itemconfig(index, bg=bg)

        if self._available_analysis_targets:
            self._analysis_listbox.selection_set(0, tk.END)

        if not date_str and mode == "By Date":
            self._available_analysis_ids = []
            self._analysis_hint_lbl.config(text="AnalysisIDs: 0")
            return

        if not self._available_analysis_ids:
            self._analysis_hint_lbl.config(text="AnalysisIDs: 0")
            return

        preview = ", ".join(self._available_analysis_ids[:4])
        if len(self._available_analysis_ids) > 4:
            preview += ", ..."
        self._analysis_hint_lbl.config(
            text=(
                f"AnalysisIDs: {len(self._available_analysis_ids)}  ({preview})"
                if mode == "By Date"
                else f"All dates AnalysisIDs: {len(self._available_analysis_ids)}  ({preview})"
            )
        )

    def _open_issue_report_dialog(self) -> None:
        date_str = self._date_var.get().strip()
        date_path = self._root_path / date_str if date_str else None
        if not date_str or date_path is None or not date_path.is_dir():
            messagebox.showerror(
                "Report Issue",
                f"Date folder not found: {date_str or '-'}\nRoot: {self._root_path}",
            )
            return

        analysis_ids = self._list_analysis_ids(date_path)
        if not analysis_ids:
            messagebox.showerror("Report Issue", f"No AnalysisID folders found in: {date_path}")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Report issue cell")
        dialog.transient(self.winfo_toplevel())
        dialog.grab_set()
        dialog.resizable(False, False)

        analysis_var = tk.StringVar(value=analysis_ids[0])
        cell_var = tk.StringVar()
        face_var = tk.StringVar(value="FRONT")
        image_var = tk.StringVar()
        category_var = tk.StringVar(value=_ISSUE_CATEGORIES[0])
        note_var = tk.StringVar()

        cell_paths: dict[str, Path] = {}
        image_paths: dict[str, Path] = {}

        frame = ttk.Frame(dialog, padding=12)
        frame.grid(row=0, column=0, sticky=tk.NSEW)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text=f"Date: {date_str}").grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8)
        )

        ttk.Label(frame, text="Analysis ID").grid(row=1, column=0, sticky=tk.W, padx=(0, 8))
        analysis_cb = ttk.Combobox(
            frame,
            textvariable=analysis_var,
            values=analysis_ids,
            state="readonly",
            width=24,
        )
        analysis_cb.grid(row=1, column=1, sticky=tk.EW, pady=2)

        ttk.Label(frame, text="Cell").grid(row=2, column=0, sticky=tk.W, padx=(0, 8))
        cell_cb = ttk.Combobox(frame, textvariable=cell_var, state="readonly", width=24)
        cell_cb.grid(row=2, column=1, sticky=tk.EW, pady=2)

        ttk.Label(frame, text="Face").grid(row=3, column=0, sticky=tk.W, padx=(0, 8))
        face_cb = ttk.Combobox(
            frame,
            textvariable=face_var,
            values=("FRONT", "BACK"),
            state="readonly",
            width=24,
        )
        face_cb.grid(row=3, column=1, sticky=tk.EW, pady=2)

        ttk.Label(frame, text="Image").grid(row=4, column=0, sticky=tk.W, padx=(0, 8))
        image_cb = ttk.Combobox(frame, textvariable=image_var, state="readonly", width=44)
        image_cb.grid(row=4, column=1, sticky=tk.EW, pady=2)

        ttk.Label(frame, text="Issue category").grid(row=5, column=0, sticky=tk.W, padx=(0, 8))
        category_cb = ttk.Combobox(
            frame,
            textvariable=category_var,
            values=_ISSUE_CATEGORIES,
            state="readonly",
            width=24,
        )
        category_cb.grid(row=5, column=1, sticky=tk.EW, pady=2)

        ttk.Label(frame, text="Other note").grid(row=6, column=0, sticky=tk.W, padx=(0, 8))
        note_entry = ttk.Entry(frame, textvariable=note_var, width=44)
        note_entry.grid(row=6, column=1, sticky=tk.EW, pady=2)

        def update_cells(_event=None) -> None:
            cell_paths.clear()
            analysis_path = date_path / analysis_var.get()
            for label, path in self._list_issue_cell_paths(analysis_path):
                cell_paths[label] = path
            values = list(cell_paths.keys())
            cell_cb["values"] = values
            cell_var.set(values[0] if values else "")
            update_images()

        def update_images(_event=None) -> None:
            image_paths.clear()
            cell_path = cell_paths.get(cell_var.get())
            face_name = face_var.get()
            face_path = cell_path / face_name if cell_path is not None else None
            images = self._list_issue_images(face_path) if face_path is not None else []
            for image_path in images:
                image_paths[image_path.name] = image_path
            values = list(image_paths.keys())
            image_cb["values"] = values
            image_var.set(values[0] if values else "")

        def submit() -> None:
            selected_image = image_paths.get(image_var.get())
            cell_path = cell_paths.get(cell_var.get())
            if selected_image is None or cell_path is None:
                messagebox.showerror("Report Issue", "Select an image to report.")
                return

            category = category_var.get()
            note = note_var.get().strip()
            if category == "기타 (other)" and not note:
                messagebox.showerror("Report Issue", "Enter a note for 기타 (other).")
                return

            try:
                new_path = self._mark_issue_image(
                    date_path=date_path,
                    analysis_id=analysis_var.get(),
                    cell_label=cell_var.get(),
                    cell_path=cell_path,
                    face_name=face_var.get(),
                    image_path=selected_image,
                    category=category,
                    note=note,
                )
            except Exception as exc:
                logger.exception("Issue report failed")
                messagebox.showerror("Report Issue", str(exc))
                return

            self._status_lbl.config(
                text=f"Issue reported: {new_path.name}",
                foreground=_C_NORMAL,
            )
            dialog.destroy()

        analysis_cb.bind("<<ComboboxSelected>>", update_cells)
        cell_cb.bind("<<ComboboxSelected>>", update_images)
        face_cb.bind("<<ComboboxSelected>>", update_images)

        actions = ttk.Frame(frame)
        actions.grid(row=7, column=0, columnspan=2, sticky=tk.E, pady=(10, 0))
        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT)
        ttk.Button(actions, text="Report", command=submit).pack(side=tk.RIGHT, padx=(0, 6))

        update_cells()
        dialog.wait_window()

    def _list_issue_cell_paths(self, analysis_path: Path) -> list[tuple[str, Path]]:
        cell_pattern = re.compile(r"^cell\s*#\d+$", re.IGNORECASE)
        cell_dirs = sorted(
            [p for p in analysis_path.iterdir() if p.is_dir() and cell_pattern.match(p.name)],
            key=lambda p: p.name.lower(),
        )
        if cell_dirs:
            return [(p.name, p) for p in cell_dirs]
        return [("Cell #1", analysis_path)]

    def _list_issue_images(self, face_path: Path) -> list[Path]:
        if not face_path.is_dir():
            return []
        return sorted(
            [
                p for p in face_path.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            ],
            key=lambda p: p.name,
        )

    def _mark_issue_image(
        self,
        *,
        date_path: Path,
        analysis_id: str,
        cell_label: str,
        cell_path: Path,
        face_name: str,
        image_path: Path,
        category: str,
        note: str,
    ) -> Path:
        analysis_path = date_path / analysis_id
        output_folder = cell_path / f"{face_name}_pass"
        mask_folder = cell_path / f"{face_name}_mask"
        old_stem = image_path.stem
        new_stem = self._choose_issue_stem(image_path, output_folder, mask_folder)
        new_image_path = image_path.with_name(f"{new_stem}{image_path.suffix}")

        if new_image_path != image_path:
            image_path.rename(new_image_path)

        renamed_pass_path = self._rename_if_exists(
            output_folder / f"{old_stem}.png",
            output_folder / f"{new_stem}.png",
        )
        self._rename_if_exists(
            mask_folder / f"{old_stem}_mask.png",
            mask_folder / f"{new_stem}_mask.png",
        )
        self._rename_if_exists(
            output_folder / f"{old_stem}_mask.png",
            output_folder / f"{new_stem}_mask.png",
        )

        record = {
            "analysis_id": analysis_id,
            "cell": cell_label,
            "face": face_name,
            "source_image": self._relative_or_name(new_image_path, analysis_path),
            "pass_image": (
                self._relative_or_name(renamed_pass_path, analysis_path)
                if renamed_pass_path is not None
                else ""
            ),
            "isIssue": True,
            "issueCategory": category,
            "issueNote": note,
            "reportedAt": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_issue_report(analysis_path, record)
        return new_image_path

    def _choose_issue_stem(
        self,
        image_path: Path,
        output_folder: Path,
        mask_folder: Path,
    ) -> str:
        if "_issue" in image_path.stem:
            return image_path.stem

        base = f"{image_path.stem}_issue"
        for index in range(1, 1000):
            candidate = base if index == 1 else f"{base}_{index}"
            source_candidate = image_path.with_name(f"{candidate}{image_path.suffix}")
            pass_candidate = output_folder / f"{candidate}.png"
            mask_candidate = mask_folder / f"{candidate}_mask.png"
            legacy_mask_candidate = output_folder / f"{candidate}_mask.png"
            if (
                not source_candidate.exists()
                and not pass_candidate.exists()
                and not mask_candidate.exists()
                and not legacy_mask_candidate.exists()
            ):
                return candidate
        raise RuntimeError(f"Unable to choose issue filename for: {image_path}")

    def _rename_if_exists(self, source: Path, target: Path) -> Path | None:
        if not source.exists():
            return None
        if source == target:
            return target
        if target.exists():
            raise RuntimeError(f"Target file already exists: {target}")
        source.rename(target)
        return target

    def _write_issue_report(self, analysis_path: Path, record: dict) -> None:
        report_path = analysis_path / _ISSUE_REPORT_FILE
        data = {"issues": []}
        if report_path.exists():
            try:
                data = json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Invalid issue report JSON, replacing: %s", report_path)

        issues = list(data.get("issues", []))
        source_image = record["source_image"]
        pass_image = record["pass_image"]
        issues = [
            item for item in issues
            if item.get("source_image") != source_image
            and (not pass_image or item.get("pass_image") != pass_image)
        ]
        issues.append(record)
        report_path.write_text(
            json.dumps({"issues": issues}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _relative_or_name(self, path: Path, base: Path) -> str:
        try:
            return str(path.relative_to(base))
        except ValueError:
            return path.name

    def _on_selection_mode_change(self, _event=None) -> None:
        if self._running:
            return
        self._refresh_analysis_hint()
        self._states.clear()
        self._rebuild_table()
        self._status_lbl.config(
            text=f"Selected mode: {self._selection_mode_var.get()}",
            foreground=_C_NORMAL,
        )

    def _on_date_change(self, _event=None) -> None:
        self._date_str = self._date_var.get().strip()
        self._refresh_analysis_hint()
        self._states.clear()
        self._rebuild_table()

        if self._date_str:
            self._status_lbl.config(
                text=(
                    f"Selected date: {self._date_str}  "
                    f"·  {len(self._available_analysis_ids)} AnalysisID(s)"
                ),
                foreground=_C_NORMAL,
            )
        else:
            self._status_lbl.config(
                text="No valid date folders found",
                foreground=_C_FAIL,
            )

    # ── Run QC ─────────────────────────────────────────────────────────────────

    def _selected_qc_targets(self) -> list[tuple[str, Path, str]]:
        selected_indices = list(self._analysis_listbox.curselection())
        if selected_indices:
            return [
                self._available_analysis_targets[index]
                for index in selected_indices
                if 0 <= index < len(self._available_analysis_targets)
            ]
        return list(self._available_analysis_targets)

    def _run_qc(self) -> None:
        if self._running:
            return
        self._refresh_date_options()
        targets = self._selected_qc_targets()

        if not targets:
            messagebox.showerror("QC Error", "Select at least one AnalysisID to process.")
            return

        self._running = True
        self._stop_event.clear()
        self._run_btn.config(state=tk.DISABLED)
        self._stop_btn.config(state=tk.NORMAL)
        self._issue_btn.config(state=tk.DISABLED)
        self._force_run_check.config(state=tk.DISABLED)
        self._date_combo.config(state=tk.DISABLED)
        self._selection_mode_combo.config(state=tk.DISABLED)
        self._analysis_listbox.config(state=tk.DISABLED)
        self._progress.configure(maximum=len(targets), value=0)
        self._status_lbl.config(
            text=f"Running QC: 0/{len(targets)} AnalysisID(s)",
            foreground=_C_NORMAL,
        )
        self._current_lbl.config(text="Current: -")
        self._retake_manager.clear()
        self._states = [
            _AnalysisState(
                analysis_id=analysis_id,
                analysis_path=date_path / analysis_id,
                date_str=date_str,
                customer_code=self._customer_code_for_analysis(analysis_id),
                run_status="Queued",
            )
            for date_str, date_path, analysis_id in targets
        ]
        self._rebuild_table()
        self._worker = threading.Thread(
            target=self._qc_worker,
            args=(targets, bool(self._force_run_var.get())),
            daemon=True,
        )
        self._worker.start()

    def _request_stop(self) -> None:
        if not self._running:
            return
        self._stop_event.set()
        self._status_lbl.config(text="Stopping after current AnalysisID...", foreground=_C_FAIL)

    def _qc_worker(self, targets: list[tuple[str, Path, str]], force_qc: bool) -> None:
        try:
            states: list[_AnalysisState] = []
            total = len(targets)
            for index, (date_str, date_path, analysis_id) in enumerate(targets, start=1):
                if self._stop_event.is_set():
                    self._queue.put(("stopped", (states, index - 1, total)))
                    return
                self._queue.put(("current", (analysis_id, index - 1, total)))
                state = self._process_one(analysis_id, date_str, date_path, force_qc=force_qc)
                state.run_status = "Done"
                states.append(state)
                self._queue.put(("progress", (state, index, total)))

            self._queue.put(("done", states))

        except Exception as exc:
            logger.exception("QC worker error")
            self._queue.put(("error", str(exc)))

    def _process_one(
        self, analysis_id: str, date_str: str, date_path: Path, force_qc: bool = False
    ) -> _AnalysisState:
        """Run FolderQC + CropEngine + ColorNormalizer for one AnalysisID."""
        analysis_path = date_path / analysis_id
        state = _AnalysisState(
            analysis_id=analysis_id,
            analysis_path=analysis_path,
            date_str=date_str,
            customer_code=self._customer_code_for_analysis(analysis_id),
            run_status="Running",
        )
        if is_test_analysis_id(analysis_id):
            ensure_test_output_folders(analysis_path)

        # FolderQC
        fq = self._folder_qc.validate(
            root_path=str(self._root_path),
            date_str=date_str,
            analysis_id=analysis_id,
            force=force_qc,
        )
        state.folder_qc_result = fq
        if fq.status == "FAIL":
            return state

        if is_test_analysis_id(analysis_id):
            self._process_test_analysis(state)
            return state

        # Clear stale retake items for this analysis before re-running
        self._retake_manager.remove_by_analysis(analysis_path)

        customer_code = self._customer_code_for_analysis(analysis_id)
        image_postprocess = (
            self._color_normalizer.normalize
            if self._color_normalizer is not None
            else None
        )
        for cell in fq.cells:
            for face_name, folder_path in [
                ("FRONT", cell.front_path),
                ("BACK",  cell.back_path),
            ]:
                if folder_path is None:
                    continue
                output_folder = folder_path.parent / f"{face_name}_pass"
                crop_results = self._crop_engine.process_folder(
                    folder_path=folder_path,
                    customer_code=customer_code,
                    output_folder=output_folder,
                    image_postprocess=image_postprocess,
                )
                self._retake_manager.load(
                    crop_results, folder_path, customer_code, output_folder
                )

        state.crop_done = True
        state.retake_items = [
            item for item in self._retake_manager.get_pending_items()
            if _is_subpath(item.folder_path, analysis_path)
        ]
        return state

    def _customer_code_for_analysis(self, analysis_id: str) -> str:
        return "T" if is_test_analysis_id(analysis_id) else analysis_id[1]

    def _process_test_analysis(self, state: _AnalysisState) -> None:
        analysis_path = state.analysis_path
        self._retake_manager.remove_by_analysis(analysis_path)
        customer_code = self._customer_code_for_analysis(state.analysis_id)
        image_postprocess = (
            self._color_normalizer.normalize
            if self._color_normalizer is not None
            else None
        )
        folder_result = state.folder_qc_result
        source_folders = [
            cell.front_path
            for cell in folder_result.cells
            if cell.front_path is not None
        ] if folder_result is not None else [
            source_path
            for _, source_path in find_test_source_folders(analysis_path)
        ]
        for source_folder in source_folders:
            output_folder, mask_folder = get_test_output_folders(
                analysis_path,
                source_folder,
            )
            output_folder.mkdir(parents=True, exist_ok=True)
            mask_folder.mkdir(parents=True, exist_ok=True)
            crop_results: list[CropResult] = []
            output_paths: dict[Path, Path] = {}
            for stale_path in output_folder.iterdir():
                if stale_path.is_file() and stale_path.suffix.lower() == ".png":
                    stale_path.unlink()
            for image_path in find_test_images(source_folder, _IMAGE_EXTENSIONS):
                output_path = output_folder / build_test_output_name(
                    source_folder,
                    image_path,
                )
                output_paths[image_path] = output_path
                crop_results.append(
                    self._crop_engine.process_image(
                        image_path=image_path,
                        customer_code=customer_code,
                        output_path=output_path,
                        skip_size_validation=True,
                        image_postprocess=image_postprocess,
                    )
                )
            self._retake_manager.load(
                crop_results,
                source_folder,
                customer_code,
                output_folder,
                output_paths=output_paths,
                skip_size_validation_for_all=True,
            )
        state.crop_done = True
        state.retake_items = [
            item for item in self._retake_manager.get_pending_items()
            if _is_subpath(item.folder_path, analysis_path)
        ]

    # ── Refresh (single AnalysisID) ────────────────────────────────────────────

    def _refresh_analysis(self, state: _AnalysisState) -> None:
        date_str = state.date_str or self._date_var.get().strip()
        date_path = self._root_path / date_str
        if not date_str or not date_path.is_dir():
            messagebox.showerror(
                "Error",
                f"Date folder not found: {date_str or '-'}\nRoot: {self._root_path}",
            )
            return

        self._run_btn.config(state=tk.DISABLED)
        self._issue_btn.config(state=tk.DISABLED)
        self._date_combo.config(state=tk.DISABLED)
        self._status_lbl.config(
            text=f"Re-running: {state.analysis_id}", foreground=_C_NORMAL
        )

        def worker():
            try:
                new_state = self._process_one(state.analysis_id, date_str, date_path)
                self._queue.put(("refresh_done", (state.analysis_id, new_state)))
            except Exception as exc:
                logger.exception("Refresh error")
                self._queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    # ── Retake checkbox ────────────────────────────────────────────────────────

    def _on_retake_check(
        self,
        state: _AnalysisState,
        item: RetakeItem,
        var: tk.BooleanVar,
    ) -> None:
        if not var.get():
            return
        self._retry_retake_item(state, item)

    def _retry_retake_item(self, state: _AnalysisState, item: RetakeItem) -> None:
        self._run_btn.config(state=tk.DISABLED)
        self._issue_btn.config(state=tk.DISABLED)
        self._date_combo.config(state=tk.DISABLED)
        self._status_lbl.config(
            text=f"Re-validating: {item.image_path.name}", foreground=_C_NORMAL
        )

        def worker():
            try:
                self._retake_manager.retry_image(item)
                new_retake = [
                    i for i in self._retake_manager.get_pending_items()
                    if _is_subpath(i.folder_path, state.analysis_path)
                ]
                self._queue.put(("retake_done", (state.analysis_id, new_retake)))
            except Exception as exc:
                logger.exception("Retake error")
                self._queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    # ── Queue polling ──────────────────────────────────────────────────────────

    def _force_crop_image(self, state: _AnalysisState, item: RetakeItem) -> None:
        if not messagebox.askyesno(
            "Manual Crop Override",
            "QC FAIL reason will be ignored and the cropped output will be saved "
            "if crop extraction is possible.\n\nContinue?",
        ):
            return

        self._run_btn.config(state=tk.DISABLED)
        self._issue_btn.config(state=tk.DISABLED)
        self._date_combo.config(state=tk.DISABLED)
        self._status_lbl.config(
            text=f"Manual crop: {item.image_path.name}", foreground=_C_NORMAL
        )

        def worker():
            try:
                result = self._retake_manager.force_crop_image(item)
                new_retake = [
                    i for i in self._retake_manager.get_pending_items()
                    if _is_subpath(i.folder_path, state.analysis_path)
                ]
                self._queue.put(
                    (
                        "force_crop_done",
                        (
                            state.analysis_id,
                            new_retake,
                            result.status,
                            list(result.fail_reasons),
                        ),
                    )
                )
            except Exception as exc:
                logger.exception("Manual crop override error")
                self._queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                msg_type, payload = self._queue.get_nowait()
                self._dispatch(msg_type, payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _dispatch(self, msg_type: str, payload) -> None:
        if msg_type == "error":
            self._status_lbl.config(text=f"Error: {payload}", foreground=_C_FAIL)
            messagebox.showerror("QC Error", payload)
            self._finish_run()

        elif msg_type == "current":
            analysis_id, processed, total = payload
            self._progress.configure(value=processed)
            self._status_lbl.config(
                text=f"Running QC: {processed}/{total} AnalysisID(s)",
                foreground=_C_NORMAL,
            )
            self._current_lbl.config(text=f"Current: {analysis_id}")
            for state in self._states:
                if state.analysis_id == analysis_id:
                    state.run_status = "Running"
                    break
            self._rebuild_table()

        elif msg_type == "progress":
            state, processed, total = payload
            self._progress.configure(value=processed)
            self._status_lbl.config(
                text=f"Running QC: {processed}/{total} AnalysisID(s)",
                foreground=_C_NORMAL,
            )
            self._upsert_state(state)
            self._rebuild_table()

        elif msg_type == "done":
            self._states = list(payload)
            self._rebuild_table()
            self._update_status()
            self._current_lbl.config(text="Current: -")
            self._progress.configure(value=len(self._states))
            self._finish_run()

        elif msg_type == "refresh_done":
            analysis_id, new_state = payload
            self._upsert_state(new_state)
            self._rebuild_table()
            self._update_status()
            self._finish_run()

        elif msg_type == "retake_done":
            analysis_id, new_retake_items = payload
            for s in self._states:
                if s.analysis_id == analysis_id:
                    s.retake_items = new_retake_items
                    break
            self._rebuild_table()
            self._update_status()
            self._finish_run()
            self._status_lbl.config(text="Re-validation complete", foreground=_C_NORMAL)

        elif msg_type == "force_crop_done":
            analysis_id, new_retake_items, status, fail_reasons = payload
            for s in self._states:
                if s.analysis_id == analysis_id:
                    s.retake_items = new_retake_items
                    break
            self._rebuild_table()
            self._update_status()
            self._finish_run()
            if status == "PASS":
                self._status_lbl.config(
                    text="Manual crop override complete",
                    foreground=_C_NORMAL,
                )
            else:
                reason_text = "; ".join(fail_reasons) or "Crop extraction failed"
                self._status_lbl.config(
                    text=f"Manual crop failed: {reason_text}",
                    foreground=_C_FAIL,
                )
                messagebox.showwarning(
                    "Manual Crop Override",
                    f"Could not save a crop for this image.\n\n{reason_text}",
                )

        elif msg_type == "stopped":
            states, processed, total = payload
            for state in states:
                self._upsert_state(state)
            for state in self._states:
                if state.folder_qc_result is None and state.run_status == "Queued":
                    state.run_status = "Stopped"
            self._rebuild_table()
            self._progress.configure(value=processed)
            self._current_lbl.config(text="Current: stopped")
            self._status_lbl.config(
                text=f"QC stopped: {processed}/{total} AnalysisID(s) processed",
                foreground=_C_FAIL,
            )
            self._finish_run()

    def _finish_run(self) -> None:
        self._running = False
        self._stop_btn.config(state=tk.DISABLED)
        self._run_btn.config(state=tk.NORMAL)
        self._issue_btn.config(state=tk.NORMAL)
        self._force_run_check.config(state=tk.NORMAL)
        self._selection_mode_combo.config(state="readonly")
        self._analysis_listbox.config(state=tk.NORMAL)
        self._date_combo.config(
            state=tk.DISABLED
            if self._selection_mode_var.get() == "By Analysis ID"
            else "readonly"
        )

    def _upsert_state(self, new_state: _AnalysisState) -> None:
        for i, s in enumerate(self._states):
            if s.analysis_id == new_state.analysis_id:
                self._states[i] = new_state
                return
        self._states.append(new_state)

    def _update_status(self) -> None:
        if not self._states:
            self._status_lbl.config(text="", foreground=_C_NORMAL)
            return
        if all(s.all_pass for s in self._states):
            self._status_lbl.config(text="All PASS ✓", foreground=_C_PASS)
        else:
            n = sum(1 for s in self._states if not s.all_pass)
            self._status_lbl.config(
                text=f"Done  ·  {n} AnalysisID(s) FAIL", foreground=_C_FAIL
            )

    # ── Table rendering ────────────────────────────────────────────────────────

    def _clear_table_body(self) -> None:
        for widget_list in self._row_widgets:
            for w in widget_list:
                try:
                    w.destroy()
                except tk.TclError:
                    pass
        self._row_widgets.clear()

    def _rebuild_table(self) -> None:
        self._clear_table_body()

        if not self._states:
            return

        gr = self._data_row
        for idx, state in enumerate(self._states):
            bg = _BG_EVEN if idx % 2 == 0 else _BG_ODD
            gr = self._render_state(state, gr, bg)

        self._tbl.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        self._sync_review_panel()

    def _show_all_pass_banner(self) -> None:
        lbl = tk.Label(
            self._tbl,
            text="ALL PASS  ✓",
            font=("TkDefaultFont", 18, "bold"),
            fg=_C_PASS, bg=_BG_EVEN, pady=40,
        )
        lbl.grid(
            row=self._data_row, column=0,
            columnspan=len(_COL_DEFS), sticky=tk.EW, pady=40,
        )
        self._row_widgets.append([lbl])

    def _render_state(self, state: _AnalysisState, gr: int, bg: str) -> int:
        if state.folder_qc_result is None:
            return self._render_pending_row(state, gr, bg)
        if not state.folder_qc_pass:
            return self._render_folder_fail(state, gr, bg)
        if state.crop_all_pass:
            return self._render_pass_row(state, gr, bg)
        if state.retake_items:
            return self._render_crop_fails(state, gr, bg)
        return self._render_pass_row(state, gr, bg)

    # ── Row builders ───────────────────────────────────────────────────────────

    def _render_pending_row(self, state: _AnalysisState, gr: int, bg: str) -> int:
        widgets = [
            self._cell(gr, _CI["AnalysisID"], state.analysis_id, bg=bg),
            self._cell(gr, _CI["FolderQC"], state.run_status, _C_MUTED, bg=bg),
            self._cell(gr, _CI["CropEngine"], "-", _C_MUTED, bg=bg),
            self._cell(gr, _CI["FAIL Image"], "-", _C_MUTED, bg=bg),
            self._cell(gr, _CI["FAIL Reason"], state.run_status, _C_MUTED, bg=bg),
            self._cell(gr, _CI["Retake"], "-", _C_MUTED, bg=bg),
        ]
        self._row_widgets.append(widgets)
        return gr + 1

    def _render_pass_row(self, state: _AnalysisState, gr: int, bg: str) -> int:
        ce_text  = "● PASS" if state.crop_done else "-"
        ce_color = _C_PASS  if state.crop_done else _C_MUTED
        widgets = [
            self._cell(gr, _CI["AnalysisID"],  state.analysis_id,        bg=bg),
            self._cell(gr, _CI["FolderQC"],    "● PASS", _C_PASS,        bg=bg),
            self._cell(gr, _CI["CropEngine"],  ce_text,  ce_color,        bg=bg),
            self._cell(gr, _CI["FAIL Image"],  "-",      _C_MUTED,        bg=bg),
            self._cell(gr, _CI["FAIL Reason"], "-",      _C_MUTED,        bg=bg),
            self._cell(gr, _CI["Retake"],      "-",      _C_MUTED,        bg=bg),
        ]
        self._row_widgets.append(widgets)
        return gr + 1

    def _render_folder_fail(self, state: _AnalysisState, gr: int, bg: str) -> int:
        reasons     = (state.folder_qc_result.fail_reasons
                       if state.folder_qc_result else ["Unknown error"])
        reason_text = "\n".join(reasons)

        retake_fr = self._retake_frame(gr, bg)
        ttk.Button(
            retake_fr, text="Refresh", width=9,
            command=lambda s=state: self._refresh_analysis(s),
        ).pack(anchor=tk.W, padx=4, pady=4)

        widgets = [
            self._cell(gr, _CI["AnalysisID"],  state.analysis_id,        bg=bg),
            self._cell(gr, _CI["FolderQC"],    "● FAIL", _C_FAIL,        bg=bg),
            self._cell(gr, _CI["CropEngine"],  "-",      _C_MUTED,       bg=bg),
            self._cell(gr, _CI["FAIL Image"],  "-",      _C_MUTED,       bg=bg),
            self._cell(gr, _CI["FAIL Reason"], reason_text, _C_FAIL,     bg=bg, wrap=228),
            retake_fr,
        ]
        self._row_widgets.append(widgets)
        return gr + 1

    def _render_crop_fails(self, state: _AnalysisState, gr: int, bg: str) -> int:
        items = state.retake_items
        first = True

        for idx, item in enumerate(items):
            is_last = idx == len(items) - 1

            try:
                rel_path = str(item.image_path.relative_to(state.analysis_path))
            except ValueError:
                rel_path = item.image_path.name

            reason_text = " | ".join(item.fail_reasons)

            retake_fr = self._retake_frame(gr, bg)
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                retake_fr, text="Retake complete", variable=var,
                command=lambda s=state, it=item, v=var: self._on_retake_check(s, it, v),
            ).pack(anchor=tk.W, padx=4, pady=(4, 1))

            action_fr = tk.Frame(retake_fr, bg=bg)
            action_fr.pack(anchor=tk.W, padx=4, pady=(1, 4))
            ttk.Button(
                action_fr, text="Crop anyway", width=12,
                command=lambda s=state, it=item: self._force_crop_image(s, it),
            ).pack(side=tk.LEFT)

            if is_last:
                ttk.Button(
                    action_fr, text="Refresh", width=9,
                    command=lambda s=state: self._refresh_analysis(s),
                ).pack(side=tk.LEFT, padx=(6, 0))

            image_cell = self._cell(
                gr, _CI["FAIL Image"], rel_path, _C_NORMAL, bg=bg, wrap=208
            )
            image_cell.config(cursor="hand2")
            image_cell.bind(
                "<Button-1>",
                lambda _event, s=state, it=item: self._select_review_item(s, it),
            )

            widgets = [
                self._cell(gr, _CI["AnalysisID"],
                           state.analysis_id if first else "",            bg=bg),
                self._cell(gr, _CI["FolderQC"],
                           "● PASS" if first else "",     _C_PASS,       bg=bg),
                self._cell(gr, _CI["CropEngine"],
                           "● FAIL" if first else "",     _C_FAIL,       bg=bg),
                image_cell,
                self._cell(gr, _CI["FAIL Reason"], reason_text, _C_FAIL, bg=bg, wrap=228),
                retake_fr,
            ]
            self._row_widgets.append(widgets)
            gr += 1
            first = False

        return gr

    # ── Cell helpers ───────────────────────────────────────────────────────────

    def _cell(
        self,
        grid_row: int,
        col: int,
        text: str,
        fg: str = _C_NORMAL,
        bg: str = _BG_EVEN,
        wrap: int = 0,
    ) -> tk.Label:
        width_px = _COL_DEFS[col][1]
        lbl = tk.Label(
            self._tbl,
            text=text, fg=fg, bg=bg,
            anchor=tk.W, justify=tk.LEFT,
            wraplength=wrap or (width_px - 12),
            padx=6, pady=4,
        )
        lbl.grid(row=grid_row, column=col, sticky=tk.NSEW, padx=1, pady=1)
        return lbl

    def _retake_frame(self, grid_row: int, bg: str) -> tk.Frame:
        fr = tk.Frame(self._tbl, bg=bg)
        fr.grid(row=grid_row, column=_CI["Retake"], sticky=tk.NSEW, padx=1, pady=1)
        return fr

    # ── Fail image review panel ───────────────────────────────────────────────

    def _review_items(self) -> list[tuple[_AnalysisState, RetakeItem]]:
        items: list[tuple[_AnalysisState, RetakeItem]] = []
        for state in self._states:
            for item in state.retake_items:
                items.append((state, item))
        return items

    def _review_key(self, state: _AnalysisState, item: RetakeItem) -> tuple[str, str]:
        return (state.analysis_id, str(item.image_path))

    def _selected_review_item(self) -> tuple[_AnalysisState, RetakeItem] | None:
        items = self._review_items()
        if not items:
            return None
        if self._review_selection_key is None:
            return items[0]
        for state, item in items:
            if self._review_key(state, item) == self._review_selection_key:
                return state, item
        return items[0]

    def _sync_review_panel(self) -> None:
        selected = self._selected_review_item()
        if selected is None:
            self._review_selection_key = None
            self._review_info_lbl.config(text="No fail image selected.")
            self._review_reason_lbl.config(text="")
            self._review_canvas.delete("all")
            self._review_canvas.create_text(
                12, 12, text="No fail image selected.", anchor=tk.NW, fill=_C_MUTED
            )
            self._set_review_controls_enabled(False)
            return

        state, item = selected
        self._select_review_item(state, item)

    def _select_review_item(self, state: _AnalysisState, item: RetakeItem) -> None:
        self._review_selection_key = self._review_key(state, item)
        items = self._review_items()
        index = 0
        total = len(items)
        for i, (candidate_state, candidate_item) in enumerate(items, start=1):
            if self._review_key(candidate_state, candidate_item) == self._review_selection_key:
                index = i
                break

        try:
            rel_path = str(item.image_path.relative_to(state.analysis_path))
        except ValueError:
            rel_path = item.image_path.name

        self._review_info_lbl.config(
            text=f"{index}/{total}  {state.analysis_id}  {rel_path}"
        )
        self._review_reason_lbl.config(text=" | ".join(item.fail_reasons))
        self._set_review_controls_enabled(True)
        self._draw_review_image()

    def _select_review_offset(self, offset: int) -> None:
        items = self._review_items()
        if not items:
            self._sync_review_panel()
            return
        current = self._selected_review_item()
        current_key = self._review_key(*current) if current is not None else None
        index = 0
        if current_key is not None:
            for i, (state, item) in enumerate(items):
                if self._review_key(state, item) == current_key:
                    index = i
                    break
        next_index = max(0, min(len(items) - 1, index + offset))
        state, item = items[next_index]
        self._select_review_item(state, item)

    def _draw_review_image(self) -> None:
        if not hasattr(self, "_review_canvas"):
            return

        self._review_canvas.delete("all")
        selected = self._selected_review_item()
        width = max(self._review_canvas.winfo_width(), 300)
        height = max(self._review_canvas.winfo_height(), 220)
        if selected is None:
            self._review_canvas.create_text(
                width / 2, height / 2, text="No fail image selected.", fill=_C_MUTED
            )
            return

        _state, item = selected
        image = quiet_imread(item.image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            self._review_canvas.create_text(
                width / 2, height / 2, text="Failed to load image.", fill=_C_MUTED
            )
            return

        image_h, image_w = image.shape[:2]
        max_w = max(1, width - 16)
        max_h = max(1, height - 16)
        scale = min(max_w / image_w, max_h / image_h, 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(image_w * scale)), max(1, int(image_h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            self._review_canvas.create_text(
                width / 2, height / 2, text="Failed to render image.", fill=_C_MUTED
            )
            return

        self._review_photo = tk.PhotoImage(data=buffer.tobytes())
        self._review_canvas.create_image(width / 2, height / 2, image=self._review_photo)

    def _set_review_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in (
            self._review_prev_btn,
            self._review_next_btn,
            self._review_retake_btn,
            self._review_crop_btn,
        ):
            button.config(state=state)

    def _review_retake_current(self) -> None:
        selected = self._selected_review_item()
        if selected is None:
            return
        state, item = selected
        self._retry_retake_item(state, item)

    def _review_force_crop_current(self) -> None:
        selected = self._selected_review_item()
        if selected is None:
            return
        state, item = selected
        self._force_crop_image(state, item)
