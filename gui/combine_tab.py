from modules.utils.analysis_id import compile_cell_pattern
import logging
import os
import queue
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from urllib.parse import unquote, urlparse

import yaml

from modules.report.report_combiner import (
    IMAGE_EXTENSIONS,
    REPORT_GRID_LAYOUTS,
    REPORT_SOURCE_MASK,
    REPORT_SOURCE_PASS,
    ReportCombiner,
)
from modules.utils.analysis_id import is_test_analysis_id
from modules.report.report_data import analysis_paths
from modules.report.manual_combine import combine_analysis

logger = logging.getLogger(__name__)


class CombineTab(ttk.Frame):
    def __init__(self, parent, config_path: str | Path, can_run=None) -> None:
        super().__init__(parent)
        self._can_run = can_run or (lambda: True)
        self._config_path = Path(config_path)
        self._combiner = ReportCombiner(self._config_path)
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._external_paths: list[Path] = []
        self._external_detected_analysis_ids: list[str] = []
        self._analysis_targets: list[tuple[str, str]] = []

        self._load_drop_parser_config()
        self._build_ui()
        self._refresh_dates()
        self._enable_drop_target()
        self._poll_queue()

    def reload_settings(self) -> None:
        if self._running:
            messagebox.showwarning(
                "Report Images",
                "Report image generation is running. Wait for completion before reloading settings.",
            )
            return
        self._combiner = ReportCombiner(self._config_path)
        self._load_drop_parser_config()
        self._refresh_dates()
        self._status_label.config(text="Settings reloaded")

    def _load_drop_parser_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        folder_cfg = cfg["folder_qc"]
        self._analysis_id_re = re.compile(
            folder_cfg["analysis_id_pattern"], re.IGNORECASE
        )
        self._cell_folder_re = compile_cell_pattern(
            folder_cfg.get("cell_folder_pattern", r"^cell\s*#\d+$")
        )

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        selector = ttk.Frame(self, style="Card.TFrame", padding=14)
        selector.grid(row=0, column=0, sticky=tk.EW, pady=(8, 12))
        selector.columnconfigure(1, weight=1)
        self._mode_var = tk.StringVar(value="By Analysis ID")
        self._mode_combo = ttk.Combobox(
            selector, textvariable=self._mode_var, values=("By Analysis ID",)
        )
        self._date_var = tk.StringVar()
        self._date_combo = ttk.Combobox(selector, textvariable=self._date_var)
        ttk.Label(selector, text="Analysis ID", style="Panel.TLabel").grid(
            row=0, column=0, padx=(0, 10)
        )
        self._analysis_var = tk.StringVar()
        self._analysis_combo = ttk.Combobox(
            selector, textvariable=self._analysis_var, state="readonly", width=40
        )
        self._analysis_combo.grid(row=0, column=1, sticky=tk.EW, padx=(0, 12))
        self._analysis_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._on_analysis_change()
        )
        self._refresh_btn = ttk.Button(
            selector, text="Refresh", command=self._refresh_dates
        )
        self._refresh_btn.grid(row=0, column=2)
        self._clear_path_btn = ttk.Button(
            selector, text="Clear external", command=self._clear_external_paths
        )
        self._clear_path_btn.grid(row=0, column=3, padx=(8, 0))
        options = ttk.Frame(selector, style="Panel.TFrame")
        options.grid(row=1, column=0, columnspan=4, sticky=tk.EW, pady=(12, 0))
        options.columnconfigure(6, weight=1)
        for col, title in [(0, "Columns"), (2, "Rows"), (4, "Source")]:
            ttk.Label(options, text=title, style="Panel.TLabel").grid(
                row=0, column=col, padx=(0, 8)
            )
        self._columns_var = tk.IntVar(value=3)
        self._rows_var = tk.IntVar(value=5)
        self._columns_spin = ttk.Spinbox(
            options,
            from_=1,
            to=12,
            textvariable=self._columns_var,
            width=5,
            command=self._sync_layout_preview,
        )
        self._columns_spin.grid(row=0, column=1, padx=(0, 16))
        self._rows_spin = ttk.Spinbox(
            options,
            from_=1,
            to=12,
            textvariable=self._rows_var,
            width=5,
            command=self._sync_layout_preview,
        )
        self._rows_spin.grid(row=0, column=3, padx=(0, 16))
        self._columns_spin.bind("<KeyRelease>", lambda _e: self._sync_layout_preview())
        self._rows_spin.bind("<KeyRelease>", lambda _e: self._sync_layout_preview())
        self._source_var = tk.StringVar(value="_BOTH")
        self._source_combo = ttk.Combobox(
            options,
            textvariable=self._source_var,
            values=("_BOTH", "_PASS", "_MASK"),
            state="readonly",
            width=9,
        )
        self._source_combo.grid(row=0, column=5)
        self._source_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._show_existing_combined_outputs()
        )
        self._run_btn = ttk.Button(
            options,
            text="Create combined",
            style="Accent.TButton",
            command=self._run_report,
        )
        self._run_btn.grid(row=0, column=7, sticky=tk.E)
        self._customer_label = ttk.Label(selector, text="", style="Panel.TLabel")
        self._customer_label.grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0)
        )
        self._layout_label = ttk.Label(selector, text="", style="Panel.TLabel")
        self._layout_label.grid(
            row=2, column=2, columnspan=2, sticky=tk.E, pady=(10, 0)
        )
        self._drop_label = tk.Label(
            self,
            text="Drop analysis folders or FRONT / BACK images here",
            bg="#153b63",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            pady=14,
            highlightthickness=2,
            highlightbackground="#3b82f6",
        )
        self._drop_label.grid(row=1, column=0, sticky=tk.EW, pady=(0, 12))
        middle = ttk.Frame(self)
        middle.grid(row=2, column=0, sticky=tk.EW, pady=(0, 12))
        middle.columnconfigure(1, weight=1)
        cells = ttk.LabelFrame(middle, text="Cells", padding=10)
        cells.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 12))
        self._cell_listbox = tk.Listbox(
            cells,
            selectmode=tk.EXTENDED,
            exportselection=False,
            height=5,
            width=24,
            bd=0,
            highlightthickness=0,
        )
        self._cell_listbox.pack(fill=tk.BOTH, expand=True)
        self._cell_listbox.bind(
            "<<ListboxSelect>>", lambda _e: self._show_existing_combined_outputs()
        )
        grid = ttk.LabelFrame(middle, text="Grid layout", padding=10)
        grid.grid(row=0, column=1, sticky=tk.NSEW)
        grid.columnconfigure(0, weight=1)
        self._preview_canvas = tk.Canvas(
            grid, height=105, bg="white", highlightthickness=0
        )
        self._preview_canvas.grid(row=0, column=0, sticky=tk.EW)
        self._preview_canvas.bind("<Configure>", lambda _e: self._sync_layout_preview())
        self._dropped_tree = ttk.Treeview(
            self,
            columns=("analysis_id", "cell", "face", "path"),
            show="headings",
            height=2,
        )
        # External-drop metadata is retained by the existing import / staging code.
        output = ttk.LabelFrame(self, text="Combined outputs", padding=10)
        output.grid(row=3, column=0, sticky=tk.NSEW)
        output.rowconfigure(0, weight=1)
        output.columnconfigure(0, weight=1)
        self._output_listbox = tk.Listbox(output, bd=0, highlightthickness=0, height=6)
        self._output_listbox.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar = ttk.Scrollbar(
            output, orient=tk.VERTICAL, command=self._output_listbox.yview
        )
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self._output_listbox.configure(yscrollcommand=scrollbar.set)
        footer = ttk.Frame(self)
        footer.grid(row=4, column=0, sticky=tk.EW, pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        self._status_label = ttk.Label(footer, text="Ready", style="Muted.TLabel")
        self._status_label.grid(row=0, column=0, sticky=tk.W)
        self._progress = ttk.Progressbar(footer, mode="indeterminate", length=100)
        self._progress.grid(row=0, column=1, padx=12)
        self._progress.grid_remove()
        self._clear_report_log_btn = ttk.Button(
            footer, text="Clear log", command=self._clear_report_log
        )
        self._clear_report_log_btn.grid(row=0, column=2)

    def _refresh_dates(self) -> None:
        dates = self._combiner.list_date_folders()
        previous = self._date_var.get()
        self._date_combo["values"] = dates
        self._date_var.set(
            previous if previous in dates else (dates[0] if dates else "")
        )
        self._refresh_analysis_ids()

    def _on_mode_change(self) -> None:
        self._date_combo.config(
            state=tk.DISABLED
            if self._mode_var.get() == "By Analysis ID"
            else "readonly"
        )
        self._refresh_analysis_ids()

    def _refresh_analysis_ids(self) -> None:
        if self._mode_var.get() == "By Analysis ID":
            self._analysis_targets = self._list_all_analysis_targets()
            labels = [
                f"{analysis_id} ({date_name})"
                for date_name, analysis_id in self._analysis_targets
            ]
        else:
            date_name = self._date_var.get()
            self._analysis_targets = [
                (date_name, analysis_id)
                for analysis_id in self._combiner.list_analysis_ids(date_name)
            ]
            labels = [analysis_id for _date_name, analysis_id in self._analysis_targets]
        previous = self._analysis_var.get()
        self._analysis_combo["values"] = labels
        self._analysis_var.set(
            previous if previous in labels else (labels[0] if labels else "")
        )
        self._on_analysis_change()

    def _list_all_analysis_targets(self) -> list[tuple[str, str]]:
        root = self._combiner.root_path
        return [
            (
                str(p.parent.relative_to(root))
                if p.parent.is_relative_to(root)
                else "..",
                p.name,
            )
            for p in analysis_paths(root, self._analysis_id_re.pattern)
        ]

    def _on_analysis_change(self) -> None:
        selected = self._selected_target()
        analysis_id = selected[1] if selected is not None else ""
        self._apply_analysis_context(analysis_id)
        self._refresh_cells()
        self._show_existing_combined_outputs()

    def _selected_target(self) -> tuple[str, str] | None:
        label = self._analysis_var.get().strip()
        if not label:
            return None
        labels = list(self._analysis_combo["values"])
        try:
            index = labels.index(label)
        except ValueError:
            return None
        if 0 <= index < len(self._analysis_targets):
            return self._analysis_targets[index]
        return None

    def _refresh_cells(self) -> None:
        self._cell_listbox.delete(0, tk.END)
        selected = self._selected_target()
        if selected is None:
            return
        date_name, analysis_id = selected
        analysis_path = self._combiner.root_path / date_name / analysis_id
        for cell_name, _cell_path in self._combiner.list_cells(analysis_path):
            self._cell_listbox.insert(tk.END, cell_name)

    def _show_existing_combined_outputs(self) -> None:
        if not hasattr(self, "_output_listbox") or self._running:
            return
        self._output_listbox.delete(0, tk.END)
        existing = self._existing_combined_outputs()
        if not existing:
            self._status_label.config(text="No existing combined images")
            return
        self._status_label.config(text=f"Existing combined images: {len(existing)}")
        for path in existing:
            self._output_listbox.insert(tk.END, f"[EXISTS] {path}")

    def _existing_combined_outputs(self) -> list[Path]:
        selected = self._selected_target()
        if selected is None:
            return []
        date_name, analysis_id = selected
        analysis_path = self._combiner.root_path / date_name / analysis_id
        selected_cells = set(self._selected_cell_names() or [])
        source_kind = self._selected_source_kind()
        paths: list[Path] = []
        for cell_name, cell_path in self._combiner.list_cells(analysis_path):
            if selected_cells and cell_name not in selected_cells:
                continue
            for face in ("FRONT", "BACK"):
                for kind in (
                    ("PASS", "MASK") if source_kind == "BOTH" else (source_kind,)
                ):
                    prefix = f"{face}_mask" if kind == "MASK" else face
                    folder = cell_path / f"{prefix}_combined"
                    paths.extend(sorted(folder.glob(f"{prefix}_combined_*.png")))
        return sorted(paths, key=lambda path: str(path).lower())

    def _selected_source_kind(self) -> str:
        return self._source_var.get().strip().upper().lstrip("_")

    def _sync_layout_preview(self) -> None:
        canvas = self._preview_canvas
        if not hasattr(self, "_preview_canvas"):
            return
        canvas.delete("all")
        try:
            columns = max(1, int(self._columns_var.get()))
            rows = max(1, int(self._rows_var.get()))
        except tk.TclError:
            return
        self._layout_label.config(text=f"Layout: {columns} x {rows}")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 30 or height < 30:
            return
        margin = 12
        gap = 5
        cell_w = (width - margin * 2 - gap * (columns - 1)) / columns
        cell_h = (height - margin * 2 - gap * (rows - 1)) / rows
        for row in range(rows):
            for col in range(columns):
                x1 = margin + col * (cell_w + gap)
                y1 = margin + row * (cell_h + gap)
                x2 = x1 + cell_w
                y2 = y1 + cell_h
                canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#9ca3af", fill="#f8fafc"
                )

    def _add_external_paths(self) -> None:
        folder = filedialog.askdirectory(
            title="Select FRONT/BACK folder or AnalysisID folder"
        )
        selected: list[Path] = []
        if folder:
            selected.append(Path(folder))
        files = filedialog.askopenfilenames(
            title="Select FRONT/BACK image files",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        selected.extend(Path(path) for path in files)
        if selected:
            self._external_paths = selected
            self._update_external_label()

    def _clear_external_paths(self) -> None:
        self._external_paths = []
        self._external_detected_analysis_ids = []
        self._update_external_label()
        self._on_analysis_change()

    def _update_external_label(self) -> None:
        if not self._external_paths:
            self._drop_label.config(
                text="Drag and drop FRONT/BACK folder, AnalysisID folder, or FRONT/BACK image files here",
                bg="#153b63",
                fg="white",
            )
            self._refresh_dropped_items()
            return
        preview = "; ".join(str(path) for path in self._external_paths[:3])
        if len(self._external_paths) > 3:
            preview += "; ..."
        self._drop_label.config(text=f"Dropped: {preview}", bg="#065f46", fg="white")
        self._refresh_dropped_items()
        if len(self._external_detected_analysis_ids) == 1:
            self._apply_analysis_context(self._external_detected_analysis_ids[0])

    def _refresh_dropped_items(self) -> None:
        if not hasattr(self, "_dropped_tree"):
            return
        self._dropped_tree.delete(*self._dropped_tree.get_children())
        detected_ids: set[str] = set()
        for path in self._external_paths:
            analysis_id = self._analysis_id_from_path(path)
            cell_name = self._cell_from_path(path)
            face = self._face_from_path(path)
            if analysis_id:
                detected_ids.add(analysis_id)
            self._dropped_tree.insert(
                "",
                tk.END,
                values=(
                    analysis_id or "-",
                    cell_name or "-",
                    face or "-",
                    str(path),
                ),
            )
        self._external_detected_analysis_ids = sorted(detected_ids)

    def _analysis_id_from_path(self, path: Path) -> str:
        for part in reversed(path.parts):
            cleaned = part.strip().strip("{}")
            candidates = [cleaned, Path(cleaned).stem]
            candidates.extend(
                token for token in re.split(r"[^A-Za-z0-9]+", cleaned) if token
            )
            for candidate in candidates:
                value = candidate.strip()
                if not value:
                    continue
                if self._analysis_id_re.fullmatch(value) or is_test_analysis_id(value):
                    return value.upper()
        return ""

    def _cell_from_path(self, path: Path) -> str:
        for part in reversed(path.parts):
            cleaned = part.strip().strip("{}")
            if self._cell_folder_re.fullmatch(cleaned):
                number_match = re.search(r"(\d+)", cleaned)
                return f"Cell #{number_match.group(1)}" if number_match else cleaned
            match = re.search(r"\bcell\s*#?\s*(\d+)\b", cleaned, re.IGNORECASE)
            if match:
                return f"Cell #{match.group(1)}"
        return ""

    def _face_from_path(self, path: Path) -> str:
        for part in reversed(path.parts):
            cleaned = part.strip().strip("{}").upper()
            if re.search(r"(?<![A-Z])FRONT(?![A-Z])", cleaned):
                return "FRONT"
            if re.search(r"(?<![A-Z])BACK(?![A-Z])", cleaned):
                return "BACK"
        return ""

    def _apply_analysis_context(self, analysis_id: str) -> None:
        customer = (
            self._combiner.customer_for_analysis_id(analysis_id) if analysis_id else "-"
        )
        columns, rows = (
            self._combiner.default_layout_for_analysis_id(analysis_id)
            if analysis_id
            else (3, 5)
        )
        self._columns_var.set(columns)
        self._rows_var.set(rows)
        self._customer_label.config(text=f"Customer: {customer}")
        self._layout_label.config(text=f"Layout: {columns} x {rows}")
        self._sync_layout_preview()

    def _enable_drop_target(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES
        except Exception:
            self._drop_label.config(
                text=("Drag and drop unavailable · Select an Analysis ID above."),
                bg="#475569",
                fg="white",
            )
            return
        registered = False
        for widget in (self, self._drop_label, self._dropped_tree):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self._on_drop)
                registered = True
            except Exception:
                logger.debug(
                    "tkinterdnd2 drop target not available on %s", widget, exc_info=True
                )
        if not registered:
            self._drop_label.config(
                text=("Drag and drop unavailable · Select an Analysis ID above."),
                bg="#475569",
                fg="white",
            )

    def _on_drop(self, event) -> None:
        paths = [
            self._normalize_dropped_path(item) for item in self.tk.splitlist(event.data)
        ]
        paths = [path for path in paths if path is not None]
        self._external_paths = paths
        self._update_external_label()

    def _normalize_dropped_path(self, value: str) -> Path | None:
        value = value.strip()
        if not value:
            return None
        if value.lower().startswith("file:"):
            parsed = urlparse(value)
            if parsed.netloc:
                return Path(f"//{parsed.netloc}{unquote(parsed.path)}")
            return Path(unquote(parsed.path.lstrip("/")))
        return Path(value)

    def _selected_cell_names(self) -> list[str] | None:
        indices = self._cell_listbox.curselection()
        if not indices:
            return None
        return [self._cell_listbox.get(index) for index in indices]

    def _current_layout(self) -> tuple[int, int]:
        return max(1, int(self._columns_var.get())), max(1, int(self._rows_var.get()))

    def _run_report(self) -> None:
        if self._running:
            return
        if not self._can_run():
            messagebox.showinfo(
                "Combine", "Wait for the QC / ML workflow to finish first."
            )
            return
        selected = self._selected_target()
        if self._external_paths:
            if len(self._external_detected_analysis_ids) > 1:
                messagebox.showerror(
                    "Report Images",
                    "Dropped paths contain multiple AnalysisIDs. Drop one AnalysisID at a time.",
                )
                return
            analysis_id = (
                self._external_detected_analysis_ids[0]
                if self._external_detected_analysis_ids
                else (selected[1] if selected is not None else "")
            )
            date_name = selected[0] if selected is not None else ""
            if not analysis_id:
                messagebox.showerror(
                    "Report Images",
                    "Select an AnalysisID or drop a path that contains an AnalysisID.",
                )
                return
        elif selected is None:
            messagebox.showerror("Report Images", "Select an AnalysisID first.")
            return
        else:
            date_name, analysis_id = selected
        source_kind = self._selected_source_kind()
        self._task_cells = self._selected_cell_names()
        try:
            self._task_layout = self._current_layout()
            if not all(1 <= v <= 12 for v in self._task_layout):
                raise ValueError()
        except (tk.TclError, ValueError):
            messagebox.showerror(
                "Combine", "Rows and columns must be integers from 1 to 12."
            )
            return
        self._sync_layout_preview()
        overwrite_pass = False
        self._running = True
        self._set_controls_enabled(False)
        self._progress.grid()
        self._progress.start(12)
        self._status_label.config(text="Report running...")
        self._output_listbox.delete(0, tk.END)
        worker = threading.Thread(
            target=self._report_worker,
            args=(
                self._external_paths[:],
                date_name,
                analysis_id,
                bool(overwrite_pass),
                source_kind,
            ),
            daemon=True,
        )
        worker.start()

    def _clear_report_log(self) -> None:
        if self._running:
            return
        self._output_listbox.delete(0, tk.END)
        self._status_label.config(text="Ready")

    def _report_worker(
        self,
        external_paths: list[Path],
        date_name: str,
        analysis_id: str,
        overwrite_pass: bool,
        source_kind: str,
    ) -> None:
        try:
            if external_paths:
                if source_kind == "BOTH" and not (
                    len(external_paths) == 1
                    and self._analysis_id_re.fullmatch(external_paths[0].name)
                ):
                    raise ValueError(
                        "For BOTH, drop one Analysis ID folder. Select PASS for individual images or FRONT/BACK folders."
                    )
                output_paths = []
                for kind in (
                    ("PASS", "MASK") if source_kind == "BOTH" else (source_kind,)
                ):
                    output_paths.extend(
                        self._run_external_report(external_paths, analysis_id, kind)
                    )
            else:
                output_paths = combine_analysis(
                    self._combiner,
                    self._combiner.root_path / date_name / analysis_id,
                    selected_cells=self._task_cells,
                    layout=self._task_layout,
                    source_kind=source_kind,
                    progress=self._log_report_progress,
                )
            self._queue.put(("done", output_paths))
        except Exception as exc:
            logger.exception("Report image generation failed")
            self._queue.put(("error", str(exc)))

    def _run_external_report(
        self,
        external_paths: list[Path],
        analysis_id: str,
        source_kind: str,
    ) -> list[Path]:
        with tempfile.TemporaryDirectory(prefix="photojig_report_") as tmp:
            tmp_root = Path(tmp)
            input_root = tmp_root / "input"
            output_root = tmp_root / "output"
            input_root.mkdir(parents=True, exist_ok=True)
            copied_roots: list[tuple[Path, Path]] = []
            for source in external_paths:
                destination = input_root / source.name
                self._copy_external_to_local_subprocess(source, destination)
                copied_roots.append((source, destination))

            copied_image_count = self._local_image_count(input_root)
            if copied_image_count <= 0:
                copied_items = self._local_tree_preview(input_root)
                raise RuntimeError(
                    "No image files were copied from the dropped paths.\n\n"
                    f"Dropped paths:\n{self._path_preview(external_paths)}\n\n"
                    f"Local temp contents:\n{copied_items}"
                )

            folder_sources = [(src, dst) for src, dst in copied_roots if dst.is_dir()]
            if len(folder_sources) == 1 and self._looks_like_analysis_folder(
                folder_sources[0][1]
            ):
                original_root, copied_root = folder_sources[0]
                old_passes = {
                    p
                    for p in copied_root.rglob("*.png")
                    if p.parent.name.lower().endswith("_pass")
                }
                generated = combine_analysis(
                    self._combiner,
                    copied_root,
                    selected_cells=self._task_cells,
                    layout=self._task_layout,
                    source_kind=source_kind,
                    progress=self._log_report_progress,
                )
                final_paths = []
                for path in copied_root.rglob("*.png"):
                    if (
                        path.parent.name.lower().endswith("_pass")
                        and path not in old_passes
                    ):
                        self._copy_local_file_to_external_subprocess(
                            path, original_root / path.relative_to(copied_root)
                        )
                for folder in sorted({p.parent for p in generated}):
                    destination = original_root / folder.relative_to(copied_root)
                    final_paths.extend(self._copy_side_result_back(folder, destination))
                return final_paths or generated

            if source_kind == REPORT_SOURCE_MASK:
                raise ValueError(
                    "For a _MASK report, drop the AnalysisID folder so its Date / "
                    "AnalysisID / Cell # structure can be preserved."
                )
            face_destinations = self._external_face_output_destinations(copied_roots)
            image_paths = sorted(
                [
                    path
                    for path in input_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda path: path.name,
            )
            run_result = self._combiner.create_for_paths(
                image_paths,
                output_root,
                analysis_id=analysis_id,
                layout=self._task_layout,
                progress_callback=self._log_report_progress,
            )
            final_paths: list[Path] = []
            for side_result in run_result.side_results:
                destination_base = face_destinations.get(side_result.face)
                if destination_base is None:
                    continue
                final_paths.extend(
                    self._copy_side_result_back(
                        side_result.pass_folder,
                        destination_base / side_result.pass_folder.name,
                    )
                )
                final_paths.extend(
                    self._copy_side_result_back(
                        side_result.combined_folder,
                        destination_base / side_result.combined_folder.name,
                    )
                )
            return final_paths or run_result.combined_paths

    def _log_report_progress(self, message: str) -> None:
        self._queue.put(("log", message))

    def _looks_like_analysis_folder(self, folder: Path) -> bool:
        for cell_name, cell_path in self._combiner.list_cells(folder):
            _ = cell_name
            names = {path.name.upper() for path in cell_path.iterdir() if path.is_dir()}
            if names.intersection(
                {"FRONT", "BACK", "FRONT_PASS", "BACK_PASS", "FRONT_MASK", "BACK_MASK"}
            ):
                return True
        return False

    def _copy_side_result_back(
        self, source_folder: Path, destination_folder: Path
    ) -> list[Path]:
        self._copy_local_folder_pngs_to_external_subprocess(
            source_folder, destination_folder
        )
        return [
            destination_folder / path.name
            for path in sorted(source_folder.glob("*.png"), key=lambda path: path.name)
        ]

    def _external_face_output_destinations(
        self,
        copied_roots: list[tuple[Path, Path]],
    ) -> dict[str, Path]:
        destinations: dict[str, Path] = {}
        for original_path, copied_path in copied_roots:
            for face in ("FRONT", "BACK"):
                if face in destinations:
                    continue
                if self._path_matches_face(copied_path, face):
                    destinations[face] = self._external_sibling_base(original_path)
                    continue
                matching_files = [
                    path
                    for path in copied_path.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                    and self._path_matches_face(path, face)
                ]
                if matching_files:
                    destinations[face] = self._external_sibling_base(original_path)
        if not destinations and copied_roots:
            destinations["FRONT"] = self._external_sibling_base(copied_roots[0][0])
            destinations["BACK"] = self._external_sibling_base(copied_roots[0][0])
        return destinations

    def _path_matches_face(self, path: Path, face: str) -> bool:
        text = path.name.upper()
        return face in text

    def _external_sibling_base(self, path: Path) -> Path:
        return path.parent if path.suffix.lower() in IMAGE_EXTENSIONS else path.parent

    def _copy_external_to_local_subprocess(
        self, source: Path, destination: Path
    ) -> None:
        # Do not probe source with pathlib here. Some VDI/team-room paths deny
        # Python/PowerShell stat calls but still allow cmd copy operations.
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
                copy_result = self._run_cmd_args(
                    "copy",
                    "/Y",
                    source_file_text,
                    str(destination_file),
                )
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

    def _copy_local_folder_pngs_to_external_subprocess(
        self, source: Path, destination: Path
    ) -> None:
        source = Path(source)
        destination = Path(destination)
        for png_path in sorted(source.glob("*.png"), key=lambda path: path.name):
            self._copy_local_file_to_external_subprocess(
                png_path, destination / png_path.name
            )

    def _copy_local_file_to_external_subprocess(
        self, source: Path, destination: Path
    ) -> None:
        source = Path(source)
        destination = Path(destination)
        self._run_cmd_args("mkdir", str(destination.parent))
        completed = self._run_cmd_args("copy", "/Y", str(source), str(destination))
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to copy report output back to the dropped folder.\n\n"
                f"Destination: {destination}\n"
                f"copy: {self._short_process_output(completed)}"
            )

    def _run_cmd_args(self, *args: str) -> subprocess.CompletedProcess:
        creationflags = 0
        startupinfo = None
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        return subprocess.run(
            ["cmd.exe", "/d", "/c", *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            startupinfo=startupinfo,
        )

    def _relative_text(self, child: str, parent: str) -> str:
        normalized_child = child.replace("/", "\\")
        normalized_parent = parent.replace("/", "\\").rstrip("\\")
        if normalized_child.lower().startswith(normalized_parent.lower() + "\\"):
            return normalized_child[len(normalized_parent) + 1 :]
        return Path(normalized_child).name

    def _local_image_count(self, folder: Path) -> int:
        return sum(
            1
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def _local_tree_preview(self, folder: Path, limit: int = 25) -> str:
        if not folder.exists():
            return "<local temp folder missing>"
        items: list[str] = []
        for path in sorted(folder.rglob("*"), key=lambda item: str(item).lower()):
            relative = path.relative_to(folder)
            suffix = "/" if path.is_dir() else ""
            items.append(f"{relative}{suffix}")
            if len(items) >= limit:
                break
        if not items:
            return "<empty>"
        if len(items) >= limit:
            items.append("...")
        return "\n".join(items)

    def _path_preview(self, paths: list[Path], limit: int = 12) -> str:
        lines = [str(path) for path in paths[:limit]]
        if len(paths) > limit:
            lines.append(f"... and {len(paths) - limit} more")
        return "\n".join(lines) if lines else "<none>"

    def _short_process_output(self, completed: subprocess.CompletedProcess) -> str:
        output = (completed.stderr or completed.stdout or "").strip()
        output = " ".join(output.split())
        if len(output) > 500:
            output = output[:500] + "..."
        return output or f"exit code {completed.returncode}"

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        combo_state = "readonly" if enabled else tk.DISABLED
        for widget in (
            self._mode_combo,
            self._source_combo,
            self._refresh_btn,
            self._clear_path_btn,
            self._run_btn,
            self._clear_report_log_btn,
            self._columns_spin,
            self._rows_spin,
        ):
            widget.config(state=state)
        self._mode_combo.config(state=combo_state)
        self._source_combo.config(state=combo_state)
        self._date_combo.config(
            state=tk.DISABLED
            if self._mode_var.get() == "By Analysis ID" or not enabled
            else "readonly"
        )
        self._analysis_combo.config(state=combo_state)
        self._cell_listbox.config(state=state)

    def _poll_queue(self) -> None:
        try:
            while True:
                msg_type, payload = self._queue.get_nowait()
                if msg_type == "done":
                    self._finish_run()
                    self._status_label.config(
                        text=f"Done: {len(payload)} combined image(s)"
                    )
                    for line in self._summarize_output_locations(payload):
                        self._output_listbox.insert(tk.END, line)
                elif msg_type == "log":
                    self._status_label.config(text=payload)
                    self._output_listbox.insert(tk.END, payload)
                    self._output_listbox.see(tk.END)
                elif msg_type == "error":
                    self._finish_run()
                    self._status_label.config(text=f"Error: {payload}")
                    messagebox.showerror("Report Images", payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish_run(self) -> None:
        self._running = False
        self._progress.stop()
        self._progress.grid_remove()
        self._set_controls_enabled(True)

    def _summarize_output_locations(self, paths: list[Path]) -> list[str]:
        folders = sorted({str(Path(path).parent) for path in paths}, key=str.lower)
        if not folders:
            return ["저장위치: -"]
        if len(folders) == 1:
            return [f"저장위치: {folders[0]}"]
        lines = [f"저장위치: {len(folders)}개 폴더"]
        lines.extend(f" - {folder}" for folder in folders[:4])
        if len(folders) > 4:
            lines.append(f" - ... and {len(folders) - 4} more")
        return lines
