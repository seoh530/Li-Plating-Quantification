"""PDF report browser. Each analysis ID contains all of its cells."""

from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import queue
import shutil
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import yaml

from modules.report.report_data import analysis_paths, load_dataset

logger = logging.getLogger(__name__)


class ReportTab(ttk.Frame):
    def __init__(self, parent, config_path):
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._queue = queue.Queue()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pdf-report"
        )
        self._temp = tempfile.TemporaryDirectory(prefix="li_salt_pdf_")
        self._generation = 0
        self._datasets = {}
        self._pdf_path = None
        self._document = None
        self._page = 0
        self._photo = None
        self._render_job = None
        self._future = None
        self._closed = False
        self._build_ui()
        self._refresh()
        self._poll()
        self.bind("<Destroy>", self._dispose, add="+")
        self._tab_binding = self.master.bind(
            "<<NotebookTabChanged>>",
            lambda _event: self.after_idle(self._refresh_if_visible),
            add="+",
        )

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        ttk.Label(self, text="Report", font=("Segoe UI", 19, "bold")).grid(
            row=0, column=0, sticky=tk.W, padx=24, pady=(18, 14)
        )
        toolbar = ttk.Frame(self, style="Card.TFrame", padding=(12, 8))
        toolbar.grid(row=1, column=0, sticky=tk.EW, padx=24, pady=(0, 12))
        toolbar.columnconfigure(1, weight=1)
        self._refresh_btn = ttk.Button(toolbar, text="Refresh", command=self._refresh)
        self._refresh_btn.grid(row=0, column=0)
        self._status = ttk.Label(
            toolbar, text="Finding completed analyses…", style="Panel.TLabel"
        )
        self._status.grid(row=0, column=1, sticky=tk.W, padx=14)
        self._export_btn = ttk.Button(
            toolbar,
            text="Export PDF",
            style="Accent.TButton",
            command=self._export,
            state=tk.DISABLED,
        )
        self._export_btn.grid(row=0, column=2)
        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky=tk.NSEW, padx=24, pady=(0, 18))
        left = ttk.Frame(body, style="Card.TFrame", padding=12)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        body.add(left, weight=1)
        ttk.Label(
            left,
            text="Completed analyses",
            style="Panel.TLabel",
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        self._tree = ttk.Treeview(
            left,
            columns=("id", "cells"),
            show="headings",
            selectmode="browse",
            height=14,
        )
        self._tree.heading("id", text="Analysis ID")
        self._tree.heading("cells", text="Cells")
        self._tree.column("id", width=135, minwidth=100)
        self._tree.column("cells", width=95, minwidth=70)
        self._tree.grid(row=1, column=0, sticky=tk.NSEW)
        self._tree.bind("<<TreeviewSelect>>", self._select)
        scrollbar = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self._tree.yview)
        scrollbar.grid(row=1, column=1, sticky=tk.NS)
        self._tree.configure(yscrollcommand=scrollbar.set)
        self._details = ttk.Label(
            left,
            text="Select one analysis to include all cells.",
            style="Panel.TLabel",
            wraplength=245,
            justify=tk.LEFT,
        )
        self._details.grid(row=2, column=0, sticky=tk.EW, pady=(12, 0))
        right = ttk.Frame(body, padding=(16, 0, 0, 0))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        body.add(right, weight=3)
        nav = ttk.Frame(right, style="Card.TFrame", padding=(12, 8))
        nav.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))
        nav.columnconfigure(0, weight=1)
        self._page_label = ttk.Label(nav, text="PDF preview", style="Panel.TLabel")
        self._page_label.grid(row=0, column=0, sticky=tk.W)
        self._prev = ttk.Button(
            nav,
            text="Previous page",
            command=lambda: self._change_page(-1),
            state=tk.DISABLED,
        )
        self._prev.grid(row=0, column=1, padx=(8, 0))
        self._next = ttk.Button(
            nav,
            text="Next page",
            command=lambda: self._change_page(1),
            state=tk.DISABLED,
        )
        self._next.grid(row=0, column=2, padx=(8, 0))
        self._canvas = tk.Canvas(right, bg="#18212e", highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky=tk.NSEW)
        self._canvas.bind("<Configure>", self._schedule_render)

    def _refresh_if_visible(self):
        if self.winfo_ismapped() and not getattr(self, "_scanning", False):
            self._refresh()

    def _refresh(self):
        if getattr(self, "_scanning", False):
            return
        self._scanning = True
        self._refresh_btn.configure(state=tk.DISABLED)
        cfg = yaml.safe_load(self._config_path.read_text(encoding="utf-8-sig"))
        root = Path(cfg["app"]["root_path"])
        pattern = cfg["folder_qc"]["analysis_id_pattern"]
        self._status.configure(text="Checking report inputs…")

        def scan():
            ready = []
            skipped = 0
            try:
                for path in analysis_paths(root, pattern):
                    try:
                        ready.append(load_dataset(path, self._config_path))
                    except (ValueError, OSError, KeyError) as exc:
                        skipped += 1
                        logger.debug("Report incomplete: %s: %s", path, exc)
                self._queue.put(("list", (ready, skipped)))
            except Exception as exc:
                self._queue.put(("scan_error", str(exc)))

        self._executor.submit(scan)

    def reload_settings(self):
        self._generation += 1
        self._clear_document()
        self._refresh()

    def _clear_document(self):
        if self._document is not None:
            self._document.close()
        if self._pdf_path is not None:
            self._pdf_path.unlink(missing_ok=True)
        self._document = None
        self._pdf_path = None
        self._photo = None
        self._canvas.delete("all")
        self._export_btn.configure(state=tk.DISABLED)
        self._prev.configure(state=tk.DISABLED)
        self._next.configure(state=tk.DISABLED)
        self._page_label.configure(text="PDF preview")

    def _select(self, _event=None):
        selection = self._tree.selection()
        if not selection:
            return
        data = self._datasets.get(selection[0])
        if data is None:
            return
        self._generation += 1
        token = self._generation
        self._clear_document()
        if self._future is not None:
            self._future.cancel()
        self._details.configure(
            text=f"{data.analysis_id}\n"
            + ", ".join(f"Cell #{c}" for c in data.cells)
            + f"\n{len(data.frame)} measured layers"
        )
        self._status.configure(text="Building PDF preview…")
        destination = Path(self._temp.name) / f"preview_{token}.pdf"

        def build():
            try:
                from modules.report.pdf_report import create_pdf

                create_pdf(data.path, self._config_path, destination)
                self._queue.put(("pdf", (token, destination)))
            except Exception as exc:
                logger.exception("PDF report failed")
                self._queue.put(("pdf_error", (token, str(exc))))

        self._future = self._executor.submit(build)

    def _poll(self):
        if self._closed:
            return
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "list":
                    previous = self._tree.selection()
                    old = self._datasets.get(previous[0]) if previous else None
                    self._datasets = {}
                    self._tree.delete(*self._tree.get_children())
                    for data in payload[0]:
                        row = self._tree.insert(
                            "",
                            tk.END,
                            values=(
                                data.analysis_id,
                                ", ".join(f"#{c}" for c in data.cells),
                            ),
                        )
                        self._datasets[row] = data
                        if old is not None and old.path == data.path:
                            self._tree.selection_set(row)
                    self._scanning = False
                    self._refresh_btn.configure(state=tk.NORMAL)
                    self._status.configure(
                        text=f"{len(payload[0])} ready for PDF · {payload[1]} incomplete"
                    )
                    if old is not None and not any(
                        d.path == old.path for d in payload[0]
                    ):
                        self._generation += 1
                        self._clear_document()
                elif kind == "scan_error":
                    self._scanning = False
                    self._refresh_btn.configure(state=tk.NORMAL)
                    self._status.configure(text=payload)
                elif kind == "pdf":
                    token, path = payload
                    if token != self._generation:
                        path.unlink(missing_ok=True)
                        continue
                    try:
                        import fitz

                        self._document = fitz.open(path)
                        self._pdf_path = path
                        self._page = 0
                        self._render()
                        self._export_btn.configure(state=tk.NORMAL)
                        self._status.configure(text="Report ready")
                    except Exception as exc:
                        self._status.configure(text=str(exc))
                elif kind == "pdf_error" and payload[0] == self._generation:
                    self._status.configure(text="Report could not be built")
                    messagebox.showerror("PDF report", payload[1], parent=self)
        except queue.Empty:
            pass
        self.after(100, self._poll)

    def _change_page(self, delta):
        if self._document is not None:
            self._page = max(0, min(len(self._document) - 1, self._page + delta))
            self._render()

    def _schedule_render(self, _event=None):
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(60, self._render)

    def _render(self):
        self._render_job = None
        self._canvas.delete("all")
        w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
        if self._document is None:
            self._canvas.create_text(
                w / 2,
                h / 2,
                text="Select a completed analysis to preview its PDF",
                fill="#94a3b8",
                width=max(200, w - 40),
            )
            return
        import fitz

        page = self._document[self._page]
        scale = max(0.05, min((w - 30) / page.rect.width, (h - 30) / page.rect.height))
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        self._photo = tk.PhotoImage(master=self._canvas, data=pix.tobytes("png"))
        self._canvas.create_image(w / 2, h / 2, image=self._photo, anchor=tk.CENTER)
        self._page_label.configure(
            text=f"Page {self._page + 1} / {len(self._document)}"
        )
        self._prev.configure(state=tk.NORMAL if self._page > 0 else tk.DISABLED)
        self._next.configure(
            state=tk.NORMAL if self._page < len(self._document) - 1 else tk.DISABLED
        )

    def _export(self):
        if self._pdf_path is None:
            return
        selected = self._tree.selection()
        data = self._datasets.get(selected[0]) if selected else None
        if data is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save analysis PDF",
            defaultextension=".pdf",
            initialfile=f"{data.analysis_id}_report_{datetime.now():%Y%m%d}.pdf",
            filetypes=[("PDF report", "*.pdf")],
        )
        if not path:
            return
        temporary = None
        try:
            fd, temporary = tempfile.mkstemp(
                prefix=".report_", suffix=".pdf", dir=Path(path).parent
            )
            os.close(fd)
            shutil.copyfile(self._pdf_path, temporary)
            os.replace(temporary, path)
            self._status.configure(text=f"Saved: {path}")
        except OSError as exc:
            messagebox.showerror("Export PDF", str(exc), parent=self)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _dispose(self, event):
        if event.widget is not self:
            return
        self._closed = True
        self._generation += 1
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._document is not None:
            self._document.close()
        # An in-flight worker may still be writing its own preview; let its
        # TemporaryDirectory finalizer clean up after the worker releases self.
