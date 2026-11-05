import logging
import queue
import threading
import csv
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from modules.ml.single_mask import SingleMaskEngine, SingleMaskResult
from modules.utils.image_io import atomic_imwrite_png, atomic_write_text
from modules.ml.developer_training import retrain, save_training_set


logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_CANVAS_BG = "#18212e"


class SingleMaskTab(ttk.Frame):
    def __init__(self, parent, config_path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._engine = None
        self._result_engine = None
        self._settings_version = 0
        self._result = None
        self._points = {}
        self._display_geometry = None
        self._working = False
        self._queue = queue.Queue()
        self._photo = None
        self._photos = {}
        self._preview_cache = None
        self._zoom = 1.0
        self._pan = (0.0, 0.0)
        self._drag_origin = None
        self._build_ui()
        self._enable_drop_target()
        self._poll_queue()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)
        style = ttk.Style(self)
        style.configure(
            "WorkflowOptions.TFrame",
            background="#ffffff",
            bordercolor="#d7dde5",
            relief=tk.SOLID,
            borderwidth=1,
        )
        for kind in ("TCheckbutton", "TRadiobutton"):
            style.configure(
                "WorkflowOptions." + kind, background="#ffffff", padding=(4, 4)
            )
            style.map(
                "WorkflowOptions." + kind,
                background=[("active", "#e8f0fc"), ("!active", "#ffffff")],
            )
        header = ttk.Frame(self, padding=(24, 18, 24, 14))
        header.grid(row=0, column=0, sticky=tk.EW)
        ttk.Label(
            header, text="Li-Salt Quantification", font=("Segoe UI", 19, "bold")
        ).pack(anchor=tk.W)

        toolbar = ttk.Frame(self, style="WorkflowOptions.TFrame", padding=(12, 8))
        toolbar.grid(row=1, column=0, sticky=tk.EW, padx=24, pady=(0, 12))
        toolbar.columnconfigure(4, weight=1)
        self._developer_var = tk.BooleanVar(value=False)
        self._label_var = tk.StringVar(value="l")
        self._developer_toggle = ttk.Checkbutton(
            toolbar,
            text="Developer mode",
            style="WorkflowOptions.TCheckbutton",
            variable=self._developer_var,
            command=self._toggle_developer,
        )
        self._developer_toggle.grid(row=0, column=0, padx=(0, 20))
        ttk.Label(toolbar, text="Overlay opacity", style="Panel.TLabel").grid(
            row=0, column=1, padx=(0, 8)
        )
        self._opacity_var = tk.DoubleVar(value=28)
        self._opacity_text = tk.StringVar(value="28%")
        self._opacity_slider = ttk.Scale(
            toolbar,
            from_=0,
            to=100,
            variable=self._opacity_var,
            command=self._on_opacity_changed,
            length=140,
        )
        self._opacity_slider.grid(row=0, column=2)
        ttk.Label(
            toolbar, textvariable=self._opacity_text, style="Panel.TLabel", width=5
        ).grid(row=0, column=3, padx=(8, 0))
        self._clear_btn = ttk.Button(toolbar, text="Clear", command=self._clear)
        self._clear_btn.grid(row=0, column=5, padx=(8, 8))
        self._export_btn = ttk.Button(
            toolbar,
            text="Export images + CSV",
            command=self._export_mask,
            style="Accent.TButton",
            state=tk.DISABLED,
        )
        self._export_btn.grid(row=0, column=6)

        upload = ttk.Frame(self)
        upload.grid(row=2, column=0, sticky=tk.EW, padx=24, pady=(0, 12))
        upload.columnconfigure(0, weight=1)
        self._drop_label = tk.Label(
            upload,
            text="Drop an image here",
            bg="#153b63",
            fg="white",
            font=("Segoe UI", 12, "bold"),
            padx=14,
            pady=18,
            highlightthickness=2,
            highlightbackground="#3b82f6",
            highlightcolor="#3b82f6",
        )
        self._drop_label.grid(row=0, column=0, sticky=tk.EW)
        self._browse_btn = ttk.Button(
            upload, text="Choose image", command=self._select_image
        )
        self._browse_btn.grid(row=0, column=1, padx=(8, 0))
        self._status = ttk.Label(
            upload, text="Ready for an image", style="Muted.TLabel"
        )
        self._status.grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        self._progress = ttk.Progressbar(upload, mode="indeterminate", length=120)
        self._progress.grid(row=1, column=1, sticky=tk.E, pady=(8, 0))
        self._progress.grid_remove()

        metrics = ttk.Frame(self)
        metrics.grid(row=3, column=0, sticky=tk.EW, padx=24, pady=(0, 12))
        self._metric_vars = {}
        for index, (name, color) in enumerate(
            [
                ("Li-salt", "#dc2626"),
                ("Nonactive", "#2563eb"),
                ("Background", "#16a34a"),
            ]
        ):
            metrics.columnconfigure(index, weight=1, uniform="metrics")
            card = ttk.Frame(metrics, style="WorkflowOptions.TFrame", padding=(14, 8))
            card.grid(
                row=0, column=index, sticky=tk.EW, padx=(0, 8 if index < 2 else 0)
            )
            tk.Label(card, text=name, bg="white", fg=color, font=("Segoe UI", 10)).pack(
                anchor=tk.W
            )
            value = tk.StringVar(value="—")
            self._metric_vars[name] = value
            ttk.Label(
                card,
                textvariable=value,
                font=("Segoe UI", 16, "bold"),
                style="Panel.TLabel",
            ).pack(anchor=tk.W, pady=(2, 0))

        self._developer_panel = ttk.Frame(
            self, style="WorkflowOptions.TFrame", padding=(12, 8)
        )
        self._developer_panel.grid(row=4, column=0, sticky=tk.EW, padx=24, pady=(0, 12))
        self._developer_panel.columnconfigure(3, weight=1)
        self._developer_controls = []
        for index, (name, code) in enumerate(
            [("Li-salt", "l"), ("Background", "b"), ("Nonactive", "n")]
        ):
            button = ttk.Radiobutton(
                self._developer_panel,
                text=name,
                variable=self._label_var,
                value=code,
                style="WorkflowOptions.TRadiobutton",
            )
            button.grid(row=0, column=index, sticky=tk.W, padx=(0, 10))
            self._developer_controls.append(button)
        self._annotation_status = ttk.Label(
            self._developer_panel, text="0 labeled pixels", style="Panel.TLabel"
        )
        self._annotation_status.grid(row=0, column=3, sticky=tk.E, padx=(0, 12))
        for index, (name, command) in enumerate(
            [
                ("Undo point", self._undo_point),
                ("Add train set", self._add_train_set),
                ("Train", self._train),
            ],
            4,
        ):
            button = ttk.Button(self._developer_panel, text=name, command=command)
            button.grid(row=0, column=index, padx=(8, 0))
            self._developer_controls.append(button)
        self._developer_panel.grid_remove()

        viewer = ttk.Frame(self)
        viewer.grid(row=5, column=0, sticky=tk.NSEW, padx=24, pady=(0, 12))
        viewer.rowconfigure(1, weight=1)
        self._canvases = {}
        for column, (key, title) in enumerate(
            [("overlay", "Source overlay"), ("mask", "Segmentation mask")]
        ):
            viewer.columnconfigure(column, weight=1, uniform="preview")
            gap = (0, 8) if column == 0 else (8, 0)
            ttk.Label(viewer, text=title, font=("Segoe UI", 11, "bold")).grid(
                row=0, column=column, sticky=tk.W, padx=gap, pady=(0, 8)
            )
            canvas = tk.Canvas(viewer, bg=_CANVAS_BG, highlightthickness=0)
            canvas.grid(row=1, column=column, sticky=tk.NSEW, padx=gap)
            self._canvases[key] = canvas
            canvas.bind("<Configure>", lambda _event: self._schedule_render())
            canvas.bind("<MouseWheel>", self._wheel_zoom)
            canvas.bind("<Button-4>", lambda _event: self._change_zoom(1.25))
            canvas.bind("<Button-5>", lambda _event: self._change_zoom(0.8))
            canvas.bind("<ButtonPress-3>", self._pan_start)
            canvas.bind("<B3-Motion>", self._pan_move)
        self._canvas = self._canvases["overlay"]
        self._canvas.bind("<Button-1>", self._label_click)
        self._canvas.bind("<B1-Motion>", self._label_click)
        footer = ttk.Frame(self, padding=(24, 0, 24, 12))
        footer.grid(row=6, column=0, sticky=tk.EW)
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer, text="Scroll to zoom · Right-drag to pan", style="Muted.TLabel"
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(
            footer, text="–", width=3, command=lambda: self._change_zoom(0.8)
        ).grid(row=0, column=1)
        self._zoom_text = tk.StringVar(value="Fit")
        ttk.Label(footer, textvariable=self._zoom_text, width=8, anchor=tk.CENTER).grid(
            row=0, column=2
        )
        ttk.Button(
            footer, text="+", width=3, command=lambda: self._change_zoom(1.25)
        ).grid(row=0, column=3)
        ttk.Button(footer, text="Fit to view", command=self._fit_view).grid(
            row=0, column=4, padx=(8, 0)
        )
        self._render_job = None
        self._draw_placeholder()
        self._set_controls(True)

    def _on_opacity_changed(self, value) -> None:
        opacity = max(0, min(100, float(value)))
        self._opacity_text.set(f"{opacity:.0f}%")
        if hasattr(self, "_render_job"):
            self._schedule_render()

    def _toggle_developer(self) -> None:
        if self._developer_var.get():
            self._developer_panel.grid()
        else:
            self._developer_panel.grid_remove()
        self._set_controls(not self._working)
        self._schedule_render()

    def _schedule_render(self) -> None:
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(30, self._render_mask)

    def _fit_view(self) -> None:
        self._zoom, self._pan = 1.0, (0.0, 0.0)
        self._render_mask()

    def _change_zoom(self, factor) -> None:
        self._zoom = min(8.0, max(1.0, self._zoom * factor))
        if self._zoom == 1.0:
            self._pan = (0.0, 0.0)
        self._render_mask()

    def _wheel_zoom(self, event) -> str:
        self._change_zoom(1.25 if event.delta > 0 else 0.8)
        return "break"

    def _pan_start(self, event) -> None:
        self._drag_origin = (event.x, event.y, self._pan)

    def _pan_move(self, event) -> None:
        if self._drag_origin is None or self._zoom <= 1:
            return
        x, y, prior = self._drag_origin
        self._pan = (prior[0] + event.x - x, prior[1] + event.y - y)
        self._render_mask()

    def _enable_drop_target(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES

            for target in [self._drop_label, *self._canvases.values()]:
                target.drop_target_register(DND_FILES)
                target.dnd_bind("<<Drop>>", self._on_drop)
        except Exception:
            logger.debug(
                "Native drag and drop unavailable; Choose image remains available",
                exc_info=True,
            )

    def _on_drop(self, event) -> str:
        if self._working:
            return "break"
        paths = [
            self._normalize_path(value)
            for value in self.tk.splitlist(getattr(event, "data", ""))
        ]
        self._accept_paths([path for path in paths if path is not None])
        return "break"

    @staticmethod
    def _normalize_path(value):
        value = str(value).strip().strip("{}")
        if not value:
            return None
        if value.lower().startswith("file:"):
            parsed = urlparse(value)
            if parsed.netloc:
                return Path(f"//{parsed.netloc}{unquote(parsed.path)}")
            return Path(unquote(parsed.path.lstrip("/")))
        return Path(value)

    def _select_image(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title="Select Image for Masking",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self._accept_paths([Path(selected)])

    def _accept_paths(self, paths) -> None:
        if len(paths) != 1:
            messagebox.showwarning("Single Mask Mode", "Drop exactly one image file.")
            return
        image_path = Path(paths[0])
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            messagebox.showwarning(
                "Single Mask Mode",
                "Supported formats: JPG, JPEG, PNG, BMP, TIF, TIFF.",
            )
            return
        self._start_processing(image_path)

    def _start_processing(self, image_path) -> None:
        if self._working:
            return
        self._working = True
        self._result = None
        self._points.clear()
        self._preview_cache = None
        self._photos.clear()
        self._set_controls(False)
        self._drop_label.config(text=f"Processing: {image_path.name}", bg="#153b63")
        self._status.config(text="Processing...")
        self._progress.grid()
        self._progress.start(12)
        self._zoom, self._pan = 1.0, (0.0, 0.0)
        self._display_geometry = None
        self._annotation_status.config(text="0 labeled pixels")
        for value in self._metric_vars.values():
            value.set("—")
        self._draw_placeholder("Cropping and analyzing image…")
        threading.Thread(target=self._worker, args=(image_path,), daemon=True).start()

    def _worker(self, image_path) -> None:
        try:
            settings_version = self._settings_version
            engine = self._engine
            if engine is None:
                engine = SingleMaskEngine(self._config_path)
                if settings_version == self._settings_version:
                    self._engine = engine
            result = engine.process_path(
                image_path,
                progress_callback=lambda message: self._queue.put(("stage", message)),
            )
            self._queue.put(("done", (engine, result)))
        except Exception as exc:
            logger.exception("Single mask processing failed")
            self._queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "stage":
                    self._status.config(text=payload)
                elif kind == "saved":
                    self._working = False
                    self._progress.stop()
                    self._progress.grid_remove()
                    self._set_controls(True)
                    self._status.config(text=f"Training set saved: {payload}")
                elif kind == "developer_error":
                    self._working = False
                    self._progress.stop()
                    self._progress.grid_remove()
                    self._set_controls(True)
                    self._status.config(
                        text="Developer operation failed; annotations retained"
                    )
                    messagebox.showerror("Developer mode", payload)
                elif kind == "trained":
                    engine, inference, result = payload
                    engine._inference = inference
                    self._result_engine = engine
                    if self._engine is not None:
                        self._engine = engine
                    self._finish(result)
                    self._status.config(text="Training complete · Preview updated")
                elif kind == "done":
                    self._result_engine, result = payload
                    self._finish(result)
                else:
                    self._fail(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish(self, result: SingleMaskResult) -> None:
        self._working = False
        self._result = result
        self._progress.stop()
        self._progress.grid_remove()
        self._set_controls(True)
        self._drop_label.config(
            text=f"Completed: {result.image_path.name}", bg="#065f46"
        )
        self._status.config(text="Crop complete · Ready to export")
        self._update_metrics()
        self._render_mask()

    def _fail(self, error) -> None:
        self._working = False
        self._result = None
        self._progress.stop()
        self._progress.grid_remove()
        self._set_controls(True)
        self._drop_label.config(text="Image processing failed", bg="#991b1b")
        self._status.config(text="Masking failed")
        self._draw_placeholder()
        messagebox.showerror("Single Mask Mode", str(error))

    def _export_mask(self) -> None:
        if self._result is None or self._working:
            return
        folder = filedialog.askdirectory(
            parent=self.winfo_toplevel(), title="Export images and ratios"
        )
        if not folder:
            return
        try:
            result = self._result
            stem = (
                result.image_path.stem
                + "_"
                + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            )
            target = Path(folder) / (stem + "_mask.png")
            if not atomic_imwrite_png(target, result.mask_image):
                raise RuntimeError("Mask image save failed.")
            values = self._result_engine._quantifier.quantify(
                result.label_mask,
                analysis_id=result.image_path.stem,
                cell=1,
                face="SINGLE",
                image_num=1,
            )
            values.update(
                filename=result.image_path.name,
                mask_file=target.name,
                model_path=str(self._result_engine._inference.model_dir),
                li_salt_percent=values["li_salt_ratio"] * 100,
                background_percent=values["background_ratio"] * 100,
                nonactive_percent=values["nonactive_ratio"] * 100,
                edge_percent=float(np.count_nonzero(result.label_mask == 3))
                / values["total_pixel"]
                * 100,
            )
            import io

            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(values))
            writer.writeheader()
            writer.writerow(values)
            atomic_write_text(Path(folder) / (stem + "_results.csv"), stream.getvalue())
            if result.source_image is not None:
                if not atomic_imwrite_png(
                    Path(folder) / (stem + "_source.png"), result.source_image
                ):
                    raise RuntimeError("Source image save failed.")
            self._status.config(text=f"Exported mask, source and CSV: {folder}")
        except Exception as exc:
            messagebox.showerror("Export Images", str(exc))

    def _label_click(self, event) -> None:
        if (
            self._working
            or not self._developer_var.get()
            or self._result is None
            or self._display_geometry is None
        ):
            return
        geometry = self._display_geometry
        left, top, sx = geometry[:3]
        sy = geometry[3] if len(geometry) == 4 else sx
        x, y = int((event.x - left) / sx), int((event.y - top) / sy)
        image = self._result.source_image
        if (
            image is None
            or event.x < left
            or event.y < top
            or not (0 <= y < image.shape[0] and 0 <= x < image.shape[1])
        ):
            return
        if image.shape[2] == 4 and image[y, x, 3] == 0:
            return
        self._points[(x, y)] = self._label_var.get()
        self._annotation_status.config(text=f"{len(self._points)} labeled pixels")
        self._render_mask()

    def _undo_point(self) -> None:
        if self._points and not self._working:
            self._points.pop(next(reversed(self._points)))
            self._annotation_status.config(text=f"{len(self._points)} labeled pixels")
            self._render_mask()

    def _start_developer_task(self, train_model: bool) -> None:
        if self._working or self._result is None or not self._developer_var.get():
            return
        if not self._points:
            messagebox.showwarning(
                "Developer mode", "Add labeled pixels on the source overlay first."
            )
            return
        result, engine = self._result, self._result_engine
        points = [(x, y, label) for (x, y), label in self._points.items()]
        self._working = True
        self._set_controls(False)
        self._progress.grid()
        self._progress.start(12)
        self._status.config(
            text="Training new model..."
            if train_model
            else "Saving enhanced training set..."
        )

        def work():
            try:
                if train_model:
                    inference = retrain(
                        engine._inference,
                        result.source_image,
                        points,
                        self._config_path,
                    )
                    labels = inference.predict_mask(
                        inference.extract_features(result.source_image)
                    )
                    updated = SingleMaskResult(
                        result.image_path,
                        labels,
                        engine._quantifier.build_mask_image(labels),
                        result.source_image,
                    )
                    self._queue.put(("trained", (engine, inference, updated)))
                else:
                    image_path, csv_path = save_training_set(
                        result.source_image,
                        points,
                        self._config_path,
                        feature_config=engine._inference._metadata.get(
                            "feature_extraction"
                        ),
                    )
                    # Preserve the uncropped capture separately; CSV coordinates use image_path.
                    import shutil

                    if result.image_path.is_file():
                        capture_folder = csv_path.parent / "captures"
                        capture_folder.mkdir(exist_ok=True)
                        shutil.copy2(
                            result.image_path,
                            capture_folder
                            / (image_path.stem + result.image_path.suffix),
                        )
                    self._queue.put(("saved", str(csv_path)))
            except Exception as exc:
                logger.exception("Developer operation failed")
                self._queue.put(("developer_error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _train(self) -> None:
        self._start_developer_task(True)

    def _add_train_set(self) -> None:
        self._start_developer_task(False)

    def _clear(self) -> None:
        if self._working:
            return
        self._result = None
        self._photo = None
        self._points.clear()
        self._annotation_status.config(text="0 labeled pixels")
        self._drop_label.config(text="Drop an image here", bg="#153b63")
        self._status.config(text="Ready for an image")
        self._zoom_text.set("Fit")
        self._display_geometry = None
        self._photos.clear()
        self._preview_cache = None
        self._zoom, self._pan = 1.0, (0.0, 0.0)
        for value in self._metric_vars.values():
            value.set("—")
        self._set_controls(True)
        self._draw_placeholder()

    def _set_controls(self, enabled) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self._browse_btn.config(state=state)
        for button in self._developer_controls:
            button.config(state=state if self._developer_var.get() else tk.DISABLED)
        self._clear_btn.config(state=state)
        export_state = (
            tk.NORMAL if enabled and self._result is not None else tk.DISABLED
        )
        self._export_btn.config(state=export_state)

    def _update_metrics(self) -> None:
        if self._result is None:
            return
        labels = self._result.label_mask
        total = int(np.count_nonzero(labels >= 0))
        for name, code in [("Li-salt", 1), ("Nonactive", 2), ("Background", 0)]:
            self._metric_vars[name].set(
                f"{np.count_nonzero(labels == code) / total * 100:.2f}%"
                if total
                else "—"
            )

    def _draw_placeholder(
        self, message="Drop an image or choose a file to begin"
    ) -> None:
        for key, canvas in self._canvases.items():
            canvas.delete("all")
            canvas.create_text(
                max(canvas.winfo_width(), 240) / 2,
                max(canvas.winfo_height(), 160) / 2,
                text=message,
                fill="#94a3b8",
                font=("Segoe UI", 11),
                width=max(canvas.winfo_width() - 48, 180),
                justify=tk.CENTER,
            )

    @staticmethod
    def _preview_images(result, opacity=0.28):
        mask = result.mask_image[:, :, :3].copy()
        source = result.source_image
        if source is None:
            return mask, mask.copy()
        alpha = (
            source[:, :, 3] > 0
            if source.shape[2] == 4
            else np.ones(source.shape[:2], bool)
        )
        opacity = float(np.clip(opacity, 0, 1))
        overlay = cv2.addWeighted(source[:, :, :3], 1 - opacity, mask, opacity, 0)
        overlay[~alpha] = (236, 239, 243)
        mask[~alpha] = (236, 239, 243)
        return overlay, mask

    def _render_mask(self) -> None:
        self._render_job = None
        self._set_controls(not self._working)
        if self._result is None:
            if not self._working:
                self._draw_placeholder()
            return
        self._zoom_text.set("Fit" if self._zoom == 1 else f"{self._zoom:.1f}×")
        opacity = self._opacity_var.get() / 100
        if (
            self._preview_cache is None
            or self._preview_cache[0] is not self._result
            or self._preview_cache[2] != opacity
        ):
            self._preview_cache = (
                self._result,
                self._preview_images(self._result, opacity),
                opacity,
            )
        overlay, mask = self._preview_cache[1]
        for key, array in [("overlay", overlay), ("mask", mask)]:
            canvas = self._canvases[key]
            cw, ch = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
            if cw < 10 or ch < 10:
                continue
            height, width = array.shape[:2]
            fit = min((cw - 32) / width, (ch - 32) / height)
            scale = max(fit, 0.001) * self._zoom
            dw, dh = max(1, round(width * scale)), max(1, round(height * scale))
            px, py = self._pan
            px = min(max(px, -max(dw - cw, 0) / 2), max(dw - cw, 0) / 2)
            py = min(max(py, -max(dh - ch, 0) / 2), max(dh - ch, 0) / 2)
            left, top = (cw - dw) / 2 + px, (ch - dh) / 2 + py
            sx, sy = dw / width, dh / height
            # Resize only the visible source region; zooming a large capture
            # must not allocate an entire oversized display bitmap.
            x0, y0 = max(0, int(-left / sx)), max(0, int(-top / sy))
            x1, y1 = (
                min(width, int((cw - left) / sx) + 1),
                min(height, int((ch - top) / sy) + 1),
            )
            visible = array[y0:y1, x0:x1]
            if visible.size == 0:
                continue
            shown = cv2.resize(
                visible,
                (max(1, round((x1 - x0) * sx)), max(1, round((y1 - y0) * sy))),
                interpolation=cv2.INTER_NEAREST
                if key == "mask"
                else cv2.INTER_AREA
                if scale < 1
                else cv2.INTER_LINEAR,
            )
            ok, encoded = cv2.imencode(".png", shown)
            if not ok:
                raise RuntimeError("Unable to render image preview.")
            photo = tk.PhotoImage(master=canvas, data=encoded.tobytes())
            self._photos[key] = photo
            canvas.delete("all")
            canvas.create_image(
                left + x0 * sx, top + y0 * sy, image=photo, anchor=tk.NW
            )
            if key == "overlay":
                self._display_geometry = (left, top, sx, sy)
                if self._developer_var.get():
                    colors = {"l": "#ff4545", "b": "#35dd71", "n": "#5aa2ff"}
                    for (x, y), label in self._points.items():
                        px, py = left + (x + 0.5) * sx, top + (y + 0.5) * sy
                        if 0 <= px <= cw and 0 <= py <= ch:
                            canvas.create_oval(
                                px - 3,
                                py - 3,
                                px + 3,
                                py + 3,
                                fill=colors[label],
                                outline="white",
                                width=1,
                            )

    def reload_settings(self) -> None:
        self._settings_version += 1
        self._engine = None
        if not self._working:
            self._status.config(text="Settings reloaded")
