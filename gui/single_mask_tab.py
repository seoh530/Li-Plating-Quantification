import logging
import queue
import threading
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from modules.ml.single_mask import SingleMaskEngine, SingleMaskResult
from modules.utils.image_io import atomic_imwrite_png


logger = logging.getLogger(__name__)
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
_CANVAS_BG = '#2b3036'


class SingleMaskTab(ttk.Frame):
    def __init__(self, parent, config_path) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._engine = None
        self._result = None
        self._working = False
        self._queue = queue.Queue()
        self._photo = None
        self._build_ui()
        self._enable_drop_target()
        self._poll_queue()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        header = ttk.Frame(self, padding=(18, 16, 18, 8))
        header.grid(row=0, column=0, sticky=tk.EW)
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text='Single Mask Mode',
            font=('Segoe UI', 16, 'bold'),
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            header,
            text='Background crop + masking for one image. Batch, CSV, DB, and reports are skipped.',
            style='Muted.TLabel',
        ).grid(row=1, column=0, sticky=tk.W, pady=(3, 0))

        self._drop_label = tk.Label(
            self,
            text='Drag & Drop One Image Here',
            bg='#eef6ff',
            fg='#1f2937',
            relief=tk.GROOVE,
            borderwidth=2,
            padx=20,
            pady=28,
            font=('Segoe UI', 12, 'bold'),
        )
        self._drop_label.grid(row=1, column=0, sticky=tk.EW, padx=18)

        actions = ttk.Frame(self, padding=(18, 8))
        actions.grid(row=2, column=0, sticky=tk.EW)
        actions.columnconfigure(3, weight=1)
        self._browse_btn = ttk.Button(
            actions, text='Select Image', command=self._select_image
        )
        self._browse_btn.grid(row=0, column=0)
        self._export_btn = ttk.Button(
            actions, text='Export Mask', command=self._export_mask,
            state=tk.DISABLED,
        )
        self._export_btn.grid(row=0, column=1, padx=(8, 0))
        self._clear_btn = ttk.Button(
            actions, text='Clear', command=self._clear
        )
        self._clear_btn.grid(row=0, column=2, padx=(8, 0))
        self._progress = ttk.Progressbar(actions, mode='indeterminate', length=180)
        self._progress.grid(row=0, column=4, padx=(12, 0))
        self._status = ttk.Label(actions, text='Ready', style='Muted.TLabel')
        self._status.grid(row=0, column=5, padx=(10, 0))

        self._canvas = tk.Canvas(
            self, bg=_CANVAS_BG, highlightthickness=1, highlightbackground='#475569'
        )
        self._canvas.grid(row=3, column=0, sticky=tk.NSEW, padx=18, pady=(0, 18))
        self._canvas.bind('<Configure>', lambda _event: self._render_mask())
        self._draw_placeholder()

    def _enable_drop_target(self) -> None:
        try:
            from tkinterdnd2 import DND_FILES
            self._drop_label.drop_target_register(DND_FILES)
            self._drop_label.dnd_bind('<<Drop>>', self._on_drop)
        except Exception:
            logger.debug('Single mask drop target unavailable', exc_info=True)
            self._drop_label.config(bg='#fff7ed')

    def _on_drop(self, event) -> str:
        if self._working:
            return 'break'
        paths = [
            self._normalize_path(value)
            for value in self.tk.splitlist(getattr(event, 'data', ''))
        ]
        self._accept_paths([path for path in paths if path is not None])
        return 'break'

    @staticmethod
    def _normalize_path(value):
        value = str(value).strip().strip('{}')
        if not value:
            return None
        if value.lower().startswith('file:'):
            parsed = urlparse(value)
            if parsed.netloc:
                return Path(f'//{parsed.netloc}{unquote(parsed.path)}')
            return Path(unquote(parsed.path.lstrip('/')))
        return Path(value)

    def _select_image(self) -> None:
        selected = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title='Select Image for Masking',
            filetypes=(
                ('Image files', '*.jpg *.jpeg *.png *.bmp *.tif *.tiff'),
                ('All files', '*.*'),
            ),
        )
        if selected:
            self._accept_paths([Path(selected)])

    def _accept_paths(self, paths) -> None:
        if len(paths) != 1:
            messagebox.showwarning(
                'Single Mask Mode', 'Drop exactly one image file.'
            )
            return
        image_path = Path(paths[0])
        if image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            messagebox.showwarning(
                'Single Mask Mode',
                'Supported formats: JPG, JPEG, PNG, BMP, TIF, TIFF.',
            )
            return
        self._start_processing(image_path)

    def _start_processing(self, image_path) -> None:
        if self._working:
            return
        self._working = True
        self._result = None
        self._set_controls(False)
        self._drop_label.config(
            text=f'Masking: {image_path.name}', bg='#fef3c7'
        )
        self._status.config(text='Processing...')
        self._progress.start(12)
        self._canvas.delete('all')
        self._canvas.create_text(
            self._canvas.winfo_width() / 2,
            self._canvas.winfo_height() / 2,
            text='Generating final mask...',
            fill='#e5e7eb',
            font=('Segoe UI', 12),
        )
        threading.Thread(
            target=self._worker, args=(image_path,), daemon=True
        ).start()

    def _worker(self, image_path) -> None:
        try:
            if self._engine is None:
                self._engine = SingleMaskEngine(self._config_path)
            result = self._engine.process_path(image_path)
            self._queue.put(('done', result))
        except Exception as exc:
            logger.exception('Single mask processing failed')
            self._queue.put(('error', str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == 'done':
                    self._finish(payload)
                else:
                    self._fail(payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish(self, result: SingleMaskResult) -> None:
        self._working = False
        self._result = result
        self._progress.stop()
        self._set_controls(True)
        self._drop_label.config(
            text=f'Completed: {result.image_path.name}', bg='#dcfce7'
        )
        self._status.config(text='Mask ready to export')
        self._render_mask()

    def _fail(self, error) -> None:
        self._working = False
        self._result = None
        self._progress.stop()
        self._set_controls(True)
        self._drop_label.config(text='Drag & Drop One Image Here', bg='#fee2e2')
        self._status.config(text='Masking failed')
        self._draw_placeholder()
        messagebox.showerror('Single Mask Mode', str(error))

    def _export_mask(self) -> None:
        if self._result is None:
            return
        source = self._result.image_path
        selected = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title='Export Mask Image',
            initialdir=str(source.parent),
            initialfile=f'{source.stem}_mask.png',
            defaultextension='.png',
            filetypes=(('PNG image', '*.png'),),
        )
        if not selected:
            return
        try:
            if not atomic_imwrite_png(selected, self._result.mask_image):
                raise RuntimeError('Failed to write mask image.')
        except Exception as exc:
            logger.exception('Single mask export failed')
            messagebox.showerror('Export Mask', str(exc))
            return
        self._status.config(text=f'Exported: {selected}')

    def _clear(self) -> None:
        if self._working:
            return
        self._result = None
        self._photo = None
        self._drop_label.config(
            text='Drag & Drop One Image Here', bg='#eef6ff'
        )
        self._status.config(text='Ready')
        self._set_controls(True)
        self._draw_placeholder()

    def _set_controls(self, enabled) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self._browse_btn.config(state=state)
        self._clear_btn.config(state=state)
        export_state = (
            tk.NORMAL if enabled and self._result is not None else tk.DISABLED
        )
        self._export_btn.config(state=export_state)

    def _draw_placeholder(self) -> None:
        self._canvas.delete('all')
        self._canvas.create_text(
            max(self._canvas.winfo_width(), 300) / 2,
            max(self._canvas.winfo_height(), 220) / 2,
            text='The final mask image will appear here.',
            fill='#aeb6c2',
            font=('Segoe UI', 12),
        )

    def _render_mask(self) -> None:
        if self._result is None:
            return
        image = self._result.mask_image
        height, width = image.shape[:2]
        canvas_w = max(self._canvas.winfo_width(), 300)
        canvas_h = max(self._canvas.winfo_height(), 220)
        scale = min((canvas_w - 24) / width, (canvas_h - 48) / height, 1.0)
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_NEAREST,
            )
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            self._fail('Failed to render mask image.')
            return
        self._photo = tk.PhotoImage(data=encoded.tobytes())
        self._canvas.delete('all')
        self._canvas.create_text(
            10, 10, text='Final Mask', anchor=tk.NW,
            fill='#e5e7eb', font=('Segoe UI', 11, 'bold'),
        )
        self._canvas.create_image(
            canvas_w / 2, 34, image=self._photo, anchor=tk.N
        )

    def reload_settings(self) -> None:
        if self._working:
            return
        self._engine = None
        self._status.config(text='Settings reloaded')
