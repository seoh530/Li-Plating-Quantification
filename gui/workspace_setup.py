"""First-run workspace selection, before any tabs scan the filesystem."""

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from modules.utils.app_paths import save_workspace_root


class WorkspaceSetup(ttk.Frame):
    def __init__(self, parent, config_path, on_complete):
        super().__init__(parent, padding=40)
        self._config_path = Path(config_path)
        self._on_complete = on_complete
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        card = ttk.Frame(self, style="Card.TFrame", padding=32)
        card.grid(row=1, column=0, sticky=tk.EW, padx=80)
        card.columnconfigure(0, weight=1)
        ttk.Label(card, text="Choose your workspace", style="CardTitle.TLabel").grid(
            row=0, column=0, columnspan=2, sticky=tk.W
        )
        ttk.Label(
            card,
            text="Select the root folder containing your analysis folders.\nYou can change it later in Settings.",
            style="Panel.TLabel",
            justify=tk.LEFT,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(12, 24))
        self._folder = tk.StringVar()
        self._entry = ttk.Entry(card, textvariable=self._folder)
        self._entry.grid(row=2, column=0, sticky=tk.EW)
        ttk.Button(card, text="Browse", command=self._browse).grid(
            row=2, column=1, padx=(12, 0)
        )
        self._status = ttk.Label(
            card, text="", style="Panel.TLabel", foreground="#b91c1c"
        )
        self._status.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=12)
        self._continue_btn = ttk.Button(
            card,
            text="Open workspace",
            style="Accent.TButton",
            command=self._complete,
            state=tk.DISABLED,
        )
        self._continue_btn.grid(row=4, column=0, columnspan=2, sticky=tk.E)
        self._folder.trace_add(
            "write",
            lambda *_: self._continue_btn.configure(
                state=tk.NORMAL if self._folder.get().strip() else tk.DISABLED
            ),
        )

    def _browse(self):
        folder = filedialog.askdirectory(
            parent=self.winfo_toplevel(),
            title="Choose workspace root folder",
            mustexist=True,
        )
        if folder:
            self._folder.set(folder)
            self._status.configure(text="")

    def _complete(self):
        try:
            save_workspace_root(
                self._config_path, self._folder.get().strip().strip('"')
            )
        except (OSError, ValueError) as exc:
            self._status.configure(text=str(exc))
            return
        self.destroy()
        self._on_complete()
