from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import yaml

from modules.ml.model_registry import display_model_path, list_model_options, read_config_model_path


class SettingsTab(ttk.Frame):
    """Runtime root/model settings."""

    def __init__(self, parent, config_path: str | Path, on_config_saved=None) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._on_config_saved = on_config_saved
        self._root_path_var = tk.StringVar()
        self._model_path_var = tk.StringVar()
        self._status_var = tk.StringVar()
        self._presets: list[dict[str, str]] = []

        self._load_config()
        self._build_ui()
        self._render_presets()

    def _load_config(self) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        app = cfg.setdefault("app", {})
        current_root = str(app.get("root_path", "") or "")
        current_model = display_model_path(read_config_model_path(self._config_path))
        presets = list(app.get("root_model_presets", []) or [])

        for name, root_path in dict(app.get("site_root_paths", {}) or {}).items():
            model_path = dict(app.get("site_model_paths", {}) or {}).get(name, current_model)
            presets.append({"root_path": str(root_path), "model_path": display_model_path(model_path)})

        if current_root:
            presets.append({"root_path": current_root, "model_path": current_model})

        self._presets = self._dedupe_presets(presets)
        self._root_path_var.set(current_root)
        self._model_path_var.set(current_model)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        frame = ttk.Frame(self, padding=16)
        frame.grid(row=0, column=0, sticky=tk.NSEW)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(4, weight=1)

        ttk.Label(frame, text="Settings", font=("TkDefaultFont", 14, "bold")).grid(
            row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 12)
        )

        ttk.Label(frame, text="Root Path").grid(row=1, column=0, sticky=tk.W, pady=(0, 8))
        root_entry = ttk.Entry(frame, textvariable=self._root_path_var)
        root_entry.grid(row=1, column=1, columnspan=2, sticky=tk.EW, pady=(0, 8))
        root_entry.bind("<Return>", lambda _event: self._save_current())
        root_entry.bind("<KP_Enter>", lambda _event: self._save_current())

        ttk.Label(frame, text="Model").grid(row=2, column=0, sticky=tk.W, pady=(0, 8))
        self._model_combo = ttk.Combobox(
            frame,
            textvariable=self._model_path_var,
            values=list_model_options(self._config_path),
            state="normal",
        )
        self._model_combo.grid(row=2, column=1, columnspan=2, sticky=tk.EW, pady=(0, 8))

        ttk.Button(frame, text="Save / Update", command=self._save_current).grid(
            row=3, column=1, sticky=tk.W, pady=(4, 10)
        )

        self._tree = ttk.Treeview(
            frame,
            columns=("root", "model"),
            show="headings",
            height=10,
            selectmode="browse",
        )
        self._tree.heading("root", text="Root Path")
        self._tree.heading("model", text="Model")
        self._tree.column("root", width=680, anchor=tk.W)
        self._tree.column("model", width=280, anchor=tk.W)
        self._tree.grid(row=4, column=0, columnspan=3, sticky=tk.NSEW, pady=(4, 0))
        self._tree.tag_configure("active", background="#dbeafe")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_selected)

        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._tree.yview)
        scroll.grid(row=4, column=3, sticky=tk.NS, pady=(4, 0))
        self._tree.configure(yscrollcommand=scroll.set)

        ttk.Label(frame, textvariable=self._status_var, foreground="#555555").grid(
            row=5, column=0, columnspan=3, sticky=tk.W, pady=(10, 0)
        )

    def _render_presets(self) -> None:
        for item_id in self._tree.get_children():
            self._tree.delete(item_id)
        for preset in self._presets:
            tags = ("active",) if self._is_active_preset(preset) else ()
            self._tree.insert(
                "",
                tk.END,
                values=(preset["root_path"], preset["model_path"]),
                tags=tags,
            )

    def _on_tree_selected(self, _event=None) -> None:
        selected = self._tree.selection()
        if not selected:
            return
        values = self._tree.item(selected[0], "values")
        if len(values) < 2:
            return
        root_path = str(values[0])
        model_path = str(values[1])
        self._root_path_var.set(root_path)
        self._model_path_var.set(model_path)
        self._write_config(root_path, model_path)
        self._render_presets()
        self._status_var.set("Selected setting applied")
        if self._on_config_saved is not None:
            self._on_config_saved()

    def _save_current(self) -> None:
        root_path = self._root_path_var.get().strip().strip("\"'")
        model_path = self._model_path_var.get().strip().strip("\"'")
        if not root_path or not model_path:
            messagebox.showinfo("Settings", "Enter Root Path and Model.")
            return
        self._presets = self._dedupe_presets(
            [
                preset
                for preset in self._presets
                if preset.get("root_path") != root_path
            ]
            + [{"root_path": root_path, "model_path": model_path}]
        )
        self._write_config(root_path, model_path)
        self._render_presets()
        self._status_var.set("Saved / updated setting")
        if self._on_config_saved is not None:
            self._on_config_saved()

    def _is_active_preset(self, preset: dict[str, str]) -> bool:
        root_path = str(preset.get("root_path", "") or "").strip().rstrip("\\/")
        model_path = display_model_path(str(preset.get("model_path", "") or "").strip())
        active_root = self._root_path_var.get().strip().rstrip("\\/")
        active_model = display_model_path(self._model_path_var.get().strip())
        return root_path.lower() == active_root.lower() and model_path == active_model

    def _write_config(self, root_path: str, model_path: str) -> None:
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        app = cfg.setdefault("app", {})
        ml = cfg.setdefault("ml", {})
        app["root_path"] = root_path
        app["root_model_presets"] = self._presets
        app.pop("active_site", None)
        app.pop("site_root_paths", None)
        app.pop("site_model_paths", None)
        ml["model_path"] = model_path
        with open(self._config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)

    def _dedupe_presets(self, presets: list[dict]) -> list[dict[str, str]]:
        unique: dict[str, str] = {}
        for preset in presets:
            root_path = str(preset.get("root_path", "") or "").strip()
            model_path = str(preset.get("model_path", "") or "").strip()
            if root_path and model_path:
                unique[root_path] = display_model_path(model_path)
        return [
            {"root_path": root_path, "model_path": model_path}
            for root_path, model_path in sorted(unique.items(), key=lambda item: item[0].lower())
        ]
