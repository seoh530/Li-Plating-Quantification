from collections.abc import Callable
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

from modules.ml.model_registry import (
    is_model_dir,
    list_model_options,
    project_root,
    resolve_model_path,
)
from modules.utils.app_paths import user_database_path
from modules.utils.image_io import atomic_write_text


class SettingsTab(ttk.Frame):
    """Persist deployment/runtime settings that operators may change."""

    def __init__(
        self,
        parent,
        config_path: str | Path,
        on_settings_saved: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path)
        self._on_settings_saved = on_settings_saved
        self._root_path_var = tk.StringVar()
        self._model_path_var = tk.StringVar()
        self._db_path_var = tk.StringVar()
        self._model_status_var = tk.StringVar()

        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        content = ttk.Frame(self, padding=(24, 18, 24, 16))
        content.grid(row=0, column=0, sticky=tk.NSEW)
        content.columnconfigure(1, weight=1)

        ttk.Label(
            content,
            text="Settings",
            font=("Segoe UI", 19, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 14))

        fields = ttk.Frame(content, style="Card.TFrame", padding=18)
        fields.grid(row=1, column=0, columnspan=3, sticky=tk.EW)
        fields.columnconfigure(1, weight=1)
        ttk.Label(fields, style="Panel.TLabel", text="Root folder").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10)
        )
        self._root_entry = ttk.Entry(fields, textvariable=self._root_path_var, width=60)
        self._root_entry.grid(row=1, column=1, sticky=tk.EW)
        ttk.Button(fields, text="Browse", command=self._browse_root_path).grid(
            row=1,
            column=2,
            sticky=tk.W,
            padx=(8, 0),
        )

        ttk.Label(fields, style="Panel.TLabel", text="Model path").grid(
            row=2,
            column=0,
            sticky=tk.W,
            padx=(0, 10),
            pady=(10, 0),
        )
        self._model_combo = ttk.Combobox(
            fields,
            textvariable=self._model_path_var,
            width=60,
        )
        self._model_combo.grid(row=2, column=1, sticky=tk.EW, pady=(10, 0))
        self._model_combo.bind(
            "<<ComboboxSelected>>", lambda _event: self._refresh_model_status()
        )
        ttk.Button(fields, text="Browse", command=self._browse_model_path).grid(
            row=2,
            column=2,
            sticky=tk.W,
            padx=(8, 0),
            pady=(10, 0),
        )

        ttk.Label(fields, style="Panel.TLabel", text="Model status").grid(
            row=3,
            column=0,
            sticky=tk.W,
            padx=(0, 10),
            pady=(6, 0),
        )
        ttk.Label(
            fields,
            style="Panel.TLabel",
            textvariable=self._model_status_var,
            foreground="#555555",
            wraplength=650,
        ).grid(
            row=3,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=(6, 0),
        )

        ttk.Label(fields, style="Panel.TLabel", text="SQLite DB").grid(
            row=4,
            column=0,
            sticky=tk.W,
            padx=(0, 10),
            pady=(10, 0),
        )
        ttk.Label(
            fields,
            style="Panel.TLabel",
            textvariable=self._db_path_var,
            foreground="#555555",
            wraplength=650,
        ).grid(
            row=4,
            column=1,
            columnspan=2,
            sticky=tk.EW,
            pady=(10, 0),
        )

        actions = ttk.Frame(fields, style="Panel.TFrame")
        actions.grid(row=5, column=0, columnspan=3, sticky=tk.EW, pady=(16, 0))
        ttk.Button(
            actions, text="Save settings", style="Accent.TButton", command=self._save
        ).pack(side=tk.LEFT)
        ttk.Button(actions, text="Reload", command=self._load_values).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        self._status_label = ttk.Label(
            actions, text="", style="Panel.TLabel", foreground="#555555"
        )
        self._status_label.pack(side=tk.LEFT, padx=(12, 0))

        notes = ttk.LabelFrame(self, text="Workspace settings", padding=12)
        notes.grid(row=1, column=0, sticky="new", padx=24, pady=(0, 18))
        notes.columnconfigure(1, weight=1)
        rows = [
            (
                "PhotoJig data",
                "Main, Analysis and Report discover analysis folders under this root; Analysis also offers a date filter.",
            ),
            (
                "Default model",
                "Main uses the saved model path as the default after restart.",
            ),
            (
                "Persistence",
                "Settings are stored in this Windows user's local application data folder.",
            ),
            (
                "Monitoring DB",
                "Each Windows user has a private SQLite DB under LOCALAPPDATA; no SQLite app is required.",
            ),
        ]
        for index, (label, text) in enumerate(rows):
            ttk.Label(
                notes, text=label, font=("TkDefaultFont", 10, "bold"), width=18
            ).grid(
                row=index,
                column=0,
                sticky=tk.NW,
                padx=(0, 12),
                pady=4,
            )
            ttk.Label(notes, text=text, wraplength=850).grid(
                row=index,
                column=1,
                sticky=tk.EW,
                pady=4,
            )

    def _load_values(self) -> None:
        cfg = self._read_config()
        root_path = str(cfg.get("app", {}).get("root_path", ""))
        model_path = str(cfg.get("ml", {}).get("model_path", ""))
        model_options = list_model_options(self._config_path)

        self._root_path_var.set(root_path)
        self._model_path_var.set(model_path)
        self._model_combo["values"] = model_options
        self._db_path_var.set(str(self._db_path_for_root(root_path, cfg)))
        self._refresh_model_status()
        self._status_label.config(text="Loaded")

    def _browse_root_path(self) -> None:
        initial = self._root_path_var.get().strip()
        options = {"parent": self, "title": "Select workspace root folder"}
        if initial and Path(initial).is_dir():
            options["initialdir"] = initial
        folder = filedialog.askdirectory(**options)
        if folder:
            self._root_path_var.set(folder)
            self._db_path_var.set(
                str(self._db_path_for_root(folder, self._read_config()))
            )

    def _browse_model_path(self) -> None:
        initial = self._resolved_model_path(self._model_path_var.get().strip())
        options = {"parent": self, "title": "Select model folder"}
        if initial.is_dir():
            options["initialdir"] = str(initial)
        else:
            models_root = project_root() / "models"
            if models_root.is_dir():
                options["initialdir"] = str(models_root)
        folder = filedialog.askdirectory(**options)
        if folder:
            self._model_path_var.set(self._model_path_for_save(folder))
            self._refresh_model_status()

    def _save(self) -> None:
        root_path = self._root_path_var.get().strip().strip('"')
        if not root_path:
            messagebox.showwarning("Settings", "Root path is required.")
            return

        path = Path(root_path).expanduser()
        if path.exists() and not path.is_dir():
            messagebox.showerror("Settings", "Root folder must be a directory.")
            return
        if not path.exists():
            create = messagebox.askyesno(
                "Settings",
                f"Root path does not exist:\n{path}\n\nCreate this folder?",
            )
            if not create:
                return
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                messagebox.showerror("Settings", f"Failed to create root path:\n{exc}")
                return

        model_path = self._model_path_var.get().strip().strip('"')
        if not model_path:
            messagebox.showwarning("Settings", "Model path is required.")
            return
        if not is_model_dir(model_path):
            save_anyway = messagebox.askyesno(
                "Settings",
                "This model folder does not contain metadata.json and model.txt/model.joblib.\n\nSave anyway?",
            )
            if not save_anyway:
                return

        try:
            saved_root_path = str(path.resolve())
            saved_model_path = self._model_path_for_save(model_path)
            self._replace_config_values(
                {
                    ("app", "root_path"): saved_root_path,
                    ("ml", "model_path"): saved_model_path,
                }
            )
            self._root_path_var.set(saved_root_path)
            self._model_path_var.set(saved_model_path)
            self._db_path_var.set(
                str(self._db_path_for_root(saved_root_path, self._read_config()))
            )
            self._model_combo["values"] = list_model_options(self._config_path)
            self._refresh_model_status()
        except Exception as exc:
            messagebox.showerror("Settings", f"Failed to save settings:\n{exc}")
            return

        self._status_label.config(text="Saved")
        if self._on_settings_saved is not None:
            self._on_settings_saved()

    def _read_config(self) -> dict:
        with open(self._config_path, "r", encoding="utf-8-sig") as fh:
            return yaml.safe_load(fh) or {}

    def _replace_config_value(self, section: str, key: str, value: str) -> None:
        self._replace_config_values({(section, key): value})

    def _replace_config_values(self, changes: dict[tuple[str, str], str]) -> None:
        text = self._config_path.read_text(encoding="utf-8-sig")
        for (section, key), value in changes.items():
            in_section = False
            replaced = False
            output: list[str] = []
            for line in text.splitlines(keepends=True):
                stripped = line.strip()
                if re.match(rf"^{re.escape(section)}\s*:", line):
                    in_section = True
                    output.append(line)
                    continue
                if (
                    in_section
                    and not line.startswith((" ", "\t"))
                    and stripped
                    and not stripped.startswith("#")
                ):
                    in_section = False
                if in_section and re.match(rf"^\s+{re.escape(key)}\s*:", line):
                    indent = line[: len(line) - len(line.lstrip())]
                    escaped = value.replace("'", "''")
                    output.append(f"{indent}{key}: '{escaped}'\n")
                    replaced = True
                else:
                    output.append(line)
            if not replaced:
                raise RuntimeError(f"{section}.{key} entry was not found in config.")
            text = "".join(output)
        parsed = yaml.safe_load(text)
        if any(
            parsed[section][key] != value for (section, key), value in changes.items()
        ):
            raise RuntimeError("Updated config did not preserve the requested values.")
        atomic_write_text(self._config_path, text)

    def _resolved_model_path(self, value: str) -> Path:
        return resolve_model_path(value) if value else project_root() / "models"

    def _model_path_for_save(self, value: str) -> str:
        raw = value.strip().strip('"')
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return raw.replace("\\", "/")
        try:
            return path.resolve().relative_to(project_root()).as_posix()
        except (OSError, ValueError):
            return str(path)

    def _refresh_model_status(self) -> None:
        model_path = self._model_path_var.get().strip()
        if not model_path:
            self._model_status_var.set("No model selected")
            return
        resolved = self._resolved_model_path(model_path)
        if is_model_dir(model_path):
            self._model_status_var.set(f"OK: {resolved}")
        else:
            self._model_status_var.set(f"Missing model files: {resolved}")

    def _db_path_for_root(self, root_path: str, cfg: dict) -> Path:
        del root_path, cfg
        return user_database_path()
