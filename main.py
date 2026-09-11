# main.py â€” QCML ì‹œìŠ¤í…œ ì§„ìž…ì 

import logging
import os
import sys
import time
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

DEBUG = False

PROJECT_ROOT = Path(__file__).resolve().parent
project_root_str = str(PROJECT_ROOT)
if sys.path[:1] != [project_root_str]:
    sys.path = [path for path in sys.path if path != project_root_str]
    sys.path.insert(0, project_root_str)


def _loaded_from_project(module_name: str) -> bool:
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return True
    try:
        Path(module_file).resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


for _package_name in ("modules", "gui"):
    if not _loaded_from_project(_package_name):
        for _name in list(sys.modules):
            if _name == _package_name or _name.startswith(f"{_package_name}."):
                sys.modules.pop(_name, None)

CONFIG_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"


class _StartupTimer:
    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._start = time.perf_counter()
        self._last = self._start

    def mark(self, label: str) -> None:
        if not self._enabled:
            return
        now = time.perf_counter()
        logging.debug(
            "startup %-32s step=%.3fs total=%.3fs",
            label,
            now - self._last,
            now - self._start,
        )
        self._last = now


class LazyTab(ttk.Frame):
    """Placeholder that builds the real tab only when first selected."""

    def __init__(self, parent, title: str, factory) -> None:
        super().__init__(parent)
        self._title = title
        self._factory = factory
        self._created = False
        self._content = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        ttk.Label(
            self,
            text=f"Click to load {title.strip()}",
            foreground="#555555",
        ).grid(row=0, column=0)

    @property
    def content(self):
        return self._content

    def ensure_loaded(self):
        if self._created:
            return self._content
        for child in self.winfo_children():
            child.destroy()
        self._content = self._factory(self)
        self._content.grid(row=0, column=0, sticky=tk.NSEW)
        self._created = True
        return self._content


def _create_loading_page(root: tk.Tk) -> tuple[ttk.Frame, ttk.Label, ttk.Progressbar]:
    loading = ttk.Frame(root, padding=32)
    loading.pack(fill=tk.BOTH, expand=True)
    loading.columnconfigure(0, weight=1)
    loading.rowconfigure(0, weight=1)
    loading.rowconfigure(4, weight=1)

    ttk.Label(
        loading,
        text="Cell Li-Plating Quantification System",
        font=("TkDefaultFont", 18, "bold"),
    ).grid(row=1, column=0, pady=(0, 10))

    status = ttk.Label(
        loading,
        text="Loading application...",
        foreground="#555555",
    )
    status.grid(row=2, column=0, pady=(0, 14))

    progress = ttk.Progressbar(loading, mode="indeterminate", length=420)
    progress.grid(row=3, column=0, sticky=tk.N)
    progress.start(12)

    return loading, status, progress


def main() -> None:
    debug_enabled = DEBUG or os.environ.get("QCML_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    startup_timer = _StartupTimer(debug_enabled)
    logging.basicConfig(
        level=logging.DEBUG if debug_enabled else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    root = tk.Tk()
    startup_timer.mark("tk root")
    root.title("Cell Li-Plating Quantification System")
    root.geometry("1280x820")
    root.minsize(1100, 720)
    try:
        root.state("zoomed")
    except tk.TclError:
        pass

    loading, loading_status, progress = _create_loading_page(root)
    root.update_idletasks()
    startup_timer.mark("loading page")

    def build_main_ui() -> None:
        try:
            loading_status.config(text="Loading workflow module...")
            root.update_idletasks()

            startup_timer.mark("before workflow import")
            from gui.workflow_tab import WorkflowTab
            startup_timer.mark("workflow import")

            notebook = ttk.Notebook(root)
            reloadable_tabs = []

            def reload_tabs_from_config() -> None:
                for tab in reloadable_tabs:
                    tab.reload_config()

            def add_reloadable(tab) -> None:
                if tab not in reloadable_tabs:
                    reloadable_tabs.append(tab)

            def on_tab_changed(_event=None) -> None:
                selected = notebook.nametowidget(notebook.select())
                if not isinstance(selected, LazyTab):
                    return
                startup_timer.mark(f"before lazy {selected._title.strip()}")
                try:
                    tab = selected.ensure_loaded()
                    if hasattr(tab, "reload_config"):
                        add_reloadable(tab)
                    startup_timer.mark(f"lazy {selected._title.strip()} loaded")
                except Exception as exc:
                    logging.exception("%s tab startup failed", selected._title.strip())
                    messagebox.showerror("Startup Error", f"{selected._title.strip()} tab failed: {exc}")

            loading_status.config(text="Building Workflow tab...")
            root.update_idletasks()
            startup_timer.mark("before workflow tab")
            workflow_tab = WorkflowTab(notebook, config_path=CONFIG_PATH)
            notebook.add(workflow_tab, text="  Workflow  ")
            add_reloadable(workflow_tab)
            startup_timer.mark("workflow tab")

            analysis_lazy = LazyTab(
                notebook,
                "Analysis",
                lambda parent: __import__("gui.analysis_tab", fromlist=["AnalysisTab"]).AnalysisTab(
                    parent,
                    config_path=CONFIG_PATH,
                ),
            )
            notebook.add(analysis_lazy, text="  Analysis  ")

            settings_lazy = LazyTab(
                notebook,
                "Settings",
                lambda parent: __import__("gui.settings_tab", fromlist=["SettingsTab"]).SettingsTab(
                    parent,
                    config_path=CONFIG_PATH,
                    on_config_saved=reload_tabs_from_config,
                ),
            )
            notebook.add(settings_lazy, text="  Settings  ")

            guide_lazy = LazyTab(
                notebook,
                "Guide",
                lambda parent: __import__("gui.guide_tab", fromlist=["GuideTab"]).GuideTab(
                    parent,
                    config_path=CONFIG_PATH,
                ),
            )
            notebook.add(guide_lazy, text="  Guide  ")
            notebook.bind("<<NotebookTabChanged>>", on_tab_changed, add="+")

            progress.stop()
            loading.destroy()
            notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            startup_timer.mark("notebook visible")
        except Exception as exc:
            logging.exception("Application startup failed")
            progress.stop()
            loading_status.config(text=f"Startup failed: {exc}")
            messagebox.showerror("Startup Error", str(exc))

    root.after(100, build_main_ui)

    root.mainloop()


if __name__ == "__main__":
    main()

