# main.py â€” QCML ì‹œìŠ¤í…œ ì§„ìž…ì 

import logging
import os
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path

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
    debug_enabled = os.environ.get("QCML_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    logging.basicConfig(
        level=logging.DEBUG if debug_enabled else logging.WARNING,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    root = tk.Tk()
    root.title("Cell Li-Plating Quantification System")
    root.geometry("1280x820")
    root.minsize(1100, 720)
    try:
        root.state("zoomed")
    except tk.TclError:
        pass

    loading, loading_status, progress = _create_loading_page(root)
    root.update_idletasks()

    def build_main_ui() -> None:
        try:
            loading_status.config(text="Loading workflow and analysis modules...")
            root.update_idletasks()

            from gui.workflow_tab import WorkflowTab
            from gui.analysis_tab import AnalysisTab
            from gui.settings_tab import SettingsTab
            from gui.guide_tab import GuideTab

            notebook = ttk.Notebook(root)
            reloadable_tabs = []

            def reload_tabs_from_config() -> None:
                for tab in reloadable_tabs:
                    tab.reload_config()

            loading_status.config(text="Building Workflow tab...")
            root.update_idletasks()
            workflow_tab = WorkflowTab(notebook, config_path=CONFIG_PATH)
            notebook.add(workflow_tab, text="  Workflow  ")
            reloadable_tabs.append(workflow_tab)

            loading_status.config(text="Building Analysis tab...")
            root.update_idletasks()
            analysis_tab = AnalysisTab(
                notebook,
                config_path=CONFIG_PATH,
            )
            notebook.add(analysis_tab, text="  Analysis  ")
            reloadable_tabs.append(analysis_tab)

            loading_status.config(text="Building Settings tab...")
            root.update_idletasks()
            settings_tab = SettingsTab(
                notebook,
                config_path=CONFIG_PATH,
                on_config_saved=reload_tabs_from_config,
            )
            notebook.add(settings_tab, text="  Settings  ")

            loading_status.config(text="Building Guide tab...")
            root.update_idletasks()
            guide_tab = GuideTab(
                notebook,
                config_path=CONFIG_PATH,
            )
            notebook.add(guide_tab, text="  Guide  ")
            reloadable_tabs.append(guide_tab)

            progress.stop()
            loading.destroy()
            notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        except Exception as exc:
            logging.exception("Application startup failed")
            progress.stop()
            loading_status.config(text=f"Startup failed: {exc}")
            messagebox.showerror("Startup Error", str(exc))

    root.after(100, build_main_ui)

    root.mainloop()


if __name__ == "__main__":
    main()

