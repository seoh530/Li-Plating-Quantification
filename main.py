import logging
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from modules.utils.app_paths import ensure_user_config


APP_TITLE = "Li-Plating Quant System"
APP_SUBTITLE = "QC, ML inference, and result review"

_COLOR_BG = "#f4f6f8"
_COLOR_PANEL = "#ffffff"
_COLOR_BORDER = "#d7dde5"
_COLOR_TEXT = "#17202a"
_COLOR_MUTED = "#667085"
_COLOR_PRIMARY = "#2563eb"
_COLOR_PRIMARY_DARK = "#1d4ed8"
_COLOR_SELECTED = "#cbd5e1"


def _configure_style(root: tk.Tk) -> ttk.Style:
    root.configure(bg=_COLOR_BG)

    default_font = tkfont.nametofont("TkDefaultFont")
    default_font.configure(family="Segoe UI", size=10)
    tkfont.nametofont("TkTextFont").configure(family="Segoe UI", size=10)
    tkfont.nametofont("TkHeadingFont").configure(
        family="Segoe UI",
        size=10,
        weight="bold",
    )

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=default_font, background=_COLOR_BG, foreground=_COLOR_TEXT)
    style.configure("TFrame", background=_COLOR_BG)
    style.configure("App.TFrame", background=_COLOR_BG)
    style.configure("Panel.TFrame", background=_COLOR_PANEL)
    style.configure("TLabel", background=_COLOR_BG, foreground=_COLOR_TEXT)
    style.configure("Muted.TLabel", foreground=_COLOR_MUTED, background=_COLOR_BG)
    style.configure("Panel.TLabel", background=_COLOR_PANEL, foreground=_COLOR_TEXT)
    style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), background=_COLOR_BG)
    style.configure("Subtitle.TLabel", foreground=_COLOR_MUTED, background=_COLOR_BG)
    style.configure("Status.TLabel", foreground=_COLOR_MUTED, background=_COLOR_PANEL)

    style.configure(
        "TButton",
        padding=(12, 6),
        background="#eef2f7",
        bordercolor=_COLOR_BORDER,
        focusthickness=1,
        focuscolor=_COLOR_PRIMARY,
    )
    style.map(
        "TButton",
        background=[
            ("active", "#e5eaf2"),
            ("pressed", "#d9e1ec"),
            ("disabled", "#f1f3f6"),
        ],
        foreground=[("disabled", "#98a2b3")],
    )
    style.configure(
        "Accent.TButton",
        foreground="#ffffff",
        background=_COLOR_PRIMARY,
        bordercolor=_COLOR_PRIMARY_DARK,
    )
    style.map(
        "Accent.TButton",
        background=[("active", _COLOR_PRIMARY_DARK), ("pressed", "#1e40af")],
        foreground=[("disabled", "#e5e7eb")],
    )
    style.configure(
        "TNotebook",
        background=_COLOR_BG,
        borderwidth=0,
        tabmargins=(0, 0, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        padding=(18, 9),
        background="#e9eef5",
        foreground=_COLOR_MUTED,
        bordercolor=_COLOR_BORDER,
        lightcolor="#e9eef5",
        darkcolor="#e9eef5",
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", _COLOR_PANEL), ("active", "#f8fafc")],
        foreground=[("selected", _COLOR_TEXT), ("active", _COLOR_TEXT)],
    )
    style.configure(
        "TLabelframe",
        background=_COLOR_BG,
        bordercolor=_COLOR_BORDER,
        relief=tk.SOLID,
    )
    style.configure(
        "TLabelframe.Label",
        background=_COLOR_BG,
        foreground=_COLOR_TEXT,
        font=("Segoe UI", 10, "bold"),
    )
    style.configure(
        "Treeview",
        background=_COLOR_PANEL,
        fieldbackground=_COLOR_PANEL,
        foreground=_COLOR_TEXT,
        rowheight=26,
        bordercolor=_COLOR_BORDER,
        borderwidth=1,
    )
    style.configure(
        "Treeview.Heading",
        background="#eef2f7",
        foreground=_COLOR_TEXT,
        padding=(8, 6),
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "Treeview",
        background=[("selected", _COLOR_SELECTED)],
        foreground=[("selected", _COLOR_TEXT)],
    )
    style.configure("TCombobox", padding=(6, 4))
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor="#e5eaf2",
        background=_COLOR_PRIMARY,
    )
    return style


def _create_loading_page(root: tk.Tk) -> tuple[ttk.Frame, ttk.Label, ttk.Progressbar]:
    loading = ttk.Frame(root, style="App.TFrame", padding=42)
    loading.pack(fill=tk.BOTH, expand=True)
    loading.columnconfigure(0, weight=1)
    loading.rowconfigure(0, weight=1)
    loading.rowconfigure(5, weight=1)

    ttk.Label(
        loading,
        text=APP_TITLE,
        style="Title.TLabel",
    ).grid(row=1, column=0, pady=(0, 10))

    ttk.Label(
        loading,
        text=APP_SUBTITLE,
        style="Subtitle.TLabel",
    ).grid(row=2, column=0, pady=(0, 20))

    status = ttk.Label(
        loading,
        text="Loading application...",
        style="Muted.TLabel",
    )
    status.grid(row=3, column=0, pady=(0, 14))

    progress = ttk.Progressbar(loading, mode="indeterminate", length=420)
    progress.grid(row=4, column=0, sticky=tk.N)
    progress.start(12)

    return loading, status, progress


def _create_app_shell(root: tk.Tk, config_path) -> tuple[ttk.Frame, ttk.Notebook, ttk.Label]:
    shell = ttk.Frame(root, style="App.TFrame", padding=(14, 12, 14, 10))
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(1, weight=1)

    header = ttk.Frame(shell, style="App.TFrame")
    header.grid(row=0, column=0, sticky=tk.EW, pady=(0, 10))
    header.columnconfigure(1, weight=1)

    title_group = ttk.Frame(header, style="App.TFrame")
    title_group.grid(row=0, column=0, sticky=tk.W)
    ttk.Label(title_group, text=APP_TITLE, style="Title.TLabel").grid(
        row=0,
        column=0,
        sticky=tk.W,
    )
    ttk.Label(title_group, text=APP_SUBTITLE, style="Subtitle.TLabel").grid(
        row=1,
        column=0,
        sticky=tk.W,
        pady=(2, 0),
    )

    config_label = ttk.Label(
        header,
        text=f"Config: {config_path}",
        style="Muted.TLabel",
    )
    config_label.grid(row=0, column=1, rowspan=2, sticky=tk.E, padx=(16, 0))

    notebook = ttk.Notebook(shell)
    notebook.grid(row=1, column=0, sticky=tk.NSEW)

    status_bar = ttk.Frame(shell, style="Panel.TFrame", padding=(10, 5))
    status_bar.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))
    status_bar.columnconfigure(0, weight=1)
    status_label = ttk.Label(status_bar, text="Ready", style="Status.TLabel")
    status_label.grid(row=0, column=0, sticky=tk.W)

    return shell, notebook, status_label


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config_path = ensure_user_config()
    except Exception as exc:
        messagebox.showerror("Startup Error", f"Failed to initialize user settings:\n{exc}")
        return

    try:
        from tkinterdnd2 import TkinterDnD

        root = TkinterDnD.Tk()
    except Exception:
        root = tk.Tk()
    _configure_style(root)
    root.title(APP_TITLE)
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
            loading_status.config(text="Loading main and analysis modules...")
            root.update_idletasks()

            from gui.analysis_tab import AnalysisTab
            from gui.guide_tab import GuideTab
            from gui.report_tab import ReportTab
            from gui.settings_tab import SettingsTab
            from gui.single_mask_tab import SingleMaskTab
            from gui.workflow_tab import WorkflowTab

            shell, notebook, status_label = _create_app_shell(root, config_path)

            loading_status.config(text="Building Main tab...")
            root.update_idletasks()
            workflow_tab = WorkflowTab(notebook, config_path=config_path)
            single_mask_tab = SingleMaskTab(notebook, config_path=config_path)
            notebook.add(workflow_tab, text="Main")

            loading_status.config(text="Building Analysis tab...")
            root.update_idletasks()
            analysis_tab = AnalysisTab(notebook, config_path=config_path)
            notebook.add(single_mask_tab, text='Single Mask Mode')
            notebook.add(analysis_tab, text="Analysis")

            loading_status.config(text="Building Report tab...")
            root.update_idletasks()
            report_tab = ReportTab(notebook, config_path=config_path)
            notebook.add(report_tab, text="Report")

            def reload_runtime_settings() -> None:
                for tab in (
                    workflow_tab,
                    single_mask_tab,
                    analysis_tab,
                    report_tab,
                    guide_tab,
                ):
                    reload_method = getattr(tab, "reload_settings", None)
                    if reload_method is not None:
                        reload_method()

            loading_status.config(text="Building Settings tab...")
            root.update_idletasks()
            settings_tab = SettingsTab(
                notebook,
                config_path=config_path,
                on_settings_saved=reload_runtime_settings,
            )
            notebook.add(settings_tab, text="Settings")

            loading_status.config(text="Building Guide tab...")
            root.update_idletasks()
            guide_tab = GuideTab(notebook, config_path=config_path)
            notebook.add(guide_tab, text="Guide")

            progress.stop()
            loading.destroy()
            shell.pack(fill=tk.BOTH, expand=True)
            shell.tkraise()
            status_label.config(text="Ready")
        except Exception as exc:
            logging.exception("Application startup failed")
            progress.stop()
            loading_status.config(text=f"Startup failed: {exc}")
            messagebox.showerror("Startup Error", str(exc))

    root.after(100, build_main_ui)

    root.mainloop()


if __name__ == "__main__":
    main()



