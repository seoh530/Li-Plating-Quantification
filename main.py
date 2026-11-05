import logging
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox, ttk

from modules.utils.app_paths import ensure_user_config, workspace_setup_required


APP_TITLE = "Li-Salt Quantification"

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

    style.configure(
        ".", font=default_font, background=_COLOR_BG, foreground=_COLOR_TEXT
    )
    style.configure("TFrame", background=_COLOR_BG)
    style.configure("App.TFrame", background=_COLOR_BG)
    style.configure("Panel.TFrame", background=_COLOR_PANEL)
    style.configure(
        "Card.TFrame",
        background=_COLOR_PANEL,
        bordercolor=_COLOR_BORDER,
        relief=tk.SOLID,
        borderwidth=1,
    )
    style.configure(
        "CardTitle.TLabel",
        background=_COLOR_PANEL,
        foreground=_COLOR_TEXT,
        font=("Segoe UI", 22, "bold"),
    )
    style.configure("Compact.TButton", padding=(7, 6))
    style.configure("Launch.TFrame", background="#0f233d")
    style.configure("Launch.TLabel", background="#0f233d", foreground="#ffffff")
    style.configure(
        "LaunchTitle.TLabel",
        background="#0f233d",
        foreground="#ffffff",
        font=("Segoe UI", 30, "bold"),
    )
    style.configure("LaunchStatus.TLabel", background="#0f233d", foreground="#afc6e3")
    style.configure(
        "Launch.Horizontal.TProgressbar",
        troughcolor="#243c59",
        background="#60a5fa",
        borderwidth=0,
        thickness=6,
    )
    style.configure("TLabel", background=_COLOR_BG, foreground=_COLOR_TEXT)
    style.configure("Muted.TLabel", foreground=_COLOR_MUTED, background=_COLOR_BG)
    style.configure("Panel.TLabel", background=_COLOR_PANEL, foreground=_COLOR_TEXT)
    style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), background=_COLOR_BG)
    style.configure("Subtitle.TLabel", foreground=_COLOR_MUTED, background=_COLOR_BG)
    style.configure("Status.TLabel", foreground=_COLOR_MUTED, background=_COLOR_PANEL)

    style.configure(
        "TButton",
        padding=(12, 6),
        relief=tk.FLAT,
        borderwidth=1,
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
        background=[
            ("disabled", "#dce4f0"),
            ("pressed", "#1e40af"),
            ("active", _COLOR_PRIMARY_DARK),
        ],
        foreground=[("disabled", "#7b8799")],
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
    loading = ttk.Frame(root, style="Launch.TFrame", padding=40)
    loading.pack(fill=tk.BOTH, expand=True)
    loading.columnconfigure(0, weight=1)
    loading.rowconfigure(0, weight=1)
    loading.rowconfigure(2, weight=1)
    card = ttk.Frame(loading, style="Launch.TFrame", padding=(40, 30))
    card.grid(row=1, column=0)
    mark = tk.Canvas(card, width=88, height=88, bg="#0f233d", highlightthickness=0)
    mark.grid(row=0, column=0, pady=(0, 24))
    mark.create_rectangle(8, 16, 76, 74, outline="#60a5fa", width=3)
    mark.create_rectangle(76, 34, 83, 56, fill="#60a5fa", outline="")
    for x, height in [(20, 17), (35, 29), (50, 41)]:
        mark.create_rectangle(x, 64 - height, x + 9, 64, fill="#93c5fd", outline="")
    ttk.Label(card, text=APP_TITLE, style="LaunchTitle.TLabel").grid(row=1, column=0)
    progress = ttk.Progressbar(
        card,
        mode="determinate",
        maximum=100,
        value=0,
        length=440,
        style="Launch.Horizontal.TProgressbar",
    )
    progress.grid(row=2, column=0, pady=(30, 18))
    status = ttk.Label(
        card, text="Preparing your workspace…", style="LaunchStatus.TLabel"
    )
    status.grid(row=3, column=0)
    return loading, status, progress


def _create_app_shell(
    root: tk.Tk, config_path
) -> tuple[ttk.Frame, ttk.Notebook, ttk.Label]:
    shell = ttk.Frame(root, style="App.TFrame", padding=(14, 12, 14, 10))
    shell.columnconfigure(0, weight=1)
    shell.rowconfigure(0, weight=1)

    notebook = ttk.Notebook(shell)
    notebook.grid(row=0, column=0, sticky=tk.NSEW)

    status_bar = ttk.Frame(shell, style="Panel.TFrame", padding=(10, 5))
    status_bar.grid(row=1, column=0, sticky=tk.EW, pady=(8, 0))
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
        messagebox.showerror(
            "Startup Error", f"Failed to initialize user settings:\n{exc}"
        )
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

    def start_loading() -> None:
        import queue
        import threading

        loading, loading_status, progress = _create_loading_page(root)
        imported = queue.Queue()

        def import_tabs():
            try:
                from gui.analysis_tab import AnalysisTab
                from gui.guide_tab import GuideTab
                from gui.report_tab import ReportTab
                from gui.settings_tab import SettingsTab
                from gui.single_mask_tab import SingleMaskTab
                from gui.workflow_tab import WorkflowTab

                imported.put(
                    (
                        True,
                        (
                            WorkflowTab,
                            SingleMaskTab,
                            AnalysisTab,
                            ReportTab,
                            SettingsTab,
                            GuideTab,
                        ),
                    )
                )
            except Exception as exc:
                imported.put((False, exc))

        def show_error(exc):
            logging.error("Application startup failed: %s", exc)
            loading_status.configure(text=f"Could not open workspace: {exc}")
            messagebox.showerror("Startup Error", str(exc), parent=root)

        def poll_imports():
            try:
                ok, payload = imported.get_nowait()
            except queue.Empty:
                root.after(50, poll_imports)
                return
            if not ok:
                show_error(payload)
                return
            shell, notebook, status_label = _create_app_shell(root, config_path)
            tabs = []

            def reload_runtime_settings():
                for tab in tabs:
                    reload_method = getattr(tab, "reload_settings", None)
                    if reload_method is not None:
                        reload_method()

            titles = (
                "Main",
                "Single Mask Mode",
                "Analysis",
                "Report",
                "Settings",
                "Guide",
            )

            def build_next(index=0):
                if index == len(payload):
                    progress.configure(value=100)
                    loading.destroy()
                    shell.pack(fill=tk.BOTH, expand=True)
                    status_label.configure(text="Ready")
                    return
                loading_status.configure(text=f"Opening {titles[index]}…")
                progress.configure(value=15 + index * 14)

                # Allow the stage to paint before the next constructor starts.
                def construct():
                    try:
                        options = (
                            {"on_settings_saved": reload_runtime_settings}
                            if titles[index] == "Settings"
                            else {}
                        )
                        tab = payload[index](
                            notebook, config_path=config_path, **options
                        )
                        tabs.append(tab)
                        notebook.add(tab, text=titles[index])
                        root.after(20, lambda: build_next(index + 1))
                    except Exception as exc:
                        shell.destroy()
                        show_error(exc)

                root.after(20, construct)

            build_next()

        threading.Thread(target=import_tabs, daemon=True).start()
        root.after(50, poll_imports)

    if workspace_setup_required(config_path):
        from gui.workspace_setup import WorkspaceSetup

        setup = WorkspaceSetup(root, config_path, start_loading)
        setup.pack(fill=tk.BOTH, expand=True)
    else:
        root.after(50, start_loading)

    root.mainloop()


if __name__ == "__main__":
    main()
