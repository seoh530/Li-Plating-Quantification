import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import LazyTab


def test_lazy_tab_factory_runs_only_when_selected_content_is_loaded() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        notebook = ttk.Notebook(root)
        calls = []

        def factory(parent):
            calls.append(parent)
            return ttk.Frame(parent)

        tab = LazyTab(notebook, "Analysis", factory)
        notebook.add(tab, text="Analysis")
        root.update()

        assert calls == []
        assert tab.content is None

        content = tab.ensure_loaded()

        assert calls == [tab]
        assert content is tab.content
    finally:
        root.destroy()
