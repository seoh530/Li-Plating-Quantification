import sys
from pathlib import Path
import tkinter as tk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.settings_tab import SettingsTab


def test_current_root_model_preset_is_highlighted(tmp_path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    config_path.write_text(
        """
app:
  root_path: "T:/Team Room/Current"
  root_model_presets:
    - root_path: "T:/Team Room/Other"
      model_path: "models/other"
ml:
  model_path: "models/lightgbm"
""".strip(),
        encoding="utf-8",
    )

    root = tk.Tk()
    root.withdraw()
    try:
        tab = SettingsTab(root, config_path)
        active_rows = [
            item_id
            for item_id in tab._tree.get_children()
            if "active" in tab._tree.item(item_id, "tags")
        ]

        assert len(active_rows) == 1
        assert tab._tree.item(active_rows[0], "values") == (
            "T:/Team Room/Current",
            "models/lightgbm",
        )
    finally:
        root.destroy()
