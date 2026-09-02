import sys
from pathlib import Path
import tkinter as tk

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.analysis_tab import AnalysisTab


def _write_config(path: Path, root_path: Path) -> None:
    path.write_text(
        f"""
app:
  root_path: "{root_path.as_posix()}"
folder_qc:
  analysis_id_pattern: '^AHG\\d+$'
""".strip(),
        encoding="utf-8",
    )


def _create_tab(config_path: Path) -> tuple[tk.Tk, AnalysisTab]:
    root = tk.Tk()
    root.withdraw()
    return root, AnalysisTab(root, config_path)


def _leaf_items(tab: AnalysisTab) -> list[str]:
    leaves: list[str] = []
    for date_id in tab._result_tree.get_children():
        for option_id in tab._result_tree.get_children(date_id):
            leaves.extend(tab._result_tree.get_children(option_id))
    return leaves


def test_analysis_root_load_shows_completed_ml_result_rows(tmp_path) -> None:
    data_root = tmp_path / "data"
    pass_dir = data_root / "BatchA" / "FRONT_PASS"
    mask_dir = data_root / "BatchA" / "FRONT_MASK"
    pass_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (pass_dir / "image01.png").write_bytes(b"not an image")
    (mask_dir / "image01_mask.png").write_bytes(b"not an image")
    (mask_dir / "image01_LI_SALT_12.35%.csv").write_text(
        "\n".join(
            [
                "source_image,crop_image,mask_image,AnalysisID,Cell,Face,ImageNum,nonactive_ratio,li_salt_ratio",
                f",{(pass_dir / 'image01.png').as_posix()},{(mask_dir / 'image01_mask.png').as_posix()},AHG00001,1,FRONT,1,0.2,0.42",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path, data_root)

    root, tab = _create_tab(config_path)
    try:
        tab._root_path_var.set(str(data_root))
        tab._load_root_path()

        groups = tab._result_tree.get_children()
        assert len(groups) == 1
        assert tab._result_tree.item(groups[0], "text") == "No date"
        option_groups = tab._result_tree.get_children(groups[0])
        assert len(option_groups) == 1
        assert tab._result_tree.item(option_groups[0], "text") == "BatchA"
        items = tab._result_tree.get_children(option_groups[0])
        assert len(items) == 1
        assert tab._result_tree.item(items[0], "text") == r"BatchA\FRONT_PASS\image01.png"
        values = tab._result_tree.item(items[0], "values")
        assert values[0] == "FRONT"
        assert values[1] == "42.00"
        assert "_MASK" not in tab._result_tree.item(items[0], "text")
    finally:
        root.destroy()


def test_analysis_li_salt_sort_uses_numeric_values_from_csv_and_filename(tmp_path) -> None:
    data_root = tmp_path / "data"
    mask_dir = data_root / "BACK_MASK"
    mask_dir.mkdir(parents=True)
    (mask_dir / "low_LI_SALT_80.00%.csv").write_text("Face\nBACK\n", encoding="utf-8")
    (mask_dir / "high_LI_SALT_20.00%.csv").write_text(
        "Face,li_salt_ratio\nBACK,0.9\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path, data_root)

    root, tab = _create_tab(config_path)
    try:
        tab._root_path_var.set(str(data_root))
        tab._load_root_path()
        tab._sort_results("li_salt")

        values = []
        for item_id in _leaf_items(tab):
            values.append(float(tab._result_tree.item(item_id, "values")[1]))
        assert values == [80.0, 90.0]
    finally:
        root.destroy()


def test_analysis_result_click_immediately_selects_point(tmp_path) -> None:
    data_root = tmp_path / "data"
    pass_dir = data_root / "AHG00001" / "Cell #1" / "FRONT_PASS"
    mask_dir = data_root / "AHG00001" / "Cell #1" / "FRONT_MASK"
    pass_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    (pass_dir / "layer.png").write_bytes(b"not an image")
    (mask_dir / "layer_mask.png").write_bytes(b"not an image")
    (data_root / "AHG00001" / "AHG00001_result.csv").write_text(
        "Cell,Face,ImageNum,li_salt_ratio,nonactive_ratio\n1,FRONT,1,0.15,0.2\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path, data_root)

    root, tab = _create_tab(config_path)
    try:
        tab._root_path_var.set(str(data_root))
        tab._load_root_path()
        item_id = _leaf_items(tab)[0]

        tab._select_result_item(item_id)

        assert tab._active_result_item == item_id
        assert len(tab._all_series_points) == 1
        assert tab._shown_point is not None
        assert tab._shown_point.percent == 15.0
    finally:
        root.destroy()


def test_analysis_groups_results_by_date_and_selects_work_folder(tmp_path) -> None:
    data_root = tmp_path / "data"
    work_dir = data_root / "2026-08-08" / "ets_test_cell_fast" / "issue"
    pass_dir = work_dir / "_PASS"
    mask_dir = work_dir / "_MASK"
    pass_dir.mkdir(parents=True)
    mask_dir.mkdir()
    pass_image = pass_dir / "image01.png"
    mask_image = mask_dir / "image01_mask.png"
    pass_image.write_bytes(b"not an image")
    mask = np.zeros((2, 2, 4), dtype=np.uint8)
    mask[:, :, :] = [0, 255, 0, 255]
    mask[0, 0] = [0, 0, 255, 255]
    assert cv2.imwrite(str(mask_image), mask)
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path, data_root)

    root, tab = _create_tab(config_path)
    try:
        tab._root_path_var.set(str(data_root))
        tab._load_root_path()

        date_id = tab._result_tree.get_children()[0]
        assert tab._result_tree.item(date_id, "text") == "2026-08-08"
        option_id = tab._result_tree.get_children(date_id)[0]
        assert tab._result_tree.item(option_id, "text") == "ets_test_cell_fast/issue"

        tab._select_result_option(option_id)

        assert tab._active_result_item == option_id
        assert len(tab._all_series_points) == 1
        assert tab._shown_point is not None
        assert tab._shown_point.percent == 25.0
    finally:
        root.destroy()
