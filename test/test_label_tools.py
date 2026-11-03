from pathlib import Path

import pytest

from tools import label_tools
from tools.label_tools import default_output_csv_for_image_folder, image_label_for_csv


def test_default_output_csv_for_cell_front_folder(tmp_path: Path) -> None:
    image_folder = tmp_path / "data" / "labeled" / "AHG11111" / "Cell #1" / "FRONT"
    image_folder.mkdir(parents=True)

    assert default_output_csv_for_image_folder(image_folder) == (
        tmp_path / "data" / "labeled" / "AHG11111" / "AHG11111_1_F.csv"
    )


def test_default_output_csv_for_direct_back_folder(tmp_path: Path) -> None:
    image_folder = tmp_path / "data" / "labeled" / "AHG11111" / "BACK"
    image_folder.mkdir(parents=True)

    assert default_output_csv_for_image_folder(image_folder) == (
        tmp_path / "data" / "labeled" / "AHG11111" / "AHG11111_B.csv"
    )


def test_default_output_csv_accepts_pass_suffix_folder(tmp_path: Path) -> None:
    image_folder = tmp_path / "data" / "labeled" / "AHG11111" / "Cell #1" / "front_pass"
    image_folder.mkdir(parents=True)

    assert default_output_csv_for_image_folder(image_folder) == (
        tmp_path / "data" / "labeled" / "AHG11111" / "AHG11111_1_F.csv"
    )


def test_image_label_for_csv_is_relative_to_analysis_folder(tmp_path: Path) -> None:
    csv_path = tmp_path / "data" / "labeled" / "AHG11111" / "AHG11111_1_F.csv"
    image_path = tmp_path / "data" / "labeled" / "AHG11111" / "Cell #1" / "FRONT" / "img01.png"
    image_path.parent.mkdir(parents=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"")

    assert image_label_for_csv(image_path, csv_path) == "Cell #1/FRONT/img01.png"


def test_discover_label_folders_finds_front_back_image_folders(tmp_path: Path) -> None:
    front = tmp_path / "data" / "labeled" / "CHG28282" / "Cell #1" / "FRONT"
    back = tmp_path / "data" / "labeled" / "CHG28282" / "Cell #1" / "back_pass"
    empty = tmp_path / "data" / "labeled" / "CHG28282" / "Cell #2" / "FRONT"
    front.mkdir(parents=True)
    back.mkdir(parents=True)
    empty.mkdir(parents=True)
    (front / "img01.png").write_bytes(b"")
    (back / "img01.png").write_bytes(b"")

    assert label_tools.discover_label_folders(tmp_path / "data" / "labeled") == [
        back,
        front,
    ]


def test_choose_label_folder_marks_completed_csv_before_number(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    front = tmp_path / "data" / "labeled" / "CHG28282" / "Cell #1" / "FRONT"
    back = tmp_path / "data" / "labeled" / "CHG28282" / "Cell #1" / "BACK"
    front.mkdir(parents=True)
    back.mkdir(parents=True)
    label_tools.default_output_csv_for_image_folder(front).write_text(
        "filename,x,y,r,g,b,h,s,v,label,l_lab,a_lab,b_lab\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(label_tools, "DEFAULT_LABELED_ROOT", tmp_path / "data" / "labeled")
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert label_tools.choose_label_folder([front, back]) == back

    output = capsys.readouterr().out
    front_label = Path("CHG28282") / "Cell #1" / "FRONT"
    back_label = Path("CHG28282") / "Cell #1" / "BACK"
    assert f"completed  1. {front_label}" in output
    assert f"           2. {back_label}" in output


def test_initialize_output_csv_writes_basic_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"

    label_tools.initialize_output_csv(csv_path, append=False)

    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == ["filename", "x", "y", "label"]


def test_initialize_output_csv_rejects_append_to_non_basic_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,x,y,r,g,b,h,s,v,label,l_lab,a_lab,b_lab\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="basic label format"):
        label_tools.initialize_output_csv(csv_path, append=True)


def test_add_point_records_basic_label_row() -> None:
    label_tools.pending_rows.clear()
    label_tools.click_points.clear()
    label_tools.saved_points.clear()
    stamp_records = []

    label_tools.add_point((10, 20), "img.png", 3, 4, "l", stamp_records)

    assert label_tools.pending_rows == [["img.png", 3, 4, "l"]]
    assert label_tools.click_points == [(3, 4, "l")]
    assert stamp_records[0][3] == ["img.png", 3, 4, "l"]

