from pathlib import Path

import yaml

from modules.qc.folder_qc import FolderQC


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "folder_qc": {
                    "analysis_id_pattern": r"^[A-Z]{3}[0-9]{5}$",
                    "date_formats": ["%Y-%m-%d"],
                    "customer_codes": {"H": "HKMC", "T": "TEST"},
                    "image_count_rules": {"H": 28},
                    "cell_folder_pattern": r"^Cell\s*#\d+$",
                    "valid_image_extensions": [".jpg", ".png"],
                }
            }
        ),
        encoding="utf-8",
    )


def test_test_analysis_id_ignores_front_back_requirement(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_path = tmp_path / "2026-05-12" / "TEST01"
    analysis_path.mkdir(parents=True)
    (analysis_path / "sample_01.jpg").write_bytes(b"")
    (analysis_path / "sample_02.jpg").write_bytes(b"")
    (analysis_path / "sample_03.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(tmp_path, "2026-05-12", "TEST01")

    assert result.status == "PASS"
    assert result.customer == "TEST"
    assert len(result.cells) == 1
    assert result.cells[0].front_path == analysis_path
    assert result.cells[0].front_count == 3
    assert result.cells[0].back_path is None


def test_literal_test_folder_supports_legacy_direct_images(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_path = tmp_path / "2026-05-12" / "TEST"
    analysis_path.mkdir(parents=True)
    (analysis_path / "sample_02.png").write_bytes(b"")
    pass_path = analysis_path / "_pass"
    pass_path.mkdir()
    (pass_path / "generated.png").write_bytes(b"")

    result = FolderQC(config_path).validate(tmp_path, "2026-05-12", "TEST")

    assert result.status == "PASS"
    assert result.customer == "TEST"
    assert result.cells[0].front_count == 1


def test_literal_test_folder_validates_each_named_folder(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_path = tmp_path / "2026-05-12" / "TEST"
    source_01 = analysis_path / "01(NEW)"
    source_02 = analysis_path / "02(OLD)"
    source_01.mkdir(parents=True)
    source_02.mkdir()
    nested_source = source_01 / "nested"
    nested_source.mkdir()
    (nested_source / "sample_01.jpg").write_bytes(b"")
    (source_02 / "sample_02.png").write_bytes(b"")
    pass_path = analysis_path / "01(NEW)_pass"
    pass_path.mkdir()
    (pass_path / "generated.png").write_bytes(b"")

    result = FolderQC(config_path).validate(tmp_path, "2026-05-12", "TEST")

    assert result.status == "PASS"
    assert [cell.cell_name for cell in result.cells] == ["01(NEW)", "02(OLD)"]
    assert [cell.front_path for cell in result.cells] == [source_01, source_02]
    assert [cell.front_count for cell in result.cells] == [1, 1]


def test_force_allows_analysis_id_and_count_warnings(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_id = "XH123"
    cell_path = tmp_path / "2026-05-12" / analysis_id
    front_path = cell_path / "FRONT"
    back_path = cell_path / "BACK"
    front_path.mkdir(parents=True)
    back_path.mkdir()
    (front_path / "front_01.jpg").write_bytes(b"")
    (back_path / "back_01.jpg").write_bytes(b"")

    normal = FolderQC(config_path).validate(tmp_path, "2026-05-12", analysis_id)
    forced = FolderQC(config_path).validate(
        tmp_path,
        "2026-05-12",
        analysis_id,
        force=True,
    )

    assert normal.status == "FAIL"
    assert forced.status == "PASS"
    assert forced.customer == "HKMC"
    assert forced.warnings
    assert any("AnalysisID format mismatch" in warning for warning in forced.warnings)
    assert any("count mismatch" in warning for warning in forced.warnings)


def test_single_front_folder_does_not_require_back(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_id = "AHG12345"
    front_path = tmp_path / "2026-05-12" / analysis_id / "FRONT"
    front_path.mkdir(parents=True)
    (front_path / "front_01.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(
        tmp_path,
        "2026-05-12",
        analysis_id,
        force=True,
    )

    assert result.status == "PASS"
    assert result.cells[0].front_path == front_path
    assert result.cells[0].back_path is None


def test_selected_customer_overrides_analysis_id_and_count_rule(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_id = "AFG12345"
    front_path = tmp_path / "2026-05-12" / analysis_id / "FRONT"
    front_path.mkdir(parents=True)
    for index in range(28):
        (front_path / f"front_{index:02d}.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(
        tmp_path,
        "2026-05-12",
        analysis_id,
        customer="HKMC",
    )

    assert result.status == "PASS"
    assert result.customer == "HKMC"


def test_arbitrary_work_folder_with_direct_images(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    work_id = "ets_test_cell_fast"
    work_path = tmp_path / "2026-05-12" / work_id
    work_path.mkdir(parents=True)
    (work_path / "sample_01.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(
        tmp_path,
        "2026-05-12",
        work_id,
        customer="ETC",
    )

    assert result.status == "PASS"
    assert result.customer == "ETC"
    assert result.cells[0].source_path == work_path
    assert result.cells[0].front_path is None


def test_back_only_folder_is_valid_when_count_matches(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    work_id = "back_only_work"
    back_path = tmp_path / "2026-05-12" / work_id / "BACK"
    back_path.mkdir(parents=True)
    for index in range(28):
        (back_path / f"back_{index:02d}.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(
        tmp_path,
        "2026-05-12",
        work_id,
        customer="HKMC",
    )

    assert result.status == "PASS"
    assert result.cells[0].front_path is None
    assert result.cells[0].back_path == back_path



