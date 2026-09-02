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


def test_count_mismatch_returns_warning_not_fail(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_path = tmp_path / "2026-05-12" / "AHG00001"
    front_path = analysis_path / "FRONT"
    front_path.mkdir(parents=True)
    for index in range(2):
        (front_path / f"2026-05-12 - AHG00001 - Cell #1 - FRONT - {index}.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(tmp_path, "2026-05-12", "AHG00001")

    assert result.status == "WARNING"
    assert not result.fail_reasons
    assert result.warning_reasons == [
        "[single] Warning: FRONT Expected 28 electrodes, found 2"
    ]


def test_filename_missing_date_or_analysis_id_fails(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_path = tmp_path / "2026-05-12" / "AHG00001"
    front_path = analysis_path / "FRONT"
    front_path.mkdir(parents=True)
    (front_path / "AHG00001 - Cell #1 - FRONT.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(tmp_path, "2026-05-12", "AHG00001")

    assert result.status == "FAIL"
    assert "invalid DATE" in result.fail_reasons[0]


def test_cell_and_face_filename_parts_are_optional(tmp_path: Path) -> None:
    config_path = tmp_path / "thresholds.yaml"
    _write_config(config_path)

    analysis_path = tmp_path / "2026-05-12" / "AHG00001"
    analysis_path.mkdir(parents=True)
    for index in range(28):
        (analysis_path / f"2026-05-12 - AHG00001 - {index}.jpg").write_bytes(b"")

    result = FolderQC(config_path).validate(tmp_path, "2026-05-12", "AHG00001")

    assert result.status == "PASS"
    assert result.cells[0].front_path == analysis_path
    assert result.cells[0].back_path is None



