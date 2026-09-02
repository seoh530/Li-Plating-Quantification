from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from train import train_lightGBM
from train import training_utils


def test_feature_set_normalization_accepts_basic_and_enhanced() -> None:
    assert train_lightGBM.normalize_feature_set("basic") == "basic"
    assert train_lightGBM.normalize_feature_set("enhanced") == "enhanced"


def test_resolve_config_path_falls_back_to_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    tools_dir = project_root / "tools"
    config_path = project_root / "config" / "thresholds.yaml"
    tools_dir.mkdir(parents=True)
    config_path.parent.mkdir()
    config_path.write_text("ml: {}\n", encoding="utf-8")
    monkeypatch.setattr(train_lightGBM, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tools_dir)

    assert train_lightGBM.resolve_config_path("config/thresholds.yaml") == config_path


def test_resolve_model_dir_uses_model_name_under_models(tmp_path: Path) -> None:
    cfg = {"ml": {"model_path": str(tmp_path / "models" / "lightgbm")}}

    assert train_lightGBM.resolve_model_dir(
        cfg,
        output_dir=None,
        feature_set="basic",
        model_name="my_model",
    ) == tmp_path / "models" / "my_model"


def test_resolve_model_dir_rejects_path_like_model_name(tmp_path: Path) -> None:
    cfg = {"ml": {"model_path": str(tmp_path / "models" / "lightgbm")}}

    with pytest.raises(RuntimeError):
        train_lightGBM.resolve_model_dir(
            cfg,
            output_dir=None,
            feature_set="basic",
            model_name="bad/name",
        )


def test_collect_csv_files_filters_for_feature_set(tmp_path: Path) -> None:
    source = tmp_path / "labels.csv"
    enhanced = tmp_path / "enhanced_labels.csv"
    source.write_text("filename,x,y,label\nimg.png,1,2,b\n", encoding="utf-8")
    enhanced.write_text(
        "filename,x,y,label,"
        + ",".join(train_lightGBM.FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
        + "\n",
        encoding="utf-8",
    )

    assert train_lightGBM.collect_csv_files([str(tmp_path)], "basic") == [source.resolve()]
    assert train_lightGBM.collect_csv_files([str(tmp_path)], "enhanced") == [enhanced.resolve()]


def test_collect_csv_files_includes_analysis_root_label_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "data" / "labeled" / "AHG11111" / "AHG11111.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("filename,x,y,label\nimg.png,1,2,b\n", encoding="utf-8")

    assert train_lightGBM.collect_csv_files([str(tmp_path / "data" / "labeled")], "basic") == [
        csv_path.resolve()
    ]


def test_collect_csv_files_resolves_relative_input_from_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    tools_dir = project_root / "tools"
    csv_path = project_root / "data" / "labeled" / "AHG11111" / "AHG11111.csv"
    tools_dir.mkdir(parents=True)
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("filename,x,y,label\nimg.png,1,2,b\n", encoding="utf-8")
    monkeypatch.setattr(train_lightGBM, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tools_dir)

    assert train_lightGBM.collect_csv_files(["data/labeled"], "basic") == [
        csv_path.resolve()
    ]


def test_collect_csv_files_enhanced_rejects_unprefixed_enhanced_csv(tmp_path: Path) -> None:
    source = tmp_path / "labels.csv"
    source.write_text(
        "filename,x,y,label,"
        + ",".join(train_lightGBM.FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="No enhanced CSV files found"):
        train_lightGBM.collect_csv_files([str(tmp_path)], "enhanced")


def test_collect_csv_files_enhanced_rejects_legacy_only_csv(tmp_path: Path) -> None:
    source = tmp_path / "labels.csv"
    source.write_text("filename,x,y,label,r,g,b,h,s,v,l_lab,a_lab,b_lab\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="No enhanced CSV files found"):
        train_lightGBM.collect_csv_files([str(tmp_path)], "enhanced")


def test_csv_skip_reason_reports_enhanced_filter_reasons(tmp_path: Path) -> None:
    legacy = tmp_path / "labels.csv"
    enhanced = tmp_path / "enhanced_labels.csv"
    legacy.write_text("filename,x,y,label\n", encoding="utf-8")
    enhanced.write_text("filename,x,y,label\n", encoding="utf-8")

    assert train_lightGBM.csv_skip_reason(
        legacy,
        "enhanced",
    ) == "filename does not start with enhanced_"
    assert train_lightGBM.csv_skip_reason(
        enhanced,
        "enhanced",
    ) == "missing enhanced columns"


def test_print_csv_filter_summary_reports_skipped_counts(tmp_path: Path, capsys) -> None:
    legacy = tmp_path / "labels.csv"
    enhanced = tmp_path / "enhanced_labels.csv"
    legacy.write_text("filename,x,y,label\n", encoding="utf-8")
    enhanced.write_text(
        "filename,x,y,label,"
        + ",".join(train_lightGBM.FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
        + "\n",
        encoding="utf-8",
    )

    train_lightGBM.print_csv_filter_summary([legacy, enhanced], [enhanced], "enhanced")

    output = capsys.readouterr().out
    assert "found 2 CSV file(s)" in output
    assert "eligible for enhanced: 1" in output
    assert "filename does not start with enhanced_" in output


def test_parse_csv_indices_accepts_comma_separated_unique_indices() -> None:
    assert train_lightGBM.parse_csv_indices("1, 2,2,4", 5) == [1, 2, 4]


def test_parse_csv_indices_rejects_out_of_range() -> None:
    with pytest.raises(RuntimeError, match="out of range"):
        train_lightGBM.parse_csv_indices("1,4", 3)


def test_select_training_csv_files_custom_uses_requested_indices(tmp_path: Path) -> None:
    csv_files = [tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv"]

    assert train_lightGBM.select_training_csv_files(
        csv_files,
        "custom",
        "3,1",
    ) == [csv_files[2], csv_files[0]]


def test_select_training_csv_files_all_uses_every_eligible_csv(tmp_path: Path) -> None:
    csv_files = [tmp_path / "a.csv", tmp_path / "b.csv"]

    assert train_lightGBM.select_training_csv_files(csv_files, "all") == csv_files


def test_select_feature_columns_basic_uses_color_only() -> None:
    import pandas as pd

    data = pd.DataFrame(columns=[*train_lightGBM.COLOR_FEATURE_COLUMNS, "x_norm", "y_norm"])

    assert train_lightGBM.select_feature_columns(data, "basic") == list(
        train_lightGBM.COLOR_FEATURE_COLUMNS
    )


def test_load_training_frame_accepts_file_name_and_builds_source_image_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "THG29302_1_B.csv"
    csv_path.write_text(
        "file_name,x,y,label,r,g,b,h,s,v,l_lab,a_lab,b_lab\n"
        "CELL#1/BACK_pass/IMG_0002.png,1,2,b,1,2,3,4,5,6,7,8,9\n",
        encoding="utf-8",
    )

    data = train_lightGBM.load_training_frame([csv_path])

    assert data.loc[0, "filename"] == "CELL#1/BACK_pass/IMG_0002.png"
    assert data.loc[0, "source_image_id"] == "THG29302_1_B__IMG_0002.png"
    assert data.loc[0, "cell_group_id"] == "THG29302_1"


def test_source_image_id_strips_enhanced_prefix() -> None:
    assert training_utils.source_image_id(
        "enhanced_THG29302_1_F.csv",
        "CELL#1/FRONT/IMG_0002.png",
    ) == "THG29302_1_F__IMG_0002.png"


def test_cell_group_id_keeps_front_back_together() -> None:
    assert training_utils.cell_group_id("THG29302_1_F.csv") == "THG29302_1"
    assert training_utils.cell_group_id("THG29302_1_BACK.csv") == "THG29302_1"


def test_cell_group_split_keeps_cells_and_source_images_exclusive() -> None:
    data = pd.DataFrame(
        {
            "cell_group_id": ["cell1", "cell1", "cell2", "cell2", "cell3", "cell3", "cell4", "cell4"],
            "source_image_id": ["img1", "img1", "img2", "img2", "img3", "img3", "img4", "img4"],
            "label": ["b", "l", "n", "b", "l", "n", "b", "n"],
        }
    )

    split = training_utils.build_cell_group_split(
        data,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=0,
    )

    assert training_utils.group_overlap_counts(data["cell_group_id"], split) == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }
    assert training_utils.group_overlap_counts(data["source_image_id"], split) == {
        "train_validation": 0,
        "train_test": 0,
        "validation_test": 0,
    }


def test_collect_test_images_excludes_masks_and_model_output(tmp_path: Path) -> None:
    test_dir = tmp_path / "data" / "test"
    output_dir = test_dir / "my_model"
    test_dir.mkdir(parents=True)
    output_dir.mkdir()
    image = test_dir / "sample.png"
    mask = test_dir / "sample_mask.png"
    generated = output_dir / "generated.png"
    image.write_bytes(b"")
    mask.write_bytes(b"")
    generated.write_bytes(b"")

    assert train_lightGBM.collect_test_images(test_dir, output_dir) == [image]


def test_collect_test_images_skips_existing_model_masks_when_requested(tmp_path: Path) -> None:
    test_dir = tmp_path / "data" / "test"
    output_dir = test_dir / "my_model"
    output_dir.mkdir(parents=True)
    existing_image = test_dir / "existing.png"
    new_image = test_dir / "new.png"
    existing_image.write_bytes(b"")
    new_image.write_bytes(b"")
    (output_dir / "existing_mask.png").write_bytes(b"mask")

    assert train_lightGBM.collect_test_images(
        test_dir,
        output_dir,
        skip_existing_masks=True,
    ) == [new_image]


def test_overwrite_test_output_dir_recreates_existing_model_folder(tmp_path: Path) -> None:
    test_dir = tmp_path / "data" / "test"
    model_dir = tmp_path / "models" / "model_a"
    output_dir = test_dir / "model_a"
    output_dir.mkdir(parents=True)
    (output_dir / "old_mask.png").write_bytes(b"old")
    model_dir.mkdir(parents=True)

    assert train_lightGBM.overwrite_test_output_dir(test_dir, model_dir) == output_dir
    assert output_dir.exists()
    assert list(output_dir.iterdir()) == []


def test_write_test_results_appends_and_keeps_existing_rows(tmp_path: Path) -> None:
    output_dir = tmp_path / "data" / "test" / "model_a"
    train_lightGBM.write_test_results(
        output_dir,
        [
            {"filename": "old.png", "li_salt_percent": 1.0},
            {"filename": "replace.png", "li_salt_percent": 2.0},
        ],
    )

    train_lightGBM.write_test_results(
        output_dir,
        [
            {"filename": "new.png", "li_salt_percent": 3.0},
            {"filename": "replace.png", "li_salt_percent": 4.0},
        ],
        append=True,
    )

    result = pd.read_csv(output_dir / "test_results.csv")

    assert result["filename"].tolist() == ["old.png", "new.png", "replace.png"]
    assert result["li_salt_percent"].tolist() == [1.0, 3.0, 4.0]


def test_test_result_row_uses_raw_pixel_counts_for_percent(tmp_path: Path) -> None:
    test_dir = tmp_path / "data" / "test"
    image_path = test_dir / "sample.png"
    mask_path = test_dir / "model_a" / "sample_mask.png"
    test_dir.mkdir(parents=True)
    image_path.write_bytes(b"")
    label_mask = np.array([[1, 0], [2, 3]], dtype=np.int16)

    row = train_lightGBM.test_result_row(
        label_mask,
        image_path,
        test_dir,
        mask_path,
    )

    assert row["filename"] == "sample.png"
    assert row["mask_file"] == "model_a/sample_mask.png"
    assert row["background_pixel"] == 1
    assert row["nonactive_pixel"] == 1
    assert row["li_salt_pixel"] == 1
    assert row["edge_pixel"] == 1
    assert row["li_salt_percent"] == 25.0
    assert row["edge_percent"] == 25.0


def test_write_test_results_saves_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "data" / "test" / "model_a"
    output_dir.mkdir(parents=True)

    output_path = train_lightGBM.write_test_results(
        output_dir,
        [{"filename": "a.png", "li_salt_percent": 12.5}],
    )

    assert output_path == output_dir / "test_results.csv"
    assert "li_salt_percent" in output_path.read_text(encoding="utf-8")


def test_format_source_csvs_writes_one_path_per_line(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "nested" / "b.csv"

    assert train_lightGBM.format_source_csvs([first, second]) == (
        f"{first}\n{second}\n"
    )
