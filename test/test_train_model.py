from pathlib import Path

from train import train_model
from train import train_lightGBM


def _enhanced_header() -> str:
    return (
        "filename,x,y,label,"
        + ",".join(train_lightGBM.FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
        + "\n"
    )


def test_selected_training_csvs_reuses_base_and_selects_only_additions(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    models_root = project_root / "models"
    labeled = project_root / "data" / "labeled"
    base_csv = labeled / "enhanced_AHG11111.csv"
    add_first = labeled / "enhanced_BHG11111.csv"
    add_second = labeled / "enhanced_CHG11111.csv"
    model_dir = models_root / "logical_lightgbm_1"
    labeled.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    for path in (base_csv, add_first, add_second):
        path.write_text(_enhanced_header(), encoding="utf-8")
    (model_dir / "source_csvs.txt").write_text(
        "data/labeled/enhanced_AHG11111.csv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_model, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(train_lightGBM, "PROJECT_ROOT", project_root)

    selected = train_model.selected_training_csvs(
        [base_csv.resolve(), add_first.resolve(), add_second.resolve()],
        config_path=project_root / "config" / "thresholds.yaml",
        base_model=str(model_dir),
        add_csv_indices="2",
        csv_indices=None,
    )

    assert selected == sorted({base_csv.resolve(), add_second.resolve()})


def test_ensure_enhanced_csvs_accepts_existing_enhanced_only(tmp_path: Path) -> None:
    csv_path = tmp_path / "enhanced_labels.csv"
    csv_path.write_text(_enhanced_header(), encoding="utf-8")

    assert train_model.ensure_enhanced_csvs(
        [str(csv_path)],
        [],
        config_path="config/thresholds.yaml",
        overwrite=False,
    ) == [csv_path.resolve()]


def test_read_model_source_csvs_prefers_enhanced_partner(tmp_path: Path, monkeypatch) -> None:
    project_root = tmp_path / "project"
    model_dir = project_root / "models" / "lightgbm_1"
    source_dir = project_root / "data" / "labeled"
    basic_csv = source_dir / "AHG11111.csv"
    enhanced_csv = source_dir / "enhanced_AHG11111.csv"
    model_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    basic_csv.write_text("filename,x,y,label\nimg.png,1,2,b\n", encoding="utf-8")
    enhanced_csv.write_text(_enhanced_header(), encoding="utf-8")
    (model_dir / "source_csvs.txt").write_text(
        "data/labeled/AHG11111.csv\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(train_model, "PROJECT_ROOT", project_root)

    assert train_model.read_model_source_csvs(model_dir) == [enhanced_csv.resolve()]



def test_write_missing_enhanced_csvs_lists_only_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    source_dir = tmp_path / "data" / "labeled"
    existing_source = source_dir / "AHG11111.csv"
    missing_source = source_dir / "BHG11111.csv"
    existing_enhanced = source_dir / "enhanced_AHG11111.csv"
    source_dir.mkdir(parents=True)
    existing_source.write_text("filename,x,y,label\nimg_a.png,1,2,b\n", encoding="utf-8")
    missing_source.write_text("filename,x,y,label\nimg_b.png,1,2,b\n", encoding="utf-8")
    existing_enhanced.write_text(_enhanced_header(), encoding="utf-8")
    monkeypatch.setattr(train_model.train_lightGBM, "csv_display_path", lambda path: Path(path).name)
    monkeypatch.setattr(
        train_model.enhancer,
        "build_image_roots",
        lambda _csvs, _roots: [],
    )
    monkeypatch.setattr(train_model.enhancer, "build_image_index", lambda _roots: {})
    monkeypatch.setattr(
        train_model.enhancer,
        "build_enhanced_frame",
        lambda labels, _image_index, _config_path: labels.assign(l_res=1.0).reindex(
            columns=["filename", "x", "y", "label", *train_model.enhancer.OUTPUT_FEATURE_COLUMNS],
            fill_value=1.0,
        ),
    )

    written = train_model.write_missing_enhanced_csvs(
        [existing_source, missing_source],
        [],
        config_path="config/thresholds.yaml",
        overwrite=False,
    )

    output = capsys.readouterr().out
    assert "enhanced_BHG11111.csv" in output
    assert "enhanced_AHG11111.csv" not in output
    assert written == [source_dir / "enhanced_BHG11111.csv"]
