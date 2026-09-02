from pathlib import Path

from train import enhance_training_features
from train.enhance_training_features import collect_csv_files, enhanced_output_path


def test_collect_csv_files_skips_generated_enhanced_csv(tmp_path: Path) -> None:
    source = tmp_path / "labels.csv"
    enhanced = tmp_path / "enhanced_labels.csv"
    source.write_text("filename,x,y,label\nimg.png,1,2,b\n", encoding="utf-8")
    enhanced.write_text("filename,x,y,label\nimg.png,1,2,b\n", encoding="utf-8")

    assert collect_csv_files([str(tmp_path)]) == [source.resolve()]
    assert enhanced_output_path(source) == enhanced


def test_resolve_project_path_falls_back_to_project_root(
    tmp_path: Path, monkeypatch
) -> None:
    project_root = tmp_path / "project"
    tools_dir = project_root / "tools"
    config_path = project_root / "config" / "thresholds.yaml"
    tools_dir.mkdir(parents=True)
    config_path.parent.mkdir()
    config_path.write_text("ml: {}\n", encoding="utf-8")
    monkeypatch.setattr(enhance_training_features, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(tools_dir)

    assert enhance_training_features.resolve_project_path("config/thresholds.yaml") == config_path
