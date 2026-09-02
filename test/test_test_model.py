from argparse import Namespace
from pathlib import Path

import pytest

from tools import test_model


def make_model_dir(root: Path, name: str) -> Path:
    model_dir = root / name
    model_dir.mkdir(parents=True)
    (model_dir / "model.txt").write_text("model\n", encoding="utf-8")
    (model_dir / "metadata.json").write_text("{}\n", encoding="utf-8")
    return model_dir


def test_discover_model_dirs_lists_only_lightgbm_models(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    valid = make_model_dir(models_root, "valid_model")
    joblib_model = models_root / "joblib_model"
    joblib_model.mkdir()
    (joblib_model / "model.joblib").write_bytes(b"model")
    (joblib_model / "metadata.json").write_text("{}\n", encoding="utf-8")
    invalid = models_root / "missing_metadata"
    invalid.mkdir()
    (invalid / "model.txt").write_text("model\n", encoding="utf-8")

    assert set(test_model.discover_model_dirs(models_root)) == {valid, joblib_model}


def test_choose_model_dir_uses_numbered_selection(tmp_path: Path, monkeypatch) -> None:
    first = make_model_dir(tmp_path / "models", "first")
    second = make_model_dir(tmp_path / "models", "second")
    monkeypatch.setattr("builtins.input", lambda _prompt: "2")

    assert test_model.choose_model_dir([first, second]) == second


def test_choose_model_dirs_accepts_comma_separated_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = make_model_dir(tmp_path / "models", "first")
    second = make_model_dir(tmp_path / "models", "second")
    third = make_model_dir(tmp_path / "models", "third")
    monkeypatch.setattr("builtins.input", lambda _prompt: "1,3")

    assert test_model.choose_model_dirs([first, second, third]) == [first, third]


def test_parse_model_indices_deduplicates_and_keeps_order() -> None:
    assert test_model.parse_model_indices("2, 1,2", 3) == [2, 1]


def test_resolve_model_dir_accepts_model_name(tmp_path: Path) -> None:
    model_dir = make_model_dir(tmp_path / "models", "named_model")
    args = Namespace(model_dir=None, model_name="named_model")
    cfg = {"ml": {"model_path": str(tmp_path / "models" / "lightgbm")}}

    assert test_model.resolve_model_dir(args, cfg) == model_dir


def test_resolve_model_dirs_accepts_comma_separated_model_names(tmp_path: Path) -> None:
    first = make_model_dir(tmp_path / "models", "first")
    second = make_model_dir(tmp_path / "models", "second")
    args = Namespace(model_dir=None, model_name="first,second")
    cfg = {"ml": {"model_path": str(tmp_path / "models" / "lightgbm")}}

    assert test_model.resolve_model_dirs(args, cfg) == [first, second]


def test_resolve_model_dir_rejects_invalid_model_name(tmp_path: Path) -> None:
    args = Namespace(model_dir=None, model_name="bad_model")
    cfg = {"ml": {"model_path": str(tmp_path / "models" / "lightgbm")}}

    with pytest.raises(RuntimeError, match="missing model.txt"):
        test_model.resolve_model_dir(args, cfg)


def test_resolve_model_dir_rejects_path_like_model_name(tmp_path: Path) -> None:
    args = Namespace(model_dir=None, model_name="../bad")
    cfg = {"ml": {"model_path": str(tmp_path / "models" / "lightgbm")}}

    with pytest.raises(RuntimeError, match="folder name only"):
        test_model.resolve_model_dir(args, cfg)
