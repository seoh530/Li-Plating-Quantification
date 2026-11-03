import sqlite3
import sys
from pathlib import Path

from modules.ml.model_registry import resolve_model_path
from modules.storage.sqlite_logger import SQLitePredictionLogger
from modules.utils.app_paths import (
    APP_DIRECTORY_NAME,
    ensure_user_config,
    resource_path,
    user_database_path,
)


def test_frozen_resources_use_meipass(monkeypatch, tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    bundled_config = bundle_root / "config" / "thresholds.yaml"
    bundled_config.parent.mkdir(parents=True)
    bundled_config.write_text("app:\n  root_path: C:/images\n", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    assert resource_path("config", "thresholds.yaml") == bundled_config
    assert resolve_model_path("models/lightgbm") == bundle_root / "models" / "lightgbm"

    user_config = ensure_user_config()
    assert user_config == tmp_path / "local" / APP_DIRECTORY_NAME / "config" / "thresholds.yaml"
    assert user_config.read_text(encoding="utf-8") == bundled_config.read_text(encoding="utf-8")


def test_user_config_is_not_overwritten(monkeypatch, tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    bundled_config = bundle_root / "config" / "thresholds.yaml"
    bundled_config.parent.mkdir(parents=True)
    bundled_config.write_text("bundled", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle_root), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    user_config = ensure_user_config()
    user_config.write_text("user settings", encoding="utf-8")

    assert ensure_user_config().read_text(encoding="utf-8") == "user settings"


def test_sqlite_is_created_in_per_user_localappdata(monkeypatch, tmp_path: Path) -> None:
    local_app_data = tmp_path / "local-user"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    logger = SQLitePredictionLogger(tmp_path / "unused-config.yaml")
    expected = local_app_data / APP_DIRECTORY_NAME / "photo_jig_vision.db"

    assert user_database_path() == expected
    assert logger.db_path == expected
    assert expected.is_file()
    with sqlite3.connect(expected) as connection:
        table = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'prediction_runs'"
        ).fetchone()
    assert table == ("prediction_runs",)
