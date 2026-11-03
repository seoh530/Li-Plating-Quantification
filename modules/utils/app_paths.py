"""Application paths that work both from source and a PyInstaller bundle."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_DIRECTORY_NAME = "Li-Plating_Quant_System"
DATABASE_FILENAME = "photo_jig_vision.db"


def resource_root() -> Path:
    """Return the read-only root containing bundled application resources."""
    bundle_root = getattr(sys, "_MEIPASS", None)
    if getattr(sys, "frozen", False) and bundle_root:
        return Path(bundle_root)
    return Path(__file__).resolve().parents[2]


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def user_data_dir() -> Path:
    """Return this Windows user's writable, non-roaming application folder."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / APP_DIRECTORY_NAME

    # LOCALAPPDATA is normally present on Windows. Keep a writable fallback
    # for development/test environments and unusual Windows configurations.
    return Path.home() / "AppData" / "Local" / APP_DIRECTORY_NAME


def user_database_path() -> Path:
    return user_data_dir() / DATABASE_FILENAME


def ensure_user_config() -> Path:
    """Seed a persistent user config from the bundled read-only default."""
    source = resource_path("config", "thresholds.yaml")
    target = user_data_dir() / "config" / "thresholds.yaml"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return target
