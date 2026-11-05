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


def workspace_setup_required(config_path: str | Path) -> bool:
    """Ask each user to choose a workspace once, including upgraded installs."""
    import yaml

    with open(config_path, encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh) or {}
    app = cfg.get("app", {})
    return not (
        app.get("workspace_configured") is True
        and str(app.get("root_path", "")).strip()
    )


def save_workspace_root(config_path: str | Path, folder: str | Path) -> Path:
    """Persist a selected directory and the first-run completion marker together."""
    import yaml
    from modules.utils.image_io import atomic_write_text

    path = Path(folder).expanduser()
    if not path.is_dir():
        raise ValueError("Choose an existing folder for your workspace.")
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8-sig")
    cfg = yaml.safe_load(text) or {}
    app = cfg.setdefault("app", {})
    app["root_path"] = str(path.resolve())
    app["workspace_configured"] = True
    # Change the app section only; keep operational settings and their comments.
    import re

    section = re.search(r"(?m)^app:\s*(?:#.*)?\n", text)
    if section is None:
        raise ValueError("The app section is missing from the settings file.")
    rest = text[section.end() :]
    next_section = re.search(r"(?m)^[^\s#][^\n]*:", rest)
    end = section.end() + (next_section.start() if next_section else len(rest))
    body = text[section.end() : end]
    for key in ("root_path", "workspace_configured"):
        encoded = yaml.safe_dump(
            app[key], allow_unicode=True, default_flow_style=True
        ).strip()
        encoded = encoded.removesuffix("\n...")
        pattern = rf"(?m)^  {key}:.*$"
        replacement = f"  {key}: {encoded}"
        if re.search(pattern, body):
            body = re.sub(pattern, lambda _match: replacement, body)
        else:
            body = body.rstrip() + "\n" + replacement + "\n\n"
    updated = text[: section.end()] + body + text[end:]
    parsed = yaml.safe_load(updated)
    if (
        parsed["app"]["root_path"] != str(path.resolve())
        or parsed["app"]["workspace_configured"] is not True
    ):
        raise ValueError("Could not save the workspace selection.")
    atomic_write_text(config_path, updated)
    return path.resolve()
