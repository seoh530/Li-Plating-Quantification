from pathlib import Path
import yaml

from modules.utils.app_paths import resource_root


def project_root() -> Path:
    """Return the source root or PyInstaller's read-only extraction root."""
    return resource_root()


def read_config_model_path(config_path: str | Path) -> Path:
    with open(config_path, "r", encoding="utf-8-sig") as fh:
        cfg = yaml.safe_load(fh) or {}
    return Path(cfg.get("ml", {}).get("model_path", "models/lightgbm"))


def resolve_model_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def is_model_dir(path: str | Path) -> bool:
    resolved = resolve_model_path(path)
    has_model = (resolved / "model.txt").is_file() or (resolved / "model.joblib").is_file()
    return has_model and (resolved / "metadata.json").is_file()


def display_model_path(path: str | Path) -> str:
    resolved = resolve_model_path(path)
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return str(resolved)


def list_model_options(config_path: str | Path) -> list[str]:
    configured = read_config_model_path(config_path)
    configured_resolved = resolve_model_path(configured)

    candidates: list[str] = []
    if is_model_dir(configured_resolved):
        candidates.append(display_model_path(configured_resolved))

    roots = [configured_resolved.parent, project_root() / "models"]
    seen_roots: set[Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_dir() or not is_model_dir(child):
                continue
            candidates.append(display_model_path(child))

    unique: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
