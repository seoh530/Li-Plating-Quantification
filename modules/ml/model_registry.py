from pathlib import Path
import yaml

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_config_model_path(config_path: str | Path) -> Path:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return Path(cfg.get("ml", {}).get("model_path", "models/lightgbm"))


def resolve_model_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def display_model_path(path: str | Path) -> str:
    resolved = resolve_model_path(path)
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return str(resolved)


def list_model_options(config_path: str | Path) -> list[str]:
    configured = read_config_model_path(config_path)
    configured_display = display_model_path(configured)
    candidates: list[str] = [configured_display]

    configured_resolved = resolve_model_path(configured)
    roots = [configured_resolved.parent, project_root() / "models"]
    seen_roots: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen_roots or not root.is_dir():
            continue
        seen_roots.add(root)
        for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
            if not child.is_dir():
                continue
            has_model = (child / "model.txt").exists() or (child / "model.joblib").exists()
            if not has_model or not (child / "metadata.json").exists():
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
