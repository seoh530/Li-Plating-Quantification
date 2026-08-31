import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_lightGBM import (
    DEFAULT_TEST_DIR,
    load_config,
    models_root_from_config,
    resolve_config_path,
    save_test_masks,
    validate_model_name,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run saved pixel classifier models on images under data/test."
    )
    parser.add_argument(
        "--config",
        default="config/thresholds.yaml",
        help="Path to thresholds.yaml.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Model folder name under models/. Use commas for multiple models, "
            "for example: model_a,model_b. If omitted, choose from a numbered list."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Explicit model directory. Cannot be used with --model-name.",
    )
    parser.add_argument(
        "--test-dir",
        default=str(DEFAULT_TEST_DIR),
        help="Directory of test images. Defaults to data/test.",
    )
    return parser.parse_args()


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def is_model_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "metadata.json").is_file()
        and ((path / "model.txt").is_file() or (path / "model.joblib").is_file())
    )


def is_lightgbm_model_dir(path: Path) -> bool:
    """Backward-compatible alias for older tests/imports."""
    return is_model_dir(path)


def discover_model_dirs(models_root: Path) -> list[Path]:
    if not models_root.is_dir():
        return []
    return sorted(
        (path for path in models_root.iterdir() if is_model_dir(path)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_model_dir_arg(
    model_dir_arg: str | None,
    model_name: str | None,
    models_root: Path,
) -> Path | None:
    if model_dir_arg:
        model_dir = resolve_project_path(model_dir_arg)
        return model_dir
    if model_name:
        return models_root / validate_model_name(model_name)
    return None


def parse_model_indices(raw: str, candidate_count: int) -> list[int]:
    selected: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            index = int(token)
        except ValueError as exc:
            raise ValueError("Enter numbers from the list, separated by commas.") from exc
        if not 1 <= index <= candidate_count:
            raise ValueError("Enter numbers from the list, separated by commas.")
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("Enter numbers from the list, separated by commas.")
    return selected


def choose_model_dirs(candidates: list[Path]) -> list[Path]:
    if not candidates:
        raise RuntimeError(
            "No trained model folders found. Expected metadata.json plus model.txt or model.joblib."
        )

    print("Model folders found under models/:")
    for index, model_dir in enumerate(candidates, start=1):
        print(f"  {index:>2}. {model_dir.name}")

    while True:
        raw = input(f"Select model(s) [1-{len(candidates)}], example 1,2,3: ").strip()
        try:
            selected_indices = parse_model_indices(raw, len(candidates))
        except ValueError:
            print("Enter numbers from the list, separated by commas.")
            continue
        return [candidates[index - 1] for index in selected_indices]


def choose_model_dir(candidates: list[Path]) -> Path:
    """Backward-compatible single-model selector."""
    return choose_model_dirs(candidates)[0]


def resolve_model_dirs(args: argparse.Namespace, cfg: dict) -> list[Path]:
    models_root, _configured_name = models_root_from_config(cfg)
    if args.model_dir and args.model_name:
        raise RuntimeError("--model-dir and --model-name cannot be used together.")
    if args.model_dir:
        explicit_model_dir = resolve_model_dir_arg(args.model_dir, None, models_root)
        if explicit_model_dir is not None and not is_model_dir(explicit_model_dir):
            raise RuntimeError(
                f"Model folder is missing model.txt or model.joblib or metadata.json: {explicit_model_dir}"
            )
        return [explicit_model_dir]
    if args.model_name:
        model_dirs = [
            models_root / validate_model_name(name)
            for name in args.model_name.split(",")
            if name.strip()
        ]
        if not model_dirs:
            raise RuntimeError("--model-name must include at least one model folder name.")
        for model_dir in model_dirs:
            if not is_model_dir(model_dir):
                raise RuntimeError(
                    f"Model folder is missing model.txt or model.joblib or metadata.json: {model_dir}"
                )
        return model_dirs

    return choose_model_dirs(discover_model_dirs(models_root))


def resolve_model_dir(args: argparse.Namespace, cfg: dict) -> Path:
    """Backward-compatible single-model resolver."""
    return resolve_model_dirs(args, cfg)[0]


def run_model_test(config_path: Path, model_dir: Path, test_dir: Path) -> int:
    saved_masks = save_test_masks(config_path, model_dir, test_dir, skip_existing_masks=True)
    if saved_masks:
        print(
            f"Saved {saved_masks} test mask image(s) and test_results.csv to "
            f"{test_dir / model_dir.name}"
        )
    else:
        print(f"No new test images needed processing under {test_dir} for {model_dir.name}")
    return saved_masks


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    model_dirs = resolve_model_dirs(args, cfg)
    test_dir = resolve_project_path(args.test_dir)

    for model_dir in model_dirs:
        run_model_test(config_path, model_dir, test_dir)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
