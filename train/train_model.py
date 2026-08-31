import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train import enhance_training_features as enhancer
from train import train_classical_model
from train import train_lightGBM

logger = logging.getLogger(__name__)

MODEL_KINDS = (
    "lightgbm",
    "catboost",
    "xgboost",
    "random_forest",
    "logistic_regression",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified enhanced-feature model training. Generates missing enhanced_*.csv "
            "files automatically, then trains the selected model family."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV files or directories. Defaults to data/labeled/.",
    )
    parser.add_argument(
        "--model-kind",
        "--model",
        choices=MODEL_KINDS,
        default=None,
        help="Model family to train. If omitted in a terminal, prompts for a choice.",
    )
    parser.add_argument("--config", default="config/thresholds.yaml")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Existing model folder/name whose source_csvs.txt should be reused. "
            "When set, only CSVs not already in that model are shown for addition."
        ),
    )
    parser.add_argument(
        "--add-csv-indices",
        default=None,
        help=(
            "Comma-separated indices from the add-candidate list. Only used with "
            "--base-model in non-interactive runs."
        ),
    )
    parser.add_argument(
        "--csv-indices",
        default=None,
        help="Comma-separated indices from the full eligible CSV list when no base model is used.",
    )
    parser.add_argument(
        "--image-root",
        action="append",
        default=[],
        help="Image root for auto-generating enhanced CSVs. Can be passed multiple times.",
    )
    parser.add_argument(
        "--overwrite-enhanced",
        action="store_true",
        help="Regenerate existing enhanced_*.csv files before training.",
    )
    parser.add_argument(
        "--test-dir",
        default=str(train_lightGBM.DEFAULT_TEST_DIR),
        help="Directory of test images to mask after training. Defaults to data/test.",
    )
    parser.add_argument(
        "--skip-test-masks",
        action="store_true",
        help="Train only; do not run data/test masking after saving the model.",
    )
    return parser.parse_args()


def prompt_model_kind() -> str:
    print("Model families:")
    for index, model_kind in enumerate(MODEL_KINDS, start=1):
        print(f"  {index:>2}. {model_kind}")
    while True:
        raw = input(f"Select model [1-{len(MODEL_KINDS)}] (default: 1): ").strip()
        if not raw:
            return MODEL_KINDS[0]
        try:
            selected = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= selected <= len(MODEL_KINDS):
            return MODEL_KINDS[selected - 1]
        print("Enter a number from the list.")


def resolve_model_kind(value: str | None) -> str:
    if value:
        normalized = value.strip().lower()
        if normalized in MODEL_KINDS:
            return normalized
        raise RuntimeError(f"Unsupported model kind: {value}")
    if sys.stdin.isatty():
        return prompt_model_kind()
    return "lightgbm"


def training_inputs(inputs: list[str]) -> list[str]:
    return inputs or [str(train_lightGBM.DEFAULT_INPUT_DIR)]


def pending_enhanced_sources(source_csvs: list[Path], *, overwrite: bool) -> list[Path]:
    if overwrite:
        return list(source_csvs)
    return [
        csv_path
        for csv_path in source_csvs
        if not enhancer.enhanced_output_path(csv_path).exists()
    ]


def write_missing_enhanced_csvs(
    source_csvs: list[Path],
    image_roots: list[str],
    *,
    config_path: str | Path,
    overwrite: bool,
) -> list[Path]:
    pending = pending_enhanced_sources(source_csvs, overwrite=overwrite)
    if not pending:
        print("Enhanced CSVs already exist; nothing to generate.", flush=True)
        return []

    action = "Regenerating" if overwrite else "Generating missing"
    print(f"{action} enhanced CSV files ({len(pending)}):", flush=True)
    for csv_path in pending:
        output_path = enhancer.enhanced_output_path(csv_path)
        print(f"  - {train_lightGBM.csv_display_path(output_path)}", flush=True)

    roots = enhancer.build_image_roots(pending, image_roots)
    image_index = enhancer.build_image_index(roots)
    written: list[Path] = []
    for index, csv_path in enumerate(pending, start=1):
        output_path = enhancer.enhanced_output_path(csv_path)
        print(
            f"[{index}/{len(pending)}] Enhancing {train_lightGBM.csv_display_path(csv_path)}",
            flush=True,
        )
        labels = enhancer.load_label_frame([csv_path])
        enhanced = enhancer.build_enhanced_frame(labels, image_index, config_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced.to_csv(output_path, index=False, float_format="%.6f")
        written.append(output_path)
    print(f"Enhanced CSV generation complete: {len(written)} file(s).", flush=True)
    return written


def ensure_enhanced_csvs(
    inputs: list[str],
    image_roots: list[str],
    *,
    config_path: str | Path,
    overwrite: bool,
) -> list[Path]:
    all_csvs = train_lightGBM.collect_all_csv_files(inputs)
    try:
        source_csvs = enhancer.collect_csv_files(inputs)
    except RuntimeError:
        source_csvs = []

    if source_csvs:
        write_missing_enhanced_csvs(
            source_csvs,
            image_roots,
            config_path=config_path,
            overwrite=overwrite,
        )
        all_csvs = train_lightGBM.collect_all_csv_files(inputs)

    enhanced_csvs = train_lightGBM.eligible_csv_files(all_csvs, "enhanced")
    print(f"Enhanced training CSV files ready: {len(enhanced_csvs)}", flush=True)
    return enhanced_csvs


def models_root(config_path: str | Path) -> Path:
    cfg = train_lightGBM.load_config(config_path)
    root, _name = train_lightGBM.models_root_from_config(cfg)
    return root


def model_dirs(config_path: str | Path) -> list[Path]:
    root = models_root(config_path)
    if not root.is_dir():
        return []
    return sorted(
        [path for path in root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def resolve_model_dir(value: str | Path, config_path: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = PROJECT_ROOT / path
    if direct.is_dir():
        return direct
    return models_root(config_path) / path


def prompt_base_model(config_path: str | Path) -> Path | None:
    dirs = model_dirs(config_path)
    if not dirs:
        return None
    print("Existing models for source_csvs reuse:")
    print("   0. none")
    for index, model_dir in enumerate(dirs, start=1):
        source_file = model_dir / "source_csvs.txt"
        suffix = "" if source_file.exists() else " (no source_csvs.txt)"
        print(f"  {index:>2}. {model_dir.name}{suffix}")
    while True:
        raw = input("Reuse source CSVs from existing model? [0=none]: ").strip()
        if not raw:
            return None
        try:
            selected = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if selected == 0:
            return None
        if 1 <= selected <= len(dirs):
            return dirs[selected - 1]
        print("Enter a number from the list.")


def resolve_existing_source_csv(line: str) -> Path:
    text = line.strip()
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path.exists() and not train_lightGBM.is_enhanced_csv(path):
        enhanced_path = enhancer.enhanced_output_path(path).resolve()
        if enhanced_path.exists():
            return enhanced_path
    return path


def read_model_source_csvs(model_dir: Path) -> list[Path]:
    source_path = model_dir / "source_csvs.txt"
    if not source_path.exists():
        return []
    csvs = []
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        csv_path = resolve_existing_source_csv(line)
        if csv_path.exists():
            csvs.append(csv_path)
        else:
            logger.warning("Skipping missing source CSV from %s: %s", source_path, line)
    return sorted({path.resolve() for path in csvs})


def print_csv_choices(title: str, csv_files: list[Path]) -> None:
    print(title)
    if not csv_files:
        print("  (none)")
        return
    for index, csv_path in enumerate(csv_files, start=1):
        print(f"  {index:>2}. {train_lightGBM.csv_display_path(csv_path)}")


def prompt_indices(count: int, *, default_all: bool) -> list[int]:
    if count <= 0:
        return []
    default_text = "all" if default_all else "none"
    while True:
        raw = input(f"Select CSV numbers comma-separated (blank = {default_text}): ").strip()
        if not raw:
            return list(range(1, count + 1)) if default_all else []
        try:
            return train_lightGBM.parse_csv_indices(raw, count)
        except RuntimeError as exc:
            print(str(exc))


def selected_training_csvs(
    enhanced_csvs: list[Path],
    *,
    config_path: str | Path,
    base_model: str | None,
    add_csv_indices: str | None,
    csv_indices: str | None,
) -> list[Path]:
    base_dir = resolve_model_dir(base_model, config_path) if base_model else None
    if base_dir is None and sys.stdin.isatty():
        base_dir = prompt_base_model(config_path)

    if base_dir is not None:
        base_csvs = read_model_source_csvs(base_dir)
        base_set = {path.resolve() for path in base_csvs}
        additions = [path for path in enhanced_csvs if path.resolve() not in base_set]
        print(f"Base model: {base_dir.name}")
        print(f"Existing source CSVs reused: {len(base_csvs)}")
        print_csv_choices("Additional eligible CSV files:", additions)
        if add_csv_indices:
            indices = train_lightGBM.parse_csv_indices(add_csv_indices, len(additions))
        elif sys.stdin.isatty():
            indices = prompt_indices(len(additions), default_all=False)
        else:
            indices = []
        selected_additions = [additions[index - 1] for index in indices]
        selected = sorted({*base_csvs, *selected_additions})
        if not selected:
            raise RuntimeError(f"No existing source CSVs found for base model: {base_dir}")
        return selected

    print_csv_choices("Eligible training CSV files:", enhanced_csvs)
    if csv_indices:
        indices = train_lightGBM.parse_csv_indices(csv_indices, len(enhanced_csvs))
    elif sys.stdin.isatty():
        indices = prompt_indices(len(enhanced_csvs), default_all=True)
    else:
        indices = list(range(1, len(enhanced_csvs) + 1))
    return [enhanced_csvs[index - 1] for index in indices]


def training_namespace(args: argparse.Namespace, csv_files: list[Path]) -> argparse.Namespace:
    return argparse.Namespace(
        inputs=[str(path) for path in csv_files],
        config=args.config,
        feature_set="enhanced",
        csv_selection="all",
        csv_indices=None,
        model_name=args.model_name,
        output_dir=args.output_dir,
        test_dir=args.test_dir,
        skip_test_masks=args.skip_test_masks,
    )


def train_selected_model(model_kind: str, args: argparse.Namespace, csv_files: list[Path]) -> Path:
    train_args = training_namespace(args, csv_files)
    if model_kind == "lightgbm":
        return train_lightGBM.train_model(train_args)
    return train_classical_model.train_model(model_kind, train_args)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    config_path = train_lightGBM.resolve_config_path(args.config)
    model_kind = resolve_model_kind(args.model_kind)
    inputs = training_inputs(args.inputs)
    enhanced_csvs = ensure_enhanced_csvs(
        inputs,
        args.image_root,
        config_path=config_path,
        overwrite=bool(args.overwrite_enhanced),
    )
    csv_files = selected_training_csvs(
        enhanced_csvs,
        config_path=config_path,
        base_model=args.base_model,
        add_csv_indices=args.add_csv_indices,
        csv_indices=args.csv_indices,
    )
    print(f"Training {model_kind} with {len(csv_files)} enhanced CSV file(s).")
    train_selected_model(model_kind, args, csv_files)


if __name__ == "__main__":
    main()
