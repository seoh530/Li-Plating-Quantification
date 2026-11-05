import re
from hashlib import sha1
from pathlib import Path
from typing import Iterable


_TEST_ANALYSIS_ID_RE = re.compile(r"^TEST(?:\d{2})?$", re.IGNORECASE)
_TEST_OUTPUT_DIRS = {"pass", "mask", "_pass", "_mask", "_cropped_temp"}
_TEST_OUTPUT_SUFFIXES = ("_pass", "_mask", "_cropped_temp")


def is_test_analysis_id(value: str) -> bool:
    return bool(_TEST_ANALYSIS_ID_RE.fullmatch(value.strip()))


def find_test_source_folders(analysis_path: str | Path) -> list[tuple[int, Path]]:
    """Return TEST source folders as (cell number, path) pairs."""
    analysis_path = Path(analysis_path)
    if analysis_path.name.upper() == "TEST" and analysis_path.is_dir():
        cell_dirs = sorted(
            (
                path
                for path in analysis_path.iterdir()
                if path.is_dir()
                and path.name.lower() not in _TEST_OUTPUT_DIRS
                and not path.name.lower().endswith(_TEST_OUTPUT_SUFFIXES)
            ),
            key=lambda path: path.name.lower(),
        )
        if cell_dirs:
            return list(enumerate(cell_dirs, start=1))
    return [(1, analysis_path)]


def get_test_output_folders(
    analysis_path: str | Path,
    source_folder: str | Path,
) -> tuple[Path, Path]:
    """Return the PASS and MASK folders for one TEST source folder."""
    analysis_path = Path(analysis_path)
    source_folder = Path(source_folder)
    if source_folder == analysis_path:
        return analysis_path / "_pass", analysis_path / "_mask"
    return (
        analysis_path / f"{source_folder.name}_pass",
        analysis_path / f"{source_folder.name}_mask",
    )


def get_test_mask_folder_for_pass(pass_folder: str | Path) -> Path:
    pass_folder = Path(pass_folder)
    if pass_folder.name.lower() == "_pass":
        return pass_folder.with_name("_mask")
    if pass_folder.name.lower().endswith("_pass"):
        return pass_folder.with_name(f"{pass_folder.name[:-5]}_mask")
    return pass_folder.with_name(f"{pass_folder.name}_mask")


def ensure_test_output_folders(analysis_path: str | Path) -> None:
    """Create PASS and MASK folders for every TEST source folder."""
    analysis_path = Path(analysis_path)
    for _, source_folder in find_test_source_folders(analysis_path):
        pass_folder, mask_folder = get_test_output_folders(
            analysis_path,
            source_folder,
        )
        pass_folder.mkdir(parents=True, exist_ok=True)
        mask_folder.mkdir(parents=True, exist_ok=True)


def find_test_images(
    analysis_path: str | Path,
    extensions: Iterable[str],
) -> list[Path]:
    """Return source images recursively, excluding pipeline output folders."""
    analysis_path = Path(analysis_path)
    valid_extensions = {extension.lower() for extension in extensions}
    images: list[Path] = []
    for path in analysis_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in valid_extensions:
            continue
        relative_parts = path.relative_to(analysis_path).parts[:-1]
        if any(part.lower() in _TEST_OUTPUT_DIRS for part in relative_parts):
            continue
        images.append(path)
    return sorted(
        images,
        key=lambda path: path.relative_to(analysis_path).as_posix().lower(),
    )


def build_test_output_name(analysis_path: str | Path, image_path: str | Path) -> str:
    """Build a stable flat PNG name without collisions across nested folders."""
    analysis_path = Path(analysis_path)
    image_path = Path(image_path)
    relative_path = image_path.relative_to(analysis_path)
    if len(relative_path.parts) == 1:
        return f"{image_path.stem}.png"
    digest = sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{image_path.stem}__{digest}.png"




def compile_cell_pattern(value: str):
    """Accept legacy YAML values that double-escaped regex whitespace/digits."""
    normalized=str(value).replace(r'\\s',r'\s').replace(r'\\d',r'\d')
    return re.compile(normalized,re.IGNORECASE)
