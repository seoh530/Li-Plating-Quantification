import base64
import logging
import os
import shutil
import subprocess
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DIR_TIMEOUT_SECONDS = 20
DEBUG = os.environ.get("QCML_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
logger = logging.getLogger(__name__)
_DIR_FALLBACK_FAILURES: dict[tuple[str, str], object] = {}
_FOLDER_FALLBACK_FAILURES: dict[str, OSError] = {}
_COPY_FALLBACK_FAILURES: dict[tuple[str, str, str], str] = {}


@dataclass(frozen=True)
class CopyResult:
    success: bool
    method: str
    error: str = ""


@dataclass(frozen=True)
class DirFallbackResult:
    files: list[Path]
    command: list[str]
    returncode: int | None = None
    stderr: str = ""
    reason: str = ""


@dataclass(frozen=True)
class ScanResult:
    original_path: str
    cleaned_path: str
    files: list[Path]
    total_file_count: int
    image_file_count: int
    excluded_output_count: int
    dir_fallback: DirFallbackResult | None = None

    def summary(self) -> str:
        lines = [
            f"Original path: {self.original_path}",
            f"Cleaned path: {self.cleaned_path}",
            f"Total files scanned: {self.total_file_count}",
            f"Image files found: {self.image_file_count}",
            f"Excluded by output-folder rule: {self.excluded_output_count}",
        ]
        if self.dir_fallback is not None:
            command_text = " ".join(self.dir_fallback.command)
            lines.extend(
                [
                    f"dir fallback reason: {self.dir_fallback.reason or '-'}",
                    f"dir command: {command_text}",
                    f"dir return code: {self.dir_fallback.returncode}",
                    f"dir stderr: {self.dir_fallback.stderr or '-'}",
                ]
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class FolderEntry:
    path: Path
    is_dir: bool
    is_file: bool


class FolderListingCache:
    """Session cache for shallow GUI folder navigation."""

    def __init__(self) -> None:
        self._children: dict[str, list[FolderEntry]] = {}
        self._subfolders: dict[str, list[Path]] = {}
        self._errors: dict[tuple[str, str], OSError] = {}

    def clear(self) -> None:
        self._children.clear()
        self._subfolders.clear()
        self._errors.clear()

    def clear_path(self, path_value: str | Path) -> None:
        key = _normalized_prefix(resolve_path(path_value))
        self._children.pop(key, None)
        self._subfolders.pop(key, None)
        self._errors.pop(("children", key), None)
        self._errors.pop(("subfolders", key), None)

    def children(
        self,
        path_value: str | Path,
        *,
        allow_fallback: bool = False,
    ) -> list[FolderEntry]:
        path = resolve_path(path_value)
        key = _normalized_prefix(path)
        cached = self._children.get(key)
        if cached is not None:
            return list(cached)
        error = self._errors.get(("children", key))
        if error is not None:
            raise error
        try:
            children = list_folder_children(path, allow_fallback=allow_fallback)
        except OSError as exc:
            self._errors[("children", key)] = exc
            raise
        self._children[key] = children
        return list(children)

    def subfolders(self, path_value: str | Path) -> list[Path]:
        path = resolve_path(path_value)
        key = _normalized_prefix(path)
        cached = self._subfolders.get(key)
        if cached is not None:
            return list(cached)
        error = self._errors.get(("subfolders", key))
        if error is not None:
            raise error
        try:
            folders = list_subfolders(path)
        except OSError as exc:
            self._errors[("subfolders", key)] = exc
            raise
        self._subfolders[key] = folders
        return list(folders)


def clean_path(value: str | Path) -> str:
    text = str(value).strip().strip("{}").strip("\"'")
    parsed = urlparse(text)
    if parsed.scheme == "file":
        text = unquote(parsed.path)
        if len(text) >= 4 and text[0] == "/" and text[2] == ":":
            text = text[1:]
    if len(text) > 2 and text[1] == ":" and text[2] not in {"\\", "/"}:
        text = f"{text[:2]}\\{text[2:]}"
    elif len(text) == 2 and text[1] == ":":
        text = f"{text}\\"
    return text


def resolve_path(value: str | Path) -> Path:
    return Path(clean_path(value))


def make_unique_work_dir(parent: str | Path, prefix: str = "") -> Path:
    root = Path(parent)
    root.mkdir(parents=True, exist_ok=True)
    base_name = prefix + datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / base_name
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base_name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def make_unique_stage_path(parent: str | Path, name: str) -> Path:
    safe_name = "".join("_" if char in '<>:"/\\|?*' else char for char in name).strip()
    safe_name = safe_name or "input"
    candidate = Path(parent) / safe_name
    suffix = 2
    while candidate.exists():
        candidate = Path(parent) / f"{safe_name}_{suffix}"
        suffix += 1
    return candidate


def iter_image_files(path_value: str | Path, extensions: set[str] | None = None) -> list[Path]:
    return scan_image_files(path_value, extensions).files


def clear_fallback_caches() -> None:
    _DIR_FALLBACK_FAILURES.clear()
    _FOLDER_FALLBACK_FAILURES.clear()
    _COPY_FALLBACK_FAILURES.clear()


def list_subfolders(path_value: str | Path) -> list[Path]:
    root = resolve_path(path_value)
    try:
        entries = list(os.scandir(root))
    except OSError:
        return _dir_subfolders(root)

    folders: list[Path] = []
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                folders.append(Path(entry.path))
        except OSError:
            continue
    return sorted(folders, key=lambda path: path.name.lower())


def path_looks_like_folder(path_value: str | Path, extensions: set[str] | None = None) -> bool:
    extensions = {ext.lower() for ext in (extensions or IMAGE_EXTENSIONS)}
    path = resolve_path(path_value)
    if _is_image_file(path, extensions):
        return False
    try:
        if path.is_dir():
            return True
    except OSError:
        pass
    try:
        list_folder_children(path)
        return True
    except OSError:
        return False


def list_folder_children(path_value: str | Path, *, allow_fallback: bool = True) -> list[FolderEntry]:
    root = resolve_path(path_value)
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        if not allow_fallback:
            raise
        return _dir_folder_children(root, reason=f"os.scandir failed: {exc}")

    children: list[FolderEntry] = []
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
            is_file = entry.is_file(follow_symlinks=False)
        except OSError as exc:
            logger.debug(
                "Folder list entry type fallback: root=%s entry=%s error=%s",
                root,
                entry.path,
                exc,
            )
            if not allow_fallback:
                raise
            return _dir_folder_children(
                root,
                reason=f"is_file/is_dir failed for {entry.path}: {exc}",
            )
        children.append(FolderEntry(Path(entry.path), is_dir=is_dir, is_file=is_file))
    return children


def scan_image_files(path_value: str | Path, extensions: set[str] | None = None) -> ScanResult:
    extensions = {ext.lower() for ext in (extensions or IMAGE_EXTENSIONS)}
    original_path = str(path_value)
    cleaned_path = clean_path(path_value)
    path = resolve_path(path_value)
    if _is_image_file(path, extensions):
        result = ScanResult(
            original_path=original_path,
            cleaned_path=cleaned_path,
            files=[path],
            total_file_count=1,
            image_file_count=1,
            excluded_output_count=0,
        )
        _log_scan_result(result)
        return result

    found: list[Path] = []
    seen: set[str] = set()
    excluded_output_count = 0
    candidates, dir_fallback = _iter_files_recursive(path)
    for candidate in candidates:
        if not _is_image_file(candidate, extensions):
            continue
        if _is_ignored_output(candidate):
            excluded_output_count += 1
            logger.debug("Excluded image by output-folder rule: %s", candidate)
            continue
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            found.append(candidate)

    result = ScanResult(
        original_path=original_path,
        cleaned_path=cleaned_path,
        files=sorted(found, key=lambda item: str(item).lower()),
        total_file_count=len(candidates),
        image_file_count=len(found),
        excluded_output_count=excluded_output_count,
        dir_fallback=dir_fallback,
    )
    _log_scan_result(result)
    return result


def _log_scan_result(result: ScanResult) -> None:
    if not DEBUG:
        return
    logger.debug(
        "Image scan: original=%s cleaned=%s total_files=%d image_files=%d excluded_output=%d",
        result.original_path,
        result.cleaned_path,
        result.total_file_count,
        result.image_file_count,
        result.excluded_output_count,
    )
    if result.dir_fallback is not None:
        logger.debug(
            "Image scan fallback: reason=%s command=%s returncode=%s stderr=%s",
            result.dir_fallback.reason,
            " ".join(result.dir_fallback.command),
            result.dir_fallback.returncode,
            result.dir_fallback.stderr or "-",
        )


def safe_copy_file(source: str | Path, destination: str | Path) -> CopyResult:
    src = resolve_path(source)
    dst = Path(destination)
    errors: list[str] = []

    if _ensure_parent(dst):
        try:
            shutil.copy2(src, dst)
            if dst.exists():
                return CopyResult(True, "shutil.copy2")
        except Exception as exc:
            errors.append(f"shutil.copy2: {exc}")
    else:
        errors.append(f"mkdir: failed to create {dst.parent}")

    for method_name, copier in (
        ("powershell Copy-Item", _copy_with_powershell),
        ("cmd copy", _copy_with_cmd_copy),
        ("xcopy", _copy_with_xcopy),
        ("robocopy", _copy_with_robocopy),
    ):
        failure_key = (
            method_name,
            _normalized_prefix(src.parent),
            _normalized_prefix(dst.parent),
        )
        cached_error = _COPY_FALLBACK_FAILURES.get(failure_key)
        if cached_error is not None:
            errors.append(f"{method_name}: skipped after previous failure: {cached_error}")
            continue
        result = copier(src, dst)
        if result.success:
            return result
        _COPY_FALLBACK_FAILURES[failure_key] = result.error
        errors.append(f"{method_name}: {result.error}")

    return CopyResult(False, "failed", "\n".join(error for error in errors if error))


def _is_image_file(path: Path, extensions: set[str]) -> bool:
    return path.suffix.lower() in extensions


def _is_ignored_output(path: Path) -> bool:
    return any(
        part.lower() in {"_quick_crop_ml", "__pycache__"}
        or part.lower().endswith("_pass")
        or part.lower().endswith("_mask")
        or part.lower().endswith("_masks")
        for part in path.parts
    )


def _iter_files_recursive(root: Path) -> tuple[list[Path], DirFallbackResult | None]:
    try:
        entries = list(os.scandir(root))
    except OSError as exc:
        fallback = _dir_files_recursive(root, reason=f"os.scandir failed: {exc}")
        return fallback.files, fallback

    files: list[Path] = []
    for entry in entries:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
            is_file = entry.is_file(follow_symlinks=False)
        except OSError as exc:
            fallback = _dir_files_recursive(
                root,
                reason=f"is_file/is_dir failed for {entry.path}: {exc}",
            )
            return fallback.files, fallback

        if is_dir:
            child_files, child_fallback = _iter_files_recursive(Path(entry.path))
            files.extend(child_files)
            if child_fallback is not None:
                return files, child_fallback
        elif is_file:
            files.append(Path(entry.path))
    return files, None


def _dir_files_recursive(root: Path, reason: str = "") -> DirFallbackResult:
    command = ["cmd", "/d", "/s", "/c", "dir", "/a:-d", "/b", "/s", str(root)]
    cache_key = (_normalized_prefix(root), "files_recursive")
    cached_failure = _DIR_FALLBACK_FAILURES.get(cache_key)
    if cached_failure is not None:
        return cached_failure
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=DIR_TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        fallback = DirFallbackResult(
            files=[],
            command=command,
            returncode=None,
            stderr=f"Timed out after {DIR_TIMEOUT_SECONDS}s: {_short_output(exc.stderr)}",
            reason=reason,
        )
        _DIR_FALLBACK_FAILURES[cache_key] = fallback
        return fallback
    if result.returncode != 0:
        fallback = DirFallbackResult(
            files=[],
            command=command,
            returncode=result.returncode,
            stderr=_short_output(result.stderr),
            reason=reason,
        )
        _DIR_FALLBACK_FAILURES[cache_key] = fallback
        return fallback
    return DirFallbackResult(
        files=[Path(line.strip()) for line in result.stdout.splitlines() if line.strip()],
        command=command,
        returncode=result.returncode,
        stderr=_short_output(result.stderr),
        reason=reason,
    )


def _dir_folder_children(root: Path, reason: str = "") -> list[FolderEntry]:
    cache_key = _normalized_prefix(root)
    cached_failure = _FOLDER_FALLBACK_FAILURES.get(cache_key)
    if cached_failure is not None:
        raise cached_failure
    children: list[FolderEntry] = []
    errors: list[str] = []
    for attr, is_dir, is_file in (
        ("/a:d", True, False),
        ("/a:-d", False, True),
    ):
        command = ["cmd", "/d", "/s", "/c", "dir", attr, "/b", str(root)]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=DIR_TIMEOUT_SECONDS,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            errors.append(
                f"{'folders' if is_dir else 'files'} timed out after {DIR_TIMEOUT_SECONDS}s: "
                f"{_short_output(exc.stderr) or '-'}"
            )
            continue
        logger.debug(
            "Folder list fallback: reason=%s command=%s returncode=%s stderr=%s",
            reason,
            " ".join(command),
            result.returncode,
            _short_output(result.stderr) or "-",
        )
        if result.returncode != 0:
            output = _short_output(result.stderr or result.stdout)
            errors.append(
                f"{'folders' if is_dir else 'files'} rc={result.returncode}: {output or '-'}"
            )
            continue
        for line in result.stdout.splitlines():
            name = line.strip()
            if name:
                children.append(FolderEntry(root / name, is_dir=is_dir, is_file=is_file))
    if children:
        return children
    if errors:
        inferred = _infer_folder_children_from_recursive_files(root, reason, errors)
        if inferred:
            return inferred
    if len(errors) < 2:
        return children
    error = OSError(f"Failed to list folder with cmd dir: {root}; {'; '.join(errors)}")
    _FOLDER_FALLBACK_FAILURES[cache_key] = error
    raise error


def _dir_subfolders(root: Path) -> list[Path]:
    cache_key = (_normalized_prefix(root), "subfolders")
    cached_failure = _DIR_FALLBACK_FAILURES.get(cache_key)
    if cached_failure is not None:
        return []
    command = ["cmd", "/d", "/s", "/c", "dir", "/ad", "/b", str(root)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=DIR_TIMEOUT_SECONDS,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        _DIR_FALLBACK_FAILURES[cache_key] = object()
        return []
    if result.returncode != 0:
        _DIR_FALLBACK_FAILURES[cache_key] = object()
        return []
    return sorted(
        [root / line.strip() for line in result.stdout.splitlines() if line.strip()],
        key=lambda path: path.name.lower(),
    )


def _infer_folder_children_from_recursive_files(
    root: Path,
    reason: str,
    errors: list[str],
) -> list[FolderEntry]:
    fallback = _dir_files_recursive(
        root,
        reason=f"immediate folder list failed: {reason}; {'; '.join(errors)}",
    )
    if fallback.returncode != 0 or not fallback.files:
        if fallback.stderr:
            errors.append(f"recursive files rc={fallback.returncode}: {fallback.stderr}")
        return []

    children: list[FolderEntry] = []
    seen: set[str] = set()
    root_text = _normalized_prefix(root)
    for file_path in fallback.files:
        relative = _relative_text(root_text, file_path)
        if not relative:
            continue
        parts = [part for part in re.split(r"[\\/]+", relative) if part]
        if not parts:
            continue
        name = parts[0]
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        children.append(
            FolderEntry(
                root / name,
                is_dir=len(parts) > 1,
                is_file=len(parts) == 1,
            )
        )
    return children


def _normalized_prefix(path: Path) -> str:
    return str(path).rstrip("\\/").replace("/", "\\").lower()


def _relative_text(root_text: str, path: Path) -> str:
    text = str(path).replace("/", "\\")
    lowered = text.lower()
    if lowered == root_text:
        return ""
    prefix = root_text + "\\"
    if lowered.startswith(prefix):
        return text[len(prefix) :]
    return path.name


def _should_use_cmd_dir_first(path: Path) -> bool:
    return False


def _ensure_parent(path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        pass
    result = _run_powershell_env(
        (
            "$p=$env:QCML_DESTINATION_PARENT; "
            "[System.IO.Directory]::CreateDirectory($p) | Out-Null"
        ),
        {"QCML_DESTINATION_PARENT": str(path.parent)},
    )
    if result.returncode == 0:
        return True
    result = _run_cmd_env(
        'mkdir "%QCML_DESTINATION_PARENT%"',
        {"QCML_DESTINATION_PARENT": str(path.parent)},
    )
    return result.returncode == 0


def _copy_with_powershell(source: Path, destination: Path) -> CopyResult:
    result = _run_powershell_env(
        (
            "Copy-Item "
            "-LiteralPath $env:QCML_SOURCE_PATH "
            "-Destination $env:QCML_DESTINATION_PATH "
            "-Force -ErrorAction Stop"
        ),
        {
            "QCML_SOURCE_PATH": str(source),
            "QCML_DESTINATION_PATH": str(destination),
        },
    )
    return _copy_result_from_process(result, destination, "powershell Copy-Item")


def _copy_with_cmd_copy(source: Path, destination: Path) -> CopyResult:
    result = _run_cmd_env(
        'copy /Y "%QCML_SOURCE_PATH%" "%QCML_DESTINATION_PATH%"',
        {
            "QCML_SOURCE_PATH": str(source),
            "QCML_DESTINATION_PATH": str(destination),
        },
    )
    return _copy_result_from_process(result, destination, "cmd copy")


def _copy_with_xcopy(source: Path, destination: Path) -> CopyResult:
    result = _run_cmd_env(
        'echo F|xcopy /Y "%QCML_SOURCE_PATH%" "%QCML_DESTINATION_PATH%"',
        {
            "QCML_SOURCE_PATH": str(source),
            "QCML_DESTINATION_PATH": str(destination),
        },
    )
    return _copy_result_from_process(result, destination, "xcopy")


def _copy_with_robocopy(source: Path, destination: Path) -> CopyResult:
    source_dir = str(source.parent)
    result = subprocess.run(
        [
            "robocopy",
            source_dir,
            str(destination.parent),
            source.name,
            "/NFL",
            "/NDL",
            "/NJH",
            "/NJS",
            "/NP",
        ],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode >= 8:
        return CopyResult(False, "robocopy", _short_output(result.stderr or result.stdout))
    copied = destination.parent / source.name
    if copied.exists() and copied != destination:
        try:
            if destination.exists():
                destination.unlink()
            copied.replace(destination)
        except OSError as exc:
            return CopyResult(False, "robocopy", str(exc))
    if destination.exists():
        return CopyResult(True, "robocopy")
    return CopyResult(False, "robocopy", "destination file was not created")


def _copy_result_from_process(
    result: subprocess.CompletedProcess,
    destination: Path,
    method: str,
) -> CopyResult:
    if result.returncode == 0 and destination.exists():
        return CopyResult(True, method)
    detail = _short_output(result.stderr or result.stdout)
    if result.returncode == 0 and not detail:
        detail = "destination file was not created"
    return CopyResult(False, method, detail)


def _run_powershell_env(script: str, env_vars: dict[str, str]) -> subprocess.CompletedProcess:
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    env = os.environ.copy()
    env.update(env_vars)
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            encoded,
        ],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=env,
    )


def _run_cmd_env(command: str, env_vars: dict[str, str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_vars)
    return subprocess.run(
        ["cmd", "/d", "/s", "/c", command],
        capture_output=True,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        env=env,
    )


def _short_output(text: str, limit: int = 1200) -> str:
    detail = str(text or "").strip()
    if len(detail) <= limit:
        return detail
    return detail[:limit].rstrip() + "..."
