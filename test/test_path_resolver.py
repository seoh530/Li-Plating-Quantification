import os
import subprocess
from pathlib import Path

import pytest

from modules.utils import path_resolver


@pytest.fixture(autouse=True)
def clear_path_resolver_caches():
    path_resolver.clear_fallback_caches()


def test_clean_path_recovers_drive_relative_path() -> None:
    assert path_resolver.clean_path(r"T:NA\Cell #2") == r"T:\NA\Cell #2"
    assert path_resolver.clean_path(r"{T:\NA\Cell #2}") == r"T:\NA\Cell #2"
    assert path_resolver.clean_path(r"file:///T:/NA/Cell%20%232") == "T:/NA/Cell #2"


def test_iter_image_files_uses_dir_when_scandir_fails(monkeypatch) -> None:
    def fail_scandir(_path):
        raise OSError("blocked")

    def fake_run(cmd, **kwargs):
        assert cmd == [
            "cmd",
            "/d",
            "/s",
            "/c",
            "dir",
            "/a:-d",
            "/b",
            "/s",
            r"T:\NA\Cell #2",
        ]
        return subprocess.CompletedProcess(
            cmd,
            0,
            r"T:\NA\Cell #2\FRONT\a.jpg" + os.linesep + r"T:\NA\Cell #2\memo.txt",
            "",
        )

    monkeypatch.setattr(path_resolver.os, "scandir", fail_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    assert path_resolver.iter_image_files(r"T:\NA\Cell #2") == [
        Path(r"T:\NA\Cell #2\FRONT\a.jpg")
    ]


def test_scan_image_files_reports_dir_failure(monkeypatch) -> None:
    def fail_scandir(_path):
        raise OSError("blocked")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 5, "", "Access is denied")

    monkeypatch.setattr(path_resolver.os, "scandir", fail_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    result = path_resolver.scan_image_files(r"T:\NA\Cell #2")

    assert result.files == []
    assert result.total_file_count == 0
    assert result.image_file_count == 0
    assert result.dir_fallback is not None
    assert result.dir_fallback.returncode == 5
    assert result.dir_fallback.stderr == "Access is denied"
    assert "dir return code: 5" in result.summary()


def test_scan_image_files_counts_output_folder_exclusions(monkeypatch) -> None:
    def fail_scandir(_path):
        raise OSError("blocked")

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            (
                r"T:\NA\Cell #2\FRONT\a.jpg"
                + os.linesep
                + r"T:\NA\Cell #2\FRONT_pass\a.png"
            ),
            "",
        )

    monkeypatch.setattr(path_resolver.os, "scandir", fail_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    result = path_resolver.scan_image_files(r"T:\NA\Cell #2")

    assert result.files == [Path(r"T:\NA\Cell #2\FRONT\a.jpg")]
    assert result.total_file_count == 2
    assert result.image_file_count == 1
    assert result.excluded_output_count == 1


def test_iter_image_files_uses_dir_when_entry_type_check_fails(monkeypatch, tmp_path) -> None:
    root = tmp_path / "Cell #2"

    class BrokenEntry:
        path = str(root / "FRONT")

        def is_dir(self, follow_symlinks=False):
            raise OSError("entry blocked")

        def is_file(self, follow_symlinks=False):
            raise OSError("entry blocked")

    def fake_scandir(_path):
        return [BrokenEntry()]

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            str(root / "FRONT" / "a.jpg"),
            "",
        )

    monkeypatch.setattr(path_resolver.os, "scandir", fake_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    result = path_resolver.scan_image_files(root)

    assert result.files == [root / "FRONT" / "a.jpg"]
    assert result.dir_fallback is not None
    assert "is_file/is_dir failed" in result.dir_fallback.reason


def test_list_folder_children_uses_cmd_dir_fallback_for_teamroom_path(monkeypatch) -> None:
    root = r"T:\Team Room\공정 #2"
    calls = []

    def fail_scandir(_path):
        raise OSError("blocked")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert cmd[-1] == root
        if "/a:d" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "FRONT\nBACK\n", "")
        if "/a:-d" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "overview.jpg\nmemo.txt\n", "")
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(path_resolver.os, "scandir", fail_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    children = path_resolver.list_folder_children(root)

    assert [child.path for child in children] == [
        Path(root) / "FRONT",
        Path(root) / "BACK",
        Path(root) / "overview.jpg",
        Path(root) / "memo.txt",
    ]
    assert [child.is_dir for child in children] == [True, True, False, False]
    assert calls == [
        ["cmd", "/d", "/s", "/c", "dir", "/a:d", "/b", root],
        ["cmd", "/d", "/s", "/c", "dir", "/a:-d", "/b", root],
    ]


def test_teamroom_folder_listing_uses_scandir_when_available(monkeypatch) -> None:
    root = Path(r"T:\Team Room\Cell #2")

    class Entry:
        path = str(root / "FRONT")

        def is_dir(self, follow_symlinks=False):
            return True

        def is_file(self, follow_symlinks=False):
            return False

    def fake_scandir(_path):
        return [Entry()]

    def fake_run(cmd, **kwargs):
        raise AssertionError("mapped drive scandir success should not call cmd fallback")

    monkeypatch.setattr(path_resolver.os, "scandir", fake_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    children = path_resolver.list_folder_children(r"T:\Team Room\Cell #2")

    assert children[0].path == root / "FRONT"


def test_list_subfolders_uses_scandir_without_cmd_fallback(monkeypatch) -> None:
    root = Path(r"T:\Team Room\Analysis")

    class FolderEntry:
        path = str(root / "BatchA")

        def is_dir(self, follow_symlinks=False):
            return True

    class FileEntry:
        path = str(root / "result.csv")

        def is_dir(self, follow_symlinks=False):
            return False

    def fake_scandir(_path):
        return [FileEntry(), FolderEntry()]

    def fake_run(cmd, **kwargs):
        raise AssertionError("scandir success should not call cmd fallback")

    monkeypatch.setattr(path_resolver.os, "scandir", fake_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    assert path_resolver.list_subfolders(root) == [root / "BatchA"]


def test_list_subfolders_fallback_is_non_recursive_cmd_dir(monkeypatch) -> None:
    root = Path(r"T:\Team Room\Analysis")
    calls = []

    def fail_scandir(_path):
        raise OSError("blocked")

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "BatchA\nBatchB\n", "")

    monkeypatch.setattr(path_resolver.os, "scandir", fail_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    assert path_resolver.list_subfolders(root) == [root / "BatchA", root / "BatchB"]
    assert calls == [["cmd", "/d", "/s", "/c", "dir", "/ad", "/b", str(root)]]


def test_folder_children_infers_subfolders_when_immediate_cmd_dir_is_denied(monkeypatch) -> None:
    root = r"T:\Team Room\Cell #2"

    def fail_scandir(_path):
        raise OSError("blocked")

    def fake_run(cmd, **kwargs):
        if len(cmd) >= 2 and cmd[-2] == "/s":
            return subprocess.CompletedProcess(
                cmd,
                0,
                (
                    root
                    + r"\FRONT\front_01.jpg"
                    + os.linesep
                    + root
                    + r"\BACK\back_01.jpg"
                ),
                "",
            )
        return subprocess.CompletedProcess(cmd, 5, "", "Access is denied")

    monkeypatch.setattr(path_resolver.os, "scandir", fail_scandir)
    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    children = path_resolver.list_folder_children(root)

    assert [(child.path.name, child.is_dir, child.is_file) for child in children] == [
        ("FRONT", True, False),
        ("BACK", True, False),
    ]


def test_cmd_dir_timeout_returns_scan_summary(monkeypatch) -> None:
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, timeout=kwargs["timeout"], stderr="slow")

    monkeypatch.setattr(path_resolver.subprocess, "run", fake_run)

    result = path_resolver.scan_image_files(r"T:\Team Room\Cell #2")

    assert result.files == []
    assert result.dir_fallback is not None
    assert result.dir_fallback.returncode is None
    assert "Timed out after" in result.summary()


def test_safe_copy_file_uses_shutil_first(tmp_path) -> None:
    source = tmp_path / "source.jpg"
    destination = tmp_path / "out" / "source.jpg"
    source.write_bytes(b"image")

    result = path_resolver.safe_copy_file(source, destination)

    assert result.success
    assert result.method == "shutil.copy2"
    assert destination.read_bytes() == b"image"
