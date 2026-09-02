import sys
import queue
from pathlib import Path
import re
import subprocess
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.workflow_tab import WorkflowTab, _ImageTask, _QuickQCHoldError, _StageSource


def make_tab() -> WorkflowTab:
    tab = WorkflowTab.__new__(WorkflowTab)
    tab._analysis_id_re = re.compile(r"^[A-Z]{3}[0-9]{5}$")
    tab._cell_folder_re = re.compile(r"^cell\s*#\d+$", re.IGNORECASE)
    tab._date_formats = ["%Y-%m-%d"]
    tab._image_count_rules = {"H": 3}
    return tab


class FakeVar:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value


def test_folder_status_uses_shallow_scan_for_nested_images(tmp_path) -> None:
    tab = make_tab()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "image.jpg").write_bytes(b"")

    assert tab._folder_processing_status(tmp_path) == (
        "Open to inspect",
        "folder_ready",
    )
    assert tab._folder_processing_status(nested) == (
        "PASS missing 0/1",
        "folder_fail",
    )


def test_folder_status_counts_direct_conventional_outputs_only(tmp_path) -> None:
    tab = make_tab()
    output_dir = tmp_path / "FRONT_pass"
    output_dir.mkdir()
    (output_dir / "layer.png").write_bytes(b"")

    assert tab._folder_processing_status(tmp_path) == (
        "MASK missing 0/1",
        "folder_partial",
    )


def test_folder_status_reports_li_salt_when_mask_exists(tmp_path) -> None:
    tab = make_tab()
    analysis_path = tmp_path / "AHG00001"
    front_path = analysis_path / "FRONT"
    pass_path = analysis_path / "FRONT_pass"
    mask_path = analysis_path / "FRONT_mask"
    front_path.mkdir(parents=True)
    pass_path.mkdir()
    mask_path.mkdir()
    (front_path / "layer.jpg").write_bytes(b"")
    (pass_path / "layer.png").write_bytes(b"")
    (mask_path / "layer_mask.png").write_bytes(b"")
    (analysis_path / "AHG00001_LI-SALT_12.34%.csv").write_text("", encoding="utf-8")

    assert tab._folder_processing_status(analysis_path) == (
        "MASK done 1/1 / Li-salt 12.34%",
        "folder_done",
    )


def test_folder_status_averages_li_salt_for_multiple_image_csvs(tmp_path) -> None:
    tab = make_tab()
    work_path = tmp_path / "issue"
    pass_path = tmp_path / "issue_PASS"
    mask_path = tmp_path / "issue_MASK"
    work_path.mkdir()
    pass_path.mkdir()
    mask_path.mkdir()
    for name, ratio in (("image01", 0.10), ("image02", 0.20)):
        (work_path / f"{name}.jpg").write_bytes(b"")
        (pass_path / f"{name}.png").write_bytes(b"")
        (mask_path / f"{name}_mask.png").write_bytes(b"")
        (mask_path / f"{name}_LI_SALT.csv").write_text(f"li_salt_ratio\n{ratio}\n", encoding="utf-8")

    assert tab._folder_processing_status(work_path) == (
        "MASK done 2/2 / Li-salt 15.00%",
        "folder_done",
    )


def test_work_folders_for_date_uses_direct_image_folders_only(tmp_path) -> None:
    tab = make_tab()
    date_path = tmp_path / "2026-08-08"
    issue_path = date_path / "ets_test_cell_fast" / "issue"
    ref_front = date_path / "ets_test_cell_fast" / "ref" / "FRONT"
    ref_back = date_path / "ets_test_cell_fast" / "ref" / "BACK"
    output_path = ref_front / "_PASS"
    for folder in (issue_path, ref_front, ref_back, output_path):
        folder.mkdir(parents=True)
    (issue_path / "a.jpg").write_bytes(b"")
    (ref_front / "front.jpg").write_bytes(b"")
    (ref_back / "back.jpg").write_bytes(b"")
    (output_path / "generated.png").write_bytes(b"")

    assert [path.relative_to(date_path).as_posix() for path in tab._work_folders_for_date(date_path)] == [
        "ets_test_cell_fast/issue",
        "ets_test_cell_fast/ref/BACK",
        "ets_test_cell_fast/ref/FRONT",
    ]


def test_work_folder_count_mismatch_is_warning(tmp_path) -> None:
    tab = make_tab()
    tab._customer_var = FakeVar("H - HKMC")
    date_path = tmp_path / "2026-08-08"
    work_path = date_path / "ets_test_cell_fast"
    pass_path = work_path / "_PASS"
    pass_path.mkdir(parents=True)
    for index in range(2):
        name = f"image_{index}.jpg"
        (work_path / name).write_bytes(b"")
        (pass_path / f"image_{index}.png").write_bytes(b"")

    assert tab._qc_output_status("2026-08-08", date_path, work_path) == (
        "Warning",
        "Warning: Expected 3 electrodes, found 2",
    )


def test_etc_work_folder_has_no_count_warning(tmp_path) -> None:
    tab = make_tab()
    tab._customer_var = FakeVar("ETC")
    date_path = tmp_path / "2026-08-08"
    work_path = date_path / "ets_test_cell_fast"
    pass_path = work_path / "_PASS"
    pass_path.mkdir(parents=True)
    (work_path / "image.jpg").write_bytes(b"")
    (pass_path / "image.png").write_bytes(b"")

    assert tab._qc_output_status("2026-08-08", date_path, work_path) == (
        "Pass",
        "QC ready",
    )


def test_work_folder_status_accepts_sibling_pass_folder(tmp_path) -> None:
    tab = make_tab()
    tab._customer_var = FakeVar("ETC")
    date_path = tmp_path / "2026-08-08"
    work_path = date_path / "ets_test_cell_fast" / "issue"
    pass_path = work_path.parent / "issue_PASS"
    work_path.mkdir(parents=True)
    pass_path.mkdir(parents=True)
    for index in range(2):
        (work_path / f"image_{index}.jpg").write_bytes(b"")
        pass_suffix = ".jpg" if index == 0 else ".png"
        (pass_path / f"image_{index}{pass_suffix}").write_bytes(b"")

    assert tab._qc_output_status("2026-08-08", date_path, work_path) == (
        "Pass",
        "QC ready",
    )


def test_build_ml_tasks_uses_sibling_pass_folder(tmp_path) -> None:
    tab = make_tab()
    tab._customer_var = FakeVar("ETC")
    date_path = tmp_path / "2026-08-08"
    work_path = date_path / "ets_test_cell_fast" / "issue"
    pass_path = work_path.parent / "issue_PASS"
    work_path.mkdir(parents=True)
    pass_path.mkdir(parents=True)
    (work_path / "image_1.jpg").write_bytes(b"")
    (pass_path / "image_1.png").write_bytes(b"")
    item = SimpleNamespace(
        date_name="2026-08-08",
        date_path=date_path,
        analysis_id="2026_08_08_issue",
        analysis_path=work_path,
    )

    tasks = tab._build_ml_tasks(item)

    assert [task.image_path for task in tasks] == [pass_path / "image_1.png"]
    assert tab._mask_dir_for_task(tasks[0]) == work_path.parent / "issue_MASK"


def test_workflow_stops_for_qc_hold_before_ml(tmp_path) -> None:
    tab = make_tab()
    work_path = tmp_path / "2026-08-08" / "issue"
    work_path.mkdir(parents=True)
    failed_item = {
        "id": str(work_path / "image_1.jpg"),
        "image_path": work_path / "image_1.jpg",
        "output_path": work_path / "_PASS" / "image_1.png",
        "fail_reasons": ["Aspect ratio out of tolerance"],
        "customer_code": "H",
        "skip_size_validation": False,
        "image_postprocess": None,
    }
    item = SimpleNamespace(
        key="held",
        date_name="2026-08-08",
        date_path=tmp_path / "2026-08-08",
        analysis_id="2026_08_08_issue",
        analysis_path=work_path,
        qc_status="Needed",
        summary="QC not run",
    )
    ml_calls = []

    tab._queue = queue.Queue()
    tab._root_path = tmp_path
    tab._held_qc_failures = {}
    tab._stop_event = SimpleNamespace(is_set=lambda: False)
    tab._load_ml_config = lambda: None
    tab._cleanup_runtime_artifacts = lambda _path: None
    tab._reset_qc_hold_for_rerun = lambda _item: None
    tab._run_work_folder_crop = lambda *_args, **_kwargs: [failed_item]
    tab._build_item = lambda *_args: item
    tab._run_ml_for_item = lambda *_args, **_kwargs: ml_calls.append(True)

    tab._workflow_worker(
        [item],
        skip_pairs=set(),
        preview_enabled=True,
        force_qc=False,
        run_qc=True,
        only_crop=False,
        run_ml=True,
    )

    queued_types = [tab._queue.get_nowait()[0] for _ in range(tab._queue.qsize())]
    assert "batch_qc_hold" in queued_types
    assert "done" not in queued_types
    assert not ml_calls


def test_quick_worker_stops_for_qc_hold(tmp_path) -> None:
    tab = make_tab()
    source = tmp_path / "issue" / "image_1.jpg"
    source.parent.mkdir()
    source.write_bytes(b"image")
    failed_item = {
        "id": str(source),
        "image_path": source,
        "output_path": source.parent.parent / "issue_PASS" / "image_1.png",
        "fail_reasons": ["Aspect ratio out of tolerance"],
        "customer_code": "H",
        "skip_size_validation": False,
        "image_postprocess": None,
    }

    tab._queue = queue.Queue()
    tab._stop_event = SimpleNamespace(is_set=lambda: False)
    tab._run_quick_file = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        _QuickQCHoldError("Aspect ratio out of tolerance", failed_item)
    )

    tab._quick_worker(
        [source],
        preview_enabled=True,
        run_ml=False,
        label_prefix="Folder",
    )

    queued_types = [tab._queue.get_nowait()[0] for _ in range(tab._queue.qsize())]
    assert "quick_qc_hold" in queued_types
    assert "quick_done" not in queued_types


def test_work_folder_crop_collects_all_qc_failures(tmp_path) -> None:
    tab = make_tab()
    tab._customer_var = FakeVar("H - HKMC")
    work_path = tmp_path / "2026-08-08" / "issue"
    work_path.mkdir(parents=True)
    first = work_path / "image_1.jpg"
    second = work_path / "image_2.jpg"
    first.write_bytes(b"image")
    second.write_bytes(b"image")
    calls = []

    class FakeCropEngine:
        def process_image(self, *, image_path, **_kwargs):
            calls.append(image_path)
            return SimpleNamespace(
                status="FAIL",
                output_path=None,
                fail_reasons=["Aspect ratio out of tolerance"],
            )

    tab._quick_crop_engine = FakeCropEngine()
    tab._qc_pipeline = SimpleNamespace(clear_pass_outputs=lambda _path: None)
    item = SimpleNamespace(analysis_path=work_path)

    failures = tab._run_work_folder_crop(
        item,
        skip_size_validation=False,
        force_save_failed=False,
    )

    assert calls == [first, second]
    assert len(failures) == 2
    assert failures[0]["image_path"] == first


def test_stage_paths_accepts_subprocess_visible_teamroom_path(tmp_path, monkeypatch) -> None:
    tab = make_tab()
    config_path = tmp_path / "config" / "thresholds.yaml"
    config_path.parent.mkdir()
    tab._config_path = config_path
    source_text = r"T:\Team Room\Batch 01"
    copied = []

    def fake_copy(source, destination):
        copied.append((source, destination))
        destination.mkdir(parents=True)
        (destination / "layer.jpg").write_bytes(b"")

    monkeypatch.setattr(tab, "_source_path_type", lambda text: "dir" if text == source_text else None)
    monkeypatch.setattr(tab, "_copy_path_with_subprocess", fake_copy)
    monkeypatch.setattr(tab, "_verify_stage_copy", lambda source, destination: None)

    staging_root = tab._stage_paths_to_local([source_text])

    assert staging_root.parent == tmp_path / "_local_folder_inputs"
    assert copied
    source, destination = copied[0]
    assert source.raw_path == source_text
    assert source.name == "Batch 01"
    assert source.is_dir is True
    assert destination == staging_root / "Batch 01"


def test_stage_paths_attempts_copy_when_teamroom_path_type_is_hidden(tmp_path, monkeypatch) -> None:
    tab = make_tab()
    config_path = tmp_path / "config" / "thresholds.yaml"
    config_path.parent.mkdir()
    tab._config_path = config_path
    source_text = r"T:\Team Room\Hidden Batch"
    copied = []

    def fake_copy(source, destination):
        copied.append((source, destination))
        destination.mkdir(parents=True)
        (destination / "layer.jpg").write_bytes(b"")

    monkeypatch.setattr(tab, "_source_path_type", lambda text: None)
    monkeypatch.setattr(tab, "_copy_path_with_subprocess", fake_copy)
    monkeypatch.setattr(tab, "_verify_stage_copy", lambda source, destination: None)

    staging_root = tab._stage_paths_to_local([source_text])

    assert copied
    source, destination = copied[0]
    assert source.raw_path == source_text
    assert source.name == "Hidden Batch"
    assert source.is_dir is None
    assert destination == staging_root / "Hidden Batch"


def test_clean_drop_path_recovers_drive_relative_teamroom_path() -> None:
    tab = make_tab()

    assert tab._clean_drop_path(r"T:NA\Batch 01") == r"T:\NA\Batch 01"
    assert tab._clean_drop_path(r"T:\NA\Batch 01") == r"T:\NA\Batch 01"
    assert tab._clean_drop_path(r"file:///T:/NA/Batch%2001") == "T:/NA/Batch 01"


def test_parse_dropped_files_preserves_unbraced_teamroom_path_with_spaces() -> None:
    tab = make_tab()

    assert tab._parse_dropped_files(r"T:\Team Room\Batch 01") == [
        r"T:\Team Room\Batch 01"
    ]
    assert tab._parse_dropped_files(r"{T:\Team Room\Batch 01}") == [
        r"T:\Team Room\Batch 01"
    ]


def test_parse_quick_path_entry_does_not_scan_teamroom_path(monkeypatch) -> None:
    tab = make_tab()

    def fail_exists(*_args, **_kwargs):
        raise AssertionError("Path entry parsing should not scan the teamroom folder")

    monkeypatch.setattr(tab, "_quick_path_exists", fail_exists)
    monkeypatch.setattr("gui.workflow_tab.path_resolver.iter_image_files", fail_exists)

    assert tab._parse_quick_path_entry(r"T:\Team Room\Batch 01") == [
        r"T:\Team Room\Batch 01"
    ]


def test_add_quick_paths_from_entry_keeps_path_text() -> None:
    tab = make_tab()
    source = r"T:\Team Room\Batch 01"
    loaded = []

    class FakeVar:
        def __init__(self, value: str) -> None:
            self.value = value

        def get(self) -> str:
            return self.value

        def set(self, value: str) -> None:
            self.value = value

    tab._quick_path_var = FakeVar(source)
    tab._start_folder_path_load = lambda paths: loaded.extend(paths)

    tab._add_quick_paths_from_entry()

    assert tab._quick_path_var.get() == source
    assert loaded == [source]


def test_copy_file_uses_encoded_powershell_env_for_space_paths(tmp_path, monkeypatch) -> None:
    tab = make_tab()
    source = _StageSource(
        raw_path=r"T:\~ Trade\Batch 01\layer.png",
        name="layer.png",
        is_dir=None,
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr("gui.workflow_tab.subprocess.run", fake_run)

    tab._copy_path_with_subprocess(source, tmp_path / "layer.png")

    assert calls
    cmd, kwargs = calls[0]
    assert "-EncodedCommand" in cmd
    assert source.raw_path not in cmd
    assert kwargs["env"]["QCML_SOURCE_PATH"] == source.raw_path


def test_verify_stage_copy_rejects_empty_unknown_folder(tmp_path) -> None:
    tab = make_tab()
    destination = tmp_path / "empty"
    destination.mkdir()
    source = _StageSource(
        raw_path=r"T:\Team Room\Cell #2",
        name="Cell #2",
        is_dir=None,
    )

    try:
        tab._verify_stage_copy(source, destination)
    except RuntimeError as exc:
        assert "Copied folder is empty" in str(exc)
    else:
        raise AssertionError("Expected empty staged folder to fail")


def test_refresh_root_tree_auto_expands_single_cell_folder(tmp_path) -> None:
    tab = make_tab()
    staging_root = tmp_path / "stage"
    cell_path = staging_root / "Cell #2"
    (cell_path / "FRONT").mkdir(parents=True)
    (cell_path / "BACK").mkdir()
    (cell_path / "FRONT" / "front_01.jpg").write_bytes(b"")
    (cell_path / "BACK" / "back_01.jpg").write_bytes(b"")
    tab._folder_root_override = staging_root

    root = tk.Tk()
    root.withdraw()
    try:
        tab._root_tree = ttk.Treeview(root)
        tab._root_tree_paths = {}
        tab._status_label = ttk.Label(root)
        tab._refresh_root_tree()

        root_id = tab._root_tree.get_children("")[0]
        cell_id = tab._root_tree.get_children(root_id)[0]
        child_names = [
            tab._root_tree.item(child_id, "text")
            for child_id in tab._root_tree.get_children(cell_id)
        ]
        face_image_counts = {}
        for face_id in tab._root_tree.get_children(cell_id):
            values = tab._root_tree.item(face_id, "values")
            face_image_counts[tab._root_tree.item(face_id, "text")] = values[0]

        assert tab._root_tree.item(cell_id, "open")
        assert child_names == ["BACK", "FRONT"]
        assert tab._root_tree.item(tab._root_tree.get_children(cell_id)[0], "open")
        assert tab._root_tree.item(tab._root_tree.get_children(cell_id)[1], "open")
        assert face_image_counts == {
            "BACK": "1",
            "FRONT": "1",
        }
    finally:
        root.destroy()


def test_quick_file_runs_on_local_copy_and_copies_result_back(tmp_path) -> None:
    tab = make_tab()
    tab._config_path = tmp_path / "config" / "thresholds.yaml"
    tab._config_path.parent.mkdir()
    source_dir = tmp_path / "teamroom"
    source_dir.mkdir()
    source = source_dir / "layer.jpg"
    source.write_bytes(b"image")

    class FakeCropEngine:
        def process_image(
            self,
            *,
            image_path,
            customer_code,
            output_path,
            skip_size_validation,
            image_postprocess,
        ):
            assert image_path != source
            assert image_path.parent.parent.name == "_quick_temp"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"crop")
            return SimpleNamespace(status="PASS", output_path=output_path, fail_reasons=[])

    tab._quick_crop_engine = FakeCropEngine()

    result = tab._run_quick_file(source, image_num=1, run_ml=False)

    assert result["crop_path"] == tmp_path / "teamroom_PASS" / "layer.png"
    assert result["crop_path"].read_bytes() == b"crop"
    assert not (tmp_path / "_quick_temp").exists() or not any((tmp_path / "_quick_temp").iterdir())


def test_quick_file_saves_crop_mask_and_returns_li_salt_row(tmp_path) -> None:
    tab = make_tab()
    tab._config_path = tmp_path / "config" / "thresholds.yaml"
    tab._config_path.parent.mkdir()
    source_dir = tmp_path / "Cell #2" / "FRONT"
    source_dir.mkdir(parents=True)
    source = source_dir / "image01.jpg"
    source.write_bytes(b"image")

    class FakeCropEngine:
        def process_image(
            self,
            *,
            image_path,
            customer_code,
            output_path,
            skip_size_validation,
            image_postprocess,
        ):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"crop")
            return SimpleNamespace(status="PASS", output_path=output_path, fail_reasons=[])

    class FakeInferencePipeline:
        def run_image(self, task, mask_path):
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.write_bytes(b"mask")
            return SimpleNamespace(
                mask_path=mask_path,
                result_row={
                    "AnalysisID": task.analysis_id,
                    "Cell": task.cell,
                    "Face": task.face,
                    "ImageNum": task.image_num,
                    "total_pixel": 100,
                    "background_ratio": 0.10,
                    "nonactive_ratio": 0.20,
                    "li_salt_ratio": 0.12345,
                },
                monitoring_record={"correction_applied": 1},
            )

    tab._quick_crop_engine = FakeCropEngine()
    tab._inference_pipeline = FakeInferencePipeline()

    result = tab._run_quick_file(source, image_num=1, run_ml=True)

    assert result["crop_path"] == tmp_path / "Cell #2" / "FRONT_PASS" / "image01.png"
    assert result["mask_path"] == tmp_path / "Cell #2" / "FRONT_MASK" / "image01_mask.png"
    assert result["crop_path"].read_bytes() == b"crop"
    assert result["mask_path"].read_bytes() == b"mask"
    assert result["result_row"]["source_image"] == str(source)
    assert result["result_row"]["li_salt_ratio"] == 0.12345
    assert not list((tmp_path / "Cell #2" / "FRONT_MASK").glob("*LI_SALT*.csv"))


def test_quick_folder_csv_is_integrated_per_source_folder(tmp_path) -> None:
    tab = make_tab()
    source_dir = tmp_path / "issue"
    mask_dir = tmp_path / "issue_MASK"
    source_dir.mkdir()
    mask_dir.mkdir()
    stale = mask_dir / "image01_LI_SALT_10.00%.csv"
    stale.write_text("stale", encoding="utf-8")
    image01 = source_dir / "image01.jpg"
    image02 = source_dir / "image02.jpg"
    image01.write_bytes(b"")
    image02.write_bytes(b"")

    tab._clear_quick_generated_csvs([image01, image02])
    csv_paths = tab._write_quick_folder_csvs(
        {
            source_dir: [
                {
                    "source_image": str(image01),
                    "crop_image": str(tmp_path / "issue_PASS" / "image01.png"),
                    "mask_image": str(mask_dir / "image01_mask.png"),
                    "AnalysisID": "QUICK_image01",
                    "Cell": 1,
                    "Face": "QUICK",
                    "ImageNum": 1,
                    "total_pixel": 100,
                    "background_ratio": 0.1,
                    "nonactive_ratio": 0.2,
                    "li_salt_ratio": 0.1,
                    "correction_applied": 0,
                },
                {
                    "source_image": str(image02),
                    "crop_image": str(tmp_path / "issue_PASS" / "image02.png"),
                    "mask_image": str(mask_dir / "image02_mask.png"),
                    "AnalysisID": "QUICK_image02",
                    "Cell": 1,
                    "Face": "QUICK",
                    "ImageNum": 2,
                    "total_pixel": 100,
                    "background_ratio": 0.1,
                    "nonactive_ratio": 0.2,
                    "li_salt_ratio": 0.2,
                    "correction_applied": 0,
                },
            ]
        }
    )

    assert not stale.exists()
    assert csv_paths == [mask_dir / "issue_LI-SALT_15.00%.csv"]
    frame = pd.read_csv(csv_paths[0])
    assert list(frame["source_image"]) == [str(image01), str(image02)]


def test_workflow_csv_is_written_to_mask_folder_and_stale_csvs_are_removed(tmp_path) -> None:
    tab = make_tab()
    work_path = tmp_path / "2026-08-08" / "issue"
    pass_dir = work_path / "_PASS"
    mask_dir = work_path / "_MASK"
    pass_dir.mkdir(parents=True)
    mask_dir.mkdir()
    image01 = pass_dir / "image01.png"
    image02 = pass_dir / "image02.png"
    image01.write_bytes(b"")
    image02.write_bytes(b"")
    stale = mask_dir / "image01_LI_SALT_10.00%.csv"
    stale.write_text("stale", encoding="utf-8")
    tasks = [
        _ImageTask("2026_08_08_issue", work_path, 1, "WORK", 1, image01, "ETC"),
        _ImageTask("2026_08_08_issue", work_path, 1, "WORK", 2, image02, "ETC"),
    ]
    rows = [
        {
            "AnalysisID": "2026_08_08_issue",
            "Cell": 1,
            "Face": "WORK",
            "ImageNum": 1,
            "total_pixel": 100,
            "background_ratio": 0.1,
            "nonactive_ratio": 0.2,
            "li_salt_ratio": 0.1,
            "correction_applied": 0,
        },
        {
            "AnalysisID": "2026_08_08_issue",
            "Cell": 1,
            "Face": "WORK",
            "ImageNum": 2,
            "total_pixel": 100,
            "background_ratio": 0.1,
            "nonactive_ratio": 0.2,
            "li_salt_ratio": 0.2,
            "correction_applied": 0,
        },
    ]

    tab._clear_mask_outputs_for_tasks(tasks, clear_masks=False)
    csv_path = tab._write_csv("2026_08_08_issue", tasks, rows)

    assert not stale.exists()
    assert csv_path == mask_dir / "2026_08_08_issue_LI-SALT_15.00%.csv"
    assert tab._find_result_csv(work_path, "2026_08_08_issue") == csv_path
    frame = pd.read_csv(csv_path)
    assert list(frame["ImageNum"]) == [1, 2]


def test_discover_quick_image_paths_hides_scan_details_when_empty(monkeypatch) -> None:
    tab = make_tab()

    def fake_scan(_path, _extensions):
        return SimpleNamespace(
            files=[],
            summary=lambda: (
                "Original path: T:\\NA\\Cell #2\n"
                "Cleaned path: T:\\NA\\Cell #2\n"
                "Total files scanned: 0\n"
                "Image files found: 0\n"
                "dir return code: 5\n"
                "dir stderr: Access is denied"
            ),
        )

    monkeypatch.setattr("gui.workflow_tab.path_resolver.scan_image_files", fake_scan)

    try:
        tab._discover_quick_image_paths([r"T:\NA\Cell #2"])
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected empty image scan to fail")

    assert "No supported image files found" in message
    assert "dir return code: 5" not in message
    assert "Access is denied" not in message


def test_discover_quick_image_entries_uses_relative_display_path(monkeypatch) -> None:
    tab = make_tab()
    source = r"T:\Project\CELL #2"
    image_path = Path(source) / "FRONT" / "image01.png"

    monkeypatch.setattr(
        "gui.workflow_tab.path_resolver.scan_image_files",
        lambda _path, _extensions: SimpleNamespace(
            files=[image_path],
            summary=lambda: "ok",
        ),
    )

    assert tab._discover_quick_image_entries([source]) == [
        (image_path, r"FRONT\image01.png")
    ]


def test_folder_root_from_loaded_paths_uses_single_input_folder(monkeypatch) -> None:
    tab = make_tab()
    source = r"T:\Team Room\Cell #2"

    monkeypatch.setattr(
        "gui.workflow_tab.path_resolver.path_looks_like_folder",
        lambda path, _extensions=None: str(path) == source,
    )

    assert tab._folder_root_from_loaded_paths(
        [source],
        [Path(source) / "FRONT" / "a.jpg"],
    ) == Path(source)


def test_folder_root_from_loaded_paths_ignores_multi_input(monkeypatch) -> None:
    tab = make_tab()

    monkeypatch.setattr(
        "gui.workflow_tab.path_resolver.path_looks_like_folder",
        lambda _path, _extensions=None: True,
    )

    assert tab._folder_root_from_loaded_paths(
        [r"T:\Team Room\Cell #1", r"T:\Team Room\Cell #2"],
        [Path(r"T:\Team Room\Cell #1\FRONT\a.jpg")],
    ) is None


def test_folder_path_load_worker_loads_folder_without_recursive_image_scan(monkeypatch) -> None:
    tab = make_tab()
    tab._queue = queue.Queue()
    source = r"T:\Team Room\Cell #2"

    monkeypatch.setattr(
        "gui.workflow_tab.path_resolver.scan_image_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Load Path should not recursively scan images")
        ),
    )
    monkeypatch.setattr(tab, "_list_folder_children", lambda _path: [])

    tab._folder_path_load_worker([source])

    msg_type, payload = tab._queue.get_nowait()
    assert msg_type == "quick_paths_ready"
    assert payload == {"folder_root": source, "images": [], "display": {}}
