import re
from pathlib import Path

from gui.workflow_tab import WorkflowTab


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _workflow_for_scan(root_path: Path) -> WorkflowTab:
    workflow = WorkflowTab.__new__(WorkflowTab)
    workflow._root_path = root_path
    workflow._date_formats = ["%Y-%m-%d"]
    workflow._mode_var = _Value("By Date")
    workflow._cell_folder_re = re.compile(r"^cell\s*#\d+$", re.IGNORECASE)
    return workflow


def test_scan_targets_accepts_all_supported_work_folder_shapes(tmp_path: Path) -> None:
    date_path = tmp_path / "2026-08-08"

    (date_path / "AHG12345" / "CELL #1" / "FRONT").mkdir(parents=True)
    (date_path / "AFG12345" / "BACK").mkdir(parents=True)
    (date_path / "arbitrary_cells" / "CELL #1" / "BACK").mkdir(parents=True)
    (date_path / "arbitrary_face" / "FRONT").mkdir(parents=True)
    direct = date_path / "ets_test_cell_fast"
    direct.mkdir(parents=True)
    (direct / "sample.jpg").write_bytes(b"")
    for generated_name in ("qc_test1_pass", "qc_test_1_MASK"):
        generated = date_path / generated_name
        generated.mkdir(parents=True)
        (generated / "generated.png").write_bytes(b"")
    (date_path / "not_a_work_folder").mkdir(parents=True)

    workflow = _workflow_for_scan(tmp_path)
    targets = workflow._scan_targets("2026-08-08")

    assert [work_id for _date, _path, work_id in targets] == [
        "AFG12345",
        "AHG12345",
        "arbitrary_cells",
        "arbitrary_face",
        "ets_test_cell_fast",
    ]


def test_run_requested_reuses_start_run_with_checkbox_values() -> None:
    workflow = WorkflowTab.__new__(WorkflowTab)
    workflow._run_qc_var = _Value(True)
    workflow._run_ml_var = _Value(False)
    workflow._force_run_var = _Value(True)
    workflow._selected_items = lambda: ["work"]
    workflow._auto_items = lambda: ["auto"]
    calls = []
    workflow._start_run = lambda items, force_qc, run_qc, run_ml: calls.append(
        (items, force_qc, run_qc, run_ml)
    )

    workflow._run_requested()

    assert calls == [(["work"], True, True, False)]


def test_artifact_status_does_not_depend_on_selected_customer(tmp_path: Path) -> None:
    date_path = tmp_path / "2026-08-08"
    work_path = date_path / "qc_test_1"
    work_path.mkdir(parents=True)
    (work_path / "source.jpg").write_bytes(b"")

    pass_path = date_path / "qc_test_1_pass"
    mask_path = date_path / "qc_test_1_mask"
    pass_path.mkdir()
    mask_path.mkdir()
    (pass_path / "source.png").write_bytes(b"")
    (mask_path / "source_mask.png").write_bytes(b"")

    workflow = _workflow_for_scan(tmp_path)
    workflow._selected_customer = "HKMC"
    hkmc_status = workflow._qc_complete("2026-08-08", date_path, "qc_test_1")
    workflow._selected_customer = "VW"
    vw_status = workflow._qc_complete("2026-08-08", date_path, "qc_test_1")

    assert hkmc_status == (True, "QC ready")
    assert vw_status == hkmc_status
    assert workflow._mask_outputs_complete("qc_test_1", work_path)
