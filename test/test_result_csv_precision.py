from pathlib import Path

from gui.ml_tab import MLTab, _ImageTask as MLImageTask
from gui.workflow_tab import WorkflowTab, _ImageTask as WorkflowImageTask


def _result_row() -> dict:
    return {
        "AnalysisID": "TEST01",
        "Cell": 1,
        "Face": "FRONT",
        "ImageNum": 1,
        "total_pixel": 500,
        "background_ratio": 0.994,
        "nonactive_ratio": 0.002,
        "li_salt_ratio": 0.004,
        "correction_applied": 0,
        "isIssue": False,
        "issueCategory": "",
        "issueNote": "",
    }


def test_workflow_result_csv_keeps_three_decimal_places(tmp_path: Path) -> None:
    task = WorkflowImageTask(
        analysis_id="TEST01",
        analysis_path=tmp_path,
        cell=1,
        face="FRONT",
        image_num=1,
        image_path=tmp_path / "source.png",
    )

    csv_path = WorkflowTab.__new__(WorkflowTab)._write_csv(
        "TEST01", [task], [_result_row()]
    )

    data_row = csv_path.read_text(encoding="utf-8").splitlines()[1]
    assert data_row == "TEST01,1,FRONT,1,500,0.994,0.002,0.004,0"


def test_ml_tab_result_csv_keeps_three_decimal_places(tmp_path: Path) -> None:
    task = MLImageTask(
        run_key="run",
        date_name="2026-09-29",
        analysis_id="TEST01",
        analysis_path=tmp_path,
        cell=1,
        face="FRONT",
        image_num=1,
        image_path=tmp_path / "source.png",
    )

    csv_path = MLTab.__new__(MLTab)._write_analysis_csv(
        "TEST01", [task], [_result_row()]
    )

    data_row = csv_path.read_text(encoding="utf-8").splitlines()[1]
    assert data_row == "TEST01,1,FRONT,1,500,0.994,0.002,0.004,False,,"
