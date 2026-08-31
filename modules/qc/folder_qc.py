import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from modules.utils.analysis_id import (
    find_test_images,
    find_test_source_folders,
    is_test_analysis_id,
)

logger = logging.getLogger(__name__)


# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class CellResult:
    cell_name: str
    front_path: Optional[Path]
    back_path: Optional[Path]
    front_count: int
    back_count: int
    status: str                          # "PASS" | "FAIL"
    fail_reasons: list[str] = field(default_factory=list)


@dataclass
class FolderQCResult:
    status: str                          # "PASS" | "FAIL"
    analysis_id: str
    customer: str
    cells: list[CellResult] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)


# ─── FolderQC ─────────────────────────────────────────────────────────────────

class FolderQC:
    """
    Validates PhotoJig folder structure and per-customer image count rules.

    FAIL reasons are returned in FolderQCResult.fail_reasons for display
    on the QC tab dashboard. RetakeManager is not involved.
    """

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._cfg: dict = cfg["folder_qc"]

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(
        self,
        root_path: str | Path,
        date_str: str,
        analysis_id: str,
    ) -> FolderQCResult:
        """
        Run all folder-level QC checks.

        Parameters
        ----------
        root_path   : PhotoJig capture root (e.g. C:/PhotoJig)
        date_str    : Name of the date folder exactly as it appears on disk
        analysis_id : AnalysisID folder name (e.g. AHG06113)

        Returns
        -------
        FolderQCResult with status "PASS" or "FAIL" and aggregated fail_reasons.
        """
        root_path = Path(root_path)
        result = FolderQCResult(status="PASS", analysis_id=analysis_id, customer="")

        # ── Step 1: date folder ──────────────────────────────────────────────
        date_path = root_path / date_str
        if not self._validate_date_folder(date_path, date_str):
            return self._fail(
                result,
                f"날짜 폴더 없음 또는 형식 불일치: '{date_str}'  "
                f"(허용 형식: YYYY-MM-DD, M-D-YYYY, M-D-YY)",
            )

        # ── Step 2: AnalysisID format ────────────────────────────────────────
        if not self._validate_analysis_id(analysis_id):
            return self._fail(
                result,
                f"AnalysisID 형식 불일치: '{analysis_id}'  (규칙: [A-Z]{{3}}[0-9]{{5}})",
            )

        # ── Step 3: customer code ────────────────────────────────────────────
        customer = self._get_customer(analysis_id)
        if customer is None:
            return self._fail(
                result,
                f"미지원 고객 코드: '{analysis_id[1]}'  "
                f"(지원 코드: {', '.join(self._cfg['customer_codes'].keys())})",
            )
        result.customer = customer

        # ── Step 4: AnalysisID folder exists ─────────────────────────────────
        analysis_path = date_path / analysis_id
        if not analysis_path.exists():
            return self._fail(
                result,
                f"AnalysisID 폴더를 찾을 수 없음: '{analysis_path}'",
            )

        # ── Step 5: detect cells and validate each ───────────────────────────
        if is_test_analysis_id(analysis_id):
            for cell_number, source_folder in find_test_source_folders(analysis_path):
                image_count = len(
                    find_test_images(
                        source_folder,
                        self._cfg["valid_image_extensions"],
                    )
                )
                cell_name = (
                    source_folder.name
                    if source_folder != analysis_path
                    else "single"
                )
                cell_result = CellResult(
                    cell_name=cell_name,
                    front_path=source_folder,
                    back_path=None,
                    front_count=image_count,
                    back_count=0,
                    status="PASS" if image_count > 0 else "FAIL",
                    fail_reasons=[] if image_count > 0 else ["TEST image files missing"],
                )
                result.cells.append(cell_result)
                if cell_result.status == "FAIL":
                    result.status = "FAIL"
                    result.fail_reasons.append(
                        f"[{cell_name}] TEST image files missing"
                    )
            return result

        cell_paths = self._detect_cells(analysis_path)
        expected_count: Optional[int] = self._cfg["image_count_rules"].get(analysis_id[1])

        for cell_name, cell_path in cell_paths:
            cell_result = self._validate_cell(cell_name, cell_path, expected_count)
            result.cells.append(cell_result)
            if cell_result.status == "FAIL":
                result.status = "FAIL"
                for reason in cell_result.fail_reasons:
                    result.fail_reasons.append(f"[{cell_name}] {reason}")

        logger.info(
            "FolderQC %s  customer=%s  cells=%d  → %s",
            analysis_id, result.customer, len(result.cells), result.status,
        )
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fail(self, result: FolderQCResult, reason: str) -> FolderQCResult:
        result.status = "FAIL"
        result.fail_reasons.append(reason)
        logger.warning("FolderQC FAIL — %s", reason)
        return result

    def _validate_date_folder(self, date_path: Path, date_str: str) -> bool:
        """Return True only if the folder exists AND name matches a valid date format."""
        if not date_path.exists():
            logger.debug("Date folder does not exist: %s", date_path)
            return False
        for fmt in self._cfg["date_formats"]:
            try:
                datetime.strptime(date_str, fmt)
                logger.debug("Date '%s' matched format '%s'", date_str, fmt)
                return True
            except ValueError:
                continue
        logger.debug("Date '%s' matched no known format", date_str)
        return False

    def _validate_analysis_id(self, analysis_id: str) -> bool:
        return bool(re.fullmatch(self._cfg["analysis_id_pattern"], analysis_id)) or is_test_analysis_id(analysis_id)

    def _get_customer(self, analysis_id: str) -> Optional[str]:
        if is_test_analysis_id(analysis_id):
            return "TEST"
        return self._cfg["customer_codes"].get(analysis_id[1])

    def _detect_cells(self, analysis_path: Path) -> list[tuple[str, Path]]:
        """
        Return list of (cell_name, cell_path) tuples.
        If no Cell #N subfolders found → single-cell mode: [(''single'', analysis_path)].
        """
        pattern = re.compile(self._cfg["cell_folder_pattern"], re.IGNORECASE)
        cell_dirs = sorted(
            [d for d in analysis_path.iterdir() if d.is_dir() and pattern.match(d.name)],
            key=lambda d: d.name.lower(),
        )
        if cell_dirs:
            logger.debug("Multi-cell mode: %s", [d.name for d in cell_dirs])
            return [(d.name, d) for d in cell_dirs]
        logger.debug("Single-cell mode: %s", analysis_path)
        return [("single", analysis_path)]

    def _find_front_back(
        self, cell_path: Path
    ) -> tuple[Optional[Path], Optional[Path]]:
        """Case-insensitive search for FRONT and BACK subfolders."""
        front_path: Optional[Path] = None
        back_path: Optional[Path] = None
        for d in cell_path.iterdir():
            if not d.is_dir():
                continue
            name_lower = d.name.lower()
            if name_lower == "front":
                front_path = d
            elif name_lower == "back":
                back_path = d
        return front_path, back_path

    def _count_images(self, folder_path: Path) -> int:
        extensions = set(self._cfg["valid_image_extensions"])
        count = sum(
            1 for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        logger.debug("Image count in '%s': %d", folder_path, count)
        return count

    def _validate_cell(
        self,
        cell_name: str,
        cell_path: Path,
        expected_count: Optional[int],
    ) -> CellResult:
        front_path, back_path = self._find_front_back(cell_path)
        front_count = self._count_images(front_path) if front_path else 0
        back_count = self._count_images(back_path) if back_path else 0

        cell_result = CellResult(
            cell_name=cell_name,
            front_path=front_path,
            back_path=back_path,
            front_count=front_count,
            back_count=back_count,
            status="PASS",
        )

        # FRONT / BACK existence
        if front_path is None:
            cell_result.status = "FAIL"
            cell_result.fail_reasons.append("FRONT 폴더 없음")
        if back_path is None:
            cell_result.status = "FAIL"
            cell_result.fail_reasons.append("BACK 폴더 없음")

        # Image count (skipped for S / T customers where expected_count is None)
        if expected_count is not None:
            if front_path and front_count != expected_count:
                cell_result.status = "FAIL"
                cell_result.fail_reasons.append(
                    f"FRONT 이미지 수 불일치: {front_count}장  (기준: {expected_count}장)"
                )
            if back_path and back_count != expected_count:
                cell_result.status = "FAIL"
                cell_result.fail_reasons.append(
                    f"BACK 이미지 수 불일치: {back_count}장  (기준: {expected_count}장)"
                )

        logger.info(
            "Cell '%s'  FRONT=%s(%d)  BACK=%s(%d)  → %s",
            cell_name,
            front_path.name if front_path else "MISSING",
            front_count,
            back_path.name if back_path else "MISSING",
            back_count,
            cell_result.status,
        )
        return cell_result
