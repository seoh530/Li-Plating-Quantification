import logging
import re
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


@dataclass
class CellResult:
    cell_name: str
    front_path: Optional[Path]
    back_path: Optional[Path]
    front_count: int
    back_count: int
    status: str  # "PASS" | "WARNING" | "FAIL"
    fail_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)


@dataclass
class FolderQCResult:
    status: str  # "PASS" | "WARNING" | "FAIL"
    analysis_id: str
    customer: str
    cells: list[CellResult] = field(default_factory=list)
    fail_reasons: list[str] = field(default_factory=list)
    warning_reasons: list[str] = field(default_factory=list)


class FolderQC:
    """
    Validate PhotoJig capture identity and electrode counts.

    Blocking failures are limited to DATE / AnalysisID validation and missing
    image sources. Electrode count mismatches are returned as WARNING so the
    workflow can continue while surfacing the mismatch in Summary.
    """

    def __init__(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._cfg: dict = cfg["folder_qc"]

    def validate(
        self,
        root_path: str | Path,
        date_str: str,
        analysis_id: str,
    ) -> FolderQCResult:
        root_path = Path(root_path)
        result = FolderQCResult(status="PASS", analysis_id=analysis_id, customer="")

        date_path = root_path / date_str
        if not self._validate_date_folder(date_path, date_str):
            return self._fail(
                result,
                f"Date folder missing or invalid: '{date_str}' "
                "(allowed: YYYY-MM-DD, M-D-YYYY, M-D-YY)",
            )

        if not self._validate_analysis_id(analysis_id):
            return self._fail(
                result,
                f"AnalysisID invalid: '{analysis_id}' (rule: [A-Z]{{3}}[0-9]{{5}})",
            )

        customer = self._get_customer(analysis_id)
        if customer is None:
            return self._fail(
                result,
                f"Unsupported customer code: '{analysis_id[1]}' "
                f"(supported: {', '.join(self._cfg['customer_codes'].keys())})",
            )
        result.customer = customer

        analysis_path = date_path / analysis_id
        if not analysis_path.exists():
            return self._fail(
                result,
                f"AnalysisID folder not found: '{analysis_path}'",
            )

        if is_test_analysis_id(analysis_id):
            self._validate_test_analysis(result, analysis_path)
            return result

        expected_count: Optional[int] = self._cfg["image_count_rules"].get(analysis_id[1])
        for cell_name, cell_path in self._detect_cells(analysis_path):
            cell_result = self._validate_cell(
                cell_name,
                cell_path,
                expected_count,
                date_str=date_str,
                analysis_id=analysis_id,
            )
            result.cells.append(cell_result)
            if cell_result.status == "FAIL":
                result.status = "FAIL"
                result.fail_reasons.extend(
                    f"[{cell_name}] {reason}" for reason in cell_result.fail_reasons
                )
            else:
                if cell_result.status == "WARNING" and result.status == "PASS":
                    result.status = "WARNING"
                result.warning_reasons.extend(
                    f"[{cell_name}] {reason}" for reason in cell_result.warning_reasons
                )

        logger.info(
            "FolderQC %s  customer=%s  cells=%d  -> %s",
            analysis_id,
            result.customer,
            len(result.cells),
            result.status,
        )
        return result

    def _validate_test_analysis(self, result: FolderQCResult, analysis_path: Path) -> None:
        for _cell_number, source_folder in find_test_source_folders(analysis_path):
            image_count = len(
                find_test_images(
                    source_folder,
                    self._cfg["valid_image_extensions"],
                )
            )
            cell_name = source_folder.name if source_folder != analysis_path else "single"
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
                result.fail_reasons.append(f"[{cell_name}] TEST image files missing")

    def _fail(self, result: FolderQCResult, reason: str) -> FolderQCResult:
        result.status = "FAIL"
        result.fail_reasons.append(reason)
        logger.warning("FolderQC FAIL - %s", reason)
        return result

    def _validate_date_folder(self, date_path: Path, date_str: str) -> bool:
        if not date_path.exists():
            logger.debug("Date folder does not exist: %s", date_path)
            return False
        return self._parse_date_text(date_str) is not None

    def _validate_analysis_id(self, analysis_id: str) -> bool:
        return bool(re.fullmatch(self._cfg["analysis_id_pattern"], analysis_id)) or is_test_analysis_id(analysis_id)

    def _get_customer(self, analysis_id: str) -> Optional[str]:
        if is_test_analysis_id(analysis_id):
            return "TEST"
        return self._cfg["customer_codes"].get(analysis_id[1])

    def _detect_cells(self, analysis_path: Path) -> list[tuple[str, Path]]:
        pattern = re.compile(self._cfg["cell_folder_pattern"], re.IGNORECASE)
        cell_dirs = sorted(
            [path for path in analysis_path.iterdir() if path.is_dir() and pattern.match(path.name)],
            key=lambda path: path.name.lower(),
        )
        if cell_dirs:
            logger.debug("Multi-cell mode: %s", [path.name for path in cell_dirs])
            return [(path.name, path) for path in cell_dirs]
        logger.debug("Single-cell mode: %s", analysis_path)
        return [("single", analysis_path)]

    def _find_front_back(
        self, cell_path: Path
    ) -> tuple[Optional[Path], Optional[Path]]:
        front_path: Optional[Path] = None
        back_path: Optional[Path] = None
        for path in cell_path.iterdir():
            if not path.is_dir():
                continue
            name_lower = path.name.lower()
            if name_lower == "front":
                front_path = path
            elif name_lower == "back":
                back_path = path
        return front_path, back_path

    def _image_paths(self, folder_path: Path) -> list[Path]:
        extensions = set(self._cfg["valid_image_extensions"])
        return sorted(
            [
                path
                for path in folder_path.iterdir()
                if path.is_file() and path.suffix.lower() in extensions
            ],
            key=lambda path: path.name.lower(),
        )

    def _count_images(self, folder_path: Path) -> int:
        count = len(self._image_paths(folder_path))
        logger.debug("Image count in '%s': %d", folder_path, count)
        return count

    def _validate_cell(
        self,
        cell_name: str,
        cell_path: Path,
        expected_count: Optional[int],
        *,
        date_str: str,
        analysis_id: str,
    ) -> CellResult:
        front_path, back_path = self._find_front_back(cell_path)
        direct_source_path = cell_path if front_path is None and back_path is None else None
        front_count = self._count_images(front_path) if front_path else (
            self._count_images(direct_source_path) if direct_source_path else 0
        )
        back_count = self._count_images(back_path) if back_path else 0

        cell_result = CellResult(
            cell_name=cell_name,
            front_path=front_path or direct_source_path,
            back_path=back_path,
            front_count=front_count,
            back_count=back_count,
            status="PASS",
        )

        source_folders: list[tuple[str, Path, int]] = []
        if front_path is not None:
            source_folders.append(("FRONT", front_path, front_count))
        if back_path is not None:
            source_folders.append(("BACK", back_path, back_count))
        if direct_source_path is not None:
            source_folders.append(("", direct_source_path, front_count))

        if not source_folders:
            cell_result.status = "FAIL"
            cell_result.fail_reasons.append("No image source folder found")
            return cell_result

        for face_name, folder_path, _count in source_folders:
            for image_path in self._image_paths(folder_path):
                reason = self._filename_fail_reason(
                    image_path,
                    date_str=date_str,
                    analysis_id=analysis_id,
                )
                if reason:
                    cell_result.status = "FAIL"
                    context = f"{face_name} " if face_name else ""
                    cell_result.fail_reasons.append(f"{context}{image_path.name}: {reason}")

        if expected_count is not None:
            for face_name, _folder_path, count in source_folders:
                if count == expected_count:
                    continue
                if cell_result.status != "FAIL":
                    cell_result.status = "WARNING"
                label = f"{face_name} " if face_name else ""
                cell_result.warning_reasons.append(
                    f"Warning: {label}Expected {expected_count} electrodes, found {count}"
                )

        logger.info(
            "Cell '%s'  FRONT=%s(%d)  BACK=%s(%d)  -> %s",
            cell_name,
            (front_path or direct_source_path).name if (front_path or direct_source_path) else "MISSING",
            front_count,
            back_path.name if back_path else "MISSING",
            back_count,
            cell_result.status,
        )
        return cell_result

    def _filename_fail_reason(
        self,
        image_path: Path,
        *,
        date_str: str,
        analysis_id: str,
    ) -> str | None:
        parts = [part.strip() for part in image_path.stem.split(" - ")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            return "filename must include DATE and ANALYSIS ID"

        file_date = self._parse_date_text(parts[0])
        expected_date = self._parse_date_text(date_str)
        if file_date is None:
            return f"invalid DATE: '{parts[0]}'"
        if expected_date is not None and file_date != expected_date:
            return f"DATE mismatch: '{parts[0]}' expected '{date_str}'"

        file_analysis_id = parts[1]
        if not self._validate_analysis_id(file_analysis_id):
            return f"invalid ANALYSIS ID: '{file_analysis_id}'"
        if file_analysis_id != analysis_id:
            return f"ANALYSIS ID mismatch: '{file_analysis_id}' expected '{analysis_id}'"
        return None

    def _parse_date_text(self, value: str):
        for fmt in self._cfg["date_formats"]:
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None
