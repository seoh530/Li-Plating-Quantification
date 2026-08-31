import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from modules.qc.crop_engine import CropEngine, CropResult
from modules.qc.folder_qc import FolderQC, FolderQCResult
from modules.qc.retake_manager import RetakeItem, RetakeManager
from modules.utils.analysis_id import (
    build_test_output_name,
    ensure_test_output_folders,
    find_test_images,
    find_test_source_folders,
    get_test_output_folders,
    is_test_analysis_id,
)

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class QCCropBatch:
    folder_path: Path
    customer_code: str
    output_folder: Path
    crop_results: list[CropResult]
    output_paths: dict[Path, Path] | None = None
    skip_size_validation_for_all: bool = False


@dataclass
class QCPipelineResult:
    analysis_id: str
    analysis_path: Path
    date_name: str
    customer_code: str
    folder_result: FolderQCResult
    crop_batches: list[QCCropBatch] = field(default_factory=list)
    failed_items: list[dict[str, object]] = field(default_factory=list)

    @property
    def crop_done(self) -> bool:
        return bool(self.crop_batches) or self.folder_result.status == "PASS"


class QCPipeline:
    """Run FolderQC and CropEngine for an AnalysisID without GUI dependencies."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        folder_cfg = cfg.get("folder_qc", {}) or {}
        self._cell_folder_re = re.compile(
            folder_cfg.get("cell_folder_pattern", r"^cell\s*#\d+$"),
            re.IGNORECASE,
        )
        self._folder_qc = FolderQC(self._config_path)
        self._crop_engine_instance: CropEngine | None = None
        self._retake_manager_instance: RetakeManager | None = None

    def validate_folder(
        self,
        *,
        root_path: str | Path,
        date_name: str,
        analysis_id: str,
    ) -> FolderQCResult:
        return self._folder_qc.validate(
            root_path=root_path,
            date_str=date_name,
            analysis_id=analysis_id,
        )

    def run_analysis(
        self,
        *,
        root_path: str | Path,
        date_name: str,
        analysis_id: str,
        analysis_path: str | Path | None = None,
        force_folder_qc: bool = False,
        clear_pass_outputs: bool = False,
    ) -> QCPipelineResult:
        resolved_analysis_path = (
            Path(analysis_path) if analysis_path is not None else Path(root_path) / date_name / analysis_id
        )
        customer_code = self.customer_code_for_analysis(analysis_id)
        if is_test_analysis_id(analysis_id):
            ensure_test_output_folders(resolved_analysis_path)

        folder_result = self.validate_folder(
            root_path=root_path,
            date_name=date_name,
            analysis_id=analysis_id,
        )
        pipeline_result = QCPipelineResult(
            analysis_id=analysis_id,
            analysis_path=resolved_analysis_path,
            date_name=date_name,
            customer_code=customer_code,
            folder_result=folder_result,
        )
        if folder_result.status != "PASS" and not force_folder_qc:
            return pipeline_result

        if is_test_analysis_id(analysis_id):
            self._run_test_analysis(
                pipeline_result,
                force_folder_qc=force_folder_qc,
                clear_pass_outputs=clear_pass_outputs,
            )
            return pipeline_result

        if folder_result.status == "PASS":
            source_folders = [
                (face_name, folder_path)
                for cell in folder_result.cells
                for face_name, folder_path in (
                    ("FRONT", cell.front_path),
                    ("BACK", cell.back_path),
                )
                if folder_path is not None
            ]
        else:
            source_folders = [
                (face_name, folder_path)
                for _cell_name, face_name, folder_path in self._forced_source_folders(
                    resolved_analysis_path
                )
            ]

        for face_name, folder_path in source_folders:
            output_folder = folder_path.parent / f"{face_name}_pass"
            if clear_pass_outputs:
                self.clear_pass_outputs(output_folder)
            crop_results = self._crop_engine.process_folder(
                folder_path=folder_path,
                customer_code=customer_code,
                output_folder=output_folder,
                image_postprocess=None,
            )
            batch = QCCropBatch(
                folder_path=folder_path,
                customer_code=customer_code,
                output_folder=output_folder,
                crop_results=crop_results,
            )
            pipeline_result.crop_batches.append(batch)
            pipeline_result.failed_items.extend(
                self._failed_items_for_batch(
                    batch,
                    skip_size_validation=False,
                )
            )

        return pipeline_result

    def force_crop_failed_item(self, failed_item: dict[str, object], *, force: bool) -> CropResult:
        result = self._crop_engine.process_image(
            image_path=Path(failed_item["image_path"]),
            customer_code=str(failed_item["customer_code"]),
            output_path=Path(failed_item["output_path"]),
            skip_size_validation=bool(failed_item["skip_size_validation"]),
            image_postprocess=None,
            force_save_failed=force,
        )
        if result.status != "PASS":
            logger.warning(
                "QC NG action failed: %s / %s",
                failed_item["image_path"],
                " | ".join(result.fail_reasons),
            )
        return result

    def clear_retake_items(self) -> None:
        self._retake_manager.clear()

    def remove_retake_by_analysis(self, analysis_path: Path) -> None:
        self._retake_manager.remove_by_analysis(analysis_path)

    def load_retake_batches(self, batches: list[QCCropBatch]) -> None:
        for batch in batches:
            self._retake_manager.load(
                batch.crop_results,
                batch.folder_path,
                batch.customer_code,
                batch.output_folder,
                output_paths=batch.output_paths,
                skip_size_validation_for_all=batch.skip_size_validation_for_all,
            )

    def pending_retake_items(self) -> list[RetakeItem]:
        return self._retake_manager.get_pending_items()

    def retry_retake_item(self, item: RetakeItem) -> CropResult:
        return self._retake_manager.retry_image(item)

    def force_crop_retake_item(self, item: RetakeItem) -> CropResult:
        return self._retake_manager.force_crop_image(item)

    def _run_test_analysis(
        self,
        pipeline_result: QCPipelineResult,
        *,
        force_folder_qc: bool,
        clear_pass_outputs: bool,
    ) -> None:
        folder_result = pipeline_result.folder_result
        analysis_path = pipeline_result.analysis_path
        if folder_result.status == "PASS":
            source_folders = [
                cell.front_path
                for cell in folder_result.cells
                if cell.front_path is not None
            ]
        elif force_folder_qc:
            source_folders = [
                source_path
                for _cell, source_path in find_test_source_folders(analysis_path)
            ]
        else:
            source_folders = []

        for source_folder in source_folders:
            output_folder, mask_folder = get_test_output_folders(
                analysis_path,
                source_folder,
            )
            output_folder.mkdir(parents=True, exist_ok=True)
            mask_folder.mkdir(parents=True, exist_ok=True)
            if clear_pass_outputs:
                self.clear_pass_outputs(output_folder)

            crop_results: list[CropResult] = []
            output_paths: dict[Path, Path] = {}
            for image_path in find_test_images(source_folder, _IMAGE_EXTENSIONS):
                output_path = output_folder / build_test_output_name(source_folder, image_path)
                output_paths[image_path] = output_path
                crop_results.append(
                    self._crop_engine.process_image(
                        image_path=image_path,
                        customer_code=pipeline_result.customer_code,
                        output_path=output_path,
                        skip_size_validation=True,
                        image_postprocess=None,
                    )
                )
            batch = QCCropBatch(
                folder_path=source_folder,
                customer_code=pipeline_result.customer_code,
                output_folder=output_folder,
                crop_results=crop_results,
                output_paths=output_paths,
                skip_size_validation_for_all=True,
            )
            pipeline_result.crop_batches.append(batch)
            pipeline_result.failed_items.extend(
                self._failed_items_for_batch(
                    batch,
                    skip_size_validation=True,
                )
            )

    def _failed_items_for_batch(
        self,
        batch: QCCropBatch,
        *,
        skip_size_validation: bool,
    ) -> list[dict[str, object]]:
        failures: list[dict[str, object]] = []
        first_path = batch.crop_results[0].image_path if batch.crop_results else None
        last_path = batch.crop_results[-1].image_path if len(batch.crop_results) > 1 else None
        for result in batch.crop_results:
            if result.status != "FAIL":
                continue
            item_skip_size = skip_size_validation or (
                result.image_path == first_path or result.image_path == last_path
            )
            output_path = (
                batch.output_paths.get(result.image_path)
                if batch.output_paths is not None
                else batch.output_folder / f"{result.image_path.stem}.png"
            )
            failures.append(
                {
                    "id": str(result.image_path),
                    "image_path": result.image_path,
                    "output_path": output_path,
                    "fail_reasons": list(result.fail_reasons),
                    "customer_code": batch.customer_code,
                    "skip_size_validation": item_skip_size,
                    "image_postprocess": None,
                }
            )
        return failures

    def _forced_cell_paths(self, analysis_path: Path) -> list[tuple[str, Path]]:
        if not analysis_path.is_dir():
            return []
        cell_dirs = sorted(
            [
                path
                for path in analysis_path.iterdir()
                if path.is_dir() and self._cell_folder_re.match(path.name)
            ],
            key=lambda path: path.name.lower(),
        )
        if cell_dirs:
            return [(path.name, path) for path in cell_dirs]
        return [("single", analysis_path)]

    def _forced_source_folders(self, analysis_path: Path) -> list[tuple[str, str, Path]]:
        sources: list[tuple[str, str, Path]] = []
        for cell_name, cell_path in self._forced_cell_paths(analysis_path):
            if not cell_path.is_dir():
                continue
            face_paths: dict[str, Path] = {}
            for path in cell_path.iterdir():
                if path.is_dir() and path.name.upper() in {"FRONT", "BACK"}:
                    face_paths[path.name.upper()] = path
            for face_name in ("FRONT", "BACK"):
                folder_path = face_paths.get(face_name)
                if folder_path is not None:
                    sources.append((cell_name, face_name, folder_path))
        return sources

    @property
    def _crop_engine(self) -> CropEngine:
        if self._crop_engine_instance is None:
            self._crop_engine_instance = CropEngine(self._config_path)
        return self._crop_engine_instance

    @property
    def _retake_manager(self) -> RetakeManager:
        if self._retake_manager_instance is None:
            self._retake_manager_instance = RetakeManager(self._config_path)
        return self._retake_manager_instance

    @staticmethod
    def clear_pass_outputs(output_folder: Path) -> None:
        if not output_folder.is_dir():
            return
        for path in output_folder.iterdir():
            if path.is_file() and path.suffix.lower() == ".png":
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Failed to remove stale pass output: %s", path, exc_info=True)

    @staticmethod
    def customer_code_for_analysis(analysis_id: str) -> str:
        return "T" if is_test_analysis_id(analysis_id) else analysis_id[1]
