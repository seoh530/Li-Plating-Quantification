from modules.utils.analysis_id import compile_cell_pattern
import logging
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import yaml

from modules.qc.crop_engine import CropEngine, CropResult
from modules.qc.folder_qc import FolderQC
from modules.utils.analysis_id import is_test_analysis_id
from modules.utils.image_io import atomic_imwrite_png, quiet_imread

logger = logging.getLogger(__name__)

REPORT_GRID_LAYOUTS: dict[str, tuple[int, int]] = {
    "HKMC": (3, 5),
    "FORD": (4, 4),
    "ESS": (2, 5),
    "VW": (2, 5),
}
DEFAULT_REPORT_GRID = (3, 5)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ProgressCallback = Callable[[str], None]
REPORT_SOURCE_PASS = "PASS"
REPORT_SOURCE_MASK = "MASK"
REPORT_SOURCES = (REPORT_SOURCE_PASS, REPORT_SOURCE_MASK)


@dataclass(frozen=True)
class ReportImage:
    path: Path | None
    layer_number: int


@dataclass
class ReportSideResult:
    face: str
    cell_name: str
    pass_folder: Path
    combined_folder: Path
    source: str
    crop_results: list[CropResult] = field(default_factory=list)
    combined_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReportRunResult:
    analysis_id: str
    analysis_path: Path
    customer: str
    layout: tuple[int, int]
    side_results: list[ReportSideResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def combined_paths(self) -> list[Path]:
        paths: list[Path] = []
        for side in self.side_results:
            paths.extend(side.combined_paths)
        return paths


class ReportCombiner:
    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        with open(self._config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._cfg = cfg
        self.root_path = Path(cfg["app"]["root_path"])
        folder_cfg = cfg["folder_qc"]
        self._date_formats = list(folder_cfg["date_formats"])
        self._cell_folder_re = compile_cell_pattern(folder_cfg.get("cell_folder_pattern", r"^cell\s*#\d+$"))
        self._image_count_rules = dict(folder_cfg.get("image_count_rules", {}))
        self._folder_qc = FolderQC(self._config_path)
        self._crop_engine = CropEngine(self._config_path)
        self._color_normalizer = None
        try:
            from modules.qc.color_normalizer import ColorNormalizer

            self._color_normalizer = ColorNormalizer(self._config_path)
        except Exception as exc:
            logger.warning("ReportCombiner: ColorNormalizer skipped: %s", exc)

    def list_date_folders(self) -> list[str]:
        if not self.root_path.is_dir():
            return []
        dated: list[tuple[datetime, str]] = []
        for path in self.root_path.iterdir():
            if not path.is_dir():
                continue
            parsed = self._parse_date_folder(path.name)
            if parsed is not None:
                dated.append((parsed, path.name))
        dated.sort(key=lambda item: item[0], reverse=True)
        return [name for _parsed, name in dated]

    def list_analysis_ids(self, date_name: str) -> list[str]:
        date_path = self.root_path / date_name
        if not date_path.is_dir():
            return []
        return sorted(path.name for path in date_path.iterdir() if path.is_dir())

    def list_cells(self, analysis_path: str | Path) -> list[tuple[str, Path]]:
        analysis_path = Path(analysis_path)
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

    def customer_for_analysis_id(self, analysis_id: str) -> str:
        return self._folder_qc.customer_for_analysis_id(analysis_id) or "UNKNOWN"

    def customer_code_for_analysis_id(self, analysis_id: str) -> str:
        return self._folder_qc.customer_code_for_analysis_id(analysis_id) or ""

    def default_layout_for_analysis_id(self, analysis_id: str) -> tuple[int, int]:
        return REPORT_GRID_LAYOUTS.get(
            self.customer_for_analysis_id(analysis_id),
            DEFAULT_REPORT_GRID,
        )

    def create_for_analysis(
        self,
        date_name: str,
        analysis_id: str,
        *,
        selected_cells: Iterable[str] | None = None,
        layout: tuple[int, int] | None = None,
        overwrite_pass: bool = False,
        source_kind: str = REPORT_SOURCE_PASS,
        progress_callback: ProgressCallback | None = None,
    ) -> ReportRunResult:
        analysis_path = self.root_path / date_name / analysis_id
        return self.create_for_analysis_path(
            analysis_path,
            analysis_id,
            selected_cells=selected_cells,
            layout=layout,
            overwrite_pass=overwrite_pass,
            source_kind=source_kind,
            progress_callback=progress_callback,
        )

    def create_for_analysis_path(
        self,
        analysis_path: str | Path,
        analysis_id: str,
        *,
        selected_cells: Iterable[str] | None = None,
        layout: tuple[int, int] | None = None,
        overwrite_pass: bool = False,
        source_kind: str = REPORT_SOURCE_PASS,
        progress_callback: ProgressCallback | None = None,
    ) -> ReportRunResult:
        analysis_path = Path(analysis_path)
        if not analysis_path.is_dir():
            raise FileNotFoundError(f"AnalysisID folder not found: {analysis_path}")

        source_kind = self._normalize_source_kind(source_kind)
        customer = self.customer_for_analysis_id(analysis_id)
        grid = layout or self.default_layout_for_analysis_id(analysis_id)
        result = ReportRunResult(
            analysis_id=analysis_id,
            analysis_path=analysis_path,
            customer=customer,
            layout=grid,
        )

        selected = set(selected_cells or [])
        cells = [
            (cell_name, cell_path)
            for cell_name, cell_path in self.list_cells(analysis_path)
            if not selected or cell_name in selected
        ]
        if not cells:
            raise ValueError("No cell folders selected.")

        for cell_name, cell_path in cells:
            for face in ("FRONT", "BACK"):
                side = self._create_for_side(
                    analysis_id,
                    cell_name,
                    cell_path,
                    face,
                    grid,
                    overwrite_pass=overwrite_pass,
                    source_kind=source_kind,
                    progress_callback=progress_callback,
                )
                result.side_results.append(side)
                result.warnings.extend(side.warnings)
        return result

    def existing_pass_outputs_for_analysis(
        self,
        date_name: str,
        analysis_id: str,
        *,
        selected_cells: Iterable[str] | None = None,
    ) -> list[Path]:
        return self.existing_pass_outputs_for_analysis_path(
            self.root_path / date_name / analysis_id,
            selected_cells=selected_cells,
        )

    def existing_pass_outputs_for_analysis_path(
        self,
        analysis_path: str | Path,
        *,
        selected_cells: Iterable[str] | None = None,
    ) -> list[Path]:
        analysis_path = Path(analysis_path)
        selected = set(selected_cells or [])
        outputs: list[Path] = []
        for cell_name, cell_path in self.list_cells(analysis_path):
            if selected and cell_name not in selected:
                continue
            for face in ("FRONT", "BACK"):
                pass_folder = cell_path / f"{face}_pass"
                outputs.extend(self._pass_pngs(pass_folder))
        return sorted(outputs, key=lambda path: str(path).lower())

    def create_for_paths(
        self,
        paths: Iterable[str | Path],
        output_base: str | Path,
        *,
        analysis_id: str = "REPORT",
        layout: tuple[int, int] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> ReportRunResult:
        grouped: dict[str, list[Path]] = {"FRONT": [], "BACK": []}
        image_paths: list[Path] = []
        for raw_path in paths:
            path = Path(raw_path)
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_paths.append(path)
            face = self._face_from_path(path)
            if face is not None:
                grouped[face].append(path)
        if not image_paths:
            raise ValueError("No image files were found in the dropped paths.")
        if not grouped["FRONT"] and not grouped["BACK"]:
            preview = "; ".join(str(path) for path in image_paths[:8])
            if len(image_paths) > 8:
                preview += f"; ... and {len(image_paths) - 8} more"
            raise ValueError(
                "Image files were found, but none could be classified as FRONT or BACK. "
                "Drop a FRONT/BACK folder or use paths containing FRONT/BACK.\n\n"
                f"Found image examples: {preview}"
            )

        output_base = Path(output_base)
        output_base.mkdir(parents=True, exist_ok=True)
        customer = self.customer_for_analysis_id(analysis_id)
        grid = layout or self.default_layout_for_analysis_id(analysis_id)
        run_result = ReportRunResult(
            analysis_id=analysis_id,
            analysis_path=output_base,
            customer=customer,
            layout=grid,
        )

        for face, face_paths in grouped.items():
            if not face_paths:
                continue
            pass_folder = output_base / f"{face}_pass"
            pass_folder.mkdir(parents=True, exist_ok=True)
            sorted_face_paths = sorted(face_paths, key=lambda p: p.name)
            crop_results = self._crop_report_images(
                sorted_face_paths,
                sorted_face_paths,
                self.customer_code_for_analysis_id(analysis_id),
                pass_folder,
                overwrite_pass=True,
            )
            failures = [
                crop
                for crop in crop_results
                if crop.output_path is None or not crop.output_path.exists()
            ]
            if failures:
                reasons = "; ".join(
                    f"{crop.image_path.name}: {' | '.join(crop.fail_reasons)}"
                    for crop in failures
                )
                raise RuntimeError(f"Crop failed for dropped {face} files: {reasons}")
            self._progress(
                progress_callback,
                f"{face}: CROP 완료 -> {pass_folder.name} 폴더 만들기 완료 ({len(crop_results)}장)",
            )
            images = [
                ReportImage(path=crop.output_path, layer_number=index)
                for index, crop in enumerate(crop_results, start=1)
                if crop.output_path is not None
            ]
            combined_folder = output_base / f"{face}_combined"
            combined_paths = self._combine_images(images, combined_folder, face, grid)
            self._progress(
                progress_callback,
                f"{face}: COMBINE 완료 -> {combined_folder.name} ({len(combined_paths)}개)",
            )
            run_result.side_results.append(
                ReportSideResult(
                    face=face,
                    cell_name="single",
                    pass_folder=pass_folder,
                    combined_folder=combined_folder,
                    source="crop",
                    crop_results=crop_results,
                    combined_paths=combined_paths,
                )
            )
        return run_result

    def _create_for_side(
        self,
        analysis_id: str,
        cell_name: str,
        cell_path: Path,
        face: str,
        layout: tuple[int, int],
        *,
        overwrite_pass: bool,
        source_kind: str,
        progress_callback: ProgressCallback | None = None,
    ) -> ReportSideResult:
        if source_kind == REPORT_SOURCE_MASK:
            return self._create_mask_for_side(
                analysis_id,
                cell_name,
                cell_path,
                face,
                layout,
                progress_callback=progress_callback,
            )
        source_folder = self._find_face_folder(cell_path, face)
        pass_folder = cell_path / f"{face}_pass"
        combined_folder = cell_path / f"{face}_combined"
        source_images = self._source_images(source_folder) if source_folder is not None else []
        side = ReportSideResult(
            face=face,
            cell_name=cell_name,
            pass_folder=pass_folder,
            combined_folder=combined_folder,
            source="crop",
        )

        if source_folder is None or not source_folder.is_dir():
            raise FileNotFoundError(f"{cell_name} {face} folder not found.")
        if not source_images:
            raise FileNotFoundError(f"{cell_name} {face} image files missing.")
        self._validate_source_count(analysis_id, face, source_images)

        crop_results = self._crop_report_images(
            source_images,
            source_images,
            self.customer_code_for_analysis_id(analysis_id),
            pass_folder,
            overwrite_pass=overwrite_pass,
        )
        side.crop_results = crop_results
        failures = [
            crop
            for crop in crop_results
            if crop.output_path is None or not crop.output_path.exists()
        ]
        if failures:
            reasons = "; ".join(
                f"{crop.image_path.name}: {' | '.join(crop.fail_reasons)}"
                for crop in failures
            )
            raise RuntimeError(f"Crop failed for {cell_name} {face}: {reasons}")
        self._progress(
            progress_callback,
            f"{cell_name} {face}: CROP 완료 -> {pass_folder.name} 폴더 만들기 완료 ({len(crop_results)}장)",
        )

        pass_images = self._pass_images(pass_folder, source_folder)
        if len(pass_images) != len(source_images):
            raise RuntimeError(
                f"{cell_name} {face} report image count mismatch after crop: "
                f"{len(pass_images)}/{len(source_images)}"
            )
        side.combined_paths = self._combine_images(pass_images, combined_folder, face, layout)
        self._progress(
            progress_callback,
            f"{cell_name} {face}: COMBINE 완료 -> {combined_folder.name} ({len(side.combined_paths)}개)",
        )
        return side

    def _create_mask_for_side(
        self,
        analysis_id: str,
        cell_name: str,
        cell_path: Path,
        face: str,
        layout: tuple[int, int],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> ReportSideResult:
        mask_folder = self._find_mask_folder(cell_path, face)
        combined_folder = cell_path / self._combined_folder_name(face, REPORT_SOURCE_MASK)
        side = ReportSideResult(
            face=face,
            cell_name=cell_name,
            pass_folder=mask_folder or cell_path / f"{face}_mask",
            combined_folder=combined_folder,
            source="mask",
        )
        if mask_folder is None:
            raise FileNotFoundError(f"{cell_name} {face}_mask folder not found.")
        mask_paths = self._mask_pngs(mask_folder)
        if not mask_paths:
            raise FileNotFoundError(f"{cell_name} {face}_mask PNG files missing.")
        # ML excludes the first and last source sheets; legacy exports may
        # still contain all sheets. Neither count is a partial/failed export.
        expected = self._expected_source_count(analysis_id)
        if expected is not None and len(mask_paths) not in {expected, max(1, expected - 2)}:
            raise ValueError(
                f"{face} mask count mismatch: {len(mask_paths)} "
                f"(expected: {expected} legacy or {max(1, expected - 2)} QC-filtered)"
            )

        source_folder = self._find_face_folder(cell_path, face)
        layer_index = self._source_layer_index(source_folder) if source_folder else {}
        images = [
            ReportImage(
                path=path,
                layer_number=layer_index.get(self._mask_source_stem(path), index),
            )
            for index, path in enumerate(mask_paths, start=1)
        ]
        side.combined_paths = self._combine_images(
            images,
            combined_folder,
            f"{face}_mask",
            layout,
        )
        self._progress(
            progress_callback,
            f"{cell_name} {face}: MASK combine complete -> {combined_folder.name} "
            f"({len(side.combined_paths)} file(s))",
        )
        return side

    def _progress(
        self,
        progress_callback: ProgressCallback | None,
        message: str,
    ) -> None:
        if progress_callback is not None:
            progress_callback(message)

    def _crop_report_images(
        self,
        all_source_images: list[Path],
        images_to_crop: list[Path],
        customer_code: str,
        pass_folder: Path,
        *,
        overwrite_pass: bool,
    ) -> list[CropResult]:
        pass_folder.mkdir(parents=True, exist_ok=True)
        source_index = {image_path: index for index, image_path in enumerate(all_source_images)}
        last_index = len(all_source_images) - 1
        results: list[CropResult] = []
        for image_path in images_to_crop:
            index = source_index.get(image_path, -1)
            output_path = pass_folder / f"{image_path.stem}.png"
            if output_path.exists() and not overwrite_pass:
                raise FileExistsError(f"Pass image already exists: {output_path}")
            results.append(
                self._crop_engine.process_image(
                    image_path=image_path,
                    customer_code=customer_code,
                    output_path=output_path,
                    skip_size_validation=index in {0, last_index},
                    image_postprocess=self._image_postprocess(),
                    force_save_failed=True,
                )
            )
        return results

    def _combine_images(
        self,
        images: list[ReportImage],
        output_folder: Path,
        face: str,
        layout: tuple[int, int],
    ) -> list[Path]:
        columns, rows = layout
        columns = max(1, int(columns))
        rows = max(1, int(rows))
        per_page = columns * rows
        if per_page <= 0:
            raise ValueError("Report grid must have at least one cell.")

        output_folder.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        # Render every page before replacing any existing report. A corrupt
        # later input must not erase the previous complete report.
        with tempfile.TemporaryDirectory(prefix=".report_", dir=output_folder) as stage:
            staged_paths: list[Path] = []
            for page_index, start in enumerate(range(0, len(images), per_page), start=1):
                page_images = images[start : start + per_page]
                canvas = self._render_page(page_images, columns, rows)
                name = f"{face}_combined_{page_index:02d}.png"
                staged_path = Path(stage) / name
                if not atomic_imwrite_png(staged_path, canvas, compression=1):
                    raise RuntimeError(f"Failed to write combined image: {name}")
                staged_paths.append(staged_path)
            for staged_path in staged_paths:
                output_path = output_folder / staged_path.name
                staged_path.replace(output_path)
                output_paths.append(output_path)
        for old_path in output_folder.glob(f"{face}_combined_*.png"):
            if old_path.is_file() and old_path not in output_paths:
                old_path.unlink()
        return output_paths

    def _render_page(self, images: list[ReportImage], columns: int, rows: int) -> np.ndarray:
        loaded: list[tuple[ReportImage, np.ndarray]] = []
        for report_image in images:
            if report_image.path is None:
                loaded.append((report_image, None))
                continue
            image = self._read_on_white(report_image.path)
            if image is None:
                raise RuntimeError(f"Failed to read pass image: {report_image.path}")
            loaded.append((report_image, image))

        if loaded:
            ratios = [
                image.shape[1] / max(image.shape[0], 1)
                for _report_image, image in loaded if image is not None
            ]
            target_ratio = float(np.median(ratios)) if ratios else 1.25
        else:
            target_ratio = 1.25

        if target_ratio >= 1.0:
            cell_w = 900
            cell_h = max(1, int(round(cell_w / target_ratio)))
        else:
            cell_h = 900
            cell_w = max(1, int(round(cell_h * target_ratio)))

        width = columns * cell_w
        height = rows * cell_h
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        for index, (report_image, image) in enumerate(loaded):
            row = index // columns
            col = index % columns
            x = col * cell_w
            y = row * cell_h
            max_w = cell_w
            max_h = cell_h
            if image is None:
                continue
            img_h, img_w = image.shape[:2]
            scale = min(max_w / img_w, max_h / img_h)
            resized = cv2.resize(
                image,
                (max(1, int(img_w * scale)), max(1, int(img_h * scale))),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
            )
            rx = x + (cell_w - resized.shape[1]) // 2
            ry = y + (max_h - resized.shape[0]) // 2
            canvas[ry : ry + resized.shape[0], rx : rx + resized.shape[1]] = resized
            label = str(report_image.layer_number)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness,
            )
            box_pad_x = 12
            box_pad_y = 8
            box_w = max(48, text_w + box_pad_x * 2)
            box_h = max(42, text_h + baseline + box_pad_y * 2)
            cv2.rectangle(
                canvas,
                (x, y),
                (min(x + box_w, width - 1), min(y + box_h, height - 1)),
                (255, 160, 40),
                thickness=cv2.FILLED,
            )
            cv2.putText(
                canvas,
                label,
                (x + box_pad_x, y + box_pad_y + text_h),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        return canvas

    def _clear_previous_combined(self, output_folder: Path, face: str) -> None:
        for path in output_folder.glob(f"{face}_combined_*.png"):
            if path.is_file():
                path.unlink()

    def _read_on_white(self, image_path: Path) -> np.ndarray | None:
        image = quiet_imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return None
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 3:
            return image
        bgr = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3].astype(np.float32) / 255.0
        white = np.full_like(bgr, 255.0)
        blended = bgr * alpha[:, :, None] + white * (1.0 - alpha[:, :, None])
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _pass_images(self, pass_folder: Path, source_folder: Path | None) -> list[ReportImage]:
        if not pass_folder.is_dir():
            return []
        layer_index = self._source_layer_index(source_folder) if source_folder else {}
        paths = self._pass_pngs(pass_folder)
        return [
            ReportImage(path=path, layer_number=layer_index.get(path.stem, index))
            for index, path in enumerate(paths, start=1)
        ]

    def _pass_pngs(self, pass_folder: Path) -> list[Path]:
        if not pass_folder.is_dir():
            return []
        return sorted(
            [
                path
                for path in pass_folder.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".png"
                and not path.name.endswith("_mask.png")
            ],
            key=lambda path: path.name,
        )

    def _mask_pngs(self, mask_folder: Path) -> list[Path]:
        if not mask_folder.is_dir():
            return []
        return sorted(
            [
                path
                for path in mask_folder.iterdir()
                if path.is_file()
                and path.suffix.lower() == ".png"
                and path.name.lower().endswith("_mask.png")
            ],
            key=lambda path: path.name,
        )

    def _find_mask_folder(self, cell_path: Path, face: str) -> Path | None:
        if not cell_path.is_dir():
            return None
        expected = f"{face}_MASK"
        for path in cell_path.iterdir():
            if path.is_dir() and path.name.upper() == expected:
                return path
        return None

    @staticmethod
    def _mask_source_stem(path: Path) -> str:
        stem = path.stem
        return stem[:-5] if stem.lower().endswith("_mask") else stem

    @staticmethod
    def _combined_folder_name(face: str, source_kind: str) -> str:
        if source_kind == REPORT_SOURCE_MASK:
            return f"{face}_mask_combined"
        return f"{face}_combined"

    @staticmethod
    def _normalize_source_kind(source_kind: str) -> str:
        normalized = str(source_kind or REPORT_SOURCE_PASS).strip().upper().lstrip("_")
        if normalized not in REPORT_SOURCES:
            raise ValueError(f"Unsupported report source: {source_kind}")
        return normalized

    def _source_layer_index(self, source_folder: Path | None) -> dict[str, int]:
        if source_folder is None:
            return {}
        return {
            path.stem: index
            for index, path in enumerate(self._source_images(source_folder), start=1)
        }

    def _source_images(self, folder_path: Path) -> list[Path]:
        if not folder_path.is_dir():
            return []
        return sorted(
            [
                path
                for path in folder_path.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda path: path.name,
        )

    def _find_face_folder(self, cell_path: Path, face: str) -> Path | None:
        if not cell_path.is_dir():
            return None
        for path in cell_path.iterdir():
            if path.is_dir() and self._is_source_face_folder(path.name, face):
                return path
        return None

    def _is_source_face_folder(self, folder_name: str, face: str) -> bool:
        normalized = folder_name.strip().upper()
        if normalized == face:
            return True
        if not normalized.startswith(face):
            return False
        output_tokens = ("PASS", "MASK", "COMBINED", "CROPPED", "REPORT")
        suffix = normalized[len(face) :]
        return not any(token in suffix for token in output_tokens)

    def _validate_source_count(
        self,
        analysis_id: str,
        face: str,
        source_images: list[Path],
    ) -> None:
        if is_test_analysis_id(analysis_id):
            return
        customer_code = self.customer_code_for_analysis_id(analysis_id)
        expected = self._image_count_rules.get(customer_code)
        if expected is None:
            return
        if len(source_images) != int(expected):
            raise ValueError(
                f"{face} image count mismatch: {len(source_images)} "
                f"(expected: {expected})"
            )

    def _expected_source_count(self, analysis_id: str) -> int | None:
        if is_test_analysis_id(analysis_id):
            return None
        customer_code = self.customer_code_for_analysis_id(analysis_id)
        expected = self._image_count_rules.get(customer_code)
        return int(expected) if expected is not None else None

    def _face_from_path(self, path: Path, *, allow_output_folders: bool = False) -> str | None:
        for part in reversed(path.parts):
            upper = part.upper()
            if self._path_part_matches_face(
                upper,
                "FRONT",
                allow_output_folders=allow_output_folders,
            ):
                return "FRONT"
            if self._path_part_matches_face(
                upper,
                "BACK",
                allow_output_folders=allow_output_folders,
            ):
                return "BACK"
        return None

    def _path_part_matches_face(
        self,
        value: str,
        face: str,
        *,
        allow_output_folders: bool = False,
    ) -> bool:
        if face not in value:
            return False
        if allow_output_folders and "_MASK" in value and "_COMBINED" not in value:
            return True
        if any(token in value for token in ("_PASS", "_MASK", "_COMBINED", "_REPORT")):
            return False
        return True

    def _parse_date_folder(self, folder_name: str) -> datetime | None:
        for fmt in self._date_formats:
            try:
                return datetime.strptime(folder_name, fmt)
            except ValueError:
                continue
        return None

    def _image_postprocess(self):
        return self._color_normalizer.normalize if self._color_normalizer is not None else None
