"""Manual combined sheets reuse QC PASS outputs and preserve excluded-layer slots."""

from pathlib import Path
from modules.report.report_combiner import ReportCombiner, ReportImage
from modules.report.report_data import image_paths, side_sources


def combine_analysis(
    combiner: ReportCombiner,
    analysis,
    selected_cells=None,
    layout=(3, 5),
    source_kind="PASS",
    progress=None,
):
    analysis = Path(analysis)
    outputs = []
    cols, rows = map(int, layout)
    if not (1 <= cols <= 12 and 1 <= rows <= 12):
        raise ValueError("Rows and columns must be between 1 and 12.")
    selected = set(selected_cells or [])
    pattern = combiner._cfg["folder_qc"].get("cell_folder_pattern", r"^cell\s*#\d+$")
    for cell, face, source, pass_dir, mask_dir, parent in side_sources(
        analysis, pattern
    ):
        if (
            selected
            and source.parent.name not in selected
            and source.name not in selected
            and not ("single" in selected and parent == analysis)
        ):
            continue
        originals = image_paths(source)
        if not originals:
            raise ValueError(f"Cell #{cell} {face}: no source images.")
        pass_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, path in enumerate(originals):
            target = pass_dir / (path.stem + ".png")
            if source_kind != "MASK" and not target.exists():
                crop = combiner._crop_engine.process_image(
                    path,
                    combiner.customer_code_for_analysis_id(analysis.name),
                    target,
                    skip_size_validation=i in {0, len(originals) - 1},
                    image_postprocess=combiner._image_postprocess(),
                )
                if crop.output_path is None or not target.exists():
                    raise ValueError(
                        f"{path.name}: Crop failed; resolve QC Hold first."
                    )
            paths.append(ReportImage(target, i + 1))
        if source_kind != "MASK":
            outputs += combiner._combine_images(
                paths, parent / f"{face}_combined", face, (cols, rows)
            )
        if source_kind in {"MASK", "BOTH"}:
            masks = []
            for i, path in enumerate(originals):
                excluded = len(originals) > 1 and i in {0, len(originals) - 1}
                target = mask_dir / (path.stem + "_mask.png")
                if not excluded and not target.is_file():
                    raise ValueError(
                        f"{face} layer {i + 1}: mask missing. Run ML first."
                    )
                masks.append(ReportImage(None if excluded else target, i + 1))
            outputs += combiner._combine_images(
                masks, parent / f"{face}_mask_combined", f"{face}_mask", (cols, rows)
            )
        if progress:
            progress(f"Cell #{cell} {face}: combined images created.")
    if not outputs:
        raise ValueError("No matching cell / face images found.")
    return outputs
