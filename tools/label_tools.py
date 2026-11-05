import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.utils.image_io import atomic_write_text, quiet_imread
from modules.ml.developer_training import annotation_frame
from modules.ml.feature_extractor import FeatureExtractor

DEFAULT_LABELED_ROOT = PROJECT_ROOT / "data" / "labeled"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_HEADER = [
    "filename",
    "x",
    "y",
    "label",
    *FeatureExtractor.ENHANCED_FEATURE_COLUMNS,
]

idx = 0
scale = 1.0
offset_x = 0
offset_y = 0
dragging = False
start_drag = (0, 0)
current_label = "b"

label_colors = {
    "b": (200, 200, 200),
    "l": (0, 0, 255),
    "n": (255, 0, 255),
}

click_points = []
saved_points = set()
pending_rows = []
action_history = []

mouse_x = 0
mouse_y = 0

BRUSH_RADIUS = 4
BRUSH_POINT_COUNT = 10
window_name = "Label Points (b/l/n)"
output_csv: Path | None = None
_label_source_image = None
_label_extractor = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label cropped/QC-ready images and save a CSV named from the "
            "data/labeled/<AnalysisID>/<Cell #N>/<FRONT|BACK> folder structure."
        )
    )
    parser.add_argument(
        "image_folder",
        nargs="?",
        default=None,
        help=(
            "Image folder to label. Example: "
            "data/labeled/AHG11111/Cell #1/FRONT. "
            "If omitted, labelable FRONT/BACK folders under data/labeled/ are auto-detected."
        ),
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Override output CSV path. Otherwise it is generated automatically.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing CSV instead of starting a fresh file.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly replace an existing label CSV. Cannot be used with --append.",
    )
    parser.add_argument(
        "--config",
        default="config/thresholds.yaml",
        help=(
            "Configuration used for automatic enhanced-feature CSV saving."
        ),
    )
    return parser.parse_args()


def resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return PROJECT_ROOT / path


def is_analysis_id(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{3}\d{5}|TEST(?:\d{2})?", value.upper()))


def cell_number_from_parts(parts: tuple[str, ...]) -> str | None:
    for part in reversed(parts):
        match = re.fullmatch(r"cell\s*#\s*(\d+)", part, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def face_from_folder(path: Path) -> str | None:
    name = path.name.upper()
    if name in {"FRONT", "FRONT_PASS"}:
        return "F"
    if name in {"BACK", "BACK_PASS"}:
        return "B"
    return None


def find_analysis_root(path: Path) -> Path | None:
    for candidate in [path, *path.parents]:
        if is_analysis_id(candidate.name):
            return candidate
    return None


def default_output_csv_for_image_folder(image_folder: Path) -> Path:
    image_folder = image_folder.resolve()
    analysis_root = find_analysis_root(image_folder)
    face = face_from_folder(image_folder)

    if analysis_root is None or face is None:
        return DEFAULT_LABELED_ROOT / "enhanced_spectrum_data.csv"

    relative_parts = image_folder.relative_to(analysis_root).parts
    cell_number = cell_number_from_parts(relative_parts)
    if cell_number is None:
        csv_name = f"{analysis_root.name}_{face}.csv"
    else:
        csv_name = f"{analysis_root.name}_{cell_number}_{face}.csv"
    return analysis_root / ("enhanced_" + csv_name)


def folder_contains_images(folder_path: Path) -> bool:
    if not folder_path.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in folder_path.iterdir()
    )


def discover_label_folders(root: Path = DEFAULT_LABELED_ROOT) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_dir()
        and face_from_folder(path) is not None
        and find_analysis_root(path) is not None
        and folder_contains_images(path)
    )


def label_folder_is_completed(folder_path: Path) -> bool:
    return default_output_csv_for_image_folder(folder_path).exists()


def choose_label_folder(candidates: list[Path]) -> Path:
    if not candidates:
        raise RuntimeError(
            "No label image folders found under data/labeled/. "
            "Expected folders like data/labeled/CHG28282/Cell #1/FRONT with images."
        )
    if len(candidates) == 1:
        selected = candidates[0]
        print(f"Auto-selected image folder: {selected}")
        return selected

    print("Label image folders found under data/labeled/:")
    for index, folder_path in enumerate(candidates, start=1):
        try:
            folder_label = folder_path.relative_to(DEFAULT_LABELED_ROOT)
        except ValueError:
            folder_label = folder_path
        csv_path = default_output_csv_for_image_folder(folder_path)
        status = "completed " if label_folder_is_completed(folder_path) else "          "
        print(f"  {status}{index:>2}. {folder_label}  ->  {csv_path.name}")

    while True:
        raw = input(f"Select folder [1-{len(candidates)}]: ").strip()
        if not raw:
            raw = "1"
        try:
            selected_index = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= selected_index <= len(candidates):
            return candidates[selected_index - 1]
        print("Enter a number from the list.")


def resolve_image_folder(image_folder_arg: str | None) -> Path:
    if image_folder_arg:
        image_folder = Path(image_folder_arg)
        if not image_folder.is_absolute():
            image_folder = PROJECT_ROOT / image_folder
        return image_folder
    return choose_label_folder(discover_label_folders())


def image_label_for_csv(image_path: Path, csv_path: Path) -> str:
    try:
        return image_path.resolve().relative_to(csv_path.parent.resolve()).as_posix()
    except ValueError:
        return image_path.name


def initialize_output_csv(csv_path: Path, append: bool, overwrite: bool = False) -> None:
    if append and overwrite:
        raise RuntimeError("--append and --overwrite cannot be used together.")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if append and csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            existing_header = next(csv.reader(fh), [])
        if existing_header != CSV_HEADER:
            raise RuntimeError(
                f"Existing CSV uses a legacy or incompatible header. Start a new enhanced CSV, or migrate it once with enhance_training_features.py: {csv_path}"
            )
        return
    if csv_path.exists() and not overwrite:
        raise RuntimeError(
            f"Label CSV already exists: {csv_path}. Use --append to add labels "
            "or --overwrite to explicitly replace it."
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)


def collect_images(image_folder: Path) -> list[Path]:
    if not image_folder.is_dir():
        raise RuntimeError(f"Image folder does not exist: {image_folder}")
    return sorted(
        path
        for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def flush_pending_rows():
    global pending_rows
    if not pending_rows:
        return
    if output_csv is None:
        raise RuntimeError("output_csv is not initialized")

    frame = annotation_frame(
        _label_source_image, [(row[1], row[2], row[3]) for row in pending_rows],
        pending_rows[0][0], _label_extractor,
    )
    existing = pd.read_csv(output_csv)
    merged = frame if existing.empty else pd.concat([existing, frame], ignore_index=True)
    merged = merged.drop_duplicates(['filename', 'x', 'y'], keep='last')
    atomic_write_text(output_csv, merged.loc[:, CSV_HEADER].to_csv(index=False, float_format='%.17g'))

    pending_rows = []


def get_label_counts():
    counts = {"b": 0, "l": 0, "n": 0}
    for _, _, lab in click_points:
        if lab in counts:
            counts[lab] += 1
    return counts


def add_point(
    image_shape: tuple[int, int],
    filename,
    xo,
    yo,
    label,
    stamp_records,
):
    height, width = image_shape
    if not (0 <= xo < width and 0 <= yo < height):
        return

    if _label_source_image is not None and _label_source_image.shape[2] == 4 and _label_source_image[yo, xo, 3] == 0:
        return
    key = (filename, xo, yo, label)
    if key in saved_points:
        return

    row = [
        filename,
        xo,
        yo,
        label,
    ]

    pending_rows.append(row)
    click_points.append((xo, yo, label))
    saved_points.add(key)
    stamp_records.append((xo, yo, label, row, key))


def stamp_points(
    image_shape: tuple[int, int],
    filename,
    center_x,
    center_y,
    label,
):
    stamp_records = []
    add_point(
        image_shape,
        filename,
        center_x,
        center_y,
        label,
        stamp_records,
    )

    for _ in range(BRUSH_POINT_COUNT):
        dx = np.random.randint(-BRUSH_RADIUS, BRUSH_RADIUS + 1)
        dy = np.random.randint(-BRUSH_RADIUS, BRUSH_RADIUS + 1)
        if dx * dx + dy * dy > BRUSH_RADIUS * BRUSH_RADIUS:
            continue
        add_point(
            image_shape,
            filename,
            center_x + dx,
            center_y + dy,
            label,
            stamp_records,
        )

    if stamp_records:
        action_history.append(stamp_records)


def undo_last_action():
    global pending_rows
    if not action_history:
        return

    last_stamp = action_history.pop()
    remove_keys = {record[4] for record in last_stamp}
    remove_points = {(x, y, lab) for x, y, lab, _row, _key in last_stamp}
    click_points[:] = [pt for pt in click_points if pt not in remove_points]
    for key in remove_keys:
        saved_points.discard(key)

    remove_rows = {tuple(row) for _x, _y, _lab, row, _key in last_stamp}
    pending_rows = [row for row in pending_rows if tuple(row) not in remove_rows]


def get_display_image(img, scale_value, ox, oy):
    h, w = img.shape[:2]
    scaled_w = max(1, int(w * scale_value))
    scaled_h = max(1, int(h * scale_value))
    scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    view_w = w
    view_h = h
    max_ox = max(0, scaled_w - view_w)
    max_oy = max(0, scaled_h - view_h)
    ox = max(0, min(ox, max_ox))
    oy = max(0, min(oy, max_oy))

    disp = scaled_img[oy:oy + view_h, ox:ox + view_w].copy()
    if disp.shape[0] < view_h or disp.shape[1] < view_w:
        canvas = np.zeros((view_h, view_w, 3), dtype=np.uint8)
        canvas[:disp.shape[0], :disp.shape[1]] = disp
        disp = canvas

    point_radius = max(2, int(3 * scale_value))
    for x, y, lab in click_points:
        dx = int(x * scale_value - ox)
        dy = int(y * scale_value - oy)
        if 0 <= dx < view_w and 0 <= dy < view_h:
            cv2.circle(disp, (dx, dy), point_radius, label_colors[lab], -1)

    counts = get_label_counts()
    cv2.putText(
        disp,
        f"[{current_label.upper()}] b:background, l:lithium, n:nonactive",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (50, 50, 50),
        2,
    )
    cv2.putText(
        disp,
        f"L: {counts['l']}   N: {counts['n']}   B: {counts['b']}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
    )
    cv2.putText(
        disp,
        "Hold b/l/n to paint | U=undo | right-drag=pan | wheel=zoom | SPACE=next | ESC=exit",
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (60, 60, 60),
        1,
    )
    return disp, ox, oy


def click_event(event, x, y, flags, param):
    global dragging, start_drag, offset_x, offset_y, scale, mouse_x, mouse_y

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
        if dragging:
            dx = x - start_drag[0]
            dy = y - start_drag[1]
            offset_x = max(0, offset_x - dx)
            offset_y = max(0, offset_y - dy)
            start_drag = (x, y)

    elif event == cv2.EVENT_MOUSEWHEEL:
        old_scale = scale
        scale *= 1.2 if flags > 0 else 1 / 1.2
        scale = max(0.2, min(scale, 20.0))

        img_x_before = (x + offset_x) / old_scale
        img_y_before = (y + offset_y) / old_scale
        offset_x = max(0, int(img_x_before * scale - x))
        offset_y = max(0, int(img_y_before * scale - y))

    elif event == cv2.EVENT_RBUTTONDOWN:
        dragging = True
        start_drag = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        dragging = False


def reset_image_state() -> None:
    global scale, offset_x, offset_y, mouse_x, mouse_y, dragging
    scale, offset_x, offset_y = 1.0, 0, 0
    mouse_x, mouse_y = 0, 0
    dragging = False
    click_points.clear()
    saved_points.clear()
    pending_rows.clear()
    action_history.clear()


def run_labeling(
    image_folder: Path,
    csv_path: Path,
    append: bool,
    config_path: str | Path = "config/thresholds.yaml",
    overwrite: bool = False,
) -> None:
    global idx, current_label, output_csv, _label_source_image, _label_extractor
    output_csv = csv_path
    _label_extractor = FeatureExtractor(resolve_config_path(config_path))
    images = collect_images(image_folder)
    if not images:
        raise RuntimeError(f"No images found in: {image_folder}")
    initialize_output_csv(csv_path, append=append, overwrite=overwrite)

    print(f"Image folder: {image_folder}")
    print(f"Output CSV:   {csv_path}")
    print(f"Images:       {len(images)}")

    idx = 0
    while idx < len(images):
        path = images[idx]
        filename = image_label_for_csv(path, csv_path)
        img_orig = quiet_imread(path, cv2.IMREAD_COLOR)
        if img_orig is None:
            print(f"Failed to load image: {path.name}")
            idx += 1
            continue

        _label_source_image = quiet_imread(path, cv2.IMREAD_UNCHANGED)
        reset_image_state()

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, click_event)

        while True:
            disp, new_offset_x, new_offset_y = get_display_image(img_orig, scale, offset_x, offset_y)
            globals()["offset_x"] = new_offset_x
            globals()["offset_y"] = new_offset_y
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                flush_pending_rows()
                cv2.destroyAllWindows()
                print(f"Saved labels to {csv_path}")
                return
            if key == 32:
                flush_pending_rows()
                idx += 1
                break
            if key == ord("u"):
                undo_last_action()
            elif key in (ord("b"), ord("l"), ord("n")):
                current_label = chr(key)
                xo = int((mouse_x + offset_x) / scale)
                yo = int((mouse_y + offset_y) / scale)
                if 0 <= xo < img_orig.shape[1] and 0 <= yo < img_orig.shape[0]:
                    stamp_points(
                        img_orig.shape[:2],
                        filename,
                        xo,
                        yo,
                        current_label,
                    )

    cv2.destroyAllWindows()
    print(f"Saved labels to {csv_path}")


def main() -> None:
    args = parse_args()
    image_folder = resolve_image_folder(args.image_folder)
    csv_path = Path(args.output_csv) if args.output_csv else default_output_csv_for_image_folder(image_folder)
    if not csv_path.is_absolute():
        csv_path = PROJECT_ROOT / csv_path
    try:
        run_labeling(
            image_folder, csv_path, append=bool(args.append),
            config_path=args.config, overwrite=bool(args.overwrite),
        )
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
