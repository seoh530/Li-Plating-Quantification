# compute_reference.py — One-time tool: operator clicks reference regions, outputs config values
#
# Usage:
#   python tools/compute_reference.py
#   python tools/compute_reference.py --ref-dir path/to/reference/images
#
# Run once after initial setup.
# Rerun only if camera, lens, or lighting conditions change significantly.
#
# Output:
#   Config-ready YAML values to copy into config/thresholds.yaml
#   under color_normalizer.reference_inactive_mean_std and
#   color_normalizer.reference_background_mean_std

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Constants ─────────────────────────────────────────────────────────────────

CONFIG_PATH  = Path(__file__).parent.parent / "config" / "thresholds.yaml"
DEFAULT_REF  = Path(__file__).parent.parent / "tests" / "reference"
DEFAULT_LABELED = Path(__file__).parent.parent / "data" / "labeled"

# Click-patch radius (samples a 2r+1 × 2r+1 area around each click)
PATCH_RADIUS = 3

# Margin applied when expanding click-derived HSV min/max into a mask range.
# Reads hsv_mask_margin from config; this is a fallback default.
DEFAULT_MARGIN = 5

# Display colors for marked points
COLOR_INACTIVE   = (0,   220,   0)    # green
COLOR_BACKGROUND = (0,   165, 255)    # orange

MAX_DISPLAY_W = 1400
MAX_DISPLAY_H = 900


# ── State ─────────────────────────────────────────────────────────────────────

class Session:
    def __init__(self) -> None:
        self.mode: str = ""           # "inactive" | "background" | ""
        self.clicks: dict[str, list[np.ndarray]] = {
            "inactive":   [],
            "background": [],
        }

    def add_click(self, x: int, y: int) -> None:
        if self.mode in self.clicks:
            self.clicks[self.mode].append(np.array([x, y]))

    def has_both_regions(self) -> bool:
        return (
            len(self.clicks["inactive"])   >= 1 and
            len(self.clicks["background"]) >= 1
        )


# ── Image helpers ─────────────────────────────────────────────────────────────

def scale_for_display(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale image to fit MAX_DISPLAY_W × MAX_DISPLAY_H. Returns (scaled, scale)."""
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return img.copy(), 1.0


def draw_overlay(
    display_img: np.ndarray,
    session: Session,
    scale: float,
) -> np.ndarray:
    """Draw mode indicator, instructions, and clicked point markers."""
    overlay = display_img.copy()
    h, w    = overlay.shape[:2]

    # Mode banner
    if session.mode == "inactive":
        banner_color = COLOR_INACTIVE
        mode_text    = "MODE: INACTIVE  (클릭으로 비활성 영역 마킹)"
    elif session.mode == "background":
        banner_color = COLOR_BACKGROUND
        mode_text    = "MODE: BACKGROUND  (클릭으로 배경 영역 마킹)"
    else:
        banner_color = (180, 180, 180)
        mode_text    = "MODE: 없음  (i 또는 b 키를 눌러 모드 선택)"

    cv2.rectangle(overlay, (0, 0), (w, 28), banner_color, -1)
    cv2.putText(overlay, mode_text, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # Instructions
    instructions = [
        "i: inactive 모드  b: background 모드",
        "클릭: 현재 모드로 픽셀 마킹",
        "u: 마지막 클릭 취소  r: 현재 이미지 초기화",
        "Enter/s: 저장 후 다음  q: 즉시 종료",
    ]
    for idx, line in enumerate(instructions):
        cv2.putText(overlay, line, (6, h - 14 - (len(instructions) - 1 - idx) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(overlay, line, (6, h - 14 - (len(instructions) - 1 - idx) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (30, 30, 30), 1, cv2.LINE_AA)

    # Click counters
    n_i = len(session.clicks["inactive"])
    n_b = len(session.clicks["background"])
    cv2.putText(overlay, f"inactive: {n_i}  background: {n_b}",
                (w - 230, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 20), 1, cv2.LINE_AA)

    # Marked points
    for pt in session.clicks["inactive"]:
        dx, dy = int(pt[0] * scale), int(pt[1] * scale)
        cv2.circle(overlay, (dx, dy), 5, COLOR_INACTIVE,   2)
        cv2.circle(overlay, (dx, dy), 1, COLOR_INACTIVE,  -1)

    for pt in session.clicks["background"]:
        dx, dy = int(pt[0] * scale), int(pt[1] * scale)
        cv2.circle(overlay, (dx, dy), 5, COLOR_BACKGROUND, 2)
        cv2.circle(overlay, (dx, dy), 1, COLOR_BACKGROUND, -1)

    return overlay


def collect_patch_pixels(
    bgr: np.ndarray,
    points: list[np.ndarray],
    radius: int = PATCH_RADIUS,
) -> np.ndarray:
    """
    For each clicked point, collect all BGR pixels in a (2r+1)×(2r+1) patch.
    Returns (N, 3) float64 array, or empty array if no points.
    """
    h, w = bgr.shape[:2]
    all_pixels: list[np.ndarray] = []
    for pt in points:
        px, py = int(pt[0]), int(pt[1])
        x0, x1 = max(0, px - radius), min(w, px + radius + 1)
        y0, y1 = max(0, py - radius), min(h, py + radius + 1)
        patch = bgr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
        all_pixels.append(patch)
    if not all_pixels:
        return np.empty((0, 3), dtype=np.float64)
    return np.concatenate(all_pixels, axis=0)


def collect_patch_hsv(
    bgr: np.ndarray,
    points: list[np.ndarray],
    radius: int = PATCH_RADIUS,
) -> np.ndarray:
    """Same as collect_patch_pixels but in HSV color space. Returns (N, 3) float64."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return collect_patch_pixels(hsv, points, radius)


def derive_hsv_range(
    hsv_pixels: np.ndarray,
    margin: int,
) -> tuple[list[int], list[int]]:
    """
    Derive [lower, upper] HSV range from sampled pixels with margin applied.
    H clipped to 0–179, S/V clipped to 0–255.
    """
    lo = hsv_pixels.min(axis=0).astype(np.int32) - margin
    hi = hsv_pixels.max(axis=0).astype(np.int32) + margin
    lo[0]  = int(np.clip(lo[0],  0,  179))
    hi[0]  = int(np.clip(hi[0],  0,  179))
    lo[1:] = np.clip(lo[1:], 0, 255)
    hi[1:] = np.clip(hi[1:], 0, 255)
    return lo.tolist(), hi.tolist()


def extract_full_mask(
    bgr: np.ndarray,
    lower: list[int],
    upper: list[int],
) -> np.ndarray:
    """Extract boolean mask from full image using derived HSV range."""
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8),
                            np.array(upper, dtype=np.uint8))
    return mask.astype(bool)


# ── Interactive loop for one image ────────────────────────────────────────────

def run_interactive(
    bgr: np.ndarray,
    image_name: str,
) -> Session | None:
    """
    Open an OpenCV window for the operator to click inactive and background regions.
    Returns the completed Session, or None if the operator pressed 'q'.
    """
    session = Session()
    display, scale = scale_for_display(bgr)

    win = f"compute_reference — {image_name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, display.shape[1], display.shape[0])

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Map display coords back to original image coords
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            session.add_click(orig_x, orig_y)

    cv2.setMouseCallback(win, on_mouse)

    print(f"\n[{image_name}]")
    print("  i: inactive 모드  |  b: background 모드")
    print("  클릭: 픽셀 마킹   |  u: 마지막 취소  |  r: 초기화")
    print("  Enter/s: 완료 → 다음  |  q: 즉시 종료")

    while True:
        frame = draw_overlay(display, session, scale)
        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('i'):
            session.mode = "inactive"
            print(f"  → inactive 모드")

        elif key == ord('b'):
            session.mode = "background"
            print(f"  → background 모드")

        elif key == ord('u'):
            if session.mode in session.clicks and session.clicks[session.mode]:
                removed = session.clicks[session.mode].pop()
                print(f"  취소: {session.mode} 클릭 ({int(removed[0])}, {int(removed[1])})")

        elif key == ord('r'):
            session.clicks = {"inactive": [], "background": []}
            print("  초기화: 모든 클릭 삭제됨")

        elif key in (13, ord('s')):    # Enter or 's'
            if not session.has_both_regions():
                print("  ⚠  inactive 와 background 영역을 모두 1개 이상 클릭해야 합니다.")
            else:
                break

        elif key == ord('q'):
            cv2.destroyWindow(win)
            return None

    cv2.destroyWindow(win)
    return session


# ── Stats aggregation ─────────────────────────────────────────────────────────

def compute_region_stats(
    bgr: np.ndarray,
    mask: np.ndarray,
) -> tuple[list[float], list[float]]:
    """
    Compute BGR mean and std from masked region.
    Returns ([B_mean, G_mean, R_mean], [B_std, G_std, R_std]).
    """
    pixels = bgr[mask].astype(np.float64)
    mean   = pixels.mean(axis=0).tolist()
    std    = pixels.std(axis=0).tolist()
    return mean, std


def combine_stats(
    all_inactive_pixels: list[np.ndarray],
    all_bg_pixels:       list[np.ndarray],
    use_lab: bool = False,
) -> tuple[
    list[float], list[float],
    list[float], list[float],
]:
    """
    Compute combined mean/std from accumulated BGR pixel arrays across all images.

    When use_lab=True, converts BGR pixels to float32 CIELAB
    (L∈[0,100], a/b∈[-127,127]) before computing statistics.
    Returns (inactive_mean, inactive_std, bg_mean, bg_std).
    Channel order: [B,G,R] when use_lab=False, [L,a,b] when use_lab=True.
    """
    inactive = np.concatenate(all_inactive_pixels, axis=0)   # (N, 3) float64 BGR
    bg       = np.concatenate(all_bg_pixels,       axis=0)

    if use_lab:
        # Convert accumulated BGR float64 to float32 LAB
        def bgr_to_lab(pixels: np.ndarray) -> np.ndarray:
            # pixels shape: (N, 3) float64, values in [0, 255]
            px_f32 = (pixels / 255.0).astype(np.float32).reshape(1, -1, 3)
            lab    = cv2.cvtColor(px_f32, cv2.COLOR_BGR2Lab).reshape(-1, 3)
            return lab.astype(np.float64)

        inactive = bgr_to_lab(inactive)
        bg       = bgr_to_lab(bg)

    i_mean = inactive.mean(axis=0).tolist()
    i_std  = inactive.std(axis=0).tolist()
    b_mean = bg.mean(axis=0).tolist()
    b_std  = bg.std(axis=0).tolist()
    return i_mean, i_std, b_mean, b_std


# ── Output formatting ─────────────────────────────────────────────────────────

def fmt3(vals: list[float]) -> str:
    return "[" + ", ".join(f"{v:.2f}" for v in vals) + "]"


def fmt_int_list(vals: list) -> str:
    return "[" + ", ".join(str(int(v)) for v in vals) + "]"


def print_config(
    inactive_lo:   list[int],   inactive_hi:  list[int],
    bg_lo:         list[int],   bg_hi:        list[int],
    inactive_mean: list[float], inactive_std: list[float],
    bg_mean:       list[float], bg_std:       list[float],
    margin:        int,
    use_lab:       bool = False,
) -> None:
    channel_label = "[L, a, b] (CIELAB)" if use_lab else "[B, G, R] (OpenCV BGR)"
    print()
    print("=" * 60)
    print("CONFIG-READY OUTPUT — copy into config/thresholds.yaml")
    print("=" * 60)
    print()
    print("color_normalizer:")
    print(f"  hsv_mask_margin: {margin}")
    print( "  hsv_inactive_range:")
    print(f"    lower: {fmt_int_list(inactive_lo)}")
    print(f"    upper: {fmt_int_list(inactive_hi)}")
    print( "  hsv_background_range:")
    print(f"    lower: {fmt_int_list(bg_lo)}")
    print(f"    upper: {fmt_int_list(bg_hi)}")
    print( "  reference_inactive_mean_std:")
    print(f"    mean: {fmt3(inactive_mean)}")
    print(f"    std:  {fmt3(inactive_std)}")
    print( "  reference_background_mean_std:")
    print(f"    mean: {fmt3(bg_mean)}")
    print(f"    std:  {fmt3(bg_std)}")
    print()
    print(f"  # Channel order: {channel_label}")
    print(f"  # Matches config use_lab: {'true' if use_lab else 'false'}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_labeled_csvs(inputs: list[str]) -> list[Path]:
    if not inputs:
        inputs = [str(DEFAULT_LABELED)]

    csv_files: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            csv_files.extend(sorted(path.rglob("*.csv")))
        elif path.is_file() and path.suffix.lower() == ".csv":
            csv_files.append(path)
        else:
            raise RuntimeError(f"Input path is not a CSV file or directory: {path}")

    unique = sorted({path.resolve() for path in csv_files})
    if not unique:
        raise RuntimeError("No labeled CSV files found for baseline generation.")
    return unique


def load_labeled_data(csv_files: list[Path]) -> "pd.DataFrame":
    frames = []
    required = {"r", "g", "b", "h", "s", "v", "label"}
    for csv_path in csv_files:
        frame = pd.read_csv(csv_path)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"Missing required columns in {csv_path}: {missing}")
        frame = frame.copy()
        frame["source_csv"] = str(csv_path)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise RuntimeError("Labeled CSV data is empty.")
    return data


def ensure_lab_columns(data: "pd.DataFrame") -> "pd.DataFrame":
    if {"l_lab", "a_lab", "b_lab"}.issubset(data.columns):
        return data

    rgb = data.loc[:, ["r", "g", "b"]].to_numpy(dtype=np.uint8, copy=True)
    bgr = rgb[:, [2, 1, 0]].reshape(1, -1, 3)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3)
    data = data.copy()
    data["l_lab"] = lab[:, 0]
    data["a_lab"] = lab[:, 1]
    data["b_lab"] = lab[:, 2]
    return data


def channel_stats(frame: "pd.DataFrame", columns: list[str]) -> dict:
    values = frame.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    return {
        "mean": {name: float(value) for name, value in zip(columns, values.mean(axis=0))},
        "std": {name: float(value) for name, value in zip(columns, values.std(axis=0))},
    }


def label_ratios(data: "pd.DataFrame") -> dict[str, float]:
    total = max(len(data), 1)
    labels = data["label"].astype(str).str.lower()
    return {
        "background": float((labels == "b").sum() / total),
        "li_salt": float((labels == "l").sum() / total),
        "nonactive": float((labels == "n").sum() / total),
    }


def build_baseline_stats(data: "pd.DataFrame", csv_files: list[Path]) -> dict:
    data = ensure_lab_columns(data)
    labels = data["label"].astype(str).str.lower()
    background = data[labels == "b"]
    if background.empty:
        raise RuntimeError("No background rows found in labeled data.")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_csvs": [str(path) for path in csv_files],
        "global": {
            "rgb": channel_stats(data, ["r", "g", "b"]),
            "hsv": channel_stats(data, ["h", "s", "v"]),
            "lab": channel_stats(data, ["l_lab", "a_lab", "b_lab"]),
            "background_rgb": channel_stats(background, ["r", "g", "b"]),
            "background_hsv": channel_stats(background, ["h", "s", "v"]),
            "background_lab": {
                "mean": {
                    "l": float(background["l_lab"].mean()),
                    "a": float(background["a_lab"].mean()),
                    "b": float(background["b_lab"].mean()),
                },
                "std": {
                    "l": float(background["l_lab"].std(ddof=0)),
                    "a": float(background["a_lab"].std(ddof=0)),
                    "b": float(background["b_lab"].std(ddof=0)),
                },
            },
            "label_ratios": label_ratios(data),
        },
    }


def run_baseline_generation(
    inputs: list[str],
    config_path: Path,
    output_path: Path | None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    model_dir = Path(cfg["ml"]["model_path"])
    if not model_dir.is_absolute():
        model_dir = config_path.parent.parent / model_dir
    target_path = output_path or (model_dir / "baseline_stats.json")

    csv_files = collect_labeled_csvs(inputs)
    data = load_labeled_data(csv_files)
    baseline = build_baseline_stats(data, csv_files)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Baseline stats written: {target_path}")
    print(f"Background LAB mean: {baseline['global']['background_lab']['mean']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="기준 이미지 통계 계산 (최초 1회 실행)"
    )
    parser.add_argument(
        "--ref-dir", type=Path, default=DEFAULT_REF,
        help=f"기준 이미지 폴더 (기본값: {DEFAULT_REF})",
    )
    parser.add_argument(
        "--baseline-from-labeled",
        nargs="*",
        default=None,
        metavar="PATH",
        help="Generate baseline_stats.json from labeled training CSV files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Config path (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=None,
        help="Optional output path for baseline_stats.json.",
    )
    args = parser.parse_args()

    if args.baseline_from_labeled is not None:
        run_baseline_generation(
            list(args.baseline_from_labeled),
            args.config,
            args.baseline_output,
        )
        return

    ref_dir: Path = args.ref_dir
    if not ref_dir.exists():
        print(f"오류: 기준 이미지 폴더를 찾을 수 없습니다: {ref_dir}")
        sys.exit(1)

    # Load config for margin and use_lab
    margin  = DEFAULT_MARGIN
    use_lab = True
    if args.config.exists():
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cn = cfg.get("color_normalizer", {})
        if cn.get("hsv_mask_margin") is not None:
            margin = int(cn["hsv_mask_margin"])
        use_lab = bool(cn.get("use_lab", True))
    print(f"hsv_mask_margin: {margin}  (from config)")
    print(f"use_lab: {use_lab}  (from config)  — stats will be in {'LAB [L,a,b]' if use_lab else 'BGR [B,G,R]'}")

    # List available images
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in exts)

    if not images:
        print(f"오류: 기준 이미지 없음 — {ref_dir}")
        sys.exit(1)

    print(f"\n기준 이미지 목록 ({ref_dir}):")
    for idx, img_path in enumerate(images):
        print(f"  {idx + 1}. {img_path.name}")

    # Image selection
    print()
    raw = input("처리할 이미지 번호를 입력하세요 (예: 1  또는  1,2): ").strip()
    selected_indices: list[int] = []
    for token in raw.replace(",", " ").split():
        try:
            n = int(token)
            if 1 <= n <= len(images):
                selected_indices.append(n - 1)
            else:
                print(f"  무시: {n} (범위 초과)")
        except ValueError:
            print(f"  무시: '{token}' (숫자 아님)")

    if not selected_indices:
        print("선택된 이미지 없음. 종료합니다.")
        sys.exit(0)

    # Process each selected image
    all_inactive_pixels: list[np.ndarray] = []
    all_bg_pixels:       list[np.ndarray] = []

    # Accumulate HSV click samples across images to derive ranges
    all_inactive_hsv_samples: list[np.ndarray] = []
    all_bg_hsv_samples:       list[np.ndarray] = []

    for idx in selected_indices:
        img_path = images[idx]
        print(f"\n로드 중: {img_path.name}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  오류: 이미지 로드 실패 — {img_path.name}")
            continue

        session = run_interactive(bgr, img_path.name)
        if session is None:
            print("사용자 중단 (q). 지금까지 수집된 데이터로 계산합니다.")
            break

        # Collect patch HSV from clicked points
        i_hsv = collect_patch_hsv(bgr, session.clicks["inactive"])
        b_hsv = collect_patch_hsv(bgr, session.clicks["background"])

        if i_hsv.shape[0] == 0 or b_hsv.shape[0] == 0:
            print(f"  경고: 클릭 데이터 부족 — {img_path.name} 건너뜀")
            continue

        all_inactive_hsv_samples.append(i_hsv)
        all_bg_hsv_samples.append(b_hsv)

        # Derive per-image HSV ranges → extract full masks → collect BGR pixels
        i_lo, i_hi = derive_hsv_range(i_hsv, margin)
        b_lo, b_hi = derive_hsv_range(b_hsv, margin)

        i_mask = extract_full_mask(bgr, i_lo, i_hi)
        b_mask = extract_full_mask(bgr, b_lo, b_hi)

        i_pixels = bgr[i_mask].astype(np.float64)
        b_pixels = bgr[b_mask].astype(np.float64)

        print(
            f"  inactive: {len(session.clicks['inactive'])}클릭 → "
            f"마스크 {i_mask.sum():,}px  "
            f"BGR mean={[round(v, 1) for v in i_pixels.mean(axis=0).tolist()]}"
        )
        print(
            f"  background: {len(session.clicks['background'])}클릭 → "
            f"마스크 {b_mask.sum():,}px  "
            f"BGR mean={[round(v, 1) for v in b_pixels.mean(axis=0).tolist()]}"
        )

        if i_pixels.shape[0] > 0:
            all_inactive_pixels.append(i_pixels)
        if b_pixels.shape[0] > 0:
            all_bg_pixels.append(b_pixels)

    # ── Compute final output ───────────────────────────────────────────────────

    if not all_inactive_pixels or not all_bg_pixels:
        print("\n오류: 유효한 픽셀 데이터가 없습니다. 종료합니다.")
        sys.exit(1)

    # Combined HSV ranges from all click samples
    all_i_hsv = np.concatenate(all_inactive_hsv_samples, axis=0)
    all_b_hsv = np.concatenate(all_bg_hsv_samples,       axis=0)
    final_i_lo, final_i_hi = derive_hsv_range(all_i_hsv, margin)
    final_b_lo, final_b_hi = derive_hsv_range(all_b_hsv, margin)

    # Combined stats (BGR or LAB depending on use_lab)
    i_mean, i_std, b_mean, b_std = combine_stats(
        all_inactive_pixels, all_bg_pixels, use_lab=use_lab
    )

    ch = "LAB" if use_lab else "BGR"
    print(f"\nImages processed: {len(all_inactive_pixels)}")
    print(
        f"inactive    {sum(p.shape[0] for p in all_inactive_pixels):,} px  "
        f"{ch} mean={fmt3(i_mean)}  std={fmt3(i_std)}"
    )
    print(
        f"background  {sum(p.shape[0] for p in all_bg_pixels):,} px  "
        f"{ch} mean={fmt3(b_mean)}  std={fmt3(b_std)}"
    )

    print_config(
        final_i_lo, final_i_hi,
        final_b_lo, final_b_hi,
        i_mean, i_std,
        b_mean, b_std,
        margin,
        use_lab=use_lab,
    )


if __name__ == "__main__":
    main()
