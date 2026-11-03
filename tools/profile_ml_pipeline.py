"""
ML inference pipeline profiler - post-optimization version.

Creates a synthetic electrode image (HKMC size: 2955x894 BGRA)
and a tiny LightGBM model, then times each pipeline stage using
the actual updated module code paths.

Usage:
    python tools/profile_ml_pipeline.py
"""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import lightgbm as lgb
import numpy as np
import yaml

from modules.ml.feature_extractor import FeatureExtractor

CONFIG_PATH = ROOT / "config" / "thresholds.yaml"
with open(CONFIG_PATH, encoding="utf-8") as fh:
    CFG = yaml.safe_load(fh)

ML_CFG = CFG["ml"]
PNG_COMPRESSION = int(ML_CFG.get("png_compression_level", 1))
DNS_FACTOR = float(ML_CFG.get("downsample_factor", 1.0))

IMG_W, IMG_H = 2955, 894

N_FEATURES = len(FeatureExtractor.FEATURE_COLUMNS)
LABEL_BACKGROUND = 0
LABEL_LI_SALT = 1
LABEL_INACTIVE = 2
LABEL_EDGE = 3


def tick(label: str, t0: float) -> float:
    elapsed = time.perf_counter() - t0
    print(f"  {label:<48s} {elapsed*1000:8.1f} ms")
    return time.perf_counter()


print(
    f"Building synthetic LightGBM model (3-class, {N_FEATURES} features, 150 rounds) ..."
)
rng = np.random.default_rng(42)
n_train = 60_000
X_train = rng.random((n_train, N_FEATURES), dtype=np.float32)
y_train = rng.integers(0, 3, size=n_train)
ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
params = {
    "objective": "multiclass",
    "num_class": 3,
    "num_leaves": 31,
    "verbosity": -1,
    "num_threads": int(ML_CFG.get("num_threads", -1)),
}
booster = lgb.train(params, ds, num_boost_round=150, valid_sets=[], callbacks=[])
model_dir = ROOT / "tmp_profile_model"
model_dir.mkdir(exist_ok=True)
model_file = model_dir / "model.txt"
booster.save_model(str(model_file))
(model_dir / "metadata.json").write_text(
    json.dumps(
        {
            "feature_columns": list(FeatureExtractor.FEATURE_COLUMNS),
            "classes": [0, 1, 2],
        }
    )
)
print("  Model ready.\n")

tmp_path = ROOT / "tmp_profile_img.png"
bgra = np.zeros((IMG_H, IMG_W, 4), dtype=np.uint8)
bgra[:, :, :3] = rng.integers(60, 160, (IMG_H, IMG_W, 3), dtype=np.uint8)
bgra[:, :, 3] = 255
bgra[:5, :, 3] = 0
bgra[-5:, :, 3] = 0
bgra[:, :5, 3] = 0
bgra[:, -5:, 3] = 0
cv2.imwrite(str(tmp_path), bgra, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])

print("=== Pipeline timing (post-optimization) ===")
t_start = t = time.perf_counter()

img = cv2.imread(str(tmp_path), cv2.IMREAD_UNCHANGED)
t = tick("1. Image load (cv2.imread)", t)

extractor = FeatureExtractor()
feature_batch = extractor.extract(img, downsample_factor=DNS_FACTOR)
n_valid = feature_batch.features.shape[0]
t = tick(
    f"2. Feature extraction ({n_valid:,} valid pixels, downsample={DNS_FACTOR:.2f})",
    t,
)

scores = booster.predict(feature_batch.features, validate_features=False)
codes = np.argmax(scores, axis=1).astype(np.uint8)
label_mask = np.full(feature_batch.image_shape, fill_value=-1, dtype=np.int16)
label_mask[feature_batch.valid_mask] = codes.astype(np.int16)
t = tick("3. LightGBM inference (single batch)", t)

from modules.ml.inference import (
    LABEL_BACKGROUND,
    LABEL_EDGE,
    LABEL_INACTIVE,
    LABEL_LI_SALT,
)

nonactive_lower = np.array(ML_CFG["nonactive_hsv_range"]["lower"], dtype=np.uint8)
nonactive_upper = np.array(ML_CFG["nonactive_hsv_range"]["upper"], dtype=np.uint8)
m = int(ML_CFG["edge_margin_px"])

hsv = feature_batch.hsv
valid = label_mask >= 0
H, W = label_mask.shape

valid_u8 = valid.astype(np.uint8) * 255
contours, _ = cv2.findContours(valid_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)
pts = contour[:, 0, :]
xs, ys = pts[:, 0], pts[:, 1]
y_min, y_max = int(ys.min()), int(ys.max())
y_range = y_max - y_min

nonactive_hsv_match = cv2.inRange(hsv, nonactive_lower, nonactive_upper).astype(bool)

result = label_mask.copy()
x_tab_tip = int(xs.min())
if x_tab_tip > 0:
    result[valid & (np.arange(W) < x_tab_tip)[np.newaxis, :]] = LABEL_EDGE
body_mask_pts = (ys < y_min + y_range * 0.25) | (ys > y_max - y_range * 0.25)
x_body_left = int(xs[body_mask_pts].min()) if body_mask_pts.any() else x_tab_tip
x_right = int(xs.max())
left_zone = np.zeros((H, W), dtype=bool)
left_zone[:, x_body_left : x_body_left + m] = True
right_zone = np.zeros((H, W), dtype=bool)
right_zone[:, max(0, x_right - m + 1) : x_right + 1] = True
top_zone = np.zeros((H, W), dtype=bool)
top_zone[y_min : min(H, y_min + m), :] = True
bottom_zone = np.zeros((H, W), dtype=bool)
bottom_zone[max(0, y_max - m + 1) : y_max + 1, :] = True
outer_band = left_zone | right_zone | top_zone | bottom_zone
result[valid & outer_band & (result == LABEL_LI_SALT)] = LABEL_EDGE
remaining_li_salt = result == LABEL_LI_SALT
result[valid & remaining_li_salt & nonactive_hsv_match] = LABEL_INACTIVE
t = tick("4. Edge post-processing (HSV reused from FeatureBatch)", t)

MASK_COLORS = {
    LABEL_BACKGROUND: (0, 255, 0),
    LABEL_INACTIVE: (255, 0, 0),
    LABEL_EDGE: (128, 128, 128),
    LABEL_LI_SALT: (0, 0, 255),
}
mask_img = np.zeros((H, W, 3), dtype=np.uint8)
mask_img[result == LABEL_BACKGROUND] = MASK_COLORS[LABEL_BACKGROUND]
mask_img[result == LABEL_INACTIVE] = MASK_COLORS[LABEL_INACTIVE]
mask_img[result == LABEL_EDGE] = MASK_COLORS[LABEL_EDGE]
mask_img[result == LABEL_LI_SALT] = MASK_COLORS[LABEL_LI_SALT]
t = tick("5. Mask image build (unrolled)", t)

mask_path = ROOT / "tmp_profile_mask.png"
cv2.imwrite(str(mask_path), mask_img, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
t = tick(f"   Mask PNG write (compression={PNG_COMPRESSION})", t)

total_pixel = int(valid.sum())
bg = int(np.count_nonzero(result == LABEL_BACKGROUND))
inact = int(np.count_nonzero((result == LABEL_INACTIVE) | (result == LABEL_EDGE)))
bg_r = round(bg / total_pixel, 6)
inact_r = round(inact / total_pixel, 6)
li_r = round(max(0.0, 1.0 - bg_r - inact_r), 6)
t = tick("6. CSV row assembly (quantification)", t)

total = time.perf_counter() - t_start
print(f"\n  {'TOTAL':<48s} {total*1000:8.1f} ms")
print(f"\n  (image: {IMG_W}x{IMG_H}, valid_pixels={n_valid:,})")

tmp_path.unlink(missing_ok=True)
mask_path.unlink(missing_ok=True)
shutil.rmtree(model_dir, ignore_errors=True)
