# Battery PhotoJig QC + ML Vision System — GUI Test Guide

---

## Current Deployment Notes

- The main app now opens with `Workflow`, `Analysis`, `Settings`, and `Guide` tabs.
- Change PhotoJig `root_path` in the `Settings` tab; the value is saved to `config/thresholds.yaml` and reused until changed again.
- Relative SQLite DB paths are stored under `<root_path>/.qcml/`.
- The `Analysis` tab can export selected/current-date result data to CSV or Excel and capture the current Analysis screen to PNG.
- ML logs now include `mask_write` timing so mask PNG save cost can be compared with model prediction time.

---

## Running the Application

```bash
cd C:/Users/og03980/Desktop/QCML_AGENT
python main.py
```

The window opens with the **QC** tab selected by default.

QC TAB date behavior:
- A date dropdown shows all valid date folders under `app.root_path`.
- Default selection is **today's date** if it exists.
- If today's date folder does not exist, the **most recent available date** is selected.
- When the date changes, the AnalysisID preview refreshes automatically.

---

## Screen Layout

| Area | Description |
|------|-------------|
| Root | `app.root_path` from `config/thresholds.yaml` (read-only) |
| Date | Selectable dropdown of available date folders under root path |
| Run QC | Scans all AnalysisIDs in the date folder and runs QC |
| Table | Per-AnalysisID FolderQC / CropEngine results |

### Table Columns

| Column | Content |
|--------|---------|
| AnalysisID | Folder name (e.g. AHG06113). Shown only on the first row if multiple FAIL images exist for the same ID |
| FolderQC | ● PASS / ● FAIL |
| CropEngine | ● PASS / ● FAIL / - (if FolderQC failed) |
| FAIL Image | Relative path of the FAIL image (e.g. FRONT/electrode_015.jpg) |
| FAIL Reason | FolderQC or CropEngine failure reason |
| Retake | FolderQC FAIL → [Refresh] / CropEngine FAIL → ☐ Retake complete + Crop anyway + [Refresh] |

---

## CropEngine Behavior Notes

- **First and last image** in each FRONT/BACK folder (sorted by filename) are **QC-exempt PASS**. They are not written to `_pass` folders, so ML does not process those edge sheets.
- **Aspect ratio tolerance** is ±2% from the customer reference ratio.
- **PNG output** is written through a Windows-safe atomic writer:
  encode to memory → write unique temp file in the target folder → flush/fsync → retry `os.replace()`.
- This avoids intermittent `[WinError 5] Access is denied` errors from temp-file collisions and reduces corrupted PNG writes that can trigger `libpng warning: phYs: CRC error`.

---

## ML Performance Notes

**Target: 2–3 seconds per image on CPU (HKMC 2955×894 = 2.6 M pixels)**

The following optimizations are applied:

| Optimization | Before | After | Gain |
|---|---|---|---|
| Single `booster.predict()` call (no batching) | 11 batches × ~150 ms overhead | 1 call | ~1,300 ms |
| `num_threads: -1` (all CPU cores) | default / 1 core | all cores | varies |
| `png_compression_level: 1` (mask write) | OpenCV default 6 | 1 | ~250 ms |
| HSV computed once, reused in edge post-processing | 2× `cvtColor` | 1× | ~30 ms |
| `build_mask_image` — loop unrolled to 4 direct assignments | loop | unrolled | ~50 ms |
| Feature extraction — `np.stack + boolean mask` | 6 separate channel slices | 1 `(H,W,6)` stack | ~15 ms |

**Tuning knobs in `config/thresholds.yaml` (`ml` section):**

| Key | Default | Effect |
|-----|---------|--------|
| `num_threads` | `-1` | LightGBM CPU threads (-1 = all cores) |
| `png_compression_level` | `1` | Mask PNG write speed (0–9; 1 recommended) |
| `downsample_factor` | `1.0` | Pre-inference resize (1.0 = off; 0.5 halves pixels) |

If a single image still exceeds 3 seconds after the above, reduce `downsample_factor`
(e.g. `0.5` reduces pixel count to 25 % of original). The mask is currently returned
at inference resolution; upsampling to original size is left for a future extension.

---

## ML Inference Pipeline

ML inference runs the following steps in order for each image:

1. **Feature extraction** — per-pixel RGB + HSV features from the QC-pass PNG
2. **Model inference** — LightGBM classifies each pixel (background / li-salt / inactive)
3. **Edge post-processing** — rule-based overrides applied before mask generation:
   - Electrode contour is derived from the valid-pixel (non-transparent) region
   - **Tab tip zone**: all columns left of the leftmost contour X → forced to `edge` (no HSV check)
   - **Left body / right / top / bottom zones**: `edge_margin_px`-wide strips at each border → pixels matching edge HSV range → `edge`
   - Body left X is found using the top-25% / bottom-25% contour y-range, consistent with the CropEngine 4-point strategy (ignores tab region in middle 50%)
   - **Nonactive HSV override**: any pixel the model predicted as `li-salt` that matches the nonactive HSV range → overridden to `inactive`
   - Edge overrides take priority; nonactive override only affects remaining `li-salt` pixels
4. **Mask image** generated from finalized labels (edge → gray, inactive → blue, background → green, li-salt → red)
5. **Quantification** — edge pixels are counted inside `inactive_ratio` in CSV (no separate edge column)

All HSV thresholds and `edge_margin_px` are read from `config/thresholds.yaml` (`ml.edge_hsv_range`, `ml.nonactive_hsv_range`, `ml.edge_margin_px`).

---

## ColorNormalizer Behavior Notes

- Normalization runs in **CIELAB color space** (`use_lab: true` in config), which gives more natural color shifts than BGR-based correction.
- A **blend factor** (default 0.7) softens the correction:  
  `output = 0.7 × corrected + 0.3 × original`
- Correction coefficients are **clamped** (scale: 0.8–1.2, shift: ±25 units) to prevent over-saturation.
- If `reference_inactive_mean_std.mean` or `reference_background_mean_std.mean` is `null` in config, ColorNormalizer is skipped — run `tools/compute_reference.py` first.
- ColorNormalizer rewrites PNG outputs through the same atomic writer, so normalized pass images do not reuse partially written PNG files.

---

## Test 1 — FolderQC FAIL

### Setup

1. Rename the `BACK` folder to `BACK_temp` in:
   ```
   C:/Users/SKBA_FA/Documents/VIISAN OfficeCam/<today's date>/AHG06113/BACK
   ```

2. Click **Run QC** in the GUI.

### Expected Result

```
AnalysisID │ FolderQC │ CropEngine │ FAIL Image │ FAIL Reason        │ Retake
AHG06113   │ ● FAIL   │ -          │ -          │ BACK folder missing │ [Refresh]
```

- CropEngine does not run if FolderQC fails.

### Recovery

1. Rename `BACK_temp` back to `BACK`.
2. Click **[Refresh]** on that row.
3. Expected: `FolderQC = ● PASS`, `CropEngine = ● PASS`.

---

## Test 2 — CropEngine FAIL

### Setup

1. Copy any image without a white background (e.g. a solid-color photo) into:
   ```
   C:/Users/SKBA_FA/Documents/VIISAN OfficeCam/<today's date>/AHG06113/FRONT/
   ```

2. Click **Run QC**.

### Expected Result

```
AnalysisID │ FolderQC │ CropEngine │ FAIL Image              │ FAIL Reason                     │ Retake
AHG06113   │ ● PASS   │ ● FAIL     │ FRONT/bad_image.jpg     │ Contour extraction failed ...   │ ☐ Retake complete
           │          │            │                         │                                 │ [Refresh]
```

- The **☐ Retake complete** checkbox appears for each FAIL image.
- The **Fail image review** panel shows FAIL images in order. Use **Previous / Next** to review them in the app window and choose **Retake complete** or **Crop anyway** from that panel.
- **[Refresh]** appears on the last FAIL row.

### Retake Simulation

1. Delete the bad image from `FRONT/`.
2. Copy a valid electrode image into the same `FRONT/` folder.
3. Check the **☐ Retake complete** checkbox for that image.
4. The system re-runs QC on **that specific image only** (not the entire folder).
5. If the image passes, the row updates to `● PASS`.

> Checking "Retake complete" triggers re-validation of **the individual image file only**.
> If more FAIL images remain in the folder, their rows stay until each is retaken.

---

## Test 3 — ColorNormalizer Output Check

ColorNormalizer is applied automatically to every CropEngine PASS image.
There is no separate status indicator in the GUI.

### How to Verify

After running QC, check that PNG files were created at:

```
.../AHG06113/FRONT_pass/*.png
.../AHG06113/BACK_pass/*.png
```

Open a file in an image viewer and confirm:
- White paper background is **transparent** (checkerboard pattern).
- Colors look natural — subtle shift, not fluorescent or over-saturated.

> If `reference_inactive_mean_std.mean` is `null` in config, ColorNormalizer is skipped.
> Run `tools/compute_reference.py` (with `use_lab: true` in config) to generate the values.

---

## Final State — ALL PASS

When all AnalysisIDs pass FolderQC and CropEngine:

- The table area is replaced with:
  ```
  ALL PASS  ✓
  ```
- Status label: **All PASS ✓** (green)

---

## Common Errors

| Error Message | Cause | Action |
|---------------|-------|--------|
| Date folder not found | Selected date folder does not exist under capture root | Create the folder or check `config.app.root_path` |
| No AnalysisID folders found | Date folder exists but no `[A-Z]{3}[0-9]{5}` subfolder | Check AnalysisID folder name (e.g. AHG06113) |
| ColorNormalizer skipped | Reference statistics not set | Run `tools/compute_reference.py` and update config |

---

## LightGBM Training

Place labeled training CSV files in:

```text
data/labeled/
```

Required CSV columns:

```text
filename, x, y, r, g, b, h, s, v, label
```

Run the training script from the **project root**:

```bash
cd C:/Users/og03980/Desktop/QCML_AGENT
python tools/train_model.py
```

Notes:
- `python tools/train_model.py` must be run from the project root.
- If no input path is passed, the script automatically reads all CSV files under `data/labeled/`.
- You may still pass one or more CSV files or directories explicitly if needed.

Expected outputs:
- Console report with overall accuracy, per-class precision/recall/F1, and confusion matrix
- Saved model files in `models/lgbm_predictor/`
- `models/lgbm_predictor/model.txt`
- `models/lgbm_predictor/metadata.json`
- `models/lgbm_predictor/report.txt`
