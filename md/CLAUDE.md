# Battery PhotoJig QC + ML Vision System

---

## Project Purpose

Build an operator-assisted computer vision system that ensures all PhotoJig electrode images are correctly structured, deterministically cropped, and color-normalized before any machine learning inference is executed.

The system separates:
- **QC Stage**: Data acquisition quality control
- **ML Stage**: Machine learning analysis

This guarantees that downstream lithium-residue quantification is performed only on validated and standardized image inputs.

---

## System Architecture Overview

```
PhotoJig Capture → QC TAB → QC-READY dataset → ML TAB (manual trigger) → Mask + Quantification
```

---

## Technology Stack

| Category          | Stack                        |
|-------------------|------------------------------|
| Language          | Python 3.10                  |
| Core Libraries    | OpenCV, NumPy, pandas, LightGBM |
| GUI Framework     | tkinter                      |
| Packaging         | PyInstaller                  |

---

## Project Folder Structure

```
project_root/
├── main.py
├── CLAUDE.md
├── ARCHITECTURE.md
├── GUIDE.md
├── requirements.txt
├── data/
│   └── labeled/         # labeled CSV files for tools/train_model.py
├── config/
│   └── thresholds.yaml
├── modules/
│   ├── qc/
│   │   ├── folder_qc.py
│   │   ├── crop_engine.py
│   │   ├── color_normalizer.py
│   │   └── retake_manager.py
│   └── ml/
│       ├── feature_extractor.py
│       ├── inference.py
│       └── quantifier.py
├── gui/
│   ├── qc_tab.py
│   └── ml_tab.py
├── tools/
│   └── compute_reference.py
├── tests/
│   ├── reference/       # old PhotoJig images for compute_reference.py
│   ├── new_images/      # new PhotoJig images for CropEngine + ColorNormalizer
│   └── results/         # cropped + normalized output
└── logs/
```

---

## Commands

```bash
python main.py              # run
pytest tests/               # test
pyinstaller main.spec       # package
pip install -r requirements.txt  # install dependencies
```

---

## Config

- File: `config/thresholds.yaml`
- Manages: crop tolerance, HSV ranges, image count rules, hsv_mask_margin, tab_y_range_ratio, customer reference aspect ratios, pixel count deviation threshold
- No hardcoding allowed — all values must be read from config

---

## Coding Conventions

- Modular architecture
- Deterministic processing
- Config-driven thresholds
- CPU-efficient operations
- Logging-first debugging
- Folder-based data pipeline

---

## Folder Structure Specification

All PhotoJig data must follow this hierarchical structure:

```
ROOT/
└── DATE/
    └── ANALYSIS_ID/
        └── (optional) CELL_FOLDER/
            ├── FRONT/
            └── BACK/
```

### Full Structure Example — Single Cell

```
C:/PhotoJig/
└── 2026-02-09/
    └── AHG06113/
        ├── FRONT/
        └── BACK/
```

### Full Structure Example — Multiple Cells

```
C:/PhotoJig/
└── 2026-02-09/
    └── AHG06113/
        ├── Cell #1/
        │   ├── FRONT/
        │   └── BACK/
        └── Cell #2/
            ├── FRONT/
            └── BACK/
```

### Structure Rules

| Level        | Rule |
|--------------|------|
| Root Path    | Fixed to PhotoJig capture location (e.g. `C:/Users/SKBA_FA/Documents/VIISAN OfficeCam`) |
| Date Folder  | Must exist under root. Formats: `YYYY-MM-DD`, `M-D-YYYY`, `M-D-YY` |
| AnalysisID   | Must follow AnalysisID naming rule. Located directly under date folder |
| Cell Folder  | Optional. Used only when multiple cells exist. Pattern: `Cell #1`, `Cell #2` (case-insensitive). If single cell → may be skipped |
| FRONT / BACK | Must exist inside each cell or directly under AnalysisID. Case-insensitive |

### Validation Conditions

- Date folder exists
- AnalysisID format is valid
- FRONT and BACK folders exist
- If cell folders exist: each must contain FRONT and BACK
- Structure mismatch → **QC FAIL**

---

## AnalysisID Rule

Format:

```
[A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][0-9]
```

Example: `AHG06113`, `THF43301`

### Customer Identification

The **second character** of AnalysisID defines the customer:

| Code | Customer |
|------|----------|
| H    | HKMC     |
| F    | FORD     |
| V    | VW       |
| S    | Slate    |
| T    | Test     |

---

## Image Count Rule

Each AnalysisID must contain FRONT and BACK images with counts based on customer.

| Customer | Code | Required Count (Front / Back) |
|----------|------|-------------------------------|
| HKMC     | H    | 30                            |
| FORD     | F    | 38                            |
| VW       | V    | 23                            |
| Slate    | S    | No restriction                |
| Test     | T    | No restriction                |

### Validation Logic

- FRONT and BACK must exist
- Image count must match expected count (except S, T)
- Count mismatch → **QC FAIL**

---

# Development Checklist

## QC TAB

- [ ] FolderQC (`modules/qc/folder_qc.py`)
- [ ] CropEngine (`modules/qc/crop_engine.py`)
- [ ] ColorNormalizer (`modules/qc/color_normalizer.py`) + `tools/compute_reference.py`
- [ ] RetakeManager (`modules/qc/retake_manager.py`)
- [ ] QC TAB GUI (`gui/qc_tab.py`)

## ML TAB

- [ ] train_model.py
- [ ] feature_extractor.py
- [ ] inference.py
- [ ] quantifier.py
- [ ] ML TAB GUI (`ml_tab.py`)

---

## Purpose

Validate and standardize captured electrode (Anode) images before ML analysis.

## Inputs

- Root PhotoJig directory
- Date folder
- Raw FRONT / BACK images

## Outputs

- Cropped + color-normalized images
- QC status
- Retake list

## Stack

OpenCV, NumPy, tkinter

---

## Main Modules

### FolderQC
- Naming validation
- Customer image count rule

### CropEngine
- White background detection
- Electrode contour detection
- Perspective transform (lens distortion correction)
- Outer edge color validation
- Tolerance validation

**Pipeline**

```
1. White paper background mask extraction
2. Largest contour extraction (main body + tab as single contour)
3. 4-point extraction for perspective transform
4. Perspective transform → rectified rectangle
5. Outer edge pixel color validation (must be inactive/edge color)
6. White background residual pixels → alpha (transparent), save as PNG
7. Size tolerance validation
8. Pixel count outlier check within same FRONT/BACK folder
```

**4-Point Extraction Strategy**

Tab is always located at the left-center of the cell. To avoid tab distorting the left corner points:

```
Right-top    : point with largest x and smallest y in contour
Right-bottom : point with largest x and largest y in contour
Left-top     : among points in top 25% y-range of bounding rect,
               find point closest in y to right-top
Left-bottom  : among points in bottom 25% y-range of bounding rect,
               find point closest in y to right-bottom
```

Middle 50% y-range on the left side is treated as the tab region and ignored during corner extraction.

**Outer Edge Validation**

After perspective transform, outermost pixels must fall within inactive/edge HSV range.
If outermost pixels are background color (yellow) → **QC FAIL** (cell is out of frame).

**Tolerance Rules**

- Angle tolerance: within ±4 degrees — if exceeded → **QC FAIL**
- Size deviation: within ±0.3% of customer reference aspect ratio (tab included)
- Pixel count deviation: if any image differs by more than ±10% from the mean pixel count of other images in the same FRONT/BACK folder → **QC FAIL**
- All failed images forwarded to RetakeManager

**Customer Reference Size (tab included, width/height ratio basis)**

| Customer | Width (px) | Height (px) | Aspect Ratio |
|----------|------------|-------------|--------------|
| HKMC     | 2955       | 894         | 3.306        |
| FORD     | 3404       | 1326        | 2.566        |
| VW       | 4995       | 960         | 5.203        |

Note: File size is not used for validation (varies by lens). Aspect ratio is used instead.

**White Paper Mask (HSV threshold)**

```python
paper_mask = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
```

**Output Format**

- Format: PNG (alpha channel for background masking)
- Save location: `FRONT_pass/` and `BACK_pass/` in the same directory as FRONT/BACK
- ML TAB uses only images from `_pass` folders

### ColorNormalizer

**Pipeline**: Crop → ColorNormalize (applied after crop is complete)

**Method**: Mean/Std Matching based on a single reference image

1. Extract inactive / background region masks from reference image using HSV threshold
2. Compute per-channel mean and std for each region → save to config
3. Extract masks from new image using the same HSV threshold
4. Compute affine correction coefficients based on mean/std
5. Apply the same coefficients to the entire image

**Reference Setup**

- Run once via separate script: `tools/compute_reference.py`
- Operator clicks points on inactive and background regions in 1–2 reference images
- Script automatically extracts mean and std for each region and prints config-ready values
- Operator manually copies output values into `config/thresholds.yaml`
- Reference does not need to be recomputed unless camera or lighting conditions change significantly

**Note**
- ColorNormalizer is not a PASS/FAIL target
- Automatically applied to all CropEngine PASS images before saving to `FRONT_pass/` and `BACK_pass/`
- No separate status display in GUI

**HSV Color Range for Mask Estimation**

The ranges below are used exclusively for ColorNormalizer mask estimation.
They are not used for pixel classification or model inference.
Apply `hsv_mask_margin` from config to both ends of each range during mask extraction.

| Region     | H range | S range | V range |
|------------|---------|---------|---------|
| inactive   | 0–179   | 4–164   | 72–158  |
| background | 19–22   | 155–197 | 201–255 |

- `hsv_mask_margin`: read from config (recommended default: 5)
- Application: `[H_min - margin, S_min - margin, V_min - margin]` ~ `[H_max + margin, S_max + margin, V_max + margin]`
- H channel must be clipped to stay within 0–179

### RetakeManager

**Triggered by**: CropEngine FAIL only (not FolderQC FAIL)

**Flow**:
1. CropEngine FAIL → display failed image filename + FAIL reason in GUI
2. Show instruction message: "Please delete this image and retake"
3. Operator manually deletes the failed image
4. Operator retakes and saves new image to the same folder
5. Operator checks checkbox ("Retake complete") in GUI
6. GUI auto-refreshes → CropEngine re-runs on that image → re-validates

### Dashboard

**FolderQC FAIL handling**
- Display FAIL reason to operator
- Operator manually fixes the folder structure or image count
- Operator clicks manual refresh button in GUI
- FolderQC re-runs → re-validates

**CropEngine FAIL handling**
- Forwarded to RetakeManager panel
- See RetakeManager flow above

**GUI Panel Layout**

| Panel | Content |
|-------|---------|
| FolderQC | PASS/FAIL status + FAIL reason + manual refresh button |
| CropEngine | PASS/FAIL status + list of FAIL images with reasons |
| RetakeManager | FAIL image filename + instruction message + checkbox per image ("Retake complete") |

---

## Reference Cell Image Color Spectrum

**Full label spectrum for visual reference (surveyed from 5–10 cells)**
This table is for engineer / operator reference only.
Not used directly for model inference or rule-based segmentation.

Note: `nonactive` (uncharged region) and `edge` are reported together as `inactive_ratio` in the CSV output. In the mask image they are rendered with separate colors (blue vs gray).

| Label      | R range  | G range  | B range  | H range | S range | V range |
|------------|----------|----------|----------|---------|---------|---------|
| nonactive  | 65–153   | 24–158   | 19–156   | 0–179   | 57–164  | 93–155  |
| edge       | 65–153   | 24–158   | 19–156   | 0–177   | 4–62    | 72–158  |
| background | 198–255  | 145–210  | 46–94    | 19–22   | 155–197 | 201–255 |
| li-salt    | 221–236  | 183–197  | 102–122  | 20–20   | 123–137 | 221–236 |

---

# ML TAB

## Purpose

Run model inference and lithium residue quantification on QC-approved images.

## Inputs

- QC-READY cropped + color-normalized images from `FRONT_pass/` and `BACK_pass/` folders

## Outputs

- Mask images
- Quantification CSV

## Stack

LightGBM, NumPy, pandas

---

## Pipeline

```
Image → Feature Extraction → Model Inference → Edge Post-Processing → Mask → Pixel Count → Quantification
```

### Pipeline Order (detail)

```
1. Model inference        → raw per-pixel label predictions
2. Edge zone boundaries   calculated from electrode contour (valid-pixel region)
3. Tab tip zone           → force LABEL_EDGE (no HSV check)
4. Left body / right / top / bottom zones + edge HSV match → LABEL_EDGE
5. Nonactive HSV (anywhere) + model li-salt → override to LABEL_INACTIVE
6. Remaining li-salt confirmed as li-salt
7. Mask image generated from finalized labels
8. CSV quantification from finalized labels
```

### Mask Color Mapping

| Label      | Color  | BGR           |
|------------|--------|---------------|
| background | green  | (0, 255, 0)   |
| inactive   | blue   | (255, 0, 0)   |
| edge       | gray   | (128, 128, 128) |
| li-salt    | red    | (0, 0, 255)   |

## Quantification Metrics

- Denominator: total cropped image pixels (inactive + edge + background + li-salt)
- inactive_ratio + background_ratio + li_salt_ratio = 1.0
- inactive_ratio includes both nonactive (uncharged) region and edge region
- Edge pixels are NOT reported as a separate column in CSV

## CSV Output Schema

- One CSV file per AnalysisID
- Save location: inside the AnalysisID folder
- Filename: `{AnalysisID}_result.csv`

| Column | Type | Description |
|--------|------|-------------|
| AnalysisID | string | e.g. AHG06113 |
| Cell | integer | Cell number (1, 2, 3 ...) |
| Face | string | FRONT or BACK |
| ImageNum | integer | Image index within the Face folder |
| total_pixel | integer | Total cropped image pixel count |
| background_ratio | float | background pixels / total_pixel |
| inactive_ratio | float | inactive pixels / total_pixel |
| li_salt_ratio | float | li-salt pixels / total_pixel |

## Execution

- Manual start button
- Batch processing

---

## Model

- Model path: `models/lgbm_predictor/`
- Input unit: per-pixel (image decomposed into pixels, features extracted per pixel)
- Currently testing with pre-trained model; retraining via separate script if needed

---

## Future Extensions

- Real-time watcher
- Blur detection
- Glove detection
- Heatmap tab

---

## References

- System architecture: see `ARCHITECTURE.md`

---

# Claude Behavior Rules

## Communication Style

- Always respond in Korean
- Be concise and technical
- No unnecessary praise or filler phrases

## Before Writing Any Code

Before writing any code, confirm all of the following:

1. Do I fully understand the purpose of this module?
2. Do I know the exact input format and data type?
3. Do I know the exact output format and where it is saved?
4. Do I know all edge cases and failure conditions?
5. Do I know all config values this module depends on?

If any answer is NO → ask the user first
If all answers are YES → explain the implementation plan, wait for approval, then write code

## Question Protocol

- Ask one question at a time
- Do not bundle multiple questions in one message
- Do not assume or guess missing specifications
