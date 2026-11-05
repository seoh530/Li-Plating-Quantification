# Li-Plating Quant System

Desktop application for photo-jig QC, lithium-residue/nonactive segmentation, quantification, and result review.

## Run

Use Python 3.10+ with Tkinter available. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python main.py
```

Supply trained model files separately. The default source configuration is `config/thresholds.yaml`. At application startup it is copied once into `%LOCALAPPDATA%/Li-Plating_Quant_System/config/thresholds.yaml`; subsequent operator changes are saved there. Editing the source YAML does not replace an existing user configuration.

In **Settings**, choose the capture root and a model directory containing `metadata.json` and `model.txt` or `model.joblib`. Main workflow, Single Mask Mode, and the ML module now honor `ml.model_path`. Deployments previously using the hardcoded `models/V2F` should select that directory in Settings before processing.

## Operator workflow

- **Main**: recursively discovers AnalysisID folders below the configured root, without date/customer selectors. The second letter maps H/F/V/E to HKMC/FORD/VW/ESS. Root may itself be an AnalysisID. Nonstandard containers may appear anywhere above an AnalysisID.
- **Show completed**: off by default; enable to include completed QC + ML + DB + combined-report items. A folder with missing combined reports stays visible and can generate reports without repeating ML.
- **Single Mask Mode**: crop/segment an image; Export Images + CSV saves source, mask and class percentages to a selected directory.
- **Developer mode**: select Li-salt/Background/Nonactive and click/drag on the source-mask overlay. Points use cropped source pixel coordinates; transparent pixels are ignored. Undo point removes the most recently added point.
- **Train (old:new = 3:7)**: trains a NEW LightGBM model from original training CSVs plus the new sparse labels. Total sample-weight mass is 30% original / 70% new, independent of row counts. Original weights/model are preserved; the new model is active in Single Mask for this session. To use it throughout the app or on restart, select its directory in Settings. This interactive run has no independent accuracy claim. If original source CSVs/features are missing, training stops rather than inventing replay data. Grouped validation remains available in the normal training script.
- **Add train set**: writes a unique cropped source and enhanced point CSV under `data/labeled/TEST/`. The uncropped capture is preserved separately under `TEST/captures/`. CSV coordinates refer to the cropped PNG, not the uncropped capture. Saving a dataset and training are separate actions. In a frozen EXE, developer datasets/models use the persistent application data directory instead of the temporary bundle directory.
- **Analysis**: browse saved cell results, layer trends, images, and heatmaps.
- **Report**: retained for manual reporting. Workflow automatically creates `<FACE>_combined` and `<FACE>_mask_combined` after successful QC/ML. The QC pass report includes end-sheet crops. Mask reports leave the first/last slots blank without shifting subsequent layer numbers. ML still excludes the same end sheets and does not infer artificial endpoint masks.
- **Settings**: capture root and deployed model settings; **Guide** remains available.

`gui/qc_tab.py`, `gui/ml_tab.py`, and `modules/qc/retake_manager.py` are retained. `TEST`/`TESTxx` analysis folders remain supported operator inputs. The current QC/ML policy excludes both end sheets for multi-image source sets; summaries use actual analyzed rows. No customer-specific physical-layer policy was changed.

## Sparse labels and enhanced features

```bash
python tools/label_tools.py
```

New label CSVs are named `enhanced_*.csv` and include enhanced features immediately; running `enhance_training_features.py` separately is only needed to migrate old basic CSVs.

Labels contain `filename,x,y,label`: `b` background, `l` lithium residue, and `n` nonactive. Coordinates refer to the cropped/QC-ready image, with x increasing rightward and y downward. Store image paths relative to the CSV location when possible.

An existing label CSV is protected from accidental truncation. Use `--append` to add labels to an enhanced CSV or `--overwrite` to explicitly start over. Conflicting labels on the same image pixel and invalid coordinates are rejected before feature generation or training. A later label for the same coordinate replaces its earlier value when saving with the label tool. Conflicting legacy annotations are rejected during training. Duplicate image basenames must be disambiguated by a relative or absolute path.

Enhanced CSVs retain the four label columns and add features computed by `modules/ml/feature_extractor.py`. The current repository supports `basic`/`enhanced` training; it does not contain `enhanced_v2`.

`tools/compute_reference.py` is the optional interactive tool for color-normalization reference values. Rerun calibration when the capture conditions change.

## Train LightGBM

```bash
python tools/train_lightGBM.py --feature-set enhanced --model-name my_model
```

The script selects eligible CSVs and saves a model, metadata, validation report, source CSV list, and feature importance under `models/`. Existing nonempty model output directories are protected from overwrite.

Validation holds out independent groups instead of neighboring labeled pixels. Configure `ml.training.validation_group_by` as:

- `cell` (default): group both faces/layers of an identified cell together; fall back to the AnalysisID group or source CSV when cell identity is unavailable.
- `analysis`: hold out entire AnalysisIDs.
- `image`: exploratory image-level validation; this does not estimate performance on independent cells.
- `csv`: hold out entire source CSVs.

Set `validation_fraction` and `seed` under `ml.training`. Every labeled class must occur in both partitions. If no suitable grouped split can be found, training stops with an explanation; it does not silently revert to a pixel split. Metadata records the selected training and validation groups.

To train without generating optional test-image previews:

```bash
python tools/train_lightGBM.py --feature-set enhanced --model-name my_model --skip-test-masks
```

`tools/test_model.py` is retained for optional saved-model preview on `data/test`. Unlike unit tests, it is an inference utility. Generated previews no longer recursively delete their whole output directory; unrelated files are preserved.

## Output and data integrity

Pixel codes are `-1` outside the valid image, `0` background, `1` lithium residue, `2` nonactive, and `3` postprocessed edge. PNG colors use OpenCV BGRA order: background green, lithium red, nonactive blue, and edge gray; invalid pixels are transparent.

The existing quantification denominator includes all nonnegative labels, including edge pixels. Ratios are stored as values from 0 to 1, with full floating-point precision in CSV exports. UI percent formatting is separate. Model preprocessing parameters saved in metadata take precedence over the current feature configuration during inference.

Image/CSV/settings writes use temporary files and replacement. DB cell replacement happens with new rows in a single SQLite transaction after analysis and CSV save; a failed insert rolls back the replacement. Images are decoded from bytes to support Unicode Windows paths.

## Packaging

```bash
python -m pip install pyinstaller
python -m PyInstaller Li-Plating_Quant_System.spec
```

The spec resolves config/models relative to its own location. If distributing a bundled model, populate `models/` before building; otherwise select an external model in Settings. The existing spec excludes optional XGBoost/CatBoost dependencies and does not bundle PyTorch. U-Net training/inference is not part of this patch.

## Validation limits

The code-review patch was checked with synthetic-image/data regression cases, actual LightGBM training/export/reload/inference, Python compilation, and unresolved-name checks. It has not been validated against production photos/model files or through interactive Windows GUI operation. Before deployment, check Main/Single Mask, QC failure and retake/resume, report export, settings reload, and result CSVs on a copy of representative production data.
