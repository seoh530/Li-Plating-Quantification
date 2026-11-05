"""Sparse annotations, dataset persistence, and weighted LightGBM retraining."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pandas as pd

from modules.ml.feature_extractor import FeatureExtractor
from modules.ml.inference import InferenceEngine, TRAINING_LABEL_TO_CODE
from modules.ml.model_registry import project_root
from modules.utils.image_io import atomic_imwrite_png, atomic_write_text
from modules.utils.app_paths import user_data_dir


def annotation_frame(image, points, filename, extractor, columns=None):
    if not points:
        raise RuntimeError('Add labeled points first.')
    columns = tuple(columns or FeatureExtractor.ENHANCED_FEATURE_COLUMNS)
    # One class per coordinate; a later correction replaces the earlier class.
    labels = {(int(x), int(y)): label for x, y, label in points}
    batch = extractor.extract(image, feature_columns=columns)
    values, valid = extractor.select_point_features(batch, labels)
    if not valid.all():
        raise RuntimeError('Annotations must be inside the non-transparent image area.')
    frame = pd.DataFrame([(filename, x, y, label) for (x, y), label in labels.items()],
                         columns=['filename', 'x', 'y', 'label'])
    if not frame.label.isin(TRAINING_LABEL_TO_CODE).all():
        raise RuntimeError('Unsupported annotation label.')
    return pd.concat([frame, pd.DataFrame(values, columns=columns)], axis=1)


def save_training_set(image, points, config_path, *, feature_config=None, root=None):
    base = user_data_dir() if getattr(sys, 'frozen', False) else project_root()
    root = Path(root) if root is not None else base / 'data' / 'labeled'
    session = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    analysis = root / 'TEST'
    relative = f'Cell #1/FRONT/dev_{session}.png'
    extractor = FeatureExtractor(config_path, feature_config=feature_config)
    frame = annotation_frame(image, points, relative, extractor)
    image_path = analysis / relative
    csv_path = analysis / f'enhanced_TEST_{session}.csv'
    if not atomic_imwrite_png(image_path, image):
        raise RuntimeError('Failed to save the training source image.')
    atomic_write_text(csv_path, frame.to_csv(index=False, float_format='%.17g'))
    return image_path, csv_path


def replay_paths(metadata):
    paths = []
    for name in metadata.get('source_csvs', []):
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = project_root() / candidate
        if not candidate.is_file():
            matches = list((project_root() / 'data' / 'labeled').rglob(Path(name).name))
            if len(matches) != 1:
                raise RuntimeError(f'Original training CSV unavailable or ambiguous: {name}. '
                                   '3:7 retraining needs the original labeled data.')
            candidate = matches[0]
        paths.append(candidate.resolve())
    if not paths:
        raise RuntimeError('Model metadata has no source_csvs. Cannot perform 3:7 replay retraining.')
    return list(dict.fromkeys(paths))


def retrain(engine, image, points, config_path):
    if engine.model_type != 'lightgbm_booster':
        raise RuntimeError('Developer retraining currently supports LightGBM models only.')
    columns = engine.feature_columns
    metadata = engine._metadata
    sources = replay_paths(metadata)
    frames = []
    for path in sources:
        frame = pd.read_csv(path)
        if '_replay_source' not in frame:
            frame['_replay_source'] = str(path)
        frames.append(frame)
    if any(not set(columns).issubset(frame.columns) for frame in frames):
        raise RuntimeError('Original CSVs lack the model feature columns. Regenerate their enhanced features first.')
    old = pd.concat(frames, ignore_index=True).drop_duplicates(['_replay_source', 'filename', 'x', 'y', 'label'])
    if old.empty or not old.label.isin(TRAINING_LABEL_TO_CODE).all():
        raise RuntimeError('Original training labels are empty or invalid.')
    extractor = FeatureExtractor(config_path, feature_config=metadata.get('feature_extraction'))
    new = annotation_frame(image, points, 'developer.png', extractor, columns)
    new['_replay_source'] = 'developer_' + uuid4().hex
    total = len(old) + len(new)
    # Normalize total sample weight, not weight per row: old:new = 30:70.
    weights = np.r_[np.full(len(old), .3 * total / len(old)),
                    np.full(len(new), .7 * total / len(new))]
    data = pd.concat([old, new], ignore_index=True)
    values = data.loc[:, list(columns)].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError('Training features contain missing or non-finite values.')
    labels = data.label.map(TRAINING_LABEL_TO_CODE).to_numpy(np.int32)
    params = dict(engine._booster.params)
    # Saved model strings may omit parameters; keep multiclass explicit.
    params.update(objective='multiclass', num_class=3, metric='multi_logloss',
                  learning_rate=.03, num_leaves=31, min_data_in_leaf=5,
                  num_threads=4, verbosity=-1, seed=42)
    booster = lgb.train(params, lgb.Dataset(values, label=labels, weight=weights, feature_name=list(columns)),
                        num_boost_round=100)
    # Store the replay rows used by this model so later retraining is reproducible.
    output_root = user_data_dir() / 'models' if getattr(sys, 'frozen', False) else engine.model_dir.parent
    output = output_root / ('developer_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    replay = output / 'training_rows.csv'
    atomic_write_text(replay, data.to_csv(index=False, float_format='%.17g'))
    booster.save_model(str(output / 'model.txt'))
    updated = dict(metadata)
    updated.update(model_type='lightgbm_booster', model_version=output.name,
                   source_csvs=[str(replay.resolve())], parent_model=str(engine.model_dir),
                   best_iteration=0, developer_training={'old_weight':.3, 'new_weight':.7,
                   'old_rows':len(old), 'new_rows':len(new), 'validation':'not independently validated'})
    # Old validation metrics/groups do not describe this interactive fine-tuning run.
    for key in ['training_groups', 'validation_groups', 'validation_group_by']:
        updated.pop(key, None)
    atomic_write_text(output / 'metadata.json', json.dumps(updated, indent=2))
    return InferenceEngine(config_path, output)
