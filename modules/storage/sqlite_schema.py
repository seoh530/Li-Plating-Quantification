from pathlib import Path
import sqlite3


PREDICTION_RUN_COLUMNS: tuple[str, ...] = (
    "run_timestamp",
    "analysis_id",
    "cell",
    "face",
    "image_num",
    "image_name",
    "customer",
    "model_version",
    "total_pixel",
    "background_ratio",
    "nonactive_ratio",
    "li_salt_ratio",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_h",
    "mean_s",
    "mean_v",
    "mean_l",
    "mean_a",
    "mean_b_lab",
    "mean_confidence",
    "low_confidence_ratio",
    "high_conf_bg_ratio",
    "raw_bg_l",
    "raw_bg_a",
    "raw_bg_b_lab",
    "baseline_bg_l",
    "baseline_bg_a",
    "baseline_bg_b_lab",
    "l_shift",
    "a_shift",
    "b_shift",
    "drift_score_before",
    "drift_score_after",
    "correction_applied",
    "normalization_skipped_reason",
    "retraining_candidate",
    "qc_warning",
    "csv_path",
    "mask_path",
    "pass_image_path",
)


CREATE_PREDICTION_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp TEXT,

    analysis_id TEXT,
    cell TEXT,
    face TEXT,
    image_num INTEGER,
    image_name TEXT,
    customer TEXT,
    model_version TEXT,

    total_pixel INTEGER,
    background_ratio REAL,
    nonactive_ratio REAL,
    li_salt_ratio REAL,

    mean_r REAL,
    mean_g REAL,
    mean_b REAL,

    mean_h REAL,
    mean_s REAL,
    mean_v REAL,

    mean_l REAL,
    mean_a REAL,
    mean_b_lab REAL,

    mean_confidence REAL,
    low_confidence_ratio REAL,

    high_conf_bg_ratio REAL,

    raw_bg_l REAL,
    raw_bg_a REAL,
    raw_bg_b_lab REAL,

    baseline_bg_l REAL,
    baseline_bg_a REAL,
    baseline_bg_b_lab REAL,

    l_shift REAL,
    a_shift REAL,
    b_shift REAL,

    drift_score_before REAL,
    drift_score_after REAL,

    correction_applied INTEGER,
    normalization_skipped_reason TEXT,
    retraining_candidate INTEGER,
    qc_warning INTEGER,

    csv_path TEXT,
    mask_path TEXT,
    pass_image_path TEXT
)
"""


PREDICTION_RUN_COLUMN_DEFINITIONS: dict[str, str] = {
    "run_timestamp": "TEXT",
    "analysis_id": "TEXT",
    "cell": "TEXT",
    "face": "TEXT",
    "image_num": "INTEGER",
    "image_name": "TEXT",
    "customer": "TEXT",
    "model_version": "TEXT",
    "total_pixel": "INTEGER",
    "background_ratio": "REAL",
    "nonactive_ratio": "REAL",
    "li_salt_ratio": "REAL",
    "mean_r": "REAL",
    "mean_g": "REAL",
    "mean_b": "REAL",
    "mean_h": "REAL",
    "mean_s": "REAL",
    "mean_v": "REAL",
    "mean_l": "REAL",
    "mean_a": "REAL",
    "mean_b_lab": "REAL",
    "mean_confidence": "REAL",
    "low_confidence_ratio": "REAL",
    "high_conf_bg_ratio": "REAL",
    "raw_bg_l": "REAL",
    "raw_bg_a": "REAL",
    "raw_bg_b_lab": "REAL",
    "baseline_bg_l": "REAL",
    "baseline_bg_a": "REAL",
    "baseline_bg_b_lab": "REAL",
    "l_shift": "REAL",
    "a_shift": "REAL",
    "b_shift": "REAL",
    "drift_score_before": "REAL",
    "drift_score_after": "REAL",
    "correction_applied": "INTEGER",
    "normalization_skipped_reason": "TEXT",
    "retraining_candidate": "INTEGER",
    "qc_warning": "INTEGER",
    "csv_path": "TEXT",
    "mask_path": "TEXT",
    "pass_image_path": "TEXT",
}


def ensure_prediction_schema(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_PREDICTION_RUNS_SQL)
    existing = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(prediction_runs)").fetchall()
    }
    for column, definition in PREDICTION_RUN_COLUMN_DEFINITIONS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE prediction_runs ADD COLUMN {column} {definition}")


def sqlite_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return int(value)
    return value
