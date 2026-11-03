import logging
import sqlite3
from pathlib import Path
from typing import Mapping

from modules.utils.app_paths import user_database_path

logger = logging.getLogger(__name__)


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


class SQLitePredictionLogger:
    """Small fault-tolerant SQLite writer for ML production monitoring."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._db_path: Path | None = None
        self._enabled = False

        try:
            self._db_path = user_database_path()
            self._test_db_path(self._db_path)
            self._ensure_schema()
            self._enabled = True
            logger.info("SQLite monitoring DB ready: %s", self._db_path)
        except Exception:
            logger.exception("SQLite monitoring disabled: initialization failed")
            self._db_path = None
            self._enabled = False

    @property
    def db_path(self) -> Path | None:
        return self._db_path

    def insert_prediction_run(self, record: Mapping[str, object]) -> None:
        """Insert a prediction row; log and continue if SQLite fails."""
        if not self._enabled or self._db_path is None:
            return

        values = [self._sqlite_value(record.get(column)) for column in PREDICTION_RUN_COLUMNS]
        placeholders = ", ".join("?" for _ in PREDICTION_RUN_COLUMNS)
        columns_sql = ", ".join(PREDICTION_RUN_COLUMNS)
        sql = f"INSERT INTO prediction_runs ({columns_sql}) VALUES ({placeholders})"

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(sql, values)
                conn.commit()
        except Exception:
            logger.exception("SQLite monitoring insert failed")

    def existing_prediction_cells(
        self,
        pairs: set[tuple[str, str]],
    ) -> set[tuple[str, str]]:
        """Return AnalysisID/cell pairs already present in prediction_runs."""
        if not self._enabled or self._db_path is None or not pairs:
            return set()

        existing: set[tuple[str, str]] = set()
        try:
            with sqlite3.connect(self._db_path) as conn:
                for analysis_id, cell in pairs:
                    row = conn.execute(
                        """
                        SELECT 1
                        FROM prediction_runs
                        WHERE analysis_id = ? AND cell = ?
                        LIMIT 1
                        """,
                        (analysis_id, cell),
                    ).fetchone()
                    if row is not None:
                        existing.add((analysis_id, cell))
        except Exception:
            logger.exception("SQLite monitoring duplicate check failed")
        return existing

    def delete_prediction_cells(self, pairs: set[tuple[str, str]]) -> None:
        """Delete existing monitoring rows for selected AnalysisID/cell pairs."""
        if not self._enabled or self._db_path is None or not pairs:
            return

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.executemany(
                    """
                    DELETE FROM prediction_runs
                    WHERE analysis_id = ? AND cell = ?
                    """,
                    list(pairs),
                )
                conn.commit()
        except Exception:
            logger.exception("SQLite monitoring duplicate replacement failed")

    def correction_summary(self, analysis_id: str) -> tuple[int, int] | None:
        """Return (total_rows, corrected_rows) for one AnalysisID."""
        if not self._enabled or self._db_path is None:
            return None

        try:
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*),
                        COALESCE(SUM(CASE WHEN correction_applied = 1 THEN 1 ELSE 0 END), 0)
                    FROM prediction_runs
                    WHERE analysis_id = ?
                    """,
                    (analysis_id,),
                ).fetchone()
            if row is None:
                return None
            return int(row[0]), int(row[1])
        except Exception:
            logger.exception("SQLite monitoring correction summary failed")
            return None

    def correction_details(self, analysis_id: str) -> list[dict[str, object]] | None:
        """Return layer-level correction rows for one AnalysisID."""
        if not self._enabled or self._db_path is None:
            return None

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT
                        cell,
                        face,
                        image_num,
                        correction_applied,
                        raw_bg_l,
                        raw_bg_a,
                        raw_bg_b_lab,
                        baseline_bg_l,
                        baseline_bg_a,
                        baseline_bg_b_lab,
                        l_shift,
                        a_shift,
                        b_shift,
                        drift_score_before,
                        drift_score_after
                    FROM prediction_runs
                    WHERE analysis_id = ?
                    ORDER BY cell, face, image_num
                    """,
                    (analysis_id,),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            logger.exception("SQLite monitoring correction detail lookup failed")
            return None

    def _test_db_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("SELECT 1")

    def _ensure_schema(self) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLite DB path was not resolved.")
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(_CREATE_PREDICTION_RUNS_SQL)
            self._ensure_prediction_run_columns(conn)
            conn.commit()

    def _ensure_prediction_run_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(prediction_runs)").fetchall()
        }
        for column, definition in _PREDICTION_RUN_COLUMN_DEFINITIONS.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE prediction_runs ADD COLUMN {column} {definition}")

    @staticmethod
    def _sqlite_value(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bool):
            return int(value)
        return value


_CREATE_PREDICTION_RUNS_SQL = """
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


_PREDICTION_RUN_COLUMN_DEFINITIONS: dict[str, str] = {
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
