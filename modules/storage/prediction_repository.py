import logging
import sqlite3
from pathlib import Path
from typing import Mapping

from modules.storage.sqlite_schema import PREDICTION_RUN_COLUMNS, sqlite_value

logger = logging.getLogger(__name__)


class PredictionRunRepository:
    """Write and deduplicate prediction_run rows."""

    def __init__(self, db_path: Path | None, *, enabled: bool) -> None:
        self._db_path = db_path
        self._enabled = enabled

    def insert_prediction_run(self, record: Mapping[str, object]) -> None:
        if not self._enabled or self._db_path is None:
            return

        values = [sqlite_value(record.get(column)) for column in PREDICTION_RUN_COLUMNS]
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
