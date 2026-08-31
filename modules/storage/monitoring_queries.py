import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class MonitoringQueries:
    """Read monitoring summaries and layer-level drift details."""

    def __init__(self, db_path: Path | None, *, enabled: bool) -> None:
        self._db_path = db_path
        self._enabled = enabled

    def correction_summary(self, analysis_id: str) -> tuple[int, int] | None:
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
                        qc_warning,
                        normalization_skipped_reason,
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
