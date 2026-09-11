import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Mapping

import yaml

from modules.storage.monitoring_queries import MonitoringQueries
from modules.storage.prediction_repository import PredictionRunRepository
from modules.storage.sqlite_schema import ensure_prediction_schema

logger = logging.getLogger(__name__)


class SQLitePredictionLogger:
    """Facade for prediction-run storage, duplicate checks, and monitoring queries."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        self._app_root = self._config_path.parent.parent
        self._db_path: Path | None = None
        self._enabled = False
        self._repository = PredictionRunRepository(None, enabled=False)
        self._queries = MonitoringQueries(None, enabled=False)

        try:
            cfg = self._load_config()
            self._db_path = self._select_db_path(cfg)
            self._ensure_schema()
            self._enabled = True
            self._repository = PredictionRunRepository(self._db_path, enabled=True)
            self._queries = MonitoringQueries(self._db_path, enabled=True)
            logger.info("SQLite monitoring DB ready: %s", self._db_path)
        except Exception:
            logger.exception("SQLite monitoring disabled: initialization failed")
            self._db_path = None
            self._enabled = False

    @property
    def db_path(self) -> Path | None:
        return self._db_path

    def insert_prediction_run(self, record: Mapping[str, object]) -> None:
        self._repository.insert_prediction_run(record)

    def existing_prediction_cells(
        self,
        pairs: set[tuple[str, str]],
    ) -> set[tuple[str, str]]:
        return self._repository.existing_prediction_cells(pairs)

    def delete_prediction_cells(self, pairs: set[tuple[str, str]]) -> None:
        self._repository.delete_prediction_cells(pairs)

    def correction_summary(self, analysis_id: str) -> tuple[int, int] | None:
        return self._queries.correction_summary(analysis_id)

    def correction_details(self, analysis_id: str) -> list[dict[str, object]] | None:
        return self._queries.correction_details(analysis_id)

    def upsert_qc_failure(
        self,
        *,
        image_path: str | Path,
        folder_path: str | Path,
        failure_reason: str,
        status: str = "FAIL",
    ) -> None:
        if not self._enabled or self._db_path is None:
            return
        now = datetime.now().isoformat(timespec="seconds")
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO qc_failures (
                        image_path,
                        folder_path,
                        failure_reason,
                        status,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_path) DO UPDATE SET
                        folder_path = excluded.folder_path,
                        failure_reason = excluded.failure_reason,
                        status = excluded.status
                    """,
                    (
                        str(image_path),
                        str(folder_path),
                        failure_reason,
                        status,
                        now,
                    ),
                )
                conn.commit()
        except Exception:
            logger.exception("SQLite QC failure upsert failed")

    def list_qc_failures(self, statuses: tuple[str, ...] = ("FAIL", "PENDING")) -> list[dict[str, object]]:
        if not self._enabled or self._db_path is None:
            return []
        placeholders = ", ".join("?" for _ in statuses)
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"""
                    SELECT
                        image_path,
                        folder_path,
                        failure_reason,
                        status,
                        created_at
                    FROM qc_failures
                    WHERE status IN ({placeholders})
                    ORDER BY created_at, image_path
                    """,
                    statuses,
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            logger.exception("SQLite QC failure list failed")
            return []

    def update_qc_failure_status(
        self,
        image_path: str | Path,
        status: str,
        *,
        failure_reason: str | None = None,
    ) -> None:
        if not self._enabled or self._db_path is None:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                if failure_reason is None:
                    conn.execute(
                        "UPDATE qc_failures SET status = ? WHERE image_path = ?",
                        (status, str(image_path)),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE qc_failures
                        SET status = ?, failure_reason = ?
                        WHERE image_path = ?
                        """,
                        (status, failure_reason, str(image_path)),
                    )
                conn.commit()
        except Exception:
            logger.exception("SQLite QC failure status update failed")

    def _load_config(self) -> dict:
        with open(self._config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _select_db_path(self, cfg: dict) -> Path:
        storage_cfg = cfg.get("storage", {}) or {}
        sqlite_cfg = storage_cfg.get("sqlite", {}) or {}

        db_name = (
            sqlite_cfg.get("path")
            or storage_cfg.get("sqlite_path")
            or cfg.get("sqlite_path")
            or "photo_jig_vision.db"
        )
        shared_path = (
            sqlite_cfg.get("shared_path")
            or storage_cfg.get("shared_path")
            or cfg.get("sqlite_shared_path")
        )

        local_path = self._resolve_db_path(db_name)
        if shared_path:
            shared_db_path = Path(shared_path) / Path(db_name).name
            try:
                self._test_db_path(shared_db_path)
                return shared_db_path
            except Exception:
                logger.warning(
                    "Shared SQLite path unavailable, falling back to local DB: %s",
                    shared_db_path,
                    exc_info=True,
                )

        self._test_db_path(local_path)
        return local_path

    def _resolve_db_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self._app_root / path

    def _test_db_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("SELECT 1")

    def _ensure_schema(self) -> None:
        if self._db_path is None:
            raise RuntimeError("SQLite DB path was not resolved.")
        with sqlite3.connect(self._db_path) as conn:
            ensure_prediction_schema(conn)
            conn.commit()
