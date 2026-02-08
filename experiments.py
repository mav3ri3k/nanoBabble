from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

from config import Config


class ExperimentTracker:
    def __init__(self, config: Config):
        self.db_path = Path(config.experiments_db)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()
        self._ensure_experiment(config.experiment_id)

    def _create_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                status TEXT NOT NULL,
                config_json TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')),
                ended_at TEXT,
                result_json TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
            """
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)")
        self.conn.commit()

    def _ensure_experiment(self, experiment_id: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO experiments(id) VALUES (?)", (experiment_id,))
        self.conn.commit()

    def start_run(self, experiment_id: str, config: Config) -> str:
        run_id = f"run_{int(time.time())}_{os.getpid()}"
        self.conn.execute(
            """
            INSERT INTO runs(id, experiment_id, status, config_json)
            VALUES (?, ?, 'running', ?)
            """,
            (run_id, experiment_id, json.dumps(config.to_dict(), sort_keys=True)),
        )
        self.conn.commit()
        return run_id

    def end_run(self, run_id: str, status: str, result: dict) -> None:
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = datetime('now'), result_json = ?
            WHERE id = ?
            """,
            (status, json.dumps(result, sort_keys=True), run_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
