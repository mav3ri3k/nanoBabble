from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from config import Config

_REMOTE_USER_HOST = "apurva@192.168.50.153"
_REMOTE_DIR = "/mnt/hdd_volume2/apurva_150gb/nanoBabble"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrackingStore:
    def __init__(self, db_path: str):
        path = Path(db_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute("PRAGMA busy_timeout=5000;")
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL REFERENCES experiments(id),
                config_json TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                run_desc TEXT,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL REFERENCES runs(id),
                step INTEGER NOT NULL,
                loss REAL NOT NULL,
                PRIMARY KEY (run_id, step)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs(experiment_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_metrics_run_step ON metrics(run_id, step)"
        )
        self._conn.commit()

    def get_experiment_id(self, name: str) -> str | None:
        row = self._conn.execute(
            "SELECT id FROM experiments WHERE name = ?",
            (name,),
        ).fetchone()
        return None if row is None else str(row[0])

    def create_experiment(self, name: str, description: str = "") -> str:
        existing = self.get_experiment_id(name)
        if existing is not None:
            return existing
        experiment_id = str(uuid4())
        self._conn.execute(
            "INSERT INTO experiments (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (experiment_id, name, description, _utc_now()),
        )
        self._conn.commit()
        return experiment_id

    def list_experiments(self) -> list[tuple[str, str, str, str]]:
        rows = self._conn.execute(
            "SELECT id, name, COALESCE(description, ''), created_at FROM experiments ORDER BY created_at"
        ).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in rows]

    def update_experiment(
        self,
        name: str,
        *,
        new_name: str | None = None,
        description: str | None = None,
    ) -> None:
        exp_id = self.get_experiment_id(name)
        if exp_id is None:
            raise ValueError(f"Experiment `{name}` does not exist.")
        if new_name is not None:
            self._conn.execute("UPDATE experiments SET name = ? WHERE id = ?", (new_name, exp_id))
        if description is not None:
            self._conn.execute(
                "UPDATE experiments SET description = ? WHERE id = ?",
                (description, exp_id),
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class RunTracker:
    def __init__(self, cfg: Config):
        self.enabled = bool(cfg.enable_metrics)
        self._cfg = cfg
        self._store: TrackingStore | None = None
        self._conn: sqlite3.Connection | None = None
        self._run_id: str | None = None
        self._loss_rows: list[tuple[str, int, float]] = []
        self._flush_every = max(1, int(cfg.metrics_flush_every))

        if not self.enabled:
            return

        self._store = TrackingStore(cfg.metrics_db_path)
        self._conn = self._store._conn

    def start_run(self) -> str | None:
        if not self.enabled:
            return None
        assert self._store is not None
        assert self._conn is not None

        experiment_id = self._store.get_experiment_id(self._cfg.experiment_name)
        if experiment_id is None:
            raise ValueError(
                f"Experiment `{self._cfg.experiment_name}` not found. "
                f"Create it first using: `uv run tracking.py create --name \"{self._cfg.experiment_name}\"`."
            )

        self._run_id = str(uuid4())
        config_json = json.dumps(
            self._cfg.to_plain_dict(),
            sort_keys=True,
            separators=(",", ":"),
        )
        self._conn.execute(
            """
            INSERT INTO runs (
                id, experiment_id, config_json, config_hash, run_desc, status, started_at, ended_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._run_id,
                experiment_id,
                config_json,
                self._cfg.config_hash(),
                self._cfg.run_description,
                "running",
                _utc_now(),
                None,
            ),
        )
        self._conn.commit()
        return self._run_id

    def _flush_metrics(self) -> None:
        if not self.enabled or not self._loss_rows:
            return
        assert self._conn is not None
        self._conn.executemany(
            "INSERT OR REPLACE INTO metrics (run_id, step, loss) VALUES (?, ?, ?)",
            self._loss_rows,
        )
        self._conn.commit()
        self._loss_rows.clear()

    def log_loss(self, step: int, loss: float) -> None:
        if not self.enabled:
            return
        if self._run_id is None:
            raise RuntimeError("RunTracker.start_run() must be called before logging metrics.")
        self._loss_rows.append((self._run_id, int(step), float(loss)))
        if len(self._loss_rows) >= self._flush_every:
            self._flush_metrics()

    def finish_run(self, status: str) -> None:
        if not self.enabled:
            return
        assert self._conn is not None
        if self._run_id is None:
            return
        self._flush_metrics()
        self._conn.execute(
            "UPDATE runs SET status = ?, ended_at = ? WHERE id = ?",
            (status, _utc_now(), self._run_id),
        )
        self._conn.commit()

    def close(self) -> None:
        if self._store is None:
            return
        self._flush_metrics()
        self._store.close()
        self._store = None
        self._conn = None


def create_tracker(cfg: Config) -> RunTracker:
    return RunTracker(cfg)


def _cli_list(store: TrackingStore) -> int:
    rows = store.list_experiments()
    if not rows:
        print("No experiments found.")
        return 0
    for exp_id, name, description, created_at in rows:
        print(f"id={exp_id} name={name} created_at={created_at} description={description}")
    return 0


def _cli_create(store: TrackingStore, name: str, description: str) -> int:
    existing = store.get_experiment_id(name)
    if existing is not None:
        print(f"experiment already exists: name={name} experiment_id={existing}")
        return 0
    exp_id = store.create_experiment(name=name, description=description)
    print(f"experiment created: name={name} experiment_id={exp_id}")
    return 0


def _cli_update(
    store: TrackingStore,
    name: str,
    new_name: str | None,
    description: str | None,
) -> int:
    if new_name is None and description is None:
        raise ValueError("Pass at least one of `--new-name` or `--description`.")
    store.update_experiment(name=name, new_name=new_name, description=description)
    print(f"updated name={name}")
    return 0


def _cli_sync(db_path: str, to: str, dry_run: bool) -> int:
    local_db = Path(db_path).resolve()
    remote_dir = f"{_REMOTE_USER_HOST}:{_REMOTE_DIR}"
    remote_db = f"{_REMOTE_USER_HOST}:{_REMOTE_DIR}/experiments.db"

    if to == "server":
        cmd = ["rsync", "-az", str(local_db), remote_dir]
    else:
        cmd = ["rsync", "-az", remote_db, str(local_db)]

    if dry_run:
        print("dry-run command:", " ".join(cmd))
        return 0

    subprocess.run(cmd, check=True)
    print(f"sync complete: to={to}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Experiment tracking CLI")
    parser.add_argument("--db", type=str, default="experiments.db", help="Path to SQLite DB.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List experiments.")

    p_create = sub.add_parser("create", help="Create experiment.")
    p_create.add_argument("--name", type=str, required=True)
    p_create.add_argument("--description", type=str, default="")

    p_update = sub.add_parser("update", help="Update experiment name/description.")
    p_update.add_argument("--name", type=str, required=True, help="Current experiment name.")
    p_update.add_argument("--new-name", type=str, default=None)
    p_update.add_argument("--description", type=str, default=None)

    p_sync = sub.add_parser("sync", help="Sync experiments.db between laptop and server.")
    p_sync.add_argument("--to", type=str, choices=["server", "laptop"], required=True)
    p_sync.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd == "sync":
        return _cli_sync(db_path=args.db, to=args.to, dry_run=bool(args.dry_run))

    store = TrackingStore(args.db)
    try:
        if args.cmd == "list":
            return _cli_list(store)
        if args.cmd == "create":
            return _cli_create(store, name=args.name, description=args.description)
        if args.cmd == "update":
            return _cli_update(
                store,
                name=args.name,
                new_name=args.new_name,
                description=args.description,
            )
        return 1
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
