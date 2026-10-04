"""SQLite storage for workers, their recipients (clients) and the processed-document history."""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import paths

WORKER_COLORS = ["#6366f1", "#06b6d4", "#d946ef", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6", "#0ea5e9", "#ec4899", "#84cc16"]

DOCUMENT_STATUSES = ("queued", "processing", "filed", "review", "failed", "ignored")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    description TEXT NOT NULL DEFAULT '',
    folder_name TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '#6366f1',
    email TEXT NOT NULL DEFAULT '',
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recipients (
    id INTEGER PRIMARY KEY,
    worker_id INTEGER NOT NULL REFERENCES workers(id) ON DELETE CASCADE,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
    aliases TEXT NOT NULL DEFAULT '[]',
    folder_name TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recipients_worker ON recipients(worker_id);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    original_name TEXT NOT NULL,
    source_path TEXT NOT NULL,
    current_path TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT '',
    worker_id INTEGER REFERENCES workers(id) ON DELETE SET NULL,
    recipient_id INTEGER REFERENCES recipients(id) ON DELETE SET NULL,
    recipient_name TEXT NOT NULL DEFAULT '',
    sender TEXT NOT NULL DEFAULT '',
    letter_date TEXT NOT NULL DEFAULT '',
    doc_type TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    confidence REAL,
    ocr_source TEXT NOT NULL DEFAULT '',
    ocr_text TEXT NOT NULL DEFAULT '',
    trace TEXT NOT NULL DEFAULT '{}',
    review_reason TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER,
    fingerprint TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_path);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
"""
_SCHEMA_VERSION = 1


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Worker:
    id: int
    name: str
    description: str = ""
    folder_name: str = ""
    color: str = WORKER_COLORS[0]
    email: str = ""
    active: bool = True
    created_at: str = ""
    recipient_count: int = 0

    @property
    def folder(self) -> str:
        return self.folder_name or self.name

    @property
    def initials(self) -> str:
        parts = [p for p in self.name.replace("_", " ").split() if p]
        return ("".join(p[0] for p in parts[:2]) or "?").upper()


@dataclass
class Recipient:
    id: int
    worker_id: int
    name: str
    aliases: list[str] = field(default_factory=list)
    folder_name: str = ""
    notes: str = ""
    created_at: str = ""
    worker_name: str = ""

    @property
    def folder(self) -> str:
        return self.folder_name or self.name


@dataclass
class Document:
    id: int
    original_name: str
    source_path: str
    current_path: str
    status: str
    stage: str
    worker_id: int | None
    recipient_id: int | None
    recipient_name: str
    sender: str
    letter_date: str
    doc_type: str
    subject: str
    confidence: float | None
    ocr_source: str
    ocr_text: str
    trace: dict[str, Any]
    review_reason: str
    error: str
    duration_ms: int | None
    created_at: str
    updated_at: str
    worker_name: str = ""
    recipient_label: str = ""


class Database:
    """Small repository over SQLite. Safe to share between the UI and the processing thread."""

    def __init__(self, path: Path | str | None = None):
        self.path = str(path or paths.database_file())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ----------------------------------------------------------------- workers
    @staticmethod
    def _worker(row: sqlite3.Row) -> Worker:
        keys = row.keys()
        return Worker(
            id=row["id"], name=row["name"], description=row["description"], folder_name=row["folder_name"],
            color=row["color"], email=row["email"], active=bool(row["active"]), created_at=row["created_at"],
            recipient_count=row["recipient_count"] if "recipient_count" in keys else 0,
        )

    def list_workers(self, active_only: bool = False) -> list[Worker]:
        sql = """SELECT w.*, (SELECT COUNT(*) FROM recipients r WHERE r.worker_id = w.id) AS recipient_count
                 FROM workers w {} ORDER BY w.name COLLATE NOCASE"""
        rows = self._query(sql.format("WHERE w.active = 1" if active_only else ""))
        return [self._worker(r) for r in rows]

    def get_worker(self, worker_id: int) -> Worker | None:
        rows = self._query(
            """SELECT w.*, (SELECT COUNT(*) FROM recipients r WHERE r.worker_id = w.id) AS recipient_count
               FROM workers w WHERE w.id = ?""",
            (worker_id,),
        )
        return self._worker(rows[0]) if rows else None

    def find_worker(self, name: str) -> Worker | None:
        rows = self._query("SELECT * FROM workers WHERE name = ? COLLATE NOCASE", (name.strip(),))
        return self._worker(rows[0]) if rows else None

    def create_worker(self, name: str, description: str = "", color: str | None = None, folder_name: str = "",
                      email: str = "") -> Worker:
        name = name.strip()
        if not name:
            raise ValueError("A worker needs a name.")
        if self.find_worker(name):
            raise ValueError(f"A worker called '{name}' already exists.")
        if color is None:
            count = self._query("SELECT COUNT(*) AS n FROM workers")[0]["n"]
            color = WORKER_COLORS[count % len(WORKER_COLORS)]
        cur = self._exec(
            "INSERT INTO workers (name, description, color, folder_name, email, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (name, description.strip(), color, folder_name.strip(), email.strip(), now_iso()),
        )
        return self.get_worker(cur.lastrowid)  # type: ignore[return-value]

    def update_worker(self, worker_id: int, **fields: Any) -> Worker:
        allowed = {"name", "description", "folder_name", "color", "email", "active"}
        changes = {k: (v.strip() if isinstance(v, str) else v) for k, v in fields.items() if k in allowed}
        if "name" in changes:
            if not changes["name"]:
                raise ValueError("A worker needs a name.")
            other = self.find_worker(changes["name"])
            if other and other.id != worker_id:
                raise ValueError(f"A worker called '{changes['name']}' already exists.")
        if changes:
            sets = ", ".join(f"{k} = :{k}" for k in changes)
            self._exec(f"UPDATE workers SET {sets} WHERE id = :id", {**changes, "id": worker_id})
        return self.get_worker(worker_id)  # type: ignore[return-value]

    def delete_worker(self, worker_id: int) -> None:
        self._exec("DELETE FROM workers WHERE id = ?", (worker_id,))

    # -------------------------------------------------------------- recipients
    @staticmethod
    def _recipient(row: sqlite3.Row) -> Recipient:
        keys = row.keys()
        return Recipient(
            id=row["id"], worker_id=row["worker_id"], name=row["name"], aliases=json.loads(row["aliases"] or "[]"),
            folder_name=row["folder_name"], notes=row["notes"], created_at=row["created_at"],
            worker_name=row["worker_name"] if "worker_name" in keys else "",
        )

    def list_recipients(self, worker_id: int | None = None, active_workers_only: bool = False) -> list[Recipient]:
        where, params = [], []
        if worker_id is not None:
            where.append("r.worker_id = ?")
            params.append(worker_id)
        if active_workers_only:
            where.append("w.active = 1")
        sql = (
            "SELECT r.*, w.name AS worker_name FROM recipients r JOIN workers w ON w.id = r.worker_id "
            + (f"WHERE {' AND '.join(where)} " if where else "")
            + "ORDER BY r.name COLLATE NOCASE"
        )
        return [self._recipient(r) for r in self._query(sql, tuple(params))]

    def get_recipient(self, recipient_id: int) -> Recipient | None:
        rows = self._query(
            "SELECT r.*, w.name AS worker_name FROM recipients r JOIN workers w ON w.id = r.worker_id WHERE r.id = ?",
            (recipient_id,),
        )
        return self._recipient(rows[0]) if rows else None

    def find_recipient(self, name: str) -> Recipient | None:
        rows = self._query(
            "SELECT r.*, w.name AS worker_name FROM recipients r JOIN workers w ON w.id = r.worker_id "
            "WHERE r.name = ? COLLATE NOCASE",
            (name.strip(),),
        )
        return self._recipient(rows[0]) if rows else None

    def create_recipient(self, worker_id: int, name: str, aliases: list[str] | None = None, folder_name: str = "",
                         notes: str = "") -> Recipient:
        name = " ".join(name.split())
        if not name:
            raise ValueError("A recipient needs a name.")
        existing = self.find_recipient(name)
        if existing:
            raise ValueError(f"'{name}' is already assigned to {existing.worker_name}.")
        cur = self._exec(
            "INSERT INTO recipients (worker_id, name, aliases, folder_name, notes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (worker_id, name, json.dumps(_clean_aliases(aliases or [], name)), folder_name.strip(), notes.strip(), now_iso()),
        )
        return self.get_recipient(cur.lastrowid)  # type: ignore[return-value]

    def update_recipient(self, recipient_id: int, **fields: Any) -> Recipient:
        current = self.get_recipient(recipient_id)
        if current is None:
            raise ValueError("Recipient not found.")
        changes: dict[str, Any] = {}
        if "name" in fields:
            name = " ".join(str(fields["name"]).split())
            if not name:
                raise ValueError("A recipient needs a name.")
            other = self.find_recipient(name)
            if other and other.id != recipient_id:
                raise ValueError(f"'{name}' is already assigned to {other.worker_name}.")
            changes["name"] = name
        if "aliases" in fields:
            changes["aliases"] = json.dumps(_clean_aliases(fields["aliases"], changes.get("name", current.name)))
        for key in ("worker_id", "folder_name", "notes"):
            if key in fields:
                changes[key] = fields[key].strip() if isinstance(fields[key], str) else fields[key]
        if changes:
            sets = ", ".join(f"{k} = :{k}" for k in changes)
            self._exec(f"UPDATE recipients SET {sets} WHERE id = :id", {**changes, "id": recipient_id})
        return self.get_recipient(recipient_id)  # type: ignore[return-value]

    def add_alias(self, recipient_id: int, alias: str) -> bool:
        """Remember another spelling of a recipient (used when a user corrects the AI). Returns True if added."""
        recipient = self.get_recipient(recipient_id)
        alias = " ".join(alias.split())
        if not recipient or not alias or alias.lower() == recipient.name.lower():
            return False
        if alias.lower() in (a.lower() for a in recipient.aliases):
            return False
        self.update_recipient(recipient_id, aliases=[*recipient.aliases, alias])
        return True

    def delete_recipient(self, recipient_id: int) -> None:
        self._exec("DELETE FROM recipients WHERE id = ?", (recipient_id,))

    # ---------------------------------------------------------- import/export
    def import_legacy_csv_folder(self, folder: str | Path) -> tuple[int, int]:
        """Import the old LetterEye format: one 'Firstname_Lastname.csv' per worker, one recipient per row."""
        workers_added = recipients_added = 0
        for csv_file in sorted(Path(folder).glob("*.csv")):
            worker_name = csv_file.stem.replace("_", " ").strip()
            worker = self.find_worker(worker_name)
            if worker is None:
                worker = self.create_worker(worker_name)
                workers_added += 1
            for row in _read_csv_rows(csv_file):
                if row and row[0].strip() and not self.find_recipient(row[0]):
                    self.create_recipient(worker.id, row[0])
                    recipients_added += 1
        return workers_added, recipients_added

    def import_csv(self, file: str | Path) -> tuple[int, int]:
        """Import a CSV with the columns: worker, recipient[, aliases separated by ';']."""
        workers_added = recipients_added = 0
        rows = _read_csv_rows(Path(file))
        if rows and [c.strip().lower() for c in rows[0][:2]] == ["worker", "recipient"]:
            rows = rows[1:]
        for row in rows:
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                continue
            worker = self.find_worker(row[0])
            if worker is None:
                worker = self.create_worker(row[0])
                workers_added += 1
            if not self.find_recipient(row[1]):
                aliases = [a for column in row[2:] for a in column.split(";") if a.strip()]
                self.create_recipient(worker.id, row[1], aliases=aliases)
                recipients_added += 1
        return workers_added, recipients_added

    def export_csv(self, file: str | Path) -> int:
        recipients = self.list_recipients()
        with open(file, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["worker", "recipient", "aliases"])
            for r in sorted(recipients, key=lambda r: (r.worker_name.lower(), r.name.lower())):
                writer.writerow([r.worker_name, r.name, ";".join(r.aliases)])
        return len(recipients)

    # --------------------------------------------------------------- documents
    def _document(self, row: sqlite3.Row) -> Document:
        keys = row.keys()
        return Document(
            id=row["id"], original_name=row["original_name"], source_path=row["source_path"],
            current_path=row["current_path"], status=row["status"], stage=row["stage"], worker_id=row["worker_id"],
            recipient_id=row["recipient_id"], recipient_name=row["recipient_name"], sender=row["sender"],
            letter_date=row["letter_date"], doc_type=row["doc_type"], subject=row["subject"],
            confidence=row["confidence"], ocr_source=row["ocr_source"], ocr_text=row["ocr_text"],
            trace=json.loads(row["trace"] or "{}"), review_reason=row["review_reason"], error=row["error"],
            duration_ms=row["duration_ms"], created_at=row["created_at"], updated_at=row["updated_at"],
            worker_name=row["worker_name"] if "worker_name" in keys and row["worker_name"] else "",
            recipient_label=row["recipient_label"] if "recipient_label" in keys and row["recipient_label"] else "",
        )

    _DOC_SELECT = """SELECT d.*, w.name AS worker_name, r.name AS recipient_label FROM documents d
                     LEFT JOIN workers w ON w.id = d.worker_id LEFT JOIN recipients r ON r.id = d.recipient_id"""

    def create_document(self, source_path: str | Path, status: str = "queued", fingerprint: str = "") -> Document:
        source = Path(source_path)
        ts = now_iso()
        cur = self._exec(
            "INSERT INTO documents (original_name, source_path, current_path, status, fingerprint, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (source.name, str(source), str(source), status, fingerprint, ts, ts),
        )
        return self.get_document(cur.lastrowid)  # type: ignore[return-value]

    def update_document(self, doc_id: int, **fields: Any) -> None:
        if not fields:
            return
        if "trace" in fields and not isinstance(fields["trace"], str):
            fields["trace"] = json.dumps(fields["trace"], ensure_ascii=False, default=str)
        fields["updated_at"] = now_iso()
        sets = ", ".join(f"{k} = :{k}" for k in fields)
        self._exec(f"UPDATE documents SET {sets} WHERE id = :id", {**fields, "id": doc_id})

    def get_document(self, doc_id: int) -> Document | None:
        rows = self._query(f"{self._DOC_SELECT} WHERE d.id = ?", (doc_id,))
        return self._document(rows[0]) if rows else None

    def list_documents(self, status: str | list[str] | None = None, search: str = "", limit: int = 200,
                       offset: int = 0) -> list[Document]:
        where, params = [], []
        if status:
            statuses = [status] if isinstance(status, str) else list(status)
            where.append(f"d.status IN ({','.join('?' * len(statuses))})")
            params += statuses
        if search.strip():
            like = f"%{search.strip()}%"
            where.append("(d.original_name LIKE ? OR d.sender LIKE ? OR d.recipient_name LIKE ? OR d.subject LIKE ? "
                         "OR w.name LIKE ? OR r.name LIKE ? OR d.doc_type LIKE ?)")
            params += [like] * 7
        sql = f"{self._DOC_SELECT} {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY d.id DESC LIMIT ? OFFSET ?"
        return [self._document(r) for r in self._query(sql, (*params, limit, offset))]

    def is_tracked(self, path: str | Path, fingerprint: str = "") -> bool:
        """True if this file is queued/being processed, or (same content stamp) was already processed.

        The second case matters in 'copy' mode, where originals stay in the inbox.
        """
        rows = self._query(
            "SELECT 1 FROM documents WHERE (current_path = ? AND status IN ('queued', 'processing')) "
            "OR (? != '' AND source_path = ? AND fingerprint = ?) LIMIT 1",
            (str(path), fingerprint, str(path), fingerprint),
        )
        return bool(rows)

    def count_by_status(self, since: str | None = None) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) AS n FROM documents"
        params: tuple = ()
        if since:
            sql += " WHERE created_at >= ?"
            params = (since,)
        counts = dict.fromkeys(DOCUMENT_STATUSES, 0)
        for row in self._query(sql + " GROUP BY status", params):
            counts[row["status"]] = row["n"]
        return counts

    def interrupted_documents(self) -> list[Document]:
        """Documents left 'queued'/'processing' because the app was closed while working on them."""
        return self.list_documents(status=["queued", "processing"], limit=100_000)

    def delete_document(self, doc_id: int) -> None:
        self._exec("DELETE FROM documents WHERE id = ?", (doc_id,))


def _clean_aliases(aliases: list[str], name: str) -> list[str]:
    seen, result = {name.strip().lower()}, []
    for alias in aliases:
        alias = " ".join(str(alias).split())
        if alias and alias.lower() not in seen:
            seen.add(alias.lower())
            result.append(alias)
    return result


def _read_csv_rows(path: Path) -> list[list[str]]:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        return []
    first = next((line for line in text.splitlines() if line.strip()), "")
    # German Excel writes ';', others ',' or tab: pick whichever the first line uses most.
    delimiter = max((",", ";", "\t"), key=first.count) if first else ","
    if first and first.count(delimiter) == 0:
        delimiter = ","
    return [row for row in csv.reader(text.splitlines(), delimiter=delimiter) if any(c.strip() for c in row)]
