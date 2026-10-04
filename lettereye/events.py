"""Activity feed shared between the processing thread and the UI.

The UI never gets called from worker threads; it polls `version` on a timer and re-reads what changed.
"""

from __future__ import annotations

import itertools
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger("lettereye")


@dataclass(frozen=True)
class Activity:
    seq: int
    time: datetime
    level: str  # info | success | warning | error | ai
    message: str
    doc_id: int | None = None


@dataclass
class ActivityFeed:
    max_entries: int = 500
    _entries: deque[Activity] = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _seq: itertools.count = field(default_factory=lambda: itertools.count(1))
    version: int = 0  # bumps on every activity *and* every data change the UI should show

    def add(self, message: str, level: str = "info", doc_id: int | None = None) -> None:
        getattr(log, "error" if level == "error" else "warning" if level == "warning" else "info")(message)
        with self._lock:
            self._entries.append(Activity(next(self._seq), datetime.now(), level, message, doc_id))
            while len(self._entries) > self.max_entries:
                self._entries.popleft()
            self.version += 1

    def touch(self) -> None:
        """Signal a data change (document/worker updated) without adding a feed entry."""
        with self._lock:
            self.version += 1

    def recent(self, limit: int = 50) -> list[Activity]:
        with self._lock:
            return list(self._entries)[-limit:][::-1]
