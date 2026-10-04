"""Application services shared by all UI pages."""

from __future__ import annotations

from dataclasses import dataclass

from ..db import Database
from ..events import ActivityFeed
from ..services.engine import Engine
from ..settings import SettingsStore


@dataclass
class AppContext:
    db: Database
    store: SettingsStore
    feed: ActivityFeed
    engine: Engine
    native: bool = False


_ctx: AppContext | None = None


def init(ctx: AppContext) -> None:
    global _ctx
    _ctx = ctx


def ctx() -> AppContext:
    if _ctx is None:
        raise RuntimeError("UI context not initialised")
    return _ctx
