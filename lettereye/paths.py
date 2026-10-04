"""Where LetterEye keeps its data. Everything lives in the per-user app-data folder:

* Windows: %LOCALAPPDATA%\\LetterEye
* macOS:   ~/Library/Application Support/LetterEye
* Linux:   ~/.local/share/LetterEye

Set LETTEREYE_DATA_DIR (or pass --data-dir) to use a different folder, e.g. a portable install.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import user_data_dir

APP_DIR_NAME = "LetterEye"
_ENV = "LETTEREYE_DATA_DIR"


def set_data_dir(path: str | os.PathLike) -> None:
    os.environ[_ENV] = str(Path(path).expanduser().resolve())


def data_dir() -> Path:
    override = os.environ.get(_ENV)
    path = Path(override) if override else Path(user_data_dir(APP_DIR_NAME, appauthor=False, roaming=False))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sub(name: str) -> Path:
    path = data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def settings_file() -> Path:
    return data_dir() / "settings.json"


def database_file() -> Path:
    return data_dir() / "lettereye.db"


def log_dir() -> Path:
    return _sub("logs")


def ocr_models_dir() -> Path:
    """PP-OCR model cache (kept out of site-packages so installs in Program Files work)."""
    return _sub("ocr-models")


def previews_dir() -> Path:
    return _sub("previews")


def assets_dir() -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    bundled = base / "lettereye" / "assets"
    return bundled if bundled.exists() else Path(__file__).resolve().parent / "assets"


def default_inbox() -> Path:
    return Path.home() / "LetterEye" / "Inbox"


def default_output() -> Path:
    return Path.home() / "LetterEye" / "Sorted"
