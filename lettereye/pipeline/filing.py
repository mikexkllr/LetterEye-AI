"""Turning a decision into a folder + file name, safely on Windows, macOS and Linux."""

from __future__ import annotations

import re
import shutil
import string
import time
from dataclasses import dataclass
from pathlib import Path

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}

FOLDER_TOKENS = {
    "worker": "Worker name",
    "recipient": "Recipient name",
    "year": "Year of the letter",
    "type": "Document type",
}
FILENAME_TOKENS = {
    "date": "Letter date (YYYY-MM-DD)",
    "year": "Year",
    "month": "Month",
    "sender": "Sender (company or person)",
    "type": "Document type",
    "subject": "Short subject",
    "recipient": "Recipient",
    "worker": "Worker",
    "original": "Original file name",
}


def sanitize(name: str, max_len: int = 80, spaces: str = " ") -> str:
    """Make one path component valid on every OS (Windows is the strictest)."""
    name = _INVALID.sub("-", str(name))
    name = " ".join(name.split())
    if spaces != " ":
        name = name.replace(" ", spaces)
    name = name.strip(" .-_")[:max_len].rstrip(" .-_")
    if name.split(".")[0].upper() in _RESERVED:
        name = f"{name}_"
    return name


class _Missing(dict):
    def __missing__(self, key: str) -> str:
        return ""


def render(template: str, values: dict[str, str]) -> str:
    """Fill {tokens}; unknown tokens become empty, doubled separators are removed."""
    allowed = {k: v for k, v in values.items()}
    try:
        text = string.Formatter().vformat(template, (), _Missing(allowed))
    except (ValueError, IndexError):
        text = template
    text = re.sub(r"([_\- ])[_\- ]+", r"\1", text)
    return text.strip("_- ")


@dataclass
class FilingValues:
    worker: str
    recipient: str
    date: str = ""
    sender: str = ""
    doc_type: str = ""
    subject: str = ""
    original: str = ""

    def as_dict(self) -> dict[str, str]:
        year, month = (self.date[:4], self.date[5:7]) if len(self.date) >= 7 else ("", "")
        return {
            "worker": self.worker, "recipient": self.recipient, "date": self.date, "year": year, "month": month,
            "sender": self.sender, "type": self.doc_type, "subject": self.subject, "original": self.original,
        }


def build_relative_path(folder_template: str, filename_template: str, values: FilingValues, suffix: str) -> Path:
    data = values.as_dict()
    folder_parts = []
    for part in re.split(r"[\\/]+", folder_template):
        rendered = sanitize(render(part, {k: sanitize(v) for k, v in data.items()}))
        if rendered:
            folder_parts.append(rendered)
    file_values = {k: sanitize(v, max_len=60, spaces="_") for k, v in data.items()}
    stem = sanitize(render(filename_template, file_values), max_len=150, spaces="_")
    if not stem:
        stem = sanitize(Path(values.original).stem, max_len=150, spaces="_") or "document"
    return Path(*folder_parts, stem + suffix.lower())


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for n in range(2, 10_000):
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(path)


def transfer(source: Path, target: Path, move: bool = True, attempts: int = 6) -> Path:
    """Move/copy with retries: on Windows, virus scanners and sync tools briefly lock new files."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target = unique_path(target)
    delay = 0.5
    for attempt in range(attempts):
        try:
            if move:
                shutil.move(str(source), str(target))
            else:
                shutil.copy2(str(source), str(target))
            return target
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return target
