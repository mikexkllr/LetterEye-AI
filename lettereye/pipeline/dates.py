"""Normalizing the many ways letters print dates to ISO (YYYY-MM-DD)."""

from __future__ import annotations

import re
from datetime import date

_MONTHS = {
    # German
    "januar": 1, "jänner": 1, "jan": 1, "februar": 2, "feb": 2, "märz": 3, "maerz": 3, "mär": 3, "mrz": 3,
    "april": 4, "apr": 4, "mai": 5, "juni": 6, "jun": 6, "juli": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "oktober": 10, "okt": 10, "november": 11, "nov": 11,
    "dezember": 12, "dez": 12,
    # English
    "january": 1, "february": 2, "march": 3, "mar": 3, "may": 5, "june": 6, "july": 7, "october": 10,
    "oct": 10, "december": 12, "dec": 12,
    # French / Spanish / Italian / Dutch (common forms)
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "juin": 6, "juillet": 7, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12, "enero": 1, "febrero": 2, "marzo": 3,
    "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12, "gennaio": 1, "febbraio": 2, "aprile": 4, "maggio": 5, "giugno": 6,
    "luglio": 7, "settembre": 9, "ottobre": 10, "dicembre": 12, "januari": 1, "februari": 2, "maart": 3,
    "mei": 5, "augustus": 8,
}

_MONTH_RE = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))
_PATTERNS = [
    ("ymd", re.compile(r"\b(\d{4})[-./](\d{1,2})[-./](\d{1,2})\b")),
    ("dmy", re.compile(r"\b(\d{1,2})\s?[./-]\s?(\d{1,2})\s?[./-]\s?(\d{4}|\d{2})\b")),
    ("d_month_y", re.compile(rf"\b(\d{{1,2}})\.?\s*(?:de\s+)?({_MONTH_RE})\.?\s*(?:de\s+)?(\d{{4}})\b", re.IGNORECASE)),
    ("month_d_y", re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)),
]


def _make(year: int, month: int, day: int) -> date | None:
    if year < 100:
        year += 2000 if year < 70 else 1900
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _matches(text: str, us_order: bool) -> list[date]:
    found: list[tuple[int, date]] = []
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            g = m.groups()
            if kind == "ymd":
                d = _make(int(g[0]), int(g[1]), int(g[2]))
            elif kind == "dmy":
                first, second, year = int(g[0]), int(g[1]), int(g[2])
                if us_order and "/" in m.group(0):
                    first, second = second, first
                d = _make(year, second, first)
            elif kind == "d_month_y":
                d = _make(int(g[2]), _MONTHS[g[1].lower()], int(g[0]))
            else:
                d = _make(int(g[2]), _MONTHS[g[0].lower()], int(g[1]))
            if d:
                found.append((m.start(), d))
    return [d for _, d in sorted(found, key=lambda x: x[0])]


def normalize_date(value: str, language: str = "de") -> str:
    """'14.03.2026' / '14. März 2026' / 'March 14, 2026' -> '2026-03-14'. Empty string if not a date."""
    if not value:
        return ""
    dates = _matches(value, us_order=language == "en")
    return dates[0].isoformat() if dates else ""


def find_letter_date(text: str, language: str = "de") -> str:
    """Best guess for the letter date in a whole OCR text: the first plausible date in the top part."""
    today = date.today()
    for d in _matches(text[:2500], us_order=language == "en"):
        if 1990 <= d.year <= today.year + 1:
            return d.isoformat()
    return ""
