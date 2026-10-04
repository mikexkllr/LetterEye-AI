"""Cleaning person names read from letters."""

from __future__ import annotations

import re

# Salutations and titles in front of a name ("Herrn Bob Smith", "Frau Dr. G. Kelly", "Mrs. Adams").
# Single-letter initials such as "M." are kept: they are part of the name.
_PREFIX = re.compile(
    r"^(?:(?:herrn?|frau|fräulein|familie|eheleute|mr|mrs|ms|miss|mx|dr|prof|dipl\.?-?\w*|monsieur|madame|mme|"
    r"mlle|señora?|sra?|signora?|sig|dhr|mevr|mw)\.?\s+)+",
    re.IGNORECASE,
)


def clean_person_name(name: str) -> str:
    name = " ".join((name or "").replace("\n", " ").split()).strip(" ,;:")
    cleaned = _PREFIX.sub("", name).strip()
    return cleaned or name
