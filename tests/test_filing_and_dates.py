from pathlib import Path

import pytest

from lettereye.pipeline.dates import find_letter_date, normalize_date
from lettereye.pipeline.filing import (
    FilingValues,
    build_relative_path,
    render,
    sanitize,
    transfer,
    unique_path,
)


@pytest.mark.parametrize("raw, expected", [
    ('a<b>c:"d"/e\\f|g?h*', "a-b-c--d--e-f-g-h"),
    ("CON", "CON_"),
    ("lpt1.txt", "lpt1.txt_"),
    ("  trailing dots...  ", "trailing dots"),
    ("", ""),
])
def test_sanitize_is_windows_safe(raw, expected):
    assert sanitize(raw) == expected


def test_render_ignores_missing_tokens_and_collapses_separators():
    assert render("{date}_{sender}_{type}", {"date": "2026-03-14", "type": "Rechnung"}) == "2026-03-14_Rechnung"
    assert render("{unknown}", {}) == ""


def test_build_relative_path():
    values = FilingValues(worker="John Doe", recipient="Bob Smith", date="2026-03-14", sender="Stadtwerke München GmbH",
                          doc_type="Rechnung", subject="Jahresabrechnung", original="scan_1")
    rel = build_relative_path("{worker}/{recipient}", "{date}_{sender}_{type}", values, ".PDF")
    assert rel == Path("John Doe", "Bob Smith", "2026-03-14_Stadtwerke_München_GmbH_Rechnung.pdf")


def test_build_relative_path_falls_back_to_original_name():
    values = FilingValues(worker="W", recipient="R", original="scan 7")
    assert build_relative_path("{worker}/{year}", "{date}", values, ".pdf") == Path("W", "scan_7.pdf")


def test_unique_path_and_transfer(tmp_path):
    source = tmp_path / "in.pdf"
    source.write_bytes(b"%PDF-1.4")
    target = tmp_path / "out" / "doc.pdf"
    first = transfer(source, target, move=False)
    assert first == target and source.exists()
    second = transfer(source, target, move=True)
    assert second.name == "doc (2).pdf" and not source.exists()
    assert unique_path(target).name == "doc (3).pdf"


@pytest.mark.parametrize("raw, expected", [
    ("14.03.2026", "2026-03-14"),
    ("14.3.26", "2026-03-14"),
    ("2026-03-14", "2026-03-14"),
    ("14. März 2026", "2026-03-14"),
    ("02. Februar 2026", "2026-02-02"),
    ("March 14, 2026", "2026-03-14"),
    ("14 March 2026", "2026-03-14"),
    ("31.02.2026", ""),
    ("no date", ""),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw, "de") == expected


def test_us_order_for_english():
    assert normalize_date("03/14/2026", "en") == "2026-03-14"


def test_find_letter_date_skips_implausible_years():
    text = "Kundennummer 01.01.1890\nBerlin, den 20.01.2026\nZahlbar bis 23.02.2026"
    assert find_letter_date(text) == "2026-01-20"


@pytest.mark.parametrize("raw, expected", [
    ("Frau G. Kelly", "G. Kelly"),
    ("Herrn Dr. Bob Smith", "Bob Smith"),
    ("Mrs. Eve Adams", "Eve Adams"),
    ("M. Kelly", "M. Kelly"),
    ("Familie Weber", "Weber"),
    ("ACME GmbH", "ACME GmbH"),
    ("Frau", "Frau"),
])
def test_clean_person_name(raw, expected):
    from lettereye.pipeline.names import clean_person_name

    assert clean_person_name(raw) == expected
