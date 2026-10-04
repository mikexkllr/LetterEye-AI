import pytest

from .conftest import LEGACY_CSV


def test_workers_and_recipients_crud(db):
    worker = db.create_worker("John Doe", description="Energy")
    assert worker.initials == "JD"
    with pytest.raises(ValueError):
        db.create_worker("john doe")  # names are unique, case-insensitive
    bob = db.create_recipient(worker.id, "  Bob   Smith ")
    assert bob.name == "Bob Smith" and bob.worker_name == "John Doe"
    with pytest.raises(ValueError, match="already assigned"):
        db.create_recipient(worker.id, "bob smith")
    other = db.create_worker("Jane Smith")
    db.update_recipient(bob.id, worker_id=other.id, aliases=["B. Smith", "b. smith", "Bob Smith"])
    moved = db.get_recipient(bob.id)
    assert moved.worker_id == other.id and moved.aliases == ["B. Smith"]
    assert [w.recipient_count for w in db.list_workers()] == [1, 0]  # Jane, John
    db.delete_worker(other.id)
    assert db.get_recipient(bob.id) is None  # recipients go with their worker


def test_add_alias_learns_once(db):
    worker = db.create_worker("John Doe")
    bob = db.create_recipient(worker.id, "Bob Smith")
    assert db.add_alias(bob.id, "B. Smith")
    assert not db.add_alias(bob.id, "b. smith")
    assert not db.add_alias(bob.id, "Bob Smith")
    assert db.get_recipient(bob.id).aliases == ["B. Smith"]


def test_import_legacy_csv_folder(db):
    workers, recipients = db.import_legacy_csv_folder(LEGACY_CSV)
    assert workers == 2 and recipients == 9
    assert db.find_recipient("Herr Bauer").worker_name == "John Doe"
    assert db.import_legacy_csv_folder(LEGACY_CSV) == (0, 0)  # idempotent


def test_import_and_export_csv(db, tmp_path):
    source = tmp_path / "people.csv"
    source.write_text("worker;recipient;aliases\nAnna Berg;ACME GmbH;ACME;Acme Corp\nAnna Berg;Max Muster;\n",
                      encoding="utf-8")
    assert db.import_csv(source) == (1, 2)
    assert db.find_recipient("ACME GmbH").aliases == ["ACME", "Acme Corp"]
    out = tmp_path / "export.csv"
    assert db.export_csv(out) == 2
    assert "Anna Berg,ACME GmbH,ACME;Acme Corp" in out.read_text(encoding="utf-8-sig")


def test_documents_tracking_and_fingerprints(db, tmp_path):
    path = tmp_path / "scan.pdf"
    doc = db.create_document(path, fingerprint="10:1")
    assert db.is_tracked(path)
    db.update_document(doc.id, status="filed", current_path=str(tmp_path / "out.pdf"), trace={"a": 1}, confidence=0.9)
    assert not db.is_tracked(path)  # no longer queued...
    assert db.is_tracked(path, "10:1")  # ...but the same file content was already processed (copy mode)
    assert not db.is_tracked(path, "11:2")  # a new scan with the same name is processed
    stored = db.get_document(doc.id)
    assert stored.trace == {"a": 1} and stored.status == "filed"
    assert db.count_by_status()["filed"] == 1
    assert db.list_documents(search="scan")[0].id == doc.id
