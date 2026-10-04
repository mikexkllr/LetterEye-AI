"""The engine end-to-end with a real inbox folder and watcher, but a fake AI."""

import shutil
import time
from pathlib import Path

import pytest
from PIL import Image

from lettereye.ai.extraction import LetterFacts
from lettereye.events import ActivityFeed
from lettereye.pipeline.processor import Processor
from lettereye.services.engine import Engine
from lettereye.settings import SettingsStore

from .conftest import CLEAN_LETTER, FakeExtractor, FakeOCR


def wait_for(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


@pytest.fixture
def engine(db, tmp_path, decision_engine):
    store = SettingsStore(tmp_path / "settings.json")
    store.update(inbox_folder=str(tmp_path / "inbox"), output_folder=str(tmp_path / "out"), setup_completed=True,
                 workflow_mode="automatic")
    (tmp_path / "inbox").mkdir()
    john = db.create_worker("John Doe")
    db.create_recipient(john.id, "Bob Smith")
    eng = Engine(db, store, ActivityFeed())
    texts = {"bob": CLEAN_LETTER, "unknown": CLEAN_LETTER.replace("Bob Smith", "Klaus Weber").replace("Herr Smith", "Herr Weber")}

    class OcrByName(FakeOCR):
        def __init__(self):
            super().__init__("")
            self.current = ""

        def recognize(self, images):
            self.text = texts[self.current]
            return super().recognize(images)

    ocr = OcrByName()

    def build(settings):
        facts = LetterFacts(sender_organization="Stadtwerke München", letter_date="14.03.2026", recipient_name="Herrn B. Smith")
        processor = Processor(settings, decision_engine, FakeExtractor(facts), ocr, None, eng.directory)
        original = processor.analyze

        def analyze(path, on_stage=None):
            ocr.current = "bob" if "bob" in Path(path).name else "unknown"
            return original(path, on_stage)

        processor.analyze = analyze
        return processor

    eng.preflight = lambda settings=None: []
    eng.build_processor = build
    eng._warm_up = lambda settings: None
    yield eng
    eng.stop()


def drop(folder: Path, name: str) -> Path:
    tmp = folder.parent / f"tmp-{name}"
    Image.new("RGB", (200, 280), "white").save(tmp, format="PNG")
    target = folder / name
    shutil.move(tmp, target)
    return target


def test_letters_are_filed_or_parked_for_review(engine, db, tmp_path):
    assert engine.start() == []
    drop(tmp_path / "inbox", "bob.png")
    drop(tmp_path / "inbox", "other.png")
    assert wait_for(lambda: len([d for d in db.list_documents() if d.status in ("filed", "review")]) == 2)

    filed = db.list_documents(status="filed")[0]
    assert Path(filed.current_path) == tmp_path / "out" / "John Doe" / "Bob Smith" / "2026-03-14_Stadtwerke_München_Rechnung.png"
    assert Path(filed.current_path).exists()
    assert not (tmp_path / "inbox" / "bob.png").exists()

    review = db.list_documents(status="review")[0]
    assert Path(review.current_path) == tmp_path / "out" / "_Review" / "other.png"
    assert review.trace["decisions"][0]["key"] == "recipient"


def test_manual_assignment_files_and_learns_the_spelling(engine, db, tmp_path):
    engine.start()
    drop(tmp_path / "inbox", "other.png")
    assert wait_for(lambda: db.count_by_status()["review"] == 1)
    doc = db.list_documents(status="review")[0]
    bob = db.find_recipient("Bob Smith")
    target = engine.assign(doc.id, recipient_id=bob.id, doc_type="Mahnung")
    assert target.exists() and target.parent == tmp_path / "out" / "John Doe" / "Bob Smith"
    assert "Mahnung" in target.name
    assert db.get_document(doc.id).status == "filed"
    assert db.get_recipient(bob.id).aliases == ["B. Smith"]


def test_copy_mode_keeps_originals_and_does_not_reprocess(engine, db, tmp_path):
    engine.store.update(move_files=False)
    engine.start()
    drop(tmp_path / "inbox", "bob.png")
    assert wait_for(lambda: db.count_by_status()["filed"] == 1)
    assert (tmp_path / "inbox" / "bob.png").exists()
    engine._watcher.scan()  # the rescan sees the original again...
    time.sleep(1.0)
    assert len(db.list_documents()) == 1  # ...but does not process it twice


def test_start_reports_problems(db, tmp_path):
    store = SettingsStore(tmp_path / "s.json")
    eng = Engine(db, store, ActivityFeed())
    problems = eng.start()
    assert problems and eng.state == "stopped"
    assert any("inbox" in p.lower() for p in problems)


def test_approval_mode_waits_for_a_human_and_records_feedback(engine, db, tmp_path):
    engine.store.update(workflow_mode="approve")
    engine.start()
    drop(tmp_path / "inbox", "bob.png")
    assert wait_for(lambda: db.count_by_status()["pending"] == 1)
    doc = db.list_documents(status="pending")[0]
    assert Path(doc.current_path).parent == tmp_path / "out" / "_Review"  # nothing filed yet
    assert doc.proposal["recipient"] == "Bob Smith" and doc.proposal["verdict"] == "filed"

    target = engine.approve(doc.id)
    assert target == tmp_path / "out" / "John Doe" / "Bob Smith" / "2026-03-14_Stadtwerke_München_Rechnung.png"
    feedback = db.last_feedback(doc.id)
    assert feedback.action == "accepted" and feedback.changed == [] and feedback.verified
    assert feedback.mode == "approve" and feedback.seconds_to_decide is not None

    engine.undo(doc.id)  # back into the queue, file back in the review folder
    doc = db.get_document(doc.id)
    assert doc.status == "pending" and Path(doc.current_path).parent == tmp_path / "out" / "_Review"
    assert not target.exists() and db.last_feedback(doc.id) is None


def test_corrections_are_recorded_field_by_field(engine, db, tmp_path):
    engine.store.update(workflow_mode="approve")
    jane = db.create_worker("Jane Smith")
    eve = db.create_recipient(jane.id, "Eve Adams")
    engine.start()
    drop(tmp_path / "inbox", "bob.png")
    assert wait_for(lambda: db.count_by_status()["pending"] == 1)
    doc = db.list_documents(status="pending")[0]
    target = engine.approve(doc.id, recipient_id=eve.id, doc_type="Mahnung", subject="Stromrechnung")
    assert target.parent == tmp_path / "out" / "Jane Smith" / "Eve Adams"
    feedback = db.last_feedback(doc.id)
    assert feedback.action == "corrected"
    assert feedback.changed == ["worker", "recipient", "doc_type", "subject"]
    assert feedback.ai["recipient"] == "Bob Smith" and feedback.final["recipient"] == "Eve Adams"

    # fixing it again after filing moves the file and replaces the earlier verdict
    bob = db.find_recipient("Bob Smith")
    fixed = engine.approve(doc.id, recipient_id=bob.id, doc_type="Rechnung", subject=doc.subject)
    assert fixed.parent == tmp_path / "out" / "John Doe" / "Bob Smith" and not target.exists()
    verdicts = [f for f in db.list_feedback() if f.doc_id == doc.id]
    assert [f.action for f in verdicts] == ["confirmed"]  # the correction was superseded


def test_reject_sets_the_letter_aside(engine, db, tmp_path):
    engine.store.update(workflow_mode="approve")
    engine.start()
    drop(tmp_path / "inbox", "bob.png")
    assert wait_for(lambda: db.count_by_status()["pending"] == 1)
    doc = db.list_documents(status="pending")[0]
    target = engine.reject(doc.id, "Advertising")
    assert target.parent == tmp_path / "out" / "_Rejected"
    assert db.get_document(doc.id).status == "ignored"
    assert db.last_feedback(doc.id).action == "rejected"
    engine.undo(doc.id)
    assert db.get_document(doc.id).status == "pending"


def test_auto_filed_letters_can_be_confirmed(engine, db, tmp_path):
    engine.start()
    drop(tmp_path / "inbox", "bob.png")
    assert wait_for(lambda: db.count_by_status()["filed"] == 1)
    doc = db.list_documents(status="filed")[0]
    assert db.last_feedback(doc.id).action == "auto_filed"
    engine.confirm(doc.id)
    assert db.last_feedback(doc.id).action == "confirmed"
    assert Path(db.get_document(doc.id).current_path).exists()
