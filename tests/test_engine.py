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
    store.update(inbox_folder=str(tmp_path / "inbox"), output_folder=str(tmp_path / "out"), setup_completed=True)
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
