from lettereye.ai.extraction import LetterFacts
from lettereye.pipeline.processor import Directory, Processor
from lettereye.settings import Settings

from .conftest import CLEAN_LETTER, GARBLED, FakeExtractor, FakeOCR


def make_processor(db, decision_engine, fast_text, llm_text=CLEAN_LETTER, **settings):
    s = Settings(confidence_threshold=0.8, **settings)
    fast = FakeOCR(fast_text, confidence=0.98 if fast_text != GARBLED else 0.55)
    llm = FakeOCR(llm_text, source="ocr_llm", model="glm-ocr")
    facts = LetterFacts(sender_organization="Stadtwerke München GmbH", recipient_name="Bob Smith",
                        letter_date="14.03.2026", subject="Jahresabrechnung Strom")
    extractor = FakeExtractor(facts)
    processor = Processor(s, decision_engine, extractor, fast, llm,
                          lambda: Directory(db.list_workers(active_only=True), db.list_recipients(active_workers_only=True)))
    return processor, fast, llm, extractor


def seed(db):
    john = db.create_worker("John Doe", description="Energy suppliers and rent")
    jane = db.create_worker("Jane Smith")
    db.create_worker("Maria Schulz", description="Everything from the Finanzamt (tax office)")
    for name in ("Alice Johnson", "Bob Smith", "Charlie Brown"):
        db.create_recipient(john.id, name)
    for name in ("Eve Adams", "Grace Kelly"):
        db.create_recipient(jane.id, name)
    return john, jane


def test_clean_letter_is_filed_without_the_ocr_llm(db, decision_engine, scan_image):
    john, _ = seed(db)
    processor, fast, llm, _ = make_processor(db, decision_engine, CLEAN_LETTER)
    analysis = processor.analyze(scan_image)
    assert analysis.status == "filed"
    assert analysis.decision.recipient.name == "Bob Smith"
    assert analysis.decision.worker.id == john.id
    assert analysis.confidence >= 0.8
    assert analysis.decision.doc_type == "Rechnung"
    assert analysis.letter_date == "2026-03-14"
    assert not analysis.escalated
    assert fast.calls == 1 and llm.calls == 0
    keys = [d["key"] for d in analysis.decision.decisions]
    assert keys == ["recipient", "verify_recipient", "doc_type"]


def test_bad_ocr_escalates_to_the_ocr_llm(db, decision_engine, scan_image):
    seed(db)
    processor, fast, llm, extractor = make_processor(db, decision_engine, GARBLED)
    analysis = processor.analyze(scan_image)
    assert analysis.escalated
    assert llm.calls == 1
    assert analysis.status == "filed"
    assert analysis.decision.recipient.name == "Bob Smith"
    assert analysis.ocr.source == "ocr_llm"
    assert extractor.texts == [CLEAN_LETTER]  # facts come from the better text
    rounds = {d["round"] for d in analysis.decision.decisions}
    assert rounds == {"decisions", "decisions_after_ocr_llm"}


def test_fast_only_mode_never_uses_the_ocr_llm(db, decision_engine, scan_image):
    seed(db)
    processor, _, llm, _ = make_processor(db, decision_engine, GARBLED, ocr_mode="fast_only")
    analysis = processor.analyze(scan_image)
    assert llm.calls == 0
    assert analysis.status == "review"


def test_unknown_recipient_with_clean_text_goes_to_review_without_escalation(db, decision_engine, scan_image):
    seed(db)
    text = CLEAN_LETTER.replace("Bob Smith", "Klaus Weber").replace("Herr Smith", "Herr Weber")
    processor, _, llm, extractor = make_processor(db, decision_engine, text)
    extractor.facts = LetterFacts(recipient_name="Klaus Weber", sender_organization="Stadtwerke")
    analysis = processor.analyze(scan_image)
    assert analysis.status == "review"
    assert "Klaus Weber" in analysis.reason
    assert llm.calls == 0  # the decision model was sure: better OCR would not change anything


def test_unknown_recipient_is_routed_by_worker_description(db, decision_engine, scan_image):
    seed(db)
    text = "Finanzamt Berlin\nHerrn Klaus Weber\nSehr geehrter Herr Weber,\nIhr Steuerbescheid"
    processor, _, _, extractor = make_processor(db, decision_engine, text, auto_create_recipients=True)
    extractor.facts = LetterFacts(recipient_name="Klaus Weber", sender_organization="Finanzamt Berlin")
    analysis = processor.analyze(scan_image)
    assert analysis.decision.worker.name == "Maria Schulz"
    assert analysis.status == "filed"
    assert analysis.new_recipient_name == "Klaus Weber"


def test_aliases_are_offered_to_the_decision_model(db, decision_engine, scan_image):
    _, jane = seed(db)
    grace = db.find_recipient("Grace Kelly")
    db.add_alias(grace.id, "G. Kelly")
    text = "Allianz\nFrau G. Kelly\nSehr geehrte Frau Kelly,\nKündigungsbestätigung"
    processor, *_ = make_processor(db, decision_engine, text)
    analysis = processor.analyze(scan_image)
    assert analysis.decision.recipient.name == "Grace Kelly"
    assert analysis.status == "filed"


def test_no_workers_yet_goes_to_review(db, decision_engine, scan_image):
    processor, *_ = make_processor(db, decision_engine, CLEAN_LETTER)
    analysis = processor.analyze(scan_image)
    assert analysis.status == "review"
    assert "No workers" in analysis.reason


def test_extraction_failure_does_not_fail_the_letter(db, decision_engine, scan_image):
    seed(db)
    processor, _, _, extractor = make_processor(db, decision_engine, CLEAN_LETTER)
    extractor.fail = True
    analysis = processor.analyze(scan_image)
    assert analysis.status == "filed"
    assert analysis.letter_date == "2026-03-14"  # found in the text instead
