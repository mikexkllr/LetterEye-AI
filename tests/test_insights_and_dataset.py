import json
import zipfile

from lettereye import dataset, insights


def seed(db, outcomes):
    """outcomes: list of (confidence, action, changed)."""
    worker = db.create_worker("John Doe")
    for name in ("Bob Smith", "Alice Johnson"):
        db.create_recipient(worker.id, name)
    for index, (confidence, action, changed) in enumerate(outcomes):
        doc = db.create_document(f"/in/scan{index}.pdf")
        trace = {"decisions": [
            {"key": "recipient", "round": "decisions", "question": "Who is the addressee (recipient) of this letter?",
             "choice": "Bob Smith", "confidence": confidence,
             "options": [{"option": "Alice Johnson", "p": 0.1}, {"option": "Bob Smith", "p": confidence},
                         {"option": "Someone else (not in this list)", "p": 0.05}]},
            {"key": "doc_type", "round": "decisions", "question": "Which category best describes this document?",
             "choice": "Rechnung: Rechnung oder Abrechnung", "confidence": 0.9,
             "options": [{"option": "Rechnung: Rechnung oder Abrechnung", "p": 0.9}, {"option": "Mahnung", "p": 0.1}]},
        ]}
        db.update_document(doc.id, status="filed", confidence=confidence, ocr_text="Herrn Bob Smith ...", trace=trace,
                           duration_ms=3000, proposal={"decision_model": "qwen3.5:4b"})
        ai = {"worker": "John Doe", "recipient": "Bob Smith", "doc_type": "Rechnung", "letter_date": "2026-03-14",
              "sender": "Stadtwerke", "subject": "Strom"}
        final = dict(ai)
        if "recipient" in changed:
            final["recipient"] = "Alice Johnson"
        if "doc_type" in changed:
            final["doc_type"] = "Mahnung"
        db.add_feedback(doc.id, action, ai=ai, final=final, changed=changed, ai_confidence=confidence,
                        decision_model="qwen3.5:4b", seconds_to_decide=5.0, processing_ms=3000)


def test_metrics_count_only_verified_letters(db):
    seed(db, [(0.97, "accepted", []), (0.92, "accepted", []), (0.85, "corrected", ["recipient"]),
              (0.6, "corrected", ["doc_type"]), (0.99, "auto_filed", []), (0.7, "rejected", [])])
    data = insights.compute(db, days=30)
    assert data.processed == 6 and data.verified == 4
    assert data.outcomes["auto"] == 1 and data.outcomes["rejected"] == 1
    assert data.routing_accuracy == 0.75  # only the recipient correction counts as a routing mistake
    assert data.unchanged_rate == 0.5
    assert data.field_accuracy["doc_type"] == (3, 4)
    assert data.median_decide_s == 5.0
    at_90 = data.at_threshold(0.9)
    assert at_90["n_auto"] == 2 and at_90["errors"] == 0  # the two ≥ 0.9 letters were right
    at_80 = data.at_threshold(0.8)
    assert at_80["n_auto"] == 3 and at_80["errors"] == 1
    assert data.corrections[0]["ai"] in ("Bob Smith", "Rechnung")
    assert sum(day["accepted"] for day in data.daily) == 2
    assert data.models[0]["model"] == "qwen3.5:4b" and data.models[0]["verified"] == 4


def test_export_writes_labelled_examples(db, tmp_path):
    seed(db, [(0.95, "accepted", []), (0.7, "corrected", ["recipient", "doc_type"]), (0.99, "auto_filed", [])])
    path, examples, letters = dataset.export(db, tmp_path)
    assert letters == 2  # the unchecked auto-filed letter is not training data
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        assert {"decisions.jsonl", "chat_sft.jsonl", "extraction.jsonl", "documents.csv", "feedback.csv",
                "README.md"} <= names
        decisions = [json.loads(line) for line in archive.read("decisions.jsonl").decode().splitlines()]
        chats = [json.loads(line) for line in archive.read("chat_sft.jsonl").decode().splitlines()]
    assert examples == len(decisions) == len(chats) == 4
    corrected = [d for d in decisions if d["task"] == "recipient" and not d["ai_correct"]][0]
    assert corrected["label"] == "Alice Johnson" and corrected["ai_choice"] == "Bob Smith"
    doc_type = [d for d in decisions if d["task"] == "doc_type" and not d["ai_correct"]][0]
    assert doc_type["label"] == "Mahnung"
    chat = chats[decisions.index(corrected)]
    assert chat["messages"][-1] == {"role": "assistant", "content": "A"}  # Alice Johnson is option A
    assert "OPTIONS:" in chat["messages"][1]["content"]
