"""Metrics about how well the AI does, computed from the human feedback log.

Only letters a human has looked at count as *verified* (accepted, confirmed or corrected). Auto-filed letters
nobody checked are counted for volume, never for accuracy.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from .db import Database, Document, Feedback

FIELDS = ("worker", "recipient", "doc_type", "letter_date", "sender", "subject")
FIELD_TITLES = {"worker": "Worker", "recipient": "Recipient", "doc_type": "Document type", "letter_date": "Letter date",
                "sender": "Sender", "subject": "Subject"}
CALIBRATION_BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 1.0001)]


@dataclass
class Outcome:
    """The current verdict on one letter: the latest feedback that was not undone."""
    doc_id: int
    kind: str  # auto | accepted | corrected | rejected
    feedback: Feedback

    @property
    def verified(self) -> bool:
        return self.kind in ("accepted", "corrected")

    @property
    def routing_correct(self) -> bool:
        """The AI picked the right recipient (and therefore the right worker)."""
        return self.verified and not {"recipient", "worker"} & set(self.feedback.changed)


@dataclass
class Insights:
    days: int | None
    processed: int = 0
    waiting: int = 0
    outcomes: Counter = field(default_factory=Counter)
    verified: int = 0
    routing_accuracy: float | None = None
    unchanged_rate: float | None = None
    field_accuracy: dict[str, tuple[int, int]] = field(default_factory=dict)
    median_processing_s: float | None = None
    median_decide_s: float | None = None
    escalation_rate: float | None = None
    daily: list[dict] = field(default_factory=list)
    calibration: list[dict] = field(default_factory=list)
    tradeoff: list[dict] = field(default_factory=list)
    corrections: list[dict] = field(default_factory=list)
    models: list[dict] = field(default_factory=list)

    def at_threshold(self, threshold: float) -> dict | None:
        rows = [r for r in self.tradeoff if abs(r["threshold"] - threshold) < 0.005]
        return rows[0] if rows else None


def latest_outcomes(feedback: list[Feedback]) -> dict[int, Outcome]:
    outcomes: dict[int, Outcome] = {}
    for fb in feedback:  # ordered by id, so later verdicts replace earlier ones
        if fb.doc_id is None or fb.undone:
            continue
        kind = {"auto_filed": "auto", "accepted": "accepted", "confirmed": "accepted", "corrected": "corrected",
                "rejected": "rejected"}[fb.action]
        outcomes[fb.doc_id] = Outcome(fb.doc_id, kind, fb)
    return outcomes


def _local_day(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).astimezone().date().isoformat()
    except ValueError:
        return iso[:10]


def compute(db: Database, days: int | None = 30) -> Insights:
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds") if days else None
    result = Insights(days=days)
    docs: list[Document] = [d for d in db.list_documents(limit=1_000_000)
                            if since is None or d.created_at >= since]
    finished = [d for d in docs if d.status not in ("queued", "processing")]
    result.processed = len(finished)
    result.waiting = sum(1 for d in db.list_documents(status=["pending", "review"], limit=100_000))
    durations = [d.duration_ms / 1000 for d in finished if d.duration_ms]
    result.median_processing_s = round(statistics.median(durations), 1) if durations else None
    with_trace = [d for d in finished if d.trace]
    if with_trace:
        result.escalation_rate = sum(1 for d in with_trace if d.trace.get("escalated")) / len(with_trace)

    feedback = db.list_feedback(since=since)
    outcomes = latest_outcomes(feedback)
    result.outcomes = Counter(o.kind for o in outcomes.values())
    verified = [o for o in outcomes.values() if o.verified]
    result.verified = len(verified)
    if verified:
        result.routing_accuracy = sum(o.routing_correct for o in verified) / len(verified)
        result.unchanged_rate = sum(1 for o in verified if not o.feedback.changed) / len(verified)
        for name in FIELDS:
            correct = sum(1 for o in verified if name not in o.feedback.changed)
            result.field_accuracy[name] = (correct, len(verified))
    decide = [f.seconds_to_decide for f in feedback
              if f.action in ("accepted", "corrected", "rejected") and f.seconds_to_decide is not None
              and f.seconds_to_decide < 7 * 86400]
    result.median_decide_s = round(statistics.median(decide), 1) if decide else None

    # ------------------------------------------------------------- per day
    span = days or 30
    today = datetime.now().astimezone().date()
    per_day: dict[str, Counter] = defaultdict(Counter)
    for outcome in outcomes.values():
        per_day[_local_day(outcome.feedback.created_at)][outcome.kind] += 1
    result.daily = [{"day": (today - timedelta(days=offset)).isoformat(),
                     **{k: per_day[(today - timedelta(days=offset)).isoformat()][k]
                        for k in ("auto", "accepted", "corrected", "rejected")}}
                    for offset in range(span - 1, -1, -1)]

    # ------------------------------------------------------------- calibration
    scored = [o for o in verified if o.feedback.ai_confidence is not None]
    for low, high in CALIBRATION_BINS:
        inside = [o for o in scored if low <= o.feedback.ai_confidence < high]
        result.calibration.append({
            "label": f"{low:.0%}–{min(high, 1):.0%}", "low": low, "high": min(high, 1.0), "n": len(inside),
            "accuracy": (sum(o.routing_correct for o in inside) / len(inside)) if inside else None,
            "expected": (low + min(high, 1.0)) / 2,
        })

    # ------------------------------------------------------------- threshold trade-off
    if scored:
        for step in range(50, 100):
            threshold = step / 100
            auto = [o for o in scored if o.feedback.ai_confidence >= threshold]
            wrong = sum(1 for o in auto if not o.routing_correct)
            result.tradeoff.append({
                "threshold": threshold, "automated": len(auto) / len(scored), "n_auto": len(auto),
                "errors": wrong, "error_rate": (wrong / len(auto)) if auto else 0.0, "n": len(scored),
            })

    # ------------------------------------------------------------- common corrections
    pairs: Counter = Counter()
    for outcome in outcomes.values():
        if outcome.kind != "corrected":
            continue
        changed = outcome.feedback.changed
        for name in changed:
            if name == "worker" and "recipient" in changed:
                continue  # the worker follows the recipient; the recipient row says it all
            if name in ("recipient", "worker", "doc_type"):
                pairs[(name, outcome.feedback.ai.get(name) or "–", outcome.feedback.final.get(name) or "–")] += 1
    result.corrections = [{"field": FIELD_TITLES[f], "ai": a, "human": h, "count": n}
                          for (f, a, h), n in pairs.most_common(12)]

    # ------------------------------------------------------------- per model
    by_model: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes.values():
        by_model[outcome.feedback.decision_model or "unknown"].append(outcome)
    for model, items in sorted(by_model.items()):
        checked = [o for o in items if o.verified]
        times = [o.feedback.processing_ms / 1000 for o in items if o.feedback.processing_ms]
        result.models.append({
            "model": model, "letters": len(items), "verified": len(checked),
            "accuracy": (sum(o.routing_correct for o in checked) / len(checked)) if checked else None,
            "median_s": round(statistics.median(times), 1) if times else None,
            "escalated": sum(1 for o in items if o.feedback.escalated) / len(items),
        })
    return result
