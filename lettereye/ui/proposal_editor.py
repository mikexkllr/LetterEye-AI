"""The AI's proposal for one letter, editable before it is accepted (human in the loop)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from nicegui import run, ui

from ..ai.extraction import LetterFacts
from ..db import Document
from ..pipeline.processor import relative_target
from .context import ctx
from .widgets import probability_bars, worker_avatar

FIELD_LABELS = {"worker": "worker", "recipient": "recipient", "doc_type": "type", "letter_date": "date",
                "sender": "sender", "subject": "subject"}


class ProposalEditor:
    """Renders the proposal and the Accept / Reject actions. `on_done(message, doc_id)` after a decision."""

    def __init__(self, doc: Document, on_done: Callable[[str, int], None], *, compact: bool = False):
        self.c = ctx()
        self.doc = doc
        self.on_done = on_done
        self.settings = self.c.store.get()
        self.proposal = doc.proposal or {}
        self.recipients = self.c.db.list_recipients()
        self.workers = self.c.db.list_workers()
        self.recipient_select: ui.select | None = None
        self.accept_button: ui.button | None = None
        self.busy = False
        self._build(compact)

    # ------------------------------------------------------------------ layout
    def _build(self, compact: bool) -> None:
        doc, proposal = self.doc, self.proposal
        threshold = self.settings.confidence_threshold
        confidence = doc.confidence or 0.0
        sure = doc.status == "pending" or (doc.status == "filed" and confidence >= threshold)

        with ui.column().classes("w-full gap-4"):
            # ------------------------------------------------ verdict header
            with ui.row().classes("w-full items-center gap-4 no-wrap"):
                color = ("positive" if sure else "warning") if confidence >= 0.5 else "negative"
                with ui.circular_progress(value=confidence, max=1, show_value=False, size="64px", color=color) \
                        .props("thickness=0.18 track-color=grey-3").classes("shrink-0"):
                    ui.label(f"{confidence:.0%}").classes("text-sm font-bold")
                with ui.column().classes("gap-0").style("min-width: 0"):
                    if sure:
                        ui.label("AI proposal").classes("text-xs uppercase tracking-wider le-muted font-semibold")
                        ui.label(f"{confidence:.0%} sure this is right").classes("text-lg font-bold leading-tight")
                    else:
                        ui.label("AI is unsure").classes("text-xs uppercase tracking-wider text-amber-600 font-semibold")
                        ui.label(doc.review_reason or "Please choose the recipient.").classes(
                            "text-[15px] font-semibold leading-snug")

            # ------------------------------------------------ who
            with ui.column().classes("w-full gap-2 p-4 rounded-2xl").style("background: var(--le-soft)"):
                ui.label("File for").classes("text-xs uppercase tracking-wider le-muted font-semibold")
                self.worker_row = ui.row().classes("items-center gap-3 no-wrap")
                options = {r.id: f"{r.name}  ·  {r.worker_name}" for r in self.recipients}
                start_new = not doc.recipient_id and bool(proposal.get("new_recipient") or doc.recipient_name)
                self.mode = ui.toggle({"existing": "Known recipient", "new": "New recipient"},
                                      value="new" if start_new else "existing") \
                    .props("dense no-caps unelevated toggle-color=primary size=sm")
                self.recipient_select = ui.select(options, value=doc.recipient_id, with_input=True,
                                                  label="Recipient", on_change=lambda: self._refresh()) \
                    .props("outlined dense options-dense behavior=menu").classes("w-full")
                with ui.row().classes("w-full gap-2 no-wrap") as self.new_row:
                    self.new_name = ui.input("Name", value=proposal.get("recipient") or doc.recipient_name,
                                             on_change=lambda: self._refresh()).props("outlined dense").classes("flex-grow")
                    self.new_worker = ui.select({w.id: w.name for w in self.workers},
                                                value=proposal.get("worker_id") or doc.worker_id, label="Worker",
                                                on_change=lambda: self._refresh()) \
                        .props("outlined dense").classes("w-40")
                self.recipient_select.bind_visibility_from(self.mode, "value", value="existing")
                self.new_row.bind_visibility_from(self.mode, "value", value="new")
                self.mode.on_value_change(lambda: self._refresh())

            # ------------------------------------------------ details
            types = [t.name for t in self.settings.document_types]
            if doc.doc_type and doc.doc_type not in types:
                types.append(doc.doc_type)
            with ui.grid(columns=2).classes("w-full gap-3"):
                self.doc_type = ui.select(types, value=doc.doc_type or None, label="Document type",
                                          on_change=lambda: self._refresh()).props("outlined dense")
                with ui.input("Letter date", value=doc.letter_date, on_change=lambda: self._refresh()) \
                        .props("outlined dense") as self.letter_date:
                    with ui.menu().props("no-parent-event") as menu:
                        ui.date(mask="YYYY-MM-DD").bind_value(self.letter_date)
                    with self.letter_date.add_slot("append"):
                        ui.icon("event").classes("cursor-pointer").on("click", menu.open)
                self.sender = ui.input("Sender", value=doc.sender, on_change=lambda: self._refresh()) \
                    .props("outlined dense").classes("col-span-2")
                self.subject = ui.input("Subject", value=doc.subject, on_change=lambda: self._refresh()) \
                    .props("outlined dense").classes("col-span-2")

            # ------------------------------------------------ result preview
            with ui.column().classes("w-full gap-1"):
                ui.label("Will be saved as").classes("text-xs uppercase tracking-wider le-muted font-semibold")
                self.target_label = ui.label("").classes("le-kbd break-all text-[12px]")
                self.changed_row = ui.row().classes("gap-1 items-center")

            # ------------------------------------------------ actions
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                self.accept_button = ui.button("Accept & file", icon="check", on_click=self.accept) \
                    .props("unelevated color=positive size=lg no-caps").classes("flex-grow").style("border-radius: 14px")
                with ui.button(icon="close").props("outline color=negative size=lg").style("border-radius: 14px") \
                        .tooltip("Reject (X)"):
                    with ui.menu().classes("rounded-xl"):
                        ui.menu_item("Wrong recipient – I'll choose", on_click=self.focus_recipient)
                        if doc.status in ("pending", "review"):
                            ui.menu_item("Run the AI again", on_click=self.rerun)
                        ui.separator()
                        ui.menu_item("Not a letter – set aside", on_click=lambda: self.reject("Not a letter"))
                        ui.menu_item("Advertising – set aside", on_click=lambda: self.reject("Advertising"))
                        ui.menu_item("Duplicate – set aside", on_click=lambda: self.reject("Duplicate"))

            if not compact:
                decisions = [d for d in doc.trace.get("decisions", []) if d.get("options")]
                if decisions:
                    with ui.expansion("Why did the AI decide this?", icon="psychology").classes("w-full le-card") \
                            .props("dense header-class='text-sm font-semibold'"):
                        for decision in decisions:
                            probability_bars(decision)
                            ui.separator().classes("my-2")
        self._refresh()

    # ------------------------------------------------------------------ state
    def _current(self) -> dict:
        """The decision as currently shown in the form."""
        if self.mode.value == "existing":
            recipient = next((r for r in self.recipients if r.id == self.recipient_select.value), None)
            worker = next((w for w in self.workers if recipient and w.id == recipient.worker_id), None)
            recipient_name, recipient_folder = (recipient.name, recipient.folder) if recipient else ("", "")
        else:
            worker = next((w for w in self.workers if w.id == self.new_worker.value), None)
            recipient_name = recipient_folder = " ".join((self.new_name.value or "").split())
        return {
            "worker": worker, "recipient_name": recipient_name, "recipient_folder": recipient_folder,
            "doc_type": self.doc_type.value or "", "letter_date": (self.letter_date.value or "").strip(),
            "sender": (self.sender.value or "").strip(), "subject": (self.subject.value or "").strip(),
        }

    def _changes(self, current: dict) -> list[str]:
        p = self.proposal
        values = {"worker": current["worker"].name if current["worker"] else "", "recipient": current["recipient_name"],
                  "doc_type": current["doc_type"], "letter_date": current["letter_date"],
                  "sender": current["sender"], "subject": current["subject"]}

        def norm(v) -> str:
            return " ".join(str(v or "").split()).casefold()

        return [k for k, v in values.items() if norm(v) != norm(p.get(k, ""))]

    def _refresh(self) -> None:
        current = self._current()
        self.worker_row.clear()
        with self.worker_row:
            worker_avatar(current["worker"], "34px")
            ui.label(current["worker"].name if current["worker"] else "No worker yet").classes("font-semibold")
        if current["worker"] and current["recipient_name"]:
            facts = LetterFacts(sender_organization=current["sender"], subject=current["subject"])
            rel = relative_target(self.settings, current["worker"], current["recipient_name"],
                                  current["recipient_folder"], facts, current["letter_date"], current["doc_type"],
                                  Path(self.doc.original_name))
            self.target_label.text = str(rel)
        else:
            self.target_label.text = "Choose a recipient"
        changes = self._changes(current) if self.proposal else []
        self.changed_row.clear()
        with self.changed_row:
            if changes and self.proposal.get("recipient"):
                ui.icon("edit", size="14px").classes("text-amber-600")
                for change in changes:
                    ui.chip(f"changed {FIELD_LABELS[change]}").props("dense outline color=amber-8").classes("text-[11px]")
        if self.accept_button is not None:
            corrected = bool(changes) and bool(self.proposal.get("recipient"))
            label = "Save correction & file" if corrected else ("Confirm" if self.doc.status == "filed" and not changes
                                                                 else "Accept & file")
            self.accept_button.text = label
            self.accept_button.props(f"color={'primary' if corrected else 'positive'}")

    # ------------------------------------------------------------------ actions
    def focus_recipient(self) -> None:
        self.mode.value = "existing"
        if self.recipient_select is not None:
            self.recipient_select.run_method("focus")
            self.recipient_select.run_method("showPopup")

    async def accept(self) -> None:
        if self.busy:
            return
        current = self._current()
        if not current["worker"] or not current["recipient_name"]:
            ui.notify("Choose who this letter is for first", type="warning")
            self.focus_recipient()
            return
        kwargs: dict = dict(doc_type=current["doc_type"], letter_date=current["letter_date"], sender=current["sender"],
                            subject=current["subject"])
        if self.mode.value == "existing":
            kwargs["recipient_id"] = self.recipient_select.value
        else:
            kwargs.update(worker_id=current["worker"].id, new_recipient_name=current["recipient_name"])
        self.busy = True
        try:
            await run.io_bound(lambda: self.c.engine.approve(self.doc.id, **kwargs))
        except Exception as exc:
            ui.notify(str(exc), type="negative")
            return
        finally:
            self.busy = False
        changes = self._changes(current)
        verb = "Corrected and filed" if changes and self.proposal.get("recipient") else "Filed"
        self.on_done(f"{verb}: {current['worker'].name} / {current['recipient_name']}", self.doc.id)

    async def reject(self, reason: str) -> None:
        try:
            await run.io_bound(self.c.engine.reject, self.doc.id, reason)
        except Exception as exc:
            ui.notify(str(exc), type="negative")
            return
        self.on_done(f"Set aside ({reason.lower()})", self.doc.id)

    def rerun(self) -> None:
        try:
            self.c.engine.reprocess(self.doc.id)
        except Exception as exc:
            ui.notify(str(exc), type="warning")
            return
        ui.notify("The AI looks at it again", type="info")
