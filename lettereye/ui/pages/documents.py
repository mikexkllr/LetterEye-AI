"""History of all processed letters with search and status filter."""

from __future__ import annotations

from datetime import datetime

from nicegui import ui

from .. import theme
from ..context import ctx
from ..document_detail import show_document
from ..widgets import on_data_change, page_header

COLUMNS = [
    {"name": "created", "label": "Processed", "field": "created", "sortable": True, "align": "left"},
    {"name": "name", "label": "File", "field": "name", "sortable": True, "align": "left",
     "classes": "ellipsis", "style": "max-width: 200px"},
    {"name": "status", "label": "Status", "field": "status", "sortable": True, "align": "left"},
    {"name": "target", "label": "Worker / Recipient", "field": "target", "sortable": True, "align": "left",
     "classes": "ellipsis", "style": "max-width: 240px"},
    {"name": "type", "label": "Type", "field": "type", "sortable": True, "align": "left"},
    {"name": "sender", "label": "Sender", "field": "sender", "sortable": True, "align": "left",
     "classes": "ellipsis", "style": "max-width: 180px"},
    {"name": "date", "label": "Letter date", "field": "date", "sortable": True, "align": "left"},
    {"name": "confidence", "label": "Confidence", "field": "confidence", "sortable": True, "align": "left"},
]

FILTERS = {"all": "All", "filed": "Filed", "review": "Review", "failed": "Failed", "ignored": "Dismissed"}
COLORS = {"positive": "green-6", "warning": "amber-7", "negative": "red-6"}


def page() -> None:
    c = ctx()
    with ui.column().classes("le-page"):
        page_header("Documents", "Every letter LetterEye has handled. Click a row for details, AI decisions and text.")
        with ui.row().classes("w-full items-center gap-3"):
            search = ui.input(placeholder="Search file, sender, recipient, worker, subject…").props(
                "outlined dense clearable").classes("flex-grow").style("max-width: 520px")
            with search.add_slot("prepend"):
                ui.icon("search")
            status = ui.toggle(FILTERS, value="all").props("unelevated no-caps toggle-color=primary")
        table = ui.table(columns=COLUMNS, rows=[], row_key="id", pagination={"rowsPerPage": 25, "sortBy": "created",
                                                                               "descending": True}) \
            .classes("w-full").props("flat")
        table.add_slot("body-cell-status", """
            <q-td :props="props">
              <q-chip dense square :color="props.row.status_color" text-color="white" :icon="props.row.status_icon"
                      class="text-xs text-weight-bold">{{ props.row.status_label }}</q-chip>
            </q-td>""")
        table.add_slot("body-cell-confidence", """
            <q-td :props="props">
              <div v-if="props.row.confidence !== null" class="row items-center no-wrap" style="gap:8px">
                <q-linear-progress rounded size="7px" :value="props.row.confidence" color="primary" track-color="grey-3"
                                   style="width:70px" />
                <span class="text-caption text-weight-medium">{{ Math.round(props.row.confidence * 100) }}%</span>
              </div>
              <span v-else class="text-grey-6">–</span>
            </q-td>""")
        table.on("rowClick", lambda e: show_document(e.args[1]["id"]))

    def load() -> None:
        selected = None if status.value == "all" else status.value
        docs = c.db.list_documents(status=selected, search=search.value or "", limit=2000)
        rows = []
        for d in docs:
            label, color, icon = theme.STATUS.get(d.status, (d.status, "grey", "help"))
            try:
                created = datetime.fromisoformat(d.created_at).astimezone().strftime("%d.%m.%y %H:%M")
            except ValueError:
                created = d.created_at
            rows.append({
                "id": d.id, "created": created, "name": d.original_name, "status": d.status, "status_label": label,
                "status_color": COLORS.get(color, color), "status_icon": icon,
                "target": " / ".join(x for x in (d.worker_name, d.recipient_label or d.recipient_name) if x) or "–",
                "type": d.doc_type or "–",
                "sender": d.sender or "–", "date": d.letter_date or "–", "confidence": d.confidence,
            })
        table.rows = rows
        table.update()

    search.on("update:model-value", lambda: load(), throttle=0.3)
    status.on_value_change(lambda: load())
    load()
    on_data_change(load, interval=2.0)
