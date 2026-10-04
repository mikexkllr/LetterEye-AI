"""Insights: how good is the AI, where does it go wrong, and what would a different threshold do?

Charts follow one palette (validated for colour-vision deficiency in light and dark mode), thin marks,
hover tooltips, and every chart has a table view.
"""

from __future__ import annotations

from nicegui import run, ui

from ... import dataset
from ...insights import FIELD_TITLES, Insights, compute
from ..context import ctx
from ..widgets import card, open_path, page_header, pick_folder

PERIODS = {7: "7 days", 30: "30 days", 90: "90 days", 0: "All time"}

PALETTE = {
    False: {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "ink": "#0f172a", "ink2": "#52514e", "muted": "#898781",
            "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff", "ideal": "#898781"},
    True: {"s1": "#3987e5", "s2": "#d95926", "s3": "#199e70", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
           "grid": "#2c2c2a", "axis": "#383835", "surface": "#151924", "ideal": "#898781"},
}


def _pct(value: float | None, digits: int = 0) -> str:
    return "–" if value is None else f"{value * 100:.{digits}f}%"


def _secs(value: float | None) -> str:
    if value is None:
        return "–"
    return f"{value:.0f} s" if value < 120 else f"{value / 60:.0f} min"


async def page() -> None:
    c = ctx()
    state = {"days": 30, "dark": False}
    try:
        await ui.context.client.connected(timeout=5)
        state["dark"] = bool(await ui.run_javascript("document.body.classList.contains('body--dark')", timeout=3))
    except Exception:
        state["dark"] = c.store.get().theme == "dark"

    with ui.column().classes("le-page"):
        page_header("Insights", "How well the AI sorts your letters – measured on the letters you checked. "
                                "Use it to pick the threshold, spot recurring mistakes, and decide when to fine-tune.")
        with ui.row().classes("w-full items-center gap-3"):
            ui.toggle(PERIODS, value=30, on_change=lambda e: (state.update(days=e.value), render())) \
                .props("unelevated no-caps toggle-color=primary")
        content = ui.column().classes("w-full gap-5")

    def render() -> None:
        data = compute(c.db, state["days"] or None)
        colors = PALETTE[state["dark"]]
        content.clear()
        with content:
            _kpis(data)
            if not data.verified:
                with card("p-6"):
                    with ui.row().classes("items-center gap-4 no-wrap"):
                        ui.icon("insights", size="40px").classes("le-gradient-text")
                        with ui.column().classes("gap-1"):
                            ui.label("Accuracy appears once you check letters").classes("font-semibold")
                            ui.label("Every letter you accept or correct in Approvals (or confirm under Documents) "
                                     "becomes a verified data point. Approval mode collects them fastest.") \
                                .classes("le-muted text-sm")
            with ui.grid(columns=2).classes("w-full gap-5"):
                _daily_chart(data, colors)
                _field_chart(data, colors)
                _calibration_chart(data, colors)
                _tradeoff(data, colors, render)
            with ui.grid(columns=2).classes("w-full gap-5"):
                _corrections_table(data)
                _models_table(data)
            _export_card()

    render()


# ------------------------------------------------------------------------- building blocks
def _kpis(data: Insights) -> None:
    tiles = [
        ("Letters processed", f"{data.processed:,}", f"{data.waiting} waiting for you" if data.waiting else "none waiting"),
        ("Routing accuracy", _pct(data.routing_accuracy), f"of {data.verified} checked letters"),
        ("Unchanged", _pct(data.unchanged_rate), "accepted without edits"),
        ("Time to decide", _secs(data.median_decide_s), "median, per letter"),
        ("Processing time", _secs(data.median_processing_s), "median, OCR + AI"),
        ("OCR LLM needed", _pct(data.escalation_rate), "re-read because unsure"),
    ]
    with ui.grid(columns=6).classes("w-full gap-4"):
        for label, value, note in tiles:
            with card("p-4 gap-1"):
                ui.label(label).classes("le-muted text-sm font-medium")
                ui.label(value).classes("le-stat-value")
                ui.label(note).classes("le-muted text-xs")


def _chart_card(title: str, subtitle: str, options: dict, columns: list[dict], rows: list[dict],
                height: int = 300) -> None:
    """A chart with a table-view twin (the accessible equivalent)."""
    with card("p-5 gap-2"):
        with ui.row().classes("w-full items-start justify-between no-wrap"):
            with ui.column().classes("gap-0"):
                ui.label(title).classes("le-section-title")
                ui.label(subtitle).classes("le-muted text-xs")
            toggle = ui.toggle({"chart": "Chart", "table": "Table"}, value="chart") \
                .props("dense unelevated no-caps size=sm toggle-color=primary")
        chart = ui.echart(options).classes("w-full").style(f"height: {height}px")
        table = ui.table(columns=columns, rows=rows, row_key=columns[0]["name"]).props("flat dense") \
            .classes("w-full").style(f"max-height: {height}px")
        chart.bind_visibility_from(toggle, "value", value="chart")
        table.bind_visibility_from(toggle, "value", value="table")


def _axis(colors: dict, **extra) -> dict:
    return {"axisLine": {"lineStyle": {"color": colors["axis"]}}, "axisTick": {"show": False},
            "axisLabel": {"color": colors["muted"], "fontSize": 11},
            "splitLine": {"lineStyle": {"color": colors["grid"], "width": 1, "type": "solid"}}, **extra}


def _tooltip(colors: dict, **extra) -> dict:
    return {"backgroundColor": colors["surface"], "borderColor": colors["grid"], "textStyle": {"color": colors["ink"]},
            **extra}


def _daily_chart(data: Insights, colors: dict) -> None:
    days = [d["day"][5:] for d in data.daily]
    totals = {k: sum(d[k] for d in data.daily) for k in ("accepted", "corrected", "auto")}
    series_spec = [("accepted", f"Accepted as proposed · {totals['accepted']}", colors["s1"]),
                   ("corrected", f"Corrected · {totals['corrected']}", colors["s2"]),
                   ("auto", f"Filed automatically, unchecked · {totals['auto']}", colors["s3"])]
    series = []
    for index, (key, name, color) in enumerate(series_spec):
        series.append({
            "name": name, "type": "bar", "stack": "outcome", "barMaxWidth": 24,
            "data": [d[key] for d in data.daily],
            "itemStyle": {"color": color, "borderColor": colors["surface"], "borderWidth": 1,
                          "borderRadius": [4, 4, 0, 0] if index == len(series_spec) - 1 else 0},
            "emphasis": {"focus": "series"},
        })
    options = {
        "animationDuration": 300,
        "grid": {"left": 36, "right": 12, "top": 44, "bottom": 28},
        "legend": {"top": 0, "left": 0, "icon": "rect", "itemWidth": 12, "itemHeight": 12,
                   "textStyle": {"color": colors["ink2"], "fontSize": 12}},
        "tooltip": _tooltip(colors, trigger="axis", axisPointer={"type": "shadow"}),
        "xAxis": _axis(colors, type="category", data=days, splitLine={"show": False}),
        "yAxis": _axis(colors, type="value", minInterval=1),
        "series": series,
    }
    rows = [{"day": d["day"], "accepted": d["accepted"], "corrected": d["corrected"], "auto": d["auto"]}
            for d in data.daily if d["accepted"] or d["corrected"] or d["auto"]]
    columns = [{"name": "day", "label": "Day", "field": "day", "align": "left"},
               {"name": "accepted", "label": "Accepted", "field": "accepted"},
               {"name": "corrected", "label": "Corrected", "field": "corrected"},
               {"name": "auto", "label": "Auto-filed", "field": "auto"}]
    _chart_card("Letters per day", "By what happened to the AI's proposal", options, columns, rows)


def _field_chart(data: Insights, colors: dict) -> None:
    names = list(FIELD_TITLES)
    values = [round(100 * data.field_accuracy[n][0] / data.field_accuracy[n][1], 1) if n in data.field_accuracy
              and data.field_accuracy[n][1] else None for n in names]
    options = {
        "animationDuration": 300,
        "grid": {"left": 104, "right": 48, "top": 8, "bottom": 24},
        "tooltip": _tooltip(colors, trigger="item", formatter="{b}: {c}% right"),
        "xAxis": _axis(colors, type="value", min=0, max=100, axisLabel={"color": colors["muted"], "formatter": "{value}%"}),
        "yAxis": _axis(colors, type="category", data=[FIELD_TITLES[n] for n in names][::-1], splitLine={"show": False},
                       axisLabel={"color": colors["ink2"], "fontSize": 12}),
        "series": [{"type": "bar", "barMaxWidth": 20, "data": values[::-1],
                    "itemStyle": {"color": colors["s1"], "borderRadius": [0, 4, 4, 0]},
                    "label": {"show": True, "position": "right", "color": colors["ink2"], "formatter": "{c}%"}}],
    }
    rows = [{"field": FIELD_TITLES[n], "right": data.field_accuracy.get(n, (0, 0))[0],
             "checked": data.field_accuracy.get(n, (0, 0))[1], "accuracy": f"{v}%" if v is not None else "–"}
            for n, v in zip(names, values, strict=True)]
    columns = [{"name": "field", "label": "Field", "field": "field", "align": "left"},
               {"name": "right", "label": "Right", "field": "right"},
               {"name": "checked", "label": "Checked", "field": "checked"},
               {"name": "accuracy", "label": "Accuracy", "field": "accuracy"}]
    _chart_card("Accuracy per field", "Share of checked letters where the AI's value was kept", options, columns, rows)


def _calibration_chart(data: Insights, colors: dict) -> None:
    labels = [f"{b['label']}\nn={b['n']}" for b in data.calibration]
    actual = [round(b["accuracy"] * 100, 1) if b["accuracy"] is not None else None for b in data.calibration]
    ideal = [round(b["expected"] * 100, 1) for b in data.calibration]
    options = {
        "animationDuration": 300,
        "grid": {"left": 40, "right": 12, "top": 40, "bottom": 44},
        "legend": {"top": 0, "left": 0, "itemWidth": 14, "itemHeight": 10,
                   "textStyle": {"color": colors["ink2"], "fontSize": 12}},
        "tooltip": _tooltip(colors, trigger="axis", axisPointer={"type": "shadow"},
                            **{":valueFormatter": "(v) => v == null ? '–' : v + '%'"}),
        "xAxis": _axis(colors, type="category", data=labels, splitLine={"show": False},
                       axisLabel={"color": colors["muted"], "fontSize": 11, "lineHeight": 14}),
        "yAxis": _axis(colors, type="value", min=0, max=100, axisLabel={"color": colors["muted"], "formatter": "{value}%"}),
        "series": [
            {"name": "Actually right", "type": "bar", "barMaxWidth": 24, "data": actual,
             "itemStyle": {"color": colors["s1"], "borderRadius": [4, 4, 0, 0]},
             "label": {"show": True, "position": "top", "color": colors["ink2"], "formatter": "{c}%", "fontSize": 11,
                       "textBorderColor": colors["surface"], "textBorderWidth": 3}},
            {"name": "Ideal", "type": "line", "data": ideal, "symbol": "none", "smooth": False,
             "lineStyle": {"color": colors["ideal"], "width": 2}},
        ],
    }
    rows = [{"bin": b["label"], "n": b["n"], "right": _pct(b["accuracy"]), "ideal": _pct(b["expected"])}
            for b in data.calibration]
    columns = [{"name": "bin", "label": "AI confidence", "field": "bin", "align": "left"},
               {"name": "n", "label": "Letters", "field": "n"},
               {"name": "right", "label": "Actually right", "field": "right"},
               {"name": "ideal", "label": "Ideal", "field": "ideal"}]
    _chart_card("Can you trust the percentage?", "Routing accuracy per confidence band – bars near the line mean the "
                "AI's confidence is honest", options, columns, rows)


def _tradeoff(data: Insights, colors: dict, rerender) -> None:
    c = ctx()
    current = c.store.get().confidence_threshold
    points = [[round(r["automated"] * 100, 1), round(r["error_rate"] * 100, 2), r["threshold"]] for r in data.tradeoff]
    now = data.at_threshold(current)
    options = {
        "animationDuration": 300,
        "grid": {"left": 48, "right": 16, "top": 16, "bottom": 40},
        "tooltip": _tooltip(colors, trigger="axis", axisPointer={"type": "line", "lineStyle": {"color": colors["axis"]}},
                            **{":formatter": "(p) => { const d = p[0].data; return 'Threshold ' + Math.round(d[2]*100) + "
                                      "'%<br/><b>' + d[0] + '%</b> filed automatically<br/><b>' + d[1] + "
                                      "'%</b> of those wrong'; }"}),
        "xAxis": _axis(colors, type="value", min=0, max=100, name="Filed automatically", nameLocation="middle",
                       nameGap=26, nameTextStyle={"color": colors["muted"]},
                       axisLabel={"color": colors["muted"], "formatter": "{value}%"}),
        "yAxis": _axis(colors, type="value", min=0, name="Wrong", nameTextStyle={"color": colors["muted"]},
                       axisLabel={"color": colors["muted"], "formatter": "{value}%"}),
        "series": [{
            "type": "line", "data": points, "symbol": "none", "lineStyle": {"color": colors["s1"], "width": 2},
            "markPoint": {"symbol": "circle", "symbolSize": 10,
                          "itemStyle": {"color": colors["s1"], "borderColor": colors["surface"], "borderWidth": 2},
                          "label": {"show": True, "position": "top", "color": colors["ink2"],
                                    "formatter": f"now {current:.0%}"},
                          "data": [{"coord": [round(now["automated"] * 100, 1), round(now["error_rate"] * 100, 2)]}]
                          if now else []},
        }],
    }
    rows = [{"threshold": f"{r['threshold']:.0%}", "automated": f"{r['automated']:.0%}", "n_auto": r["n_auto"],
             "errors": r["errors"], "error_rate": f"{r['error_rate']:.1%}"} for r in data.tradeoff[::5]]
    columns = [{"name": "threshold", "label": "Threshold", "field": "threshold", "align": "left"},
               {"name": "automated", "label": "Automatic", "field": "automated"},
               {"name": "n_auto", "label": "Letters", "field": "n_auto"},
               {"name": "errors", "label": "Wrong", "field": "errors"},
               {"name": "error_rate", "label": "Wrong %", "field": "error_rate"}]
    with ui.column().classes("w-full gap-0"):
        _chart_card("Choose your threshold", "Each point is a threshold: more automation vs. more mistakes slipping "
                    "through (measured on checked letters)", options, columns, rows, height=250)
        with card("p-4 gap-2 mt-3"):
            readout = ui.label("").classes("text-sm")
            slider = ui.slider(min=0.5, max=0.99, step=0.01, value=current).props("color=primary label")

            def show(value: float) -> None:
                row = data.at_threshold(round(value, 2))
                if row is None:
                    readout.text = "Check a few letters first – then you can simulate thresholds here."
                    return
                readout.text = (f"At {value:.0%}: {row['automated']:.0%} of letters would be filed automatically, "
                                f"{row['errors']} of {row['n_auto']} of those wrong ({row['error_rate']:.1%}).")

            def apply() -> None:
                c.store.update(confidence_threshold=round(slider.value, 2))
                ui.notify(f"Threshold set to {slider.value:.0%}", type="positive")
                rerender()

            slider.on_value_change(lambda e: show(e.value))
            show(current)
            with ui.row().classes("w-full justify-end"):
                ui.button("Use this threshold", icon="check", on_click=apply).props("unelevated color=primary no-caps dense")


def _corrections_table(data: Insights) -> None:
    with card("p-5 gap-2"):
        ui.label("Most common corrections").classes("le-section-title")
        ui.label("What the AI proposed and what you chose instead – add spellings or worker descriptions for the "
                 "top ones.").classes("le-muted text-xs")
        if not data.corrections:
            ui.label("No corrections yet.").classes("le-muted text-sm py-4")
            return
        ui.table(columns=[{"name": "field", "label": "Field", "field": "field", "align": "left"},
                          {"name": "ai", "label": "AI said", "field": "ai", "align": "left"},
                          {"name": "human", "label": "You chose", "field": "human", "align": "left"},
                          {"name": "count", "label": "Times", "field": "count"}],
                 rows=data.corrections, row_key="ai").props("flat dense").classes("w-full")


def _models_table(data: Insights) -> None:
    with card("p-5 gap-2"):
        ui.label("Models compared").classes("le-section-title")
        ui.label("Switch the decision model in Settings and compare here – a bigger model is worth it when accuracy "
                 "rises more than the time per letter.").classes("le-muted text-xs")
        if not data.models:
            ui.label("No data yet.").classes("le-muted text-sm py-4")
            return
        rows = [{"model": m["model"], "letters": m["letters"], "verified": m["verified"], "accuracy": _pct(m["accuracy"]),
                 "time": _secs(m["median_s"]), "escalated": _pct(m["escalated"])} for m in data.models]
        ui.table(columns=[{"name": "model", "label": "Decision model", "field": "model", "align": "left"},
                          {"name": "letters", "label": "Letters", "field": "letters"},
                          {"name": "verified", "label": "Checked", "field": "verified"},
                          {"name": "accuracy", "label": "Accuracy", "field": "accuracy"},
                          {"name": "time", "label": "Time / letter", "field": "time"},
                          {"name": "escalated", "label": "OCR LLM", "field": "escalated"}],
                 rows=rows, row_key="model").props("flat dense").classes("w-full")


def _export_card() -> None:
    c = ctx()
    with card("p-5"):
        with ui.row().classes("w-full items-center gap-4 no-wrap"):
            ui.icon("model_training", size="36px").classes("le-gradient-text")
            with ui.column().classes("gap-1 flex-grow"):
                ui.label("Training data").classes("le-section-title")
                ui.label("Export every letter you checked as labelled data: typed decisions (for evaluating or "
                         "fine-tuning the decision model), chat-format examples for LoRA fine-tuning, extraction "
                         "corrections and CSVs. Stays on this computer until you share it.").classes("le-muted text-sm")
            images = ui.checkbox("Include page images", value=False).classes("text-sm")

            async def export() -> None:
                folder = await pick_folder(c.store.get().output_folder, "Save the training data into…")
                if not folder:
                    return
                path, examples, letters = await run.io_bound(dataset.export, c.db, folder, images.value)
                ui.notify(f"Exported {examples} decisions from {letters} letters", type="positive")
                open_path(path.parent)

            ui.button("Export…", icon="download", on_click=export).props("unelevated color=primary no-caps")
