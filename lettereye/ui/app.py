"""The application window: navigation, live status, page routing."""

from __future__ import annotations

from nicegui import app, ui

from .. import APP_NAME, paths
from . import theme
from .context import ctx
from .pages import approvals, dashboard, documents, insights, settings_page, setup, workers

NAV = [
    ("/", "Dashboard", "space_dashboard"),
    ("/approvals", "Approvals", "fact_check"),
    ("/documents", "Documents", "inventory_2"),
    ("/insights", "Insights", "insights"),
    ("/workers", "Workers", "groups"),
    ("/settings", "Settings", "tune"),
]

ROUTES = {
    "/": dashboard.page,
    "/approvals": approvals.page,
    "/review": approvals.page,
    "/documents": documents.page,
    "/insights": insights.page,
    "/workers": workers.page,
    "/settings": settings_page.page,
    "/setup": setup.page,
}


def _nav_matches(item_path: str, current: str) -> bool:
    current = current.split("?")[0].split("#")[0] or "/"
    return current == item_path if item_path == "/" else current.startswith(item_path)


def root() -> None:
    c = ctx()
    settings = c.store.get()
    theme.apply({"auto": None, "light": False, "dark": True}[settings.theme])
    ui.page_title(APP_NAME)

    if not settings.setup_completed:
        # First run: full-screen setup assistant without navigation.
        setup.page(first_run=True)
        return

    router = ui.context.client.sub_pages_router
    nav_items: dict[str, ui.row] = {}

    with ui.left_drawer(value=True, fixed=True).props("width=252 breakpoint=0").classes("le-drawer p-0"):
        with ui.column().classes("w-full h-full p-4 gap-1 no-wrap"):
            with ui.row().classes("items-center gap-3 px-2 pt-2 pb-5 no-wrap"):
                ui.image("/assets/icon.png").classes("w-10 h-10")
                with ui.column().classes("gap-0"):
                    ui.label("LetterEye").classes("text-white text-lg font-bold leading-tight")
                    ui.label("AI · local").classes("text-xs text-slate-400 leading-tight")
            for path, label, icon in NAV:
                with ui.row().classes("le-nav-item items-center w-full no-wrap") as item:
                    ui.icon(icon, size="20px")
                    ui.label(label).classes("text-[14px] font-medium")
                    if path == "/approvals":
                        review_badge = ui.label("").classes("le-nav-badge")
                item.on("click", lambda p=path: ui.navigate.to(p))
                nav_items[path] = item
            ui.space()
            with ui.column().classes("w-full gap-2 p-3 rounded-2xl").style("background: rgba(255,255,255,.05)"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    status_dot = ui.element("span").classes("le-status-dot le-dot-off")
                    status_label = ui.label("Stopped").classes("text-sm text-white font-medium")
                status_detail = ui.label("").classes("text-xs text-slate-400 break-all")

    def highlight(path: str) -> None:
        for item_path, element in nav_items.items():
            element.classes(add="active") if _nav_matches(item_path, path) else element.classes(remove="active")

    router.on_path_changed(highlight)
    highlight(router.current_path)

    last_version = {"v": -1}

    def refresh_shell() -> None:
        if c.feed.version == last_version["v"] and c.engine.state not in ("starting", "stopping"):
            return
        last_version["v"] = c.feed.version
        counts = c.db.count_by_status()
        count = counts["review"] + counts["pending"]
        review_badge.text = str(count) if count else ""
        review_badge.set_visibility(bool(count))
        state = c.engine.state
        status_dot.classes(remove="le-dot-on le-dot-off le-dot-warn")
        if state == "running":
            status_dot.classes(add="le-dot-on")
            current = c.engine.current
            status_label.text = "Working" if current else "Watching"
            status_detail.text = (f"{current.name} · {current.stage}" if current
                                  else c.store.get().inbox_folder)
        elif state in ("starting", "stopping"):
            status_dot.classes(add="le-dot-warn")
            status_label.text = state.capitalize() + "…"
            status_detail.text = ""
        else:
            status_dot.classes(add="le-dot-off")
            status_label.text = "Stopped"
            status_detail.text = c.engine.last_error

    ui.timer(1.0, refresh_shell)

    ui.sub_pages(ROUTES).classes("w-full")


def configure_static_files() -> None:
    app.add_static_files("/assets", str(paths.assets_dir()))
    app.add_media_files("/previews", str(paths.previews_dir()))
