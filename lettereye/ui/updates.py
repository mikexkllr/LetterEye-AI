"""Update status and controls: the settings card and the restart banner."""

from __future__ import annotations

from nicegui import run, ui

from .. import buildinfo
from .context import ctx

CHANNELS = {"stable": "Stable – the main branch", "dev": "Dev – the dev branch, newest changes for testing"}


def version_line() -> str:
    parts = [f"Version {buildinfo.VERSION}"]
    if buildinfo.CHANNEL in ("stable", "dev"):
        parts.append(f"{buildinfo.CHANNEL} build")
    else:
        parts.append("running from source")
    if buildinfo.COMMIT:
        parts.append(f"commit {buildinfo.COMMIT[:7]}")
    if buildinfo.BUILT:
        parts.append(buildinfo.BUILT)
    return " · ".join(parts)


def updates_card(draft: dict) -> None:
    """Settings → General → Updates. Changes here apply immediately (no Save needed)."""
    c = ctx()
    updater = c.updater
    settings = c.store.get()

    def save(key: str, value) -> None:
        draft[key] = value  # keep the settings page's draft in sync so "Save" does not undo this
        c.store.update(**{key: value})
        if updater is not None and key == "update_channel":
            updater.check_soon()

    with ui.column().classes("w-full gap-3"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("system_update", size="28px").classes("le-gradient-text")
            with ui.column().classes("gap-0"):
                ui.label("Updates").classes("le-section-title")
                ui.label(version_line()).classes("le-muted text-xs")
        if updater is None or not updater.installed:
            ui.label(updater.message if updater else "Updates are not available.").classes("text-sm le-muted")
            ui.markdown("Install LetterEye from the "
                        "[GitHub releases](https://github.com/mikexkllr/LetterEye-AI/releases) to get automatic "
                        "updates: **stable** follows the `main` branch, **dev** follows `dev`.").classes("text-sm")
            return
        with ui.row().classes("gap-4 items-start"):
            ui.select(CHANNELS, value=settings.update_channel or updater.channel, label="Update channel",
                      on_change=lambda e: save("update_channel", e.value)).props("outlined dense").classes("w-96")
            ui.select({0: "Automatic (dev: 5 min, stable: 1 h)", 5: "Every 5 minutes", 15: "Every 15 minutes",
                       60: "Every hour", 360: "Every 6 hours"}, value=settings.update_check_minutes,
                      label="Check for updates", on_change=lambda e: save("update_check_minutes", e.value)) \
                .props("outlined dense").classes("w-64")
        ui.switch("Install updates automatically when no letter is being processed (after a 60 s countdown)",
                  value=settings.auto_update, on_change=lambda e: save("auto_update", e.value))

        status = ui.row().classes("w-full items-center gap-3 p-3 rounded-xl").style("background: var(--le-soft)")

        def render_status() -> None:
            status.clear()
            with status:
                icon = {"checking": "sync", "downloading": "downloading", "ready": "new_releases",
                        "available": "new_releases", "error": "error", "up_to_date": "check_circle",
                        "applying": "restart_alt"}.get(updater.state, "info")
                ui.icon(icon, color="negative" if updater.state == "error" else "primary")
                with ui.column().classes("gap-0 flex-grow"):
                    ui.label(updater.message or "No check yet.").classes("text-sm font-medium")
                    if updater.last_check:
                        ui.label(f"Last check {updater.last_check:%H:%M:%S} · {updater.channel} channel") \
                            .classes("le-muted text-xs")
                    if updater.state == "downloading" and updater.progress is not None:
                        ui.linear_progress(value=updater.progress, show_value=False).props("rounded size=6px")

                async def check() -> None:
                    found = await run.io_bound(updater.check)
                    if found and c.store.get().auto_update:
                        await run.io_bound(updater.download)
                    render_status()

                async def install() -> None:
                    if updater.state == "available":
                        await run.io_bound(updater.download)
                    await run.io_bound(updater.apply_now)

                if updater.state in ("ready", "available"):
                    ui.button(f"Install {updater.available} now", icon="restart_alt", on_click=install) \
                        .props("unelevated color=primary no-caps dense")
                ui.button("Check now", icon="refresh", on_click=check).props("flat no-caps dense")
            if updater.notes and updater.state in ("ready", "available"):
                with status:
                    with ui.expansion("What's new", icon="notes").classes("w-full"):
                        ui.markdown(updater.notes).classes("text-sm")

        render_status()
        last = {"key": None}

        def tick() -> None:
            key = (updater.state, updater.message, round(updater.progress or 0, 2))
            if key != last["key"]:
                last["key"] = key
                render_status()

        ui.timer(1.0, tick)


def update_banner() -> None:
    """A banner above every page while an update countdown runs or an update waits."""
    c = ctx()
    updater = c.updater
    if updater is None or not updater.installed:
        return
    with ui.element("div").classes("le-update-banner") as banner:
        ui.icon("system_update", size="20px")
        text = ui.label("").classes("text-sm font-medium")
        ui.space()
        ui.button("Restart now", on_click=lambda: run.io_bound(updater.apply_now)) \
            .props("unelevated dense no-caps color=white text-color=primary")
        ui.button("Later", on_click=lambda: (updater.postpone(30), banner.set_visibility(False))) \
            .props("flat dense no-caps color=white")
    banner.set_visibility(False)

    def tick() -> None:
        seconds = updater.countdown
        if updater.state == "ready" and seconds is not None:
            text.text = f"LetterEye restarts in {seconds} s to install version {updater.available}."
            banner.set_visibility(True)
        elif updater.state == "ready" and not c.store.get().auto_update:
            text.text = f"Version {updater.available} is ready. Restart when it suits you."
            banner.set_visibility(True)
        elif updater.state == "applying":
            text.text = f"Installing version {updater.available} – LetterEye restarts in a moment…"
            banner.set_visibility(True)
        else:
            banner.set_visibility(False)

    ui.timer(1.0, tick)
