"""Reusable UI building blocks."""

from __future__ import annotations

import os
import string
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from nicegui import app, ui

from ..db import Worker
from . import theme
from .context import ctx


# ----------------------------------------------------------------------------- layout bits
def page_header(title: str, subtitle: str = "") -> ui.row:
    with ui.row().classes("w-full items-end justify-between gap-4 no-wrap") as row:
        with ui.column().classes("gap-1"):
            ui.label(title).classes("le-title")
            if subtitle:
                ui.label(subtitle).classes("le-subtitle")
    return row


def card(classes: str = "") -> ui.card:
    return ui.card().classes(f"le-card w-full {classes}").props("flat")


def empty_state(icon: str, title: str, text: str = "") -> None:
    with ui.column().classes("le-empty w-full items-center gap-2"):
        ui.icon(icon, size="56px").classes("le-gradient-text")
        ui.label(title).classes("text-lg font-semibold")
        if text:
            ui.label(text).classes("le-muted max-w-md")


def status_chip(status: str) -> None:
    label, color, icon = theme.STATUS.get(status, (status, "grey", "help"))
    ui.chip(label, icon=icon, color=color, text_color="white").props("dense square").classes("text-xs font-semibold")


def confidence_bar(p: float | None, width: str = "w-28") -> None:
    if p is None:
        ui.label("–").classes("le-muted")
        return
    with ui.row().classes("items-center gap-2 no-wrap"):
        with ui.element("div").classes(f"le-bar {width}"):
            ui.element("div").style(f"width: {max(2, min(100, p * 100)):.0f}%")
        ui.label(f"{p:.0%}").classes("text-xs font-semibold le-muted w-9")


def worker_avatar(worker: Worker | None, size: str = "md") -> None:
    # ui.avatar's first argument is an icon name, so the initials go into a label inside it.
    with ui.avatar(color=worker.color if worker else "grey-5", text_color="white", size=size):
        ui.label(worker.initials if worker else "?").classes("font-bold").style("font-size: .8em; letter-spacing: .03em")


def relative_time(iso: str) -> str:
    try:
        then = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return ""
    seconds = (datetime.now(UTC) - then).total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)} min ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)} h ago"
    return then.astimezone().strftime("%d.%m.%Y %H:%M")


def probability_bars(decision: dict) -> None:
    """Show one decision of the decision model: question, chosen option, probabilities."""
    options = sorted(decision.get("options", []), key=lambda o: o["p"], reverse=True)
    with ui.column().classes("w-full gap-1"):
        ui.label(decision.get("question", "")).classes("text-sm font-semibold")
        for option in options[:6]:
            chosen = option["option"] == decision.get("choice")
            with ui.row().classes("w-full items-center gap-3 no-wrap"):
                ui.label(option["option"]).classes(
                    f"text-sm truncate {'font-semibold' if chosen else 'le-muted'}").style("width: 46%")
                with ui.element("div").classes("le-bar flex-grow"):
                    ui.element("div").style(f"width: {max(1, option['p'] * 100):.1f}%; {'' if chosen else 'opacity:.35'}")
                ui.label(f"{option['p']:.1%}").classes("text-xs font-semibold w-12 text-right")


# ----------------------------------------------------------------------------- OS integration
def open_path(path: str | Path) -> None:
    path = str(path)
    if not Path(path).exists():
        ui.notify("That file or folder no longer exists.", type="warning")
        return
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as exc:
        ui.notify(f"Could not open: {exc}", type="negative")


def reveal_path(path: str | Path) -> None:
    """Open the folder containing a file, selecting the file where the OS supports it."""
    p = Path(path)
    if sys.platform == "win32" and p.is_file():
        subprocess.Popen(f'explorer /select,"{p}"')
    elif sys.platform == "darwin" and p.exists():
        subprocess.Popen(["open", "-R", str(p)])
    else:
        open_path(p if p.is_dir() else p.parent)


# ----------------------------------------------------------------------------- folder picker
def _shortcuts() -> list[tuple[str, str, Path]]:
    home = Path.home()
    items = [("Home", "home", home)]
    for name, icon in (("Desktop", "desktop_windows"), ("Documents", "description"), ("Downloads", "download")):
        if (home / name).is_dir():
            items.append((name, icon, home / name))
    if sys.platform == "win32":
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                items.append((f"{letter}:", "storage", drive))
    else:
        items.append(("Computer", "storage", Path("/")))
        if sys.platform == "darwin" and Path("/Volumes").is_dir():
            items.append(("Volumes", "usb", Path("/Volumes")))
    return items


class FolderBrowser(ui.dialog):
    def __init__(self, initial: str = "", title: str = "Choose a folder"):
        super().__init__()
        start = Path(initial).expanduser() if initial else Path.home()
        while not start.is_dir() and start != start.parent:
            start = start.parent
        self.current = start if start.is_dir() else Path.home()
        with self, ui.card().classes("le-card").style("width: 680px; max-width: 95vw"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(title).classes("text-lg font-semibold")
                ui.button(icon="close", on_click=lambda: self.submit(None)).props("flat round dense")
            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                ui.button(icon="arrow_upward", on_click=self._up).props("flat round").tooltip("Parent folder")
                self.path_input = ui.input(value=str(self.current)).props("outlined dense").classes("flex-grow")
                self.path_input.on("keydown.enter", lambda: self._go(Path(self.path_input.value)))
            with ui.row().classes("w-full gap-1"):
                for label, icon, target in _shortcuts():
                    ui.button(label, icon=icon, on_click=lambda t=target: self._go(t)).props("flat dense no-caps size=sm")
            with ui.scroll_area().classes("w-full").style("height: 300px; border: 1px solid var(--le-border); border-radius: 12px"):
                self.listing = ui.column().classes("w-full gap-0")
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-2"):
                    self.new_name = ui.input(placeholder="New folder name").props("dense outlined").classes("w-48")
                    ui.button("Create", icon="create_new_folder", on_click=self._create).props("flat")
                with ui.row().classes("gap-2"):
                    ui.button("Cancel", on_click=lambda: self.submit(None)).props("flat")
                    ui.button("Select this folder", icon="check", on_click=lambda: self.submit(str(self.current))) \
                        .props("unelevated color=primary")
        self._render()

    def _render(self) -> None:
        self.path_input.value = str(self.current)
        self.listing.clear()
        try:
            folders = sorted((p for p in self.current.iterdir() if p.is_dir() and not p.name.startswith(".")),
                             key=lambda p: p.name.lower())
        except OSError as exc:
            with self.listing:
                ui.label(f"Cannot open this folder: {exc}").classes("le-muted p-4")
            return
        with self.listing:
            if not folders:
                ui.label("No subfolders").classes("le-muted p-4")
            for folder in folders[:500]:
                with ui.item(on_click=lambda f=folder: self._go(f)).props("clickable dense").classes("rounded-lg"):
                    with ui.item_section().props("avatar"):
                        ui.icon("folder", color="amber-7")
                    with ui.item_section():
                        ui.label(folder.name)

    def _go(self, path: Path) -> None:
        if path.is_dir():
            self.current = path
            self._render()
        else:
            ui.notify("Folder not found", type="warning")

    def _up(self) -> None:
        self._go(self.current.parent)

    def _create(self) -> None:
        name = (self.new_name.value or "").strip()
        if not name:
            return
        try:
            (self.current / name).mkdir(parents=False, exist_ok=True)
        except OSError as exc:
            ui.notify(f"Could not create folder: {exc}", type="negative")
            return
        self.new_name.value = ""
        self._go(self.current / name)


async def pick_folder(initial: str = "", title: str = "Choose a folder") -> str | None:
    if ctx().native:
        try:
            import webview

            dialog_type = webview.FileDialog.FOLDER if hasattr(webview, "FileDialog") else webview.FOLDER_DIALOG
            result = await app.native.main_window.create_file_dialog(  # type: ignore[union-attr]
                dialog_type, directory=initial or str(Path.home()))
            if result:
                return str(result[0] if isinstance(result, list | tuple) else result)
            return None
        except Exception:
            pass  # fall back to the built-in browser
    return await FolderBrowser(initial, title)


def folder_field(label: str, value: str, on_change, hint: str = "") -> ui.input:
    """Text field + browse button for a folder."""
    with ui.row().classes("w-full items-start gap-2 no-wrap"):
        field = ui.input(label, value=value, on_change=lambda e: on_change(e.value)).props("outlined").classes("flex-grow")
        if hint:
            field.props(f'hint="{hint}"')

        async def browse() -> None:
            chosen = await pick_folder(field.value, label)
            if chosen:
                field.value = chosen

        ui.button(icon="folder_open", on_click=browse).props("unelevated color=primary").classes("mt-1").style(
            "height: 48px; width: 52px").tooltip("Browse…")
    return field


# ----------------------------------------------------------------------------- live updates
def on_data_change(callback, interval: float = 1.0) -> ui.timer:
    """Call `callback` whenever the engine reports new activity or data changes."""
    feed = ctx().feed
    state = {"version": feed.version}

    def tick() -> None:
        if feed.version != state["version"]:
            state["version"] = feed.version
            callback()

    return ui.timer(interval, tick)
