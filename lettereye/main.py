"""Command line entry point.

    lettereye                      start the app (native window; falls back to the browser)
    lettereye --browser            start and open in the default browser instead
    lettereye analyze letter.pdf   run the pipeline on one file and print the result (nothing is moved)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import logging.handlers
import socket
import sys
import threading
import webbrowser

from . import APP_NAME, __version__, paths


def _setup_logging(verbose: bool) -> None:
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(paths.log_dir() / "lettereye.log", maxBytes=2_000_000, backupCount=3,
                                             encoding="utf-8"),
    ]
    if sys.stderr is not None:  # windowed PyInstaller builds have no console
        try:  # a redirected Windows console may not be UTF-8; never fail on an umlaut or arrow
            sys.stderr.reconfigure(errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "watchdog", "RapidOCR", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _native_available() -> bool:
    if importlib.util.find_spec("webview") is None:
        return False
    if sys.platform.startswith("linux"):
        # pywebview needs GTK (python-gi) or Qt bindings on Linux.
        return any(importlib.util.find_spec(m) for m in ("gi", "qtpy", "PyQt6", "PySide6", "PyQt5"))
    return True


def _cmd_analyze(path: str) -> int:
    from .db import Database
    from .events import ActivityFeed
    from .services.engine import Engine
    from .settings import SettingsStore

    db, store = Database(), SettingsStore()
    engine = Engine(db, store, ActivityFeed())
    analysis = engine.analyze_file(path, on_stage=lambda s: print(f"… {s}", file=sys.stderr))
    print(json.dumps({
        "status": analysis.status,
        "reason": analysis.reason,
        "worker": analysis.decision.worker.name if analysis.decision.worker else None,
        "recipient": analysis.decision.recipient.name if analysis.decision.recipient else None,
        "confidence": round(analysis.confidence, 4),
        "doc_type": analysis.decision.doc_type,
        "letter_date": analysis.letter_date,
        "facts": analysis.facts.model_dump(),
        "ocr_source": analysis.ocr.source,
        "escalated": analysis.escalated,
        "trace": analysis.trace(),
    }, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lettereye", description=f"{APP_NAME} {__version__}")
    parser.add_argument("command", nargs="?", choices=["run", "analyze"], default="run")
    parser.add_argument("file", nargs="?", help="file for 'analyze'")
    parser.add_argument("--browser", action="store_true", help="open in the web browser instead of a window")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: only this computer)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", help="store settings, database and models here instead of the app-data folder")
    parser.add_argument("--no-autostart", action="store_true", help="do not start watching automatically")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    args = parser.parse_args(argv)

    if args.data_dir:
        paths.set_data_dir(args.data_dir)
    _setup_logging(args.verbose)

    if args.command == "analyze":
        if not args.file:
            parser.error("analyze needs a file")
        return _cmd_analyze(args.file)

    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    if _port_in_use(browse_host, args.port):
        # Already running: just bring it up in the browser instead of starting a second instance.
        webbrowser.open(f"http://{browse_host}:{args.port}")
        return 0

    from nicegui import app, ui

    from .db import Database
    from .events import ActivityFeed
    from .services.engine import Engine
    from .settings import SettingsStore
    from .ui import context
    from .ui.app import configure_static_files, root

    native = not args.browser and _native_available()
    db, store, feed = Database(), SettingsStore(), ActivityFeed()
    engine = Engine(db, store, feed)
    context.init(context.AppContext(db=db, store=store, feed=feed, engine=engine, native=native))
    configure_static_files()

    def autostart() -> None:
        settings = store.get()
        if settings.setup_completed and settings.autostart and not args.no_autostart:
            problems = engine.start()
            if problems:
                feed.add(f"Could not start watching: {problems[0]}", "warning")

    app.on_startup(lambda: threading.Thread(target=autostart, name="autostart", daemon=True).start())
    app.on_shutdown(engine.stop)

    ui.run(
        root,
        title=APP_NAME,
        host=args.host,
        port=args.port,
        native=native,
        window_size=(1380, 900) if native else None,
        reload=False,
        show=not native,
        favicon=paths.assets_dir() / "icon.png",
        dark=None,
        show_welcome_message=False,
        storage_secret="lettereye-local",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
