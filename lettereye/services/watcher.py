"""Watching the inbox folder for new scans.

watchdog delivers events instantly; a periodic rescan catches what it misses (network shares, files
copied while the app was busy). Before a file is handed on, we wait until the scanner has finished
writing it: the size must be stable and, on Windows, the file must not be locked by another process.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..ocr.documents import is_supported

log = logging.getLogger(__name__)

_TEMP_PREFIXES = ("~$", ".", "._")
_TEMP_SUFFIXES = (".tmp", ".part", ".crdownload", ".partial", ".download")


def is_candidate(path: Path) -> bool:
    name = path.name
    return is_supported(path) and not name.startswith(_TEMP_PREFIXES) and not name.lower().endswith(_TEMP_SUFFIXES)


def _windows_exclusive_open_fails(path: Path) -> bool:
    """Try to open the file with no sharing allowed. Fails while any other process still has it open."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    generic_read, open_existing, file_attribute_normal = 0x80000000, 3, 0x80
    handle = kernel32.CreateFileW(str(path), generic_read, 0, None, open_existing, file_attribute_normal, None)
    if handle is None or handle == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        return error in (32, 33)  # ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
    kernel32.CloseHandle(handle)
    return False


def is_locked(path: Path) -> bool:
    """True while another process (scanner, sync tool, copy) still holds the file."""
    try:
        if sys.platform == "win32":
            return _windows_exclusive_open_fails(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        os.close(fd)
        return False
    except FileNotFoundError:
        return False
    except OSError:
        return True


def wait_until_ready(path: Path, stop: threading.Event | None = None, timeout: float = 300,
                     interval: float = 1.0, stable_checks: int = 2) -> bool:
    """Block until the file has stopped growing and is unlocked. False if it vanished or timed out."""
    deadline = time.monotonic() + timeout
    last: tuple[int, int] | None = None
    stable = 0
    while time.monotonic() < deadline:
        if stop is not None and stop.is_set():
            return False
        try:
            st = path.stat()
        except FileNotFoundError:
            return False
        current = (st.st_size, st.st_mtime_ns)
        if current == last and st.st_size > 0:
            stable += 1
            if stable >= stable_checks and not is_locked(path):
                return True
        else:
            stable = 0
        last = current
        time.sleep(interval)
    return False


class _Handler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[Path], None]):
        self.callback = callback

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.callback(Path(str(event.src_path)))

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self.callback(Path(str(event.dest_path)))


class InboxWatcher:
    def __init__(self, folder: str | Path, on_file: Callable[[Path], None], recursive: bool = False,
                 rescan_interval: float = 20.0, ignore: Callable[[Path], bool] | None = None):
        self.folder = Path(folder)
        self.on_file = on_file
        self.recursive = recursive
        self.rescan_interval = rescan_interval
        self.ignore = ignore or (lambda _p: False)
        self._observer: Observer | None = None
        self._stop = threading.Event()
        self._rescan_thread: threading.Thread | None = None

    def _offer(self, path: Path) -> None:
        if is_candidate(path) and not self.ignore(path):
            self.on_file(path)

    def scan(self) -> list[Path]:
        pattern = "**/*" if self.recursive else "*"
        found = sorted((p for p in self.folder.glob(pattern) if p.is_file() and is_candidate(p)),
                       key=lambda p: p.stat().st_mtime)
        for path in found:
            if not self.ignore(path):
                self.on_file(path)
        return found

    def start(self, scan_existing: bool = True) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._observer = Observer()
        self._observer.schedule(_Handler(self._offer), str(self.folder), recursive=self.recursive)
        self._observer.daemon = True
        self._observer.start()
        if scan_existing:
            self.scan()
        self._rescan_thread = threading.Thread(target=self._rescan_loop, name="inbox-rescan", daemon=True)
        self._rescan_thread.start()

    def _rescan_loop(self) -> None:
        while not self._stop.wait(self.rescan_interval):
            try:
                self.scan()
            except OSError as exc:  # folder temporarily unavailable (e.g. network drive)
                log.warning("Rescan of %s failed: %s", self.folder, exc)

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
