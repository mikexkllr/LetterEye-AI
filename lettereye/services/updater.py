"""Automatic updates from GitHub releases (Velopack).

Two channels, built by the release pipeline:
* stable – every push to `main`
* dev    – every push to `dev` (pre-releases), for testing changes directly on the target computer

The installed app checks GitHub, downloads updates in the background and applies them when no letter is
being processed – after a visible countdown that can be postponed. A copy running from source (git checkout)
does not update itself.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime

from .. import buildinfo
from ..events import ActivityFeed
from ..settings import SettingsStore

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/mikexkllr/LetterEye-AI"
COUNTDOWN_SECONDS = 60


def platform_tag() -> str:
    return "win" if sys.platform == "win32" else "osx" if sys.platform == "darwin" else "linux"


def velopack_channel(channel: str) -> str:
    """Release feed name, e.g. 'stable-win' or 'dev-osx' (matches `vpk pack --channel` in the pipeline)."""
    return f"{channel}-{platform_tag()}"


class Updater:
    def __init__(self, store: SettingsStore, feed: ActivityFeed, is_idle: Callable[[], bool],
                 shutdown: Callable[[], None]):
        self.store = store
        self.feed = feed
        self.is_idle = is_idle
        self.shutdown = shutdown
        self.state = "idle"  # disabled | idle | checking | up_to_date | downloading | ready | applying | error
        self.message = ""
        self.progress: float | None = None
        self.available: str = ""
        self.notes: str = ""
        self.last_check: datetime | None = None
        self.countdown_until: float | None = None
        self.postponed_until: float = 0.0
        self._info = None
        self._manager = None
        self._manager_channel = ""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._check_now = threading.Event()
        self.installed = self._detect_installation()
        if not self.installed:
            self.state = "disabled"
            self.message = ("Automatic updates work in the installed app (from GitHub releases). This copy runs "
                            "from source – update it with git pull.")

    # ----------------------------------------------------------------- environment
    @property
    def channel(self) -> str:
        chosen = self.store.get().update_channel
        if chosen:
            return chosen
        return buildinfo.CHANNEL if buildinfo.CHANNEL in ("stable", "dev") else "stable"

    def _detect_installation(self) -> bool:
        if not getattr(sys, "frozen", False):
            return False
        try:
            self._get_manager()
            return True
        except Exception as exc:
            log.info("Updates disabled (not a Velopack installation): %s", _first_line(exc))
            return False

    def _get_manager(self):
        channel = self.channel
        if self._manager is None or self._manager_channel != channel:
            import velopack

            switching = buildinfo.CHANNEL in ("stable", "dev") and channel != buildinfo.CHANNEL
            options = velopack.UpdateOptions(AllowVersionDowngrade=switching, MaximumDeltasBeforeFallback=10,
                                             ExplicitChannel=velopack_channel(channel))
            source = velopack.GithubSource(REPO_URL, None, channel == "dev")
            self._manager = velopack.UpdateManager(source, options)
            self._manager_channel = channel
            self._info = None
        return self._manager

    # ----------------------------------------------------------------- actions
    def check(self) -> bool:
        """Look for a newer release on the current channel. Returns True if one is available."""
        if not self.installed:
            return False
        with self._lock:
            self.state, self.message = "checking", "Looking for updates…"
            try:
                info = self._get_manager().check_for_updates()
            except Exception as exc:
                self.state, self.message = "error", f"Update check failed: {_first_line(exc)}"
                log.warning(self.message)
                return False
            finally:
                self.last_check = datetime.now()
            if not info:
                self.state, self.message, self.available = "up_to_date", "LetterEye is up to date.", ""
                return False
            self._info = info
            target = info.TargetFullRelease
            self.available = target.Version
            self.notes = target.NotesMarkdown or ""
            self.state, self.message = "available", f"Version {target.Version} is available."
            return True

    def download(self) -> bool:
        if self._info is None:
            return False
        with self._lock:
            self.state, self.progress = "downloading", 0.0
            self.message = f"Downloading version {self.available}…"
            try:
                self._get_manager().download_updates(self._info, lambda p: setattr(self, "progress", p / 100))
            except Exception as exc:
                self.state, self.message = "error", f"Download failed: {_first_line(exc)}"
                log.warning(self.message)
                return False
            self.state, self.progress = "ready", 1.0
            self.message = f"Version {self.available} is ready to install."
            self.feed.add(f"Update {self.available} downloaded ({self.channel} channel)", "info")
            return True

    def apply_now(self) -> None:
        """Restart into the new version. Velopack replaces the files once this process has exited."""
        if self._info is None or self.state not in ("ready", "available"):
            return
        self.state, self.message = "applying", f"Installing version {self.available}…"
        self.feed.add(f"Restarting to install version {self.available}", "info")
        try:
            self._get_manager().wait_exit_then_apply_updates(self._info, True, True, None)
        except Exception as exc:
            self.state, self.message = "error", f"Could not install the update: {_first_line(exc)}"
            log.error(self.message)
            return
        self.shutdown()

    def postpone(self, minutes: int = 30) -> None:
        self.countdown_until = None
        self.postponed_until = time.monotonic() + minutes * 60

    def check_soon(self) -> None:
        self._check_now.set()

    @property
    def countdown(self) -> int | None:
        if self.countdown_until is None:
            return None
        return max(0, int(self.countdown_until - time.monotonic()))

    # ----------------------------------------------------------------- background loop
    def start(self) -> None:
        if self.installed:
            threading.Thread(target=self._loop, name="updater", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        self._check_now.set()

    def _interval(self) -> float:
        minutes = self.store.get().update_check_minutes or (5 if self.channel == "dev" else 60)
        return max(1, minutes) * 60

    def _loop(self) -> None:
        next_check = time.monotonic() + 20  # let the app start first
        while not self._stop.is_set():
            settings = self.store.get()
            now = time.monotonic()
            if (now >= next_check or self._check_now.is_set()) and self.state not in ("downloading", "applying"):
                self._check_now.clear()
                next_check = now + self._interval()
                if self.check() and settings.auto_update:
                    self.download()
            if self.state == "ready" and settings.auto_update and now >= self.postponed_until:
                if not self.is_idle():
                    self.countdown_until = None  # never interrupt a letter
                elif self.countdown_until is None:
                    self.countdown_until = now + COUNTDOWN_SECONDS
                    self.feed.touch()
                elif now >= self.countdown_until:
                    self.apply_now()
            self._stop.wait(2)


def _first_line(exc: BaseException) -> str:
    """Velopack errors can carry a long native backtrace; people only need the first line."""
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
