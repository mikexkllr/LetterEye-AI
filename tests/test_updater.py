import sys
import time
import types

import pytest

from lettereye import buildinfo
from lettereye.events import ActivityFeed
from lettereye.services import updater as updater_module
from lettereye.services.updater import Updater, velopack_channel
from lettereye.settings import SettingsStore


class FakeAsset:
    def __init__(self, version):
        self.Version = version
        self.NotesMarkdown = "- faster OCR"


class FakeVelopack(types.ModuleType):
    def __init__(self, newer: str | None):
        super().__init__("velopack")
        self.newer = newer
        self.created = []
        self.applied = []
        module = self

        class UpdateOptions:
            def __init__(self, AllowVersionDowngrade, MaximumDeltasBeforeFallback, ExplicitChannel=None):  # noqa: N803
                self.AllowVersionDowngrade = AllowVersionDowngrade
                self.ExplicitChannel = ExplicitChannel

        class GithubSource:
            def __init__(self, repo_url, access_token=None, prerelease=False):
                self.repo_url, self.prerelease = repo_url, prerelease

        class UpdateInfo:
            def __init__(self, version):
                self.TargetFullRelease = FakeAsset(version)

        class UpdateManager:
            def __init__(self, source, options):
                module.created.append((source, options))

            def check_for_updates(self):
                return UpdateInfo(module.newer) if module.newer else None

            def download_updates(self, info, progress_callback=None):
                for p in (10, 60, 100):
                    progress_callback(p)

            def wait_exit_then_apply_updates(self, info, silent=False, restart=True, restart_args=None):
                module.applied.append((info.TargetFullRelease.Version, silent, restart))

        self.UpdateOptions, self.GithubSource, self.UpdateManager = UpdateOptions, GithubSource, UpdateManager


@pytest.fixture
def installed(monkeypatch, tmp_path):
    def make(newer="2.1.99", channel="dev", idle=True):
        fake = FakeVelopack(newer)
        monkeypatch.setitem(sys.modules, "velopack", fake)
        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(buildinfo, "CHANNEL", channel)
        store = SettingsStore(tmp_path / "settings.json")
        state = {"idle": idle, "shutdown": 0}
        up = Updater(store, ActivityFeed(), is_idle=lambda: state["idle"],
                     shutdown=lambda: state.__setitem__("shutdown", state["shutdown"] + 1))
        return up, fake, store, state
    return make


def test_channel_names_match_the_release_pipeline(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert velopack_channel("stable") == "stable-win"
    monkeypatch.setattr(sys, "platform", "darwin")
    assert velopack_channel("dev") == "dev-osx"


def test_dev_builds_follow_prereleases_and_can_switch_to_stable(installed):
    up, fake, store, _ = installed(channel="dev")
    assert up.installed and up.channel == "dev"
    source, options = fake.created[-1]
    assert source.prerelease and source.repo_url.endswith("mikexkllr/LetterEye-AI")
    assert options.ExplicitChannel.startswith("dev-") and not options.AllowVersionDowngrade
    store.update(update_channel="stable")
    up.check()
    source, options = fake.created[-1]
    assert not source.prerelease and options.ExplicitChannel.startswith("stable-")
    assert options.AllowVersionDowngrade  # dev -> stable may go to a lower version number


def test_check_download_and_apply_when_idle(installed, monkeypatch):
    monkeypatch.setattr(updater_module, "COUNTDOWN_SECONDS", 0)
    up, fake, _, state = installed(newer="2.1.99")
    assert up.check() and up.available == "2.1.99" and "faster OCR" in up.notes
    assert up.download() and up.state == "ready" and up.progress == 1.0
    up._stop.clear()
    loop = __import__("threading").Thread(target=up._loop, daemon=True)
    loop.start()
    deadline = time.time() + 10
    while not fake.applied and time.time() < deadline:
        time.sleep(0.1)
    up.stop()
    assert fake.applied == [("2.1.99", True, True)]
    assert state["shutdown"] == 1


def test_never_interrupts_a_letter(installed, monkeypatch):
    monkeypatch.setattr(updater_module, "COUNTDOWN_SECONDS", 0)
    up, fake, _, _ = installed(newer="2.1.99", idle=False)
    up.check()
    up.download()
    loop = __import__("threading").Thread(target=up._loop, daemon=True)
    loop.start()
    time.sleep(2.5)
    up.stop()
    assert fake.applied == [] and up.countdown is None


def test_up_to_date_and_source_mode(installed, monkeypatch, tmp_path):
    up, _, _, _ = installed(newer=None)
    assert not up.check() and up.state == "up_to_date"
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    source_copy = Updater(SettingsStore(tmp_path / "s2.json"), ActivityFeed(), lambda: True, lambda: None)
    assert not source_copy.installed and source_copy.state == "disabled"
