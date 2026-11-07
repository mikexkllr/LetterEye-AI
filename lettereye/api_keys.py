"""API keys for the providers you bring your own key for.

Keys are never written to settings.json. They are kept in `secrets.json` in the app-data folder:

* Windows: encrypted with DPAPI for the current Windows user (only that user on that PC can decrypt them).
* macOS / Linux: in a file only the current user can read (mode 0600), like ~/.aws/credentials.

The OS keychain is not used on purpose: LetterEye is not code-signed and updates itself, and macOS would
ask for keychain access again after every update.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import threading
from pathlib import Path

from . import paths

log = logging.getLogger(__name__)

_FORMAT = 1


class SecretStore:
    def __init__(self, path: Path | None = None):
        self._path = path or paths.data_dir() / "secrets.json"
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ public API
    def get(self, name: str) -> str:
        with self._lock:
            value = self._read().get(name)
        if not value:
            return ""
        try:
            return _unprotect(value)
        except Exception as exc:  # e.g. the file was copied from another Windows user
            log.warning("Could not decrypt the stored key %r: %s", name, exc)
            return ""

    def set(self, name: str, value: str) -> None:
        """Store a key; an empty value removes it."""
        value = value.strip()
        with self._lock:
            data = self._read()
            if value:
                data[name] = _protect(value)
            else:
                data.pop(name, None)
            self._write(data)

    def delete(self, name: str) -> None:
        self.set(name, "")

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._read())

    def prune(self, keep: set[str]) -> None:
        """Remove the keys of providers that no longer exist."""
        with self._lock:
            data = self._read()
            stale = [name for name in data if name not in keep]
            if stale:
                for name in stale:
                    data.pop(name)
                self._write(data)

    # ------------------------------------------------------------------ storage
    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in (raw.get("secrets") or {}).items()}
        except (OSError, ValueError, AttributeError) as exc:
            log.error("Could not read %s (%s); stored API keys are ignored", self._path, exc)
            return {}

    def _write(self, data: dict[str, str]) -> None:
        payload = json.dumps({"format": _FORMAT, "protection": _PROTECTION, "secrets": data}, indent=2)
        tmp = self._path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        if sys.platform != "win32":
            os.chmod(tmp, 0o600)  # in case the file existed with wider permissions
        os.replace(tmp, self._path)


def mask(key: str) -> str:
    """What the UI shows of a key: its last four characters."""
    key = key.strip()
    return key[-4:] if len(key) >= 12 else ""


# ---------------------------------------------------------------------- protection per platform
if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _PROTECTION = "dpapi"

    class _Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _crypt32.CryptProtectData.argtypes = [ctypes.POINTER(_Blob), wintypes.LPCWSTR, ctypes.POINTER(_Blob),
                                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    _crypt32.CryptProtectData.restype = wintypes.BOOL
    _crypt32.CryptUnprotectData.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.POINTER(_Blob),
                                            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(_Blob)]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p
    _UI_FORBIDDEN = 0x01

    def _call(function, data: bytes) -> bytes:
        buffer = ctypes.create_string_buffer(data, len(data))
        blob_in = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
        blob_out = _Blob()
        if function is _crypt32.CryptProtectData:
            ok = function(ctypes.byref(blob_in), "LetterEye", None, None, None, _UI_FORBIDDEN, ctypes.byref(blob_out))
        else:
            ok = function(ctypes.byref(blob_in), None, None, None, None, _UI_FORBIDDEN, ctypes.byref(blob_out))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            _kernel32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))

    def _protect(value: str) -> str:
        return "dpapi:" + base64.b64encode(_call(_crypt32.CryptProtectData, value.encode("utf-8"))).decode("ascii")

    def _unprotect(stored: str) -> str:
        if not stored.startswith("dpapi:"):
            return stored
        return _call(_crypt32.CryptUnprotectData, base64.b64decode(stored[6:])).decode("utf-8")

else:
    _PROTECTION = "user-only-file"

    def _protect(value: str) -> str:
        return value

    def _unprotect(stored: str) -> str:
        return stored
