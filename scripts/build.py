"""Build a standalone LetterEye app with PyInstaller.

    uv run --extra build python scripts/build.py

Output: dist/LetterEye/ (with LetterEye.exe on Windows) or dist/LetterEye.app (macOS). The release pipeline
(.github/workflows/release.yml) turns this into an installer with automatic updates (Velopack).
Ollama is not bundled – users install it once from https://ollama.com/download (the app's setup
assistant walks them through it). Models are downloaded by the app into the user's data folder.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "lettereye" / "assets"


def main() -> None:
    icon = ASSETS / ("icon.ico" if sys.platform == "win32" else "icon.icns" if sys.platform == "darwin" else "icon.png")
    args = [
        str(ROOT / "lettereye" / "__main__.py"),
        "--name", "LetterEye",
        "--noconfirm",
        "--clean",
        "--windowed",
        "--onedir",
        "--icon", str(icon),
        "--add-data", f"{ASSETS}{os.pathsep}lettereye/assets",
        # packages with data files or native libraries that PyInstaller does not find by itself
        "--collect-all", "nicegui",
        "--collect-all", "rapidocr",
        "--collect-all", "pypdfium2",
        "--collect-all", "pypdfium2_raw",
        "--collect-all", "onnxruntime",
        "--collect-all", "webview",
        "--collect-all", "velopack",
        "--collect-submodules", "langchain_ollama",
        "--collect-submodules", "langchain_core",
        "--copy-metadata", "nicegui",
        "--copy-metadata", "langchain-core",
        "--copy-metadata", "langchain-ollama",
        "--copy-metadata", "rapidocr",
    ]
    if sys.platform == "darwin":
        args += ["--osx-bundle-identifier", "ai.lettereye.app"]
    PyInstaller.__main__.run(args)
    print(f"\nDone: {ROOT / 'dist' / 'LetterEye'}")


if __name__ == "__main__":
    main()
