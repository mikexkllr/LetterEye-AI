"""Write lettereye/buildinfo.py for a release build (used by the GitHub Actions pipeline)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "lettereye" / "buildinfo.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", required=True, choices=["stable", "dev"])
    parser.add_argument("--commit", default="")
    args = parser.parse_args()
    TARGET.write_text(
        '"""Build metadata. The release pipeline overwrites this file (scripts/write_buildinfo.py) before packaging."""\n\n'
        f"VERSION = {args.version!r}\n"
        f"CHANNEL = {args.channel!r}\n"
        f"COMMIT = {args.commit[:12]!r}\n"
        f"BUILT = {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')!r}\n",
        encoding="utf-8",
    )
    print(TARGET.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
