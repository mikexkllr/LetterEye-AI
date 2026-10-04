import multiprocessing
import os
import sys


def _frozen_startup() -> None:
    """Things an installed (PyInstaller + Velopack) build must do before anything else."""
    # Windowed builds have no console: give libraries that print somewhere harmless to write to.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    try:  # handles Velopack's install/update/uninstall hooks (exits for those) and finishes pending updates
        import velopack

        velopack.App().run()
    except Exception:  # not installed through Velopack (e.g. a plain PyInstaller folder)
        pass


if __name__ == "__main__":
    multiprocessing.freeze_support()  # must run first: child processes of the native window exit here
    if getattr(sys, "frozen", False):
        _frozen_startup()
    from lettereye.main import main

    sys.exit(main())
