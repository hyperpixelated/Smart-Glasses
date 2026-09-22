"""Verify that the smart-glasses development dependencies are available."""

from __future__ import annotations

import importlib
import io
import sys
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout


def check_python() -> tuple[bool, str]:
    """Return the Python interpreter version currently executing this script."""
    return True, sys.version.split()[0]


def check_package(module_name: str) -> tuple[bool, str]:
    """Import one package and return its version, without triggering any ML work."""
    try:
        # Some libraries print setup notices during import; keep verification output
        # limited to the standardized status lines below.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "version unavailable")
        return True, str(version)
    except Exception as error:  # Report any unavailable or broken dependency.
        return False, f"{type(error).__name__}: {error}"


def main() -> int:
    """Print dependency status and return a process exit status."""
    checks: tuple[tuple[str, Callable[[], tuple[bool, str]]], ...] = (
        ("Python", check_python),
        ("PyTorch", lambda: check_package("torch")),
        ("Ultralytics", lambda: check_package("ultralytics")),
        ("OpenCV", lambda: check_package("cv2")),
        ("NumPy", lambda: check_package("numpy")),
        ("pytest", lambda: check_package("pytest")),
    )

    print("Smart Glasses Environment Verification")
    print("=" * 39)

    all_passed = True
    for name, check in checks:
        passed, version = check()
        status = "PASS" if passed else "FAIL"
        print(f"{status:<4} {name}: {version}")
        all_passed = all_passed and passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
