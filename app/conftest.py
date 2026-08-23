"""Pytest configuration: ensure the app package is importable, keep CLI help plain."""

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# typer forces rich terminal rendering (ANSI styles, narrow wrapping) when
# GITHUB_ACTIONS / FORCE_COLOR are set, which splits option names in ``--help``
# output and breaks substring assertions. Read at typer import time, so set here.
os.environ.setdefault("_TYPER_FORCE_DISABLE_TERMINAL", "1")
os.environ.setdefault("TERMINAL_WIDTH", "200")
