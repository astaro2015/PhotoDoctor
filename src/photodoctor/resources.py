from __future__ import annotations

import sys
from pathlib import Path


def asset_path(name: str) -> Path:
    """Return a packaged/source asset path for both editable and PyInstaller runs."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "photodoctor" / "assets" / name
    return Path(__file__).resolve().parent / "assets" / name
