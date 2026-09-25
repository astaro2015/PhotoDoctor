from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_dir: str | Path | None = None) -> Path:
    root = Path(log_dir) if log_dir else (Path.home() / ".photodoctor" / "logs")
    root.mkdir(parents=True, exist_ok=True)
    log_file = root / "photodoctor.log"
    logger = logging.getLogger()
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return log_file
