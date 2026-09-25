from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable, Iterable

from .database import AnalysisDatabase
from .service import analyze_file
from .image_formats import SUPPORTED_INPUT_EXTENSIONS

SUPPORTED = SUPPORTED_INPUT_EXTENSIONS


@dataclass(slots=True)
class BatchItemResult:
    path: Path
    status: str
    error: str | None = None


@dataclass(slots=True)
class BatchSummary:
    total: int
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    cached: int = 0
    cancelled: bool = False
    items: list[BatchItemResult] = field(default_factory=list)


def collect_images(root: str | Path, recursive: bool = True) -> list[Path]:
    p = Path(root)
    if p.is_file():
        return [p] if p.suffix.lower() in SUPPORTED else []
    if not p.is_dir():
        return []
    it = p.rglob("*") if recursive else p.glob("*")
    return sorted(x for x in it if x.is_file() and x.suffix.lower() in SUPPORTED)


def analyze_batch(
    paths: Iterable[str | Path],
    database: AnalysisDatabase | None = None,
    cancel_event: Event | None = None,
    progress: Callable[[int, int, Path, str], None] | None = None,
    pause_event: Event | None = None,
    precision: str = "normal",
) -> BatchSummary:
    files = [Path(p) for p in paths]
    summary = BatchSummary(total=len(files))
    cancel_event = cancel_event or Event()
    pause_event = pause_event or Event()

    for index, path in enumerate(files, start=1):
        while pause_event.is_set() and not cancel_event.is_set():
            cancel_event.wait(0.05)
        if cancel_event.is_set():
            summary.cancelled = True
            break
        if database and database.is_current(path, precision=precision):
            summary.cached += 1
            summary.processed += 1
            summary.items.append(BatchItemResult(path, "cached"))
            if progress:
                progress(index, len(files), path, "cached")
            continue
        try:
            manual = database.load_manual_vision_overrides(path) if database else {"faces": [], "eyes": []}
            result = analyze_file(
                path, precision=precision,
                manual_face_boxes=manual.get("faces", []), manual_eye_boxes=manual.get("eyes", []),
            )
            if database:
                database.save(result)
            summary.succeeded += 1
            status = "ok"
            error = None
        except Exception as exc:  # fault isolation: one bad image must not kill the batch
            summary.failed += 1
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        summary.processed += 1
        summary.items.append(BatchItemResult(path, status, error))
        if progress:
            progress(index, len(files), path, status)
    return summary
