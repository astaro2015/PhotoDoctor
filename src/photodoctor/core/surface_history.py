from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image

HISTORY_KEY = "PhotoDoctorSurfaceHistory"
JPEG_HISTORY_PREFIX = b"PhotoDoctorSurfaceHistory:"
HISTORY_VERSION = 1
MAX_HISTORY_CANDIDATES = 256


def _round01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return round(min(1.0, max(0.0, number)), 6)


def compact_surface_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    contour = candidate.get("contour")
    if isinstance(contour, list):
        pts: list[dict[str, float]] = []
        for point in contour:
            if not isinstance(point, Mapping):
                continue
            pts.append({"x": _round01(point.get("x", 0.0)), "y": _round01(point.get("y", 0.0))})
        if len(pts) >= 2:
            out["contour"] = pts[:32]
    endpoints = candidate.get("line_endpoints")
    if isinstance(endpoints, list):
        pts = []
        for point in endpoints[:2]:
            if not isinstance(point, Mapping):
                continue
            pts.append({"x": _round01(point.get("x", 0.0)), "y": _round01(point.get("y", 0.0))})
        if len(pts) == 2:
            out["line_endpoints"] = pts
    for key in ("x", "y", "w", "h"):
        if key in candidate:
            out[key] = _round01(candidate.get(key, 0.0))
    for key in ("polarity", "candidate_kind", "detection_branch"):
        value = candidate.get(key)
        if value is not None:
            out[key] = str(value)[:64]
    for key in ("stroke_width_px", "orientation_deg"):
        if key in candidate:
            try:
                out[key] = round(float(candidate.get(key, 0.0) or 0.0), 3)
            except (TypeError, ValueError, OverflowError):
                pass
    return out


def normalize_surface_history(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"version": HISTORY_VERSION, "passes": 0, "candidates": []}
    candidates = payload.get("candidates", [])
    clean = [compact_surface_candidate(item) for item in candidates if isinstance(item, Mapping)] if isinstance(candidates, list) else []
    clean = [item for item in clean if item][:MAX_HISTORY_CANDIDATES]
    try:
        passes = max(0, int(payload.get("passes", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        passes = 0
    return {"version": HISTORY_VERSION, "passes": passes, "candidates": clean}


def merge_surface_history(existing: Any, candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    base = normalize_surface_history(existing)
    merged = list(base["candidates"])
    seen = {json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True) for item in merged}
    for candidate in candidates:
        compact = compact_surface_candidate(candidate)
        if not compact:
            continue
        key = json.dumps(compact, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(compact)
    if len(merged) > MAX_HISTORY_CANDIDATES:
        merged = merged[-MAX_HISTORY_CANDIDATES:]
    return {"version": HISTORY_VERSION, "passes": int(base["passes"]) + 1, "candidates": merged}


def encode_surface_history(payload: Any) -> str:
    normalized = normalize_surface_history(payload)
    return json.dumps(normalized, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def decode_surface_history(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return normalize_surface_history(None)
    if not isinstance(value, str) or not value.strip():
        return normalize_surface_history(None)
    try:
        return normalize_surface_history(json.loads(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return normalize_surface_history(None)


def jpeg_history_comment(payload: Any) -> bytes:
    return JPEG_HISTORY_PREFIX + encode_surface_history(payload).encode("ascii")


def is_jpeg_history_comment(payload: bytes) -> bool:
    return bytes(payload).startswith(JPEG_HISTORY_PREFIX)


def _jpeg_comments(path: Path) -> tuple[bytes, ...]:
    try:
        data = path.read_bytes()
    except OSError:
        return ()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return ()
    out: list[bytes] = []
    pos = 2
    while pos + 1 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker == 0xDA:
            break
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(data):
            break
        seg_len = int.from_bytes(data[pos:pos + 2], "big")
        if seg_len < 2 or pos + seg_len > len(data):
            break
        payload = data[pos + 2:pos + seg_len]
        if marker == 0xFE:
            out.append(payload)
        pos += seg_len
    return tuple(out)


def read_surface_history(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return normalize_surface_history(None)
    source = Path(path)
    if not source.exists() or not source.is_file():
        return normalize_surface_history(None)
    suffix = source.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        for comment in reversed(_jpeg_comments(source)):
            if is_jpeg_history_comment(comment):
                return decode_surface_history(comment[len(JPEG_HISTORY_PREFIX):])
        return normalize_surface_history(None)
    if suffix == ".png":
        try:
            with Image.open(source) as image:
                text_map = getattr(image, "text", None)
                if isinstance(text_map, dict) and HISTORY_KEY in text_map:
                    return decode_surface_history(text_map.get(HISTORY_KEY))
        except OSError:
            pass
    return normalize_surface_history(None)
