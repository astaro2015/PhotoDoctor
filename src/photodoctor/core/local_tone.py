from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .precision import get_precision, resolution_aware_positions
from .spatial_windows import adaptive_square_windows


@dataclass(frozen=True, slots=True)
class LocalToneResult:
    confidence: float
    dark_area_pct: float
    bright_area_pct: float
    deep_shadow_area_pct: float
    highlight_clip_area_pct: float
    total_cells: int
    cells_norm: list[dict[str, float | str]]
    map_window_px: int = 0
    map_step_px: int = 0
    map_target_positions: int = 0
    map_width: int = 0
    map_height: int = 0


def _status_area_percentages(
    cells: list[dict[str, float | str]], height: int, width: int, *, max_long_edge: int = 512
) -> dict[str, float]:
    """Estimate unique image-area coverage from overlapping categorical windows.

    Summing window areas directly overweights central pixels because overlapping
    windows cover them more often than border pixels.  Fractional per-pixel votes
    on a bounded working grid keep the reported percentages location-stable while
    preserving the window-based map itself.
    """
    statuses = ("deep_shadow", "dark", "bright", "clipped_highlight", "mid")
    if not cells or height <= 0 or width <= 0:
        return {key: 0.0 for key in statuses}
    scale = min(1.0, float(max_long_edge) / max(height, width))
    gh = max(32, int(round(height * scale)))
    gw = max(32, int(round(width * scale)))
    coverage = np.zeros((gh, gw), dtype=np.float32)
    votes = {key: np.zeros((gh, gw), dtype=np.float32) for key in statuses}
    for cell in cells:
        try:
            x = float(cell.get("x", 0.0)); y = float(cell.get("y", 0.0))
            cw = float(cell.get("w", 0.0)); ch = float(cell.get("h", 0.0))
        except (TypeError, ValueError):
            continue
        x0 = max(0, min(gw - 1, int(np.floor(x * gw))))
        y0 = max(0, min(gh - 1, int(np.floor(y * gh))))
        x1 = max(x0 + 1, min(gw, int(np.ceil((x + cw) * gw))))
        y1 = max(y0 + 1, min(gh, int(np.ceil((y + ch) * gh))))
        coverage[y0:y1, x0:x1] += 1.0
        status = str(cell.get("status", "mid"))
        if status in votes:
            votes[status][y0:y1, x0:x1] += 1.0
    active = coverage > 0.0
    if not np.any(active):
        return {key: 0.0 for key in statuses}
    out: dict[str, float] = {}
    for key in statuses:
        fraction = np.divide(votes[key], coverage, out=np.zeros_like(coverage), where=active)
        out[key] = 100.0 * float(np.mean(fraction[active]))
    return out


def analyze_local_tone(linear_rgb: np.ndarray, *, precision: str | None = None) -> LocalToneResult:
    """Describe local tonal distribution without declaring scene content a defect."""
    h, w = linear_rgb.shape[:2]
    if h < 32 or w < 32:
        return LocalToneResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, [])

    luma = (
        0.2126 * linear_rgb[..., 0]
        + 0.7152 * linear_rgb[..., 1]
        + 0.0722 * linear_rgb[..., 2]
    ).astype(np.float32)

    if precision is None:
        target_positions = 12
        overlap = 0.35
    else:
        profile = get_precision(precision)
        target_positions = resolution_aware_positions(
            min(h, w),
            profile.tone_base_positions,
            min_positions=profile.min_positions,
            max_positions=profile.max_positions,
        )
        overlap = profile.tone_overlap
    spec, windows = adaptive_square_windows(
        h, w, target_short_positions=target_positions, overlap=overlap, min_window=32, max_window=320
    )

    cells: list[dict[str, float | str]] = []

    for y0, y1, x0, x1 in windows:
        tile = luma[y0:y1, x0:x1]
        if tile.size < 64:
            continue
        p05, p25, p50, p75, p95 = np.percentile(tile, [5, 25, 50, 75, 95])
        mean = float(np.mean(tile))
        shadow_clip = float(np.mean(tile <= 0.0031308))
        highlight_clip = float(np.mean(tile >= 0.99))
        # A small black/white object must not paint the whole measurement window
        # as clipped.  Require percentile support when clipping fraction alone
        # triggers the extreme class; isolated speculars remain available to the
        # dedicated highlight detector through ``shadow_clip/highlight_clip``.
        if p50 < 0.055 or (shadow_clip >= 0.10 and p25 < 0.035):
            status = "deep_shadow"
        elif mean < 0.20:
            status = "dark"
        elif highlight_clip >= 0.08 and p75 > 0.92:
            # Bright is not the same thing as clipped.  A near-white but still
            # recoverable region (for example sRGB ~240) must remain ``bright``;
            # reserve clipped_highlight for windows with real near-1.0 support.
            status = "clipped_highlight"
        elif mean > 0.68:
            status = "bright"
        else:
            status = "mid"

        cells.append({
            "x": x0 / w,
            "y": y0 / h,
            "w": (x1 - x0) / w,
            "h": (y1 - y0) / h,
            "mean": mean,
            "p05": float(p05),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p95": float(p95),
            "shadow_clip": shadow_clip,
            "highlight_clip": highlight_clip,
            "status": status,
        })
    area_pct = _status_area_percentages(cells, h, w)
    total_cells = len(cells)
    confidence = float(np.clip(0.55 + min(total_cells / 60.0, 1.0) * 0.35, 0.0, 0.90)) if cells else 0.0
    return LocalToneResult(
        confidence=confidence,
        dark_area_pct=area_pct["dark"],
        bright_area_pct=area_pct["bright"],
        deep_shadow_area_pct=area_pct["deep_shadow"],
        highlight_clip_area_pct=area_pct["clipped_highlight"],
        total_cells=total_cells,
        cells_norm=cells,
        map_window_px=spec.window,
        map_step_px=spec.step,
        map_target_positions=target_positions,
        map_width=w,
        map_height=h,
    )
