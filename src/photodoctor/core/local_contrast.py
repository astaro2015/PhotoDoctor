from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .precision import get_precision, resolution_aware_positions
from .spatial_windows import adaptive_square_windows


@dataclass(frozen=True, slots=True)
class LocalContrastResult:
    score: float
    confidence: float
    low_contrast_area_pct: float
    median_local_range: float
    fading_likelihood: float
    classification: str
    informative_cells: int
    total_cells: int
    map_cells: int
    cells_norm: list[dict[str, float | str]]
    map_window_px: int = 0
    map_step_px: int = 0
    map_target_positions: int = 0
    map_width: int = 0
    map_height: int = 0


def _score_from_range(local_range: float) -> float:
    # p90-p10 around 0.10 is visibly flat; ~0.32 is already healthy local separation.
    return float(np.clip((local_range - 0.055) / 0.275 * 100.0, 0.0, 100.0))


def _percentiles_gray8(values: np.ndarray, qs: tuple[float, ...]) -> tuple[float, ...]:
    """Exact linear percentiles for an 8-bit gray tile, returned on the old 0..1 scale."""
    flat = np.asarray(values, dtype=np.uint8).ravel()
    n = int(flat.size)
    if n <= 0:
        return tuple(0.0 for _ in qs)
    cumulative = np.cumsum(np.bincount(flat, minlength=256), dtype=np.int64)
    levels = np.arange(256, dtype=np.float32) / np.float32(255.0)
    out: list[float] = []
    for q in qs:
        pos = float(np.clip(q, 0.0, 100.0)) * 0.01 * (n - 1)
        lo_i = int(np.floor(pos)); hi_i = int(np.ceil(pos)); frac = pos - lo_i
        lo_bin = int(np.searchsorted(cumulative, lo_i + 1, side="left"))
        hi_bin = int(np.searchsorted(cumulative, hi_i + 1, side="left"))
        lo_v = float(levels[lo_bin]); hi_v = float(levels[hi_bin])
        out.append(float(lo_v + (hi_v - lo_v) * frac))
    return tuple(out)


def _summary_windows(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Fixed, overlapping summary scale independent of UI precision.

    A hard non-overlapping grid made low-contrast area depend too strongly on
    whether a region happened to cross one of the grid boundaries.  Keep a
    fixed physical summary scale, but overlap measurements and later combine
    them as fractional votes over unique image area.
    """
    _, windows = adaptive_square_windows(
        height, width, target_short_positions=9, overlap=0.35, min_window=40, max_window=280
    )
    return windows


def _unique_summary_stats(
    measured_windows: list[tuple[int, int, int, int, dict[str, float | str]]],
    height: int,
    width: int,
    *,
    max_long_edge: int = 320,
) -> tuple[float, float, float]:
    """Return score/range/low-area without double-counting overlapping windows."""
    if not measured_windows or height <= 0 or width <= 0:
        return 0.0, 0.0, 0.0
    scale = min(1.0, float(max_long_edge) / max(height, width))
    gh = max(32, int(round(height * scale)))
    gw = max(32, int(round(width * scale)))
    coverage = np.zeros((gh, gw), dtype=np.float32)
    low_votes = np.zeros((gh, gw), dtype=np.float32)
    score_weight = np.zeros((gh, gw), dtype=np.float32)
    score_acc = np.zeros((gh, gw), dtype=np.float32)
    range_acc = np.zeros((gh, gw), dtype=np.float32)

    for y0, y1, x0, x1, measured in measured_windows:
        if str(measured.get("status", "")) == "low_texture":
            continue
        yy0 = max(0, min(gh - 1, int(np.floor(y0 / height * gh))))
        xx0 = max(0, min(gw - 1, int(np.floor(x0 / width * gw))))
        yy1 = max(yy0 + 1, min(gh, int(np.ceil(y1 / height * gh))))
        xx1 = max(xx0 + 1, min(gw, int(np.ceil(x1 / width * gw))))
        confidence = max(float(measured.get("confidence", 0.0)), 0.20)
        score = float(measured.get("score", 0.0))
        local_range = float(measured.get("range", 0.0))
        coverage[yy0:yy1, xx0:xx1] += 1.0
        score_weight[yy0:yy1, xx0:xx1] += confidence
        score_acc[yy0:yy1, xx0:xx1] += score * confidence
        range_acc[yy0:yy1, xx0:xx1] += local_range * confidence
        if str(measured.get("status", "")) == "low":
            low_votes[yy0:yy1, xx0:xx1] += 1.0

    active = score_weight > 0.0
    if not np.any(active):
        return 0.0, 0.0, 0.0
    score_map = np.divide(score_acc, score_weight, out=np.zeros_like(score_acc), where=active)
    range_map = np.divide(range_acc, score_weight, out=np.zeros_like(range_acc), where=active)
    low_fraction = np.divide(low_votes, coverage, out=np.zeros_like(low_votes), where=coverage > 0.0)
    return (
        float(np.mean(score_map[active])),
        float(np.median(range_map[active])),
        100.0 * float(np.mean(low_fraction[active])),
    )


def _measure_tile(gray8: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> dict[str, float | str] | None:
    tile8 = gray8[y0:y1, x0:x1]
    if tile8.size < 256:
        return None

    p10, p25, p50, p75, p90 = _percentiles_gray8(tile8, (10.0, 25.0, 50.0, 75.0, 90.0))
    local_range = float(p90 - p10)
    iqr = float(p75 - p25)
    gx = cv2.Sobel(tile8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tile8, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    edge_density = float(np.mean(grad >= 12.0))
    grad_mean = float(np.mean(grad))

    # Low-amplitude random grain used to satisfy the raw Sobel texture test and
    # could make a flat noisy image look like a faded/low-contrast photograph.
    # Structural edges persist after smoothing; stochastic grain largely vanishes.
    guide = cv2.GaussianBlur(tile8, (0, 0), sigmaX=1.0)
    sgx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    sgy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    structured_grad = cv2.magnitude(sgx, sgy)
    structured_edge_density = float(np.mean(structured_grad >= 12.0))
    structured_grad_mean = float(np.mean(structured_grad))
    structure_persistence = float(np.clip(structured_grad_mean / max(grad_mean, 1e-6), 0.0, 1.0))

    texture_signal = 0.58 * np.clip(structured_edge_density / 0.11, 0.0, 1.0) + 0.42 * np.clip(structured_grad_mean / 24.0, 0.0, 1.0)
    informative = bool((texture_signal >= 0.30 or local_range >= 0.14) and structure_persistence >= 0.52)
    structure_gate = float(np.clip((structure_persistence - 0.46) / 0.28, 0.0, 1.0))
    confidence = float(np.clip(0.16 + 0.78 * texture_signal * structure_gate, 0.0, 0.92))
    score = _score_from_range(local_range)

    if not informative:
        status = "low_texture"
    elif score < 38.0:
        status = "low"
    elif score < 64.0:
        status = "moderate"
    else:
        status = "good"

    return {
        "score": score,
        "range": local_range,
        "iqr": iqr,
        "confidence": confidence,
        "structure_persistence": structure_persistence,
        "status": status,
        "median": float(p50),
    }


def local_contrast_summary_score(rgb: np.ndarray) -> float:
    """Return the stable scalar local-contrast score without building the fine map.

    Validator uses only this scalar.  The detailed localization map is intentionally
    skipped here; summary windows and math are exactly the same as in
    ``analyze_local_contrast``.
    """
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return 0.0
    gray8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    measured_windows: list[tuple[int, int, int, int, dict[str, float | str]]] = []
    for y0, y1, x0, x1 in _summary_windows(h, w):
        measured = _measure_tile(gray8, y0, y1, x0, x1)
        if measured is not None:
            measured_windows.append((y0, y1, x0, x1, measured))
    score, _median_range, _low_pct = _unique_summary_stats(measured_windows, h, w)
    return score


def analyze_local_contrast(
    rgb: np.ndarray,
    *,
    map_rgb: np.ndarray | None = None,
    precision: str | None = None,
) -> LocalContrastResult:
    """Estimate perceived local tonal separation and possible fading.

    The summary uses a stable coarse scale for calibration.  The displayed map
    uses much finer overlapping windows so a face, sleeve and background are not
    forced into one giant diagnostic tile.  Smooth areas remain excluded from the
    fading decision at both scales.
    """
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return LocalContrastResult(0.0, 0.0, 0.0, 0.0, 0.0, "unknown", 0, 0, 0, [])

    gray8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Fine map for localization.  It may use a larger spatial copy and denser
    # precision profile without changing the stable scalar summary below.
    map_source = rgb if map_rgb is None else map_rgb
    map_h, map_w = map_source.shape[:2]
    map_gray8 = cv2.cvtColor(map_source, cv2.COLOR_RGB2GRAY)
    if map_rgb is None or precision is None:
        target_positions = 13
        overlap = 0.40
    else:
        profile = get_precision(precision)
        target_positions = resolution_aware_positions(
            min(map_h, map_w),
            profile.contrast_base_positions,
            min_positions=profile.min_positions,
            max_positions=profile.max_positions,
        )
        overlap = profile.contrast_overlap
    map_spec, map_windows = adaptive_square_windows(
        map_h, map_w, target_short_positions=target_positions, overlap=overlap, min_window=32, max_window=320
    )
    cells: list[dict[str, float | str]] = []
    for y0, y1, x0, x1 in map_windows:
        measured = _measure_tile(map_gray8, y0, y1, x0, x1)
        if measured is None:
            continue
        cells.append({
            "x": x0 / map_w,
            "y": y0 / map_h,
            "w": (x1 - x0) / map_w,
            "h": (y1 - y0) / map_h,
            **measured,
        })

    # Fixed summary scale for scalar score/fading.  It intentionally does not
    # depend on Fast/Normal/Precise, but overlapping windows prevent a region
    # from changing apparent area merely because it crossed a coarse grid line.
    summary_windows = _summary_windows(h, w)
    summary_measured: list[tuple[int, int, int, int, dict[str, float | str]]] = []
    summary_informative = 0
    summary_total = 0

    for y0, y1, x0, x1 in summary_windows:
        measured = _measure_tile(gray8, y0, y1, x0, x1)
        if measured is None:
            continue
        summary_total += 1
        summary_measured.append((y0, y1, x0, x1, measured))
        if measured["status"] != "low_texture":
            summary_informative += 1

    score, median_range, low_pct = _unique_summary_stats(summary_measured, h, w)

    coverage = summary_informative / max(summary_total, 1)
    confidence = float(np.clip(0.28 + 0.58 * coverage, 0.0, 0.86)) if summary_total else 0.0

    low_factor = np.clip((low_pct - 15.0) / 60.0, 0.0, 1.0)
    range_factor = np.clip((0.19 - median_range) / 0.15, 0.0, 1.0)
    fading_likelihood = float(np.clip(0.56 * low_factor + 0.44 * range_factor, 0.0, 1.0))
    if confidence < 0.42:
        classification = "unknown"
    elif fading_likelihood >= 0.62:
        classification = "fading_candidate"
    elif fading_likelihood >= 0.36:
        classification = "mixed"
    else:
        classification = "normal"

    return LocalContrastResult(
        score=score,
        confidence=confidence,
        low_contrast_area_pct=low_pct,
        median_local_range=median_range,
        fading_likelihood=fading_likelihood,
        classification=classification,
        informative_cells=summary_informative,
        total_cells=summary_total,
        map_cells=len(cells),
        cells_norm=cells,
        map_window_px=map_spec.window,
        map_step_px=map_spec.step,
        map_target_positions=target_positions,
        map_width=map_w,
        map_height=map_h,
    )
