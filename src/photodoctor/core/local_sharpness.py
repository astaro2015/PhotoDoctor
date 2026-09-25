from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from .precision import get_precision, resolution_aware_positions
from .spatial_windows import WindowSpec, adaptive_square_windows


@dataclass(frozen=True, slots=True)
class LocalSharpnessResult:
    score: float
    confidence: float
    soft_area_pct: float
    informative_cells: int
    total_cells: int
    cells_norm: list[dict[str, float | str]]
    map_window_px: int = 0
    map_step_px: int = 0
    map_target_positions: int = 0
    map_width: int = 0
    map_height: int = 0


def _clip100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _percentiles_u8(values: np.ndarray, qs: tuple[float, ...]) -> tuple[float, ...]:
    """Exact NumPy-linear percentiles for uint8 samples via a 256-bin histogram."""
    flat = np.asarray(values, dtype=np.uint8).ravel()
    n = int(flat.size)
    if n <= 0:
        return tuple(0.0 for _ in qs)
    cumulative = np.cumsum(np.bincount(flat, minlength=256), dtype=np.int64)
    out: list[float] = []
    for q in qs:
        pos = float(np.clip(q, 0.0, 100.0)) * 0.01 * (n - 1)
        lo_i = int(np.floor(pos)); hi_i = int(np.ceil(pos)); frac = pos - lo_i
        lo_v = int(np.searchsorted(cumulative, lo_i + 1, side="left"))
        hi_v = int(np.searchsorted(cumulative, hi_i + 1, side="left"))
        out.append(float(lo_v + (hi_v - lo_v) * frac))
    return tuple(out)


def _measure_windows(gray: np.ndarray, windows: list[tuple[int, int, int, int]]) -> tuple[list[dict[str, float | str]], list[float], list[float], float, float]:
    h, w = gray.shape[:2]
    cells: list[dict[str, float | str]] = []
    informative_scores: list[float] = []
    informative_weights: list[float] = []
    soft_weight = 0.0
    total_informative_weight = 0.0

    for y0, y1, x0, x1 in windows:
        tile = gray[y0:y1, x0:x1]
        if tile.size < 256:
            continue

        lap_var = float(cv2.Laplacian(tile, cv2.CV_64F).var())
        gx = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3)
        grad2 = gx * gx + gy * gy
        tenengrad = float(np.mean(grad2))
        grad = cv2.magnitude(gx, gy)
        edge_density = float(np.mean(grad >= 18.0))
        grad_mean = float(np.mean(grad))

        # Random sensor/JPEG grain can have enormous raw high-frequency energy and
        # previously made a smooth noisy patch look ``sharp``.  Real scene edges
        # survive a mild Gaussian guide much better than unstructured grain.  The
        # persistence ratio is therefore an independent structure gate, not a
        # replacement for the calibrated Laplacian/Tenengrad score.
        guide = cv2.GaussianBlur(tile, (0, 0), sigmaX=1.0)
        sgx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
        sgy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
        structured_grad_mean = float(np.mean(cv2.magnitude(sgx, sgy)))
        structure_persistence = float(np.clip(structured_grad_mean / max(grad_mean, 1e-6), 0.0, 1.0))

        lap_score = _clip100(20.0 * np.log10(1.0 + lap_var))
        ten_score = _clip100(14.0 * np.log10(1.0 + tenengrad))
        energy_score = float(0.45 * lap_score + 0.55 * ten_score)

        # Absolute edge energy alone is contrast-dependent: a crisp pale edge can
        # score below a blurred high-contrast edge.  Laplacian/Tenengrad ratio is
        # largely invariant to edge amplitude and tracks edge acutance instead.
        # Use it only when the tile has enough robust signal; the independent
        # structure-persistence gate below prevents random grain from exploiting
        # the ratio (noise has a very large ratio but poor persistence).
        p05, p95 = _percentiles_u8(tile, (5.0, 95.0))
        robust_range = float(p95 - p05)
        acutance_ratio = float(lap_var / max(tenengrad, 1e-6))
        if robust_range >= 10.0:
            log_lo = math.log10(0.003)
            log_hi = math.log10(0.080)
            acutance_score = _clip100(100.0 * (math.log10(max(acutance_ratio, 1e-6)) - log_lo) / (log_hi - log_lo))
            score = float(0.35 * energy_score + 0.65 * acutance_score)
        else:
            acutance_score = 0.0
            score = energy_score

        texture_signal = 0.62 * np.clip(edge_density / 0.16, 0.0, 1.0) + 0.38 * np.clip(grad_mean / 32.0, 0.0, 1.0)
        structure_gate = float(np.clip((structure_persistence - 0.42) / 0.28, 0.0, 1.0))
        confidence = float(np.clip(0.18 + 0.78 * texture_signal * structure_gate, 0.0, 0.94))
        informative = confidence >= 0.46 and structure_persistence >= 0.50

        if not informative:
            status = "low_texture"
        elif score < 45.0:
            status = "soft"
        elif score < 65.0:
            status = "medium"
        else:
            status = "sharp"

        area_weight = float((x1 - x0) * (y1 - y0))
        if informative:
            informative_scores.append(score)
            informative_weights.append(confidence * area_weight)
            total_informative_weight += area_weight
            if status == "soft":
                soft_weight += area_weight

        cells.append({
            "x": x0 / w,
            "y": y0 / h,
            "w": (x1 - x0) / w,
            "h": (y1 - y0) / h,
            "score": score,
            "confidence": confidence,
            "structure_persistence": structure_persistence,
            "robust_range": robust_range,
            "acutance_ratio": acutance_ratio,
            "acutance_score": acutance_score,
            "status": status,
        })
    return cells, informative_scores, informative_weights, soft_weight, total_informative_weight


def analyze_local_sharpness(
    rgb: np.ndarray,
    *,
    map_rgb: np.ndarray | None = None,
    precision: str | None = None,
) -> LocalSharpnessResult:
    """Estimate local high-frequency detail without letting map density recalibrate Quality.

    ``rgb`` is the stable technical image used for the scalar score and soft-area
    summary.  ``map_rgb`` may be a larger copy used only for localization.  This
    lets high-resolution originals get a denser map while keeping the calibrated
    numeric result comparable between precision modes.
    """
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return LocalSharpnessResult(0.0, 0.0, 0.0, 0, 0, [])

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, summary_windows = adaptive_square_windows(
        h, w, target_short_positions=15, overlap=0.50, min_window=40, max_window=280
    )
    summary_cells, informative_scores, informative_weights, soft_weight, total_informative_weight = _measure_windows(gray, summary_windows)

    if informative_scores:
        weights = np.asarray(informative_weights, dtype=np.float64)
        values = np.asarray(informative_scores, dtype=np.float64)
        score = float(np.average(values, weights=weights))
        soft_pct = float(100.0 * soft_weight / max(total_informative_weight, 1.0))
    else:
        score = 0.0
        soft_pct = 0.0
    summary_informative = sum(1 for cell in summary_cells if cell["status"] != "low_texture")
    coverage = summary_informative / max(len(summary_cells), 1)
    confidence = float(np.clip(0.25 + 0.65 * coverage, 0.0, 0.90)) if summary_cells else 0.0

    map_source = rgb if map_rgb is None else map_rgb
    mh, mw = map_source.shape[:2]
    if map_rgb is None or precision is None:
        target_positions = 15
        overlap = 0.50
    else:
        profile = get_precision(precision)
        target_positions = resolution_aware_positions(
            min(mh, mw),
            profile.sharp_base_positions,
            min_positions=profile.min_positions,
            max_positions=profile.max_positions,
        )
        overlap = profile.sharp_overlap
    map_gray = cv2.cvtColor(map_source, cv2.COLOR_RGB2GRAY)
    spec, map_windows = adaptive_square_windows(
        mh, mw, target_short_positions=target_positions, overlap=overlap, min_window=32, max_window=320
    )
    map_cells, _, _, _, _ = _measure_windows(map_gray, map_windows)

    return LocalSharpnessResult(
        score=score,
        confidence=confidence,
        soft_area_pct=soft_pct,
        informative_cells=summary_informative,
        total_cells=len(summary_cells),
        cells_norm=map_cells,
        map_window_px=spec.window,
        map_step_px=spec.step,
        map_target_positions=target_positions,
        map_width=mw,
        map_height=mh,
    )
