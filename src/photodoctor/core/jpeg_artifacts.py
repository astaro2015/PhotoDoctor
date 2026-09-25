from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class JpegArtifactResult:
    score: float
    confidence: float
    classification: str
    block_ratio: float
    boundary_excess: float
    best_offset: int
    grid_dominance: float
    source_is_jpeg: bool


def _axis_phase_stats(diff: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary/ordinary means for all eight JPEG grid phases.

    The previous implementation boolean-indexed almost the whole derivative image
    sixteen times (boundary + ordinary for each phase).  Here each phase slice is
    visited once and the complementary mean is derived from the total sum.
    """
    length = diff.shape[axis]
    boundary_means = np.zeros(8, dtype=np.float64)
    ordinary_means = np.zeros(8, dtype=np.float64)
    total_count = int(diff.size)
    total_sum = float(np.sum(diff, dtype=np.float64))
    for offset in range(8):
        residue = (offset - 1) % 8
        phase = diff[:, residue::8] if axis == 1 else diff[residue::8, :]
        count = int(phase.size)
        other_count = total_count - count
        if count < 2 or other_count < 2:
            continue
        phase_sum = float(np.sum(phase, dtype=np.float64))
        boundary_means[offset] = phase_sum / count
        ordinary_means[offset] = (total_sum - phase_sum) / other_count
    return boundary_means, ordinary_means


def analyze_jpeg_artifacts(rgb: np.ndarray, source_format: str | None = None) -> JpegArtifactResult:
    """Estimate JPEG 8x8 blocking without assuming every JPEG is bad.

    The detector compares pixel discontinuities at candidate 8-pixel grid
    boundaries with ordinary neighbouring differences. Repeating scene geometry
    can mimic a grid, so confidence is intentionally limited unless the source is
    actually JPEG and the strongest grid aligns with the decoded 8x8 origin.
    """
    h, w = rgb.shape[:2]
    source_is_jpeg = str(source_format or "").upper() in {"JPEG", "JPG"}
    if h < 48 or w < 48:
        return JpegArtifactResult(100.0, 0.0, "unknown", 1.0, 0.0, 0, 1.0, source_is_jpeg)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))

    bx_all, ox_all = _axis_phase_stats(dx, 1)
    by_all, oy_all = _axis_phase_stats(dy, 0)
    candidates: list[tuple[float, float, int]] = []
    for offset in range(8):
        boundary_mean = 0.5 * (float(bx_all[offset]) + float(by_all[offset]))
        ordinary_mean = 0.5 * (float(ox_all[offset]) + float(oy_all[offset]))
        ratio = boundary_mean / max(ordinary_mean, 1e-3)
        excess = max(0.0, boundary_mean - ordinary_mean)
        candidates.append((float(ratio), float(excess), offset))

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_ratio, best_excess, best_offset = candidates[0]
    ratios = np.asarray([x[0] for x in candidates], dtype=np.float64)
    median_ratio = float(np.median(ratios))
    grid_dominance = float(best_ratio / max(median_ratio, 1e-3))

    ratio_severity = float(np.clip((best_ratio - 1.12) / 4.0, 0.0, 1.0))
    excess_severity = float(np.clip(best_excess / 3.0, 0.0, 1.0))
    severity = float(np.clip(0.72 * ratio_severity + 0.28 * excess_severity, 0.0, 1.0))
    score = float(100.0 * (1.0 - severity))

    if best_ratio < 1.30 or severity < 0.08:
        classification = "none"
    elif severity < 0.22:
        classification = "mild"
    elif severity < 0.48:
        classification = "moderate"
    else:
        classification = "strong"

    alignment_bonus = 0.12 if best_offset == 0 else 0.0
    source_bonus = 0.17 if source_is_jpeg else 0.0
    dominance_bonus = float(np.clip((grid_dominance - 1.0) * 0.35, 0.0, 0.16))
    confidence = float(np.clip(0.38 + source_bonus + alignment_bonus + dominance_bonus, 0.0, 0.84))
    if classification == "none":
        confidence = float(np.clip(confidence + 0.05, 0.0, 0.86))

    return JpegArtifactResult(
        score=score,
        confidence=confidence,
        classification=classification,
        block_ratio=best_ratio,
        boundary_excess=best_excess,
        best_offset=best_offset,
        grid_dominance=grid_dominance,
        source_is_jpeg=source_is_jpeg,
    )
