from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class EdgeArtifactResult:
    score: float
    confidence: float
    classification: str
    halo_likelihood: float
    ringing_likelihood: float
    near_edge_p90: float
    outer_ring_excess: float
    edge_coverage: float
    polarity_balance: float


@dataclass(frozen=True, slots=True)
class PosterizationResult:
    score: float
    confidence: float
    classification: str
    severity: float
    occupied_luma_levels: int
    occupied_channel_levels_median: float
    tonal_span: float
    plateau_ratio: float
    jump_ratio: float
    smooth_transition_samples: int


def _safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    if values.size == 0:
        return default
    return float(np.percentile(values, q))


def analyze_edge_artifacts(rgb: np.ndarray) -> EdgeArtifactResult:
    """Estimate halos, oversharpening and ringing around strong edges.

    The mathematics is unchanged, but large temporary maps are consumed in
    sequence.  Precise mode can feed 12--24 MP images here; keeping gray, Sobel
    maps, distance, three masks and three masked value copies alive together
    caused pathological allocator/memory stalls on otherwise ordinary photos.
    """
    h, w = rgb.shape[:2]
    if h < 64 or w < 64:
        return EdgeArtifactResult(100.0, 0.0, "unknown", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.9)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    del gx, gy, smooth

    threshold = max(float(np.percentile(grad, 90.0)), 0.12)
    edges = (grad >= threshold).astype(np.uint8)
    del grad
    edge_coverage = float(edges.mean())
    edge_pixels = int(edges.sum())
    if edge_pixels < 64:
        return EdgeArtifactResult(100.0, 0.12, "unknown", 0.0, 0.0, 0.0, 0.0, edge_coverage, 0.0)

    # Distance from the nearest strong edge. This lets us distinguish the first
    # halo band from oscillation that continues farther away (ringing).
    distance_input = 1 - edges
    del edges
    distance = cv2.distanceTransform(distance_input, cv2.DIST_L2, 3)
    del distance_input

    residual_blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    residual = gray - residual_blur
    del residual_blur, gray
    abs_residual = np.abs(residual)

    # Consume masks one at a time.  This preserves the exact pixel sets and
    # percentile formulas while avoiding three full-frame bool masks plus three
    # large masked copies being resident simultaneously.
    near = distance <= 2.0
    near_values = abs_residual[near]
    near_p90 = _safe_percentile(near_values, 90.0)
    del near_values
    near_residual = residual[near]
    if near_residual.size:
        pos = float(np.mean(near_residual > 0.03))
        neg = float(np.mean(near_residual < -0.03))
    else:
        pos = 0.0
        neg = 0.0
    del near_residual, near

    outer = (distance > 2.0) & (distance <= 5.0)
    outer_values = abs_residual[outer]
    outer_p90 = _safe_percentile(outer_values, 90.0)
    del outer_values, outer

    far = distance > 7.0
    far_values = abs_residual[far]
    far_p90 = _safe_percentile(far_values, 90.0)
    del far_values, far, distance, abs_residual, residual

    outer_excess = max(0.0, outer_p90 - far_p90)
    polarity_balance = float(2.0 * min(pos, neg) / max(pos + neg, 1e-6))

    # ~0.09 is a crisp but ordinary edge in our calibration scenes; sustained
    # values above ~0.13 increasingly resemble haloed/unsharp-mask output.
    halo_likelihood = float(np.clip((near_p90 - 0.095) / 0.105, 0.0, 1.0))
    halo_likelihood *= float(np.clip(0.45 + 0.55 * polarity_balance, 0.0, 1.0))

    # Ringing is specifically energy a few pixels away from the edge beyond the
    # ordinary image background. Keep the threshold above typical print grain.
    ringing_likelihood = float(np.clip((outer_excess - 0.007) / 0.022, 0.0, 1.0))

    severity = float(np.clip(max(halo_likelihood, 0.88 * ringing_likelihood), 0.0, 1.0))
    score = float(100.0 * (1.0 - severity))

    if severity < 0.12:
        classification = "none"
    elif ringing_likelihood >= 0.55 and ringing_likelihood >= halo_likelihood * 0.80:
        classification = "ringing_candidate"
    elif severity < 0.38:
        classification = "mild_halo_candidate"
    elif severity < 0.68:
        classification = "halo_candidate"
    else:
        classification = "strong_halo_candidate"

    # Enough edge support improves confidence. Dense line art lowers it because
    # graphics can imitate this signal without any sharpening artifact.
    support = float(np.clip(edge_coverage / 0.035, 0.0, 1.0))
    graphic_penalty = float(np.clip((edge_coverage - 0.16) / 0.18, 0.0, 0.32))
    confidence = float(np.clip(0.34 + 0.28 * support + 0.13 * polarity_balance - graphic_penalty, 0.0, 0.78))
    if classification == "none":
        confidence = float(np.clip(confidence + 0.05, 0.0, 0.82))

    return EdgeArtifactResult(
        score=score,
        confidence=confidence,
        classification=classification,
        halo_likelihood=halo_likelihood,
        ringing_likelihood=ringing_likelihood,
        near_edge_p90=near_p90,
        outer_ring_excess=outer_excess,
        edge_coverage=edge_coverage,
        polarity_balance=polarity_balance,
    )


def _occupied_levels(channel: np.ndarray, min_count: int) -> int:
    hist = np.bincount(channel.astype(np.uint8).ravel(), minlength=256)
    return int(np.count_nonzero(hist >= min_count))


def analyze_posterization(rgb: np.ndarray) -> PosterizationResult:
    """Estimate posterization/banding in otherwise smooth tonal transitions.

    Two clues are combined: suspiciously few occupied tone levels over a broad
    tonal span, and step/plateau behaviour where a low-frequency guide indicates
    that the image should change smoothly. Uniform walls are not enough by
    themselves to trigger a posterization warning.
    """
    h, w = rgb.shape[:2]
    if h < 48 or w < 48:
        return PosterizationResult(100.0, 0.0, "unknown", 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0)

    gray8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = gray8.astype(np.float32)
    p1, p99 = np.percentile(gray, [1.0, 99.0])
    tonal_span = float((p99 - p1) / 255.0)
    if tonal_span < 0.10:
        return PosterizationResult(100.0, 0.22, "unknown", 0.0, 1, 1.0, tonal_span, 0.0, 0.0, 0)

    min_count = max(4, int(gray.size * 0.00001))
    occupied_luma = _occupied_levels(gray8, min_count)
    channel_occupied = [_occupied_levels(rgb[..., c], min_count) for c in range(3)]
    occupied_channel_median = float(np.median(np.asarray(channel_occupied, dtype=np.float32)))
    effective_occupied = min(float(occupied_luma), occupied_channel_median)

    guide = cv2.GaussianBlur(gray, (0, 0), 2.0)
    raw_dx = np.abs(np.diff(gray, axis=1))
    raw_dy = np.abs(np.diff(gray, axis=0))
    guide_dx = np.abs(np.diff(guide, axis=1))
    guide_dy = np.abs(np.diff(guide, axis=0))

    # Keep horizontal and vertical transitions separate instead of concatenating
    # two full-frame derivative arrays plus a third boolean mask and a fourth
    # selected-values copy.  Counts are mathematically identical, but peak memory
    # is dramatically lower on 4K/24-MP precise analysis.
    smooth_x = (guide_dx >= 0.18) & (guide_dx <= 3.5)
    smooth_y = (guide_dy >= 0.18) & (guide_dy <= 3.5)
    smooth_samples_x = int(np.count_nonzero(smooth_x))
    smooth_samples_y = int(np.count_nonzero(smooth_y))
    smooth_samples = smooth_samples_x + smooth_samples_y

    if smooth_samples >= 256:
        plateau_count = int(np.count_nonzero((raw_dx < 0.5) & smooth_x))
        plateau_count += int(np.count_nonzero((raw_dy < 0.5) & smooth_y))
        jump_count = int(np.count_nonzero((raw_dx >= 3.0) & smooth_x))
        jump_count += int(np.count_nonzero((raw_dy >= 3.0) & smooth_y))
        plateau_ratio = float(plateau_count / smooth_samples)
        jump_ratio = float(jump_count / smooth_samples)
        banding_factor = float(np.clip(plateau_ratio * np.clip(jump_ratio * 8.0, 0.0, 1.0), 0.0, 1.0))
    else:
        plateau_ratio = 0.0
        jump_ratio = 0.0
        banding_factor = 0.0

    level_factor = float(np.clip((96.0 - effective_occupied) / 88.0, 0.0, 1.0))
    span_factor = float(np.clip((tonal_span - 0.16) / 0.58, 0.0, 1.0))
    occupancy_severity = level_factor * span_factor

    # Very coarse quantisation can create wide flat steps, so the smooth-guide
    # mask may miss the exact jump pixels. The occupied-level clue covers that
    # case; banding_factor improves confidence for subtler 32/64-level banding.
    severity = float(np.clip(0.66 * occupancy_severity + 0.34 * banding_factor, 0.0, 1.0))
    score = float(100.0 * (1.0 - severity))

    if severity < 0.12:
        classification = "none"
    elif severity < 0.30:
        classification = "mild"
    elif severity < 0.56:
        classification = "moderate"
    else:
        classification = "strong"

    transition_support = float(np.clip(smooth_samples / max(gray.size * 0.12, 1.0), 0.0, 1.0))
    level_evidence = float(np.clip((128.0 - effective_occupied) / 112.0, 0.0, 1.0))
    confidence = float(np.clip(0.30 + 0.24 * span_factor + 0.18 * transition_support + 0.12 * level_evidence, 0.0, 0.78))
    if classification == "none" and effective_occupied >= 150:
        confidence = float(np.clip(confidence + 0.08, 0.0, 0.82))

    return PosterizationResult(
        score=score,
        confidence=confidence,
        classification=classification,
        severity=severity,
        occupied_luma_levels=occupied_luma,
        occupied_channel_levels_median=occupied_channel_median,
        tonal_span=tonal_span,
        plateau_ratio=plateau_ratio,
        jump_ratio=jump_ratio,
        smooth_transition_samples=smooth_samples,
    )
