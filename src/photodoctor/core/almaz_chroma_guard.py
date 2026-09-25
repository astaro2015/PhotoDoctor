from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class AlmazChromaGuardResult:
    applied: bool
    archival_like: bool
    source_chroma_center: float
    source_chroma_spread_p95: float
    raw_local_chroma_p95: float
    raw_local_chroma_p99: float
    guarded_local_chroma_p95: float
    guarded_local_chroma_p99: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlmazDeblurChromaGuardResult:
    applied: bool
    raw_local_chroma_p95: float
    raw_local_chroma_p99: float
    guarded_local_chroma_p95: float
    guarded_local_chroma_p99: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lab_float(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)


def _baseline_for_shape(before: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    if before.shape == shape:
        return before
    h, w = shape[:2]
    return cv2.resize(before, (w, h), interpolation=cv2.INTER_CUBIC)


def _chroma_stats(rgb: np.ndarray) -> tuple[float, float, float, float]:
    lab = _lab_float(rgb)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    center_a = float(np.median(a))
    center_b = float(np.median(b))
    center = float(np.hypot(center_a, center_b))
    spread = np.hypot(a - center_a, b - center_b)
    p95 = float(np.percentile(spread, 95.0))
    p99 = float(np.percentile(spread, 99.0))
    return center, p95, p99, float(np.mean(spread))


def archival_chroma_profile(rgb: np.ndarray) -> tuple[bool, float, float]:
    """Return whether a source behaves like monochrome/sepia archival material.

    The detector is intentionally conservative.  It looks for a mostly coherent
    chroma field, not merely low global saturation, so warm sepia prints are
    accepted while ordinary colourful photographs are left untouched.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        return False, 999.0, 999.0
    center, p95, p99, _mean = _chroma_stats(rgb)
    # Neutral B/W: tiny centre chroma.  Sepia: a coherent warm cast may have a
    # larger centre, but local chroma variation still stays compact.
    archival_like = bool(
        p95 <= 10.0
        and p99 <= 18.0
        and center <= 42.0
    )
    return archival_like, center, p95


def _local_chroma_delta(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    ref_lab = _lab_float(reference)
    cand_lab = _lab_float(candidate)
    da = cand_lab[..., 1] - ref_lab[..., 1]
    db = cand_lab[..., 2] - ref_lab[..., 2]
    dist = np.hypot(da, db)
    return float(np.percentile(dist, 95.0)), float(np.percentile(dist, 99.0))


def guard_archival_chroma(
    before: np.ndarray,
    after: np.ndarray,
    *,
    retain_ai_chroma: float = 0.12,
    max_local_delta: float = 2.5,
) -> tuple[np.ndarray, AlmazChromaGuardResult]:
    """Keep ALMAZ luminance/detail while suppressing invented colour islands.

    For near-monochrome or sepia archival sources, the output L channel comes
    from ALMAZ, while a/b chroma remains anchored to the source.  A small bounded
    portion of the AI chroma change is retained so legitimate gentle cleanup is
    not completely discarded.
    """
    if before.ndim != 3 or before.shape[2] != 3 or before.dtype != np.uint8:
        raise ValueError("ALMAZ chroma guard expects uint8 RGB source.")
    if after.ndim != 3 or after.shape[2] != 3 or after.dtype != np.uint8:
        raise ValueError("ALMAZ chroma guard expects uint8 RGB result.")

    baseline = _baseline_for_shape(before, after.shape)
    archival_like, center, spread_p95 = archival_chroma_profile(before)
    raw_p95, raw_p99 = _local_chroma_delta(baseline, after)
    if not archival_like:
        result = AlmazChromaGuardResult(
            False, False, center, spread_p95,
            raw_p95, raw_p99, raw_p95, raw_p99,
            "ALMAZ chroma guard: обычное цветное фото, архивное ограничение не требуется.",
        )
        return after, result

    retain = float(np.clip(retain_ai_chroma, 0.0, 1.0))
    limit = float(max(0.0, max_local_delta))
    base_lab = _lab_float(baseline)
    ai_lab = _lab_float(after)

    delta_ab = ai_lab[..., 1:3] - base_lab[..., 1:3]
    mag = np.linalg.norm(delta_ab, axis=2, keepdims=True)
    if limit > 0.0:
        scale = np.minimum(1.0, limit / np.maximum(mag, 1e-6))
        delta_ab = delta_ab * scale
    else:
        delta_ab[:] = 0.0

    guarded_lab = ai_lab.copy()
    guarded_lab[..., 1:3] = base_lab[..., 1:3] + delta_ab * retain
    guarded_lab = np.clip(np.rint(guarded_lab), 0, 255).astype(np.uint8)
    guarded = cv2.cvtColor(guarded_lab, cv2.COLOR_LAB2RGB)

    guarded_p95, guarded_p99 = _local_chroma_delta(baseline, guarded)
    result = AlmazChromaGuardResult(
        True, True, center, spread_p95,
        raw_p95, raw_p99, guarded_p95, guarded_p99,
        (
            "ALMAZ chroma guard: архивное/сепийное фото; сохраняю исходную цветность, "
            f"локальный chroma p99 {raw_p99:.2f} -> {guarded_p99:.2f} Lab."
        ),
    )
    return guarded, result


def guard_deblur_chroma(
    before: np.ndarray,
    after: np.ndarray,
    *,
    retain_ai_chroma: float = 0.12,
    max_local_delta: float = 5.0,
    apply_threshold_p99: float = 2.5,
) -> tuple[np.ndarray, AlmazDeblurChromaGuardResult]:
    """Suppress colour hallucinations from domain-specific deblur models.

    NAFNet-GoPro is trained for GoPro-style motion blur and may create coloured
    islands on unrelated blur domains.  Deblurring is primarily a luminance /
    edge-restoration task, so we preserve the model's L channel while tightly
    anchoring a/b chroma to the input.  Small colour changes are left untouched
    to avoid needless Lab round-trips on already well-behaved outputs.
    """
    if before.ndim != 3 or before.shape[2] != 3 or before.dtype != np.uint8:
        raise ValueError("ALMAZ Deblur chroma guard expects uint8 RGB source.")
    if after.ndim != 3 or after.shape[2] != 3 or after.dtype != np.uint8:
        raise ValueError("ALMAZ Deblur chroma guard expects uint8 RGB result.")
    if before.shape != after.shape:
        raise ValueError("ALMAZ Deblur chroma guard expects x1 output with unchanged dimensions.")

    raw_p95, raw_p99 = _local_chroma_delta(before, after)
    threshold = float(max(0.0, apply_threshold_p99))
    if raw_p99 <= threshold:
        info = AlmazDeblurChromaGuardResult(
            False, raw_p95, raw_p99, raw_p95, raw_p99,
            f"ALMAZ Deblur chroma guard: local p99 {raw_p99:.2f} Lab, цветовая защита не требуется.",
        )
        return after, info

    retain = float(np.clip(retain_ai_chroma, 0.0, 1.0))
    limit = float(max(0.0, max_local_delta))
    base_lab = _lab_float(before)
    ai_lab = _lab_float(after)
    delta_ab = ai_lab[..., 1:3] - base_lab[..., 1:3]
    mag = np.linalg.norm(delta_ab, axis=2, keepdims=True)
    if limit > 0.0:
        delta_ab = delta_ab * np.minimum(1.0, limit / np.maximum(mag, 1e-6))
    else:
        delta_ab[:] = 0.0

    guarded_lab = ai_lab.copy()
    guarded_lab[..., 1:3] = base_lab[..., 1:3] + delta_ab * retain
    guarded_lab = np.clip(np.rint(guarded_lab), 0, 255).astype(np.uint8)
    guarded = cv2.cvtColor(guarded_lab, cv2.COLOR_LAB2RGB)
    guarded_p95, guarded_p99 = _local_chroma_delta(before, guarded)
    info = AlmazDeblurChromaGuardResult(
        True, raw_p95, raw_p99, guarded_p95, guarded_p99,
        (
            "ALMAZ Deblur chroma guard: подавляю паразитные цветовые островки; "
            f"local chroma p99 {raw_p99:.2f} -> {guarded_p99:.2f} Lab."
        ),
    )
    return guarded, info
