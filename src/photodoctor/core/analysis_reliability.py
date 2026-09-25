from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Mapping

import numpy as np

from .models import MetricResult


@dataclass(frozen=True, slots=True)
class AnalysisReliabilityResult:
    status: str  # reliable | caution | unknown | ood_candidate
    calibrated_confidence: float
    support_score: float
    ood_score: float
    low_support_metrics: tuple[str, ...]
    unknown_components: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_raw(self) -> dict[str, object]:
        return {
            "status": self.status,
            "calibrated_confidence": self.calibrated_confidence,
            "support_score": self.support_score,
            "ood_score": self.ood_score,
            "low_support_metrics": list(self.low_support_metrics),
            "unknown_components": list(self.unknown_components),
            "reasons": list(self.reasons),
            "method": "heuristic_analysis_reliability_v1",
            "probability_interpretation": False,
            "note": "Heuristic support calibration; not a learned probability of correctness.",
        }


def _raw(metrics: Mapping[str, MetricResult], key: str) -> dict[str, object]:
    metric = metrics.get(key)
    return metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}


def _ratio(raw: dict[str, object], numerator: str, denominator: str) -> float | None:
    try:
        n = float(raw.get(numerator, 0.0) or 0.0)
        d = float(raw.get(denominator, 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if d <= 0:
        return None
    return float(np.clip(n / d, 0.0, 1.0))


def analyze_analysis_reliability(metrics: Mapping[str, MetricResult]) -> AnalysisReliabilityResult:
    """Estimate whether the current evidence supports confident automated action.

    This is deliberately *not* a learned OOD detector and its confidence is not a
    calibrated probability.  It aggregates measurement support and several
    conservative unknown/low-information guards so downstream code can avoid
    turning weak evidence into an automatic correction.
    """
    core_keys = (
        "brightness", "shadow_clipping", "highlight_clipping", "contrast",
        "local_contrast", "local_sharpness", "noise", "jpeg_artifacts",
        "edge_artifacts", "posterization", "detail_loss_type",
    )
    confidences: list[float] = []
    low_support: list[str] = []
    for key in core_keys:
        metric = metrics.get(key)
        if metric is None:
            low_support.append(key)
            continue
        conf = float(np.clip(metric.confidence, 0.0, 1.0))
        confidences.append(conf)
        if conf < 0.45:
            low_support.append(key)

    if confidences:
        support = float(np.clip(0.55 * median(confidences) + 0.45 * (sum(confidences) / len(confidences)), 0.0, 1.0))
    else:
        support = 0.0

    penalty = min(0.18, 0.03 * len(low_support))
    ood_score = 0.0
    reasons: list[str] = []
    unknown_components: list[str] = []

    sharp_raw = _raw(metrics, "local_sharpness")
    sharp_coverage = _ratio(sharp_raw, "informative_cells", "total_cells")
    if sharp_coverage is not None and sharp_coverage < 0.20:
        penalty += 0.08
        ood_score += 0.16
        reasons.append("low_informative_sharpness_coverage")

    contrast_raw = _raw(metrics, "local_contrast")
    contrast_coverage = _ratio(contrast_raw, "informative_cells", "total_cells")
    if contrast_coverage is not None and contrast_coverage < 0.20:
        penalty += 0.07
        ood_score += 0.14
        reasons.append("low_informative_contrast_coverage")

    noise_raw = _raw(metrics, "noise")
    noise_coverage = _ratio(noise_raw, "usable_tiles", "total_tiles")
    if noise_coverage is not None and noise_coverage < 0.15:
        penalty += 0.06
        ood_score += 0.10
        reasons.append("low_noise_measurement_support")

    nss_raw = _raw(metrics, "nss_baseline")
    nss_informative = bool(nss_raw.get("informative", False)) if nss_raw else False
    try:
        image_std = float(nss_raw.get("image_std", 0.0) or 0.0)
    except (TypeError, ValueError):
        image_std = 0.0
    if nss_raw and not nss_informative:
        penalty += 0.06
        ood_score += 0.10
        reasons.append("nss_not_informative")
    if nss_raw and image_std < 0.012:
        penalty += 0.10
        ood_score += 0.30
        reasons.append("extremely_low_visual_variance")
    elif nss_raw and image_std < 0.020:
        ood_score += 0.10
        reasons.append("low_visual_variance")

    detail_raw = _raw(metrics, "detail_loss_type")
    detail_kind = str(detail_raw.get("classification", "unknown"))
    if detail_kind == "unknown":
        penalty += 0.09
        unknown_components.append("detail_loss_type")
        reasons.append("detail_loss_unknown")
    elif detail_kind in {"mixed", "mixed_or_degraded"}:
        penalty += 0.04
        unknown_components.append("detail_loss_type")
        reasons.append("detail_loss_mixed")

    subject_raw = _raw(metrics, "main_subject")
    if subject_raw and str(subject_raw.get("subject_kind", "unknown")) == "unknown":
        penalty += 0.03
        unknown_components.append("main_subject")
        reasons.append("main_subject_unknown")

    semantic = metrics.get("semantic_context")
    if semantic is not None and semantic.confidence < 0.40:
        penalty += 0.04
        unknown_components.append("semantic_context")
        reasons.append("semantic_context_weak")

    profile_raw = _raw(metrics, "analysis_profile")
    try:
        tw = int(profile_raw.get("technical_width", 0) or 0)
        th = int(profile_raw.get("technical_height", 0) or 0)
    except (TypeError, ValueError):
        tw = th = 0
    short_edge = min(tw, th) if tw > 0 and th > 0 else 0
    if 0 < short_edge < 64:
        penalty += 0.20
        ood_score += 0.35
        reasons.append("very_small_input")
    elif 0 < short_edge < 128:
        penalty += 0.08
        ood_score += 0.15
        reasons.append("small_input")

    if (
        sharp_coverage is not None and sharp_coverage < 0.15
        and contrast_coverage is not None and contrast_coverage < 0.15
        and (not nss_informative or image_std < 0.02)
    ):
        ood_score += 0.18
        reasons.append("multiple_low_information_signals")

    calibrated = float(np.clip(support - penalty, 0.05, 0.95))
    ood_score = float(np.clip(ood_score, 0.0, 0.95))

    if ood_score >= 0.70:
        status = "ood_candidate"
    elif calibrated < 0.45 or len(unknown_components) >= 3:
        status = "unknown"
    elif (
        calibrated < 0.65
        or ood_score >= 0.40
        or ("detail_loss_type" in unknown_components and calibrated < 0.72)
    ):
        status = "caution"
    else:
        status = "reliable"

    if not reasons:
        reasons.append("measurement_support_consistent")

    return AnalysisReliabilityResult(
        status=status,
        calibrated_confidence=calibrated,
        support_score=support,
        ood_score=ood_score,
        low_support_metrics=tuple(low_support),
        unknown_components=tuple(unknown_components),
        reasons=tuple(reasons),
    )
