from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

from .blur_type import BlurTypeResult, analyze_blur_type
from .exif_context import ExifContext
from .local_sharpness import LocalSharpnessResult
from .main_subject import MainSubjectResult


@dataclass(frozen=True, slots=True)
class DetailLossContextResult:
    classification: str
    confidence: float
    raw_classification: str
    explanation_code: str
    subject_sharpness: float | None
    background_sharpness: float | None
    subject_background_delta: float | None
    subject_blur_classification: str | None
    subject_blur_confidence: float
    shutter_risk: str
    reasons: tuple[str, ...]


def _cell_center(cell: dict[str, float | str]) -> tuple[float, float]:
    return (
        float(cell.get("x", 0.0)) + float(cell.get("w", 0.0)) * 0.5,
        float(cell.get("y", 0.0)) + float(cell.get("h", 0.0)) * 0.5,
    )


def _in_box(cx: float, cy: float, box: dict[str, float]) -> bool:
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])
    return x <= cx <= x + w and y <= cy <= y + h


def _subject_background_sharpness(
    local: LocalSharpnessResult,
    subject: MainSubjectResult,
) -> tuple[float | None, float | None, float | None]:
    box = subject.protection_box_norm or subject.box_norm
    if box is None or subject.confidence < 0.42:
        return None, None, None
    subject_scores: list[float] = []
    background_scores: list[float] = []
    for cell in local.cells_norm:
        if str(cell.get("status", "")) == "low_texture":
            continue
        conf = float(cell.get("confidence", 0.0) or 0.0)
        if conf < 0.46:
            continue
        score = float(cell.get("score", 0.0) or 0.0)
        cx, cy = _cell_center(cell)
        if _in_box(cx, cy, box):
            subject_scores.append(score)
        else:
            background_scores.append(score)
    if len(subject_scores) < 2 or len(background_scores) < 2:
        return None, None, None
    s = float(median(subject_scores))
    b = float(median(background_scores))
    return s, b, b - s


def _subject_crop(rgb: np.ndarray, subject: MainSubjectResult) -> np.ndarray | None:
    box = subject.protection_box_norm or subject.box_norm
    if box is None or subject.confidence < 0.42:
        return None
    h, w = rgb.shape[:2]
    x0 = int(np.clip(round(float(box["x"]) * w), 0, max(w - 1, 0)))
    y0 = int(np.clip(round(float(box["y"]) * h), 0, max(h - 1, 0)))
    x1 = int(np.clip(round((float(box["x"]) + float(box["w"])) * w), x0 + 1, w))
    y1 = int(np.clip(round((float(box["y"]) + float(box["h"])) * h), y0 + 1, h))
    crop = rgb[y0:y1, x0:x1]
    if crop.shape[0] < 96 or crop.shape[1] < 96:
        return None
    return crop


def refine_detail_loss_type(
    rgb: np.ndarray,
    raw_blur: BlurTypeResult,
    local_sharpness: LocalSharpnessResult,
    subject: MainSubjectResult,
    exif: ExifContext,
    *,
    archival_likelihood: float,
    surface_candidate_count: int,
) -> DetailLossContextResult:
    """Refine raw blur cues with scene/subject/exposure context.

    The function is intentionally conservative.  A single image cannot prove
    physical camera motion versus subject motion, so specific labels require
    spatial or EXIF support and confidence is capped below certainty.
    """
    raw = raw_blur.classification
    reasons: list[str] = [f"raw:{raw}"]
    s_sharp, b_sharp, delta = _subject_background_sharpness(local_sharpness, subject)

    subject_blur: BlurTypeResult | None = None
    crop = _subject_crop(rgb, subject)
    if crop is not None:
        subject_blur = analyze_blur_type(crop)
        reasons.append(f"subject_raw:{subject_blur.classification}")
    subj_kind = subject_blur.classification if subject_blur is not None else None
    subj_conf = float(subject_blur.confidence if subject_blur is not None else 0.0)

    # Old prints/scans often combine texture loss, print grain, scratches and
    # processing history. Do not force those into a photographic blur kernel.
    archival_likelihood = float(np.clip(archival_likelihood, 0.0, 1.0))
    if archival_likelihood >= 0.50 and surface_candidate_count >= 8 and raw in {"mixed_or_degraded", "defocus_like"}:
        conf = float(np.clip(max(raw_blur.confidence, 0.46) + 0.12 * archival_likelihood, 0.0, 0.68))
        reasons.extend(("archival_context", "surface_ageing_candidates", f"archival_likelihood:{archival_likelihood:.3f}"))
        return DetailLossContextResult(
            "degradation_like", conf, raw, "archival_degradation_context",
            s_sharp, b_sharp, delta, subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
        )

    # If the primary subject is substantially softer than textured background,
    # classify the loss as local before making any global blur claim.
    subject_is_locally_soft = (
        delta is not None and delta >= 11.0 and b_sharp is not None and b_sharp >= 52.0
        and s_sharp is not None and s_sharp < 58.0
    )
    if subject_is_locally_soft:
        reasons.append("subject_softer_than_background")
        if subj_kind == "motion_like" and subj_conf >= 0.50:
            conf = float(np.clip(0.48 + 0.24 * subj_conf + min(delta or 0.0, 30.0) / 150.0, 0.0, 0.82))
            return DetailLossContextResult(
                "subject_motion_like", conf, raw, "localized_directional_subject_softness",
                s_sharp, b_sharp, delta, subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
            )
        conf = float(np.clip(0.46 + min(delta or 0.0, 28.0) / 110.0 + 0.10 * subject.confidence, 0.0, 0.76))
        return DetailLossContextResult(
            "local_subject_softness", conf, raw, "subject_background_sharpness_gap",
            s_sharp, b_sharp, delta, subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
        )

    if raw == "motion_like":
        # A high shutter-risk EXIF signal supports global camera shake, but does
        # not prove it. Require the raw image cue as well and cap confidence.
        if exif.shutter_risk == "high" and exif.confidence >= 0.40:
            reasons.extend(("global_directional_blur", "slow_shutter_support"))
            conf = float(np.clip(0.50 + 0.30 * raw_blur.confidence + 0.10 * exif.confidence, 0.0, 0.84))
            return DetailLossContextResult(
                "camera_shake_like", conf, raw, "directional_blur_with_shutter_risk",
                s_sharp, b_sharp, delta, subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
            )
        reasons.append("directional_blur_without_causal_proof")
        return DetailLossContextResult(
            "motion_like", float(np.clip(raw_blur.confidence, 0.0, 0.78)), raw,
            "directional_blur_unspecified_source", s_sharp, b_sharp, delta,
            subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
        )

    if raw == "defocus_like":
        reasons.append("global_isotropic_softening")
        return DetailLossContextResult(
            "defocus_like", float(np.clip(raw_blur.confidence, 0.0, 0.80)), raw,
            "global_isotropic_softening", s_sharp, b_sharp, delta,
            subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
        )

    if raw == "mixed_or_degraded":
        return DetailLossContextResult(
            "mixed", float(np.clip(raw_blur.confidence, 0.0, 0.56)), raw,
            "mixed_blur_signals", s_sharp, b_sharp, delta,
            subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
        )

    return DetailLossContextResult(
        "unknown", float(np.clip(raw_blur.confidence, 0.0, 0.42)), raw,
        "insufficient_evidence", s_sharp, b_sharp, delta,
        subj_kind, subj_conf, exif.shutter_risk, tuple(reasons),
    )
