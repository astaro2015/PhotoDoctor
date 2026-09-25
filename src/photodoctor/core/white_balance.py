from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence, Mapping

import cv2
import numpy as np

from .auto_tone_color import AutoToneColorProfile, analyze_auto_tone_color
from .loader import srgb_to_linear
from .spatial_windows import adaptive_square_windows


MODEL_ID = "native_white_balance_advisor_v1"
FEATURE_SCHEMA = "white_balance_features_v1"


@dataclass(frozen=True, slots=True)
class WhiteBalanceAdvice:
    gain_r: float
    gain_g: float
    gain_b: float
    technical_temperature_shift_k: float
    recommended_temperature_shift_k: float
    technical_tint_shift: float
    recommended_tint_shift: float
    neutralization_strength: float
    atmosphere_preservation: float
    cast_strength: float
    cast_direction: str
    confidence: float
    estimator_agreement: float
    neutral_fraction: float
    face_guard: float
    correction_needed: bool
    model_id: str = MODEL_ID
    feature_schema: str = FEATURE_SCHEMA
    method: str = "hybrid_awb_feature_fusion_v1"
    advisor_kind: str = "hybrid_expert_baseline"
    neural_model_used: bool = False

    def to_raw(self) -> dict[str, object]:
        return asdict(self)



@dataclass(frozen=True, slots=True)
class SpatialWhiteBalanceResult:
    cells_norm: tuple[dict[str, float | str], ...]
    classification: str
    confidence: float
    mixed_light_score: float
    coverage: float
    supported_cells: int
    total_cells: int
    global_consistency: float
    neutral_anchor_cells: int = 0
    hybrid_anchor_cells: int = 0
    spatial_coherence: float = 0.0
    method: str = "neutral_overlap_spatial_wb_v1"

    def to_raw(self) -> dict[str, object]:
        return {
            "cells_norm": [dict(x) for x in self.cells_norm],
            "classification": self.classification,
            "confidence": self.confidence,
            "mixed_light_score": self.mixed_light_score,
            "coverage": self.coverage,
            "supported_cells": self.supported_cells,
            "total_cells": self.total_cells,
            "global_consistency": self.global_consistency,
            "neutral_anchor_cells": self.neutral_anchor_cells,
            "hybrid_anchor_cells": self.hybrid_anchor_cells,
            "spatial_coherence": self.spatial_coherence,
            "method": self.method,
        }


def advice_from_raw(raw: Mapping[str, object]) -> WhiteBalanceAdvice:
    """Rebuild a validated advice object from analyzer MetricResult.raw_value."""
    def f(name: str, default: float = 0.0) -> float:
        try:
            value = float(raw.get(name, default))
        except (TypeError, ValueError, OverflowError):
            value = float(default)
        return value if np.isfinite(value) else float(default)

    return WhiteBalanceAdvice(
        gain_r=f("gain_r", 1.0), gain_g=f("gain_g", 1.0), gain_b=f("gain_b", 1.0),
        technical_temperature_shift_k=f("technical_temperature_shift_k"),
        recommended_temperature_shift_k=f("recommended_temperature_shift_k"),
        technical_tint_shift=f("technical_tint_shift"),
        recommended_tint_shift=f("recommended_tint_shift"),
        neutralization_strength=float(np.clip(f("neutralization_strength"), 0.0, 1.0)),
        atmosphere_preservation=float(np.clip(f("atmosphere_preservation", 1.0), 0.0, 1.0)),
        cast_strength=float(np.clip(f("cast_strength"), 0.0, 1.0)),
        cast_direction=str(raw.get("cast_direction", "neutral") or "neutral"),
        confidence=float(np.clip(f("confidence", 0.35), 0.0, 1.0)),
        estimator_agreement=float(np.clip(f("estimator_agreement"), 0.0, 1.0)),
        neutral_fraction=float(np.clip(f("neutral_fraction"), 0.0, 1.0)),
        face_guard=float(np.clip(f("face_guard", 1.0), 0.0, 1.0)),
        correction_needed=bool(raw.get("correction_needed", False)),
        model_id=str(raw.get("model_id", MODEL_ID) or MODEL_ID),
        feature_schema=str(raw.get("feature_schema", FEATURE_SCHEMA) or FEATURE_SCHEMA),
        method=str(raw.get("method", "hybrid_awb_feature_fusion_v1") or "hybrid_awb_feature_fusion_v1"),
        advisor_kind=str(raw.get("advisor_kind", "hybrid_expert_baseline") or "hybrid_expert_baseline"),
        neural_model_used=bool(raw.get("neural_model_used", False)),
    )

def _safe_image(rgb: np.ndarray) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    return rgb


def _normalised_log_gains(illuminant: Sequence[float]) -> np.ndarray:
    illum = np.asarray(illuminant, dtype=np.float64)
    illum = np.clip(illum, 1e-6, None)
    log_gain = -np.log(illum)
    log_gain -= float(log_gain.mean())
    return np.clip(log_gain, -0.55, 0.55)


def _shades_of_gray(rgb: np.ndarray, p: float = 6.0) -> tuple[np.ndarray, float]:
    x = rgb.astype(np.float32) / 255.0
    mask = (x.max(axis=2) < 0.985) & (x.min(axis=2) > 0.018)
    values = x[mask]
    if values.shape[0] < 256:
        values = x.reshape(-1, 3)
    illum = np.power(np.mean(np.power(np.clip(values, 1e-6, 1.0), p), axis=0), 1.0 / p)
    return _normalised_log_gains(illum), min(1.0, values.shape[0] / max(rgb.shape[0] * rgb.shape[1] * 0.25, 1.0))


def _white_patch(rgb: np.ndarray, q: float = 0.97) -> tuple[np.ndarray, float]:
    x = rgb.astype(np.float32) / 255.0
    flat = x.reshape(-1, 3)
    illum = np.quantile(flat, q, axis=0)
    span = float(np.max(illum) - np.min(illum))
    reliability = float(np.clip(0.82 - max(0.0, float(np.max(illum)) - 0.99) * 3.0 - max(0.0, 0.15 - span) * 0.3, 0.45, 0.85))
    return _normalised_log_gains(illum), reliability


def _gray_edge(rgb: np.ndarray, p: float = 6.0) -> tuple[np.ndarray, float]:
    x = rgb.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(x, (0, 0), 1.0)
    illum: list[float] = []
    useful = 0
    for c in range(3):
        gx = cv2.Sobel(blur[..., c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur[..., c], cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        values = mag[(mag > 1e-4) & (x[..., c] < 0.985)]
        useful += int(values.size)
        if values.size < 128:
            illum.append(1.0)
        else:
            illum.append(float(np.power(np.mean(np.power(values, p)), 1.0 / p)))
    reliability = float(np.clip(useful / max(rgb.shape[0] * rgb.shape[1] * 0.18, 1.0), 0.25, 0.72))
    return _normalised_log_gains(illum), reliability


def _neutral_midtone_estimator(
    rgb: np.ndarray,
    profile: AutoToneColorProfile | None = None,
) -> tuple[np.ndarray | None, float, float]:
    # Analyzer already computes the same AutoToneColorProfile earlier in the
    # pipeline. Reuse it when supplied: this is strictly equivalent for the same
    # RGB input and avoids a second full-frame neutral/quantile pass late in
    # Precise analysis, where allocator/thread state can make the duplicate pass
    # pathologically slow on some images. Standalone callers keep the old path.
    profile = profile or analyze_auto_tone_color(rgb)
    med = np.asarray(
        [profile.neutral_median_r, profile.neutral_median_g, profile.neutral_median_b],
        dtype=np.float64,
    )
    if profile.neutral_fraction <= 0.0 or np.any(med <= 1.0):
        return None, 0.0, profile.neutral_fraction
    reliability = float(np.clip(profile.confidence * min(1.0, profile.neutral_fraction / 0.05), 0.15, 0.88))
    return _normalised_log_gains(med), reliability, profile.neutral_fraction


def _weighted_fusion(estimates: list[tuple[np.ndarray, float]]) -> tuple[np.ndarray, float]:
    if not estimates:
        return np.zeros(3, dtype=np.float64), 0.0
    weights = np.asarray([max(1e-6, float(weight)) for _value, weight in estimates], dtype=np.float64)
    values = np.vstack([value for value, _weight in estimates])

    # First weighted mean, then downweight estimators that disagree strongly with
    # the consensus. This keeps Gray-Edge useful without allowing one colourful
    # edge population to override mutually agreeing global/white-patch estimates.
    centre = np.average(values, axis=0, weights=weights)
    distances = np.sqrt(np.mean((values - centre[None, :]) ** 2, axis=1))
    robust_weights = weights * np.exp(-np.square(distances / 0.20))
    if float(robust_weights.sum()) <= 1e-9:
        robust_weights = weights
    centre = np.average(values, axis=0, weights=robust_weights)
    centre -= float(centre.mean())

    dispersion = float(np.sqrt(np.average(np.mean((values - centre[None, :]) ** 2, axis=1), weights=robust_weights)))
    agreement = float(np.clip(np.exp(-dispersion / 0.16), 0.0, 1.0))
    return np.clip(centre, -0.50, 0.50), agreement


def _face_guard_factor(
    rgb: np.ndarray,
    log_gains: np.ndarray,
    face_boxes: Iterable[tuple[int, int, int, int]] | None,
) -> float:
    if not face_boxes:
        return 1.0
    linear = srgb_to_linear(rgb).astype(np.float32)
    gains = np.exp(log_gains).astype(np.float32)
    guarded: list[float] = []
    h_img, w_img = rgb.shape[:2]
    for x, y, w, h in face_boxes:
        x = max(0, int(x)); y = max(0, int(y))
        w = min(int(w), w_img - x); h = min(int(h), h_img - y)
        if w < 12 or h < 12:
            continue
        roi = linear[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        before_y = 0.2126 * roi[..., 0] + 0.7152 * roi[..., 1] + 0.0722 * roi[..., 2]
        after = roi * gains[None, None, :]
        after_y = 0.2126 * after[..., 0] + 0.7152 * after[..., 1] + 0.0722 * after[..., 2]
        med_before = float(np.median(before_y))
        med_after = float(np.median(after_y))
        relative_luma = abs(med_after - med_before) / max(med_before, 0.04)
        clip_fraction = float(np.mean(np.max(after, axis=2) > 1.02))
        penalty = max(0.0, relative_luma - 0.14) * 1.8 + clip_fraction * 2.5

        # Chroma safety for skin: a WB shift is allowed to correct illumination,
        # but it should not push previously plausible skin far outside a broad
        # skin-colour envelope. This is a guard, not a beauty/skin-tone model.
        src_u8 = rgb[y:y+h, x:x+w]
        src_ycc = cv2.cvtColor(src_u8, cv2.COLOR_RGB2YCrCb)
        sy, scr, scb = cv2.split(src_ycc)
        skin = (sy >= 38) & (sy <= 248) & (scr >= 132) & (scr <= 182) & (scb >= 72) & (scb <= 138)
        if np.count_nonzero(skin) >= 32:
            corrected_u8 = _linear_to_srgb(np.clip(after, 0.0, 1.0))
            dst_ycc = cv2.cvtColor(corrected_u8, cv2.COLOR_RGB2YCrCb)
            _dy, dcr, dcb = cv2.split(dst_ycc)
            chroma_shift = np.sqrt(
                np.square(dcr.astype(np.float32) - scr.astype(np.float32))
                + np.square(dcb.astype(np.float32) - scb.astype(np.float32))
            )
            median_shift = float(np.median(chroma_shift[skin]))
            retained = float(np.mean((dcr[skin] >= 126) & (dcr[skin] <= 188) & (dcb[skin] >= 68) & (dcb[skin] <= 144)))
            penalty += max(0.0, median_shift - 10.0) / 24.0 + max(0.0, 0.70 - retained) * 0.9
        guarded.append(float(np.clip(1.0 - penalty, 0.52, 1.0)))
    return float(min(guarded)) if guarded else 1.0


def analyze_white_balance(
    rgb: np.ndarray,
    *,
    face_boxes: Iterable[tuple[int, int, int, int]] | None = None,
    tone_class: str = "color",
    archival_likelihood: float = 0.0,
    auto_tone_profile: AutoToneColorProfile | None = None,
) -> WhiteBalanceAdvice:
    """Estimate a conservative WB correction from multiple independent cues.

    This v1 advisor is intentionally *not* a neural network. It is the auditable
    expert baseline for the future WB AI model: its outputs are numeric parameters
    (channel gains, temperature/tint approximation and neutralisation strength),
    not generated pixels.  A trained model may later refine these parameters while
    the deterministic transform and safety checks stay unchanged.
    """
    _safe_image(rgb)
    tone_class = str(tone_class or "unknown")
    try:
        archival = float(archival_likelihood)
    except (TypeError, ValueError, OverflowError):
        archival = 0.0
    archival = float(np.clip(archival, 0.0, 1.0))

    estimates: list[tuple[np.ndarray, float]] = []
    sg, sg_conf = _shades_of_gray(rgb)
    estimates.append((sg, 0.40 * sg_conf))
    wp, wp_conf = _white_patch(rgb)
    estimates.append((wp, 0.34 * wp_conf))
    ge, ge_conf = _gray_edge(rgb)
    estimates.append((ge, 0.16 * ge_conf))
    neutral, neutral_conf, neutral_fraction = _neutral_midtone_estimator(rgb, auto_tone_profile)
    if neutral is not None:
        estimates.append((neutral, 0.20 * neutral_conf))

    log_gains, agreement = _weighted_fusion(estimates)
    # If a large part of the frame is independently identified as neutral
    # midtones, let that evidence dominate colourful-object area statistics.
    # The gate starts only above 20% neutral support so a small accidental
    # low-chroma patch cannot override three agreeing illuminant estimators.
    if neutral is not None and neutral_fraction >= 0.20:
        neutral_anchor_weight = float(np.clip((neutral_fraction - 0.20) / 0.35, 0.0, 0.88))
        log_gains = (1.0 - neutral_anchor_weight) * log_gains + neutral_anchor_weight * neutral
        log_gains -= float(np.mean(log_gains))
    gains = np.exp(log_gains)
    gains /= float(np.exp(np.mean(np.log(np.clip(gains, 1e-9, None)))))

    spread = float(np.max(log_gains) - np.min(log_gains))
    cast_strength = float(np.clip(spread / 0.70, 0.0, 1.0))
    face_guard = _face_guard_factor(rgb, log_gains, face_boxes)

    neutral_support = float(np.clip(neutral_fraction / 0.06, 0.0, 1.0))
    evidence = float(np.clip(0.16 + 0.54 * neutral_support + 0.30 * agreement, 0.0, 1.0))
    confidence = float(np.clip(0.20 + 0.38 * agreement + 0.42 * evidence, 0.30, 0.93))
    # A colourful scene can make Gray-World/White-Patch agree for the wrong reason.
    # Without neutral evidence, prefer abstention over confidently neutralising the
    # actual atmosphere of the scene.
    if neutral_fraction < 0.008:
        confidence = min(confidence, 0.58)
    elif neutral_fraction < 0.020:
        confidence = min(confidence, 0.68)

    technical_temp = float(np.clip(3500.0 * np.log(max(gains[0], 1e-6) / max(gains[2], 1e-6)), -3500.0, 3500.0))
    rb_mid = float(np.sqrt(max(gains[0] * gains[2], 1e-9)))
    technical_tint = float(np.clip(-80.0 * np.log(max(gains[1], 1e-6) / rb_mid), -30.0, 30.0))

    if tone_class in {"sepia", "monochrome"} or archival >= 0.78:
        neutralization = 0.0
        confidence = min(confidence, 0.52)
    else:
        cast_gate = float(np.clip((cast_strength - 0.045) / 0.40, 0.0, 1.0))
        neutralization = 0.42 + 0.28 * confidence + 0.18 * cast_gate
        neutralization = float(np.clip(neutralization * face_guard, 0.0, 0.86))
        if neutral_fraction < 0.008:
            neutralization = min(neutralization, 0.38)
        elif neutral_fraction < 0.020:
            neutralization = min(neutralization, 0.52)
        if cast_strength < 0.06:
            neutralization = min(neutralization, 0.35)

    rec_temp = technical_temp * neutralization
    rec_tint = technical_tint * neutralization
    correction_needed = bool(
        neutralization >= 0.35
        and confidence >= 0.52
        and (abs(rec_temp) >= 280.0 or abs(rec_tint) >= 2.5)
        and tone_class not in {"sepia", "monochrome"}
        and archival < 0.78
        and neutral_fraction >= 0.008
    )

    if rec_temp <= -450.0:
        cast_direction = "warm"
    elif rec_temp >= 450.0:
        cast_direction = "cool"
    elif rec_tint >= 4.0:
        cast_direction = "green"
    elif rec_tint <= -4.0:
        cast_direction = "magenta"
    else:
        cast_direction = "neutral"

    return WhiteBalanceAdvice(
        gain_r=float(gains[0]), gain_g=float(gains[1]), gain_b=float(gains[2]),
        technical_temperature_shift_k=technical_temp,
        recommended_temperature_shift_k=float(rec_temp),
        technical_tint_shift=technical_tint,
        recommended_tint_shift=float(rec_tint),
        neutralization_strength=neutralization,
        atmosphere_preservation=float(np.clip(1.0 - neutralization, 0.0, 1.0)),
        cast_strength=cast_strength,
        cast_direction=cast_direction,
        confidence=confidence,
        estimator_agreement=agreement,
        neutral_fraction=float(neutral_fraction),
        face_guard=face_guard,
        correction_needed=correction_needed,
    )


def _local_neutral_estimate(tile_rgb: np.ndarray) -> tuple[np.ndarray | None, float, float, str]:
    """Estimate a local illuminant from conservative anchors.

    First preference is genuinely low-chroma midtones. If a strong cast makes
    those disappear, a bounded multi-estimator fallback is allowed only in a
    structured, internally varied window where independent estimators agree.
    Uniform coloured walls therefore do not become pretend gray cards.
    """
    if tile_rgb.size == 0:
        return None, 0.0, 0.0, "none"
    linear = srgb_to_linear(tile_rgb).astype(np.float32)
    luma = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    vmax = np.max(linear, axis=2)
    vmin = np.min(linear, axis=2)
    mean = np.mean(linear, axis=2)
    relative_chroma = (vmax - vmin) / np.maximum(mean, 0.035)
    usable = (luma >= 0.055) & (luma <= 0.88)
    neutral = usable & (relative_chroma <= 0.145)
    count = int(np.count_nonzero(neutral))
    area = int(tile_rgb.shape[0] * tile_rgb.shape[1])
    neutral_fraction = float(count / max(area, 1))
    if count >= max(48, int(round(area * 0.008))):
        values = linear[neutral]
        illum = np.median(values, axis=0).astype(np.float64)
        if np.all(illum > 1e-5):
            log_gains = _normalised_log_gains(illum)
            chroma_quality = float(np.clip(1.0 - np.median(relative_chroma[neutral]) / 0.145, 0.0, 1.0))
            support = float(np.clip((neutral_fraction - 0.008) / 0.070, 0.0, 1.0))
            luma_span = float(np.quantile(luma[neutral], 0.85) - np.quantile(luma[neutral], 0.15)) if count >= 64 else 0.0
            range_quality = float(np.clip(0.45 + luma_span / 0.30, 0.45, 1.0))
            confidence = float(np.clip(0.18 + 0.52 * support + 0.20 * chroma_quality + 0.10 * range_quality, 0.0, 0.94))
            return log_gains, confidence, neutral_fraction, "neutral"

    # Strong colour casts can move a true gray surface outside the low-chroma
    # threshold. Allow a fallback only if the tile has real scene structure and
    # three classical estimators broadly agree. This is intentionally weaker.
    gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    guide = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    luma_std = float(np.std(luma[usable])) if np.any(usable) else 0.0
    edge_fraction = float(np.mean(grad > 5.0))
    if luma_std < 0.028 or edge_fraction < 0.012:
        return None, 0.0, neutral_fraction, "none"
    sg, sg_conf = _shades_of_gray(tile_rgb)
    wp, wp_conf = _white_patch(tile_rgb)
    ge, ge_conf = _gray_edge(tile_rgb)
    log_gains, agreement = _weighted_fusion([(sg, 0.45 * sg_conf), (wp, 0.35 * wp_conf), (ge, 0.20 * ge_conf)])
    if agreement < 0.56:
        return None, 0.0, neutral_fraction, "none"
    structure = float(np.clip((luma_std - 0.028) / 0.12, 0.0, 1.0))
    edge_support = float(np.clip((edge_fraction - 0.012) / 0.10, 0.0, 1.0))
    evidence = float(np.clip(0.5 * structure + 0.5 * edge_support, 0.0, 1.0))
    confidence = float(np.clip(0.28 + 0.34 * agreement + 0.20 * evidence, 0.0, 0.72))
    if confidence < 0.50:
        return None, 0.0, neutral_fraction, "none"
    support_fraction = max(neutral_fraction, 0.010 + 0.040 * evidence)
    return log_gains, confidence, float(support_fraction), "hybrid"


def _log_gains_to_temp_tint(log_gains: np.ndarray) -> tuple[float, float]:
    gains = np.exp(np.asarray(log_gains, dtype=np.float64))
    gains /= float(np.exp(np.mean(np.log(np.clip(gains, 1e-9, None)))))
    temp = float(np.clip(3500.0 * np.log(max(gains[0], 1e-6) / max(gains[2], 1e-6)), -3500.0, 3500.0))
    rb_mid = float(np.sqrt(max(gains[0] * gains[2], 1e-9)))
    tint = float(np.clip(-80.0 * np.log(max(gains[1], 1e-6) / rb_mid), -30.0, 30.0))
    return temp, tint


def _spatial_gain_coherence(cells: Sequence[Mapping[str, object]]) -> float:
    """How much of local WB variation follows a coherent 2D illumination field.

    A coloured object can bias several classical AWB estimators, but those biases
    usually do not form a smooth scene-wide field. Real mixed illumination often
    does.  Fit a weighted colour plane over cell centres and report explained
    variance. This deliberately favours precision over recalling every tiny local
    lamp pool; weak/nonlinear cases remain Review/uncertain instead of inventing
    multiple light sources from object colours.
    """
    if len(cells) < 4:
        return 0.0
    xy=[]; vv=[]; ww=[]
    for cell in cells:
        try:
            x=float(cell.get("x",0.0))+0.5*float(cell.get("w",0.0))
            y=float(cell.get("y",0.0))+0.5*float(cell.get("h",0.0))
            temp=float(cell.get("temperature_shift_k",0.0))/1800.0
            tint=float(cell.get("tint_shift",0.0))/12.0
            conf=float(cell.get("confidence",0.0))
            nf=float(cell.get("neutral_fraction",0.0))
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(np.isfinite(v) for v in (x,y,temp,tint,conf,nf)):
            continue
        xy.append((x-0.5,y-0.5)); vv.append((temp,tint)); ww.append(max(1e-4,conf*min(1.0,nf/0.05)))
    if len(xy) < 4:
        return 0.0
    xy_arr=np.asarray(xy,dtype=np.float64)
    values=np.asarray(vv,dtype=np.float64)
    weights=np.asarray(ww,dtype=np.float64)
    centre=np.average(values,axis=0,weights=weights)
    total=float(np.sum(weights[:,None]*np.square(values-centre[None,:])))
    if total <= 1e-9:
        return 0.0
    design=np.column_stack([np.ones(len(xy_arr)),xy_arr])
    sw=np.sqrt(weights)[:,None]
    try:
        coeff=np.linalg.lstsq(design*sw,values*sw,rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    pred=design@coeff
    residual=float(np.sum(weights[:,None]*np.square(values-pred)))
    return float(np.clip(1.0-residual/total,0.0,1.0))


def analyze_spatial_white_balance(
    rgb: np.ndarray,
    *,
    tone_class: str = "color",
    archival_likelihood: float = 0.0,
    max_long_edge: int = 960,
) -> SpatialWhiteBalanceResult:
    """Build a sparse map of trustworthy local illuminant estimates.

    Only windows with real neutral evidence are admitted. This makes the map
    intentionally incomplete on colourful scenes and allows mixed-light detection
    without independently white-balancing every tile.
    """
    _safe_image(rgb)
    tone_class = str(tone_class or "unknown")
    try:
        archival = float(np.clip(float(archival_likelihood), 0.0, 1.0))
    except (TypeError, ValueError, OverflowError):
        archival = 0.0
    if tone_class in {"sepia", "monochrome"} or archival >= 0.78:
        return SpatialWhiteBalanceResult((), "protected", 0.0, 0.0, 0.0, 0, 0, 1.0)

    h0, w0 = rgb.shape[:2]
    scale = min(1.0, float(max_long_edge) / max(h0, w0))
    if scale < 1.0:
        work = cv2.resize(rgb, (max(1, round(w0 * scale)), max(1, round(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        work = rgb
    h, w = work.shape[:2]
    _spec, windows = adaptive_square_windows(h, w, target_short_positions=6, overlap=0.50, min_window=56, max_window=220)
    cells: list[dict[str, float | str]] = []
    vectors: list[np.ndarray] = []
    weights: list[float] = []
    for y0, y1, x0, x1 in windows:
        log_gains, confidence, neutral_fraction, anchor_kind = _local_neutral_estimate(work[y0:y1, x0:x1])
        if log_gains is None or confidence < 0.42:
            continue
        temp, tint = _log_gains_to_temp_tint(log_gains)
        gains = np.exp(log_gains)
        gains /= float(np.exp(np.mean(np.log(np.clip(gains, 1e-9, None)))))
        spread = float(np.max(log_gains) - np.min(log_gains))
        cast_strength = float(np.clip(spread / 0.70, 0.0, 1.0))
        cells.append({
            "x": float(x0 / w), "y": float(y0 / h), "w": float((x1 - x0) / w), "h": float((y1 - y0) / h),
            "gain_r": float(gains[0]), "gain_g": float(gains[1]), "gain_b": float(gains[2]),
            "log_gain_r": float(log_gains[0]), "log_gain_g": float(log_gains[1]), "log_gain_b": float(log_gains[2]),
            "temperature_shift_k": temp, "tint_shift": tint, "cast_strength": cast_strength,
            "neutral_fraction": float(neutral_fraction), "confidence": float(confidence), "anchor_kind": anchor_kind,
        })
        vectors.append(np.asarray([temp / 1800.0, tint / 12.0], dtype=np.float64))
        weights.append(max(1e-4, confidence * min(1.0, neutral_fraction / 0.05)))

    total = len(windows)
    supported = len(cells)
    if supported == 0:
        return SpatialWhiteBalanceResult((), "uncertain", 0.0, 0.0, 0.0, 0, total, 0.0)

    # Unique support coverage, not a sum of overlapping window areas.
    mask = np.zeros((128, 128), dtype=np.uint8)
    for cell in cells:
        x0 = max(0, min(127, int(np.floor(float(cell["x"]) * 128))))
        y0 = max(0, min(127, int(np.floor(float(cell["y"]) * 128))))
        x1 = max(x0 + 1, min(128, int(np.ceil((float(cell["x"]) + float(cell["w"])) * 128))))
        y1 = max(y0 + 1, min(128, int(np.ceil((float(cell["y"]) + float(cell["h"])) * 128))))
        mask[y0:y1, x0:x1] = 1
    coverage = float(np.mean(mask))

    arr = np.vstack(vectors)
    ww = np.asarray(weights, dtype=np.float64)
    centre = np.average(arr, axis=0, weights=ww)
    dispersion = float(np.sqrt(np.average(np.sum(np.square(arr - centre[None, :]), axis=1), weights=ww)))
    temps = np.asarray([float(c["temperature_shift_k"]) for c in cells], dtype=np.float64)
    tints = np.asarray([float(c["tint_shift"]) for c in cells], dtype=np.float64)
    strong = np.asarray([float(c["confidence"]) >= 0.52 for c in cells], dtype=bool)
    opposite_temp = bool(np.any((temps <= -550.0) & strong) and np.any((temps >= 550.0) & strong))
    opposite_tint = bool(np.any((tints <= -6.0) & strong) and np.any((tints >= 6.0) & strong))
    raw_mixed_score = float(np.clip(max(dispersion / 1.15, 0.78 if opposite_temp else 0.0, 0.72 if opposite_tint else 0.0), 0.0, 1.0))
    spatial_coherence = _spatial_gain_coherence(cells)
    # Object colours can make local AWB estimates disagree even under one lamp.
    # Genuine mixed light must also form a spatially coherent field.
    coherence_factor = 0.28 + 0.72 * float(np.sqrt(spatial_coherence))
    mixed_score = float(np.clip(raw_mixed_score * coherence_factor, 0.0, 1.0))
    consistency = float(np.clip(np.exp(-dispersion / 0.72), 0.0, 1.0))
    support_ratio = supported / max(total, 1)
    mean_conf = float(np.average(np.asarray([float(c["confidence"]) for c in cells]), weights=ww))
    confidence = float(np.clip(mean_conf * min(1.0, support_ratio / 0.28) * (0.72 + 0.28 * min(1.0, coverage / 0.30)), 0.0, 0.93))
    neutral_anchor_cells = sum(1 for c in cells if str(c.get("anchor_kind", "")) == "neutral")
    hybrid_anchor_cells = supported - neutral_anchor_cells
    extreme_fraction = float(np.mean((np.abs(temps) >= 3000.0) | (np.abs(tints) >= 25.0)))
    hybrid_only_suspicious = neutral_anchor_cells == 0 and extreme_fraction >= 0.25
    if hybrid_only_suspicious:
        confidence = min(confidence, 0.46)
        mixed_score = min(mixed_score, 0.49)

    if supported < 3 or coverage < 0.10 or confidence < 0.40:
        classification = "uncertain"
    elif hybrid_only_suspicious:
        classification = "uncertain"
    elif neutral_anchor_cells == 0:
        # Hybrid-only evidence is useful under strong casts, but colourful object
        # layouts can imitate multiple illuminants. Demand a much stronger spatial
        # field before naming mixed/local light without a single true neutral anchor.
        if mixed_score >= 0.62 and spatial_coherence >= 0.58:
            classification = "mixed_light"
        elif mixed_score >= 0.45 and spatial_coherence >= 0.55:
            classification = "local_cast"
        else:
            classification = "uniform"
    elif mixed_score >= 0.55 and spatial_coherence >= 0.20:
        classification = "mixed_light"
    elif raw_mixed_score >= 0.18 or (mixed_score >= 0.34 and spatial_coherence >= 0.12):
        classification = "local_cast"
    else:
        classification = "uniform"
    return SpatialWhiteBalanceResult(
        tuple(cells), classification, confidence, mixed_score, coverage, supported, total, consistency,
        neutral_anchor_cells, hybrid_anchor_cells, spatial_coherence,
    )


def _wb_skin_residual_guard(rgb: np.ndarray, face_regions: Sequence[Mapping[str, object]] | None) -> np.ndarray:
    h, w = rgb.shape[:2]
    guard = np.ones((h, w), dtype=np.float32)
    if not face_regions:
        return guard
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    yy, cr, cb = cv2.split(ycc)
    skin = (yy >= 38) & (yy <= 248) & (cr >= 132) & (cr <= 182) & (cb >= 72) & (cb <= 138)
    face_mask = np.zeros((h, w), dtype=np.float32)
    for region in face_regions:
        try:
            x=float(region.get("x",0)); y=float(region.get("y",0)); rw=float(region.get("w",0)); rh=float(region.get("h",0))
        except (TypeError, ValueError, OverflowError):
            continue
        x0=max(0,min(w,int(round(x*w)))); y0=max(0,min(h,int(round(y*h))))
        x1=max(0,min(w,int(round((x+rw)*w)))); y1=max(0,min(h,int(round((y+rh)*h))))
        if x1>x0 and y1>y0:
            face_mask[y0:y1,x0:x1]=1.0
    if np.any(face_mask):
        feather=max(1.0,min(h,w)/320.0)
        face_mask=cv2.GaussianBlur(face_mask,(0,0),sigmaX=feather,sigmaY=feather)
        # The local residual must be conservative over the whole confirmed face,
        # even when a strong colour cast moves real skin outside classical YCrCb
        # thresholds.  Global/base WB still applies normally; only spatial
        # variation is damped here.
        guard *= 1.0 - 0.22 * np.clip(face_mask,0.0,1.0)
        protected=np.clip(face_mask * skin.astype(np.float32),0.0,1.0)
        guard *= 1.0 - 0.30 * protected
    return np.clip(guard,0.45,1.0)


def apply_spatial_white_balance(rgb: np.ndarray, plan: Mapping[str, object], *, strength: float = 1.0) -> np.ndarray:
    """Apply a global WB base plus smooth local residuals from trustworthy cells."""
    _safe_image(rgb)
    strength=float(np.clip(float(strength),0.0,1.0))
    if strength <= 0.0:
        return rgb.copy()
    cells=plan.get("white_balance_cells") if isinstance(plan, Mapping) else None
    global_raw=plan.get("global_advice") if isinstance(plan, Mapping) else None
    if not isinstance(cells,(list,tuple)) or not cells:
        if isinstance(global_raw, Mapping):
            return apply_white_balance(rgb, advice_from_raw(global_raw), strength=strength)
        return rgb.copy()
    if isinstance(global_raw, Mapping):
        advice=advice_from_raw(global_raw)
        global_log=np.log(np.clip(np.asarray([advice.gain_r, advice.gain_g, advice.gain_b],dtype=np.float32),0.60,1.80))
        base_strength=float(np.clip(float(plan.get("white_balance_base_strength", min(advice.neutralization_strength,0.55))),0.0,0.70))
    else:
        global_log=np.zeros(3,dtype=np.float32); base_strength=0.0

    # Local targets are already conservative; the UI strength scales the complete
    # chromatic correction, not the map support.
    from .local_correction_planner import rasterize_plan_fields
    h,w=rgb.shape[:2]
    fields, support = rasterize_plan_fields(
        cells, h, w, value_keys=("target_log_r", "target_log_g", "target_log_b")
    )
    local=np.stack(fields,axis=2)
    base=(global_log * base_strength)[None,None,:]
    residual=local-base
    skin_guard=_wb_skin_residual_guard(rgb, plan.get("face_regions") if isinstance(plan, Mapping) else None)
    eye_regions=plan.get("eye_regions") if isinstance(plan, Mapping) else None
    if isinstance(eye_regions,(list,tuple)) and eye_regions:
        eye_guard=np.ones((h,w),dtype=np.float32)
        for region in eye_regions:
            try:
                x=float(region.get("x",0)); y=float(region.get("y",0)); rw=float(region.get("w",0)); rh=float(region.get("h",0))
            except (TypeError, ValueError, OverflowError):
                continue
            x0=max(0,min(w,int(round(x*w)))); y0=max(0,min(h,int(round(y*h))))
            x1=max(0,min(w,int(round((x+rw)*w)))); y1=max(0,min(h,int(round((y+rh)*h))))
            if x1>x0 and y1>y0: eye_guard[y0:y1,x0:x1]=0.45
        eye_guard=cv2.GaussianBlur(eye_guard,(0,0),sigmaX=max(1.0,min(h,w)/500.0))
        skin_guard*=eye_guard
    log_field=base + residual * support[...,None] * skin_guard[...,None]
    log_field*=strength
    gains=np.exp(np.clip(log_field,-0.42,0.42)).astype(np.float32)
    linear=srgb_to_linear(rgb).astype(np.float32)
    corrected=linear*gains
    before_y=0.2126*linear[...,0]+0.7152*linear[...,1]+0.0722*linear[...,2]
    after_y=0.2126*corrected[...,0]+0.7152*corrected[...,1]+0.0722*corrected[...,2]
    # Per-pixel chromatic normalization prevents the local WB field from acting as
    # a local exposure brush. Keep the scalar bounded near highlights/shadows.
    lum_scale=np.clip(before_y/np.maximum(after_y,1e-5),0.90,1.10)
    corrected*=lum_scale[...,None]
    over=corrected>1.0
    if np.any(over):
        extra=corrected[over]-1.0
        corrected[over]=1.0+extra/(1.0+4.0*extra)
    return _linear_to_srgb(np.clip(corrected,0.0,1.0))


def spatial_neutral_error(rgb: np.ndarray, cells: Sequence[Mapping[str, object]]) -> float:
    """Robust neutral chroma error over the exact windows used by spatial WB."""
    h,w=rgb.shape[:2]
    errors=[]; weights=[]
    linear=srgb_to_linear(rgb).astype(np.float32)
    # These per-pixel quantities do not depend on the window. Spatial WB cells
    # overlap heavily, so recomputing them inside every ROI repeated the same
    # arithmetic dozens of times. Precompute once and slice the exact same maps.
    lum_full=0.2126*linear[...,0]+0.7152*linear[...,1]+0.0722*linear[...,2]
    vmax_full=linear.max(axis=2); vmin_full=linear.min(axis=2); mean_full=linear.mean(axis=2)
    rel_full=(vmax_full-vmin_full)/np.maximum(mean_full,0.035)
    usable_full=(lum_full>=0.055)&(lum_full<=0.88)
    for cell in cells:
        try:
            x=float(cell.get("x",0)); y=float(cell.get("y",0)); rw=float(cell.get("w",0)); rh=float(cell.get("h",0)); conf=float(cell.get("confidence",0.5))
        except (TypeError, ValueError, OverflowError):
            continue
        x0=max(0,min(w-1,int(np.floor(x*w)))); y0=max(0,min(h-1,int(np.floor(y*h))))
        x1=max(x0+1,min(w,int(np.ceil((x+rw)*w)))); y1=max(y0+1,min(h,int(np.ceil((y+rh)*h))))
        roi=linear[y0:y1,x0:x1]
        if roi.size==0: continue
        rel=rel_full[y0:y1,x0:x1]
        usable=usable_full[y0:y1,x0:x1]
        vals=rel[usable]
        if vals.size<32: continue
        # Lowest-chroma quartile approximates the same neutral anchors without
        # requiring the original pixel mask to be stored in the result payload.
        q=float(np.quantile(vals,0.30)); chosen=usable & (rel<=max(q,0.08))
        pix=roi[chosen]
        if pix.shape[0]<24: continue
        med=np.median(pix,axis=0)
        lg=np.log(np.clip(med,1e-5,None)); lg-=lg.mean()
        errors.append(float(np.sqrt(np.mean(np.square(lg)))))
        weights.append(max(0.1,conf))
    if not errors:
        return 0.0
    return float(np.average(np.asarray(errors),weights=np.asarray(weights)))


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8)


def apply_white_balance(
    rgb: np.ndarray,
    advice: WhiteBalanceAdvice,
    *,
    strength: float | None = None,
) -> np.ndarray:
    """Apply advisor gains in linear sRGB with bounded highlight protection."""
    _safe_image(rgb)
    if strength is None:
        strength = advice.neutralization_strength
    strength = float(np.clip(float(strength), 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()

    full_gains = np.asarray([advice.gain_r, advice.gain_g, advice.gain_b], dtype=np.float32)
    gains = np.power(np.clip(full_gains, 0.60, 1.80), strength).astype(np.float32)
    linear = srgb_to_linear(rgb).astype(np.float32)
    corrected = linear * gains[None, None, :]

    # Preserve global linear luminance. WB should change chromaticity, not act as
    # an exposure slider. Limit the scalar so a pathological scene cannot cause
    # large global relighting.
    before_y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    after_y = 0.2126 * corrected[..., 0] + 0.7152 * corrected[..., 1] + 0.0722 * corrected[..., 2]
    scale = float(np.mean(before_y) / max(float(np.mean(after_y)), 1e-6))
    scale = float(np.clip(scale, 0.88, 1.12))
    corrected *= scale

    # Smoothly compress only values above 1.0 instead of hard-clipping all gained
    # highlights. Values below 1 are untouched.
    over = corrected > 1.0
    if np.any(over):
        extra = corrected[over] - 1.0
        corrected[over] = 1.0 + extra / (1.0 + 4.0 * extra)
    corrected = np.clip(corrected, 0.0, 1.0)
    return _linear_to_srgb(corrected)
