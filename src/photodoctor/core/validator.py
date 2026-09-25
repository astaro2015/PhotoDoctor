from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Mapping, Any, Callable

import cv2
import numpy as np

from .loader import LoadedImage, srgb_to_linear
from .local_contrast import analyze_local_contrast, local_contrast_summary_score
from .local_correction_planner import (
    LocalCorrectionPlan,
    build_local_correction_plan,
    apply_local_exposure,
    apply_local_contrast,
    apply_local_sharpness,
    apply_local_denoise,
    rasterize_plan_field,
)
from .noise import analyze_noise
from .rendering_artifacts import analyze_edge_artifacts
from .models import MetricResult
from .precision import get_precision
from .performance import get_validator_profile
from .red_eye import correct_red_eye, build_red_eye_mask
from .auto_tone_color import analyze_auto_tone_color, apply_auto_tone_color, measure_neutral_midtone_cast
from .white_balance import analyze_white_balance, apply_white_balance, advice_from_raw, apply_spatial_white_balance, spatial_neutral_error
from .super_resolution import apply_super_resolution_x2, face_identity_drift
from .almaz_restoration import apply_almaz_restoration
from .almaz_safety import validate_almaz_transition
from .almaz_chroma_guard import guard_archival_chroma
from photodoctor.ai.restoration_runtime import inspect_installed_restoration_model


@dataclass(slots=True)
class ValidationItem:
    action_key: str
    tested: bool
    accepted: bool
    candidate: str
    target_before: float | None
    target_after: float | None
    confidence: float
    regressions: dict[str, float]
    message: str
    source_candidates: list[dict[str, Any]] | None = None
    preview_available: bool = True
    auto_eligible: bool = True
    technical_passed: bool | None = None
    adjustable: bool = False
    default_strength: float = 1.0
    parameter_label: str = "Сила"
    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _technical_copy(rgb: np.ndarray, long_edge: int = 1024) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, long_edge / max(h, w))
    if scale >= 1.0:
        return rgb.copy()
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _basic_probe(
    rgb: np.ndarray,
    fields: set[str] | frozenset[str] | tuple[str, ...] | list[str] | None = None,
    *,
    cache: dict[str, float] | None = None,
) -> dict[str, float]:
    """Measure only the requested validator signals, reusing an optional cache.

    The old implementation always ran every expensive probe (noise, edge, local
    contrast, sharpness) even when a validator needed only clipping/brightness.
    Selective lazy evaluation preserves the exact formulas while avoiding unrelated
    work.
    """
    all_fields = {
        "brightness", "highlight_clip_pct", "shadow_clip_pct", "contrast",
        "sharpness", "noise_score", "edge_score", "local_contrast_score",
    }
    need = all_fields if fields is None else set(fields)
    unknown = need - all_fields
    if unknown:
        raise KeyError(f"Unknown probe fields: {sorted(unknown)}")
    store = cache if cache is not None else {}
    missing = need - store.keys()
    if not missing:
        return {key: float(store[key]) for key in need}

    tone_fields = {"brightness", "highlight_clip_pct", "shadow_clip_pct", "contrast"}
    if missing & tone_fields:
        linear = srgb_to_linear(rgb)
        gray_lin = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
        if "brightness" in missing:
            store["brightness"] = float(gray_lin.mean())
        if "highlight_clip_pct" in missing:
            store["highlight_clip_pct"] = float(np.mean(gray_lin >= 0.99) * 100.0)
        if "shadow_clip_pct" in missing:
            store["shadow_clip_pct"] = float(np.mean(gray_lin <= 0.0031308) * 100.0)
        if "contrast" in missing:
            p5, p95 = np.percentile(gray_lin, [5, 95])
            store["contrast"] = float(p95 - p5)

    if "sharpness" in missing:
        gray8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap = float(cv2.Laplacian(gray8, cv2.CV_64F).var())
        gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
        ten = float(np.mean(gx * gx + gy * gy))
        lap_score = float(np.clip(20.0 * np.log10(1.0 + lap), 0.0, 100.0))
        ten_score = float(np.clip(14.0 * np.log10(1.0 + ten), 0.0, 100.0))
        store["sharpness"] = (lap_score + ten_score) / 2.0
    if "noise_score" in missing:
        store["noise_score"] = float(analyze_noise(rgb).score)
    if "edge_score" in missing:
        store["edge_score"] = float(analyze_edge_artifacts(rgb).score)
    if "local_contrast_score" in missing:
        store["local_contrast_score"] = float(local_contrast_summary_score(rgb))

    return {key: float(store[key]) for key in need}


_PROBE_TONE = frozenset({"brightness", "highlight_clip_pct", "shadow_clip_pct", "contrast"})
_PROBE_WB = frozenset({"brightness", "highlight_clip_pct", "shadow_clip_pct"})
_PROBE_EXPOSURE = frozenset({"brightness", "highlight_clip_pct", "shadow_clip_pct", "sharpness", "noise_score", "edge_score"})
_PROBE_CONTRAST = frozenset({"highlight_clip_pct", "shadow_clip_pct", "local_contrast_score", "edge_score", "sharpness"})
_PROBE_DETAIL = frozenset({"sharpness", "noise_score", "edge_score"})


def _gamma_lift(rgb: np.ndarray, gamma_value: float) -> np.ndarray:
    x = rgb.astype(np.float32) / 255.0
    y = np.power(np.clip(x, 0.0, 1.0), gamma_value)
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


def _mild_local_contrast(rgb: np.ndarray, strength: float = 1.0) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    effect = 0.65 * strength
    mixed = cv2.addWeighted(l, 1.0 - effect, l2, effect, 0.0)
    out = cv2.merge((mixed, a, b))
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def _edge_aware_sharpen(rgb: np.ndarray, strength: float = 0.35) -> np.ndarray:
    """Bounded luminance-only unsharp mask for explicit manual preview.

    It intentionally ignores very small differences to avoid amplifying flat-area noise
    and caps the amount well below typical editor sharpen filters.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32)
    blur = cv2.GaussianBlur(lf, (0, 0), sigmaX=1.0)
    detail = lf - blur
    gate = np.clip((np.abs(detail) - 1.5) / 7.0, 0.0, 1.0)
    amount = 0.62 * strength
    l2 = np.clip(lf + amount * detail * gate, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)


def _mild_denoise(rgb: np.ndarray, strength: float = 0.30) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()
    # Conservative coloured NLM. Kept manual-only because hair/fabric detail can
    # look like noise and should be judged by the user at 100%.
    h = 1.5 + 5.0 * strength
    hc = 1.5 + 4.0 * strength
    return cv2.fastNlMeansDenoisingColored(rgb, None, h, hc, 7, 21)


def _mild_deblock(rgb: np.ndarray, strength: float = 0.30) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()
    filtered = cv2.bilateralFilter(rgb, 5, 8.0 + 22.0 * strength, 3.0 + 2.0 * strength)
    alpha = 0.18 + 0.42 * strength
    return np.clip(np.rint(rgb.astype(np.float32) * (1.0 - alpha) + filtered.astype(np.float32) * alpha), 0, 255).astype(np.uint8)


def _mild_edge_soften(rgb: np.ndarray, strength: float = 0.25) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    edge_alpha = np.clip((lap - 12.0) / 55.0, 0.0, 1.0) * (0.12 + 0.38 * strength)
    softened = cv2.GaussianBlur(rgb, (0, 0), sigmaX=0.45 + 0.45 * strength)
    a = edge_alpha[..., None].astype(np.float32)
    return np.clip(np.rint(rgb.astype(np.float32) * (1.0 - a) + softened.astype(np.float32) * a), 0, 255).astype(np.uint8)


def _mild_deband(rgb: np.ndarray, strength: float = 0.25) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()
    # Deterministic sub-LSB ordered dither plus tiny luminance smoothing. This does
    # not invent texture; it only makes visible quantisation steps less abrupt.
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    smooth = cv2.GaussianBlur(l, (0, 0), sigmaX=0.35 + 0.45 * strength)
    lf = l.astype(np.float32) * (1.0 - 0.18 * strength) + smooth.astype(np.float32) * (0.18 * strength)
    bayer = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
    ], dtype=np.float32)
    pattern = (bayer / 63.0 - 0.5) * (1.2 * strength)
    tiled = np.tile(pattern, (int(np.ceil(l.shape[0] / 8)), int(np.ceil(l.shape[1] / 8))))[:l.shape[0], :l.shape[1]]
    l2 = np.clip(np.rint(lf + tiled), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)


def _candidate_contour_mask(shape: tuple[int, int], candidates: list[dict[str, Any]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        contour = cand.get("contour")
        points: list[tuple[int, int]] = []
        if isinstance(contour, list):
            for point in contour:
                if not isinstance(point, dict):
                    continue
                try:
                    px = float(point.get("x", 0.0)); py = float(point.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                x = int(round(px * w)) if 0.0 <= px <= 1.0 else int(round(px))
                y = int(round(py * h)) if 0.0 <= py <= 1.0 else int(round(py))
                points.append((max(0, min(w - 1, x)), max(0, min(h - 1, y))))
        if len(points) >= 2:
            arr = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            if len(points) >= 3:
                cv2.fillPoly(mask, [arr], 255)
            else:
                cv2.polylines(mask, [arr], False, 255, 2, cv2.LINE_AA)
            continue
        try:
            x = float(cand.get("x", 0.0)); y = float(cand.get("y", 0.0))
            bw = float(cand.get("w", 0.0)); bh = float(cand.get("h", 0.0))
        except (TypeError, ValueError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= bw <= 1.0 and 0.0 <= bh <= 1.0:
            x = int(round(x * w)); y = int(round(y * h)); bw = int(round(bw * w)); bh = int(round(bh * h))
        else:
            x = int(round(x)); y = int(round(y)); bw = int(round(bw)); bh = int(round(bh))
        if bw <= 0 or bh <= 0:
            continue
        x = max(0, min(w - 1, x)); y = max(0, min(h - 1, y))
        x2 = max(x + 1, min(w, x + bw)); y2 = max(y + 1, min(h, y + bh))
        cv2.rectangle(mask, (x, y), (x2 - 1, y2 - 1), 255, -1)
    if np.any(mask):
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return mask


def _candidate_line_endpoints_px(candidate: Mapping[str, Any], shape: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float]] | None:
    h, w = shape
    raw = candidate.get("line_endpoints")
    if not isinstance(raw, list) or len(raw) < 2:
        return None
    pts: list[tuple[float, float]] = []
    for point in raw[:2]:
        if not isinstance(point, Mapping):
            return None
        try:
            x = float(point.get("x", 0.0)); y = float(point.get("y", 0.0))
        except (TypeError, ValueError, OverflowError):
            return None
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x *= w; y *= h
        pts.append((float(np.clip(x, 0.0, w - 1.0)), float(np.clip(y, 0.0, h - 1.0))))
    return pts[0], pts[1]


def _angle_delta(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _surface_mask_residual_gray(gray: np.ndarray, blur_sigma: float = 4.5) -> np.ndarray:
    """Return the local bright-ridge residual from a precomputed float gray frame."""
    sigma = float(max(2.0, blur_sigma))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101)
    return gray - background


def _surface_mask_residual(rgb: np.ndarray, blur_sigma: float = 4.5) -> np.ndarray:
    """Return a local bright-ridge residual used to estimate real crack width.

    A Surface candidate is only the *seed* of a physical crack.  Old paper damage
    can be 1 px wide in one place and 8-12 px wide a few millimetres later.  Using
    one fixed stroke width leaves a bright rim behind, especially on faces/light
    paper.  The residual is deliberately luminance-only and local, so growth cannot
    wander far from a confirmed seed.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return _surface_mask_residual_gray(gray, blur_sigma)


def _grow_surface_mask_to_ridge_width(
    rgb: np.ndarray,
    seed_mask: np.ndarray,
    candidates: list[dict[str, Any]],
    *,
    gray: np.ndarray | None = None,
) -> np.ndarray:
    """Expand only from confirmed seeds into locally bright crack pixels.

    Surface v8 replaces the old almost-fixed-width repair mask with a bounded
    geodesic growth.  Expansion is capped tightly (and tighter on faces) and every
    newly added pixel must still look like the same local bright ridge.  This is
    what lets a wide white emulsion loss be removed without turning a face-sized
    rectangle into an inpaint target.
    """
    if seed_mask.size == 0 or not np.any(seed_mask):
        return seed_mask.copy()
    h, w = seed_mask.shape
    widths: list[float] = []
    semantic = 0.0
    for cand in candidates:
        if not isinstance(cand, Mapping):
            continue
        try:
            widths.append(float(cand.get("stroke_width_px", cand.get("oriented_width_px", 2.0)) or 2.0))
            semantic = max(semantic, float(cand.get("semantic_risk", 0.0) or 0.0))
        except (TypeError, ValueError, OverflowError):
            continue
    typical_width = float(np.percentile(widths, 70.0)) if widths else 2.0
    max_growth = int(round(np.clip(typical_width * 1.20 + 4.0, 4.0, 11.0)))
    if semantic >= 0.35:
        max_growth = min(max_growth, 7)  # face-safe: accurate width, never a broad patch

    if gray is None:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    residual = _surface_mask_residual_gray(gray, blur_sigma=max(3.5, max_growth * 0.75))
    seed = seed_mask > 0
    outer = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_growth * 2 + 1, max_growth * 2 + 1))) > 0
    ring_outer = cv2.dilate(seed.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_growth * 2 + 5, max_growth * 2 + 5))) > 0
    ring = ring_outer & ~outer
    noise = residual[ring]
    if noise.size >= 32:
        med = float(np.median(noise))
        mad = float(np.median(np.abs(noise - med)))
        sigma = max(0.8, 1.4826 * mad)
    else:
        med, sigma = 0.0, 1.2
    # Faint archival cracks can be only a couple of code values above the paper.
    # Seed proximity plus geodesic growth is the safety constraint, not a huge
    # global threshold.
    threshold = max(0.9, med + 0.72 * sigma)

    # A flat bright region next to a natural edge can also have positive Gaussian
    # residual.  Require a two-sided ridge at *some* axis using a sample distance
    # larger than the expected crack half-width.  A real scratch is locally higher
    # than both sides; a one-sided object boundary is not.
    offset = int(np.clip(max_growth + 1, 4, 12))
    ridge_pairs: list[np.ndarray] = []
    for dx, dy in ((offset, 0), (0, offset), (offset, offset), (offset, -offset)):
        m1 = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
        m2 = np.asarray([[1.0, 0.0, float(-dx)], [0.0, 1.0, float(-dy)]], dtype=np.float32)
        side1 = cv2.warpAffine(gray, m1, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        side2 = cv2.warpAffine(gray, m2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        ridge_pairs.append(np.minimum(gray - side1, gray - side2))
    ridge_support = np.maximum.reduce(ridge_pairs)
    support = outer & (residual >= threshold) & (ridge_support >= max(0.7, threshold * 0.45))

    grown = seed.copy()
    kernel3 = np.ones((3, 3), np.uint8)
    for _ in range(max_growth):
        expanded = cv2.dilate(grown.astype(np.uint8), kernel3, iterations=1) > 0
        nxt = grown | (expanded & support)
        if np.array_equal(nxt, grown):
            break
        grown = nxt

    out = (grown.astype(np.uint8) * 255)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return out


def _linked_surface_repair_mask(rgb: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    """Build a crack-track repair mask with ridge-verified long-gap linking.

    Surface v8 treats confirmed collinear fragments as samples of one physical
    crack.  Gaps may now be much longer than v7's 18 px, but only when the bridge
    itself follows the segment direction and retains a two-sided bright-ridge
    signature.  The resulting seed is then expanded to the *measured* crack width.
    """
    shape = rgb.shape[:2]
    h, w = shape
    mask = _candidate_contour_mask(shape, candidates)
    if not np.any(mask):
        return mask
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    min_dim = min(h, w)
    max_gap = float(np.clip(min_dim * 0.080, 18.0, 76.0))

    lines: list[tuple[dict[str, Any], tuple[tuple[float, float], tuple[float, float]], float, float]] = []
    for cand in candidates:
        if not isinstance(cand, dict) or str(cand.get("candidate_kind", "")) != "line":
            continue
        if str(cand.get("polarity", "bright")) != "bright":
            continue
        endpoints = _candidate_line_endpoints_px(cand, shape)
        if endpoints is None:
            continue
        try:
            angle = float(cand.get("orientation_deg", 0.0) or 0.0)
            width = float(cand.get("stroke_width_px", cand.get("oriented_width_px", 2.0)) or 2.0)
        except (TypeError, ValueError, OverflowError):
            continue
        lines.append((cand, endpoints, angle, float(np.clip(width, 1.0, 10.0))))

    def take(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        xi = np.clip(np.rint(xx).astype(np.int32), 0, w - 1)
        yi = np.clip(np.rint(yy).astype(np.int32), 0, h - 1)
        return gray[yi, xi]

    for i in range(len(lines)):
        _, ep_a, angle_a, width_a = lines[i]
        for j in range(i + 1, len(lines)):
            _, ep_b, angle_b, width_b = lines[j]
            if _angle_delta(angle_a, angle_b) > 28.0:
                continue
            pairs = [
                (ep_a[0], ep_b[0]), (ep_a[0], ep_b[1]),
                (ep_a[1], ep_b[0]), (ep_a[1], ep_b[1]),
            ]
            p1, p2 = min(pairs, key=lambda pair: float(np.hypot(pair[0][0]-pair[1][0], pair[0][1]-pair[1][1])))
            dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
            dist = float(np.hypot(dx, dy))
            if dist < 1.0 or dist > max_gap:
                continue
            bridge_angle = float((np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0)
            # Two parallel but unrelated seams must not be connected sideways.
            if _angle_delta(bridge_angle, angle_a) > 34.0 or _angle_delta(bridge_angle, angle_b) > 34.0:
                continue
            ux, uy = dx / dist, dy / dist
            nx, ny = -uy, ux
            samples = max(7, int(np.ceil(dist)) + 1)
            t = np.linspace(0.0, 1.0, samples, dtype=np.float32)
            xs = p1[0] + dx * t; ys = p1[1] + dy * t
            offset = float(np.clip(max(width_a, width_b) * 1.8, 2.5, 8.0))
            center = take(xs, ys)
            side1 = take(xs + nx * offset, ys + ny * offset)
            side2 = take(xs - nx * offset, ys - ny * offset)
            ridge = np.minimum(center - side1, center - side2)
            # Long faint gaps get a slightly lower median requirement, but need
            # broad support along the whole bridge.
            required_median = 1.25 if dist <= 24.0 else 0.75
            required_fraction = 0.38 if dist <= 24.0 else 0.48
            if float(np.median(ridge)) < required_median or float(np.mean(ridge >= 1.25)) < required_fraction:
                continue
            thickness = max(2, int(round(max(width_a, width_b))))
            cv2.line(mask, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])), int(round(p2[1]))), 255, thickness, cv2.LINE_AA)

    return _grow_surface_mask_to_ridge_width(rgb, mask, candidates, gray=gray)


def _surface_bright_contrast(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Measure remaining local bright-ridge contrast inside the exact repair mask."""
    if mask.size == 0 or not np.any(mask):
        return 0.0
    residual = _surface_mask_residual(rgb, blur_sigma=4.5)
    vals = residual[mask > 0]
    if vals.size == 0:
        return 0.0
    positive = np.clip(vals, 0.0, None)
    # Mean of the upper half is stable for mixed masks containing tiny bridge gaps.
    floor = float(np.percentile(positive, 50.0))
    upper = positive[positive >= floor]
    return float(np.mean(upper)) if upper.size else float(np.mean(positive))


def _surface_repair_effectiveness(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    before_c = _surface_bright_contrast(before, mask)
    after_c = _surface_bright_contrast(after, mask)
    if before_c <= 1e-6:
        return before_c, after_c, 1.0
    reduction = float(np.clip((before_c - after_c) / before_c, -1.0, 1.0))
    return before_c, after_c, reduction

def _surface_history_overlap(candidate: Mapping[str, Any], history_mask: np.ndarray) -> float:
    if history_mask.size == 0 or not np.any(history_mask):
        return 0.0
    cand_mask = _candidate_contour_mask(history_mask.shape, [dict(candidate)])
    count = int(np.count_nonzero(cand_mask))
    if count <= 0:
        return 0.0
    expanded = cv2.dilate(history_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    overlap = int(np.count_nonzero((cand_mask > 0) & (expanded > 0)))
    return float(overlap / count)


def _bounded_surface_heal(
    rgb: np.ndarray,
    candidates: list[dict[str, Any]],
    strength: float = 0.5,
    *,
    repair_mask: np.ndarray | None = None,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or not candidates:
        return rgb.copy()
    mask = repair_mask if repair_mask is not None else _linked_surface_repair_mask(rgb, candidates)
    if not np.any(mask):
        return rgb.copy()
    widths: list[float] = []
    for cand in candidates:
        try:
            widths.append(float(cand.get("stroke_width_px", cand.get("oriented_width_px", 2.0)) or 2.0))
        except (TypeError, ValueError, OverflowError, AttributeError):
            continue
    typical_width = float(np.percentile(widths, 75.0)) if widths else 2.0
    radius = float(np.clip(1.6 + typical_width * 0.55, 2.0, 5.0))
    healed = cv2.inpaint(rgb, mask, radius, cv2.INPAINT_TELEA)
    alpha = strength
    return np.clip(np.rint(rgb.astype(np.float32) * (1.0 - alpha) + healed.astype(np.float32) * alpha), 0, 255).astype(np.uint8)


def _surface_candidate_repairable_for_manual_preview(candidate: Mapping[str, Any]) -> bool:
    """Return whether a Surface candidate is technically safe enough for a manual preview.

    This is deliberately *not* an auto-application decision. Surface v9 is always
    user-armed: the helper only keeps obviously unsafe/nonsensical candidates out
    of the bounded repair preview after the user chooses them.
    """
    if bool(candidate.get("previously_repaired", False)):
        return False
    label = str(candidate.get("verification_label", candidate.get("ai_label", "unprocessed")))
    if label != "defect":
        return False
    polarity = str(candidate.get("polarity", "bright"))
    kind = str(candidate.get("candidate_kind", ""))
    branch = str(candidate.get("detection_branch", "primary"))
    try:
        semantic_risk = float(candidate.get("semantic_risk", 0.0) or 0.0)
        quality = float(candidate.get("candidate_quality", 0.0) or 0.0)
        confidence = float(candidate.get("verification_confidence", candidate.get("ai_confidence", 0.0)) or 0.0)
    except (TypeError, ValueError, OverflowError):
        return False
    if polarity != "bright" or kind not in {"line", "irregular"}:
        return False
    if confidence < 0.56:
        return False
    # Eye regions remain protected by default, but a high-confidence bright
    # physical ridge may cross an eye.  Do not veto the *damage* merely because
    # the underlying anatomy is important; instead require stronger evidence and
    # keep the adaptive mask tightly bounded.
    if semantic_risk >= 0.80:
        try:
            context_contrast = float(candidate.get("context_contrast", 0.0) or 0.0)
            side_similarity = float(candidate.get("side_similarity", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            return False
        face_safe_override = bool(
            semantic_risk < 0.98
            and confidence >= 0.72
            and quality >= 0.55
            and context_contrast >= 0.50
            and side_similarity >= 0.52
        )
        if not face_safe_override:
            return False
    if branch == "low_contrast_hough" and quality < 0.42:
        return False
    return True


def surface_candidate_auto_repair_eligible(candidate: Mapping[str, Any]) -> bool:
    """Surface v9 never auto-arms a defect repair.

    Kept as a public compatibility guard because older GUI/tests imported this
    helper. Returning False here makes the manual-only contract fail closed.
    """
    return False


def refresh_surface_validation_after_refinement(
    rgb: np.ndarray, metrics: dict[str, MetricResult], surface_history: Mapping[str, Any] | None = None
) -> None:
    """Synchronize executable surface repair with the post-AI refined map.

    Classical validation runs before local Surface AI. Without this second pass the
    GUI can show a confirmed faint crack while preview still receives only the old
    pre-AI first-eight fallback. Surface v8 rebuilds the executable candidate list
    from the fused map and validates the exact narrow-mask repair once more.
    """
    validation_metric = metrics.get("recommendation_validation")
    refinement_metric = metrics.get("surface_refinement")
    if validation_metric is None or refinement_metric is None:
        return
    validation_raw = validation_metric.raw_value if isinstance(validation_metric.raw_value, dict) else None
    refinement_raw = refinement_metric.raw_value if isinstance(refinement_metric.raw_value, dict) else None
    if not isinstance(validation_raw, dict) or not isinstance(refinement_raw, dict):
        return
    items = validation_raw.get("items", [])
    boxes = refinement_raw.get("refined_boxes_norm", [])
    if not isinstance(items, list) or not isinstance(boxes, list):
        return

    history_payload = surface_history if isinstance(surface_history, Mapping) else {}
    history_boxes = history_payload.get("candidates", []) if isinstance(history_payload, Mapping) else []
    if not isinstance(history_boxes, list):
        history_boxes = []
    history_boxes = [dict(box) for box in history_boxes if isinstance(box, Mapping)]
    history_mask = _candidate_contour_mask(rgb.shape[:2], history_boxes) if history_boxes else np.zeros(rgb.shape[:2], np.uint8)
    try:
        history_passes = max(0, int(history_payload.get("passes", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        history_passes = 0

    marked_boxes: list[dict[str, Any]] = []
    history_filtered = 0
    for raw_box in boxes:
        if not isinstance(raw_box, dict):
            continue
        box = dict(raw_box)
        overlap = _surface_history_overlap(box, history_mask) if history_boxes else 0.0
        box["repair_history_overlap"] = overlap
        box["previously_repaired"] = bool(overlap >= 0.42)
        if box["previously_repaired"]:
            history_filtered += 1
        marked_boxes.append(box)
    boxes = marked_boxes
    candidates = [dict(box) for box in boxes if _surface_candidate_repairable_for_manual_preview(box)]

    # Keep the visible/refined map in sync so the GUI cannot auto-check a region
    # that the validator has already suppressed from a previous saved Surface pass.
    ref_raw = dict(refinement_raw)
    ref_raw["refined_boxes_norm"] = boxes
    ref_raw["history_filtered_count"] = history_filtered
    ref_raw["history_passes"] = history_passes
    metrics["surface_refinement"] = MetricResult(
        refinement_metric.name, ref_raw, refinement_metric.normalized_value, refinement_metric.confidence,
        refinement_metric.scale, region=refinement_metric.region,
        diagnostic=refinement_metric.diagnostic + f"; Surface v9 history filtered={history_filtered}, passes={history_passes}",
    )

    updated_items: list[dict[str, Any]] = []
    found = False
    for original in items:
        if not isinstance(original, dict):
            continue
        item = dict(original)
        if str(item.get("action_key", "")) != "surface_defects":
            updated_items.append(item)
            continue
        found = True
        if not candidates:
            item.update({
                "accepted": False,
                "auto_eligible": False,
                "technical_passed": False,
                "source_candidates": [],
                "preview_available": False,
                "message": (
                    "Surface v9 не нашёл новых подтверждённых светлых линий, безопасных для автоматического лечения. "
                    "Неоднозначные кандидаты остаются на вкладке «Дефекты» для ручного выбора."
                ),
            })
            updated_items.append(item)
            continue

        # Surface v9 is strictly manual-only. At analysis time we rank/filter
        # candidates and expose them to the Defects tab, but we do NOT build one
        # giant repair mask and trial-inpaint every confirmed candidate before the
        # user has selected anything. The exact bounded heal + efficacy check runs
        # later for the user-selected subset when Preview is requested.
        confidences = [
            float(box.get("verification_confidence", box.get("ai_confidence", 0.0)) or 0.0)
            for box in candidates
        ]
        confidence = float(np.clip(np.mean(confidences) if confidences else 0.0, 0.0, 0.98))
        # Preserve the useful V7/V8 convergence diagnostic without running any
        # repair trial.  Geometry-only mask area is cheap and does not alter the
        # photo; it lets the UI say that only a few tiny residuals remain after
        # earlier saved Surface passes while manual-only remains absolute.
        candidate_mask = _candidate_contour_mask(rgb.shape[:2], candidates)
        mask_pct = float(np.mean(candidate_mask > 0) * 100.0) if candidate_mask.size else 0.0
        residual_manual_only = bool(
            (history_passes >= 1 and len(candidates) <= 2 and mask_pct <= 0.35 and confidence < 0.82)
            or (history_passes >= 2 and len(candidates) <= 3 and mask_pct <= 0.15 and confidence < 0.82)
        )
        regressions = dict(item.get("regressions", {}) or {})
        regressions.update({
            "candidate_count": float(len(candidates)),
            "post_ai_refined": 1.0,
            "repair_mask_pct": mask_pct,
            "history_filtered_count": float(history_filtered),
            "history_passes": float(history_passes),
            "residual_manual_only": float(residual_manual_only),
            "repair_trial_deferred_until_manual_selection": 1.0,
        })
        if residual_manual_only:
            message = (
                f"После {history_passes} сохранённых Surface-проходов осталось только {len(candidates)} "
                f"маленьких слабых кандидата(ов) ({mask_pct:.3f}% кадра). Они оставлены только для ручной проверки. "
                "Автолечение отключено."
            )
        else:
            message = (
                f"Surface v9 нашёл {len(candidates)} пригодных кандидата(ов) после AI/Context-проверки; "
                f"уже исправленных областей пропущено: {history_filtered}. "
                "Автолечение отключено. Пробная маска и проверка эффективности строятся только для участков, "
                "которые вы сами отметите в колонке «Лечить»."
            )
        item.update({
            "accepted": False,
            "auto_eligible": False,
            # None means no repair trial has run yet; False would incorrectly mean
            # that a trial was attempted and failed.
            "technical_passed": None,
            "candidate": "bounded_surface_heal",
            "confidence": confidence,
            "regressions": regressions,
            "source_candidates": candidates,
            "preview_available": bool(candidates),
            "adjustable": True,
            "default_strength": 0.90,
            "parameter_label": "Сила",
            "message": message,
        })
        updated_items.append(item)

    if not found:
        return
    raw = dict(validation_raw)
    raw["items"] = updated_items
    raw["tested_count"] = len(updated_items)
    raw["accepted_count"] = sum(bool(item.get("accepted", False)) for item in updated_items)
    raw["rejected_count"] = len(updated_items) - int(raw["accepted_count"])
    metrics["recommendation_validation"] = MetricResult(
        validation_metric.name, raw, validation_metric.normalized_value,
        validation_metric.confidence, validation_metric.scale,
        region=validation_metric.region,
        diagnostic=validation_metric.diagnostic + "; Surface v9 post-AI repair synchronization",
    )


def _decision_actions(metrics: Mapping[str, MetricResult]) -> dict[str, str]:
    raw = metrics.get("decision_plan")
    payload = raw.raw_value if raw is not None and isinstance(raw.raw_value, dict) else {}
    items = payload.get("items", []) if isinstance(payload, dict) else []
    actions: dict[str, str] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision", ""))
            key = str(item.get("key", ""))
            if key and decision in {"fix", "review"}:
                actions[key] = decision
    return actions


def _normalized_box_slice(cell: Mapping[str, Any], height: int, width: int) -> tuple[slice, slice] | None:
    try:
        x = float(cell.get("x", 0.0)); y = float(cell.get("y", 0.0))
        rw = float(cell.get("w", 0.0)); rh = float(cell.get("h", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite([x, y, rw, rh]).all() or rw <= 0.0 or rh <= 0.0:
        return None
    x0 = max(0, min(width, int(round(x * width)))); y0 = max(0, min(height, int(round(y * height))))
    x1 = max(0, min(width, int(round((x + rw) * width)))); y1 = max(0, min(height, int(round((y + rh) * height))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return slice(y0, y1), slice(x0, x1)


def _exposure_plan_error(rgb: np.ndarray, cells: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> float:
    linear = srgb_to_linear(rgb)
    luma = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    errors: list[float] = []
    weights: list[float] = []
    h, w = luma.shape
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        box = _normalized_box_slice(cell, h, w)
        if box is None:
            continue
        tile = luma[box]
        if tile.size < 16:
            continue
        p50 = float(np.percentile(tile, 50))
        target = float(cell.get("target_luma", p50) or p50)
        kind = str(cell.get("kind", ""))
        error = max(0.0, target - p50) if kind == "lift" else max(0.0, p50 - target)
        errors.append(error)
        weights.append(float(tile.size))
    if not errors:
        return 0.0
    return float(np.average(np.asarray(errors), weights=np.asarray(weights)))


def _contrast_plan_range(rgb: np.ndarray, cells: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    values: list[float] = []
    weights: list[float] = []
    h, w = gray.shape
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        box = _normalized_box_slice(cell, h, w)
        if box is None:
            continue
        tile = gray[box]
        if tile.size < 64:
            continue
        p10, p90 = np.percentile(tile, [10, 90])
        values.append(float(p90 - p10))
        weights.append(float(tile.size))
    if not values:
        return 0.0
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))


def _outside_change(rgb: np.ndarray, corrected: np.ndarray, cells: list[dict[str, Any]] | tuple[dict[str, Any], ...], value_key: str) -> float:
    _field, support = rasterize_plan_field(cells, rgb.shape[0], rgb.shape[1], value_key=value_key)
    outside = support < 0.05
    if not np.any(outside):
        return 0.0
    diff = np.abs(corrected.astype(np.float32) - rgb.astype(np.float32)) / 255.0
    return float(np.mean(diff[outside]))


def _resolve_local_plan(
    metrics: Mapping[str, MetricResult] | None,
    local_plan: LocalCorrectionPlan | None = None,
) -> LocalCorrectionPlan | None:
    """Reuse Analyzer's exact plan; rebuild only for legacy/incomplete metric sets."""
    if local_plan is not None:
        return local_plan
    if metrics is None:
        return None
    metric = metrics.get("local_correction_plan")
    raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, Mapping) else None
    if raw:
        try:
            return LocalCorrectionPlan.from_raw(raw)
        except (TypeError, ValueError, KeyError):
            pass
    return build_local_correction_plan(metrics)


def _validate_exposure(
    rgb: np.ndarray, metrics: Mapping[str, MetricResult] | None = None,
    local_plan: LocalCorrectionPlan | None = None,
    before_cache: dict[str, float] | None = None,
) -> ValidationItem:
    before = _basic_probe(rgb, _PROBE_EXPOSURE, cache=before_cache)
    if metrics is not None:
        spatial = _resolve_local_plan(metrics, local_plan)
        if spatial is not None and len(spatial.exposure_cells) >= 2 and spatial.exposure_confidence >= 0.48:
            parameters = spatial.to_raw()
            strength = 0.80
            corrected = apply_local_exposure(rgb, parameters, strength=strength)
            after = _basic_probe(corrected, _PROBE_EXPOSURE)
            err_before = _exposure_plan_error(rgb, list(spatial.exposure_cells))
            err_after = _exposure_plan_error(corrected, list(spatial.exposure_cells))
            improvement = err_before - err_after
            highlight_reg = after["highlight_clip_pct"] - before["highlight_clip_pct"]
            shadow_reg = after["shadow_clip_pct"] - before["shadow_clip_pct"]
            outside_delta = _outside_change(rgb, corrected, list(spatial.exposure_cells), "delta_gamma")
            technical = bool(
                err_before >= 0.008
                and improvement >= max(0.003, err_before * 0.10)
                and highlight_reg <= 0.45
                and shadow_reg <= 0.70
                and outside_delta <= 0.008
            )
            return ValidationItem(
                "exposure", True, technical, "spatial_exposure_v1", err_before, err_after,
                float(np.clip(0.64 + spatial.exposure_confidence * 0.30 + (0.04 if technical else -0.04), 0.55, 0.92)),
                {
                    "target_error_delta": float(err_after - err_before),
                    "highlight_clip_delta_pct": float(highlight_reg),
                    "shadow_clip_delta_pct": float(shadow_reg),
                    "outside_mean_abs_delta": float(outside_delta),
                    "coverage_pct": float(spatial.exposure_coverage * 100.0),
                },
                (
                    f"Пространственная коррекция уменьшила ошибку яркости именно в согласованных областях ({spatial.exposure_coverage * 100:.0f}% кадра), "
                    "не меняя нормальную часть изображения заметно."
                    if technical else
                    "Локальная карта нашла проблемные области, но пробная пространственная коррекция не прошла ограничения безопасности; оставлена для ручного сравнения."
                ),
                preview_available=True, auto_eligible=technical, technical_passed=technical,
                adjustable=True, default_strength=strength, parameter_label="Сила", parameters=parameters,
            )

    # Gamma operates in display space while the probe is measured in linear
    # light.  Keep the proposal intentionally mild; the previous target around
    # 0.40 linear mean was too bright for many perfectly normal photographs.
    candidates = (0.94, 0.90, 0.86, 0.82)
    default_strength = 0.70
    target = min(0.24, max(0.17, before["brightness"] + 0.035))
    best: tuple[float, dict[str, float], float] | None = None

    for gamma_value in candidates:
        effective_gamma = 1.0 - default_strength * (1.0 - gamma_value)
        candidate = _gamma_lift(rgb, effective_gamma)
        after = _basic_probe(candidate, _PROBE_EXPOSURE)
        brightness_gain = after["brightness"] - before["brightness"]
        highlight_reg = after["highlight_clip_pct"] - before["highlight_clip_pct"]
        sharp_reg = before["sharpness"] - after["sharpness"]
        edge_reg = before["edge_score"] - after["edge_score"]
        if brightness_gain < 0.006:
            continue
        if after["brightness"] > 0.42 or highlight_reg > 0.8 or sharp_reg > 3.5 or edge_reg > 7.0:
            continue
        score = abs(after["brightness"] - target) + max(0.0, highlight_reg) * 0.04
        if best is None or score < best[0]:
            best = (score, after, gamma_value)

    if best is None:
        manual_gamma = candidates[0]
        effective_gamma = 1.0 - default_strength * (1.0 - manual_gamma)
        manual_after = _basic_probe(_gamma_lift(rgb, effective_gamma), _PROBE_EXPOSURE)
        return ValidationItem(
            "exposure", True, False, f"gamma={manual_gamma:.2f}", before["brightness"], manual_after["brightness"], 0.68,
            {
                "highlight_clip_delta_pct": manual_after["highlight_clip_pct"] - before["highlight_clip_pct"],
                "sharpness_delta": manual_after["sharpness"] - before["sharpness"],
                "edge_score_delta": manual_after["edge_score"] - before["edge_score"],
                "noise_score_delta": manual_after["noise_score"] - before["noise_score"],
            },
            "Даже консервативный подъём средних тонов не прошёл ограничения по светам/деталям; доступен только ручной мягкий предпросмотр.",
            adjustable=True, default_strength=default_strength, parameter_label="Сила"
        )

    after = best[1]
    best_gamma = best[2]
    regressions = {
        "highlight_clip_delta_pct": after["highlight_clip_pct"] - before["highlight_clip_pct"],
        "sharpness_delta": after["sharpness"] - before["sharpness"],
        "edge_score_delta": after["edge_score"] - before["edge_score"],
        "noise_score_delta": after["noise_score"] - before["noise_score"],
    }
    confidence = float(np.clip(0.70 + min(after["brightness"] - before["brightness"], 0.08), 0.70, 0.80))
    return ValidationItem(
        "exposure", True, True, f"gamma={best_gamma:.2f}", before["brightness"], after["brightness"], confidence,
        regressions,
        "Консервативный подъём средних тонов даёт небольшой выигрыш на технической копии без заметного роста потерь в светах или деталях.",
        adjustable=True, default_strength=default_strength, parameter_label="Сила"
    )


def _validate_auto_tone_color(rgb: np.ndarray) -> ValidationItem:
    before = _basic_probe(rgb, _PROBE_TONE)
    profile_before = analyze_auto_tone_color(rgb)
    strength = 0.80
    corrected = apply_auto_tone_color(rgb, strength=strength, profile=profile_before)
    after = _basic_probe(corrected, _PROBE_TONE)
    before_cast, _before_neutral, _fraction = measure_neutral_midtone_cast(rgb, reference_rgb=rgb)
    after_cast, _after_neutral, _fraction2 = measure_neutral_midtone_cast(corrected, reference_rgb=rgb)

    contrast_gain = after["contrast"] - before["contrast"]
    cast_gain = before_cast - after_cast
    highlight_reg = after["highlight_clip_pct"] - before["highlight_clip_pct"]
    shadow_reg = after["shadow_clip_pct"] - before["shadow_clip_pct"]
    brightness_delta = after["brightness"] - before["brightness"]
    technical = (
        (contrast_gain >= 0.010 or cast_gain >= 0.008)
        and highlight_reg <= 0.8
        and shadow_reg <= 2.0
        and abs(brightness_delta) <= 0.045
    )
    regressions = {
        "contrast_delta": float(contrast_gain),
        "neutral_cast_delta": float(after_cast - before_cast),
        "highlight_clip_delta_pct": float(highlight_reg),
        "shadow_clip_delta_pct": float(shadow_reg),
        "brightness_delta": float(brightness_delta),
    }
    message = (
        "Комбинированная поканальная коррекция расширила тональный диапазон/нейтрализовала полутона без заметного сдвига средней яркости. "
        "Режим откалиброван по реальной паре Photoshop Автотон → Автоконтраст → Автоцвет; это не точная копия закрытого алгоритма Adobe."
        if technical else
        "На пробной копии комбинированная автокоррекция не дала достаточно безопасного выигрыша; оставлена только для явного ручного сравнения."
    )
    return ValidationItem(
        "auto_tone_color", True, bool(technical), "reference_auto_tone_contrast_color_v1",
        float(before["contrast"]), float(after["contrast"]), 0.76 if technical else 0.58,
        regressions, message, preview_available=True, auto_eligible=False,
        technical_passed=bool(technical), adjustable=True, default_strength=strength, parameter_label="Сила",
    )


def _face_skin_chroma_drift(before_rgb: np.ndarray, after_rgb: np.ndarray, face_regions: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> float:
    if not face_regions:
        return 0.0
    h, w = before_rgb.shape[:2]
    before_ycc = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2YCrCb)
    before_lab = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    after_lab = cv2.cvtColor(after_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    yy, cr, cb = cv2.split(before_ycc)
    skin = (yy >= 38) & (yy <= 248) & (cr >= 132) & (cr <= 182) & (cb >= 72) & (cb <= 138)
    values: list[float] = []
    for region in face_regions:
        box = _normalized_box_slice(region, h, w)
        if box is None:
            continue
        local_skin = skin[box]
        before_roi = before_lab[box][..., 1:3]
        after_roi = after_lab[box][..., 1:3]
        if np.count_nonzero(local_skin) >= 24:
            sample = local_skin
        else:
            # Strong warm/cool/magenta casts can move genuine skin outside the
            # classical YCrCb envelope. Never turn that into an automatic safety
            # pass. Fall back to the central face interior with moderate luma.
            local_y = yy[box]
            rh, rw = local_y.shape[:2]
            inner = np.zeros((rh, rw), dtype=bool)
            y0 = max(0, int(round(rh * 0.16))); y1 = min(rh, int(round(rh * 0.86)))
            x0 = max(0, int(round(rw * 0.16))); x1 = min(rw, int(round(rw * 0.84)))
            if y1 > y0 and x1 > x0:
                inner[y0:y1, x0:x1] = True
            sample = inner & (local_y >= 35) & (local_y <= 245)
            if np.count_nonzero(sample) < 24:
                continue
        ba = before_roi[sample]
        aa = after_roi[sample]
        delta = np.sqrt(np.sum(np.square(aa - ba), axis=1))
        values.append(float(np.median(delta)))
    return float(max(values)) if values else 0.0


def _validate_white_balance(
    rgb: np.ndarray, metrics: Mapping[str, MetricResult],
    local_plan: LocalCorrectionPlan | None = None,
    before_cache: dict[str, float] | None = None,
) -> ValidationItem:
    metric = metrics.get("white_balance_advisor")
    raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
    try:
        advice = advice_from_raw(raw) if raw else analyze_white_balance(rgb)
    except Exception:
        advice = analyze_white_balance(rgb)

    spatial = _resolve_local_plan(metrics, local_plan)
    use_spatial = bool(
        spatial is not None and
        spatial.white_balance_kind in {"mixed_light", "local_cast"}
        and len(spatial.white_balance_cells) >= 2
        and spatial.white_balance_confidence >= 0.48
    )
    before = _basic_probe(rgb, _PROBE_WB, cache=before_cache)
    if use_spatial:
        parameters = spatial.to_raw()
        parameters["global_advice"] = advice.to_raw()
        strength = float(np.clip(max(0.62, min(0.88, 0.58 + 0.32 * spatial.white_balance_confidence)), 0.0, 1.0))
        corrected = apply_spatial_white_balance(rgb, parameters, strength=strength)
        after = _basic_probe(corrected, _PROBE_WB)
        cells = list(spatial.white_balance_cells)
        error_before = spatial_neutral_error(rgb, cells)
        error_after = spatial_neutral_error(corrected, cells)
        improvement = error_before - error_after
        highlight_reg = float(after["highlight_clip_pct"] - before["highlight_clip_pct"])
        shadow_reg = float(after["shadow_clip_pct"] - before["shadow_clip_pct"])
        brightness_delta = float(after["brightness"] - before["brightness"])
        skin_drift = _face_skin_chroma_drift(rgb, corrected, list(spatial.face_regions))
        technical = bool(
            error_before >= 0.010
            and improvement >= max(0.0025, error_before * 0.08)
            and highlight_reg <= 0.65
            and shadow_reg <= 1.2
            and abs(brightness_delta) <= 0.035
            and skin_drift <= 7.0
        )
        regressions = {
            "local_neutral_error_delta": float(error_after - error_before),
            "highlight_clip_delta_pct": highlight_reg,
            "shadow_clip_delta_pct": shadow_reg,
            "brightness_delta": brightness_delta,
            "skin_chroma_drift_lab": skin_drift,
            "mixed_light_score": float(spatial.white_balance_mixed_score),
            "local_coverage": float(spatial.white_balance_coverage),
        }
        message = (
            f"Локальный баланс белого уменьшил цветовой разброс в подтверждённых нейтральных областях без заметной порчи кожи и экспозиции. "
            f"Карта mixed-light {spatial.white_balance_mixed_score * 100:.0f}%, покрытие {spatial.white_balance_coverage * 100:.0f}%."
            if technical else
            "Пробный spatial WB не дал достаточно безопасного локального выигрыша или затронул кожу/тон сильнее допустимого; оставлен только для ручного сравнения."
        )
        confidence = float(np.clip(0.52 + 0.38 * spatial.white_balance_confidence + (0.04 if technical else -0.05), 0.48, 0.94))
        return ValidationItem(
            "white_balance", True, technical, "spatial_white_balance_v1",
            float(error_before * 100.0), float(error_after * 100.0), confidence, regressions, message,
            preview_available=True, auto_eligible=False, technical_passed=technical, adjustable=True,
            default_strength=strength, parameter_label="Нейтрализация", parameters=parameters,
        )

    strength = float(np.clip(advice.neutralization_strength, 0.0, 1.0))
    corrected = apply_white_balance(rgb, advice, strength=strength)
    after = _basic_probe(corrected, _PROBE_WB)
    after_advice = analyze_white_balance(corrected)

    cast_gain = float(advice.cast_strength - after_advice.cast_strength)
    highlight_reg = float(after["highlight_clip_pct"] - before["highlight_clip_pct"])
    shadow_reg = float(after["shadow_clip_pct"] - before["shadow_clip_pct"])
    brightness_delta = float(after["brightness"] - before["brightness"])
    skin_drift = _face_skin_chroma_drift(rgb, corrected, list(spatial.face_regions))
    technical = bool(
        cast_gain >= max(0.025, advice.cast_strength * 0.12)
        and highlight_reg <= 0.8
        and shadow_reg <= 1.5
        and abs(brightness_delta) <= 0.035
        and skin_drift <= 9.5
    )
    regressions = {
        "cast_strength_delta": float(after_advice.cast_strength - advice.cast_strength),
        "highlight_clip_delta_pct": highlight_reg,
        "shadow_clip_delta_pct": shadow_reg,
        "brightness_delta": brightness_delta,
        "skin_chroma_drift_lab": skin_drift,
        "temperature_shift_k": float(advice.recommended_temperature_shift_k),
        "tint_shift": float(advice.recommended_tint_shift),
    }
    message = (
        f"Баланс белого стал нейтральнее без заметного изменения экспозиции. "
        f"Совет: температура {advice.recommended_temperature_shift_k:+.0f} K, tint {advice.recommended_tint_shift:+.1f}, "
        f"нейтрализация {strength * 100:.0f}% с сохранением {advice.atmosphere_preservation * 100:.0f}% атмосферы исходного света."
        if technical else
        "Пробная коррекция баланса белого не дала достаточно безопасного выигрыша; оставлена только для ручного сравнения."
    )
    confidence = float(np.clip(advice.confidence + (0.06 if technical else -0.05), 0.45, 0.95))
    return ValidationItem(
        "white_balance", True, technical, "hybrid_awb_linear_rgb_v1",
        float(advice.cast_strength * 100.0), float(after_advice.cast_strength * 100.0), confidence,
        regressions, message, preview_available=True, auto_eligible=False,
        technical_passed=technical, adjustable=True, default_strength=strength, parameter_label="Нейтрализация",
        parameters=advice.to_raw(),
    )


def _validate_contrast(
    rgb: np.ndarray, metrics: Mapping[str, MetricResult] | None = None,
    local_plan: LocalCorrectionPlan | None = None,
    before_cache: dict[str, float] | None = None,
) -> ValidationItem:
    before = _basic_probe(rgb, _PROBE_CONTRAST, cache=before_cache)
    if metrics is not None:
        spatial = _resolve_local_plan(metrics, local_plan)
        if spatial is not None and len(spatial.contrast_cells) >= 2 and spatial.contrast_confidence >= 0.46:
            parameters = spatial.to_raw()
            strength = 0.82
            corrected = apply_local_contrast(rgb, parameters, strength=strength)
            after = _basic_probe(corrected, _PROBE_CONTRAST)
            range_before = _contrast_plan_range(rgb, list(spatial.contrast_cells))
            range_after = _contrast_plan_range(corrected, list(spatial.contrast_cells))
            gain = range_after - range_before
            highlight_reg = after["highlight_clip_pct"] - before["highlight_clip_pct"]
            shadow_reg = after["shadow_clip_pct"] - before["shadow_clip_pct"]
            outside_delta = _outside_change(rgb, corrected, list(spatial.contrast_cells), "strength")
            technical = bool(
                range_before >= 0.035
                and gain >= 0.004
                and highlight_reg <= 0.55
                and shadow_reg <= 0.80
                and outside_delta <= 0.010
            )
            return ValidationItem(
                "contrast", True, technical, "spatial_contrast_v1", range_before, range_after,
                float(np.clip(0.62 + spatial.contrast_confidence * 0.32 + (0.05 if technical else -0.03), 0.54, 0.93)),
                {
                    "target_local_range_delta": float(gain),
                    "highlight_clip_delta_pct": float(highlight_reg),
                    "shadow_clip_delta_pct": float(shadow_reg),
                    "outside_mean_abs_delta": float(outside_delta),
                    "coverage_pct": float(spatial.contrast_coverage * 100.0),
                },
                (
                    f"Локальный контраст усилен только в согласованных слабоконтрастных областях ({spatial.contrast_coverage * 100:.0f}% кадра); "
                    "остальная часть сохранена."
                    if technical else
                    "Карта нашла слабоконтрастные области, но пространственная проба не дала достаточного безопасного выигрыша; оставлена для ручного сравнения."
                ),
                preview_available=True, auto_eligible=technical, technical_passed=technical,
                adjustable=True, default_strength=strength, parameter_label="Сила", parameters=parameters,
            )

    tested: list[tuple[float, dict[str, float]]] = []
    best: tuple[float, dict[str, float]] | None = None
    for strength in (0.25, 0.40, 0.55, 0.70, 1.00):
        after = _basic_probe(_mild_local_contrast(rgb, strength), _PROBE_CONTRAST)
        tested.append((strength, after))
        gain = after["local_contrast_score"] - before["local_contrast_score"]
        highlight_reg = after["highlight_clip_pct"] - before["highlight_clip_pct"]
        shadow_reg = after["shadow_clip_pct"] - before["shadow_clip_pct"]
        edge_reg = before["edge_score"] - after["edge_score"]
        if gain >= 2.5 and highlight_reg <= 1.0 and shadow_reg <= 1.0 and edge_reg <= 7.0:
            best = (strength, after)
            break
    if best is None:
        strength, after = tested[1] if len(tested) > 1 else tested[0]
        accepted = False
    else:
        strength, after = best
        accepted = True
    regressions = {
        "highlight_clip_delta_pct": after["highlight_clip_pct"] - before["highlight_clip_pct"],
        "shadow_clip_delta_pct": after["shadow_clip_pct"] - before["shadow_clip_pct"],
        "edge_score_delta": after["edge_score"] - before["edge_score"],
        "sharpness_delta": after["sharpness"] - before["sharpness"],
    }
    return ValidationItem(
        "contrast", True, accepted, "mild_lab_clahe", before["local_contrast_score"], after["local_contrast_score"],
        0.74 if accepted else 0.58, regressions,
        "Подобрана минимальная сила локального контраста, которая дала измеримый выигрыш без заметных побочных изменений." if accepted else
        "Даже мягкий локальный контраст не прошёл автоматические ограничения; он доступен только для ручного предпросмотра.",
        adjustable=True, default_strength=float(strength), parameter_label="Сила"
    )


def _validate_red_eye(rgb: np.ndarray, metrics: Mapping[str, MetricResult]) -> ValidationItem:
    payload = metrics.get("red_eye")
    raw = payload.raw_value if payload is not None and isinstance(payload.raw_value, dict) else {}
    candidates = [item for item in raw.get("candidates", []) if isinstance(item, dict) and bool(item.get("suspicious", False))]
    suspicious_before = int(raw.get("suspicious_eye_count", 0) or 0)
    if suspicious_before <= 0 or not candidates:
        return ValidationItem(
            "red_eye", True, False, "localized_pupil_neutralization", 0.0, None, 0.55,
            {}, "Надёжные кандидаты красных глаз не подтверждены; автоматическое применение не требуется.", candidates, False
        )
    corrected, stats = correct_red_eye(rgb, candidates)
    if stats.corrected_count <= 0:
        return ValidationItem(
            "red_eye", True, False, "localized_pupil_neutralization_relaxed", float(suspicious_before), None, 0.58,
            {}, "Строгая пробная маска не прошла проверку безопасности. Для явного ручного предпросмотра доступен более мягкий поиск красного рефлекса строго внутри уже найденных глаз.",
            candidates, True, False, False, adjustable=True, default_strength=0.60, parameter_label="Сила"
        )
    # Rebuild suspicious estimate from the already-known candidate boxes.
    remaining = 0
    before_strength = 0.0
    after_strength = 0.0
    nonred_channel_error = 0.0
    roi_pixel_count = 0
    outside_mask_changed = 0
    total_changed = 0
    lower_eye_changed = 0
    max_changed_fraction = 0.0
    for cand in candidates:
        x = int(round(float(cand.get("x", 0.0)) * rgb.shape[1])) if 0.0 <= float(cand.get("x", 0.0)) <= 1.0 else int(cand.get("x", 0))
        y = int(round(float(cand.get("y", 0.0)) * rgb.shape[0])) if 0.0 <= float(cand.get("y", 0.0)) <= 1.0 else int(cand.get("y", 0))
        w = int(round(float(cand.get("w", 0.0)) * rgb.shape[1])) if 0.0 <= float(cand.get("w", 0.0)) <= 1.0 else int(cand.get("w", 0))
        h = int(round(float(cand.get("h", 0.0)) * rgb.shape[0])) if 0.0 <= float(cand.get("h", 0.0)) <= 1.0 else int(cand.get("h", 0))
        x = max(0, x); y = max(0, y); w = min(w, rgb.shape[1] - x); h = min(h, rgb.shape[0] - y)
        if w < 4 or h < 4:
            continue
        roi_before = rgb[y:y+h, x:x+w].astype(np.float32)
        roi_after = corrected[y:y+h, x:x+w].astype(np.float32)
        def red_strength(roi: np.ndarray) -> float:
            r = roi[..., 0]; g = roi[..., 1]; b = roi[..., 2]
            mean_gb = (g + b) * 0.5
            mask = (r > g * 1.15) & (r > b * 1.15)
            if not np.any(mask):
                return 0.0
            return float(np.mean(np.maximum(0.0, r[mask] - mean_gb[mask])))
        b_strength = red_strength(roi_before)
        a_strength = red_strength(roi_after)
        before_strength += b_strength
        after_strength += a_strength
        # V2+ is deliberately red-channel-only. Any meaningful G/B drift means
        # collateral recolouring and must block automatic acceptance.
        if roi_before.size and roi_after.size:
            gb_delta = np.abs(roi_after[..., 1:3] - roi_before[..., 1:3])
            nonred_channel_error += float(gb_delta.sum())
            roi_pixel_count += int(gb_delta.size)

            allowed_mask = build_red_eye_mask(np.clip(np.rint(roi_before), 0, 255).astype(np.uint8))
            red_delta = np.abs(roi_after[..., 0] - roi_before[..., 0])
            changed = red_delta >= 0.75
            changed_count = int(changed.sum())
            total_changed += changed_count
            if changed_count:
                outside_mask_changed += int((changed & ~allowed_mask).sum())
                lower_start = int(round(h * 0.68))
                lower_eye_changed += int(changed[max(0, lower_start):, :].sum())
                max_changed_fraction = max(max_changed_fraction, changed_count / float(max(w * h, 1)))
        if a_strength >= max(16.0, b_strength * 0.62):
            remaining += 1
    nonred_mae = nonred_channel_error / max(roi_pixel_count, 1)
    outside_change_ratio = outside_mask_changed / max(total_changed, 1)
    lower_change_ratio = lower_eye_changed / max(total_changed, 1)
    brightness_delta = float(corrected.mean() - rgb.mean())
    accepted = (
        remaining < suspicious_before
        and after_strength <= before_strength * 0.74
        and nonred_mae <= 0.75
        and outside_change_ratio <= 0.002
        and lower_change_ratio <= 0.10
        and max_changed_fraction <= 0.12
        and abs(brightness_delta) <= 1.5
    )
    regressions = {
        "remaining_suspicious": float(remaining),
        "corrected_eyes": float(stats.corrected_count),
        "red_strength_delta": after_strength - before_strength,
        "brightness_delta": brightness_delta,
        "nonred_channel_mae": float(nonred_mae),
        "outside_pupil_change_ratio": float(outside_change_ratio),
        "lower_eye_change_ratio": float(lower_change_ratio),
        "max_eye_changed_fraction": float(max_changed_fraction),
    }
    confidence = float(np.clip((payload.confidence if payload is not None else 0.6) + (0.10 if accepted else -0.04), 0.50, 0.95))
    regressions["source_candidates"] = float(len(candidates))
    return ValidationItem(
        "red_eye", True, accepted, "localized_pupil_neutralization", float(suspicious_before), float(remaining), confidence,
        regressions,
        "Красный рефлекс локально нейтрализован с сохранением цвета радужки и блика." if accepted else
        "Пробная коррекция красных глаз не показала достаточно надёжного выигрыша; автоматическое применение не подтверждено.",
        candidates,
        adjustable=True, default_strength=0.72, parameter_label="Сила"
    )


def _sharpness_plan_score(rgb: np.ndarray, cells: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    values: list[float] = []
    weights: list[float] = []
    h, w = gray.shape
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        box = _normalized_box_slice(cell, h, w)
        if box is None:
            continue
        tile = gray[box]
        if tile.size < 256:
            continue
        lap = float(cv2.Laplacian(tile, cv2.CV_64F).var())
        gx = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3)
        ten = float(np.mean(gx * gx + gy * gy))
        lap_score = float(np.clip(20.0 * np.log10(1.0 + lap), 0.0, 100.0))
        ten_score = float(np.clip(14.0 * np.log10(1.0 + ten), 0.0, 100.0))
        values.append(0.45 * lap_score + 0.55 * ten_score)
        weights.append(float(tile.size))
    if not values:
        return 0.0
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))


def _noise_plan_sigma(rgb: np.ndarray, cells: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    guide = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    flat_mask = cv2.magnitude(gx, gy) <= 12.0
    kernel = np.asarray([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)
    response = np.abs(cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101))
    scale = float(np.sqrt(np.pi / 2.0) / 6.0)
    values: list[float] = []
    weights: list[float] = []
    h, w = gray.shape
    for cell in cells:
        if not isinstance(cell, Mapping):
            continue
        box = _normalized_box_slice(cell, h, w)
        if box is None:
            continue
        mask = flat_mask[box]
        if mask.size < 128 or float(mask.mean()) < 0.16:
            continue
        sample = response[box][mask]
        if sample.size < 96:
            continue
        cap = float(np.percentile(sample, 98.0))
        sample = sample[sample <= cap]
        if sample.size < 64:
            continue
        values.append(scale * float(sample.mean()))
        weights.append(float(sample.size))
    if not values:
        return 0.0
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))


def _validate_sharpness(
    rgb: np.ndarray, metrics: Mapping[str, MetricResult] | None = None,
    local_plan: LocalCorrectionPlan | None = None,
    before_cache: dict[str, float] | None = None,
) -> ValidationItem:
    before = _basic_probe(rgb, _PROBE_DETAIL, cache=before_cache)
    if metrics is not None:
        spatial = _resolve_local_plan(metrics, local_plan)
        if spatial is not None and len(spatial.sharpness_cells) >= 2 and spatial.sharpness_confidence >= 0.48:
            parameters = spatial.to_raw()
            strength = 0.82
            corrected = apply_local_sharpness(rgb, parameters, strength=strength)
            after = _basic_probe(corrected, _PROBE_DETAIL)
            target_before = _sharpness_plan_score(rgb, list(spatial.sharpness_cells))
            target_after = _sharpness_plan_score(corrected, list(spatial.sharpness_cells))
            gain = target_after - target_before
            outside_delta = _outside_change(rgb, corrected, list(spatial.sharpness_cells), "strength")
            noise_delta = after["noise_score"] - before["noise_score"]
            edge_delta = after["edge_score"] - before["edge_score"]
            technical = bool(
                target_before > 0.0
                and gain >= 0.25
                and outside_delta <= 0.006
                and noise_delta >= -3.5
                and edge_delta >= -5.0
            )
            return ValidationItem(
                "sharpness", True, technical, "spatial_sharpness_v1", target_before, target_after,
                float(np.clip(0.61 + spatial.sharpness_confidence * 0.32 + (0.05 if technical else -0.04), 0.52, 0.93)),
                {
                    "target_sharpness_delta": float(gain),
                    "outside_mean_abs_delta": float(outside_delta),
                    "noise_score_delta": float(noise_delta),
                    "edge_score_delta": float(edge_delta),
                    "coverage_pct": float(spatial.sharpness_coverage * 100.0),
                },
                (
                    f"Резкость усилена только в согласованных мягких областях ({spatial.sharpness_coverage * 100:.0f}% кадра); шум, лица и глаза защищены ограничителями."
                    if technical else
                    "Карта мягкости есть, но локальная проба резкости не прошла ограничения по шуму/ореолам или не дала достаточного выигрыша; оставлена для ручного сравнения."
                ),
                preview_available=True, auto_eligible=technical, technical_passed=technical,
                adjustable=True, default_strength=strength, parameter_label="Сила", parameters=parameters,
            )

    strength = 0.35
    after = _basic_probe(_edge_aware_sharpen(rgb, strength), _PROBE_DETAIL)
    gain = after["sharpness"] - before["sharpness"]
    edge_delta = after["edge_score"] - before["edge_score"]
    noise_delta = after["noise_score"] - before["noise_score"]
    return ValidationItem(
        "sharpness", True, False, "edge_aware_unsharp", before["sharpness"], after["sharpness"], 0.58,
        {"sharpness_delta": gain, "edge_score_delta": edge_delta, "noise_score_delta": noise_delta},
        "Доступно только как ручная проба: мягкая резкость усиливает уже существующие детали, но не восстанавливает отсутствующие. Лица и глаза обязательно сравнивать в 100%.",
        preview_available=True, auto_eligible=False, technical_passed=(gain > 0.8 and edge_delta > -10.0),
        adjustable=True, default_strength=strength, parameter_label="Сила"
    )


def _validate_noise(
    rgb: np.ndarray, metrics: Mapping[str, MetricResult] | None = None,
    local_plan: LocalCorrectionPlan | None = None,
    before_cache: dict[str, float] | None = None,
) -> ValidationItem:
    if metrics is not None:
        spatial = _resolve_local_plan(metrics, local_plan)
        if spatial is not None and len(spatial.noise_cells) >= 2 and spatial.noise_confidence >= 0.48:
            parameters = spatial.to_raw()
            strength = 0.84
            before = _basic_probe(rgb, _PROBE_DETAIL, cache=before_cache)
            target_before = _noise_plan_sigma(rgb, list(spatial.noise_cells))
            corrected = apply_local_denoise(rgb, parameters, strength=strength)
            after = _basic_probe(corrected, _PROBE_DETAIL)
            target_after = _noise_plan_sigma(corrected, list(spatial.noise_cells))
            improvement = target_before - target_after
            outside_delta = _outside_change(rgb, corrected, list(spatial.noise_cells), "strength")
            sharp_delta = after["sharpness"] - before["sharpness"]
            edge_delta = after["edge_score"] - before["edge_score"]
            technical = bool(
                target_before >= 2.4
                and improvement >= max(0.08, target_before * 0.025)
                and outside_delta <= 0.006
                and sharp_delta >= -4.0
                and edge_delta >= -4.5
            )
            return ValidationItem(
                "noise", True, technical, "spatial_denoise_v1", target_before, target_after,
                float(np.clip(0.61 + spatial.noise_confidence * 0.32 + (0.05 if technical else -0.04), 0.52, 0.93)),
                {
                    "target_sigma_delta": float(target_after - target_before),
                    "outside_mean_abs_delta": float(outside_delta),
                    "sharpness_delta": float(sharp_delta),
                    "edge_score_delta": float(edge_delta),
                    "coverage_pct": float(spatial.noise_coverage * 100.0),
                },
                (
                    f"Шум уменьшен только в подтверждённых слаботекстурных областях ({spatial.noise_coverage * 100:.0f}% кадра); сильные границы, лица и глаза защищены."
                    if technical else
                    "Локальный шум найден, но пробное шумоподавление не прошло защиту детализации или не дало достаточного выигрыша; оставлено для ручного сравнения."
                ),
                preview_available=True, auto_eligible=technical, technical_passed=technical,
                adjustable=True, default_strength=strength, parameter_label="Сила", parameters=parameters,
            )
    return _validate_manual_filter(rgb, "noise")


def _surface_preview_candidates(metrics: Mapping[str, MetricResult]) -> list[dict[str, Any]]:
    # Executable repair targets must come from the classical detector (or later
    # explicit user choices in the GUI).  surface_refinement is advisory-only and
    # must never silently decide which candidates will be treated.
    surface_metric = metrics.get("surface_defects")
    raw = surface_metric.raw_value if surface_metric is not None and isinstance(surface_metric.raw_value, dict) else {}
    classical = raw.get("boxes_norm", []) if isinstance(raw, dict) else []
    # No confident AI confirmation: keep the repair strictly manual and bounded to
    # a small number of the strongest bright candidates shown by the classical map.
    fallback = [dict(box) for box in classical if isinstance(box, dict) and str(box.get("polarity", "bright")) == "bright"]
    if not fallback and isinstance(classical, list):
        fallback = [dict(box) for box in classical if isinstance(box, dict)]
    return fallback[:8]


def _validate_surface_defects(rgb: np.ndarray, metrics: Mapping[str, MetricResult]) -> ValidationItem:
    candidates = _surface_preview_candidates(metrics)
    if not candidates:
        return ValidationItem(
            "surface_defects", True, False, "bounded_surface_heal", None, None, 0.45, {},
            "Кандидаты обнаружены, но ни один не подходит даже для ограниченной пробной коррекции. Сначала подтвердите дефект во вкладке «Дефекты».",
            [], False, False, False, adjustable=True, default_strength=0.45, parameter_label="Сила"
        )
    strength = 0.45
    healed = _bounded_surface_heal(rgb, candidates, strength)
    changed = float(np.mean(np.any(healed != rgb, axis=2)) * 100.0)
    return ValidationItem(
        "surface_defects", True, False, "bounded_surface_heal", None, None, 0.52,
        {"changed_pixel_pct": changed, "candidate_count": float(len(candidates))},
        f"Ручная проба лечит только {len(candidates)} наиболее обоснованных локальных кандидатов, а не все найденные линии. Обязательно проверить волосы, швы и контуры в 100%.",
        candidates, True, False, changed <= 3.0, adjustable=True, default_strength=strength, parameter_label="Сила"
    )



def _validate_surface_defects_deferred(metrics: Mapping[str, MetricResult]) -> ValidationItem:
    """Cheap pre-AI placeholder for the main analysis pipeline.

    The executable Surface decision is rebuilt after Surface AI/refinement by
    ``refresh_surface_validation_after_refinement``.  Rendering the classical
    first-eight trial here only to discard it later is expensive on 4K images.
    Keep the same pre-AI confidence/candidate payload needed by the surrounding
    validator, but defer mask construction/inpaint until the final refined set is
    known.  Standalone ``validate_recommendations`` keeps the original full probe
    unless the caller explicitly opts into deferral.
    """
    candidates = _surface_preview_candidates(metrics)
    if not candidates:
        return ValidationItem(
            "surface_defects", True, False, "bounded_surface_heal", None, None, 0.45, {},
            "Кандидаты обнаружены, но ни один не подходит даже для ограниченной пробной коррекции. Сначала подтвердите дефект во вкладке «Дефекты».",
            [], False, False, False, adjustable=True, default_strength=0.45, parameter_label="Сила"
        )
    return ValidationItem(
        "surface_defects", True, False, "bounded_surface_heal", None, None, 0.52,
        {"changed_pixel_pct": 0.0, "candidate_count": float(len(candidates))},
        "Предварительная Surface-проба отложена до post-AI уточнения; финальная маска будет проверена по подтверждённым кандидатам.",
        candidates, True, False, False, adjustable=True, default_strength=0.45, parameter_label="Сила"
    )

def _validate_super_resolution(
    rgb: np.ndarray, metrics: Mapping[str, MetricResult], before_cache: dict[str, float] | None = None
) -> ValidationItem:
    """Prepare a lazy ALMAZ x2 candidate without rendering the expensive preview.

    Ordinary analysis must stay cheap.  The actual x2 transform is only executed
    when the user explicitly asks for corrected preview/save-copy via
    ``apply_selected_preview``.
    """
    metric = metrics.get("super_resolution")
    raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, Mapping) else {}
    strength = float(np.clip(float(raw.get("strength", 0.72) or 0.72), 0.0, 1.0))
    face_protection = float(np.clip(float(raw.get("face_protection", 0.85) or 0.85), 0.0, 1.0))
    deblock_strength = float(np.clip(0.10 + 0.45 * float(raw.get("jpeg_need", 0.0) or 0.0), 0.0, 0.45))
    denoise_strength = float(np.clip(0.06 + 0.32 * float(raw.get("noise_need", 0.0) or 0.0), 0.0, 0.32))
    detail_strength = float(np.clip(0.42 + 0.32 * float(raw.get("detail_deficit", 0.0) or 0.0), 0.25, 0.76))
    faces_metric = metrics.get("faces")
    faces_raw = faces_metric.raw_value if faces_metric is not None and isinstance(faces_metric.raw_value, Mapping) else {}
    params = {
        "backend": "auto_swinir_or_classical",
        "backend_preference": "auto",
        "suggested_scale": 2,
        "face_protection": face_protection,
        "deblock_strength": deblock_strength,
        "denoise_strength": denoise_strength,
        "detail_strength": detail_strength,
        "face_boxes": faces_raw.get("faces", []),
    }
    recommend = bool(raw.get("recommend", False))
    need_score = float(raw.get("need_score", 0.0) or 0.0)
    long_edge = int(raw.get("long_edge_px", max(rgb.shape[:2])) or max(rgb.shape[:2]))
    megapixels = float(raw.get("megapixels", (rgb.shape[0] * rgb.shape[1]) / 1_000_000.0) or 0.0)
    technical = bool(recommend and int(raw.get("suggested_scale", 2) or 2) == 2 and long_edge <= 2600 and megapixels <= 5.5)
    regressions = {
        "lazy_preview": 1.0,
        "need_score": need_score,
        "face_protection": face_protection,
    }
    message = (
        "ALMAZ x2 рекомендован. Тяжёлое увеличение не считается во время анализа: оно будет построено только при явном предпросмотре/сохранении. "
        "Лица защищаются отдельным identity-guard, а результат следует оценивать в 100%."
        if technical else
        "ALMAZ x2 доступен только как ручной эксперимент: автоматическая рекомендация не прошла быстрые входные ограничения."
    )
    confidence = float(np.clip(float(raw.get("confidence", 0.64) or 0.64), 0.50, 0.90))
    return ValidationItem(
        "super_resolution", True, False, "almaz_x2_identity_guard_v1",
        need_score, None, confidence, regressions, message,
        preview_available=True, auto_eligible=False, technical_passed=technical,
        adjustable=True, default_strength=strength, parameter_label="Сила восстановления", parameters=params,
    )


def _validate_manual_filter(
    rgb: np.ndarray, action_key: str, before_cache: dict[str, float] | None = None
) -> ValidationItem:
    before = _basic_probe(rgb, _PROBE_DETAIL, cache=before_cache)
    configs = {
        "noise": ("mild_nlm", _mild_denoise, 0.30, "Шумоподавление доступно только вручную: волосы, ткань и мелкие детали проверять в 100%."),
        "jpeg_artifacts": ("mild_deblock", _mild_deblock, 0.30, "Мягкое сглаживание блоков доступно только вручную; оно может уменьшить мелкую фактуру."),
        "edge_artifacts": ("edge_halo_soften", _mild_edge_soften, 0.25, "Смягчение ореолов действует только на сильные границы и доступно для ручного сравнения."),
        "posterization": ("mild_deband", _mild_deband, 0.25, "Мягкое сглаживание полосатости градаций использует минимальный детерминированный дизеринг; оценивать на гладких градиентах в 100%."),
    }
    candidate, func, strength, message = configs[action_key]
    after = _basic_probe(func(rgb, strength), _PROBE_DETAIL)
    regressions = {
        "sharpness_delta": after["sharpness"] - before["sharpness"],
        "noise_score_delta": after["noise_score"] - before["noise_score"],
        "edge_score_delta": after["edge_score"] - before["edge_score"],
    }
    return ValidationItem(
        action_key, True, False, candidate, None, None, 0.52, regressions, message,
        preview_available=True, auto_eligible=False, technical_passed=True,
        adjustable=True, default_strength=strength, parameter_label="Сила"
    )



def _almaz_restoration_task_for_action(
    action_key: str, metrics: Mapping[str, MetricResult]
) -> str | None:
    if action_key == "noise":
        return "denoise"
    if action_key == "jpeg_artifacts":
        return "jpeg_recovery"
    if action_key != "sharpness":
        return None
    detail_metric = metrics.get("detail_loss_type")
    raw = detail_metric.raw_value if detail_metric is not None and isinstance(detail_metric.raw_value, Mapping) else {}
    kind = str(raw.get("classification", "unknown"))
    if kind in {
        "camera_shake_like", "subject_motion_like", "motion_like", "defocus_like",
        "mixed", "mixed_or_degraded", "degradation_like",
    }:
        return "deblur"
    return None


def _metric_score_for_almaz(metrics: Mapping[str, MetricResult], key: str, default: float) -> float:
    metric = metrics.get(key)
    if metric is None or metric.normalized_value is None:
        return float(default)
    try:
        value = float(metric.normalized_value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _decision_severity_for_almaz(metrics: Mapping[str, MetricResult], action_key: str) -> float:
    metric = metrics.get("decision_plan")
    raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, Mapping) else {}
    items = raw.get("items", []) if isinstance(raw, Mapping) else []
    if isinstance(items, list):
        for row in items:
            if not isinstance(row, Mapping) or str(row.get("key", "")) != action_key:
                continue
            try:
                value = float(row.get("severity", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                return 0.0
            return float(np.clip(value, 0.0, 100.0)) if np.isfinite(value) else 0.0
    return 0.0


def _adaptive_almaz_restoration_strength(
    task: str, action_key: str, metrics: Mapping[str, MetricResult], *, has_faces: bool
) -> tuple[float, float]:
    """Choose a conservative preview strength from measured degradation.

    This is still a recommendation, never automatic application.  The goal is
    merely to avoid using one magic strength for a mildly noisy phone photo and
    a badly compressed old 1-MP frame.
    """
    severity = _decision_severity_for_almaz(metrics, action_key) / 100.0
    if task == "denoise":
        score = _metric_score_for_almaz(metrics, "noise", 70.0)
        measured = float(np.clip((72.0 - score) / 38.0, 0.0, 1.0))
        degradation = max(measured, severity * 0.90)
        strength = 0.46 + 0.28 * degradation
        if has_faces:
            strength -= 0.04
        return float(np.clip(strength, 0.42, 0.74)), degradation
    if task == "deblur":
        values = [
            _metric_score_for_almaz(metrics, "local_sharpness", 70.0),
            _metric_score_for_almaz(metrics, "faces", 70.0) if has_faces else 100.0,
            _metric_score_for_almaz(metrics, "eyes", 70.0) if has_faces else 100.0,
        ]
        score = min(values)
        measured = float(np.clip((70.0 - score) / 38.0, 0.0, 1.0))
        degradation = max(measured, severity * 0.85)
        # The stock NAFNet-GoPro checkpoint is intentionally conservative here:
        # it is trained for GoPro-style motion blur, not arbitrary soft photos.
        strength = 0.26 + 0.20 * degradation
        if has_faces:
            strength -= 0.035
        return float(np.clip(strength, 0.22, 0.46)), degradation
    # jpeg_recovery
    score = _metric_score_for_almaz(metrics, "jpeg_artifacts", 70.0)
    measured = float(np.clip((72.0 - score) / 40.0, 0.0, 1.0))
    degradation = max(measured, severity * 0.90)
    strength = 0.40 + 0.29 * degradation
    if has_faces:
        strength -= 0.035
    return float(np.clip(strength, 0.36, 0.69)), degradation


def _prefer_almaz_restoration_if_available(
    action_key: str,
    item: ValidationItem,
    metrics: Mapping[str, MetricResult],
) -> ValidationItem:
    task = _almaz_restoration_task_for_action(action_key, metrics)
    if task is None:
        return item
    try:
        status = inspect_installed_restoration_model(task)
    except Exception:
        return item
    if not status.ready:
        return item

    faces_metric = metrics.get("faces")
    faces_raw = faces_metric.raw_value if faces_metric is not None and isinstance(faces_metric.raw_value, Mapping) else {}
    face_boxes = faces_raw.get("faces", []) if isinstance(faces_raw, Mapping) else []
    if not isinstance(face_boxes, list):
        face_boxes = []

    candidate_by_task = {
        "denoise": "almaz_ai_denoise_v1",
        "deblur": "almaz_ai_deblur_v1",
        "jpeg_recovery": "almaz_ai_jpeg_recovery_v1",
    }
    has_faces = bool(face_boxes)
    adaptive_strength, degradation_score = _adaptive_almaz_restoration_strength(
        task, action_key, metrics, has_faces=has_faces
    )
    protection_by_task = {"denoise": 0.80, "deblur": 0.92, "jpeg_recovery": 0.88}
    label_by_task = {
        "denoise": "AI Denoise",
        "deblur": "AI Deblur",
        "jpeg_recovery": "AI JPEG/Blur Recovery",
    }
    params = {
        "backend": "almaz_onnx_x1",
        "task": task,
        "model_id": status.model_id,
        "provider": status.provider,
        "provider_label": status.provider_label,
        "face_protection": protection_by_task[task],
        "face_boxes": face_boxes,
        "adaptive_strength": adaptive_strength,
        "degradation_score": degradation_score,
    }
    regressions = dict(item.regressions or {})
    regressions["almaz_model_ready"] = 1.0
    return ValidationItem(
        action_key=action_key,
        tested=True,
        accepted=False,
        candidate=candidate_by_task[task],
        target_before=item.target_before,
        target_after=None,
        confidence=float(np.clip(max(item.confidence, 0.64), 0.0, 0.90)),
        regressions=regressions,
        message=(
            f"ALMAZ {label_by_task[task]} доступен через проверенную модель {status.model_id} "
            f"({status.provider_label or status.provider}). Тяжёлый inference выполняется только по явному предпросмотру; "
            f"Рекомендуемая сила {adaptive_strength * 100:.0f}% выбрана по измеренной деградации; "
            "до калибровки на реальных старых фото автоматическое применение запрещено."
        ),
        source_candidates=item.source_candidates,
        preview_available=True,
        auto_eligible=False,
        technical_passed=True,
        adjustable=True,
        default_strength=adaptive_strength,
        parameter_label="Сила ALMAZ",
        parameters=params,
    )

def _standalone_almaz_restoration_item(
    task: str, metrics: Mapping[str, MetricResult]
) -> ValidationItem | None:
    """Expose one ALMAZ x1 model as an independent manual correction."""
    if task not in {"denoise", "deblur", "jpeg_recovery"}:
        return None
    try:
        status = inspect_installed_restoration_model(task)
    except Exception:
        return None
    if not status.ready:
        return None
    source_key = {"denoise": "noise", "deblur": "sharpness", "jpeg_recovery": "jpeg_artifacts"}[task]
    action_key = {"denoise": "almaz_denoise", "deblur": "almaz_deblur", "jpeg_recovery": "almaz_jpeg_recovery"}[task]
    candidate = {"denoise": "almaz_ai_denoise_v1", "deblur": "almaz_ai_deblur_v1", "jpeg_recovery": "almaz_ai_jpeg_recovery_v1"}[task]
    label = {"denoise": "AI Denoise", "deblur": "AI Deblur", "jpeg_recovery": "JPEG Recovery"}[task]
    protection = {"denoise": 0.80, "deblur": 0.92, "jpeg_recovery": 0.88}[task]
    faces_metric = metrics.get("faces")
    faces_raw = faces_metric.raw_value if faces_metric is not None and isinstance(faces_metric.raw_value, Mapping) else {}
    face_boxes = faces_raw.get("faces", []) if isinstance(faces_raw, Mapping) else []
    if not isinstance(face_boxes, list):
        face_boxes = []
    strength, degradation = _adaptive_almaz_restoration_strength(task, source_key, metrics, has_faces=bool(face_boxes))
    source_score = _metric_score_for_almaz(metrics, source_key, 70.0)
    params = {
        "backend": "almaz_onnx_x1", "task": task, "source_action_key": source_key,
        "model_id": status.model_id, "provider": status.provider, "provider_label": status.provider_label,
        "face_protection": protection, "face_boxes": face_boxes,
        "adaptive_strength": strength, "degradation_score": degradation,
    }
    return ValidationItem(
        action_key=action_key, tested=True, accepted=False, candidate=candidate,
        target_before=source_score, target_after=None,
        confidence=float(np.clip(0.62 + 0.18 * degradation, 0.62, 0.84)),
        regressions={"almaz_model_ready": 1.0, "degradation_score": degradation},
        message=(
            f"ALMAZ {label} — отдельный ручной модуль через модель {status.model_id} "
            f"({status.provider_label or status.provider}). Его можно включать независимо от обычных "
            "шумоподавления/резкости/JPEG-коррекции для покадрового сравнения и диагностики артефактов."
            + (" Базовая Deblur-модель обучена на GoPro motion blur; до нашего fine-tune используется "
               "пониженная сила и усиленная защита цветности." if task == "deblur" else "")
        ),
        preview_available=True, auto_eligible=False, technical_passed=True, adjustable=True,
        default_strength=strength, parameter_label="Сила ALMAZ", parameters=params,
    )


def validate_recommendations(
    image: LoadedImage,
    metrics: Mapping[str, MetricResult],
    *,
    precision: str = "normal",
    defer_surface_refinement: bool = False,
) -> list[ValidationItem]:
    profile = get_precision(precision)
    validation_long_edge = 4096 if profile.key == "maximum" else (2048 if profile.deep_analysis else 1024)
    rgb = _technical_copy(image.srgb, long_edge=validation_long_edge)
    actions = _decision_actions(metrics)
    # WB/Auto Tone need a richer colour sample than the ordinary Normal-mode
    # 1024px validator copy, but never need a 24–50 MP full-frame transform just
    # to decide whether a correction is safe. Create that copy only when needed.
    colour_validation_rgb: np.ndarray | None = None

    def colour_probe_rgb() -> np.ndarray:
        nonlocal colour_validation_rgb
        if colour_validation_rgb is None:
            colour_validation_rgb = rgb if profile.deep_analysis else _technical_copy(image.srgb, long_edge=2048)
        return colour_validation_rgb

    items: list[ValidationItem] = []
    local_plan = _resolve_local_plan(metrics) if any(
        key in actions for key in ("white_balance", "exposure", "contrast", "sharpness", "noise")
    ) else None

    # Build immutable before-caches once.  In serial mode this is equivalent to
    # the old lazy shared cache.  In parallel mode worker threads only read these
    # dictionaries, avoiding locks and duplicate full-frame probes.
    rgb_before_cache: dict[str, float] = {}
    colour_before_cache: dict[str, float] = rgb_before_cache if profile.deep_analysis else {}
    rgb_need: set[str] = set()
    if "exposure" in actions:
        rgb_need.update(_PROBE_EXPOSURE)
    if "contrast" in actions:
        rgb_need.update(_PROBE_CONTRAST)
    if "sharpness" in actions or "noise" in actions or any(
        key in actions for key in ("jpeg_artifacts", "edge_artifacts", "posterization")
    ):
        rgb_need.update(_PROBE_DETAIL)
    if "white_balance" in actions and profile.deep_analysis:
        rgb_need.update(_PROBE_WB)
    if rgb_need:
        _basic_probe(rgb, rgb_need, cache=rgb_before_cache)

    if "white_balance" in actions and not profile.deep_analysis:
        colour_rgb = colour_probe_rgb()
        _basic_probe(colour_rgb, _PROBE_WB, cache=colour_before_cache)

    def finish_item(key: str, item: ValidationItem) -> ValidationItem:
        decision = actions.get(key, "")
        if item.technical_passed is None:
            item.technical_passed = bool(item.accepted)
        if decision != "fix":
            # A review recommendation may be previewed explicitly by the user,
            # but it must never become an automatic correction merely because
            # the bounded technical probe happened to look acceptable.
            item.auto_eligible = False
            item.accepted = False
            if item.preview_available:
                item.message = (
                    "Модуль решений оставил эту задачу для ручной проверки. "
                    "Пробная коррекция доступна только для явного ручного предпросмотра. " + item.message
                )
        return item

    Task = tuple[str, Callable[[], ValidationItem]]
    tasks: list[Task] = []
    if "red_eye" in actions:
        tasks.append(("red_eye", lambda: _validate_red_eye(rgb, metrics)))
    if "super_resolution" in actions:
        tasks.append(("super_resolution", lambda: _validate_super_resolution(rgb, metrics, rgb_before_cache)))
    if "white_balance" in actions:
        wb_validation_rgb = colour_probe_rgb()
        wb_cache = colour_before_cache if wb_validation_rgb is not rgb else rgb_before_cache
        tasks.append(("white_balance", lambda arr=wb_validation_rgb, cache=wb_cache: _validate_white_balance(arr, metrics, local_plan, cache)))
    if "auto_tone_color" in actions:
        atc_validation_rgb = colour_probe_rgb()
        tasks.append(("auto_tone_color", lambda arr=atc_validation_rgb: _validate_auto_tone_color(arr)))
    if "exposure" in actions:
        tasks.append(("exposure", lambda: _validate_exposure(rgb, metrics, local_plan, rgb_before_cache)))
    if "contrast" in actions:
        tasks.append(("contrast", lambda: _validate_contrast(rgb, metrics, local_plan, rgb_before_cache)))
    if "sharpness" in actions:
        tasks.append(("sharpness", lambda: _validate_sharpness(rgb, metrics, local_plan, rgb_before_cache)))
    if "surface_defects" in actions:
        surface_validator = (
            (lambda: _validate_surface_defects_deferred(metrics))
            if defer_surface_refinement else
            (lambda: _validate_surface_defects(rgb, metrics))
        )
        tasks.append(("surface_defects", surface_validator))
    if "noise" in actions:
        tasks.append(("noise", lambda: _validate_noise(rgb, metrics, local_plan, rgb_before_cache)))
    for key in ("jpeg_artifacts", "edge_artifacts", "posterization"):
        if key in actions:
            tasks.append((key, lambda action_key=key: _validate_manual_filter(rgb, action_key, rgb_before_cache)))

    validator_cv_threads, validator_workers = get_validator_profile()
    # NLM is unusually memory-hungry. If denoising itself is among the actions,
    # keep at most two concurrent Validator tasks even when the common profile
    # selected more workers. This preserves the same math while avoiding RAM
    # pressure on large photos.
    effective_workers = min(validator_workers, 2) if "noise" in actions else validator_workers
    serial_cv_threads = max(1, int(cv2.getNumThreads()))
    if effective_workers >= 2 and len(tasks) >= 2:
        # Validators are independent bounded previews of the same immutable input.
        # Reuse the CPU profile selected at startup: fewer OpenCV threads per task
        # and a limited worker count.  This changes scheduling only, never formulas,
        # image size, thresholds or action order.
        cv2.setNumThreads(validator_cv_threads)
        try:
            with ThreadPoolExecutor(max_workers=min(effective_workers, len(tasks)), thread_name_prefix="pd-validator") as pool:
                futures = [pool.submit(func) for _, func in tasks]
                for (key, _), future in zip(tasks, futures):
                    items.append(finish_item(key, future.result()))
        finally:
            cv2.setNumThreads(serial_cv_threads)
    else:
        for key, func in tasks:
            items.append(finish_item(key, func()))

    # Keep ALMAZ x1 modules separately selectable instead of replacing the
    # ordinary Noise/Sharpness/JPEG actions. They are manual-only by design.
    for almaz_task in ("denoise", "deblur", "jpeg_recovery"):
        standalone = _standalone_almaz_restoration_item(almaz_task, metrics)
        if standalone is not None:
            items.append(standalone)
    return items


def preview_action_available(item: Mapping[str, Any]) -> bool:
    """Return whether a Validator item has an executable bounded preview transform."""
    if not isinstance(item, Mapping) or not bool(item.get("tested", True)):
        return False
    if item.get("preview_available") is False:
        return False
    key = str(item.get("action_key", ""))
    candidate = str(item.get("candidate", ""))
    if key == "super_resolution":
        return candidate == "almaz_x2_identity_guard_v1" and isinstance(item.get("parameters"), Mapping)
    if key == "almaz_denoise":
        return candidate == "almaz_ai_denoise_v1" and isinstance(item.get("parameters"), Mapping)
    if key == "almaz_deblur":
        return candidate == "almaz_ai_deblur_v1" and isinstance(item.get("parameters"), Mapping)
    if key == "almaz_jpeg_recovery":
        return candidate == "almaz_ai_jpeg_recovery_v1" and isinstance(item.get("parameters"), Mapping)
    if key == "red_eye":
        return candidate in {"localized_pupil_neutralization", "localized_pupil_neutralization_relaxed"} and isinstance(item.get("source_candidates"), list) and bool(item.get("source_candidates"))
    if key == "exposure":
        if candidate == "spatial_exposure_v1":
            return isinstance(item.get("parameters"), Mapping)
        if not candidate.startswith("gamma="):
            return False
        try:
            gamma_value = float(candidate.split("=", 1)[1])
        except (TypeError, ValueError):
            return False
        return 0.55 <= gamma_value <= 1.05
    if key == "white_balance":
        return candidate in {"hybrid_awb_linear_rgb_v1", "spatial_white_balance_v1"} and isinstance(item.get("parameters"), Mapping)
    if key == "auto_tone_color":
        return candidate == "reference_auto_tone_contrast_color_v1"
    if key == "contrast":
        if candidate == "spatial_contrast_v1":
            return isinstance(item.get("parameters"), Mapping)
        return candidate == "mild_lab_clahe"
    if key == "sharpness":
        if candidate == "almaz_ai_deblur_v1":
            return isinstance(item.get("parameters"), Mapping)
        if candidate == "spatial_sharpness_v1":
            return isinstance(item.get("parameters"), Mapping)
        return candidate == "edge_aware_unsharp"
    if key == "surface_defects":
        return candidate == "bounded_surface_heal" and isinstance(item.get("source_candidates"), list) and bool(item.get("source_candidates"))
    if key == "noise":
        if candidate == "almaz_ai_denoise_v1":
            return isinstance(item.get("parameters"), Mapping)
        if candidate == "spatial_denoise_v1":
            return isinstance(item.get("parameters"), Mapping)
        return candidate == "mild_nlm"
    if key == "jpeg_artifacts":
        if candidate == "almaz_ai_jpeg_recovery_v1":
            return isinstance(item.get("parameters"), Mapping)
        return candidate == "mild_deblock"
    if key == "edge_artifacts":
        return candidate == "edge_halo_soften"
    if key == "posterization":
        return candidate == "mild_deband"
    return False


def _blend_correction_region(
    before: np.ndarray, after: np.ndarray, region: Mapping[str, float] | None
) -> np.ndarray:
    """Blend a correction into one normalized user-selected rectangle with a soft edge.

    A malformed *explicit* local region is a fail-closed condition: returning the
    globally corrected image would turn a local user request into a whole-frame edit.
    """
    if not isinstance(region, Mapping):
        return after
    h, w = before.shape[:2]
    try:
        x = float(region.get("x", 0.0)); y = float(region.get("y", 0.0))
        rw = float(region.get("w", 0.0)); rh = float(region.get("h", 0.0))
    except (TypeError, ValueError, OverflowError):
        return before.copy()
    if not np.isfinite([x, y, rw, rh]).all() or rw <= 0.0 or rh <= 0.0:
        return before.copy()
    try:
        x0 = max(0, min(w, int(round(x * w))))
        y0 = max(0, min(h, int(round(y * h))))
        x1 = max(0, min(w, int(round((x + rw) * w))))
        y1 = max(0, min(h, int(round((y + rh) * h))))
    except (TypeError, ValueError, OverflowError):
        return before.copy()
    if x1 - x0 < 2 or y1 - y0 < 2:
        return before.copy()
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    feather = max(1.5, min(x1 - x0, y1 - y0) * 0.06)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    mixed = before.astype(np.float32) * (1.0 - mask) + after.astype(np.float32) * mask
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def _soften_almaz_result(before: np.ndarray, corrected: np.ndarray, factor: float) -> np.ndarray:
    """Blend an unsafe ALMAZ result back toward a non-AI baseline.

    x1 tasks blend toward the exact pre-action image.  x2 SR blends toward a
    bicubic x2 baseline, preserving the requested output size while reducing
    model-induced clipping/identity drift.
    """
    factor = float(np.clip(factor, 0.0, 1.0))
    if corrected.shape == before.shape:
        baseline = before
    elif (
        corrected.ndim == 3 and before.ndim == 3
        and corrected.shape[0] == before.shape[0] * 2
        and corrected.shape[1] == before.shape[1] * 2
        and corrected.shape[2] == before.shape[2]
    ):
        baseline = cv2.resize(before, (corrected.shape[1], corrected.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        return corrected
    mixed = baseline.astype(np.float32) * (1.0 - factor) + corrected.astype(np.float32) * factor
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


# Stable execution stages for multi-correction previews.  Keep these values
# spaced so future stages can be inserted without reshuffling unrelated work.
PREVIEW_ACTION_ORDER: dict[str, int] = {
    "surface_defects": 10,
    "red_eye": 20,
    "jpeg_artifacts": 30,
    "almaz_jpeg_recovery": 30,
    "noise": 40,
    "almaz_denoise": 40,
    "almaz_deblur": 50,
    "edge_artifacts": 60,
    "white_balance": 70,
    "auto_tone_color": 80,
    "exposure": 80,
    "contrast": 90,
    "posterization": 100,
    "super_resolution": 110,
    "sharpness": 120,
}


def _preview_order_key(item: Mapping[str, Any]) -> tuple[int, str]:
    """Return the deterministic correction pipeline stage for a preview item."""
    key = str(item.get("action_key", ""))
    candidate = str(item.get("candidate", ""))
    # Candidate identity wins over the action bucket: ALMAZ restoration may be
    # exposed either as a standalone action or as an upgraded legacy action.
    if candidate == "almaz_ai_jpeg_recovery_v1":
        return (30, key)
    if candidate == "almaz_ai_denoise_v1":
        return (40, key)
    if candidate == "almaz_ai_deblur_v1":
        return (50, key)
    return (PREVIEW_ACTION_ORDER.get(key, 999), key)


def apply_selected_preview(
    rgb: np.ndarray,
    validation_items: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    selected_action_keys: set[str] | list[str] | tuple[str, ...],
    action_strengths: Mapping[str, float] | None = None,
    action_regions: Mapping[str, Mapping[str, float]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Apply exactly the user-selected executable correction actions to a copy.

    Validator acceptance is deliberately *not* required here: rejected actions
    are only reachable when the user explicitly checks them in the UI.  Unknown
    or non-executable actions are ignored, and the source array is never mutated.
    """
    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    selected = {str(key) for key in selected_action_keys}
    # The reference auto trio is a compound alternative to the separate global
    # exposure/contrast actions. Avoid silently stacking the same tonal work.
    if "auto_tone_color" in selected:
        selected.discard("exposure")
        selected.discard("contrast")
        selected.discard("white_balance")
    elif "white_balance" in selected:
        selected.discard("auto_tone_color")
    strengths = {str(key): float(value) for key, value in (action_strengths or {}).items()}
    regions = {str(key): value for key, value in (action_regions or {}).items() if isinstance(value, Mapping)}
    out = rgb.copy()
    items = [
        item for item in validation_items
        if isinstance(item, dict)
        and str(item.get("action_key", "")) in selected
        and preview_action_available(item)
    ]
    # Correction order is deliberately independent from table/UI order.
    # Structural restoration runs on the least tone-altered pixels possible;
    # global colour/tone follows; resolution/detail work is last.
    items.sort(key=_preview_order_key)
    action_labels = {
        "red_eye": "исправляю красные глаза",
        "white_balance": "корректирую баланс белого",
        "auto_tone_color": "корректирую тон и цвет",
        "exposure": "корректирую яркость",
        "contrast": "корректирую локальный контраст",
        "surface_defects": "восстанавливаю дефекты поверхности",
        "noise": "обрабатываю шум",
        "jpeg_artifacts": "обрабатываю JPEG-артефакты",
        "edge_artifacts": "смягчаю ореолы",
        "posterization": "сглаживаю градации",
        "super_resolution": "готовлю увеличение x2",
        "almaz_denoise": "ALMAZ: выполняю AI Denoise",
        "almaz_deblur": "ALMAZ: выполняю AI Deblur",
        "almaz_jpeg_recovery": "ALMAZ: выполняю JPEG Recovery",
        "sharpness": "обрабатываю резкость",
    }
    total_items = len(items)
    for item_index, item in enumerate(items, start=1):
        key = str(item.get("action_key", ""))
        candidate = str(item.get("candidate", ""))
        emit(f"Предпросмотр {item_index}/{max(total_items, 1)}: {action_labels.get(key, key or 'применяю коррекцию')}")
        default_strength = float(item.get("default_strength", 1.0) or 1.0)
        strength = float(np.clip(strengths.get(key, default_strength), 0.0, 1.0))
        before_action = out
        corrected = out
        intrinsically_local = key in {"red_eye", "surface_defects"}
        intrinsically_global = key == "super_resolution" or candidate.startswith("almaz_ai_")
        if key == "red_eye" and candidate in {"localized_pupil_neutralization", "localized_pupil_neutralization_relaxed"}:
            candidate_payload = item.get("source_candidates")
            if isinstance(candidate_payload, list):
                corrected, _stats = correct_red_eye(
                    out, candidate_payload, strength=strength,
                    relaxed=(candidate == "localized_pupil_neutralization_relaxed"),
                )
        elif key == "white_balance" and candidate == "hybrid_awb_linear_rgb_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_white_balance(out, advice_from_raw(params), strength=strength)
        elif key == "white_balance" and candidate == "spatial_white_balance_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_spatial_white_balance(out, params, strength=strength)
        elif key == "auto_tone_color" and candidate == "reference_auto_tone_contrast_color_v1":
            corrected = apply_auto_tone_color(out, strength=strength)
        elif key == "exposure" and candidate == "spatial_exposure_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_local_exposure(out, params, strength=strength)
        elif key == "exposure" and candidate.startswith("gamma="):
            try:
                gamma_value = float(candidate.split("=", 1)[1])
            except (TypeError, ValueError):
                continue
            if 0.55 <= gamma_value <= 1.05:
                effective_gamma = 1.0 - strength * (1.0 - gamma_value)
                corrected = _gamma_lift(out, effective_gamma)
        elif key == "contrast" and candidate == "spatial_contrast_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_local_contrast(out, params, strength=strength)
        elif key == "contrast" and candidate == "mild_lab_clahe":
            corrected = _mild_local_contrast(out, strength)
        elif key == "surface_defects" and candidate == "bounded_surface_heal":
            candidate_payload = item.get("source_candidates")
            if isinstance(candidate_payload, list):
                corrected = _bounded_surface_heal(out, candidate_payload, strength)
        elif key == "almaz_deblur" and candidate == "almaz_ai_deblur_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_almaz_restoration(
                    out, task="deblur", strength=strength,
                    face_boxes=params.get("face_boxes", []),
                    face_protection=float(params.get("face_protection", 0.90) or 0.90), progress=progress,
                )
        elif key == "almaz_denoise" and candidate == "almaz_ai_denoise_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_almaz_restoration(
                    out, task="denoise", strength=strength,
                    face_boxes=params.get("face_boxes", []),
                    face_protection=float(params.get("face_protection", 0.78) or 0.78), progress=progress,
                )
        elif key == "almaz_jpeg_recovery" and candidate == "almaz_ai_jpeg_recovery_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_almaz_restoration(
                    out, task="jpeg_recovery", strength=strength,
                    face_boxes=params.get("face_boxes", []),
                    face_protection=float(params.get("face_protection", 0.86) or 0.86), progress=progress,
                )
        elif key == "sharpness" and candidate == "almaz_ai_deblur_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_almaz_restoration(
                    out, task="deblur", strength=strength,
                    face_boxes=params.get("face_boxes", []),
                    face_protection=float(params.get("face_protection", 0.90) or 0.90),
                    progress=progress,
                )
        elif key == "sharpness" and candidate == "spatial_sharpness_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_local_sharpness(out, params, strength=strength)
        elif key == "sharpness" and candidate == "edge_aware_unsharp":
            corrected = _edge_aware_sharpen(out, strength)
        elif key == "noise" and candidate == "almaz_ai_denoise_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_almaz_restoration(
                    out, task="denoise", strength=strength,
                    face_boxes=params.get("face_boxes", []),
                    face_protection=float(params.get("face_protection", 0.78) or 0.78),
                    progress=progress,
                )
        elif key == "noise" and candidate == "spatial_denoise_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_local_denoise(out, params, strength=strength)
        elif key == "noise" and candidate == "mild_nlm":
            corrected = _mild_denoise(out, strength)
        elif key == "jpeg_artifacts" and candidate == "almaz_ai_jpeg_recovery_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_almaz_restoration(
                    out, task="jpeg_recovery", strength=strength,
                    face_boxes=params.get("face_boxes", []),
                    face_protection=float(params.get("face_protection", 0.86) or 0.86),
                    progress=progress,
                )
        elif key == "jpeg_artifacts" and candidate == "mild_deblock":
            corrected = _mild_deblock(out, strength)
        elif key == "edge_artifacts" and candidate == "edge_halo_soften":
            corrected = _mild_edge_soften(out, strength)
        elif key == "posterization" and candidate == "mild_deband":
            corrected = _mild_deband(out, strength)
        elif key == "super_resolution" and candidate == "almaz_x2_identity_guard_v1":
            params = item.get("parameters")
            if isinstance(params, Mapping):
                corrected = apply_super_resolution_x2(
                    out,
                    strength=strength,
                    face_protection=float(params.get("face_protection", 0.85) or 0.85),
                    deblock_strength=float(params.get("deblock_strength", 0.22) or 0.22),
                    denoise_strength=float(params.get("denoise_strength", 0.14) or 0.14),
                    detail_strength=float(params.get("detail_strength", 0.55) or 0.55),
                    face_boxes=[(float(face.get("x", 0)), float(face.get("y", 0)), float(face.get("w", 0)), float(face.get("h", 0))) for face in params.get("face_boxes", []) if isinstance(face, Mapping)],
                    backend_preference=str(params.get("backend_preference", "auto")),
                    progress=progress,
                )
        if candidate.startswith("almaz_"):
            params_for_safety = item.get("parameters") if isinstance(item.get("parameters"), Mapping) else None
            emit("ALMAZ: проверяю, не появились ли паразитные цветные пятна")
            corrected, chroma_guard = guard_archival_chroma(before_action, corrected)
            if chroma_guard.applied:
                emit(
                    "ALMAZ: архивная защита цвета включена — сохраняю исходную сепию/монохромную цветность "
                    f"(chroma p99 {chroma_guard.raw_local_chroma_p99:.2f} -> "
                    f"{chroma_guard.guarded_local_chroma_p99:.2f} Lab)"
                )
            emit("ALMAZ: проверяю лица, цвет, пересветы и новые артефакты")
            safety = validate_almaz_transition(
                before_action, corrected, action_key=key, candidate=candidate,
                parameters=params_for_safety,
            )
            if not safety.accepted:
                emit("ALMAZ safety: исходная сила не прошла защиту; автоматически уменьшаю эффект")
                recovered = False
                for safety_factor in (0.75, 0.50, 0.35):
                    emit(f"ALMAZ safety: пробую более мягкий вариант — {safety_factor * 100:.0f}% от рассчитанного эффекта")
                    softened = _soften_almaz_result(before_action, corrected, safety_factor)
                    retry = validate_almaz_transition(
                        before_action, softened, action_key=key, candidate=candidate,
                        parameters=params_for_safety,
                    )
                    if retry.accepted:
                        corrected = softened
                        safety = retry
                        effective_pct = strength * safety_factor * 100.0
                        emit(
                            f"ALMAZ safety: найден безопасный вариант; эффективная сила около {effective_pct:.0f}% "
                            f"(clipping +{retry.clipping_delta_pct:.2f} п.п.)"
                        )
                        recovered = True
                        break
                if not recovered:
                    emit("ALMAZ safety: даже мягкий вариант не прошёл проверку; результат отклонён")
                    raise RuntimeError(
                        "Проверка безопасности ALMAZ отклонила коррекцию даже после автоматического снижения силы. "
                        "Исходный файл не изменён. Попробуйте уменьшить силу коррекции вручную. "
                        f"Технические данные: {safety.message}"
                    )
            else:
                emit("ALMAZ: проверка безопасности пройдена")
        if not intrinsically_local and not intrinsically_global and key in regions:
            out = _blend_correction_region(before_action, corrected, regions.get(key))
        else:
            out = corrected
    emit("Предпросмотр: собираю итоговое изображение")
    return out


def apply_validated_preview(rgb: np.ndarray, validation_items: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> np.ndarray:
    """Apply only Validator-accepted Fix candidates to an in-memory copy.

    Kept as the safe automatic API. Manual UI overrides use
    :func:`apply_selected_preview` with an explicit action-key set.
    """
    accepted_keys = {
        str(item.get("action_key", ""))
        for item in validation_items
        if isinstance(item, dict) and bool(item.get("accepted", False))
    }
    return apply_selected_preview(rgb, validation_items, accepted_keys)
