from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .main_subject import MainSubjectResult


@dataclass(frozen=True, slots=True)
class AestheticQualityResult:
    score: float | None
    confidence: float
    classification: str
    components: dict[str, float]
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_raw(self) -> dict[str, object]:
        return {
            "score": self.score,
            "confidence": self.confidence,
            "classification": self.classification,
            "components": dict(self.components),
            "reasons": list(self.reasons),
            "limitations": list(self.limitations),
            "method": "aesthetic_subject_composition_v1",
            "technical_quality_independent": True,
        }


def _clip100(v: float) -> float:
    return float(np.clip(v, 0.0, 100.0))


def _bell(value: float, low: float, ideal_low: float, ideal_high: float, high: float) -> float:
    if value <= low or value >= high:
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < ideal_low:
        return float((value - low) / max(ideal_low - low, 1e-9))
    return float((high - value) / max(high - ideal_high, 1e-9))


def _placement_score(kind: str, cx: float, cy: float) -> float:
    center_dist = float(np.hypot((cx - 0.5) / 0.50, (cy - 0.46) / 0.52))
    center = float(np.clip(1.0 - center_dist, 0.0, 1.0))
    thirds = min(
        float(np.hypot((cx - tx) / 0.46, (cy - ty) / 0.46))
        for tx in (1.0 / 3.0, 2.0 / 3.0)
        for ty in (1.0 / 3.0, 2.0 / 3.0)
    )
    thirds_score = float(np.clip(1.0 - thirds, 0.0, 1.0))
    if kind == "people_group":
        return max(center, 0.78 * thirds_score)
    if kind == "person":
        return max(0.92 * center, thirds_score)
    return max(0.82 * center, thirds_score)


def _prominence_score(kind: str, area: float) -> float:
    if kind == "person":
        return _bell(area, 0.025, 0.10, 0.46, 0.82)
    if kind == "people_group":
        return _bell(area, 0.04, 0.16, 0.68, 0.92)
    return _bell(area, 0.015, 0.07, 0.42, 0.72)


def _edge_safety(box: dict[str, float], kind: str) -> float:
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    margins = (x, y, 1.0 - x - w, 1.0 - y - h)
    min_margin = min(margins)
    # A large portrait/group crop can legitimately touch an edge, so do not turn
    # close framing into an automatic aesthetic defect.
    area = w * h
    relaxed = 0.012 if kind in {"person", "people_group"} and area >= 0.42 else 0.025
    if min_margin >= 0.06:
        return 1.0
    return float(np.clip((min_margin + relaxed) / (0.06 + relaxed), 0.0, 1.0))


def _subject_mask(shape: tuple[int, int], box: dict[str, float]) -> np.ndarray:
    h, w = shape
    x0 = int(np.clip(round(float(box["x"]) * w), 0, w - 1))
    y0 = int(np.clip(round(float(box["y"]) * h), 0, h - 1))
    x1 = int(np.clip(round((float(box["x"]) + float(box["w"])) * w), x0 + 1, w))
    y1 = int(np.clip(round((float(box["y"]) + float(box["h"])) * h), y0 + 1, h))
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _visual_components(rgb: np.ndarray, box: dict[str, float]) -> tuple[float, float]:
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, 720.0 / max(h0, w0))
    if scale < 1.0:
        work = cv2.resize(rgb, (max(1, round(w0 * scale)), max(1, round(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        work = rgb
    h, w = work.shape[:2]
    mask = _subject_mask((h, w), box).astype(bool)
    if mask.sum() < 64 or (~mask).sum() < 64:
        return 50.0, 50.0

    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    subject_grad = float(np.percentile(grad[mask], 70))
    bg_grad = float(np.percentile(grad[~mask], 70))
    ratio = bg_grad / max(subject_grad, 1e-4)
    background_calmness = _clip100(82.0 - max(0.0, ratio - 0.75) * 38.0)

    # Tonal separation is only a weak compositional cue. Similar tones are not
    # automatically bad, so the floor stays deliberately high.
    subject_mean = float(np.mean(gray[mask]))
    bg_mean = float(np.mean(gray[~mask]))
    delta = abs(subject_mean - bg_mean)
    tonal_separation = _clip100(52.0 + min(delta / 0.22, 1.0) * 48.0)
    return background_calmness, tonal_separation


def analyze_aesthetic_quality(rgb: np.ndarray, subject: MainSubjectResult) -> AestheticQualityResult:
    limitations = (
        "Не оценивает эмоцию, удачность момента, личную ценность или художественный замысел.",
        "Оценка композиционная и эвристическая; она не должна автоматически управлять проверкой безопасности.",
    )
    if subject.subject_kind == "unknown" or subject.box_norm is None or subject.confidence < 0.42:
        return AestheticQualityResult(
            None,
            float(np.clip(subject.confidence * 0.55, 0.0, 0.32)),
            "unknown",
            {},
            ("main_subject_unknown",),
            limitations,
        )

    box = subject.box_norm
    area = float(box["w"] * box["h"])
    cx = float(box["x"] + box["w"] * 0.5)
    cy = float(box["y"] + box["h"] * 0.5)
    placement = 100.0 * _placement_score(subject.subject_kind, cx, cy)
    prominence = 100.0 * _prominence_score(subject.subject_kind, area)
    edge = 100.0 * _edge_safety(box, subject.subject_kind)
    calmness, separation = _visual_components(rgb, box)
    components = {
        "placement": _clip100(placement),
        "prominence": _clip100(prominence),
        "edge_safety": _clip100(edge),
        "background_calmness": _clip100(calmness),
        "tonal_separation": _clip100(separation),
    }
    score = (
        0.26 * components["placement"]
        + 0.21 * components["prominence"]
        + 0.18 * components["edge_safety"]
        + 0.22 * components["background_calmness"]
        + 0.13 * components["tonal_separation"]
    )
    score = _clip100(score)
    confidence = float(np.clip(0.34 + 0.46 * subject.confidence, 0.0, 0.78))
    if score >= 76.0:
        classification = "strong_composition_signal"
    elif score >= 62.0:
        classification = "balanced"
    elif score >= 48.0:
        classification = "mixed"
    else:
        classification = "review"

    weakest = min(components, key=components.get)
    reasons = (f"weakest:{weakest}", f"subject:{subject.subject_kind}")
    return AestheticQualityResult(score, confidence, classification, components, reasons, limitations)
