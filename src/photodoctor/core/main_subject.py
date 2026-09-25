from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .eyes import EyeRegion
from .faces import FaceRegion
from .semantic_context import SemanticContext


@dataclass(frozen=True, slots=True)
class MainSubjectResult:
    subject_kind: str
    scene_kind: str
    confidence: float
    box_norm: dict[str, float] | None
    protection_box_norm: dict[str, float] | None
    source: str
    face_indices: tuple[int, ...]
    subject_area_pct: float
    reasons: tuple[str, ...]

    def to_raw(self) -> dict[str, object]:
        return {
            "subject_kind": self.subject_kind,
            "scene_kind": self.scene_kind,
            "confidence": self.confidence,
            "box_norm": dict(self.box_norm) if self.box_norm else None,
            "protection_box_norm": dict(self.protection_box_norm) if self.protection_box_norm else None,
            "source": self.source,
            "face_indices": list(self.face_indices),
            "subject_area_pct": self.subject_area_pct,
            "reasons": list(self.reasons),
            "method": "main_subject_rules_saliency_v1",
        }


def _clip_box(x: float, y: float, w: float, h: float) -> dict[str, float]:
    x0 = float(np.clip(x, 0.0, 1.0))
    y0 = float(np.clip(y, 0.0, 1.0))
    x1 = float(np.clip(x + w, x0, 1.0))
    y1 = float(np.clip(y + h, y0, 1.0))
    return {"x": x0, "y": y0, "w": max(0.0, x1 - x0), "h": max(0.0, y1 - y0)}


def _expand_box(box: dict[str, float], x_factor: float, top_factor: float, bottom_factor: float) -> dict[str, float]:
    x, y, w, h = (float(box[k]) for k in ("x", "y", "w", "h"))
    return _clip_box(
        x - w * x_factor,
        y - h * top_factor,
        w * (1.0 + 2.0 * x_factor),
        h * (1.0 + top_factor + bottom_factor),
    )


def _union_boxes(boxes: list[dict[str, float]]) -> dict[str, float]:
    x0 = min(float(b["x"]) for b in boxes)
    y0 = min(float(b["y"]) for b in boxes)
    x1 = max(float(b["x"]) + float(b["w"]) for b in boxes)
    y1 = max(float(b["y"]) + float(b["h"]) for b in boxes)
    return _clip_box(x0, y0, x1 - x0, y1 - y0)


def _face_box(face: FaceRegion, image_w: int, image_h: int) -> dict[str, float]:
    return _clip_box(face.x / image_w, face.y / image_h, face.w / image_w, face.h / image_h)


def _face_score(face: FaceRegion, image_w: int, image_h: int, eye_count: int) -> float:
    area = (face.w * face.h) / max(float(image_w * image_h), 1.0)
    area_score = float(np.clip(np.sqrt(area / 0.055), 0.0, 1.0))
    cx = (face.x + face.w * 0.5) / image_w
    cy = (face.y + face.h * 0.5) / image_h
    dist = np.hypot((cx - 0.5) / 0.72, (cy - 0.46) / 0.72)
    centrality = float(np.clip(1.0 - dist, 0.0, 1.0))
    sharpness = float(np.clip(face.sharpness_score / 100.0, 0.0, 1.0))
    measure = float(np.clip(face.measurement_confidence, 0.0, 1.0))
    eye_support = 1.0 if eye_count >= 2 else (0.55 if eye_count == 1 else 0.0)
    manual_bonus = 0.08 if face.source == "manual" else 0.0
    return float(np.clip(
        0.42 * area_score + 0.23 * centrality + 0.13 * sharpness + 0.12 * measure + 0.10 * eye_support + manual_bonus,
        0.0, 1.0,
    ))


def _scene_kind(semantic: SemanticContext, subject_kind: str) -> str:
    if semantic.classification == "archival_portrait":
        return "archival_people"
    if subject_kind == "people_group" or semantic.classification == "group_portrait":
        return "people_group"
    if subject_kind == "person" or semantic.classification == "portrait":
        return "people_portrait"
    if subject_kind == "visual_region":
        return "general_scene"
    return "unknown_scene"


def _normalize_feature(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    p95 = float(np.percentile(arr, 95)) if arr.size else 0.0
    if p95 <= 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr / p95, 0.0, 1.0).astype(np.float32)


def _visual_subject(rgb: np.ndarray) -> tuple[dict[str, float] | None, float, tuple[str, ...]]:
    h0, w0 = rgb.shape[:2]
    if min(h0, w0) < 48:
        return None, 0.0, ("image_too_small",)
    scale = min(1.0, 640.0 / max(h0, w0))
    if scale < 1.0:
        work = cv2.resize(rgb, (max(1, round(w0 * scale)), max(1, round(h0 * scale))), interpolation=cv2.INTER_AREA)
    else:
        work = rgb
    h, w = work.shape[:2]
    gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(2.0, min(h, w) / 45.0))
    local = _normalize_feature(np.abs(gray - blur))
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = _normalize_feature(cv2.magnitude(gx, gy))
    hsv = cv2.cvtColor(work, cv2.COLOR_RGB2HSV)
    sat = _normalize_feature(hsv[..., 1].astype(np.float32) / 255.0)

    intrinsic = 0.44 * local + 0.40 * grad + 0.16 * sat
    intrinsic_p95 = float(np.percentile(intrinsic, 95))
    intrinsic_mean = float(np.mean(intrinsic))
    if intrinsic_p95 < 0.16 or intrinsic_mean < 0.015:
        return None, 0.0, ("weak_visual_signal",)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / max(w * 0.52, 1.0)
    ny = (yy - h * 0.47) / max(h * 0.55, 1.0)
    center = np.exp(-(nx * nx + ny * ny) * 0.65).astype(np.float32)
    saliency = np.clip(0.90 * intrinsic + 0.10 * center, 0.0, 1.0)

    threshold = max(0.24, float(np.percentile(saliency, 82)))
    mask = (saliency >= threshold).astype(np.uint8) * 255
    radius = max(3, int(round(min(h, w) / 90)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius | 1, radius | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    best: tuple[float, tuple[int, int, int, int], float] | None = None
    total_area = float(h * w)
    for label in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[label])
        area_frac = area / max(total_area, 1.0)
        if area_frac < 0.006 or area_frac > 0.58 or bw < 8 or bh < 8:
            continue
        component = labels == label
        mean_saliency = float(np.mean(saliency[component]))
        cx, cy = centroids[label]
        dist = np.hypot((cx / w - 0.5) / 0.72, (cy / h - 0.47) / 0.72)
        centrality = float(np.clip(1.0 - dist, 0.0, 1.0))
        area_score = float(np.clip(np.sqrt(area_frac / 0.08), 0.0, 1.0))
        if area_frac > 0.34:
            area_score *= float(np.clip((0.58 - area_frac) / 0.24, 0.0, 1.0))
        score = 0.58 * mean_saliency + 0.22 * centrality + 0.20 * area_score
        if best is None or score > best[0]:
            best = (score, (x, y, bw, bh), area_frac)

    if best is None or best[0] < 0.34:
        return None, 0.0, ("no_stable_visual_component",)
    score, (x, y, bw, bh), area_frac = best
    box = _clip_box(x / w, y / h, bw / w, bh / h)
    box = _expand_box(box, 0.08, 0.08, 0.08)
    confidence = float(np.clip(0.22 + 0.55 * score + 0.12 * min(1.0, intrinsic_p95), 0.0, 0.64))
    return box, confidence, ("saliency_component", "no_confirmed_face_subject")


def analyze_main_subject(
    rgb: np.ndarray,
    faces: list[FaceRegion],
    eyes: list[EyeRegion],
    semantic: SemanticContext,
    *,
    face_detector_confidence: float,
) -> MainSubjectResult:
    h, w = rgb.shape[:2]
    if faces:
        eye_counts = [0 for _ in faces]
        for eye in eyes:
            if 0 <= eye.face_index < len(eye_counts):
                eye_counts[eye.face_index] += 1
        scores = [_face_score(face, w, h, eye_counts[index]) for index, face in enumerate(faces)]
        order = sorted(range(len(faces)), key=lambda index: scores[index], reverse=True)
        boxes = [_face_box(face, w, h) for face in faces]
        use_group = len(faces) >= 2
        if len(faces) >= 2:
            top, second = order[0], order[1]
            top_area = boxes[top]["w"] * boxes[top]["h"]
            second_area = boxes[second]["w"] * boxes[second]["h"]
            dominant = scores[top] >= scores[second] * 1.55 and top_area >= second_area * 1.8
            if dominant and semantic.classification != "group_portrait":
                use_group = False
        if use_group:
            selected = tuple(range(len(faces)))
            union = _union_boxes([boxes[index] for index in selected])
            subject_box = _expand_box(union, 0.12, 0.12, 0.42)
            protection = _expand_box(union, 0.08, 0.10, 0.18)
            avg_score = float(np.mean([scores[index] for index in selected]))
            confidence = float(np.clip(0.38 + 0.32 * face_detector_confidence + 0.24 * avg_score, 0.0, 0.92))
            kind = "people_group"
            reasons = ("multiple_faces", "faces_are_primary_semantic_signal")
        else:
            selected = (order[0],)
            base = boxes[order[0]]
            subject_box = _expand_box(base, 0.42, 0.22, 0.95)
            protection = _expand_box(base, 0.18, 0.16, 0.28)
            face = faces[order[0]]
            manual_bonus = 0.06 if face.source == "manual" else 0.0
            confidence = float(np.clip(0.42 + 0.30 * face_detector_confidence + 0.24 * scores[order[0]] + manual_bonus, 0.0, 0.94))
            kind = "person"
            reasons = ("dominant_face", "face_size_position_quality")
        area_pct = 100.0 * subject_box["w"] * subject_box["h"]
        return MainSubjectResult(
            kind,
            _scene_kind(semantic, kind),
            confidence,
            subject_box,
            protection,
            "faces",
            selected,
            float(area_pct),
            reasons,
        )

    visual_box, visual_conf, reasons = _visual_subject(rgb)
    if visual_box is not None and visual_conf >= 0.42:
        protection = _expand_box(visual_box, 0.05, 0.05, 0.05)
        return MainSubjectResult(
            "visual_region",
            _scene_kind(semantic, "visual_region"),
            visual_conf,
            visual_box,
            protection,
            "visual_saliency",
            (),
            float(100.0 * visual_box["w"] * visual_box["h"]),
            reasons,
        )

    return MainSubjectResult(
        "unknown",
        _scene_kind(semantic, "unknown"),
        float(np.clip(max(0.18, visual_conf), 0.0, 0.38)),
        None,
        None,
        "none",
        (),
        0.0,
        reasons or ("no_reliable_subject",),
    )
