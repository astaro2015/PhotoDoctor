from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class HighlightContextResult:
    confidence: float
    total_clip_pct: float
    protected_clip_pct: float
    unexplained_clip_pct: float
    candidate_count: int
    face_rejected_count: int
    border_rejected_count: int
    boxes_norm: list[dict[str, float | str]]


def _overlaps_face(x: int, y: int, w: int, h: int, face_boxes: list[tuple[int, int, int, int]]) -> bool:
    if not face_boxes:
        return False
    x1, y1 = x + w, y + h
    area = max(w * h, 1)
    for fx, fy, fw, fh in face_boxes:
        ix0 = max(x, fx); iy0 = max(y, fy)
        ix1 = min(x1, fx + fw); iy1 = min(y1, fy + fh)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        overlap = (ix1 - ix0) * (iy1 - iy0) / area
        if overlap >= 0.12:
            return True
    return False


def analyze_highlight_context(
    rgb: np.ndarray,
    linear_luma: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]] | None = None,
) -> HighlightContextResult:
    """Find compact clipped highlights that may be specular/light-source regions.

    The output is intentionally a *context* for clipping, not a claim that the
    candidate is harmless. Large white regions, border-connected highlights and
    anything substantially overlapping a detected face remain unexplained.
    """
    h, w = rgb.shape[:2]
    if h < 32 or w < 32 or linear_luma.shape[:2] != (h, w):
        return HighlightContextResult(0.0, 0.0, 0.0, 0.0, 0, 0, 0, [])

    face_boxes = list(face_boxes or [])
    clip_mask = linear_luma >= 0.99
    total_clip = int(clip_mask.sum())
    total_pixels = h * w
    total_clip_pct = float(100.0 * total_clip / max(total_pixels, 1))
    if total_clip == 0:
        return HighlightContextResult(0.82, 0.0, 0.0, 0.0, 0, 0, 0, [])

    # The connected component is based on a less extreme bright mask so that a
    # clipped core and its luminous neighbourhood are treated as one object.
    bright_mask = (linear_luma >= 0.78).astype(np.uint8)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bright_mask, 8)

    protected = np.zeros((h, w), np.uint8)
    boxes: list[dict[str, float | str]] = []
    face_rejected = 0
    border_rejected = 0

    for label in range(1, n_labels):
        x, y, bw, bh, area = (int(v) for v in stats[label])
        component = labels == label
        clipped_here = clip_mask & component
        clip_count = int(clipped_here.sum())
        if clip_count == 0:
            continue

        area_pct = 100.0 * area / total_pixels
        clip_pct = 100.0 * clip_count / total_pixels
        touches_border = x <= 1 or y <= 1 or (x + bw) >= w - 1 or (y + bh) >= h - 1
        if touches_border:
            border_rejected += 1
            continue
        if _overlaps_face(x, y, bw, bh, face_boxes):
            face_rejected += 1
            continue

        # Compare the luminous component with a local ring. A white shirt or
        # pale wall normally has less separation than a glint or light source.
        comp_u8 = component.astype(np.uint8)
        radius = int(np.clip(round(min(bw, bh) * 0.22), 3, 11))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        dilated = cv2.dilate(comp_u8, kernel, iterations=1).astype(bool)
        ring = dilated & (~component)
        if int(ring.sum()) >= 32:
            background = float(np.median(linear_luma[ring]))
        else:
            background = float(np.median(linear_luma))
        peak = float(np.percentile(linear_luma[component], 90.0))
        local_contrast = peak - background
        compactness = float(area / max(bw * bh, 1))

        # Conservative protection. Tiny glints need only moderate contrast;
        # larger light-source candidates need stronger separation and shape.
        tiny = area_pct <= 0.18 and clip_pct <= 0.12 and local_contrast >= 0.08
        compact = area_pct <= 1.25 and clip_pct <= 0.60 and local_contrast >= 0.16 and compactness >= 0.24
        light_source = area_pct <= 2.20 and clip_pct <= 1.00 and local_contrast >= 0.25 and compactness >= 0.34
        if not (tiny or compact or light_source):
            continue

        kind = "light_source_candidate" if light_source and area_pct > 0.55 else "specular_candidate"
        protected[clipped_here] = 1
        boxes.append({
            "x": x / w,
            "y": y / h,
            "w": bw / w,
            "h": bh / h,
            "kind": kind,
            "component_area_pct": float(area_pct),
            "clip_area_pct": float(clip_pct),
            "local_contrast": float(local_contrast),
            "compactness": compactness,
        })

    protected_count = int((protected.astype(bool) & clip_mask).sum())
    protected_pct = float(100.0 * protected_count / max(total_pixels, 1))
    unexplained_pct = max(0.0, total_clip_pct - protected_pct)

    component_support = float(np.clip(total_clip / max(total_pixels * 0.001, 1.0), 0.0, 1.0))
    confidence = float(np.clip(0.50 + 0.22 * component_support, 0.0, 0.72))
    if not boxes:
        confidence = float(np.clip(confidence + 0.05, 0.0, 0.77))

    return HighlightContextResult(
        confidence=confidence,
        total_clip_pct=total_clip_pct,
        protected_clip_pct=protected_pct,
        unexplained_clip_pct=unexplained_pct,
        candidate_count=len(boxes),
        face_rejected_count=face_rejected,
        border_rejected_count=border_rejected,
        boxes_norm=boxes[:48],
    )
