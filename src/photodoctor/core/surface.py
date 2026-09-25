from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class SurfaceDefectDetection:
    candidate_density: float
    candidate_count: int
    threshold: float
    confidence: float
    score: float
    boxes_norm: list[dict[str, object]]
    ai_boxes_norm: list[dict[str, object]]


def _clip_score(v: float) -> float:
    return float(np.clip(v, 0.0, 100.0))


def _iou(a: dict[str, object], b: dict[str, object]) -> float:
    ax0, ay0 = float(a["x"]), float(a["y"])
    ax1, ay1 = ax0 + float(a["w"]), ay0 + float(a["h"])
    bx0, by0 = float(b["x"]), float(b["y"])
    bx1, by1 = bx0 + float(b["w"]), by0 + float(b["h"])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = float(a["w"]) * float(a["h"]) + float(b["w"]) * float(b["h"]) - inter
    return inter / union if union > 0 else 0.0


def _robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 4.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values.astype(np.float32) - med)))
    return max(4.0, 1.4826 * mad)


def _deduplicate_ranked_candidates(
    candidates: list[tuple[float, dict[str, object]]],
    *,
    iou_threshold: float = 0.45,
) -> list[tuple[float, dict[str, object]]]:
    """Keep the historical greedy IoU order while vectorising pair checks.

    The previous Python ``any(_iou(candidate, kept) ...)`` performed millions of
    dictionary lookups and scalar Python calls on texture-heavy frames.  This is
    the same greedy algorithm: candidates are still visited in rank order and a
    candidate is rejected iff its IoU with any *already kept* box is strictly
    greater than the same threshold.  Only the pair arithmetic is batched in
    NumPy.
    """
    count = len(candidates)
    if count <= 1:
        return list(candidates)

    boxes = np.empty((count, 4), dtype=np.float64)
    areas = np.empty(count, dtype=np.float64)
    for index, (_, box) in enumerate(candidates):
        x0 = float(box["x"]); y0 = float(box["y"])
        bw = float(box["w"]); bh = float(box["h"])
        boxes[index] = (x0, y0, x0 + bw, y0 + bh)
        areas[index] = bw * bh

    kept = np.empty(count, dtype=np.int32)
    kept_count = 0
    threshold = float(iou_threshold)
    for index in range(count):
        if kept_count:
            previous = kept[:kept_count]
            ix0 = np.maximum(boxes[index, 0], boxes[previous, 0])
            iy0 = np.maximum(boxes[index, 1], boxes[previous, 1])
            ix1 = np.minimum(boxes[index, 2], boxes[previous, 2])
            iy1 = np.minimum(boxes[index, 3], boxes[previous, 3])
            iw = np.maximum(0.0, ix1 - ix0)
            ih = np.maximum(0.0, iy1 - iy0)
            inter = iw * ih
            union = areas[index] + areas[previous] - inter
            ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
            if bool(np.any(ious > threshold)):
                continue
        kept[kept_count] = index
        kept_count += 1

    return [candidates[int(index)] for index in kept[:kept_count]]


def merge_surface_detections(
    *detections: SurfaceDefectDetection,
    max_boxes: int,
    max_ai_boxes: int,
) -> SurfaceDefectDetection:
    """Merge deeper Surface passes without letting a later pass erase an earlier one.

    Maximum mode intentionally runs the proven Precise search plus its own denser
    search.  The union is ranked by the same stored candidate evidence and then
    greedily IoU-deduplicated.  Thus Maximum can add/replace overlapping geometry,
    but a distinct Precise candidate survives even when Maximum's thresholds produce
    a different component layout.
    """
    if not detections:
        return SurfaceDefectDetection(0.0, 0, 255.0, 0.20, 100.0, [], [])

    ranked: list[tuple[float, dict[str, object]]] = []
    for pass_index, detection in enumerate(detections):
        for order, raw in enumerate(detection.ai_boxes_norm):
            if not isinstance(raw, dict):
                continue
            box = dict(raw)
            box.setdefault("surface_precision_pass", "maximum" if pass_index == 0 else "precise")
            try:
                quality = float(box.get("candidate_quality", 0.0) or 0.0)
                strength = float(box.get("strength", 0.0) or 0.0)
                length = float(box.get("oriented_length_px", 0.0) or 0.0)
            except (TypeError, ValueError, OverflowError):
                quality = strength = length = 0.0
            # Stable evidence rank. `order` only breaks exact ties and is tiny enough
            # never to overpower a real quality/strength difference.
            rank = quality * 2.0 + strength * 0.45 + min(max(length, 0.0) / 220.0, 1.0) * 0.16 - order * 1e-7
            ranked.append((rank, box))

    ranked.sort(key=lambda item: item[0], reverse=True)
    deduped = _deduplicate_ranked_candidates(ranked, iou_threshold=0.45)
    shown = deduped[:max(0, int(max_boxes))]
    ai_rows = deduped[:max(0, int(max_ai_boxes))]
    densities = [float(item.candidate_density) for item in detections]
    thresholds = [float(item.threshold) for item in detections]
    confidences = [float(item.confidence) for item in detections]
    scores = [float(item.score) for item in detections]
    return SurfaceDefectDetection(
        candidate_density=max(densities) if densities else 0.0,
        candidate_count=len(deduped),
        threshold=min(thresholds) if thresholds else 255.0,
        confidence=max(confidences) if confidences else 0.20,
        score=min(scores) if scores else 100.0,
        boxes_norm=[box for _, box in shown],
        ai_boxes_norm=[box for _, box in ai_rows],
    )


def _component_context_features(
    gray: np.ndarray,
    lab: np.ndarray,
    labels: np.ndarray,
    idx: int,
    x: int,
    y: int,
    bw: int,
    bh: int,
    polarity: str,
    *,
    candidate_kind: str,
) -> dict[str, float]:
    """Describe whether a morphology candidate behaves like a physical surface defect.

    A scratch/ridge should be a local intensity extremum, not merely one side of a
    natural image edge.  We therefore inspect an annulus around the component and,
    for elongated candidates, compare both sides of the estimated scratch axis.
    The features are deliberately deterministic and remain advisory; they are also
    stored with the candidate so the UI/AI fusion can explain why a candidate was
    promoted or suppressed.
    """
    h, w = gray.shape
    long_side = max(bw, bh)
    short_side = max(1, min(bw, bh))
    pad_floor = 12 if candidate_kind == "spot" else 8
    pad = int(np.clip(round(short_side * 2.8 + long_side * 0.035), pad_floor, 36))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    local_labels = labels[y0:y1, x0:x1]
    comp = (local_labels == idx).astype(np.uint8)
    if int(comp.sum()) < 2:
        return {
            "linearity": 0.0, "context_contrast": 0.0, "side_similarity": 0.5,
            "texture_risk": 1.0, "chroma_risk": 0.5, "candidate_quality": 0.0,
        }

    g = gray[y0:y1, x0:x1]
    l = lab[y0:y1, x0:x1]
    radius = max(2, min(10, short_side + 2))
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil = cv2.dilate(comp, kernel)
    ring = (dil > 0) & (comp == 0)
    if int(ring.sum()) < 12:
        larger = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize + 4, ksize + 4)))
        ring = (larger > 0) & (comp == 0)

    center_vals = g[comp > 0].astype(np.float32)
    ring_vals = g[ring].astype(np.float32)
    center_med = float(np.median(center_vals)) if center_vals.size else 0.0
    ring_med = float(np.median(ring_vals)) if ring_vals.size else center_med
    sigma = _robust_sigma(ring_vals)
    signed_delta = (center_med - ring_med) if polarity == "bright" else (ring_med - center_med)
    contrast_z = signed_delta / sigma
    context_contrast = float(np.clip((contrast_z - 0.45) / 3.0, 0.0, 1.0))

    ys, xs = np.where(comp > 0)
    coords = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    linearity = 0.0
    minor = np.asarray([1.0, 0.0], dtype=np.float32)
    if len(coords) >= 3:
        centered = coords - coords.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(1, len(coords) - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)
        lam_small = max(float(vals[order[0]]), 1e-6)
        lam_large = max(float(vals[order[-1]]), lam_small)
        linearity = float(np.clip(1.0 - lam_small / lam_large, 0.0, 1.0))
        major = vecs[:, order[-1]].astype(np.float32)
        minor = np.asarray([-major[1], major[0]], dtype=np.float32)

    side_similarity = 0.5
    if candidate_kind == "line" and len(coords) >= 4:
        offset = float(max(2.0, short_side * 1.35 + 1.0))
        side_medians: list[float] = []
        for direction in (-1.0, 1.0):
            dx, dy = float(minor[0] * offset * direction), float(minor[1] * offset * direction)
            shifted = cv2.warpAffine(
                comp,
                np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
                (comp.shape[1], comp.shape[0]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            side_mask = (shifted > 0) & (comp == 0)
            vals_side = g[side_mask]
            if vals_side.size >= 3:
                side_medians.append(float(np.median(vals_side)))
        if len(side_medians) == 2:
            side_gap = abs(side_medians[0] - side_medians[1])
            side_similarity = float(np.clip(1.0 - side_gap / max(18.0, 3.0 * sigma), 0.0, 1.0))
            side_ref = max(side_medians) if polarity == "bright" else min(side_medians)
            signed_side_delta = (center_med - side_ref) if polarity == "bright" else (side_ref - center_med)
            side_extremum = float(np.clip((signed_side_delta / sigma - 0.35) / 2.6, 0.0, 1.0))
            context_contrast = 0.55 * context_contrast + 0.45 * side_extremum

    # Natural detail tends to live in edge-dense/texture-dense neighborhoods.  This
    # is a soft penalty only: a real scratch can cross a textured region.
    edges = cv2.Canny(g, 55, 145)
    # Inspect the whole context crop, not only the immediate ring. Repeated text,
    # hair and fabric seams otherwise look deceptively defect-like at tiny scale.
    edge_fraction = float(np.mean(edges > 0)) if edges.size else 0.0
    edge_risk = float(np.clip((edge_fraction - 0.045) / 0.20, 0.0, 1.0))
    if ring_vals.size >= 12:
        p10, p90 = np.percentile(ring_vals, [10, 90])
        ring_range = float(p90 - p10)
    else:
        ring_range = 0.0
    range_risk = float(np.clip((ring_range - 24.0) / 85.0, 0.0, 1.0))
    texture_risk = max(edge_risk, range_risk)

    # Strong chroma changes with weak luminance evidence are often painted/textured
    # scene details rather than physical dust/scratches.  Again: penalty, not veto.
    chroma_risk = 0.0
    if np.any(ring):
        center_ab = np.median(l[..., 1:3][comp > 0].astype(np.float32), axis=0)
        ring_ab = np.median(l[..., 1:3][ring].astype(np.float32), axis=0)
        chroma_delta = float(np.linalg.norm(center_ab - ring_ab))
        luma_delta = abs(center_med - ring_med)
        ratio = chroma_delta / max(6.0, luma_delta)
        chroma_risk = float(np.clip((ratio - 0.55) / 1.6, 0.0, 1.0))

    aspect = long_side / max(1.0, float(short_side))
    if candidate_kind == "line":
        aspect_score = float(np.clip((aspect - 1.8) / 6.0, 0.0, 1.0))
        geometry = 0.62 * linearity + 0.38 * aspect_score
    elif candidate_kind == "spot":
        geometry = float(np.clip(1.0 - abs(np.log(max(aspect, 1e-6))) / 1.4, 0.0, 1.0))
    else:
        geometry = 0.42 * linearity + 0.22

    context_score = float(np.clip(0.72 * context_contrast + 0.28 * side_similarity, 0.0, 1.0))
    penalty = 0.20 * texture_risk + 0.12 * chroma_risk
    if polarity == "dark":
        penalty += 0.07  # natural dark lines (hair, seams, text) are especially common
    quality = float(np.clip(0.42 * context_score + 0.38 * geometry + 0.20 * (1.0 - texture_risk) - penalty, 0.0, 1.0))
    return {
        "linearity": linearity,
        "context_contrast": float(np.clip(context_contrast, 0.0, 1.0)),
        "side_similarity": side_similarity,
        "texture_risk": texture_risk,
        "chroma_risk": chroma_risk,
        "candidate_quality": quality,
    }


def _bounded_hysteresis_mask(strong: np.ndarray, weak: np.ndarray, *, max_steps: int = 5) -> np.ndarray:
    """Grow strong evidence only through nearby weak evidence.

    Full flood-fill hysteresis is risky on faces and fabric because one real crack can
    connect to natural structure. A bounded number of one-pixel growth steps bridges
    small gaps without allowing the mask to wander across the whole photograph.
    """
    if strong.shape != weak.shape:
        raise ValueError("strong и weak должны иметь одинаковый размер")
    current = strong.astype(bool, copy=True)
    allowed = weak.astype(bool, copy=False)
    if not np.any(current):
        return np.zeros(strong.shape, np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(0, int(max_steps))):
        grown = cv2.dilate(current.astype(np.uint8), kernel, iterations=1) > 0
        updated = current | (grown & allowed)
        if np.array_equal(updated, current):
            break
        current = updated
    return current.astype(np.uint8) * 255


def _angle_delta_deg(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % 180.0
    return min(delta, 180.0 - delta)


def detect_surface_defects(
    rgb: np.ndarray,
    max_boxes: int = 20,
    max_ai_boxes: int = 64,
    protection_boxes: list[tuple[int, int, int, int, float]] | None = None,
    *,
    precision: str = "normal",
) -> SurfaceDefectDetection:
    """Context-aware candidates for scratches/dust/cracks, not a semantic verdict.

    Candidate generation remains deliberately sensitive (multi-scale top/black-hat),
    but promotion now requires local-context evidence.  This mirrors established
    dust/scratch workflows: first detect local outliers/ridges, then reject natural
    image structure using neighborhood, texture and shape information.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Ожидается изображение RGB")
    h, w = rgb.shape[:2]
    if h < 32 or w < 32:
        return SurfaceDefectDetection(0.0, 0, 255.0, 0.20, 100.0, [], [])

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    protection_boxes = protection_boxes or []
    mode = str(precision or "normal").strip().lower()
    if mode == "maximum":
        morphology_sizes = (3, 5, 7, 9, 11, 13, 15, 17, 21, 25, 31, 39, 47, 55)
        primary_robust_sigma = 5.6
        primary_pct = {"bright": 98.80, "dark": 99.30}
        primary_floor = {"bright": 10.0, "dark": 11.0}
        quality_floor_delta = -0.08
        broad_pct = 91.0
        broad_sigma_factor = 2.20
        hough_sensitivity = 1.85
        kept_line_limit = 224
    elif mode == "precise":
        morphology_sizes = (3, 5, 7, 9, 11, 13, 15, 17, 23, 31)
        primary_robust_sigma = 6.1
        primary_pct = {"bright": 99.12, "dark": 99.48}
        primary_floor = {"bright": 10.5, "dark": 11.5}
        quality_floor_delta = -0.05
        broad_pct = 92.5
        broad_sigma_factor = 2.42
        hough_sensitivity = 1.28
        kept_line_limit = 96
    elif mode == "fast":
        morphology_sizes = (5, 9, 15)
        primary_robust_sigma = 7.5
        primary_pct = {"bright": 99.60, "dark": 99.78}
        primary_floor = {"bright": 13.0, "dark": 13.0}
        quality_floor_delta = 0.0
        broad_pct = 95.0
        broad_sigma_factor = 2.8
        hough_sensitivity = 0.90
        kept_line_limit = 30
    else:
        # Surface v9 defaults to a more complete candidate search. Because repair
        # is manual-only, recall is more valuable than hiding every weak candidate.
        morphology_sizes = (3, 5, 7, 9, 13, 15, 17, 23, 31)
        primary_robust_sigma = 6.5
        primary_pct = {"bright": 99.25, "dark": 99.58}
        primary_floor = {"bright": 11.5, "dark": 12.0}
        quality_floor_delta = -0.035
        broad_pct = 93.3
        broad_sigma_factor = 2.55
        hough_sensitivity = 1.14
        kept_line_limit = 64
    bright_responses: list[np.ndarray] = []
    dark_responses: list[np.ndarray] = []
    response_by_size: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for size in morphology_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        bright_part = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        dark_part = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        bright_responses.append(bright_part)
        dark_responses.append(dark_part)
        response_by_size[int(size)] = (bright_part, dark_part)
    # Exact Surface v8 response is retained for the legacy Hough pass. Extra v9
    # scales feed the deeper pass without perturbing already-proven crack geometry.
    for legacy_size in (5, 9, 15):
        if legacy_size not in response_by_size:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (legacy_size, legacy_size))
            response_by_size[legacy_size] = (
                cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel),
                cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel),
            )
    legacy_bright_response = np.maximum.reduce([response_by_size[k][0] for k in (5, 9, 15)]).astype(np.float32)
    legacy_dark_response = np.maximum.reduce([response_by_size[k][1] for k in (5, 9, 15)]).astype(np.float32)
    bright_response = np.maximum.reduce(bright_responses).astype(np.float32)
    dark_response = np.maximum.reduce(dark_responses).astype(np.float32)

    mx = max(2, round(w * 0.018))
    my = max(2, round(h * 0.018))
    image_area = float(h * w)
    max_area = max(30.0, image_area * 0.0035)
    min_area = max(3, round(image_area / 1_500_000))
    accepted_mask = np.zeros((h, w), np.uint8)
    candidates: list[tuple[float, dict[str, object]]] = []
    thresholds: list[float] = []

    def process(response: np.ndarray, polarity: str) -> None:
        interior = response[my : h - my, mx : w - mx]
        if interior.size == 0:
            interior = response
        med = float(np.median(interior))
        mad = float(np.median(np.abs(interior - med)))
        robust = med + primary_robust_sigma * max(1.4826 * mad, 1.0)
        pct = primary_pct[polarity]
        percentile = float(np.percentile(interior, pct))
        floor = primary_floor[polarity]
        threshold = float(np.clip(max(floor, robust, percentile), floor, 78.0))
        thresholds.append(threshold)

        mask = (response >= threshold).astype(np.uint8) * 255
        mask[:my, :] = 0
        mask[h - my :, :] = 0
        mask[:, :mx] = 0
        mask[:, w - mx :] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for idx in range(1, count):
            x, y, bw, bh, area = (int(v) for v in stats[idx])
            if area < min_area or bw <= 0 or bh <= 0:
                continue
            if x <= mx or y <= my or x + bw >= w - mx or y + bh >= h - my:
                continue

            long_side = max(bw, bh)
            short_side = max(1, min(bw, bh))
            axis_aspect = long_side / short_side
            fill = area / float(bw * bh)
            comp_mask = labels[y : y + bh, x : x + bw] == idx

            # Axis-aligned bounding boxes make a diagonal scratch look almost square.
            # Measure geometry in the component's own principal-axis frame instead.
            cy, cx = np.where(comp_mask)
            oriented_aspect = axis_aspect
            oriented_length = float(long_side)
            oriented_width = float(short_side)
            component_linearity = 0.0
            orientation_deg = 90.0 if bh >= bw else 0.0
            stroke_width = float(short_side)
            if int(comp_mask.sum()) > 0:
                padded = np.pad(comp_mask.astype(np.uint8), 1, mode="constant", constant_values=0)
                dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
                stroke_width = max(1.0, float(dist.max()) * 2.0)
            if len(cx) >= 3:
                coords = np.column_stack([cx.astype(np.float32), cy.astype(np.float32)])
                centered = coords - coords.mean(axis=0, keepdims=True)
                cov = (centered.T @ centered) / max(1, len(coords) - 1)
                vals, vecs = np.linalg.eigh(cov)
                order = np.argsort(vals)
                major = vecs[:, order[-1]]
                minor = vecs[:, order[0]]
                orientation_deg = float((np.degrees(np.arctan2(float(major[1]), float(major[0]))) + 180.0) % 180.0)
                major_proj = centered @ major
                minor_proj = centered @ minor
                oriented_length = float(major_proj.max() - major_proj.min() + 1.0)
                oriented_width = float(minor_proj.max() - minor_proj.min() + 1.0)
                oriented_aspect = oriented_length / max(oriented_width, 1.0)
                component_linearity = float(np.clip(1.0 - max(float(vals[order[0]]), 1e-6) / max(float(vals[order[-1]]), 1e-6), 0.0, 1.0))

            long_thin_line = (
                oriented_aspect >= 4.0 and stroke_width <= 8.5 and oriented_length >= 18.0
                and component_linearity >= 0.88 and area <= image_area * 0.012
            )
            if area > max_area and not long_thin_line:
                continue

            candidate_kind = "irregular"
            is_broad_line = False
            if polarity == "bright":
                is_spot = area <= 96 and long_side <= 16
                is_line = (
                    oriented_aspect >= 2.4 and stroke_width <= 10.0
                    and oriented_length >= 7.0 and component_linearity >= 0.72
                )
                # Old archival emulsion loss is often a *band*, not a 2 px scratch.
                # V7 rejected these as "too thick", which is exactly why the wide
                # white damage across the reference face was never offered for repair.
                # Keep the geometry conservative and let Context/Surface AI decide
                # whether the broad bright ridge is physical damage or real content.
                is_broad_line = (
                    oriented_aspect >= 2.2 and stroke_width <= 22.0
                    and oriented_length >= 18.0 and component_linearity >= 0.64
                    and fill <= 0.58 and area <= max_area
                )
                is_irregular = oriented_aspect >= 1.7 and fill <= 0.28 and area <= max_area * 0.45
                accepted = is_spot or is_line or is_broad_line or is_irregular or long_thin_line
                polarity_weight = 1.35
            else:
                is_spot = area <= 42 and long_side <= 11 and fill >= 0.22
                is_line = (
                    oriented_aspect >= 4.0 and stroke_width <= 7.5
                    and oriented_length >= 14.0 and component_linearity >= 0.88
                )
                is_irregular = oriented_aspect >= 3.0 and fill <= 0.24 and oriented_length >= 12
                accepted = is_spot or is_line or is_irregular or long_thin_line
                polarity_weight = 0.62
            if not accepted:
                continue
            if is_spot:
                candidate_kind = "spot"
            elif is_line or is_broad_line or long_thin_line:
                candidate_kind = "line"
            strength = float(response[y : y + bh, x : x + bw][comp_mask].mean()) if area else 0.0
            context = _component_context_features(
                gray, lab, labels, idx, x, y, bw, bh, polarity, candidate_kind=candidate_kind
            )
            quality = float(context["candidate_quality"])
            semantic_risk = 0.0
            if protection_boxes and area > 0:
                gx = x + cx
                gy = y + cy
                for px, py, pw, ph, weight in protection_boxes:
                    if pw <= 0 or ph <= 0:
                        continue
                    inside = (gx >= px) & (gx < px + pw) & (gy >= py) & (gy < py + ph)
                    overlap = float(np.mean(inside)) if inside.size else 0.0
                    semantic_risk = max(semantic_risk, overlap * float(np.clip(weight, 0.0, 1.0)))
            if semantic_risk > 0.0:
                # A defect can cross a face, so this is never a veto. Dark lines in
                # protected semantic regions are much more likely to be hair/eyes.
                penalty = semantic_risk * (0.34 if polarity == "dark" else 0.12)
                quality = float(np.clip(quality - penalty, 0.0, 1.0))

            # Candidate generation remains broad, but very weak local-context matches
            # are discarded before they ever reach the AI refiner.
            quality_floor = (0.22 if polarity == "bright" else 0.31) + quality_floor_delta
            if candidate_kind == "spot":
                quality_floor = max(quality_floor, (0.52 if polarity == "bright" else 0.68) + quality_floor_delta * 0.6)
            if quality < quality_floor:
                continue

            strength_score = float(np.clip(strength / max(threshold * 1.8, 1.0), 0.0, 1.0))
            rank = (
                strength * np.sqrt(max(area, 1)) * (1.0 + min(oriented_aspect, 14.0) / 11.0)
                * polarity_weight * (0.62 + 0.85 * quality) * (0.85 + 0.25 * strength_score)
            )
            contour_mask = comp_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_norm: list[dict[str, float]] = []
            if contours:
                contour = max(contours, key=cv2.contourArea)
                perimeter = float(cv2.arcLength(contour, True))
                epsilon = max(0.65, perimeter * 0.008)
                approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                if len(approx) > 64:
                    stride = int(np.ceil(len(approx) / 64.0))
                    approx = approx[::stride]
                contour_norm = [
                    {"x": (x + int(px)) / w, "y": (y + int(py)) / h}
                    for px, py in approx
                ]

            box: dict[str, object] = {
                "x": x / w,
                "y": y / h,
                "w": bw / w,
                "h": bh / h,
                "strength": float(np.clip(strength / 255.0, 0.0, 1.0)),
                "polarity": polarity,
                "candidate_kind": candidate_kind,
                "damage_width_class": "broad" if is_broad_line else "thin",
                "oriented_aspect": float(oriented_aspect),
                "oriented_length_px": float(oriented_length),
                "oriented_width_px": float(oriented_width),
                "stroke_width_px": float(stroke_width),
                "orientation_deg": float(orientation_deg),
                "candidate_quality": quality,
                "parallel_neighbor_risk": 0.0,
                "linearity": float(context["linearity"]),
                "context_contrast": float(context["context_contrast"]),
                "side_similarity": float(context["side_similarity"]),
                "texture_risk": float(context["texture_risk"]),
                "chroma_risk": float(context["chroma_risk"]),
                "semantic_risk": float(semantic_risk),
                "contour": contour_norm,
            }
            accepted_roi = accepted_mask[y : y + bh, x : x + bw]
            accepted_roi[comp_mask] = 255
            candidates.append((float(rank), box))

    process(bright_response, "bright")
    process(dark_response, "dark")

    # Surface v8 broad bright-emulsion-loss branch.  The ordinary top-hat detector
    # is intentionally tuned for thin scratches; wide archival abrasion bands can
    # therefore disappear at their centre and survive every repair pass.  Detect
    # those bands from a locally normalised luminance residual, then subject them to
    # the same Context/AI refinement as every other candidate.
    gray_f = gray.astype(np.float32)
    broad_sigma = float(np.clip(min(h, w) / 90.0, 5.0, 10.0))
    broad_background = cv2.GaussianBlur(gray_f, (0, 0), broad_sigma, borderType=cv2.BORDER_REFLECT101)
    broad_response = gray_f - broad_background
    broad_interior = broad_response[my : h - my, mx : w - mx]
    if broad_interior.size == 0:
        broad_interior = broad_response
    broad_med = float(np.median(broad_interior))
    broad_mad = float(np.median(np.abs(broad_interior - broad_med)))
    broad_sigma_robust = max(1.0, 1.4826 * broad_mad)
    broad_threshold = float(np.clip(
        max(14.0, broad_med + broad_sigma_factor * broad_sigma_robust, float(np.percentile(broad_interior, broad_pct))),
        14.0, 38.0,
    ))
    broad_mask = (broad_response >= broad_threshold).astype(np.uint8) * 255
    broad_mask[:my, :] = 0; broad_mask[h - my :, :] = 0
    broad_mask[:, :mx] = 0; broad_mask[:, w - mx :] = 0
    broad_mask = cv2.morphologyEx(
        broad_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    count_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(broad_mask, connectivity=8)
    for idx in range(1, count_b):
        x, y, bw, bh, area = (int(v) for v in stats_b[idx])
        if area < max(18, min_area * 3) or area > max_area * 1.35 or bw <= 0 or bh <= 0:
            continue
        if x <= mx or y <= my or x + bw >= w - mx or y + bh >= h - my:
            continue
        comp = labels_b[y:y+bh, x:x+bw] == idx
        cy, cx = np.where(comp)
        if len(cx) < 12:
            continue
        coords = np.column_stack([cx.astype(np.float32), cy.astype(np.float32)])
        centered = coords - coords.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(1, len(coords) - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)
        major = vecs[:, order[-1]].astype(np.float32)
        minor = vecs[:, order[0]].astype(np.float32)
        major_proj = centered @ major; minor_proj = centered @ minor
        length = float(major_proj.max() - major_proj.min() + 1.0)
        width = float(minor_proj.max() - minor_proj.min() + 1.0)
        aspect = length / max(width, 1.0)
        linearity = float(np.clip(1.0 - max(float(vals[order[0]]), 1e-6) / max(float(vals[order[-1]]), 1e-6), 0.0, 1.0))
        padded = np.pad(comp.astype(np.uint8), 1, mode="constant", constant_values=0)
        dist = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        stroke_width = max(1.0, float(dist.max()) * 2.0)
        fill = area / float(max(1, bw * bh))
        if not (
            aspect >= 2.0 and 5.0 <= stroke_width <= 26.0 and length >= 22.0
            and linearity >= 0.60 and fill <= 0.72
        ):
            continue
        context = _component_context_features(
            gray, lab, labels_b, idx, x, y, bw, bh, "bright", candidate_kind="line"
        )
        quality = float(context["candidate_quality"])
        semantic_risk = 0.0
        if protection_boxes:
            gx = x + cx; gy = y + cy
            for px, py, pw, ph, weight in protection_boxes:
                if pw <= 0 or ph <= 0:
                    continue
                inside = (gx >= px) & (gx < px + pw) & (gy >= py) & (gy < py + ph)
                overlap = float(np.mean(inside)) if inside.size else 0.0
                semantic_risk = max(semantic_risk, overlap * float(np.clip(weight, 0.0, 1.0)))
        # Bright physical loss may legitimately cross a face/eye. Penalise it, do
        # not erase it from the candidate pool before the learned verifier sees it.
        quality = float(np.clip(quality - semantic_risk * 0.06, 0.0, 1.0))
        if quality < max(0.34, 0.44 + quality_floor_delta):
            continue

        center = coords.mean(axis=0)
        pmin = center + major * float(major_proj.min())
        pmax = center + major * float(major_proj.max())
        x1, y1 = float(x + pmin[0]), float(y + pmin[1])
        x2, y2 = float(x + pmax[0]), float(y + pmax[1])
        orientation_deg = float((np.degrees(np.arctan2(float(major[1]), float(major[0]))) + 180.0) % 180.0)
        contour_mask = comp.astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_norm: list[dict[str, float]] = []
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, max(0.8, perimeter * 0.008), True).reshape(-1, 2)
            if len(approx) > 72:
                approx = approx[::int(np.ceil(len(approx) / 72.0))]
            contour_norm = [{"x": (x + int(px)) / w, "y": (y + int(py)) / h} for px, py in approx]
        strength = float(np.mean(broad_response[y:y+bh, x:x+bw][comp]))
        box: dict[str, object] = {
            "x": x / w, "y": y / h, "w": bw / w, "h": bh / h,
            "strength": float(np.clip(strength / 128.0, 0.0, 1.0)),
            "polarity": "bright", "candidate_kind": "line",
            "detection_branch": "broad_bright_ridge", "damage_width_class": "broad",
            "oriented_aspect": aspect, "oriented_length_px": length,
            "oriented_width_px": width, "stroke_width_px": float(stroke_width),
            "orientation_deg": orientation_deg, "candidate_quality": quality,
            "parallel_neighbor_risk": 0.0, "linearity": float(context["linearity"]),
            "context_contrast": float(context["context_contrast"]),
            "side_similarity": float(context["side_similarity"]),
            "texture_risk": float(context["texture_risk"]),
            "chroma_risk": float(context["chroma_risk"]), "semantic_risk": float(semantic_risk),
            "contour": contour_norm,
            "line_endpoints": [
                {"x": float(np.clip(x1 / w, 0.0, 1.0)), "y": float(np.clip(y1 / h, 0.0, 1.0))},
                {"x": float(np.clip(x2 / w, 0.0, 1.0)), "y": float(np.clip(y2 / h, 0.0, 1.0))},
            ],
        }
        rank = float(strength * np.sqrt(max(area, 1)) * (0.72 + quality) * (1.0 + min(aspect, 10.0) / 12.0))
        candidates.append((rank, box))

    # Low-contrast continuation branch.  Connected components alone are a poor fit
    # for old cracked photographs: a real crack can cross a face, meet a wrinkle and
    # become one large non-linear blob.  Probabilistic Hough segments let us recover
    # the actual narrow ridge inside that blob, while a two-sided intensity profile
    # rejects ordinary one-sided edges.  These candidates remain advisory and still
    # pass through the normal Context/Surface-AI verifier; Surface repair itself is manual-only.
    illumination_sigma = max(3.0, min(h, w) / 150.0)
    local_illumination = cv2.GaussianBlur(gray, (0, 0), illumination_sigma, borderType=cv2.BORDER_REFLECT101)
    illum_interior = local_illumination[my : h - my, mx : w - mx]
    if illum_interior.size == 0:
        illum_interior = local_illumination
    legacy_illum_floor = float(np.clip(np.percentile(illum_interior, 35.0), 95.0, 145.0))
    sensitive_illum_floor = float(np.clip(
        np.percentile(illum_interior, 32.0 if mode in {"normal", "precise"} else (27.0 if mode == "maximum" else 35.0)),
        90.0 if mode == "maximum" else 95.0, 145.0,
    ))
    # Preserve the proven Surface v8 pass exactly, then add a more sensitive v9
    # pass. A deeper search must be monotonic: it may add candidates, not erase
    # previously recoverable crack geometry by perturbing Hough's input mask.
    eligible_low_contrast_legacy = local_illumination >= legacy_illum_floor
    eligible_low_contrast_sensitive = local_illumination >= sensitive_illum_floor
    for eligible_mask in (eligible_low_contrast_legacy, eligible_low_contrast_sensitive):
        eligible_mask[:my, :] = False
        eligible_mask[h - my :, :] = False
        eligible_mask[:, :mx] = False
        eligible_mask[:, w - mx :] = False
    lab_f = lab.astype(np.float32)

    def process_low_contrast_hough(response: np.ndarray, polarity: str, *, legacy: bool = False) -> None:
        eligible = eligible_low_contrast_legacy if legacy else eligible_low_contrast_sensitive
        values = response[eligible]
        if values.size < 128:
            return
        local_med = float(np.median(values))
        local_mad = float(np.median(np.abs(values - local_med)))
        local_sigma = max(1.0, 1.4826 * local_mad)
        effective_sensitivity = 1.0 if legacy else hough_sensitivity
        if polarity == "bright":
            strong_floor = 22.0 / effective_sensitivity
            weak_floor = 16.0 / effective_sensitivity
            strong_threshold = float(np.clip(max(strong_floor, local_med + (4.5 / effective_sensitivity) * local_sigma), strong_floor, 52.0))
            weak_threshold = float(np.clip(max(weak_floor, local_med + (3.0 / effective_sensitivity) * local_sigma), weak_floor, strong_threshold))
            quality_floor = 0.40 if legacy else max(0.30, 0.40 + quality_floor_delta)
            min_support = 0.54 if legacy else max(0.44, 0.54 - (effective_sensitivity - 1.0) * 0.16)
            min_side_similarity = 0.34 if legacy else max(0.28, 0.34 - (effective_sensitivity - 1.0) * 0.10)
            max_texture_risk = 0.72 if legacy else min(0.78, 0.72 + (effective_sensitivity - 1.0) * 0.08)
            polarity_weight = 0.72
            growth_steps = 5 if legacy else (9 if mode == "maximum" else (6 if mode == "precise" else 5))
        else:
            # Dark-line mode is deliberately stricter: hair, seams and text dominate
            # this polarity on real photographs.
            dark_gain = 1.0 if legacy else (1.0 + max(0.0, effective_sensitivity - 1.0) * 0.45)
            strong_floor = 30.0 / dark_gain
            weak_floor = 23.0 / dark_gain
            strong_threshold = float(np.clip(max(strong_floor, local_med + (6.0 / dark_gain) * local_sigma), strong_floor, 68.0))
            weak_threshold = float(np.clip(max(weak_floor, local_med + (4.0 / dark_gain) * local_sigma), weak_floor, strong_threshold))
            quality_floor = 0.58 if legacy else max(0.50, 0.58 + quality_floor_delta * 0.6)
            min_support = 0.68 if legacy else max(0.62, 0.68 - max(0.0, effective_sensitivity - 1.0) * 0.08)
            min_side_similarity = 0.62 if legacy else max(0.56, 0.62 - max(0.0, effective_sensitivity - 1.0) * 0.08)
            max_texture_risk = 0.50 if legacy else min(0.56, 0.50 + max(0.0, effective_sensitivity - 1.0) * 0.05)
            polarity_weight = 0.34
            growth_steps = 4 if legacy else (7 if mode == "maximum" else (5 if mode == "precise" else 4))
        thresholds.append(strong_threshold)

        strong = (response >= strong_threshold) & eligible
        weak = (response >= weak_threshold) & eligible
        mask = _bounded_hysteresis_mask(strong, weak, max_steps=growth_steps)
        if not np.any(mask):
            return
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        if legacy:
            min_line_length = max(14, int(round(min(h, w) * 0.035)))
            hough_votes = max(10, int(round(min_line_length * 0.70)))
            max_line_gap = max(4, int(round(min(h, w) * 0.012)))
            hough_theta = np.pi / 180.0
        else:
            min_line_length = max(8 if mode == "maximum" else 12 if mode == "precise" else 14, int(round(min(h, w) * (0.022 if mode == "maximum" else 0.030 if mode == "precise" else 0.035))))
            hough_votes = max(7 if mode == "maximum" else 9 if mode == "precise" else 10, int(round(min_line_length * (0.50 if mode == "maximum" else 0.62 if mode == "precise" else 0.70))))
            max_line_gap = max(4, int(round(min(h, w) * (0.022 if mode == "maximum" else 0.015 if mode == "precise" else 0.012))))
            hough_theta = np.pi / (720.0 if mode == "maximum" else 180.0)
        lines = cv2.HoughLinesP(
            mask,
            1.0,
            hough_theta,
            threshold=hough_votes,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None:
            return

        raw_lines: list[tuple[float, dict[str, object], tuple[float, float, float, float]]] = []
        border_x = max(mx, int(round(w * 0.028)))
        border_y = max(my, int(round(h * 0.028)))
        half_width = float(np.clip(min(h, w) * 0.0032, 1.5, 4.0))

        def sample_nearest(array: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
            xi = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
            yi = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
            return array[yi, xi]

        for row in lines[:, 0, :]:
            x1, y1, x2, y2 = (float(v) for v in row)
            dx, dy = x2 - x1, y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min_line_length:
                continue
            # Suppress the scanner/photo outer frame, not a tear that merely reaches it.
            near_left = max(x1, x2) <= border_x * 1.6
            near_right = min(x1, x2) >= w - border_x * 1.6
            near_top = max(y1, y2) <= border_y * 1.6
            near_bottom = min(y1, y2) >= h - border_y * 1.6
            if (near_left or near_right) and abs(dy) >= abs(dx) * 1.35:
                continue
            if (near_top or near_bottom) and abs(dx) >= abs(dy) * 1.35:
                continue

            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux
            count_samples = max(10, int(round(length)) + 1)
            t = np.linspace(0.0, 1.0, count_samples, dtype=np.float32)
            xs = x1 + dx * t
            ys = y1 + dy * t
            offset = max(2.5, half_width * 1.8)
            center = sample_nearest(gray_f, xs, ys).astype(np.float32)
            side1 = sample_nearest(gray_f, xs + nx * offset, ys + ny * offset).astype(np.float32)
            side2 = sample_nearest(gray_f, xs - nx * offset, ys - ny * offset).astype(np.float32)
            if polarity == "bright":
                delta1 = center - side1
                delta2 = center - side2
            else:
                delta1 = side1 - center
                delta2 = side2 - center
            both_delta = np.minimum(delta1, delta2)
            signed_delta = float(np.median(both_delta))
            support = float(np.mean(both_delta >= 4.0))
            side_gap = float(np.median(np.abs(side1 - side2)))
            side_similarity = float(np.clip(1.0 - side_gap / 24.0, 0.0, 1.0))
            contrast_score = float(np.clip((signed_delta - 1.5) / 10.0, 0.0, 1.0))
            support_score = float(np.clip((support - 0.25) / 0.65, 0.0, 1.0))
            length_score = float(np.clip((length - min_line_length) / max(90.0, min(h, w) * 0.22), 0.0, 1.0))

            xa = max(0, int(np.floor(min(x1, x2) - 8.0)))
            xb = min(w, int(np.ceil(max(x1, x2) + 9.0)))
            ya = max(0, int(np.floor(min(y1, y2) - 8.0)))
            yb = min(h, int(np.ceil(max(y1, y2) + 9.0)))
            roi = gray[ya:yb, xa:xb]
            edges = cv2.Canny(roi, 55, 145) if roi.size else np.empty((0, 0), np.uint8)
            edge_fraction = float(np.mean(edges > 0)) if edges.size else 0.0
            texture_risk = float(np.clip((edge_fraction - 0.05) / 0.22, 0.0, 1.0))

            center_ab = sample_nearest(lab_f[..., 1:3], xs, ys).reshape(-1, 2)
            side_ab = np.concatenate([
                sample_nearest(lab_f[..., 1:3], xs + nx * offset, ys + ny * offset).reshape(-1, 2),
                sample_nearest(lab_f[..., 1:3], xs - nx * offset, ys - ny * offset).reshape(-1, 2),
            ], axis=0)
            chroma_delta = float(np.linalg.norm(np.median(center_ab, axis=0) - np.median(side_ab, axis=0)))
            chroma_risk = float(np.clip((chroma_delta / max(6.0, abs(signed_delta)) - 0.55) / 1.6, 0.0, 1.0))

            quality = float(np.clip(
                0.40 * contrast_score
                + 0.24 * support_score
                + 0.20 * side_similarity
                + 0.16 * length_score
                - 0.18 * texture_risk
                - 0.08 * chroma_risk,
                0.0,
                1.0,
            ))
            if support < min_support or side_similarity < min_side_similarity or texture_risk > max_texture_risk:
                continue

            semantic_risk = 0.0
            if protection_boxes:
                for px, py, pw, ph, weight in protection_boxes:
                    if pw <= 0 or ph <= 0:
                        continue
                    inside = (xs >= px) & (xs < px + pw) & (ys >= py) & (ys < py + ph)
                    overlap = float(np.mean(inside)) if inside.size else 0.0
                    semantic_risk = max(semantic_risk, overlap * float(np.clip(weight, 0.0, 1.0)))
            if semantic_risk:
                quality = float(np.clip(quality - semantic_risk * (0.10 if polarity == "bright" else 0.30), 0.0, 1.0))
            if quality < quality_floor:
                continue

            response_values = sample_nearest(response, xs, ys).astype(np.float32)
            strength = float(np.mean(response_values))
            angle = float((np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0)
            # Narrow rotated polygon: safe for overlay and for bounded inpainting.
            corners = np.asarray([
                [x1 + nx * half_width, y1 + ny * half_width],
                [x2 + nx * half_width, y2 + ny * half_width],
                [x2 - nx * half_width, y2 - ny * half_width],
                [x1 - nx * half_width, y1 - ny * half_width],
            ], dtype=np.float32)
            corners[:, 0] = np.clip(corners[:, 0], 0.0, w - 1.0)
            corners[:, 1] = np.clip(corners[:, 1], 0.0, h - 1.0)
            bx0, by0 = corners.min(axis=0)
            bx1, by1 = corners.max(axis=0)
            bw, bh = max(1.0, float(bx1 - bx0 + 1.0)), max(1.0, float(by1 - by0 + 1.0))
            contour_norm = [{"x": float(px / w), "y": float(py / h)} for px, py in corners]
            box: dict[str, object] = {
                "x": float(bx0 / w), "y": float(by0 / h), "w": float(bw / w), "h": float(bh / h),
                "strength": float(np.clip(strength / 255.0, 0.0, 1.0)),
                "polarity": polarity,
                "candidate_kind": "line",
                "detection_branch": "low_contrast_hough",
                # Keep the provenance of the additive v9 search. Sensitive-only
                # candidates may be penalised by the denser candidate field, but
                # they must never lower the ranking of geometry already found by
                # the proven v8 pass.
                "surface_search_pass": "legacy_v8" if legacy else "sensitive_v9",
                "oriented_aspect": float(length / max(half_width * 2.0, 1.0)),
                "oriented_length_px": length,
                "oriented_width_px": float(half_width * 2.0),
                "stroke_width_px": float(half_width * 2.0),
                "orientation_deg": angle,
                "candidate_quality": quality,
                "parallel_neighbor_risk": 0.0,
                "linearity": 1.0,
                "context_contrast": contrast_score,
                "side_similarity": side_similarity,
                "texture_risk": texture_risk,
                "chroma_risk": chroma_risk,
                "semantic_risk": float(semantic_risk),
                "contour": contour_norm,
                "line_endpoints": [
                    {"x": float(x1 / w), "y": float(y1 / h)},
                    {"x": float(x2 / w), "y": float(y2 / h)},
                ],
            }
            rank = float(
                strength * np.sqrt(max(length * half_width * 2.0, 1.0))
                * polarity_weight * (0.55 + quality) * (1.0 + min(length, 180.0) / 180.0)
            )
            raw_lines.append((rank, box, (x1, y1, x2, y2)))

        # Hough returns many near-identical lines through the same crack. Keep one
        # representative for overlapping collinear segments, but preserve adjacent
        # bent segments so a curved crack is represented as a chain rather than one
        # unsafe giant rectangle.
        raw_lines.sort(key=lambda item: item[0], reverse=True)
        kept: list[tuple[float, dict[str, object], tuple[float, float, float, float]]] = []
        for item in raw_lines:
            _, box, segment = item
            x1, y1, x2, y2 = segment
            length = float(box["oriented_length_px"])
            angle = float(box["orientation_deg"])
            ux = (x2 - x1) / max(length, 1e-6)
            uy = (y2 - y1) / max(length, 1e-6)
            nx, ny = -uy, ux
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            duplicate = False
            for _, other, other_segment in kept:
                if _angle_delta_deg(angle, float(other["orientation_deg"])) > 9.0:
                    continue
                ox1, oy1, ox2, oy2 = other_segment
                olen = float(other["oriented_length_px"])
                ocx, ocy = (ox1 + ox2) * 0.5, (oy1 + oy2) * 0.5
                perp = abs((ocx - cx) * nx + (ocy - cy) * ny)
                if perp > max(5.0, half_width * 2.2):
                    continue
                p1, p2 = x1 * ux + y1 * uy, x2 * ux + y2 * uy
                q1, q2 = ox1 * ux + oy1 * uy, ox2 * ux + oy2 * uy
                amin, amax = min(p1, p2), max(p1, p2)
                bmin, bmax = min(q1, q2), max(q1, q2)
                overlap = max(0.0, min(amax, bmax) - max(amin, bmin))
                gap = max(0.0, max(amin, bmin) - min(amax, bmax))
                if overlap >= 0.42 * min(length, olen) or gap <= 3.0:
                    duplicate = True
                    break
            if duplicate:
                continue
            kept.append(item)
            if len(kept) >= kept_line_limit:
                break

        for rank, box, _ in kept:
            contour = np.asarray([
                [int(round(float(point["x"]) * w)), int(round(float(point["y"]) * h))]
                for point in box["contour"]
            ], dtype=np.int32)
            if len(contour) >= 3:
                cv2.fillPoly(accepted_mask, [contour], 255)
            candidates.append((rank, box))

    process_low_contrast_hough(legacy_bright_response, "bright", legacy=True)
    process_low_contrast_hough(legacy_dark_response, "dark", legacy=True)
    if mode != "fast":
        process_low_contrast_hough(bright_response, "bright", legacy=False)
        process_low_contrast_hough(dark_response, "dark", legacy=False)

    # Repeated, near-parallel thin lines are much more often fabric, blinds, text
    # strokes or other deliberate scene structure than independent film scratches.
    # A second-stage neighborhood check can see this pattern; an isolated-patch
    # classifier cannot.  This is a soft penalty until the repetition becomes
    # unmistakable, then very weak candidates are removed before AI evaluation.
    adjusted: list[tuple[float, dict[str, object]]] = []
    line_rows = [item for item in candidates if str(item[1].get("candidate_kind", "")) == "line"]
    baseline_line_rows = [
        item for item in line_rows
        if str(item[1].get("surface_search_pass", "")) != "sensitive_v9"
    ]
    for rank, box in candidates:
        if str(box.get("candidate_kind", "")) != "line":
            adjusted.append((rank, box))
            continue
        # The extra v9 pass is additive.  Its denser set of Hough lines must not
        # make an old v8/primary candidate look artificially repetitive.
        repetition_rows = (
            line_rows
            if str(box.get("surface_search_pass", "")) == "sensitive_v9"
            else baseline_line_rows
        )
        branch_name = str(box.get("detection_branch", "") or "")
        try:
            texture_now = float(box.get("texture_risk", 0.0) or 0.0)
            context_now = float(box.get("context_contrast", 0.0) or 0.0)
            quality_now = float(box.get("candidate_quality", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            texture_now = context_now = quality_now = 0.0
        # When recall thresholds are lowered, tiny bright gaps inside printed text
        # can resemble a short scratch. Very high edge texture plus almost no local
        # ridge contrast is a strong hard-negative signal and does not describe an
        # isolated physical crack. Keep this guard independent of precision mode.
        if branch_name in {"", "primary"} and texture_now >= 0.92 and context_now < 0.22 and quality_now < 0.30:
            continue
        if len(repetition_rows) < 4:
            adjusted.append((rank, box))
            continue

        angle = float(box.get("orientation_deg", 0.0) or 0.0)
        cx = float(box.get("x", 0.0)) + float(box.get("w", 0.0)) * 0.5
        cy = float(box.get("y", 0.0)) + float(box.get("h", 0.0)) * 0.5
        similar = 0
        for _, other in repetition_rows:
            if other is box:
                continue
            other_angle = float(other.get("orientation_deg", 0.0) or 0.0)
            da = abs(angle - other_angle) % 180.0
            da = min(da, 180.0 - da)
            if da > 8.0:
                continue
            ocx = float(other.get("x", 0.0)) + float(other.get("w", 0.0)) * 0.5
            ocy = float(other.get("y", 0.0)) + float(other.get("h", 0.0)) * 0.5
            if np.hypot(cx - ocx, cy - ocy) <= 0.48:
                similar += 1
        repetition_risk = float(np.clip((similar - 2) / 7.0, 0.0, 1.0))
        if repetition_risk > 0.0:
            old_quality = float(box.get("candidate_quality", 0.0) or 0.0)
            polarity = str(box.get("polarity", "bright"))
            branch = str(box.get("detection_branch", "primary"))
            if branch == "low_contrast_hough":
                # The Hough branch is intentionally sensitive. Repetition is therefore
                # much stronger evidence against it than against a primary morphology
                # candidate; this kills blinds/fabric/text strokes before AI evaluation.
                penalty = repetition_risk * (0.92 if polarity == "dark" else 0.72)
                floor = 0.58 if polarity == "dark" else 0.40
            else:
                penalty = repetition_risk * (0.57 if polarity == "dark" else 0.42)
                floor = 0.31 if polarity == "dark" else 0.22
            new_quality = float(np.clip(old_quality - penalty, 0.0, 1.0))
            box["parallel_neighbor_risk"] = repetition_risk
            box["candidate_quality"] = new_quality
            rank *= 0.45 + 0.55 * (new_quality / max(old_quality, 0.05))
            if new_quality < floor:
                continue
        adjusted.append((rank, box))
    candidates = adjusted

    candidates.sort(key=lambda item: item[0], reverse=True)
    deduped = _deduplicate_ranked_candidates(candidates, iou_threshold=0.45)

    shown = deduped[:max(0, int(max_boxes))]
    ai_candidates = deduped[:max(0, int(max_ai_boxes))]
    density = float(np.mean(accepted_mask > 0) * 100.0)
    quality_mean = float(np.mean([float(box.get("candidate_quality", 0.0)) for _, box in shown])) if shown else 0.0
    score = _clip_score(100.0 - density * 13.0 - min(len(deduped), 30) * 0.38 - quality_mean * min(len(shown), 12) * 0.22)
    if not shown:
        confidence = 0.54
    else:
        avg_strength = float(np.mean([float(box["strength"]) for _, box in shown]))
        # Still below automatic-action territory.  Context quality raises our
        # confidence in the *candidate map*, not in a semantic defect verdict.
        confidence = float(np.clip(0.39 + avg_strength * 0.06 + quality_mean * 0.05, 0.39, 0.49))

    return SurfaceDefectDetection(
        candidate_density=density,
        candidate_count=len(deduped),
        threshold=max(thresholds) if thresholds else 255.0,
        confidence=confidence,
        score=score,
        boxes_norm=[box for _, box in shown],
        ai_boxes_norm=[box for _, box in ai_candidates],
    )
