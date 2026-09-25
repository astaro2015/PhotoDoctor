from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .eyes import EyeRegion
from .faces import FaceRegion


@dataclass(slots=True)
class RedEyeCandidate:
    face_index: int
    eye_index: int
    x: int
    y: int
    w: int
    h: int
    red_pixel_ratio: float
    red_pixel_count: int
    mean_red_excess: float
    mean_red_level: float
    severity: float
    confidence: float
    eye_source: str = "auto"
    relaxed_detection: bool = False
    component_center_distance: float = 1.0
    component_circularity: float = 0.0
    component_fill: float = 0.0
    local_red_contrast: float = 0.0
    red_dominance: float = 0.0
    catchlight_nearby: bool = False
    geometry_pair_confirmed: bool = False
    confirmation_score: float = 0.0

    def to_raw(self, image_w: int, image_h: int) -> dict[str, float | int | bool]:
        return {
            "face_index": self.face_index,
            "eye_index": self.eye_index,
            "x": self.x / image_w,
            "y": self.y / image_h,
            "w": self.w / image_w,
            "h": self.h / image_h,
            "red_pixel_ratio": self.red_pixel_ratio,
            "red_pixel_count": self.red_pixel_count,
            "mean_red_excess": self.mean_red_excess,
            "mean_red_level": self.mean_red_level,
            "severity": self.severity,
            "confidence": self.confidence,
            "eye_source": self.eye_source,
            "relaxed_detection": self.relaxed_detection,
            "component_center_distance": self.component_center_distance,
            "component_circularity": self.component_circularity,
            "component_fill": self.component_fill,
            "local_red_contrast": self.local_red_contrast,
            "red_dominance": self.red_dominance,
            "catchlight_nearby": self.catchlight_nearby,
            "geometry_pair_confirmed": self.geometry_pair_confirmed,
            "confirmation_score": self.confirmation_score,
            "suspicious": _candidate_is_suspicious(self),
        }


@dataclass(slots=True)
class RedEyeAnalysis:
    suspicious_eye_count: int
    candidates: list[RedEyeCandidate]
    confidence: float
    detector_available: bool = True
    detector_name: str = "eye_roi_red_reflex_v7_pupil_guard"


@dataclass(slots=True)
class RedEyeCorrectionStats:
    corrected_count: int
    total_mask_pixels: int



def _candidate_is_suspicious(candidate: RedEyeCandidate) -> bool:
    """Require several independent signs before calling a flash reflex red-eye.

    A red patch inside an *expected* eye box is not enough.  The candidate must
    look like a pupil reflex relative to its immediate iris/eye surroundings.
    Geometry-only eye boxes are deliberately fail-closed: because the eye itself
    was not detected, a unilateral red patch cannot become an automatic defect.
    """
    if candidate.red_pixel_count <= 0:
        return False

    manual = candidate.eye_source == "manual"
    geometry = candidate.eye_source == "red_eye_geometry"

    # First reject the common false-positive shape seen on warm eyelids, hair and
    # skin: the red component is near an edge of the guessed eye box and its
    # neighbourhood is almost equally red.  A retinal flash reflex is normally a
    # much stronger local chromatic outlier inside the pupil.
    max_center_distance = 1.02 if manual else (0.68 if geometry else 0.78)
    min_local_contrast = 7.0 if (manual and candidate.relaxed_detection) else (16.0 if geometry else 11.0)
    min_dominance = 1.08 if (manual and candidate.relaxed_detection) else (1.38 if geometry else 1.22)
    if candidate.component_center_distance > max_center_distance:
        return False
    if candidate.local_red_contrast < min_local_contrast:
        return False
    if candidate.red_dominance < min_dominance:
        return False
    if candidate.component_circularity < 0.18 or candidate.component_fill < 0.14:
        return False

    if geometry:
        # Published automatic red-eye systems use facial/binocular geometry as a
        # confirmation stage.  For our geometry fallback the eye was never
        # independently detected, so require matching evidence from the other eye
        # on the same face.  This prevents hair/eyelid/skin from being promoted to
        # a defect merely because it lies where an eye was expected.
        if not candidate.geometry_pair_confirmed:
            return False
        return bool(
            candidate.red_pixel_count >= 3
            and candidate.mean_red_excess >= 22.0
            and candidate.mean_red_level >= 78.0
            and candidate.confirmation_score >= 58.0
        )

    if candidate.relaxed_detection:
        if manual:
            # Manual markup confirms the eye location, therefore a small/JPEG-
            # diluted reflex may be retained, but it still needs local contrast.
            return bool(
                candidate.red_pixel_count >= 2
                and candidate.mean_red_excess >= 14.0
                and candidate.mean_red_level >= 68.0
                and candidate.confirmation_score >= 30.0
            )
        return bool(
            candidate.red_pixel_count >= 3
            and candidate.mean_red_excess >= 22.0
            and candidate.mean_red_level >= 82.0
            and candidate.confirmation_score >= 50.0
        )

    return bool(
        candidate.red_pixel_count >= 2
        and candidate.mean_red_excess >= 18.0
        and candidate.mean_red_level >= 65.0
        and candidate.confirmation_score >= (36.0 if manual else 46.0)
    )


def supplement_red_eye_eye_regions(
    rgb: np.ndarray,
    faces: list[FaceRegion],
    eyes: list[EyeRegion],
) -> list[EyeRegion]:
    """Add geometry-only eye ROIs for faces where Haar missed one/both eyes.

    These synthetic regions are *only* for red-eye search. They are not fed back
    into eye sharpness/quality metrics. This is important on group photos where
    faces can be 30-60 px wide and Haar eye detection often returns nothing.
    """
    h_img, w_img = rgb.shape[:2]
    out = list(eyes)
    existing_by_face: dict[int, list[EyeRegion]] = {}
    for eye in eyes:
        existing_by_face.setdefault(int(eye.face_index), []).append(eye)

    for face_index, face in enumerate(faces):
        face_eyes = existing_by_face.get(face_index, [])
        occupied = set()
        for eye in face_eyes:
            cx = eye.x + eye.w * 0.5
            rel_x = (cx - face.x) / max(face.w, 1)
            occupied.add("left" if rel_x < 0.5 else "right")
        if len(occupied) >= 2:
            continue

        bw = max(6, int(round(face.w * 0.26)))
        bh = max(5, int(round(face.h * 0.20)))
        cy = face.y + face.h * 0.38
        for side, cx_ratio in (("left", 0.31), ("right", 0.69)):
            if side in occupied:
                continue
            cx = face.x + face.w * cx_ratio
            x = max(0, int(round(cx - bw / 2.0)))
            y = max(0, int(round(cy - bh / 2.0)))
            w = min(bw, w_img - x)
            h = min(bh, h_img - y)
            if w < 5 or h < 5:
                continue
            confidence = float(np.clip(face.measurement_confidence * 0.65, 0.28, 0.52))
            out.append(EyeRegion(
                face_index=face_index, x=x, y=y, w=w, h=h,
                laplacian=0.0, tenengrad=0.0, sharpness_score=0.0,
                measurement_confidence=confidence, source="red_eye_geometry",
            ))
    return out

def _clip100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ry = max(h * 0.42, 1.0)
    rx = max(w * 0.42, 1.0)
    norm = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return norm <= 1.0


def _central_mask(h: int, w: int) -> np.ndarray:
    y0 = int(round(h * 0.10)); y1 = int(round(h * 0.90))
    x0 = int(round(w * 0.10)); x1 = int(round(w * 0.90))
    mask = np.zeros((h, w), dtype=bool)
    mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return mask


def _catchlight_mask(roi_rgb: np.ndarray) -> np.ndarray:
    """Protect small neutral/near-neutral specular highlights in the pupil."""
    if roi_rgb.size == 0:
        return np.zeros(roi_rgb.shape[:2], dtype=bool)
    arr = roi_rgb.astype(np.float32)
    hi = arr.max(axis=2)
    lo = arr.min(axis=2)
    # A catchlight is bright and approximately neutral. We intentionally keep
    # the threshold permissive because JPEG can tint a white reflection.
    mask = (hi >= 210.0) & ((hi - lo) <= 55.0)
    if mask.any():
        # Protect the immediate antialiased rim as well; this avoids a dark ring
        # around the catchlight after red neutralisation.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return mask


def _compact_components(mask: np.ndarray, roi_rgb: np.ndarray) -> np.ndarray:
    """Keep one compact pupil-like red component, never a skin/eyelid sweep.

    An eye ROI contains plenty of naturally warm pixels.  The old implementation
    could keep several red components at once; on a real portrait that allowed a
    long under-eye skin crescent to join the pupil mask and produced a blue-gray
    "bruise" after red-channel neutralisation.  A single eye has a single pupil,
    so automatic correction deliberately keeps only the strongest compact core.
    """
    if not mask.any():
        return mask
    h, w = mask.shape
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=bool)
    rf = roi_rgb[..., 0].astype(np.float32)
    gf = roi_rgb[..., 1].astype(np.float32)
    bf = roi_rgb[..., 2].astype(np.float32)
    mean_gb = (gf + bf) * 0.5
    red_excess = rf - mean_gb
    min_area = max(2, int(round(min(h, w) * 0.06)))
    max_area = max(min_area + 1, int(round(h * w * 0.12)))
    cx0 = (w - 1) * 0.5
    cy0 = (h - 1) * 0.5

    candidates: list[tuple[float, int]] = []
    for label in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        if area < min_area or area > max_area or bw <= 0 or bh <= 0:
            continue
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        fill = area / float(bw * bh)
        if aspect > 2.0 or fill < 0.18:
            continue

        component = labels == label
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
        circularity = float(np.clip(4.0 * np.pi * area / max(perimeter * perimeter, 1e-6), 0.0, 1.0))
        if circularity < 0.24:
            continue

        cx, cy = centroids[label]
        dx = (float(cx) - cx0) / max(w * 0.42, 1.0)
        dy = (float(cy) - cy0) / max(h * 0.42, 1.0)
        center_dist = float(np.hypot(dx, dy))
        if center_dist > 0.82:
            continue

        excess = float(red_excess[component].mean()) if np.any(component) else 0.0
        component_r = float(rf[component].mean()) if np.any(component) else 0.0
        component_gb = float(mean_gb[component].mean()) if np.any(component) else 1.0
        red_dominance = component_r / max(component_gb, 1.0)
        if red_dominance < 1.15:
            continue

        area_fraction = area / float(max(h * w, 1))
        compact_bonus = 1.0 - min(area_fraction / 0.12, 1.0)
        score = (
            excess
            + 34.0 * (1.0 - center_dist)
            + 28.0 * circularity
            + 8.0 * min(fill, 1.0)
            + 8.0 * compact_bonus
        )
        candidates.append((score, label))

    if not candidates:
        return out
    _score, best_label = max(candidates, key=lambda item: item[0])
    out |= labels == best_label
    return out


def build_red_eye_mask(roi_rgb: np.ndarray, *, relaxed: bool = False) -> np.ndarray:
    if roi_rgb.size == 0:
        return np.zeros(roi_rgb.shape[:2], dtype=bool)
    rf = roi_rgb[..., 0].astype(np.float32)
    gf = roi_rgb[..., 1].astype(np.float32)
    bf = roi_rgb[..., 2].astype(np.float32)
    mean_gb = (gf + bf) * 0.5
    red_excess = rf - mean_gb
    value = np.maximum.reduce([rf, gf, bf])
    ellipse = _ellipse_mask(roi_rgb.shape[0], roi_rgb.shape[1])
    center = _central_mask(roi_rgb.shape[0], roi_rgb.shape[1])
    catchlight = _catchlight_mask(roi_rgb)
    search = ellipse & center & ~catchlight

    if not np.any(search):
        return np.zeros(roi_rgb.shape[:2], dtype=bool)

    # Critical detail: compare redness to the *local eye ROI*, not to a fixed
    # absolute value only. Warm skin/eyelids can have R-(G+B)/2 around 20-35 and
    # used to join the pupil into one huge component which was then rejected.
    # A flash-red pupil is a compact local outlier relative to its surroundings.
    baseline_excess = float(np.percentile(red_excess[search], 35.0))
    gb_delta = np.abs(gf - bf)
    gb_balanced = gb_delta <= np.maximum(12.0, mean_gb * 0.30)

    if relaxed:
        warm = (rf >= 44.0) & (rf > gf * 1.08) & (rf > bf * 1.08)
        local_red = red_excess >= max(12.0, baseline_excess + 8.0)
        not_bright = value < 244.0
        raw = warm & gb_balanced & local_red & not_bright & search
    else:
        warm = (rf >= 50.0) & (rf > gf * 1.13) & (rf > bf * 1.13)
        local_red = red_excess >= max(18.0, baseline_excess + 12.0)
        not_bright = value < 240.0
        raw = warm & gb_balanced & local_red & not_bright & search

    if not raw.any():
        return raw

    # Close tiny JPEG holes but do not broadly dilate the pupil into skin/iris.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = _compact_components(cleaned.astype(bool), roi_rgb)
    mask &= search
    return mask



def _confirmation_features(roi_rgb: np.ndarray, mask: np.ndarray) -> dict[str, float | bool]:
    """Measure pupil-like evidence independent from raw red-pixel thresholds."""
    if roi_rgb.size == 0 or not np.any(mask):
        return {
            "center_distance": 1.5,
            "circularity": 0.0,
            "fill": 0.0,
            "local_red_contrast": 0.0,
            "red_dominance": 0.0,
            "catchlight_nearby": False,
            "confirmation_score": 0.0,
        }

    h, w = mask.shape
    ys, xs = np.where(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    fill = float(mask.sum() / max(bw * bh, 1))

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = float(np.clip(4.0 * np.pi * float(mask.sum()) / max(perimeter * perimeter, 1e-6), 0.0, 1.0))

    cx = float(xs.mean()); cy = float(ys.mean())
    nx = (cx - (w - 1) * 0.5) / max(w * 0.42, 1.0)
    ny = (cy - (h - 1) * 0.5) / max(h * 0.42, 1.0)
    center_distance = float(np.hypot(nx, ny))

    arr = roi_rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mean_gb = (g + b) * 0.5
    red_excess = r - mean_gb
    component_excess = float(red_excess[mask].mean())
    component_r = float(r[mask].mean())
    component_gb = float(mean_gb[mask].mean())
    red_dominance = float(component_r / max(component_gb, 1.0))

    # Compare the red component with an annulus immediately around it.  This is
    # the key discriminator between a retinal reflex and warm skin/hair: the
    # latter tends to stay red in the surrounding ring as well.
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    outer_size = 7 if min(h, w) >= 12 else 5
    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_size, outer_size))
    inner = cv2.dilate(mask.astype(np.uint8), inner_kernel, iterations=1).astype(bool)
    outer = cv2.dilate(mask.astype(np.uint8), outer_kernel, iterations=1).astype(bool)
    ring = outer & ~inner & _ellipse_mask(h, w)
    if np.any(ring):
        surrounding_excess = float(np.median(red_excess[ring]))
    else:
        surrounding_excess = float(np.median(red_excess[~mask])) if np.any(~mask) else component_excess
    local_red_contrast = float(component_excess - surrounding_excess)

    catchlight = _catchlight_mask(roi_rgb)
    near_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    near = cv2.dilate(mask.astype(np.uint8), near_kernel, iterations=1).astype(bool)
    catchlight_nearby = bool(np.any(catchlight & near))

    redness_score = float(np.clip((component_excess - 10.0) / 90.0, 0.0, 1.0))
    local_score = float(np.clip((local_red_contrast - 5.0) / 70.0, 0.0, 1.0))
    dominance_score = float(np.clip((red_dominance - 1.05) / 1.35, 0.0, 1.0))
    center_score = float(np.clip(1.0 - center_distance / 0.95, 0.0, 1.0))
    shape_score = float(np.clip(0.60 * circularity + 0.40 * min(fill / 0.55, 1.0), 0.0, 1.0))
    confirmation_score = 100.0 * (
        0.28 * redness_score
        + 0.25 * local_score
        + 0.16 * dominance_score
        + 0.17 * center_score
        + 0.10 * shape_score
        + 0.04 * (1.0 if catchlight_nearby else 0.0)
    )
    return {
        "center_distance": center_distance,
        "circularity": circularity,
        "fill": fill,
        "local_red_contrast": local_red_contrast,
        "red_dominance": red_dominance,
        "catchlight_nearby": catchlight_nearby,
        "confirmation_score": float(np.clip(confirmation_score, 0.0, 100.0)),
    }


def _confirm_geometry_pairs(candidates: list[RedEyeCandidate]) -> None:
    """Confirm geometry-only eye boxes only when binocular evidence agrees."""
    by_face: dict[int, list[RedEyeCandidate]] = {}
    for candidate in candidates:
        if candidate.red_pixel_count <= 0:
            continue
        by_face.setdefault(int(candidate.face_index), []).append(candidate)

    for face_candidates in by_face.values():
        if len(face_candidates) < 2:
            continue
        for i, left in enumerate(face_candidates[:-1]):
            for right in face_candidates[i + 1:]:
                if left.eye_source != "red_eye_geometry" and right.eye_source != "red_eye_geometry":
                    continue
                # Both sides must already look at least somewhat pupil-like before
                # binocular geometry can boost them.  Pairing two weak skin patches
                # must never manufacture confidence out of nothing.
                if min(left.confirmation_score, right.confirmation_score) < 38.0:
                    continue
                if min(left.local_red_contrast, right.local_red_contrast) < 12.0:
                    continue
                lcx = left.x + left.w * 0.5; lcy = left.y + left.h * 0.5
                rcx = right.x + right.w * 0.5; rcy = right.y + right.h * 0.5
                mean_w = max((left.w + right.w) * 0.5, 1.0)
                mean_h = max((left.h + right.h) * 0.5, 1.0)
                separation = abs(rcx - lcx) / mean_w
                vertical = abs(rcy - lcy) / mean_h
                if not (0.85 <= separation <= 4.5 and vertical <= 0.70):
                    continue
                left.geometry_pair_confirmed = True
                right.geometry_pair_confirmed = True
                left.confirmation_score = min(100.0, left.confirmation_score + 10.0)
                right.confirmation_score = min(100.0, right.confirmation_score + 10.0)

def analyze_red_eye(rgb: np.ndarray, eyes: list[EyeRegion]) -> RedEyeAnalysis:
    h_img, w_img = rgb.shape[:2]
    candidates: list[RedEyeCandidate] = []
    suspicious = 0
    for eye_index, eye in enumerate(eyes):
        x = max(0, int(eye.x)); y = max(0, int(eye.y))
        w = min(int(eye.w), w_img - x); h = min(int(eye.h), h_img - y)
        min_dim = 5 if eye.source == "red_eye_geometry" else 8
        if w < min_dim or h < min_dim:
            continue
        roi = rgb[y:y+h, x:x+w]
        mask = build_red_eye_mask(roi)
        relaxed_detection = False
        # JPEG blending can turn an obvious flash pupil into only a few pink-red
        # pixels. Once the ROI is a *confirmed eye* (auto/manual), a second relaxed
        # pass is safe enough and must not depend on eye-box size. Geometry-only
        # ROIs use the same pass but face stricter final evidence in
        # _candidate_is_suspicious().
        if not mask.any():
            mask = build_red_eye_mask(roi, relaxed=True)
            relaxed_detection = bool(mask.any())
        area = max(int(mask.sum()), 0)
        area_ratio = float(area / max(mask.size, 1))
        if area > 0:
            rf = roi[..., 0].astype(np.float32)
            gf = roi[..., 1].astype(np.float32)
            bf = roi[..., 2].astype(np.float32)
            red_excess = rf - (gf + bf) * 0.5
            mean_red_excess = float(red_excess[mask].mean())
            mean_red_level = float(rf[mask].mean())
            compact_bonus = min(1.0, area / max(min(w, h) * 2.0, 1.0))
            severity = _clip100(area_ratio * 1500.0 + max(0.0, mean_red_excess - 24.0) * 1.2)
            confidence = float(np.clip(
                eye.measurement_confidence
                + 0.18 * compact_bonus
                + 0.10 * min(mean_red_excess / 60.0, 1.0)
                - (0.05 if eye.source == "red_eye_geometry" else 0.0)
                - (0.04 if relaxed_detection else 0.0),
                0.30, 0.96,
            ))
        else:
            mean_red_excess = 0.0
            mean_red_level = 0.0
            severity = 0.0
            confidence = float(np.clip(eye.measurement_confidence, 0.30, 0.80))
        features = _confirmation_features(roi, mask)
        candidate = RedEyeCandidate(
            face_index=eye.face_index,
            eye_index=eye_index,
            x=x,
            y=y,
            w=w,
            h=h,
            red_pixel_ratio=area_ratio,
            red_pixel_count=area,
            mean_red_excess=mean_red_excess,
            mean_red_level=mean_red_level,
            severity=severity,
            confidence=confidence,
            eye_source=eye.source,
            relaxed_detection=relaxed_detection,
            component_center_distance=float(features["center_distance"]),
            component_circularity=float(features["circularity"]),
            component_fill=float(features["fill"]),
            local_red_contrast=float(features["local_red_contrast"]),
            red_dominance=float(features["red_dominance"]),
            catchlight_nearby=bool(features["catchlight_nearby"]),
            confirmation_score=float(features["confirmation_score"]),
        )
        candidates.append(candidate)
    _confirm_geometry_pairs(candidates)
    suspicious = sum(1 for candidate in candidates if _candidate_is_suspicious(candidate))
    if not candidates:
        return RedEyeAnalysis(0, [], 0.0)
    if suspicious:
        conf = float(np.clip(
            sum(
                0.45 * c.confidence + 0.55 * (c.confirmation_score / 100.0)
                for c in candidates if _candidate_is_suspicious(c)
            ) / max(suspicious, 1),
            0.42, 0.96,
        ))
    else:
        conf = float(np.clip(sum(c.confidence for c in candidates) / len(candidates) * 0.8, 0.20, 0.75))
    return RedEyeAnalysis(suspicious, candidates, conf)


def _resolve_candidate_box(
    item: dict | RedEyeCandidate,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    if isinstance(item, RedEyeCandidate):
        x, y, w, h = float(item.x), float(item.y), float(item.w), float(item.h)
        normalized = False
    else:
        # Keep sub-unit coordinates as floats until coordinate-space detection.
        # The old code converted 0.42 -> 0 before this check, corrupting real ROIs.
        x = float(item.get("x", 0.0) or 0.0)
        y = float(item.get("y", 0.0) or 0.0)
        w = float(item.get("w", 0.0) or 0.0)
        h = float(item.get("h", 0.0) or 0.0)
        normalized = all(0.0 <= v <= 1.0 for v in (x, y, w, h))
    if normalized:
        x *= image_w; y *= image_h; w *= image_w; h *= image_h
    xi = max(0, int(round(x)))
    yi = max(0, int(round(y)))
    wi = max(0, int(round(w)))
    hi = max(0, int(round(h)))
    wi = min(wi, image_w - xi)
    hi = min(hi, image_h - yi)
    return xi, yi, wi, hi


def _correction_alpha(roi_u8: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    h, w = mask.shape
    rf = roi_u8[..., 0].astype(np.float32)
    gf = roi_u8[..., 1].astype(np.float32)
    bf = roi_u8[..., 2].astype(np.float32)
    mean_gb = (gf + bf) * 0.5
    excess = np.maximum(rf - mean_gb, 0.0)
    red_weight = np.clip((excess - 12.0) / 55.0, 0.0, 1.0)

    # Feather *inside* the verified pupil mask only.  We intentionally do not
    # dilate the support: even a one-pixel spill onto warm lower-eyelid skin can
    # become a visible blue/gray halo when only the red channel is reduced.
    sigma = max(0.45, min(w, h) * 0.035)
    feather = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = feather * (0.28 + 0.72 * red_weight) * float(strength)
    alpha *= mask.astype(np.float32)
    alpha[_catchlight_mask(roi_u8)] = 0.0
    return np.clip(alpha, 0.0, 1.0)


def correct_red_eye(
    rgb: np.ndarray,
    candidates: list[dict] | list[RedEyeCandidate],
    *,
    strength: float = 1.0,
    relaxed: bool = False,
) -> tuple[np.ndarray, RedEyeCorrectionStats]:
    out = rgb.copy().astype(np.float32)
    strength = float(np.clip(strength, 0.0, 1.0))
    corrected_count = 0
    total_mask_pixels = 0
    h_img, w_img = out.shape[:2]

    for item in candidates:
        x, y, w, h = _resolve_candidate_box(item, w_img, h_img)
        if w < 8 or h < 8:
            continue
        roi = out[y:y+h, x:x+w]
        roi_u8 = np.clip(np.rint(roi), 0, 255).astype(np.uint8)
        mask = build_red_eye_mask(roi_u8, relaxed=relaxed)
        pixels = int(mask.sum())
        if pixels <= 0:
            continue

        alpha = _correction_alpha(roi_u8, mask, strength)
        if float(alpha.max()) <= 0.0:
            continue

        r = roi[..., 0].copy()
        g = roi[..., 1].copy()
        b = roi[..., 2].copy()
        mean_gb = (g + b) * 0.5

        # Red-eye is a red-channel flash reflex. Correct that channel only.
        # Keeping G/B intact preserves the natural iris colour and texture; reducing
        # R towards local G/B also naturally restores the dark pupil without painting
        # it flat black/gray.
        target_r = np.minimum(r, mean_gb * 1.02)
        roi[..., 0] = r * (1.0 - alpha) + target_r * alpha

        # Preserve catchlights exactly. Alpha should already be zero there; explicit
        # copy makes this invariant robust against future mask changes.
        catchlight = _catchlight_mask(roi_u8)
        if np.any(catchlight):
            roi[catchlight] = roi_u8.astype(np.float32)[catchlight]

        corrected_count += 1
        total_mask_pixels += pixels

    return (
        np.clip(np.rint(out), 0, 255).astype(np.uint8),
        RedEyeCorrectionStats(corrected_count, total_mask_pixels),
    )
