from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations

import cv2
import numpy as np

from .faces import FaceRegion, _iou, _normalized_box_to_pixels, _cached_cascade


@dataclass(slots=True)
class EyeRegion:
    face_index: int
    x: int
    y: int
    w: int
    h: int
    laplacian: float
    tenengrad: float
    sharpness_score: float
    measurement_confidence: float
    source: str = "auto"

    def to_raw(self, image_w: int, image_h: int) -> dict[str, float | int]:
        return {
            "face_index": self.face_index,
            "x": self.x / image_w,
            "y": self.y / image_h,
            "w": self.w / image_w,
            "h": self.h / image_h,
            "laplacian": self.laplacian,
            "tenengrad": self.tenengrad,
            "sharpness_score": self.sharpness_score,
            "measurement_confidence": self.measurement_confidence,
            "source": self.source,
        }


@dataclass(slots=True)
class EyeAnalysis:
    detector_available: bool
    eyes: list[EyeRegion]
    detector_confidence: float
    faces_with_eyes: int
    detector_name: str = "opencv_haar_eye_quad_v3"


def _clip100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _filter_candidates(face: FaceRegion, candidates) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for box in candidates:
        x, y, w, h = (int(v) for v in box)
        cx = (x + w / 2.0) / max(face.w, 1)
        cy = (y + h / 2.0) / max(face.h, 1)
        size_ratio = max(w, h) / max(min(face.w, face.h), 1)
        aspect = w / max(h, 1)
        if not (0.08 <= cx <= 0.92):
            continue
        if not (0.18 <= cy <= 0.53):
            continue
        if not (0.08 <= size_ratio <= 0.24):
            continue
        if not (0.65 <= aspect <= 1.55):
            continue
        out.append((x, y, w, h))
    return out


def _candidate_cost(face: FaceRegion, box: tuple[int, int, int, int]) -> float:
    x, y, w, h = box
    cx = (x + w / 2.0) / max(face.w, 1)
    cy = (y + h / 2.0) / max(face.h, 1)
    # Distance to either expected eye location in a frontal face.
    dx = min(abs(cx - 0.32), abs(cx - 0.68))
    return dx * 1.6 + abs(cy - 0.37) * 1.2 + abs((w + h) / (2 * max(face.w, 1)) - 0.14)


def select_eye_candidates(face: FaceRegion, candidates) -> list[tuple[int, int, int, int]]:
    """Select at most two geometrically plausible eyes inside one face."""
    filtered = _filter_candidates(face, candidates)
    if len(filtered) <= 1:
        return filtered

    best_pair = None
    best_cost = 1e9
    for left, right in combinations(filtered, 2):
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        lc = ((lx + lw / 2.0) / face.w, (ly + lh / 2.0) / face.h)
        rc = ((rx + rw / 2.0) / face.w, (ry + rh / 2.0) / face.h)
        if lc[0] > rc[0]:
            left, right = right, left
            lx, ly, lw, lh = left
            rx, ry, rw, rh = right
            lc = ((lx + lw / 2.0) / face.w, (ly + lh / 2.0) / face.h)
            rc = ((rx + rw / 2.0) / face.w, (ry + rh / 2.0) / face.h)
        separation = rc[0] - lc[0]
        if not (0.18 <= separation <= 0.62):
            continue
        if abs(lc[1] - rc[1]) > 0.11:
            continue
        size_ratio = max(lw * lh, rw * rh) / max(min(lw * lh, rw * rh), 1)
        if size_ratio > 2.0:
            continue
        cost = _candidate_cost(face, left) + _candidate_cost(face, right) + abs(lc[1] - rc[1])
        if cost < best_cost:
            best_cost = cost
            best_pair = [left, right]
    if best_pair is not None:
        return best_pair
    return [min(filtered, key=lambda box: _candidate_cost(face, box))]


def measure_eye_regions(
    rgb: np.ndarray,
    face_index: int,
    boxes: list[tuple[int, int, int, int]],
    pair_complete: bool,
    *,
    source: str = "auto",
) -> list[EyeRegion]:
    h_img, w_img = rgb.shape[:2]
    out: list[EyeRegion] = []
    for x, y, w, h in boxes:
        x = max(0, int(x)); y = max(0, int(y))
        w = min(int(w), w_img - x); h = min(int(h), h_img - y)
        if w < 8 or h < 8:
            continue
        roi = rgb[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ten = float(np.mean(gx * gx + gy * gy))
        lap_score = _clip100(20.0 * np.log10(1.0 + lap))
        ten_score = _clip100(14.0 * np.log10(1.0 + ten))
        sharpness = (lap_score + ten_score) / 2.0
        short_side = min(w, h)
        confidence = float(np.clip(0.30 + min(short_side, 34) / 34.0 * 0.22 + (0.08 if pair_complete else 0.0), 0.30, 0.60))
        out.append(EyeRegion(face_index, x, y, w, h, lap, ten, sharpness, confidence, source))
    return out


def analyze_eyes(
    rgb: np.ndarray,
    faces: list[FaceRegion],
    manual_boxes: list[dict[str, float]] | None = None,
    *,
    precision: str = "normal",
) -> EyeAnalysis:
    mode = str(precision or "normal").strip().lower()
    detector_name = "opencv_haar_eye_quad_v5_maximum" if mode == "maximum" else ("opencv_haar_eye_quad_v4_deep" if mode == "precise" else "opencv_haar_eye_quad_v3")
    base = Path(getattr(cv2.data, "haarcascades", ""))
    cascade_names = (
        "haarcascade_eye_tree_eyeglasses.xml",
        "haarcascade_eye.xml",
        "haarcascade_lefteye_2splits.xml",
        "haarcascade_righteye_2splits.xml",
    )
    cascades: list[cv2.CascadeClassifier] = []
    for name in cascade_names:
        path = base / name
        if not path.is_file():
            continue
        cascade = _cached_cascade(str(path))
        if cascade is not None:
            cascades.append(cascade)
    if not cascades:
        return EyeAnalysis(False, [], 0.0, 0, detector_name=detector_name)
    if not faces:
        return EyeAnalysis(True, [], 0.50, 0, detector_name=detector_name)

    all_eyes: list[EyeRegion] = []
    faces_with_eyes = 0

    for face_index, face in enumerate(faces):
        upper_h = max(16, int(round(face.h * 0.68)))
        # Grayscale conversion is pixel-local. Convert only the face ROI instead
        # of allocating a full-frame gray copy merely to slice a tiny region.
        # This is bit-identical inside the ROI and avoids a large Precise-mode
        # allocation after the face cascade stage.
        face_rgb = rgb[face.y:face.y + upper_h, face.x:face.x + face.w]
        if face_rgb.size == 0:
            continue
        roi = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        equalized = cv2.equalizeHist(roi)
        min_eye = max(6, round(min(face.w, face.h) * 0.06))
        max_eye = max(min_eye + 2, round(min(face.w, face.h) * 0.28))

        combined: list[tuple[int, int, int, int]] = []
        passes: list[tuple[np.ndarray, float, int]] = [(equalized, 1.04, 2)]
        if mode in {"precise", "maximum"}:
            # A denser scale pyramid plus CLAHE catches small/dim eyes that the
            # ordinary equalised pass can miss. Candidate geometry still has
            # to pass select_eye_candidates().
            clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(4, 4)).apply(roi)
            passes.extend(((equalized, 1.025, 3), (clahe, 1.025, 2)))
            if mode == "maximum":
                # Additional fine-scale passes; same geometry guardrails, more search.
                clahe_fine = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(6, 6)).apply(roi)
                passes.extend(((equalized, 1.016, 3), (clahe, 1.016, 3), (clahe_fine, 1.018, 2)))

        for view, scale_factor, neighbors in passes:
            for cascade in cascades:
                detected = cascade.detectMultiScale(
                    view,
                    scaleFactor=scale_factor,
                    minNeighbors=neighbors,
                    minSize=(min_eye, min_eye),
                    maxSize=(max_eye, max_eye),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                combined.extend(tuple(int(v) for v in box) for box in detected)

        selected = select_eye_candidates(face, combined)
        if selected:
            faces_with_eyes += 1
        global_boxes = [(face.x + x, face.y + y, w, h) for x, y, w, h in selected]
        all_eyes.extend(measure_eye_regions(rgb, face_index, global_boxes, pair_complete=len(selected) == 2))

    # Manual eye markup is a fail-safe for old/tilted/partly closed eyes. The box is
    # assigned to the face containing its centre and replaces overlapping automatic
    # eye candidates. It never creates a face by itself: the user marks a face first.
    h_img, w_img = rgb.shape[:2]
    for raw_box in manual_boxes or []:
        box = _normalized_box_to_pixels(raw_box, w_img, h_img, min_size=8)
        if box is None:
            continue
        x, y, w, h = box
        cx = x + w / 2.0; cy = y + h / 2.0
        face_index = -1
        for idx, face in enumerate(faces):
            if face.x <= cx <= face.x + face.w and face.y <= cy <= face.y + face.h:
                face_index = idx
                break
        if face_index < 0:
            continue
        all_eyes = [
            eye for eye in all_eyes
            if not (
                eye.face_index == face_index
                and _iou((eye.x, eye.y, eye.w, eye.h), box) >= 0.28
            )
        ]
        measured = measure_eye_regions(rgb, face_index, [box], False, source="manual")
        if measured:
            measured[0].measurement_confidence = max(measured[0].measurement_confidence, 0.96)
            all_eyes.extend(measured)

    faces_with_eyes = len({eye.face_index for eye in all_eyes})

    if all_eyes:
        coverage = faces_with_eyes / max(len(faces), 1)
        paired_faces = sum(1 for idx in range(len(faces)) if sum(eye.face_index == idx for eye in all_eyes) >= 2)
        pair_bonus = 0.04 * (paired_faces / max(len(faces), 1))
        detector_confidence = float(np.clip(0.48 + 0.15 * coverage + pair_bonus, 0.48, 0.67))
    else:
        detector_confidence = 0.38
    return EyeAnalysis(True, all_eyes, detector_confidence, faces_with_eyes, detector_name=detector_name)
