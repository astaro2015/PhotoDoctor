from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np

from .loader import srgb_to_linear


@dataclass(slots=True)
class FaceRegion:
    x: int
    y: int
    w: int
    h: int
    brightness_linear: float
    laplacian: float
    tenengrad: float
    sharpness_score: float
    measurement_confidence: float
    source: str = "auto"

    def to_raw(self, image_w: int, image_h: int) -> dict[str, float]:
        return {
            "x": self.x / image_w,
            "y": self.y / image_h,
            "w": self.w / image_w,
            "h": self.h / image_h,
            "brightness_linear": self.brightness_linear,
            "laplacian": self.laplacian,
            "tenengrad": self.tenengrad,
            "sharpness_score": self.sharpness_score,
            "measurement_confidence": self.measurement_confidence,
            "source": self.source,
        }


@dataclass(slots=True)
class FaceAnalysis:
    detector_available: bool
    faces: list[FaceRegion]
    detector_confidence: float
    detector_name: str = "opencv_haar_frontal_rotation_multiscale_v5_eye_consensus"


@lru_cache(maxsize=16)
def _cached_cascade(path_text: str) -> cv2.CascadeClassifier | None:
    """Load one immutable Haar cascade once per process.

    Face confirmation and eye analysis reuse the same OpenCV XML classifiers
    many times.  Re-parsing those files for every candidate adds pure startup/I/O
    overhead and does not change any detection parameters or pixels.
    """
    path = Path(path_text)
    if not path.is_file():
        return None
    cascade = cv2.CascadeClassifier(str(path))
    return None if cascade.empty() else cascade


def _clip100(v: float) -> float:
    return float(np.clip(v, 0.0, 100.0))


def _linear_luma(rgb: np.ndarray) -> float:
    # Decoded photos are uint8. Reuse the loader's exact 256-entry sRGB transfer
    # LUT instead of evaluating pow/where again for every detected face.
    linear = srgb_to_linear(rgb)
    y = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    return float(np.mean(y))


def measure_face_regions(
    rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], *, source: str = "auto"
) -> list[FaceRegion]:
    h_img, w_img = rgb.shape[:2]
    out: list[FaceRegion] = []
    for x, y, w, h in boxes:
        x = max(0, int(x)); y = max(0, int(y))
        w = min(int(w), w_img - x); h = min(int(h), h_img - y)
        if w < 24 or h < 24:
            continue
        roi = rgb[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        ten = float(np.mean(gx * gx + gy * gy))
        lap_score = _clip100(20.0 * np.log10(1.0 + lap))
        ten_score = _clip100(14.0 * np.log10(1.0 + ten))
        sharpness = (lap_score + ten_score) / 2.0
        short_side = min(w, h)
        # Small face crops are intrinsically less reliable for detail assessment.
        measurement_confidence = float(np.clip(0.42 + min(short_side, 220) / 220.0 * 0.38, 0.42, 0.80))
        out.append(FaceRegion(
            x=x,
            y=y,
            w=w,
            h=h,
            brightness_linear=_linear_luma(roi),
            laplacian=lap,
            tenengrad=ten,
            sharpness_score=sharpness,
            measurement_confidence=measurement_confidence,
            source=source,
        ))
    return out


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(union, 1))


def _detect_with_weight(
    cascade: cv2.CascadeClassifier,
    gray: np.ndarray,
    *,
    scale_factor: float,
    min_neighbors: int,
    min_face: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Return rectangles plus cascade level weights when OpenCV exposes them."""
    try:
        rects, _reject, weights = cascade.detectMultiScale3(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face, min_face),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True,
        )
        return [
            (tuple(int(v) for v in rect), float(weight))
            for rect, weight in zip(rects, weights)
        ]
    except (AttributeError, cv2.error):
        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(min_face, min_face),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [(tuple(int(v) for v in rect), 0.0) for rect in rects]


@dataclass(slots=True)
class _EyeEvidence:
    count: int
    robust_pair: bool
    supported_eye_count: int


def _plausible_eye_evidence(
    gray: np.ndarray, box: tuple[int, int, int, int], cascade_dir: Path
) -> _EyeEvidence:
    """Cross-check eye geometry with several independent Haar eye cascades.

    A floral pattern can fool one eye cascade surprisingly well.  A real frontal
    face, however, usually yields approximately co-located responses from more
    than one eye model.  We keep the old permissive count as recovery evidence
    but expose a stronger ``robust_pair`` signal for automatic face acceptance.
    """
    x, y, w, h = box
    upper_h = max(16, int(round(h * 0.68)))
    roi = gray[y:y + upper_h, x:x + w]
    if roi.size == 0:
        return _EyeEvidence(0, False, 0)
    eq = cv2.equalizeHist(roi)
    min_eye = max(6, round(min(w, h) * 0.06))
    max_eye = max(min_eye + 2, round(min(w, h) * 0.28))
    plausible: list[tuple[float, float, int, int, str]] = []
    eye_specs = (
        ("tree", "haarcascade_eye_tree_eyeglasses.xml"),
        ("generic", "haarcascade_eye.xml"),
        ("left_split", "haarcascade_lefteye_2splits.xml"),
        ("right_split", "haarcascade_righteye_2splits.xml"),
    )
    for family, name in eye_specs:
        path = cascade_dir / name
        if not path.is_file():
            continue
        eye_cascade = _cached_cascade(str(path))
        if eye_cascade is None:
            continue
        detected = eye_cascade.detectMultiScale(
            eq,
            scaleFactor=1.04,
            minNeighbors=2,
            minSize=(min_eye, min_eye),
            maxSize=(max_eye, max_eye),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for ex, ey, ew, eh in detected:
            cx = (int(ex) + int(ew) / 2.0) / max(w, 1)
            cy = (int(ey) + int(eh) / 2.0) / max(h, 1)
            size_ratio = max(int(ew), int(eh)) / max(min(w, h), 1)
            aspect = int(ew) / max(int(eh), 1)
            if not (0.08 <= cx <= 0.92 and 0.16 <= cy <= 0.56):
                continue
            if not (0.07 <= size_ratio <= 0.29 and 0.60 <= aspect <= 1.70):
                continue
            plausible.append((cx, cy, int(ew), int(eh), family))

    if not plausible:
        return _EyeEvidence(0, False, 0)

    # Cluster co-located responses from independent eye cascades.  The source set
    # is the useful part: two generic texture hits do not become two confirmed
    # eyes merely because their rectangles happen to be horizontal.
    clusters: list[dict[str, object]] = []
    for cx, cy, ew, eh, family in sorted(plausible, key=lambda item: item[2] * item[3], reverse=True):
        hit = next(
            (row for row in clusters
             if abs(cx - float(row["cx"])) < 0.085 and abs(cy - float(row["cy"])) < 0.085),
            None,
        )
        if hit is None:
            clusters.append({"cx": cx, "cy": cy, "ew": ew, "eh": eh, "sources": {family}})
        else:
            sources = hit.get("sources")
            if isinstance(sources, set):
                sources.add(family)
            # Keep the larger crop as a stable centre estimate.
            if ew * eh > int(hit["ew"]) * int(hit["eh"]):
                hit["cx"], hit["cy"], hit["ew"], hit["eh"] = cx, cy, ew, eh

    supported_eye_count = sum(
        1 for row in clusters if isinstance(row.get("sources"), set) and len(row["sources"]) >= 2
    )
    if len(clusters) < 2:
        return _EyeEvidence(len(clusters), False, supported_eye_count)

    clusters.sort(key=lambda row: float(row["cx"]))
    found_pair = False
    robust_pair = False
    for left_index in range(len(clusters) - 1):
        for right_index in range(left_index + 1, len(clusters)):
            left = clusters[left_index]; right = clusters[right_index]
            lx, ly = float(left["cx"]), float(left["cy"])
            rx, ry = float(right["cx"]), float(right["cy"])
            separation = rx - lx
            size_similarity = min(int(left["ew"]), int(right["ew"])) / max(int(left["ew"]), int(right["ew"]), 1)
            if not (0.18 <= separation <= 0.64 and abs(ly - ry) <= 0.14 and size_similarity >= 0.52):
                continue
            found_pair = True
            left_sources = left.get("sources", set())
            right_sources = right.get("sources", set())
            if (isinstance(left_sources, set) and isinstance(right_sources, set)
                    and len(left_sources) >= 2 and len(right_sources) >= 2):
                robust_pair = True
                break
        if robust_pair:
            break
    return _EyeEvidence(2 if found_pair else 1, robust_pair, supported_eye_count)


def _plausible_eye_count(gray: np.ndarray, box: tuple[int, int, int, int], cascade_dir: Path) -> int:
    return _plausible_eye_evidence(gray, box, cascade_dir).count


def _merge_face_candidates(
    primary: list[tuple[tuple[int, int, int, int], float]],
    secondary: list[tuple[tuple[int, int, int, int], float]],
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for source, candidates in (("default", primary), ("alt", secondary)):
        for box, weight in candidates:
            hit = next((entry for entry in merged if _iou(entry["box"], box) >= 0.34), None)
            if hit is None:
                merged.append({"box": box, "sources": {source}, "weights": {source: weight}})
                continue
            hit["sources"].add(source)
            hit["weights"][source] = max(float(hit["weights"].get(source, -1e9)), float(weight))
            # Prefer the larger overlapping crop; it is usually the more complete face ROI.
            old_box = hit["box"]
            if box[2] * box[3] > old_box[2] * old_box[3]:
                hit["box"] = box
    return merged


def _box_source_rank(source: str) -> tuple[int, int]:
    text = str(source)
    parts = text.split("@")
    family = _source_family(text)
    angle = 0.0
    if len(parts) >= 3:
        try:
            angle = float(parts[-1])
        except ValueError:
            angle = 0.0
    zero_angle = abs(angle) < 1e-6
    family_rank = {"default": 0, "alt2": 1, "alt": 2}.get(family, 3)
    return (0 if zero_angle else 1, family_rank)


def _merge_labeled_face_candidates(
    groups: list[tuple[str, list[tuple[tuple[int, int, int, int], float]]]],
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for source, candidates in groups:
        for box, weight in candidates:
            hit = next((entry for entry in merged if _iou(entry["box"], box) >= 0.34), None)
            if hit is None:
                merged.append({
                    "box": box, "box_source": source, "sources": {source}, "weights": {source: weight}
                })
                continue
            hit["sources"].add(source)
            hit["weights"][source] = max(float(hit["weights"].get(source, -1e9)), float(weight))
            old_source = str(hit.get("box_source", ""))
            old_rank = _box_source_rank(old_source)
            new_rank = _box_source_rank(source)
            # Rotated detections become larger axis-aligned rectangles after mapping
            # back. Prefer an unrotated/default crop whenever one exists; this keeps
            # the face ROI tight enough for eye detection instead of swallowing hair,
            # shoulders or the neighbouring face.
            replace = new_rank < old_rank
            if new_rank == old_rank and _source_family(source) == _source_family(old_source):
                old_weight = float(hit["weights"].get(old_source, -1e9))
                replace = float(weight) > old_weight
            if replace:
                hit["box"] = box
                hit["box_source"] = source
    return merged


def _rescale_detections(
    rows: list[tuple[tuple[int, int, int, int], float]], scale: float,
) -> list[tuple[tuple[int, int, int, int], float]]:
    if abs(scale - 1.0) < 1e-6:
        return rows
    out: list[tuple[tuple[int, int, int, int], float]] = []
    inv = 1.0 / max(scale, 1e-9)
    for (x, y, w, h), weight in rows:
        out.append(((
            int(round(x * inv)), int(round(y * inv)),
            int(round(w * inv)), int(round(h * inv)),
        ), weight))
    return out


def _rotate_gray(gray: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate around the image centre without changing canvas size.

    Returns the rotated image and the inverse affine transform that maps points
    from the rotated canvas back into the pre-rotation image coordinates.
    """
    if abs(float(angle_deg)) < 1e-6:
        inv = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return gray, inv
    h, w = gray.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    return rotated, cv2.invertAffineTransform(matrix)


def _unrotate_detections(
    rows: list[tuple[tuple[int, int, int, int], float]],
    inverse_matrix: np.ndarray,
    width: int,
    height: int,
) -> list[tuple[tuple[int, int, int, int], float]]:
    out: list[tuple[tuple[int, int, int, int], float]] = []
    m = np.asarray(inverse_matrix, dtype=np.float64)
    for (x, y, w, h), weight in rows:
        corners = np.asarray(
            [[x, y, 1.0], [x + w, y, 1.0], [x, y + h, 1.0], [x + w, y + h, 1.0]],
            dtype=np.float64,
        )
        mapped = corners @ m.T
        x0 = max(0, int(np.floor(np.min(mapped[:, 0]))))
        y0 = max(0, int(np.floor(np.min(mapped[:, 1]))))
        x1 = min(width, int(np.ceil(np.max(mapped[:, 0]))))
        y1 = min(height, int(np.ceil(np.max(mapped[:, 1]))))
        if x1 - x0 >= 24 and y1 - y0 >= 24:
            out.append(((x0, y0, x1 - x0, y1 - y0), float(weight)))
    return out


def _source_family(name: str) -> str:
    text = str(name)
    if text.startswith("default"):
        return "default"
    if text.startswith("alt2"):
        return "alt2"
    if text.startswith("alt"):
        return "alt"
    return text.split("@", 1)[0]


def _source_angle(name: str) -> float:
    parts = str(name).split("@")
    if len(parts) < 3:
        return 0.0
    try:
        return float(parts[-1])
    except ValueError:
        return 0.0


def _accept_face_candidate(entry: dict[str, object], eye_count: int) -> bool:
    sources = entry.get("sources", set())
    weights = entry.get("weights", {})
    if not isinstance(sources, set):
        sources = set()
    if not isinstance(weights, dict):
        weights = {}
    families = {_source_family(str(name)) for name in sources}
    default_sources = [str(name) for name in sources if _source_family(str(name)) == "default"]
    alternate_sources = [str(name) for name in sources if _source_family(str(name)) in {"alt", "alt2"}]
    zero_default = [name for name in default_sources if abs(_source_angle(name)) < 1e-6]
    primary_weight = max((float(weights.get(name, 0.0)) for name in zero_default), default=0.0)
    multi_family = len(families & {"default", "alt", "alt2"}) >= 2
    zero_angle_families = {
        _source_family(str(name)) for name in sources if abs(_source_angle(str(name))) < 1e-6
    }
    zero_angle_multi_family = len(zero_angle_families & {"default", "alt", "alt2"}) >= 2
    alternate_angles = {round(_source_angle(name), 1) for name in alternate_sources}
    repeated_alternate_angles = len(alternate_angles) >= 2

    # Strong unrotated default evidence may survive closed/obscured eyes. Rotated
    # detections are recovery evidence only: they need eye or independent-cascade
    # support because furniture/clothing can produce very convincing Haar hits.
    if primary_weight >= 4.0:
        return True
    if zero_angle_multi_family:
        return True
    if eye_count >= 2:
        return True
    if eye_count >= 1 and repeated_alternate_angles:
        return True
    return False


def _accept_face_candidate_strict(
    entry: dict[str, object], eye_evidence: _EyeEvidence, relative_size: float
) -> bool:
    """Precision-first gate used for automatic face boxes.

    Haar face cascades are excellent proposal generators but patterned fabric and
    flowers can receive absurdly confident scores.  Small candidates therefore
    need independently corroborated eye geometry.  Large portrait faces retain a
    conservative recovery path for closed/occluded eyes.
    """
    if not _accept_face_candidate(entry, eye_evidence.count):
        return False
    sources = entry.get("sources", set())
    weights = entry.get("weights", {})
    if not isinstance(sources, set):
        sources = set()
    if not isinstance(weights, dict):
        weights = {}

    zero_default = [
        str(name) for name in sources
        if _source_family(str(name)) == "default" and abs(_source_angle(str(name))) < 1e-6
    ]
    primary_weight = max((float(weights.get(name, 0.0)) for name in zero_default), default=-99.0)
    zero_families = {
        _source_family(str(name)) for name in sources if abs(_source_angle(str(name))) < 1e-6
    }
    multi_family = len(zero_families & {"default", "alt", "alt2"}) >= 2
    default_scales = {
        str(name).split("@")[1] for name in zero_default if len(str(name).split("@")) >= 2
    }
    repeated_default = len(default_scales) >= 2

    if eye_evidence.robust_pair:
        return True
    if eye_evidence.count >= 2:
        return bool(
            relative_size >= 0.105
            and primary_weight >= 3.5
            and (multi_family or repeated_default)
        )
    if eye_evidence.count == 1:
        return bool(
            relative_size >= 0.18
            and primary_weight >= 5.0
            and multi_family
            and repeated_default
        )
    return bool(
        relative_size >= 0.24
        and primary_weight >= 6.5
        and multi_family
        and repeated_default
    )


def _normalized_box_to_pixels(
    box: dict[str, float], image_w: int, image_h: int, *, min_size: int = 24
) -> tuple[int, int, int, int] | None:
    try:
        x = float(box.get("x", 0.0)); y = float(box.get("y", 0.0))
        w = float(box.get("w", 0.0)); h = float(box.get("h", 0.0))
    except (TypeError, ValueError):
        return None
    x = min(max(x, 0.0), 1.0); y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0 - x); h = min(max(h, 0.0), 1.0 - y)
    px = int(round(x * image_w)); py = int(round(y * image_h))
    pw = int(round(w * image_w)); ph = int(round(h * image_h))
    if pw < min_size or ph < min_size:
        return None
    return px, py, min(pw, image_w - px), min(ph, image_h - py)


def analyze_faces(
    rgb: np.ndarray, precision: str = "normal", manual_boxes: list[dict[str, float]] | None = None
) -> FaceAnalysis:
    cascade_dir = Path(getattr(cv2.data, "haarcascades", ""))
    cascade_specs = [
        ("default", cascade_dir / "haarcascade_frontalface_default.xml", 1.08, 5),
        ("alt", cascade_dir / "haarcascade_frontalface_alt.xml", 1.05, 3),
        ("alt2", cascade_dir / "haarcascade_frontalface_alt2.xml", 1.05, 3),
    ]
    cascades: list[tuple[str, cv2.CascadeClassifier, float, int]] = []
    for family, path, scale_factor, neighbors in cascade_specs:
        if not path.is_file():
            continue
        cascade = _cached_cascade(str(path))
        if cascade is not None:
            cascades.append((family, cascade, scale_factor, neighbors))
    if not cascades or not any(family == "default" for family, *_rest in cascades):
        return FaceAnalysis(False, [], 0.0)

    original_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h_img, w_img = original_gray.shape[:2]
    long_edge = max(h_img, w_img)
    mode = str(precision or "normal").strip().lower()
    if mode == "fast":
        base_targets = (1150,)
        recovery_angles: tuple[float, ...] = ()
        recovery_target = 1050
        base_families = {"default", "alt"}
        recovery_families: set[str] = set()
    elif mode == "maximum":
        # Keep the face search deep, but spend the extra Maximum budget mainly on
        # spatial maps and Surface defects.  A very dense ±2° rotation sweep was
        # disproportionately expensive and repeated nearly identical Haar work.
        base_targets = (2300, 2000, 1750, 1500, 1250, 1000, 800, 650)
        recovery_angles = (-4.0, 4.0, -8.0, 8.0, -12.0, 12.0, -16.0, 16.0, -22.0, 22.0, -28.0, 28.0)
        recovery_target = 1600
        base_families = {"default", "alt", "alt2"}
        recovery_families = {"default", "alt", "alt2"}
    elif mode == "precise":
        # Deep mode intentionally spends more time on independent scales and
        # modest rotations.  This is where the extra runtime is useful: small
        # and slightly tilted faces are the common miss in group/archival shots.
        base_targets = (1800, 1450, 1100, 800)
        recovery_angles = (-6.0, 6.0, -12.0, 12.0, -20.0, 20.0)
        recovery_target = 1250
        base_families = {"default", "alt", "alt2"}
        recovery_families = {"default", "alt2"}
    else:
        base_targets = (1350, 900)
        recovery_angles = (-10.0, 10.0)
        recovery_target = 1050
        base_families = {"default", "alt", "alt2"}
        recovery_families = {"default", "alt2"}

    groups: list[tuple[str, list[tuple[tuple[int, int, int, int], float]]]] = []
    scaled_gray_cache: dict[int, tuple[np.ndarray, float]] = {}

    def resolved_gray(target: int) -> tuple[np.ndarray, float]:
        resolved = min(long_edge, int(target))
        cached = scaled_gray_cache.get(resolved)
        if cached is not None:
            return cached
        scale = min(1.0, resolved / max(long_edge, 1))
        if scale < 0.999:
            base_gray = cv2.resize(
                original_gray,
                (max(1, int(round(w_img * scale))), max(1, int(round(h_img * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            base_gray = original_gray
        cached = (base_gray, scale)
        scaled_gray_cache[resolved] = cached
        return cached

    def run_pass(target: int, angle: float, families: set[str]) -> None:
        resolved = min(long_edge, int(target))
        base_gray, scale = resolved_gray(resolved)
        scaled_h, scaled_w = base_gray.shape[:2]
        min_side = min(base_gray.shape[:2])
        min_face = max(30, round(min_side * 0.050))
        rotated, inverse = _rotate_gray(base_gray, angle)
        equalized = cv2.equalizeHist(rotated)
        angle_tag = f"{angle:+.0f}"
        for family, cascade, scale_factor, neighbors in cascades:
            if family not in families:
                continue
            rows = _detect_with_weight(
                cascade, equalized, scale_factor=scale_factor, min_neighbors=neighbors, min_face=min_face,
            )
            if abs(angle) > 1e-6:
                rows = _unrotate_detections(rows, inverse, scaled_w, scaled_h)
            rows = _rescale_detections(rows, scale)
            groups.append((f"{family}@{resolved}@{angle_tag}", rows))

    seen_base: set[int] = set()
    for target in base_targets:
        resolved = min(long_edge, int(target))
        if resolved in seen_base:
            continue
        seen_base.add(resolved)
        run_pass(resolved, 0.0, base_families)

    if recovery_angles:
        resolved_recovery = min(long_edge, int(recovery_target))
        for angle in recovery_angles:
            run_pass(resolved_recovery, angle, recovery_families)

    candidates = _merge_labeled_face_candidates(groups)
    accepted: list[tuple[int, int, int, int]] = []
    for entry in candidates:
        box = entry["box"]
        x, y, w, h = box
        cx = max(0, x)
        cy = max(0, y)
        clipped = (cx, cy, min(w, w_img - cx), min(h, h_img - cy))
        if clipped[2] < 24 or clipped[3] < 24:
            continue
        aspect = clipped[2] / max(clipped[3], 1)
        if not (0.62 <= aspect <= 1.60):
            continue
        relative_size = min(clipped[2], clipped[3]) / max(min(h_img, w_img), 1)
        # Very strong, large multi-scale/multi-family face evidence is accepted by
        # the existing strict gate even with zero eye support. Avoid running four
        # eye cascades when their result cannot change that decision.
        no_eye_evidence = _EyeEvidence(0, False, 0)
        if _accept_face_candidate_strict(entry, no_eye_evidence, relative_size):
            accepted.append(clipped)
            continue
        eye_evidence = _plausible_eye_evidence(original_gray, clipped, cascade_dir)
        if _accept_face_candidate_strict(entry, eye_evidence, relative_size):
            accepted.append(clipped)

    # Deduplicate again after inverse rotation and coordinate rounding.
    final_boxes: list[tuple[int, int, int, int]] = []
    for box in sorted(accepted, key=lambda b: b[2] * b[3], reverse=True):
        if any(_iou(box, kept) >= 0.42 for kept in final_boxes):
            continue
        final_boxes.append(box)
    final_boxes = final_boxes[:12]
    auto_faces = measure_face_regions(rgb, final_boxes, source="auto")

    manual_faces: list[FaceRegion] = []
    for raw_box in manual_boxes or []:
        box = _normalized_box_to_pixels(raw_box, w_img, h_img, min_size=24)
        if box is None:
            continue
        measured = measure_face_regions(rgb, [box], source="manual")
        if not measured:
            continue
        manual = measured[0]
        manual.measurement_confidence = max(manual.measurement_confidence, 0.94)
        # Human markup wins over an overlapping automatic crop.
        auto_faces = [
            face for face in auto_faces
            if _iou((face.x, face.y, face.w, face.h), box) < 0.42
        ]
        manual_faces.append(manual)

    faces = (auto_faces + manual_faces)[:12]
    detector_confidence = 0.92 if manual_faces else (0.77 if faces else 0.50)
    return FaceAnalysis(True, faces, detector_confidence)
