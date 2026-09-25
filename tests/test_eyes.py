import numpy as np

from photodoctor.core.eyes import measure_eye_regions, select_eye_candidates
from photodoctor.core.faces import FaceRegion


def face() -> FaceRegion:
    return FaceRegion(50, 40, 160, 160, 0.4, 100.0, 5000.0, 55.0, 0.75)


def test_selector_keeps_plausible_horizontal_eye_pair_and_rejects_lower_face_noise():
    f = face()
    candidates = [
        (44, 50, 24, 24),
        (94, 49, 23, 23),
        (58, 92, 22, 22),
        (54, 40, 44, 44),
    ]
    selected = select_eye_candidates(f, candidates)
    assert len(selected) == 2
    centers = sorted((x + w / 2.0) / f.w for x, y, w, h in selected)
    assert centers[0] < 0.5 < centers[1]


def test_eye_measurement_reports_lower_sharpness_after_blur():
    y, x = np.indices((220, 260))
    checker = (((x // 3 + y // 3) % 2) * 255).astype(np.uint8)
    sharp = np.repeat(checker[..., None], 3, axis=2)
    # Small manual blur without relying on detector behaviour.
    import cv2
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.0)
    boxes = [(80, 80, 32, 28)]
    a = measure_eye_regions(sharp, 0, boxes, False)[0]
    b = measure_eye_regions(blurred, 0, boxes, False)[0]
    assert a.sharpness_score > b.sharpness_score + 15


def test_selector_returns_at_most_one_when_pair_geometry_is_bad():
    f = face()
    candidates = [(45, 50, 22, 22), (52, 51, 21, 21), (60, 52, 20, 20)]
    selected = select_eye_candidates(f, candidates)
    assert len(selected) <= 1


def test_dual_detector_name_is_versioned_when_available():
    import cv2
    from photodoctor.core.eyes import analyze_eyes
    rgb = np.zeros((120, 120, 3), dtype=np.uint8)
    result = analyze_eyes(rgb, [])
    # OpenCV wheels used by Photo Doctor ship at least one eye cascade.
    assert result.detector_available
    assert result.detector_name == "opencv_haar_eye_quad_v3"


def test_manual_eye_box_is_assigned_inside_manual_face():
    import numpy as np
    from photodoctor.core.eyes import analyze_eyes
    rgb = np.full((300, 400, 3), 128, np.uint8)
    f = FaceRegion(100, 60, 160, 180, 0.4, 100.0, 5000.0, 55.0, 0.95, "manual")
    result = analyze_eyes(rgb, [f], manual_boxes=[{"x": 0.34, "y": 0.30, "w": 0.07, "h": 0.07}])
    manual = [eye for eye in result.eyes if eye.source == "manual"]
    assert len(manual) == 1
    assert manual[0].face_index == 0
    assert manual[0].measurement_confidence >= 0.90


def test_precise_eye_mode_advertises_deep_detector_path():
    from photodoctor.core.eyes import analyze_eyes
    rgb = np.zeros((120, 120, 3), dtype=np.uint8)
    result = analyze_eyes(rgb, [], precision="precise")
    assert result.detector_available
    assert result.detector_name == "opencv_haar_eye_quad_v4_deep"
