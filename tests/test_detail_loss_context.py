from __future__ import annotations

import cv2
import numpy as np

from photodoctor.core.blur_type import BlurTypeResult, analyze_blur_type
from photodoctor.core.detail_loss_context import refine_detail_loss_type
from photodoctor.core.exif_context import analyze_exif_context
from photodoctor.core.local_sharpness import LocalSharpnessResult
from photodoctor.core.main_subject import MainSubjectResult


def _texture(size: int = 720) -> np.ndarray:
    rng = np.random.default_rng(4321)
    base = rng.normal(127, 44, (size, size)).clip(0, 255).astype(np.uint8)
    for y in range(30, size, 70):
        cv2.line(base, (15, y), (size - 20, y + 19), 230, 3)
    for x in range(35, size, 85):
        cv2.line(base, (x, 10), (x + 25, size - 10), 24, 3)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)


def _motion(rgb: np.ndarray, length: int = 29) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    return cv2.filter2D(rgb, -1, k)


def _unknown_subject() -> MainSubjectResult:
    return MainSubjectResult("unknown", "unknown_scene", 0.0, None, None, "none", (), 0.0, ("none",))


def _center_subject() -> MainSubjectResult:
    box = {"x": 0.25, "y": 0.25, "w": 0.50, "h": 0.50}
    return MainSubjectResult("person", "people_portrait", 0.86, box, box, "test", (0,), 25.0, ("test",))


def _local(subject_score: float = 35.0, background_score: float = 72.0) -> LocalSharpnessResult:
    cells = []
    # 4x4 grid. Central four cells belong to the 0.25..0.75 subject box.
    for iy in range(4):
        for ix in range(4):
            x = ix * 0.25
            y = iy * 0.25
            inside = ix in {1, 2} and iy in {1, 2}
            cells.append({
                "x": x, "y": y, "w": 0.25, "h": 0.25,
                "score": subject_score if inside else background_score,
                "confidence": 0.82, "status": "soft" if inside else "sharp",
            })
    return LocalSharpnessResult(60.0, 0.85, 25.0, 16, 16, cells)


def test_global_directional_blur_plus_slow_shutter_is_camera_shake_like():
    rgb = _motion(_texture())
    raw = analyze_blur_type(rgb)
    assert raw.classification == "motion_like"
    exif = analyze_exif_context({"exposure_time": "1/15", "focal_length": "50", "iso": 200})
    result = refine_detail_loss_type(
        rgb, raw, _local(60.0, 61.0), _unknown_subject(), exif,
        archival_likelihood=0.0, surface_candidate_count=0,
    )
    assert result.classification == "camera_shake_like"
    assert result.confidence >= 0.60
    assert result.shutter_risk == "high"


def test_directionally_blurred_subject_on_sharp_background_is_subject_motion_like():
    base = _texture()
    rgb = base.copy()
    y0 = x0 = 180
    y1 = x1 = 540
    rgb[y0:y1, x0:x1] = _motion(base[y0:y1, x0:x1], 25)
    # The whole-frame low-level result may be mixed because most background is sharp;
    # feed a conservative raw motion hypothesis and require the subject crop to confirm it.
    raw = BlurTypeResult("motion_like", 0.61, 2.4, 0.0, 10.0, 8, "test_motion")
    result = refine_detail_loss_type(
        rgb, raw, _local(34.0, 76.0), _center_subject(), analyze_exif_context({}),
        archival_likelihood=0.0, surface_candidate_count=0,
    )
    assert result.subject_blur_classification == "motion_like"
    assert result.classification == "subject_motion_like"
    assert result.subject_background_delta is not None and result.subject_background_delta >= 30.0


def test_local_soft_subject_without_direction_stays_local_not_global_defocus():
    base = _texture()
    rgb = base.copy()
    rgb[180:540, 180:540] = cv2.GaussianBlur(base[180:540, 180:540], (0, 0), 4.5)
    raw = BlurTypeResult("mixed_or_degraded", 0.48, 1.4, None, 9.0, 8, "mixed")
    result = refine_detail_loss_type(
        rgb, raw, _local(31.0, 75.0), _center_subject(), analyze_exif_context({}),
        archival_likelihood=0.0, surface_candidate_count=0,
    )
    assert result.classification == "local_subject_softness"
    assert result.classification != "defocus_like"


def test_global_gaussian_softness_is_defocus_like():
    rgb = cv2.GaussianBlur(_texture(), (0, 0), 4.5)
    raw = analyze_blur_type(rgb)
    assert raw.classification in {"defocus_like", "mixed_or_degraded"}
    if raw.classification != "defocus_like":
        raw = BlurTypeResult("defocus_like", 0.68, raw.anisotropy, None, raw.fine_to_coarse_ratio, raw.informative_patches, "forced_defocus_fixture")
    result = refine_detail_loss_type(
        rgb, raw, _local(42.0, 43.0), _unknown_subject(), analyze_exif_context({}),
        archival_likelihood=0.0, surface_candidate_count=0,
    )
    assert result.classification == "defocus_like"


def test_archival_mixed_loss_prefers_degradation_context():
    rgb = _texture()
    raw = BlurTypeResult("mixed_or_degraded", 0.46, 1.5, None, 12.0, 9, "mixed")
    result = refine_detail_loss_type(
        rgb, raw, _local(48.0, 50.0), _unknown_subject(), analyze_exif_context({}),
        archival_likelihood=0.78, surface_candidate_count=18,
    )
    assert result.classification == "degradation_like"
    assert "archival_context" in result.reasons


def test_monochrome_alone_is_not_enough_for_archival_degradation():
    rgb = _texture()
    raw = BlurTypeResult("mixed_or_degraded", 0.48, 1.5, None, 12.0, 9, "mixed")
    result = refine_detail_loss_type(
        rgb, raw, _local(48.0, 50.0), _unknown_subject(), analyze_exif_context({}),
        archival_likelihood=0.24, surface_candidate_count=18,
    )
    assert result.classification != "degradation_like"
    assert result.classification == "mixed"
