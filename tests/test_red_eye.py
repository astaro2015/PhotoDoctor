import numpy as np

from photodoctor.core.decisions import build_decision_plan
from photodoctor.core.eyes import EyeRegion
from photodoctor.core.models import MetricResult
from photodoctor.core.red_eye import analyze_red_eye, correct_red_eye
from photodoctor.core.validator import apply_validated_preview


def m(name, raw, score, conf=0.8):
    return MetricResult(name, raw, score, conf, "test")


def synthetic_eye_rgb() -> np.ndarray:
    rgb = np.full((48, 72, 3), 140, dtype=np.uint8)
    yy, xx = np.mgrid[0:48, 0:72]
    iris = ((xx - 36) ** 2) / (15 ** 2) + ((yy - 24) ** 2) / (11 ** 2) <= 1.0
    pupil = ((xx - 36) ** 2) / (7 ** 2) + ((yy - 24) ** 2) / (7 ** 2) <= 1.0
    rgb[iris] = (78, 88, 95)
    rgb[pupil] = (210, 34, 34)
    rgb[20:23, 41:44] = (245, 245, 245)  # catchlight
    return rgb


def test_red_eye_detector_finds_suspicious_eye():
    rgb = synthetic_eye_rgb()
    eyes = [EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)]
    analysis = analyze_red_eye(rgb, eyes)
    assert analysis.suspicious_eye_count == 1
    assert analysis.candidates[0].severity >= 14.0
    assert analysis.candidates[0].red_pixel_ratio >= 0.012


def test_red_eye_correction_reduces_red_channel_in_affected_region():
    rgb = synthetic_eye_rgb()
    eyes = [EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)]
    analysis = analyze_red_eye(rgb, eyes)
    corrected, stats = correct_red_eye(rgb, analysis.candidates)
    assert stats.corrected_count == 1
    before_red = float(rgb[..., 0].mean())
    after_red = float(corrected[..., 0].mean())
    assert after_red < before_red - 3.0


def test_decision_plan_adds_red_eye_fix():
    metrics = {
        "brightness": m("brightness", 0.5, 70, 0.85),
        "highlight_context": m("highlight_context", {"unexplained_clip_pct": 0.0}, None, 0.7),
        "contrast": m("contrast", 0.6, 85, 0.8),
        "local_contrast": m("local_contrast", {"classification": "normal"}, 82, 0.8),
        "laplacian": m("laplacian", 100, 75, 0.72),
        "tenengrad": m("tenengrad", 5000, 78, 0.78),
        "detail_loss_type": m("detail_loss_type", {"classification": "unknown"}, None, 0.5),
        "noise": m("noise", {}, 90, 0.7),
        "jpeg_artifacts": m("jpeg_artifacts", {}, 95, 0.7),
        "edge_artifacts": m("edge_artifacts", {}, 95, 0.7),
        "posterization": m("posterization", {}, 95, 0.7),
        "surface_defects": m("surface_defects", {"candidate_count": 0}, 100, 0.45),
        "image_tone": m("image_tone", {"classification": "color"}, None, 0.8),
        "semantic_context": m("semantic_context", {"classification": "general_photo", "face_count": 1}, None, 0.5),
        "faces": m("faces", {"face_count": 1}, 72, 0.8),
        "red_eye": m("red_eye", {"suspicious_eye_count": 1, "candidates": [{"suspicious": True}]}, 60, 0.82),
    }
    plan = build_decision_plan(metrics)
    item = next(i for i in plan if i.key == "red_eye")
    assert item.decision == "fix"
    assert item.repairability >= 90


def test_preview_applies_accepted_red_eye_fix():
    rgb = synthetic_eye_rgb()
    before = rgb.copy()
    preview = apply_validated_preview(rgb, [
        {
            "action_key": "red_eye",
            "accepted": True,
            "candidate": "localized_pupil_neutralization",
            "source_candidates": [{"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "suspicious": True}],
        }
    ])
    assert np.array_equal(rgb, before)
    assert float(preview[..., 0].mean()) < float(rgb[..., 0].mean())


def test_red_eye_correction_preserves_catchlight_and_nonred_channels():
    rgb = synthetic_eye_rgb()
    eyes = [EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)]
    analysis = analyze_red_eye(rgb, eyes)
    corrected, stats = correct_red_eye(rgb, analysis.candidates)
    assert stats.corrected_count == 1

    # White catchlight must survive exactly; losing it makes the pupil look painted.
    assert np.array_equal(corrected[20:23, 41:44], rgb[20:23, 41:44])

    pupil = ((np.mgrid[0:48, 0:72][1] - 36) ** 2 + (np.mgrid[0:48, 0:72][0] - 24) ** 2) <= 7 ** 2
    pupil[20:23, 41:44] = False
    # V2 removes the red flash but does not recolor G/B, preserving iris texture/chroma.
    assert np.array_equal(corrected[..., 1][pupil], rgb[..., 1][pupil])
    assert np.array_equal(corrected[..., 2][pupil], rgb[..., 2][pupil])
    assert float(corrected[..., 0][pupil].mean()) < float(rgb[..., 0][pupil].mean()) - 50.0


def test_red_eye_correction_does_not_touch_skin_outside_pupil():
    rgb = synthetic_eye_rgb()
    before = rgb.copy()
    eyes = [EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)]
    analysis = analyze_red_eye(rgb, eyes)
    corrected, _ = correct_red_eye(rgb, analysis.candidates)
    yy, xx = np.mgrid[0:48, 0:72]
    outside = ((xx - 36) ** 2 + (yy - 24) ** 2) >= 11 ** 2
    assert np.array_equal(corrected[outside], before[outside])


def test_normalized_candidate_box_is_not_truncated_before_scaling():
    canvas = np.full((100, 160, 3), 120, dtype=np.uint8)
    eye = synthetic_eye_rgb()
    # Put the synthetic eye away from origin. Old code int(0.25)->0 and missed it.
    x, y = 40, 30
    canvas[y:y+48, x:x+72] = eye
    candidate = {
        "x": x / canvas.shape[1],
        "y": y / canvas.shape[0],
        "w": 72 / canvas.shape[1],
        "h": 48 / canvas.shape[0],
        "suspicious": True,
    }
    corrected, stats = correct_red_eye(canvas, [candidate])
    assert stats.corrected_count == 1
    # Origin/background must remain untouched, while the inserted eye changes.
    assert np.array_equal(corrected[:20, :20], canvas[:20, :20])
    assert float(corrected[y:y+48, x:x+72, 0].mean()) < float(canvas[y:y+48, x:x+72, 0].mean()) - 2.0


def test_normal_eye_is_left_unchanged():
    rgb = synthetic_eye_rgb()
    # Convert the pupil to a natural dark neutral pupil with a white catchlight.
    yy, xx = np.mgrid[0:48, 0:72]
    pupil = ((xx - 36) ** 2 + (yy - 24) ** 2) <= 7 ** 2
    rgb[pupil] = (35, 38, 42)
    rgb[20:23, 41:44] = (245, 245, 245)
    before = rgb.copy()
    eyes = [EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)]
    analysis = analyze_red_eye(rgb, eyes)
    assert analysis.suspicious_eye_count == 0
    corrected, stats = correct_red_eye(rgb, analysis.candidates)
    assert stats.corrected_count == 0
    assert np.array_equal(corrected, before)


def test_red_eye_strength_is_monotonic_and_partial():
    rgb = synthetic_eye_rgb()
    eyes = [EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)]
    analysis = analyze_red_eye(rgb, eyes)
    half, _ = correct_red_eye(rgb, analysis.candidates, strength=0.5)
    full, _ = correct_red_eye(rgb, analysis.candidates, strength=1.0)
    yy, xx = np.mgrid[0:48, 0:72]
    pupil = ((xx - 36) ** 2 + (yy - 24) ** 2) <= 7 ** 2
    pupil[20:23, 41:44] = False
    before_r = float(rgb[..., 0][pupil].mean())
    half_r = float(half[..., 0][pupil].mean())
    full_r = float(full[..., 0][pupil].mean())
    assert before_r > half_r > full_r
    half_drop = before_r - half_r
    full_drop = before_r - full_r
    assert 0.35 <= half_drop / full_drop <= 0.65

from photodoctor.core.faces import FaceRegion
from photodoctor.core.red_eye import supplement_red_eye_eye_regions


def _small_group_face() -> FaceRegion:
    return FaceRegion(
        x=20, y=12, w=42, h=50,
        brightness_linear=0.28,
        laplacian=20.0,
        tenengrad=100.0,
        sharpness_score=55.0,
        measurement_confidence=0.62,
        source="auto",
    )


def _paint_small_red_reflexes(canvas: np.ndarray, fallback_eyes: list[EyeRegion]) -> None:
    for eye in fallback_eyes:
        cx = eye.x + eye.w // 2
        cy = eye.y + eye.h // 2
        yy, xx = np.ogrid[:canvas.shape[0], :canvas.shape[1]]
        iris = (xx - cx) ** 2 + (yy - cy) ** 2 <= 3 ** 2
        pupil = (xx - cx) ** 2 + (yy - cy) ** 2 <= 2 ** 2
        canvas[iris] = (62, 66, 72)
        canvas[pupil] = (220, 26, 24)


def test_group_photo_geometry_fallback_finds_red_eyes_when_haar_found_none():
    rgb = np.full((80, 90, 3), (118, 96, 82), dtype=np.uint8)
    face = _small_group_face()
    fallback = supplement_red_eye_eye_regions(rgb, [face], [])
    assert len(fallback) == 2
    assert all(eye.source == "red_eye_geometry" for eye in fallback)
    _paint_small_red_reflexes(rgb, fallback)

    analysis = analyze_red_eye(rgb, fallback)
    assert analysis.suspicious_eye_count == 2
    assert all(c.eye_source == "red_eye_geometry" for c in analysis.candidates)
    assert any(c.relaxed_detection or c.red_pixel_ratio > 0 for c in analysis.candidates)


def test_group_photo_fallback_only_adds_missing_eye_slot():
    rgb = np.full((80, 90, 3), 120, dtype=np.uint8)
    face = _small_group_face()
    existing = [EyeRegion(
        0, face.x + 7, face.y + 13, 10, 9,
        30.0, 100.0, 50.0, 0.55, "auto"
    )]
    supplemented = supplement_red_eye_eye_regions(rgb, [face], existing)
    assert len(supplemented) == 2
    assert sum(eye.source == "red_eye_geometry" for eye in supplemented) == 1


def test_group_photo_geometry_fallback_does_not_flag_neutral_dark_pupils():
    rgb = np.full((80, 90, 3), (118, 96, 82), dtype=np.uint8)
    face = _small_group_face()
    fallback = supplement_red_eye_eye_regions(rgb, [face], [])
    for eye in fallback:
        cx = eye.x + eye.w // 2
        cy = eye.y + eye.h // 2
        yy, xx = np.ogrid[:rgb.shape[0], :rgb.shape[1]]
        iris = (xx - cx) ** 2 + (yy - cy) ** 2 <= 3 ** 2
        pupil = (xx - cx) ** 2 + (yy - cy) ** 2 <= 2 ** 2
        rgb[iris] = (58, 63, 68)
        rgb[pupil] = (30, 33, 36)
    analysis = analyze_red_eye(rgb, fallback)
    assert analysis.suspicious_eye_count == 0


def _subtle_group_eye(source: str) -> tuple[np.ndarray, EyeRegion]:
    """Confirmed eye with a tiny JPEG-like red reflex diluted by a generous ROI."""
    rgb = np.full((30, 40, 3), (82, 84, 88), dtype=np.uint8)
    # Neutral/dark iris around the centre.
    yy, xx = np.ogrid[:30, :40]
    iris = (xx - 20) ** 2 + (yy - 15) ** 2 <= 5 ** 2
    rgb[iris] = (58, 61, 65)
    # Only 2x2 mildly red pixels: strict v3 threshold rejects these, relaxed v4
    # should recover them inside an already-confirmed eye ROI.
    rgb[14:16, 19:21] = (118, 103, 100)
    eye = EyeRegion(0, 8, 6, 24, 18, 10.0, 40.0, 45.0, 0.96 if source == "manual" else 0.62, source)
    return rgb, eye


def test_manual_confirmed_eye_gets_relaxed_red_reflex_second_pass():
    rgb, eye = _subtle_group_eye("manual")
    analysis = analyze_red_eye(rgb, [eye])
    assert analysis.suspicious_eye_count == 1
    cand = analysis.candidates[0]
    assert cand.relaxed_detection is True
    assert cand.red_pixel_count >= 2
    assert cand.red_pixel_ratio < 0.012  # old whole-eye ratio rule rejected this
    assert cand.mean_red_level >= 70.0


def test_auto_confirmed_eye_weak_relaxed_reflex_is_not_promoted_to_defect():
    rgb, eye = _subtle_group_eye("auto")
    assert min(eye.w, eye.h) > 12
    analysis = analyze_red_eye(rgb, [eye])
    # Automatic eye boxes frequently contain a few warm/JPEG pixels. The relaxed
    # pass may see them, but v5 must not promote this weak 2x2 patch to red-eye.
    assert analysis.candidates[0].relaxed_detection is True
    assert analysis.suspicious_eye_count == 0


def test_relaxed_confirmed_eye_does_not_flag_brown_iris_patch_as_red_eye():
    rgb = np.full((30, 40, 3), (82, 84, 88), dtype=np.uint8)
    # Warm brown has a large G/B separation. Relaxed red-eye path should reject it.
    rgb[13:17, 18:22] = (96, 76, 54)
    eye = EyeRegion(0, 8, 6, 24, 18, 10.0, 40.0, 45.0, 0.96, "manual")
    analysis = analyze_red_eye(rgb, [eye])
    assert analysis.suspicious_eye_count == 0


def test_confirmed_eye_red_reflex_reaches_decision_plan():
    rgb, eye = _subtle_group_eye("manual")
    analysis = analyze_red_eye(rgb, [eye])
    assert analysis.suspicious_eye_count == 1
    raw = {
        "eye_count": 1,
        "checked_eye_count": 1,
        "suspicious_eye_count": analysis.suspicious_eye_count,
        "candidates": [c.to_raw(rgb.shape[1], rgb.shape[0]) for c in analysis.candidates],
        "method": analysis.detector_name,
    }
    metrics = {
        "brightness": m("brightness", 0.3, 85, 0.85),
        "highlight_context": m("highlight_context", {"unexplained_clip_pct": 0.0}, None, 0.7),
        "contrast": m("contrast", 0.6, 85, 0.8),
        "local_contrast": m("local_contrast", {"classification": "normal"}, 82, 0.8),
        "laplacian": m("laplacian", 100, 75, 0.72),
        "tenengrad": m("tenengrad", 5000, 78, 0.78),
        "detail_loss_type": m("detail_loss_type", {"classification": "unknown"}, None, 0.5),
        "noise": m("noise", {}, 90, 0.7),
        "jpeg_artifacts": m("jpeg_artifacts", {}, 95, 0.7),
        "edge_artifacts": m("edge_artifacts", {}, 95, 0.7),
        "posterization": m("posterization", {}, 95, 0.7),
        "surface_defects": m("surface_defects", {"candidate_count": 0}, 100, 0.45),
        "image_tone": m("image_tone", {"classification": "color"}, None, 0.8),
        "semantic_context": m("semantic_context", {"classification": "group_portrait", "face_count": 8}, None, 0.8),
        "faces": m("faces", {"face_count": 8}, 72, 0.8),
        "red_eye": m("red_eye", raw, 70, analysis.confidence),
    }
    plan = build_decision_plan(metrics)
    item = next(i for i in plan if i.key == "red_eye")
    assert item.title == "Красные глаза"
    assert item.decision in {"fix", "review"}


def test_relaxed_red_eye_decision_requires_manual_review():
    rgb, eye = _subtle_group_eye("manual")
    analysis = analyze_red_eye(rgb, [eye])
    assert analysis.suspicious_eye_count == 1
    raw = {
        "suspicious_eye_count": 1,
        "candidates": [c.to_raw(rgb.shape[1], rgb.shape[0]) for c in analysis.candidates],
    }
    metrics = {
        "brightness": m("brightness", 0.3, 85, 0.85),
        "highlight_context": m("highlight_context", {"unexplained_clip_pct": 0.0}, None, 0.7),
        "contrast": m("contrast", 0.6, 85, 0.8),
        "local_contrast": m("local_contrast", {"classification": "normal"}, 82, 0.8),
        "laplacian": m("laplacian", 100, 75, 0.72),
        "tenengrad": m("tenengrad", 5000, 78, 0.78),
        "detail_loss_type": m("detail_loss_type", {"classification": "unknown"}, None, 0.5),
        "noise": m("noise", {}, 90, 0.7),
        "jpeg_artifacts": m("jpeg_artifacts", {}, 95, 0.7),
        "edge_artifacts": m("edge_artifacts", {}, 95, 0.7),
        "posterization": m("posterization", {}, 95, 0.7),
        "surface_defects": m("surface_defects", {"candidate_count": 0}, 100, 0.45),
        "image_tone": m("image_tone", {"classification": "color"}, None, 0.8),
        "semantic_context": m("semantic_context", {"classification": "portrait", "face_count": 1}, None, 0.8),
        "faces": m("faces", {"face_count": 1}, 72, 0.8),
        "red_eye": m("red_eye", raw, 70, 0.95),
    }
    plan = build_decision_plan(metrics)
    item = next(i for i in plan if i.key == "red_eye")
    assert item.decision == "review"


def test_geometry_fallback_single_warm_red_patch_is_not_red_eye():
    """A guessed eye position must not turn one warm skin/hair patch into red-eye."""
    rgb = np.full((80, 90, 3), (118, 96, 82), dtype=np.uint8)
    face = _small_group_face()
    fallback = supplement_red_eye_eye_regions(rgb, [face], [])
    assert len(fallback) == 2

    # Paint only one expected eye slot with a compact but off-centre warm red patch.
    eye = fallback[1]
    x0 = eye.x + max(1, int(round(eye.w * 0.12)))
    y0 = eye.y + max(1, int(round(eye.h * 0.10)))
    rgb[y0:y0 + 3, x0:x0 + 5] = (198, 132, 114)

    analysis = analyze_red_eye(rgb, fallback)
    assert analysis.suspicious_eye_count == 0
    red = [c for c in analysis.candidates if c.red_pixel_count > 0]
    assert red
    assert all(not c.geometry_pair_confirmed for c in red)
    assert all(not c.to_raw(rgb.shape[1], rgb.shape[0])["suspicious"] for c in red)


def test_geometry_fallback_real_binocular_red_reflex_is_confirmed_as_pair():
    rgb = np.full((80, 90, 3), (118, 96, 82), dtype=np.uint8)
    face = _small_group_face()
    fallback = supplement_red_eye_eye_regions(rgb, [face], [])
    _paint_small_red_reflexes(rgb, fallback)

    analysis = analyze_red_eye(rgb, fallback)
    suspicious = [c for c in analysis.candidates if c.to_raw(rgb.shape[1], rgb.shape[0])["suspicious"]]
    assert len(suspicious) == 2
    assert all(c.geometry_pair_confirmed for c in suspicious)
    assert all(c.local_red_contrast >= 12.0 for c in suspicious)


def test_red_eye_correction_never_turns_warm_under_eye_skin_into_bruise():
    rgb = np.full((64, 80, 3), (185, 132, 118), dtype=np.uint8)
    yy, xx = np.mgrid[0:64, 0:80]
    iris = ((xx - 40) ** 2) / (15 ** 2) + ((yy - 30) ** 2) / (10 ** 2) <= 1.0
    pupil = ((xx - 40) ** 2 + (yy - 30) ** 2) <= 6 ** 2
    rgb[iris] = (78, 82, 88)
    rgb[pupil] = (190, 42, 38)
    rgb[27:30, 42:45] = (245, 245, 245)

    # Deliberately warm/red skin under the eye.  Older masking could classify
    # this long crescent together with the pupil and create a blue-gray bruise.
    under_eye = (((xx - 40) ** 2) / (30 ** 2) + ((yy - 47) ** 2) / (9 ** 2) <= 1.0) & (yy >= 40)
    rgb[under_eye] = (205, 126, 110)
    before = rgb.copy()

    eye = EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)
    analysis = analyze_red_eye(rgb, [eye])
    assert analysis.suspicious_eye_count == 1
    corrected, stats = correct_red_eye(rgb, analysis.candidates)
    assert stats.corrected_count == 1
    assert np.array_equal(corrected[under_eye], before[under_eye])
    assert np.any(corrected[pupil] != before[pupil])


def test_red_eye_correction_changes_only_verified_mask_pixels():
    rgb = synthetic_eye_rgb()
    eye = EyeRegion(0, 0, 0, rgb.shape[1], rgb.shape[0], 100.0, 100.0, 60.0, 0.8)
    analysis = analyze_red_eye(rgb, [eye])
    corrected, _stats = correct_red_eye(rgb, analysis.candidates)
    from photodoctor.core.red_eye import build_red_eye_mask
    allowed = build_red_eye_mask(rgb)
    changed = np.any(corrected != rgb, axis=2)
    assert np.all(~changed | allowed)
