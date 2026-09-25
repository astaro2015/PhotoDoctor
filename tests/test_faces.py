from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from photodoctor.core.faces import (
    analyze_faces, measure_face_regions, _merge_face_candidates, _accept_face_candidate,
    _accept_face_candidate_strict, _EyeEvidence,
)
from photodoctor.core.service import analyze_file
from photodoctor.gui.presentation import build_display_metrics


def _checker_face(size: int = 160) -> np.ndarray:
    y, x = np.indices((size, size))
    c = (((x // 8 + y // 8) % 2) * 170 + 45).astype(np.uint8)
    return np.dstack([c, c, c])


def test_face_region_blur_reduces_local_sharpness():
    sharp = _checker_face()
    blurred = np.array(Image.fromarray(sharp, "RGB").filter(ImageFilter.GaussianBlur(radius=3.0)))
    box = [(0, 0, sharp.shape[1], sharp.shape[0])]
    a = measure_face_regions(sharp, box)[0]
    b = measure_face_regions(blurred, box)[0]
    assert a.sharpness_score > b.sharpness_score
    assert a.laplacian > b.laplacian


def test_face_region_brightness_is_local():
    rgb = np.full((200, 300, 3), 220, np.uint8)
    rgb[40:160, 90:210] = 70
    face = measure_face_regions(rgb, [(90, 40, 120, 120)])[0]
    assert face.brightness_linear < 0.10
    assert 0.0 <= face.measurement_confidence <= 1.0


def test_face_boxes_export_normalized_coordinates():
    rgb = np.full((240, 320, 3), 128, np.uint8)
    face = measure_face_regions(rgb, [(80, 60, 100, 120)])[0]
    raw = face.to_raw(320, 240)
    assert raw["x"] == 0.25
    assert raw["y"] == 0.25
    assert 0 < raw["w"] <= 1
    assert 0 < raw["h"] <= 1


def test_blank_image_without_faces_is_not_an_error():
    rgb = np.full((320, 400, 3), 128, np.uint8)
    result = analyze_faces(rgb)
    assert result.detector_available
    assert result.faces == []


def test_analyzer_exposes_faces_metric(tmp_path):
    rgb = np.full((320, 400, 3), 128, np.uint8)
    path = tmp_path / "blank.png"
    Image.fromarray(rgb, "RGB").save(path)
    result = analyze_file(path)
    metric = result.metrics["faces"]
    assert isinstance(metric.raw_value, dict)
    assert metric.raw_value["detector"] == "opencv_haar_frontal_rotation_multiscale_v5_eye_consensus"
    assert "face_count" in metric.raw_value


def test_no_faces_display_row_is_informational(tmp_path):
    rgb = np.full((320, 400, 3), 128, np.uint8)
    path = tmp_path / "blank.png"
    Image.fromarray(rgb, "RGB").save(path)
    metrics = analyze_file(path).metrics
    row = {r.key: r for r in build_display_metrics(metrics)}["faces"]
    assert row.status == "Инфо"
    assert "не найдены" in row.diagnosis.lower()


def test_dual_face_candidates_merge_overlapping_default_and_alt_hits():
    merged = _merge_face_candidates(
        [((20, 20, 100, 100), 7.5)],
        [((24, 22, 96, 96), 101.0), ((180, 30, 88, 88), 99.0)],
    )
    assert len(merged) == 2
    consensus = next(row for row in merged if len(row["sources"]) == 2)
    assert consensus["sources"] == {"default", "alt"}



def test_alt_only_face_recovery_requires_two_plausible_eyes():
    entry = {"box": (10, 10, 100, 100), "sources": {"alt"}, "weights": {"alt": 100.0}}
    assert not _accept_face_candidate(entry, 0)
    assert not _accept_face_candidate(entry, 1)
    assert _accept_face_candidate(entry, 2)


def test_primary_or_dual_consensus_can_survive_closed_eyes():
    strong_primary = {"box": (10, 10, 100, 100), "sources": {"default"}, "weights": {"default": 5.0}}
    consensus = {"box": (10, 10, 100, 100), "sources": {"default", "alt"}, "weights": {"default": 1.0, "alt": 100.0}}
    assert _accept_face_candidate(strong_primary, 0)
    assert _accept_face_candidate(consensus, 0)


def test_multiscale_alt_recovery_can_use_one_eye_but_weak_default_repeats_cannot():
    alt_repeat = {
        "box": (10, 10, 100, 100),
        "sources": {"alt@1400@+10", "alt@900@-10"},
        "weights": {"alt@1400@+10": 100.0, "alt@900@-10": 99.0},
    }
    weak_default_repeat = {
        "box": (10, 10, 100, 100),
        "sources": {"default@1400", "default@900"},
        "weights": {"default@1400": 0.4, "default@900": 0.8},
    }
    assert _accept_face_candidate(alt_repeat, 1)
    assert not _accept_face_candidate(weak_default_repeat, 1)


def test_manual_face_box_is_kept_on_blank_image():
    rgb = np.full((300, 400, 3), 128, np.uint8)
    result = analyze_faces(rgb, precision="precise", manual_boxes=[{"x": 0.25, "y": 0.20, "w": 0.30, "h": 0.40}])
    manual = [face for face in result.faces if face.source == "manual"]
    assert len(manual) == 1
    assert manual[0].measurement_confidence >= 0.90


def test_service_manual_face_and_eye_flow_into_metrics(tmp_path):
    from photodoctor.core.service import analyze_file
    rgb = np.full((300, 400, 3), 128, np.uint8)
    path = tmp_path / "manual.png"
    Image.fromarray(rgb, "RGB").save(path)
    result = analyze_file(
        path,
        precision="fast",
        manual_face_boxes=[{"x": 0.25, "y": 0.20, "w": 0.40, "h": 0.50}],
        manual_eye_boxes=[{"x": 0.34, "y": 0.32, "w": 0.08, "h": 0.08}],
    )
    faces = result.metrics["faces"].raw_value["faces"]
    eyes = result.metrics["eyes"].raw_value["eyes"]
    assert len(faces) == 1 and faces[0]["source"] == "manual"
    assert len(eyes) == 1 and eyes[0]["source"] == "manual"
    assert result.metrics["red_eye"].raw_value["eye_count"] == 1


def test_strict_face_gate_rejects_small_pattern_even_with_two_generic_eye_hits():
    entry = {
        "box": (10, 10, 78, 78),
        "sources": {"default@1500@+0", "alt@1500@+0", "alt2@1100@+0"},
        "weights": {"default@1500@+0": 5.2, "alt@1500@+0": 108.0, "alt2@1100@+0": 55.0},
    }
    evidence = _EyeEvidence(count=2, robust_pair=False, supported_eye_count=0)
    assert not _accept_face_candidate_strict(entry, evidence, relative_size=0.07)


def test_strict_face_gate_accepts_small_face_with_independent_eye_consensus():
    entry = {
        "box": (10, 10, 72, 72),
        "sources": {"default@1500@+0"},
        "weights": {"default@1500@+0": 1.0},
    }
    evidence = _EyeEvidence(count=2, robust_pair=True, supported_eye_count=2)
    assert _accept_face_candidate_strict(entry, evidence, relative_size=0.06)


def test_strict_face_gate_rejects_large_weak_single_cascade_false_face():
    entry = {
        "box": (10, 10, 260, 260),
        "sources": {"default@1350@+0"},
        "weights": {"default@1350@+0": -0.7},
    }
    evidence = _EyeEvidence(count=2, robust_pair=False, supported_eye_count=0)
    assert not _accept_face_candidate_strict(entry, evidence, relative_size=0.23)


def test_strict_face_gate_keeps_large_closed_eye_portrait_only_with_strong_consensus():
    entry = {
        "box": (10, 10, 300, 300),
        "sources": {"default@1500@+0", "default@1000@+0", "alt@1500@+0"},
        "weights": {"default@1500@+0": 7.3, "default@1000@+0": 7.0, "alt@1500@+0": 109.0},
    }
    evidence = _EyeEvidence(count=0, robust_pair=False, supported_eye_count=0)
    assert _accept_face_candidate_strict(entry, evidence, relative_size=0.28)
