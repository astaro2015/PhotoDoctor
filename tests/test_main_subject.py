import numpy as np

from photodoctor.core.eyes import EyeRegion
from photodoctor.core.faces import FaceRegion
from photodoctor.core.main_subject import analyze_main_subject
from photodoctor.core.semantic_context import SemanticContext


def _semantic(kind="portrait", faces=1):
    return SemanticContext(kind, "single" if faces == 1 else "group", faces, 0.0, "face", 0.8, [])


def _face(x, y, w, h, *, sharp=75.0, source="auto"):
    return FaceRegion(x, y, w, h, 0.4, 100.0, 100.0, sharp, 0.75, source)


def _eye(face_index, x, y):
    return EyeRegion(face_index, x, y, 12, 8, 20.0, 20.0, 70.0, 0.55)


def test_single_face_becomes_main_person():
    rgb = np.zeros((400, 600, 3), dtype=np.uint8)
    face = _face(220, 80, 120, 120)
    result = analyze_main_subject(rgb, [face], [_eye(0, 250, 120), _eye(0, 300, 120)], _semantic(), face_detector_confidence=0.8)
    assert result.subject_kind == "person"
    assert result.scene_kind == "people_portrait"
    assert result.confidence > 0.7
    assert result.box_norm is not None
    assert result.protection_box_norm is not None
    assert result.face_indices == (0,)


def test_group_faces_are_group_subject():
    rgb = np.zeros((500, 800, 3), dtype=np.uint8)
    faces = [_face(170, 110, 130, 130), _face(430, 120, 125, 125)]
    result = analyze_main_subject(rgb, faces, [], _semantic("group_portrait", 2), face_detector_confidence=0.75)
    assert result.subject_kind == "people_group"
    assert result.scene_kind == "people_group"
    assert result.face_indices == (0, 1)
    assert result.box_norm["w"] > 0.45


def test_manual_face_is_valid_subject():
    rgb = np.zeros((300, 500, 3), dtype=np.uint8)
    result = analyze_main_subject(rgb, [_face(180, 60, 100, 100, source="manual")], [], _semantic(), face_detector_confidence=0.65)
    assert result.subject_kind == "person"
    assert result.confidence >= 0.65


def test_flat_image_returns_unknown_instead_of_fake_subject():
    rgb = np.full((320, 480, 3), 128, dtype=np.uint8)
    result = analyze_main_subject(rgb, [], [], _semantic("general_photo", 0), face_detector_confidence=0.5)
    assert result.subject_kind == "unknown"
    assert result.box_norm is None
    assert result.confidence <= 0.38


def test_visual_object_can_be_subject_without_faces():
    rgb = np.full((360, 540, 3), 115, dtype=np.uint8)
    rgb[100:270, 190:360] = (225, 35, 30)
    rgb[125:245, 215:335] = (245, 210, 40)
    result = analyze_main_subject(rgb, [], [], _semantic("general_photo", 0), face_detector_confidence=0.5)
    assert result.subject_kind == "visual_region"
    assert result.box_norm is not None
    assert result.confidence >= 0.42
    cx = result.box_norm["x"] + result.box_norm["w"] / 2
    cy = result.box_norm["y"] + result.box_norm["h"] / 2
    assert 0.3 < cx < 0.7
    assert 0.25 < cy < 0.75
