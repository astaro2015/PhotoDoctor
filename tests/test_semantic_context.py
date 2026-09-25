from photodoctor.core.semantic_context import analyze_semantic_context


def ctx(face_count, tone="color", tone_conf=0.8, surface=0, surface_conf=0.5, face_conf=0.7):
    return analyze_semantic_context(
        face_count=face_count,
        face_detector_confidence=face_conf,
        tone_class=tone,
        tone_confidence=tone_conf,
        surface_candidate_count=surface,
        surface_confidence=surface_conf,
    )


def test_sepia_people_with_surface_ageing_becomes_archival_portrait():
    result = ctx(2, tone="sepia", tone_conf=0.92, surface=20, surface_conf=0.65)
    assert result.classification == "archival_portrait"
    assert result.people_context == "group"
    assert result.archival_likelihood >= 0.5
    assert result.preservation_priority == "faces_and_original_character"


def test_multiple_faces_in_normal_color_is_group_portrait():
    result = ctx(3, tone="color", surface=0)
    assert result.classification == "group_portrait"
    assert result.people_context == "group"


def test_single_face_is_portrait():
    result = ctx(1, tone="color")
    assert result.classification == "portrait"
    assert result.people_context == "single"


def test_no_confident_faces_is_general_photo_without_claiming_scene_identity():
    result = ctx(0, tone="color")
    assert result.classification == "general_photo"
    assert result.people_context == "none"
    assert result.confidence <= 0.58
