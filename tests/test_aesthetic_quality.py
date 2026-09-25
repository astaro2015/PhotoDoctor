import numpy as np

from photodoctor.core.aesthetic_quality import analyze_aesthetic_quality
from photodoctor.core.main_subject import MainSubjectResult


def _subject(kind="person", box=None, confidence=0.85):
    if box is None:
        box = {"x": 0.30, "y": 0.15, "w": 0.40, "h": 0.55}
    return MainSubjectResult(kind, "people_portrait", confidence, box, box, "test", (), box["w"] * box["h"] * 100, ())


def test_unknown_subject_has_no_aesthetic_score():
    rgb = np.full((300, 400, 3), 128, np.uint8)
    result = analyze_aesthetic_quality(rgb, MainSubjectResult("unknown", "unknown_scene", 0.25, None, None, "none", (), 0.0, ()))
    assert result.score is None
    assert result.classification == "unknown"
    assert result.confidence <= 0.32


def test_balanced_subject_produces_separate_score():
    rgb = np.full((360, 540, 3), 105, np.uint8)
    rgb[70:285, 165:375] = (170, 145, 125)
    result = analyze_aesthetic_quality(rgb, _subject())
    assert result.score is not None
    assert 45 <= result.score <= 100
    assert set(result.components) == {"placement", "prominence", "edge_safety", "background_calmness", "tonal_separation"}
    assert result.confidence < 0.8


def test_edge_crowded_subject_scores_lower_than_balanced_subject():
    rgb = np.full((360, 540, 3), 110, np.uint8)
    # Distracting checker texture outside the subject area.
    y, x = np.indices((360, 540))
    checker = (((x // 8 + y // 8) % 2) * 120 + 50).astype(np.uint8)
    noisy = np.dstack([checker, checker, checker])
    balanced = analyze_aesthetic_quality(rgb, _subject())
    edge = analyze_aesthetic_quality(noisy, _subject(box={"x": 0.0, "y": 0.02, "w": 0.23, "h": 0.42}))
    assert balanced.score is not None and edge.score is not None
    assert edge.score < balanced.score
