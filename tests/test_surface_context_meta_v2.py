from pathlib import Path

from photodoctor.ai.native_surface_context_meta_v2 import (
    _MODEL_PATH,
    feature_vector,
    predict_box,
    predict_boxes,
    suggestion_threshold,
)


def _box(**overrides):
    box = {
        "x": 0.1, "y": 0.1, "w": 0.02, "h": 0.08,
        "candidate_quality": 0.82,
        "context_contrast": 0.88,
        "texture_risk": 0.12,
        "parallel_neighbor_risk": 0.0,
        "candidate_kind": "line",
        "polarity": "bright",
    }
    box.update(overrides)
    return box


def test_context_meta_asset_is_bundled_and_threshold_from_cv_gate():
    assert Path(_MODEL_PATH).is_file()
    th = suggestion_threshold()
    assert 0.30 <= th <= 0.90


def test_context_meta_feature_contract_is_finite_and_eight_dimensional():
    v = feature_vector(_box(), (1200, 800))
    assert v.shape == (8,)
    assert all(float(x) == float(x) for x in v)


def test_context_meta_prediction_is_deterministic_and_advisory_only():
    a = predict_box(_box(), (1200, 800))
    b = predict_box(_box(), (1200, 800))
    assert a == b
    assert 0.0 <= a["defect_probability"] <= 1.0
    assert a["automatic_edits"] is False
    assert a["expert_verified"] is False


def test_context_meta_contract_excludes_weak_label_features():
    from photodoctor.ai.native_surface_context_meta_v2 import _bundle
    _, meta = _bundle()
    features = set(meta.get("features", []))
    forbidden = {
        "candidate_quality", "parallel_neighbor_risk", "response_ratio",
        "response_drop", "disappearance_coherence", "restored_response_median_near",
    }
    assert features.isdisjoint(forbidden)
    guard = meta.get("label_feature_leakage_guard", {})
    assert guard.get("status") == "enforced"


def test_context_meta_packaged_candidate_meets_precision_first_gate():
    from photodoctor.ai.native_surface_context_meta_v2 import _bundle
    _, meta = _bundle()
    gate = meta.get("precision_gate", {})
    assert gate.get("precision", 0.0) >= 0.85
    assert gate.get("min_group_precision", 0.0) >= 0.85
    assert gate.get("precision_wilson_lower_95", 0.0) >= 0.80
    groups = [row for row in gate.get("groups", []) if row.get("predictions", 0) > 0]
    assert len(groups) >= 3
    assert meta.get("expert_verified") is False
    assert meta.get("automatic_edits") is False


def test_context_meta_batch_is_bit_identical_to_scalar_predictions():
    boxes = [
        _box(x=0.10 + i * 0.03, y=0.08 + (i % 3) * 0.07,
             w=0.015 + (i % 2) * 0.01, h=0.05 + (i % 4) * 0.02,
             context_contrast=0.2 + i * 0.07, texture_risk=0.8 - i * 0.06,
             candidate_kind=("line" if i % 3 == 0 else "irregular"),
             polarity=("bright" if i % 2 == 0 else "dark"))
        for i in range(10)
    ]
    scalar = [predict_box(box, (1200, 800)) for box in boxes]
    assert predict_boxes(boxes, (1200, 800)) == scalar
