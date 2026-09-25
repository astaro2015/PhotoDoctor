import photodoctor.core.validator as validator_module
from photodoctor.core.validator import surface_candidate_auto_repair_eligible
from pathlib import Path

import numpy as np
from PIL import Image

from photodoctor.core.analyzer import analyze_classical
from photodoctor.core.loader import load_image
from photodoctor.core.validator import (
    _surface_preview_candidates,
    apply_selected_preview, apply_validated_preview, preview_action_available, validate_recommendations,
)
from photodoctor.core.models import MetricResult


def save_img(tmp_path: Path, name: str, value: int) -> Path:
    p = tmp_path / name
    Image.new("RGB", (320, 240), (value, value, value)).save(p)
    return p


def test_low_information_dark_image_exposure_is_tested_but_manual_only(tmp_path):
    loaded = load_image(save_img(tmp_path, "dark.png", 72))
    result = analyze_classical(loaded)
    items = validate_recommendations(loaded, result.metrics)
    exposure = next(item for item in items if item.action_key == "exposure")
    assert exposure.tested
    assert exposure.technical_passed
    assert exposure.preview_available
    assert not exposure.auto_eligible
    assert not exposure.accepted
    assert exposure.target_after > exposure.target_before
    assert exposure.regressions["highlight_clip_delta_pct"] <= 1.0



def test_exposure_validator_uses_conservative_default_strength(tmp_path):
    loaded = load_image(save_img(tmp_path, "dark_strength.png", 72))
    result = analyze_classical(loaded)
    items = validate_recommendations(loaded, result.metrics)
    exposure = next(item for item in items if item.action_key == "exposure")
    assert exposure.default_strength <= 0.70
    assert exposure.candidate in {"gamma=0.94", "gamma=0.90", "gamma=0.86", "gamma=0.82"}


def test_validator_does_not_mutate_source_pixels(tmp_path):
    loaded = load_image(save_img(tmp_path, "dark2.png", 68))
    before = loaded.srgb.copy()
    result = analyze_classical(loaded)
    validate_recommendations(loaded, result.metrics)
    assert np.array_equal(before, loaded.srgb)


def test_clean_bright_enough_image_has_no_fake_exposure_validation(tmp_path):
    loaded = load_image(save_img(tmp_path, "mid.png", 170))
    result = analyze_classical(loaded)
    items = validate_recommendations(loaded, result.metrics)
    assert not any(item.action_key == "exposure" for item in items)


def test_preview_applies_only_accepted_candidates_without_mutating_source():
    source = np.full((80, 100, 3), 70, dtype=np.uint8)
    before = source.copy()
    preview = apply_validated_preview(source, [
        {"action_key": "exposure", "accepted": True, "candidate": "gamma=0.86"},
        {"action_key": "contrast", "accepted": False, "candidate": "mild_lab_clahe"},
    ])
    assert np.array_equal(source, before)
    assert preview.shape == source.shape
    assert float(preview.mean()) > float(source.mean())


def test_preview_ignores_unknown_or_rejected_actions():
    source = np.full((40, 50, 3), 90, dtype=np.uint8)
    preview = apply_validated_preview(source, [
        {"action_key": "mystery", "accepted": True, "candidate": "whatever"},
        {"action_key": "exposure", "accepted": False, "candidate": "gamma=0.80"},
    ])
    assert np.array_equal(preview, source)


def test_manual_selected_preview_can_apply_validator_rejected_action_without_changing_auto_behavior():
    source = np.full((80, 100, 3), 70, dtype=np.uint8)
    items = [{
        "action_key": "exposure",
        "tested": True,
        "accepted": False,
        "candidate": "gamma=0.92",
        "preview_available": True,
    }]
    auto = apply_validated_preview(source, items)
    manual = apply_selected_preview(source, items, {"exposure"})
    assert np.array_equal(auto, source)
    assert not np.array_equal(manual, source)
    assert float(manual.mean()) > float(source.mean())
    assert np.array_equal(source, np.full_like(source, 70))


def test_review_action_is_previewable_but_never_auto_accepted(tmp_path):
    loaded = load_image(save_img(tmp_path, "review.png", 72))
    result = analyze_classical(loaded)
    raw = result.metrics["decision_plan"].raw_value
    exposure_decision = next(item for item in raw["items"] if item["key"] == "exposure")
    exposure_decision["decision"] = "review"
    items = validate_recommendations(loaded, result.metrics)
    exposure = next(item for item in items if item.action_key == "exposure")
    assert exposure.tested
    assert not exposure.accepted
    assert not exposure.auto_eligible
    assert preview_action_available(exposure.to_dict())
    preview = apply_selected_preview(loaded.srgb, [exposure.to_dict()], {"exposure"})
    assert not np.array_equal(preview, loaded.srgb)


def test_adjustable_strength_zero_is_noop_and_full_strength_changes_preview():
    source = np.full((96, 128, 3), 82, dtype=np.uint8)
    items = [{
        "action_key": "contrast", "tested": True, "accepted": False,
        "candidate": "mild_lab_clahe", "preview_available": True,
        "adjustable": True, "default_strength": 0.4,
    }]
    zero = apply_selected_preview(source, items, {"contrast"}, {"contrast": 0.0})
    full = apply_selected_preview(source, items, {"contrast"}, {"contrast": 1.0})
    assert np.array_equal(zero, source)
    assert not np.array_equal(full, source)


def test_manual_sharpness_is_executable_but_never_required_for_auto_preview():
    source = np.zeros((96, 128, 3), dtype=np.uint8)
    source[:, 64:] = 150
    items = [{
        "action_key": "sharpness", "tested": True, "accepted": False,
        "candidate": "edge_aware_unsharp", "preview_available": True,
        "auto_eligible": False, "adjustable": True, "default_strength": 0.35,
    }]
    assert preview_action_available(items[0])
    auto = apply_validated_preview(source, items)
    manual = apply_selected_preview(source, items, {"sharpness"}, {"sharpness": 0.6})
    assert np.array_equal(auto, source)
    assert not np.array_equal(manual, source)


def test_surface_heal_is_bounded_to_selected_candidate_region():
    source = np.full((80, 100, 3), 120, dtype=np.uint8)
    source[35:37, 20:70] = 245
    cand = {"x": 0.18, "y": 0.40, "w": 0.56, "h": 0.12, "polarity": "bright"}
    item = {
        "action_key": "surface_defects", "tested": True, "accepted": False,
        "candidate": "bounded_surface_heal", "preview_available": True,
        "source_candidates": [cand], "adjustable": True, "default_strength": 0.5,
    }
    out = apply_selected_preview(source, [item], {"surface_defects"}, {"surface_defects": 0.6})
    assert preview_action_available(item)
    assert not np.array_equal(out, source)
    assert np.array_equal(out[:20], source[:20])
    assert np.array_equal(out[60:], source[60:])


def test_manual_filter_action_keys_have_executable_preview_contract():
    mapping = {
        "noise": "mild_nlm",
        "jpeg_artifacts": "mild_deblock",
        "edge_artifacts": "edge_halo_soften",
        "posterization": "mild_deband",
    }
    for key, candidate in mapping.items():
        assert preview_action_available({
            "action_key": key, "tested": True, "candidate": candidate, "preview_available": True,
        })


def test_user_selected_region_limits_global_correction_to_that_area():
    source = np.zeros((120, 160, 3), dtype=np.uint8)
    source[:] = 70
    source[20:100, 70:90] = 140
    item = {
        "action_key": "sharpness", "tested": True, "accepted": False,
        "candidate": "edge_aware_unsharp", "preview_available": True,
        "adjustable": True, "default_strength": 0.6,
    }
    region = {"x": 0.38, "y": 0.10, "w": 0.24, "h": 0.80}
    out = apply_selected_preview(
        source, [item], {"sharpness"}, {"sharpness": 1.0}, {"sharpness": region}
    )
    assert not np.array_equal(out, source)
    # Far from the feathered target the image must stay byte-identical.
    assert np.array_equal(out[:, :25], source[:, :25])
    assert np.array_equal(out[:, 135:], source[:, 135:])


def test_all_decision_fix_review_action_keys_have_preview_contracts():
    examples = {
        "red_eye": {"candidate": "localized_pupil_neutralization_relaxed", "source_candidates": [{"x": .1, "y": .1, "w": .1, "h": .1}]},
        "exposure": {"candidate": "gamma=0.92"},
        "contrast": {"candidate": "mild_lab_clahe"},
        "sharpness": {"candidate": "edge_aware_unsharp"},
        "surface_defects": {"candidate": "bounded_surface_heal", "source_candidates": [{"x": .1, "y": .1, "w": .1, "h": .1}]},
        "noise": {"candidate": "mild_nlm"},
        "jpeg_artifacts": {"candidate": "mild_deblock"},
        "edge_artifacts": {"candidate": "edge_halo_soften"},
        "posterization": {"candidate": "mild_deband"},
    }
    for key, extra in examples.items():
        item = {"action_key": key, "tested": True, "preview_available": True, **extra}
        assert preview_action_available(item), key


def test_invalid_explicit_local_region_fails_closed_instead_of_editing_whole_frame():
    source = np.zeros((120, 160, 3), dtype=np.uint8)
    source[:] = 70
    source[20:100, 70:90] = 140
    item = {
        "action_key": "sharpness", "tested": True, "accepted": False,
        "candidate": "edge_aware_unsharp", "preview_available": True,
        "adjustable": True, "default_strength": 1.0,
    }
    bad_regions = [
        {"x": float("nan"), "y": 0.1, "w": 0.2, "h": 0.8},
        {"x": 0.2, "y": 0.1, "w": 0.0, "h": 0.8},
        {"x": 2.0, "y": 2.0, "w": 0.2, "h": 0.2},
        {"x": "bad", "y": 0.1, "w": 0.2, "h": 0.8},
    ]
    for region in bad_regions:
        out = apply_selected_preview(
            source, [item], {"sharpness"}, {"sharpness": 1.0}, {"sharpness": region}
        )
        assert np.array_equal(out, source), region


def test_surface_preview_candidates_do_not_use_ai_label_to_choose_executable_targets():
    bright = {"x": 0.1, "y": 0.1, "w": 0.1, "h": 0.1, "polarity": "bright", "strength": 0.8}
    dark = {"x": 0.6, "y": 0.6, "w": 0.1, "h": 0.1, "polarity": "dark", "strength": 0.9}
    refined_dark = dict(dark, ai_label="defect", ai_confidence=0.99)
    metrics = {
        "surface_defects": MetricResult("surface_defects", {"boxes_norm": [bright, dark]}, 50, 0.5, "test"),
        "surface_refinement": MetricResult(
            "surface_refinement", {"refined_boxes_norm": [dict(bright, ai_label="natural_detail", ai_confidence=0.99), refined_dark]},
            None, 0.99, "ai",
        ),
    }
    selected = _surface_preview_candidates(metrics)
    assert selected == [bright]


def test_precise_validator_uses_larger_probe_copy(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    import photodoctor.core.validator as validator

    calls = []
    original = validator._technical_copy

    def capture(rgb, long_edge=1024):
        calls.append(long_edge)
        return original(rgb, long_edge=min(long_edge, 64))

    monkeypatch.setattr(validator, "_technical_copy", capture)
    image = SimpleNamespace(srgb=np.zeros((80, 120, 3), dtype=np.uint8))
    validator.validate_recommendations(image, {}, precision="normal")
    validator.validate_recommendations(image, {}, precision="precise")
    assert calls == [1024, 2048]


def test_face_skin_chroma_drift_falls_back_to_face_interior_when_skin_mask_misses():
    from photodoctor.core.validator import _face_skin_chroma_drift
    before = np.full((120, 160, 3), [220, 115, 70], dtype=np.uint8)
    after = before.copy()
    after[20:100, 35:125] = [185, 120, 110]
    regions = [{"x": 35/160, "y": 20/120, "w": 90/160, "h": 80/120}]
    assert _face_skin_chroma_drift(before, after, regions) > 7.0


def test_parallel_validator_preserves_exact_items(tmp_path, monkeypatch):
    import copy
    import photodoctor.core.validator as validator_module

    h, w = 270, 360
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.dstack((
        (62.0 + 0.18 * xx + 10.0 * np.sin(xx / 17.0)) % 180,
        (68.0 + 0.15 * yy + 8.0 * np.cos(yy / 19.0)) % 180,
        (72.0 + 0.11 * (xx + yy) + 7.0 * np.sin((xx + yy) / 23.0)) % 180,
    )).astype(np.uint8)
    path = tmp_path / "validator-parallel.png"
    Image.fromarray(rgb, "RGB").save(path)
    loaded = load_image(path)
    result = analyze_classical(loaded, precision="fast")

    raw = copy.deepcopy(result.metrics["decision_plan"].raw_value)
    raw["items"] = [
        {"key": "exposure", "title": "Exposure", "decision": "review", "severity": 20.0, "repairability": 70.0, "confidence": 0.7, "priority": 10.0, "reason": "test", "guardrail": "test"},
        {"key": "contrast", "title": "Contrast", "decision": "review", "severity": 20.0, "repairability": 70.0, "confidence": 0.7, "priority": 10.0, "reason": "test", "guardrail": "test"},
        {"key": "sharpness", "title": "Sharpness", "decision": "review", "severity": 20.0, "repairability": 70.0, "confidence": 0.7, "priority": 10.0, "reason": "test", "guardrail": "test"},
    ]
    metric = result.metrics["decision_plan"]
    result.metrics["decision_plan"] = MetricResult(
        metric.name, raw, metric.normalized_value, metric.confidence, metric.scale,
        region=metric.region, diagnostic=metric.diagnostic,
    )

    monkeypatch.setattr(validator_module, "get_validator_profile", lambda: (4, 1))
    serial = [item.to_dict() for item in validate_recommendations(loaded, result.metrics, precision="fast")]
    monkeypatch.setattr(validator_module, "get_validator_profile", lambda: (1, 3))
    parallel = [item.to_dict() for item in validate_recommendations(loaded, result.metrics, precision="fast")]
    assert parallel == serial


def test_surface_v9_is_hard_manual_only_even_for_high_confidence_face_crack():
    base = {
        "verification_label": "defect", "verification_confidence": 0.93,
        "polarity": "bright", "candidate_kind": "line",
        "detection_branch": "low_contrast_hough", "candidate_quality": 0.86,
        "context_contrast": 0.82, "side_similarity": 0.78,
    }
    for candidate in (
        dict(base, semantic_risk=0.0),
        dict(base, semantic_risk=0.58),
        dict(base, semantic_risk=0.95),
        dict(base, semantic_risk=0.0, polarity="dark"),
    ):
        assert surface_candidate_auto_repair_eligible(candidate) is False

    # Technical filtering still exists for a user-requested preview; it is a
    # separate concept from automatic application.
    assert validator_module._surface_candidate_repairable_for_manual_preview(dict(base, semantic_risk=0.58)) is True
    assert validator_module._surface_candidate_repairable_for_manual_preview(dict(base, semantic_risk=0.95)) is True
