from types import SimpleNamespace

import numpy as np

from photodoctor.core.models import MetricResult
from photodoctor.core.decisions import build_decision_plan
from photodoctor.core.validator import apply_selected_preview, preview_action_available, validate_recommendations
from photodoctor.core.white_balance import analyze_white_balance


def m(key, raw, score=None, confidence=0.8):
    return MetricResult(key, raw, score, confidence, "test")


def warm_rgb():
    base = np.full((180, 240, 3), 140, dtype=np.float32)
    base[35:145, 25:95] = [75, 95, 175]
    base[45:150, 150:225] = [170, 80, 60]
    return np.clip(base * np.array([1.35, 1.0, 0.58], dtype=np.float32), 0, 255).astype(np.uint8)


def metrics_for(rgb):
    advice = analyze_white_balance(rgb)
    metrics = {
        "semantic_context": m("semantic_context", {"classification": "general_photo", "face_count": 0, "archival_likelihood": 0.0}, None, 0.8),
        "image_tone": m("image_tone", {"classification": "color", "confidence": 0.9}, None, 0.9),
        "white_balance_advisor": m("white_balance_advisor", advice.to_raw(), 100 * (1-advice.cast_strength), advice.confidence),
        "neutral_balance": m("neutral_balance", {"cast_direction": "warm"}, 20.0, 0.75),
        "brightness": m("brightness", 0.25, 75.0, 0.8),
        "histogram": m("histogram", {"p50": 0.24}, None, 0.8),
        "contrast": m("contrast", {}, 80.0, 0.8),
        "local_contrast": m("local_contrast", {"classification": "normal"}, 80.0, 0.8),
        "faces": m("faces", {"face_count": 0, "faces": []}, None, 0.5),
        "main_subject": m("main_subject", {"subject_kind": "unknown", "confidence": 0.2}, None, 0.2),
    }
    plan = build_decision_plan(metrics)
    metrics["decision_plan"] = m("decision_plan", {"items": [x.to_dict() for x in plan]}, None, 0.8)
    return metrics, plan


def test_wb_advisor_creates_review_plan_without_duplicate_auto_color_only_action():
    metrics, plan = metrics_for(warm_rgb())
    keys = [x.key for x in plan]
    assert "white_balance" in keys
    # Contrast is already healthy, so the compound Photoshop-like action should not
    # duplicate a pure colour correction.
    assert "auto_tone_color" not in keys
    item = next(x for x in plan if x.key == "white_balance")
    assert item.decision == "review"
    assert "K" in item.reason
    assert "tint" in item.reason


def test_wb_validation_and_preview_are_executable():
    rgb = warm_rgb()
    metrics, _plan = metrics_for(rgb)
    image = SimpleNamespace(srgb=rgb)
    items = validate_recommendations(image, metrics)
    wb = next(x for x in items if x.action_key == "white_balance")
    raw = wb.to_dict()
    assert wb.technical_passed is True
    assert wb.accepted is False  # review stays manual despite successful technical probe
    assert preview_action_available(raw)
    out = apply_selected_preview(rgb, [raw], {"white_balance"}, {"white_balance": wb.default_strength})
    assert not np.array_equal(out, rgb)


def test_compound_auto_and_wb_do_not_stack():
    rgb = warm_rgb()
    advice = analyze_white_balance(rgb)
    wb = {
        "action_key": "white_balance", "tested": True, "preview_available": True,
        "candidate": "hybrid_awb_linear_rgb_v1", "adjustable": True,
        "default_strength": advice.neutralization_strength, "parameters": advice.to_raw(),
    }
    auto = {
        "action_key": "auto_tone_color", "tested": True, "preview_available": True,
        "candidate": "reference_auto_tone_contrast_color_v1", "adjustable": True,
        "default_strength": 0.8,
    }
    both = apply_selected_preview(rgb, [wb, auto], {"white_balance", "auto_tone_color"})
    auto_only = apply_selected_preview(rgb, [wb, auto], {"auto_tone_color"})
    assert np.array_equal(both, auto_only)


def test_mixed_light_uses_spatial_wb_candidate_and_validator():
    from photodoctor.core.white_balance import analyze_spatial_white_balance
    h, w = 240, 360
    base = np.full((h, w, 3), 145, dtype=np.float32)
    base[30:210, 20:100] = [130, 130, 130]
    base[50:190, 120:180] = [70, 100, 170]
    base[50:190, 200:260] = [170, 90, 70]
    base[30:210, 280:340] = [185, 185, 185]
    cast = np.ones_like(base)
    cast[:, :180] *= np.array([1.30, 1.0, 0.70], dtype=np.float32)
    cast[:, 180:] *= np.array([0.72, 1.0, 1.28], dtype=np.float32)
    rgb = np.clip(base * cast, 0, 255).astype(np.uint8)
    advice = analyze_white_balance(rgb)
    spatial = analyze_spatial_white_balance(rgb)
    metrics = {
        "semantic_context": m("semantic_context", {"classification": "general_photo", "face_count": 0, "archival_likelihood": 0.0}, None, 0.8),
        "image_tone": m("image_tone", {"classification": "color", "confidence": 0.9}, None, 0.9),
        "white_balance_advisor": m("white_balance_advisor", advice.to_raw(), 100 * (1-advice.cast_strength), advice.confidence),
        "local_white_balance": m("local_white_balance", spatial.to_raw(), 100 * (1-spatial.mixed_light_score), spatial.confidence),
        "neutral_balance": m("neutral_balance", {"cast_direction": "neutral"}, 85.0, 0.75),
        "brightness": m("brightness", 0.25, 75.0, 0.8),
        "histogram": m("histogram", {"p50": 0.24}, None, 0.8),
        "contrast": m("contrast", {}, 80.0, 0.8),
        "local_contrast": m("local_contrast", {"classification": "normal", "cells_norm": []}, 80.0, 0.8),
        "local_tone": m("local_tone", {"cells_norm": []}, 80.0, 0.8),
        "local_sharpness": m("local_sharpness", {"cells_norm": []}, 80.0, 0.8),
        "noise": m("noise", {"cells_norm": []}, 80.0, 0.8),
        "faces": m("faces", {"face_count": 0, "faces": []}, None, 0.5),
        "eyes": m("eyes", {"eyes": []}, None, 0.5),
        "main_subject": m("main_subject", {"subject_kind": "unknown", "confidence": 0.2}, None, 0.2),
    }
    plan = build_decision_plan(metrics)
    wb_decision = next(x for x in plan if x.key == "white_balance")
    assert wb_decision.decision == "review"
    assert "неоднород" in wb_decision.reason
    metrics["decision_plan"] = m("decision_plan", {"items": [x.to_dict() for x in plan]}, None, 0.8)
    items = validate_recommendations(SimpleNamespace(srgb=rgb), metrics)
    wb = next(x for x in items if x.action_key == "white_balance")
    assert wb.candidate == "spatial_white_balance_v1"
    assert wb.technical_passed is True
    assert wb.accepted is False
    assert preview_action_available(wb.to_dict())
