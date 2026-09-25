from pathlib import Path

import numpy as np

from photodoctor.core.white_balance import (
    FEATURE_SCHEMA,
    MODEL_ID,
    analyze_white_balance,
    apply_white_balance,
)


def _warm_scene() -> np.ndarray:
    # Neutral wall + coloured material under a strong warm cast.
    base = np.full((180, 240, 3), 145, dtype=np.float32)
    base[40:140, 20:90] = [70, 90, 170]
    base[50:150, 150:220] = [170, 80, 65]
    cast = np.array([1.35, 1.0, 0.58], dtype=np.float32)
    return np.clip(base * cast[None, None, :], 0, 255).astype(np.uint8)


def _channel_spread(rgb: np.ndarray) -> float:
    med = np.median(rgb.reshape(-1, 3).astype(np.float32), axis=0)
    return float(med.max() - med.min())


def test_contract_is_explicit_and_non_neural_baseline():
    advice = analyze_white_balance(_warm_scene())
    assert advice.model_id == MODEL_ID == "native_white_balance_advisor_v1"
    assert advice.feature_schema == FEATURE_SCHEMA == "white_balance_features_v1"
    assert advice.advisor_kind == "hybrid_expert_baseline"
    assert advice.neural_model_used is False


def test_warm_scene_gets_cooler_recommendation_and_preserves_atmosphere():
    rgb = _warm_scene()
    advice = analyze_white_balance(rgb)
    assert advice.cast_direction == "warm"
    assert advice.recommended_temperature_shift_k < -450
    assert 0.35 <= advice.neutralization_strength <= 0.86
    assert 0.14 <= advice.atmosphere_preservation <= 0.65
    assert advice.correction_needed


def test_apply_reduces_warm_channel_spread_without_large_luma_change():
    rgb = _warm_scene()
    advice = analyze_white_balance(rgb)
    out = apply_white_balance(rgb, advice)
    assert _channel_spread(out) < _channel_spread(rgb)
    before = rgb.astype(np.float32).mean()
    after = out.astype(np.float32).mean()
    assert abs(after - before) < 18.0


def test_neutral_gray_is_not_overcorrected():
    x = np.linspace(40, 210, 220, dtype=np.float32)
    gray = np.repeat(x[None, :, None], 140, axis=0)
    rgb = np.repeat(gray, 3, axis=2).astype(np.uint8)
    advice = analyze_white_balance(rgb)
    assert abs(advice.recommended_temperature_shift_k) < 300
    assert abs(advice.recommended_tint_shift) < 2.5
    assert not advice.correction_needed
    out = apply_white_balance(rgb, advice)
    assert np.max(np.abs(out.astype(np.int16) - rgb.astype(np.int16))) <= 1


def test_archival_or_sepia_is_guarded():
    rgb = _warm_scene()
    advice = analyze_white_balance(rgb, tone_class="sepia", archival_likelihood=0.9)
    assert advice.neutralization_strength == 0.0
    assert not advice.correction_needed


def test_faces_can_reduce_but_not_invert_recommendation():
    rgb = _warm_scene()
    no_face = analyze_white_balance(rgb)
    with_face = analyze_white_balance(rgb, face_boxes=[(20, 30, 80, 100)])
    assert with_face.neutralization_strength <= no_face.neutralization_strength + 1e-9
    assert with_face.cast_direction == no_face.cast_direction


def test_strength_zero_is_identity():
    rgb = _warm_scene()
    advice = analyze_white_balance(rgb)
    out = apply_white_balance(rgb, advice, strength=0.0)
    assert np.array_equal(out, rgb)


def _mixed_light_scene() -> tuple[np.ndarray, np.ndarray]:
    h, w = 240, 360
    base = np.full((h, w, 3), 145, dtype=np.float32)
    base[30:210, 20:100] = [130, 130, 130]
    base[50:190, 120:180] = [70, 100, 170]
    base[50:190, 200:260] = [170, 90, 70]
    base[30:210, 280:340] = [185, 185, 185]
    cast = np.ones_like(base)
    cast[:, :180] *= np.array([1.30, 1.0, 0.70], dtype=np.float32)
    cast[:, 180:] *= np.array([0.72, 1.0, 1.28], dtype=np.float32)
    return np.clip(base * cast, 0, 255).astype(np.uint8), base.astype(np.uint8)


def test_colorful_scene_without_neutral_evidence_abstains():
    rgb = np.zeros((180, 240, 3), dtype=np.uint8)
    rgb[:, :80] = [220, 80, 30]
    rgb[:, 80:160] = [190, 40, 130]
    rgb[:, 160:] = [240, 120, 20]
    advice = analyze_white_balance(rgb)
    assert advice.neutral_fraction < 0.008
    assert advice.confidence <= 0.58 + 1e-9
    assert advice.neutralization_strength <= 0.38 + 1e-9
    assert not advice.correction_needed


def test_spatial_white_balance_detects_opposing_mixed_light():
    from photodoctor.core.white_balance import analyze_spatial_white_balance
    rgb, _base = _mixed_light_scene()
    result = analyze_spatial_white_balance(rgb)
    assert result.classification == "mixed_light"
    assert result.confidence >= 0.55
    assert result.mixed_light_score >= 0.70
    temps = [float(cell["temperature_shift_k"]) for cell in result.cells_norm]
    assert min(temps) < -700
    assert max(temps) > 700


def test_spatial_white_balance_improves_known_mixed_light_scene():
    from photodoctor.core.white_balance import analyze_spatial_white_balance, apply_spatial_white_balance, spatial_neutral_error
    from photodoctor.core.local_correction_planner import build_local_correction_plan
    from photodoctor.core.models import MetricResult
    rgb, base = _mixed_light_scene()
    global_advice = analyze_white_balance(rgb)
    spatial = analyze_spatial_white_balance(rgb)
    metrics = {
        "local_white_balance": MetricResult("local_white_balance", spatial.to_raw(), None, spatial.confidence, "test"),
        "white_balance_advisor": MetricResult("white_balance_advisor", global_advice.to_raw(), None, global_advice.confidence, "test"),
        "faces": MetricResult("faces", {"faces": []}, None, 0.5, "test"),
        "eyes": MetricResult("eyes", {"eyes": []}, None, 0.5, "test"),
    }
    plan = build_local_correction_plan(metrics)
    raw = plan.to_raw()
    raw["global_advice"] = global_advice.to_raw()
    before_error = spatial_neutral_error(rgb, list(plan.white_balance_cells))
    out = apply_spatial_white_balance(rgb, raw, strength=0.80)
    after_error = spatial_neutral_error(out, list(plan.white_balance_cells))
    assert after_error < before_error * 0.90
    assert np.mean(np.abs(out.astype(np.float32) - base.astype(np.float32))) < np.mean(np.abs(rgb.astype(np.float32) - base.astype(np.float32)))
    assert np.mean(np.abs(out[:, :180].astype(np.float32) - base[:, :180].astype(np.float32))) < np.mean(np.abs(rgb[:, :180].astype(np.float32) - base[:, :180].astype(np.float32)))
    assert np.mean(np.abs(out[:, 180:].astype(np.float32) - base[:, 180:].astype(np.float32))) < np.mean(np.abs(rgb[:, 180:].astype(np.float32) - base[:, 180:].astype(np.float32)))


def test_spatial_wb_skin_guard_reduces_local_residual_on_face():
    from photodoctor.core.white_balance import apply_spatial_white_balance
    rgb = np.full((160, 220, 3), [185, 132, 108], dtype=np.uint8)
    cells = [
        {"x": 0.0, "y": 0.0, "w": 0.65, "h": 1.0, "target_log_r": -0.18, "target_log_g": 0.01, "target_log_b": 0.17, "confidence": 0.8},
        {"x": 0.35, "y": 0.0, "w": 0.65, "h": 1.0, "target_log_r": -0.18, "target_log_g": 0.01, "target_log_b": 0.17, "confidence": 0.8},
    ]
    unguarded = {
        "white_balance_cells": cells,
        "white_balance_base_strength": 0.0,
        "face_regions": [],
        "eye_regions": [],
    }
    guarded = {
        "white_balance_cells": cells,
        "white_balance_base_strength": 0.0,
        "face_regions": [{"x": 0.30, "y": 0.20, "w": 0.40, "h": 0.60}],
        "eye_regions": [],
    }
    out_unguarded = apply_spatial_white_balance(rgb, unguarded, strength=1.0)
    out_guarded = apply_spatial_white_balance(rgb, guarded, strength=1.0)
    diff_unguarded = np.mean(np.abs(out_unguarded.astype(np.float32) - rgb.astype(np.float32)), axis=2)
    diff_guarded = np.mean(np.abs(out_guarded.astype(np.float32) - rgb.astype(np.float32)), axis=2)
    face_change_unguarded = float(np.mean(diff_unguarded[32:128, 66:154]))
    face_change_guarded = float(np.mean(diff_guarded[32:128, 66:154]))
    assert face_change_guarded < face_change_unguarded * 0.70


def test_spatial_wb_does_not_call_colored_object_halves_mixed_light():
    from photodoctor.core.white_balance import analyze_spatial_white_balance
    h, w = 240, 360
    yy, xx = np.mgrid[:h, :w]
    texture = (20 * np.sin(xx / 7) + 15 * np.sin(yy / 11)).astype(np.float32)
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:, :180] = np.array([180, 65, 35], dtype=np.float32) + texture[:, :180, None]
    rgb[:, 180:] = np.array([35, 80, 190], dtype=np.float32) + texture[:, 180:, None]
    result = analyze_spatial_white_balance(np.clip(rgb, 0, 255).astype(np.uint8))
    assert result.neutral_anchor_cells == 0
    assert result.classification == "uncertain"
    assert result.confidence <= 0.46 + 1e-9


def test_gradual_mixed_light_is_at_least_local_cast_when_neutral_anchors_exist():
    from photodoctor.core.white_balance import analyze_spatial_white_balance
    h, w = 240, 360
    base = np.full((h, w, 3), 145, dtype=np.float32)
    base[40:200, 100:150] = [80, 110, 170]
    base[40:200, 210:260] = [170, 100, 70]
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    left = np.array([1.25, 1.0, 0.78], dtype=np.float32)
    right = np.array([0.78, 1.0, 1.25], dtype=np.float32)
    cast = left * (1.0 - t) + right * t
    rgb = np.clip(base * cast, 0, 255).astype(np.uint8)
    result = analyze_spatial_white_balance(rgb)
    assert result.neutral_anchor_cells >= 2
    assert result.classification in {"local_cast", "mixed_light"}
    assert result.confidence >= 0.55


def test_large_neutral_background_beats_two_saturated_objects():
    h, w = 240, 360
    yy, xx = np.mgrid[:h, :w]
    texture = (20 * np.sin(xx / 7) + 15 * np.sin(yy / 11)).astype(np.float32)
    rgb = np.full((h, w, 3), 150, dtype=np.float32) + texture[..., None] * 0.25
    rgb[40:200, 30:130] = [210, 70, 40]
    rgb[40:200, 230:330] = [40, 80, 210]
    advice = analyze_white_balance(np.clip(rgb, 0, 255).astype(np.uint8))
    assert advice.neutral_fraction >= 0.50
    assert abs(advice.recommended_temperature_shift_k) < 120
    assert abs(advice.recommended_tint_shift) < 2.0
    assert not advice.correction_needed


def test_uniform_strong_cast_is_not_misclassified_as_mixed_light():
    from photodoctor.core.white_balance import analyze_spatial_white_balance
    base = _mixed_light_scene()[1].astype(np.float32)
    for gains in (
        np.array([1.25, 1.0, 0.78], dtype=np.float32),
        np.array([0.78, 1.0, 1.25], dtype=np.float32),
        np.array([0.90, 1.18, 0.90], dtype=np.float32),
        np.array([1.12, 0.86, 1.12], dtype=np.float32),
    ):
        rgb = np.clip(base * gains, 0, 255).astype(np.uint8)
        result = analyze_spatial_white_balance(rgb)
        assert result.classification in {"uniform", "uncertain"}
        assert result.mixed_light_score < 0.55


def test_mixed_light_has_spatial_coherence_support():
    from photodoctor.core.white_balance import analyze_spatial_white_balance
    rgb, _base = _mixed_light_scene()
    result = analyze_spatial_white_balance(rgb)
    assert result.classification == "mixed_light"
    assert result.spatial_coherence >= 0.20


def test_spatial_wb_face_guard_still_works_when_casted_skin_misses_classic_ycrcb():
    from photodoctor.core.white_balance import apply_spatial_white_balance
    rgb = np.full((160, 220, 3), [220, 115, 70], dtype=np.uint8)
    cells = [
        {"x": 0.0, "y": 0.0, "w": 0.65, "h": 1.0, "target_log_r": -0.18, "target_log_g": 0.01, "target_log_b": 0.17, "confidence": 0.8},
        {"x": 0.35, "y": 0.0, "w": 0.65, "h": 1.0, "target_log_r": -0.18, "target_log_g": 0.01, "target_log_b": 0.17, "confidence": 0.8},
    ]
    unguarded = {"white_balance_cells": cells, "white_balance_base_strength": 0.0, "face_regions": [], "eye_regions": []}
    guarded = {"white_balance_cells": cells, "white_balance_base_strength": 0.0, "face_regions": [{"x": 0.30, "y": 0.20, "w": 0.40, "h": 0.60}], "eye_regions": []}
    a = apply_spatial_white_balance(rgb, unguarded, strength=1.0)
    b = apply_spatial_white_balance(rgb, guarded, strength=1.0)
    da = np.mean(np.abs(a.astype(np.float32) - rgb.astype(np.float32)), axis=2)
    db = np.mean(np.abs(b.astype(np.float32) - rgb.astype(np.float32)), axis=2)
    assert float(np.mean(db[32:128, 66:154])) < float(np.mean(da[32:128, 66:154])) * 0.88


def test_colour_mosaic_without_neutral_anchors_does_not_fake_mixed_light():
    from photodoctor.core.white_balance import analyze_spatial_white_balance
    import cv2
    colors = [
        [15,183,225],[182,69,171],[126,38,105],[157,46,179],
        [163,34,55],[37,128,55],[112,17,147],[33,117,12],
        [51,179,193],[157,153,29],[73,104,167],[47,122,130],
    ]
    h, w = 240, 360
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    idx = 0
    for row in range(3):
        for col in range(4):
            rgb[row*h//3:(row+1)*h//3, col*w//4:(col+1)*w//4] = colors[idx]
            idx += 1
    yy, xx = np.mgrid[:h, :w]
    rgb += (8*np.sin(xx/7 + 37) + 6*np.sin(yy/11 + 37*0.3))[..., None]
    result = analyze_spatial_white_balance(np.clip(rgb, 0, 255).astype(np.uint8))
    assert result.neutral_anchor_cells == 0
    assert result.classification in {"uniform", "uncertain"}
