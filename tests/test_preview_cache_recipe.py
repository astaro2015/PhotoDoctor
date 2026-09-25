from photodoctor.core.preview_cache import build_preview_recipe_signature


def _wb_item(default_strength: float = 0.55):
    return {
        "action_key": "white_balance",
        "candidate": "hybrid_awb_linear_rgb_v1",
        "default_strength": default_strength,
    }


def test_white_balance_strength_change_always_changes_cache_signature():
    items = [_wb_item()]
    sig_35 = build_preview_recipe_signature({"white_balance"}, {"white_balance": 0.35}, {}, items)
    sig_60 = build_preview_recipe_signature({"white_balance"}, {"white_balance": 0.60}, {}, items)
    assert sig_35 != sig_60


def test_default_strength_and_explicit_same_strength_share_signature():
    items = [_wb_item(0.55)]
    default_sig = build_preview_recipe_signature({"white_balance"}, {}, {}, items)
    explicit_sig = build_preview_recipe_signature({"white_balance"}, {"white_balance": 0.55}, {}, items)
    assert default_sig == explicit_sig


def test_region_change_changes_cache_signature():
    items = [_wb_item()]
    a = {"white_balance": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}}
    b = {"white_balance": {"x": 0.1, "y": 0.2, "w": 0.35, "h": 0.4}}
    sig_a = build_preview_recipe_signature({"white_balance"}, {"white_balance": 0.55}, a, items)
    sig_b = build_preview_recipe_signature({"white_balance"}, {"white_balance": 0.55}, b, items)
    assert sig_a != sig_b


def test_candidate_change_changes_cache_signature():
    first = [_wb_item()]
    second = [{**_wb_item(), "candidate": "spatial_white_balance_v1"}]
    sig_a = build_preview_recipe_signature({"white_balance"}, {"white_balance": 0.55}, {}, first)
    sig_b = build_preview_recipe_signature({"white_balance"}, {"white_balance": 0.55}, {}, second)
    assert sig_a != sig_b


def test_white_balance_preview_pixels_change_when_strength_changes():
    import numpy as np
    from photodoctor.core.validator import apply_selected_preview

    y, x = np.mgrid[0:72, 0:96]
    rgb = np.empty((72, 96, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(135 + x // 5 + y // 12, 0, 255)
    rgb[..., 1] = np.clip(108 + x // 8 + y // 10, 0, 255)
    rgb[..., 2] = np.clip(82 + x // 12 + y // 14, 0, 255)

    params = {
        "gain_r": 0.78,
        "gain_g": 1.00,
        "gain_b": 1.28,
        "technical_temperature_shift_k": -900.0,
        "recommended_temperature_shift_k": -650.0,
        "technical_tint_shift": 0.0,
        "recommended_tint_shift": 0.0,
        "neutralization_strength": 0.55,
        "atmosphere_preservation": 0.35,
        "cast_strength": 0.60,
        "cast_direction": "warm",
        "confidence": 0.90,
        "estimator_agreement": 0.90,
        "neutral_fraction": 0.25,
        "face_guard": 1.0,
        "correction_needed": True,
    }
    item = {
        "action_key": "white_balance",
        "tested": True,
        "preview_available": True,
        "candidate": "hybrid_awb_linear_rgb_v1",
        "parameters": params,
        "adjustable": True,
        "default_strength": 0.55,
    }

    p35 = apply_selected_preview(rgb, [item], {"white_balance"}, {"white_balance": 0.35}, {})
    p60 = apply_selected_preview(rgb, [item], {"white_balance"}, {"white_balance": 0.60}, {})
    assert not np.array_equal(p35, p60)
    assert float(np.mean(np.abs(p35.astype(np.int16) - p60.astype(np.int16)))) > 0.5
