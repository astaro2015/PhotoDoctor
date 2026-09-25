from __future__ import annotations

from photodoctor.core.validator import PREVIEW_ACTION_ORDER, _preview_order_key


def _item(action_key: str, candidate: str = "") -> dict[str, object]:
    return {"action_key": action_key, "candidate": candidate}


def test_multi_correction_pipeline_has_structure_tone_resolution_order():
    # Keep the pipeline independent from whatever order rows happen to have in
    # the UI.  Physical restoration first, global tone/color later, and final
    # resolution/detail processing last.
    items = [
        _item("sharpness", "edge_aware_unsharp"),
        _item("contrast", "mild_lab_clahe"),
        _item("red_eye", "localized_pupil_neutralization"),
        _item("surface_defects", "bounded_surface_heal"),
        _item("super_resolution", "almaz_x2_identity_guard_v1"),
        _item("posterization", "mild_deband"),
        _item("white_balance", "spatial_white_balance_v1"),
        _item("edge_artifacts", "edge_halo_soften"),
        _item("exposure", "gamma=0.92"),
        _item("noise", "mild_nlm"),
        _item("jpeg_artifacts", "mild_deblock"),
    ]
    ordered = sorted(items, key=_preview_order_key)
    assert [x["action_key"] for x in ordered] == [
        "surface_defects",
        "red_eye",
        "jpeg_artifacts",
        "noise",
        "edge_artifacts",
        "white_balance",
        "exposure",
        "contrast",
        "posterization",
        "super_resolution",
        "sharpness",
    ]
    assert PREVIEW_ACTION_ORDER["surface_defects"] < PREVIEW_ACTION_ORDER["white_balance"]
    assert PREVIEW_ACTION_ORDER["super_resolution"] < PREVIEW_ACTION_ORDER["sharpness"]


def test_almaz_upgrade_candidates_keep_restoration_stage_regardless_of_action_bucket():
    items = [
        _item("sharpness", "edge_aware_unsharp"),
        _item("sharpness", "almaz_ai_deblur_v1"),
        _item("noise", "almaz_ai_denoise_v1"),
        _item("jpeg_artifacts", "almaz_ai_jpeg_recovery_v1"),
        _item("white_balance", "hybrid_awb_linear_rgb_v1"),
    ]
    ordered = sorted(items, key=_preview_order_key)
    assert [x["candidate"] for x in ordered] == [
        "almaz_ai_jpeg_recovery_v1",
        "almaz_ai_denoise_v1",
        "almaz_ai_deblur_v1",
        "hybrid_awb_linear_rgb_v1",
        "edge_aware_unsharp",
    ]


def test_apply_selected_preview_executes_actions_in_pipeline_order(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from photodoctor.core import validator

    calls: list[str] = []
    rgb = np.full((24, 32, 3), 100, dtype=np.uint8)

    monkeypatch.setattr(validator, "_bounded_surface_heal", lambda x, c, s: calls.append("surface") or x.copy())
    monkeypatch.setattr(validator, "correct_red_eye", lambda x, c, **k: (calls.append("red_eye") or x.copy(), {}))
    monkeypatch.setattr(validator, "_mild_deblock", lambda x, s: calls.append("jpeg") or x.copy())
    monkeypatch.setattr(validator, "_mild_denoise", lambda x, s: calls.append("denoise") or x.copy())
    monkeypatch.setattr(validator, "apply_almaz_restoration", lambda x, **k: calls.append("deblur") or x.copy())
    monkeypatch.setattr(validator, "_mild_edge_soften", lambda x, s: calls.append("halos") or x.copy())
    monkeypatch.setattr(validator, "_gamma_lift", lambda x, g: calls.append("exposure") or x.copy())
    monkeypatch.setattr(validator, "_mild_local_contrast", lambda x, s: calls.append("contrast") or x.copy())
    monkeypatch.setattr(validator, "_mild_deband", lambda x, s: calls.append("deband") or x.copy())
    monkeypatch.setattr(validator, "apply_super_resolution_x2", lambda x, **k: calls.append("sr") or np.repeat(np.repeat(x, 2, 0), 2, 1))
    monkeypatch.setattr(validator, "_edge_aware_sharpen", lambda x, s: calls.append("sharpness") or x.copy())
    monkeypatch.setattr(validator, "validate_almaz_transition", lambda *a, **k: SimpleNamespace(accepted=True, message="ok"))
    monkeypatch.setattr(validator, "guard_archival_chroma", lambda before, after: (after, SimpleNamespace(applied=False)))

    items = [
        {"action_key":"sharpness","candidate":"edge_aware_unsharp","preview_available":True,"default_strength":0.5},
        {"action_key":"super_resolution","candidate":"almaz_x2_identity_guard_v1","preview_available":True,"default_strength":0.5,"parameters":{}},
        {"action_key":"posterization","candidate":"mild_deband","preview_available":True,"default_strength":0.5},
        {"action_key":"contrast","candidate":"mild_lab_clahe","preview_available":True,"default_strength":0.5},
        {"action_key":"exposure","candidate":"gamma=0.92","preview_available":True,"default_strength":0.5},
        {"action_key":"edge_artifacts","candidate":"edge_halo_soften","preview_available":True,"default_strength":0.5},
        {"action_key":"sharpness","candidate":"almaz_ai_deblur_v1","preview_available":True,"default_strength":0.5,"parameters":{}},
        {"action_key":"noise","candidate":"mild_nlm","preview_available":True,"default_strength":0.5},
        {"action_key":"jpeg_artifacts","candidate":"mild_deblock","preview_available":True,"default_strength":0.5},
        {"action_key":"red_eye","candidate":"localized_pupil_neutralization","preview_available":True,"tested":True,"default_strength":0.5,"source_candidates":[{"x":0.2,"y":0.2,"w":0.1,"h":0.1}]},
        {"action_key":"surface_defects","candidate":"bounded_surface_heal","preview_available":True,"tested":True,"default_strength":0.5,"source_candidates":[{"x":0.2,"y":0.2,"w":0.2,"h":0.1,"polarity":"bright"}]},
    ]
    selected = {str(x["action_key"]) for x in items}
    validator.apply_selected_preview(rgb, items, selected)
    assert calls == ["surface", "red_eye", "jpeg", "denoise", "deblur", "halos", "exposure", "contrast", "deband", "sr", "sharpness"]
