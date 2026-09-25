from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from photodoctor.core import validator


def _item(key: str, candidate: str, parameters=None):
    return {
        "action_key": key,
        "tested": True,
        "preview_available": True,
        "candidate": candidate,
        "parameters": parameters or {},
        "default_strength": 0.5,
    }


def test_almaz_deblur_runs_before_sr_and_ordinary_sharpen_after(monkeypatch):
    calls: list[str] = []
    def restoration(rgb, **kwargs):
        calls.append("deblur")
        return rgb.copy()
    def sr(rgb, **kwargs):
        calls.append("sr")
        return np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
    def sharpen(rgb, strength):
        calls.append("sharpen")
        return rgb.copy()
    monkeypatch.setattr(validator, "apply_almaz_restoration", restoration)
    monkeypatch.setattr(validator, "apply_super_resolution_x2", sr)
    monkeypatch.setattr(validator, "_edge_aware_sharpen", sharpen)
    monkeypatch.setattr(validator, "validate_almaz_transition", lambda *a, **k: SimpleNamespace(accepted=True, message="ok"))
    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    items = [
        _item("sharpness", "edge_aware_unsharp"),
        _item("super_resolution", "almaz_x2_identity_guard_v1", {"face_boxes": []}),
        _item("sharpness", "almaz_ai_deblur_v1", {"face_boxes": []}),
    ]
    out = validator.apply_selected_preview(rgb, items, {"sharpness", "super_resolution"})
    assert calls == ["deblur", "sr", "sharpen"]
    assert out.shape == (32, 40, 3)


def test_almaz_preview_auto_softens_after_safety_block(monkeypatch):
    calls: list[str] = []
    progress: list[str] = []

    def restoration(rgb, **kwargs):
        calls.append("denoise")
        out = rgb.copy()
        out[:, :] = 180
        return out

    safety_calls = {"count": 0}
    def safety(*args, **kwargs):
        safety_calls["count"] += 1
        if safety_calls["count"] == 1:
            return SimpleNamespace(accepted=False, message="ALMAZ safety BLOCK (clip)", clipping_delta_pct=2.47)
        return SimpleNamespace(accepted=True, message="ALMAZ safety PASS", clipping_delta_pct=1.20)

    monkeypatch.setattr(validator, "apply_almaz_restoration", restoration)
    monkeypatch.setattr(validator, "validate_almaz_transition", safety)
    rgb = np.full((12, 16, 3), 100, dtype=np.uint8)
    items = [_item("noise", "almaz_ai_denoise_v1", {"face_boxes": []})]
    out = validator.apply_selected_preview(
        rgb, items, {"noise"}, action_strengths={"noise": 0.8}, progress=progress.append
    )
    assert calls == ["denoise"]
    assert safety_calls["count"] == 2
    assert np.all(out == 160)  # 75% of the model delta: 100 -> 180 becomes 160
    assert any("автоматически уменьшаю эффект" in msg for msg in progress)
    assert any("найден безопасный вариант" in msg for msg in progress)
