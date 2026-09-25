from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from photodoctor.core.models import MetricResult
from photodoctor.core.validator import ValidationItem, _prefer_almaz_restoration_if_available, apply_selected_preview
from photodoctor.core import validator
from photodoctor.core import almaz_restoration


def _metric(name: str, raw: object, value: float | None = None) -> MetricResult:
    return MetricResult(name, raw, value, 0.8, "test")


def _legacy_item(key: str) -> ValidationItem:
    return ValidationItem(
        key, True, False,
        {"noise": "mild_nlm", "sharpness": "edge_aware_unsharp", "jpeg_artifacts": "mild_deblock"}[key],
        40.0, 50.0, 0.6, {}, "legacy", preview_available=True, auto_eligible=False,
        technical_passed=True, adjustable=True, default_strength=0.4,
    )


def _ready(task: str):
    return SimpleNamespace(
        ready=True, task=task, model_id=f"fake_{task}", provider="CPUExecutionProvider",
        provider_label="ONNX Runtime CPU", detail="ready",
    )


def test_noise_action_upgrades_to_almaz_backend_when_verified_model_is_ready(monkeypatch):
    monkeypatch.setattr(validator, "inspect_installed_restoration_model", lambda task: _ready(task))
    metrics = {"faces": _metric("faces", {"face_count": 1, "faces": [{"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}]})}
    item = _prefer_almaz_restoration_if_available("noise", _legacy_item("noise"), metrics)
    assert item.candidate == "almaz_ai_denoise_v1"
    assert item.accepted is False
    assert item.auto_eligible is False
    assert item.parameters["task"] == "denoise"
    assert item.parameters["face_boxes"][0]["x"] == 0.2


def test_sharpness_upgrades_only_for_real_blur_classification(monkeypatch):
    monkeypatch.setattr(validator, "inspect_installed_restoration_model", lambda task: _ready(task))
    base = {"faces": _metric("faces", {"face_count": 0, "faces": []})}
    local = dict(base)
    local["detail_loss_type"] = _metric("detail_loss_type", {"classification": "local_subject_softness"})
    assert _prefer_almaz_restoration_if_available("sharpness", _legacy_item("sharpness"), local).candidate == "edge_aware_unsharp"

    motion = dict(base)
    motion["detail_loss_type"] = _metric("detail_loss_type", {"classification": "camera_shake_like"})
    upgraded = _prefer_almaz_restoration_if_available("sharpness", _legacy_item("sharpness"), motion)
    assert upgraded.candidate == "almaz_ai_deblur_v1"
    assert upgraded.parameters["task"] == "deblur"


def test_jpeg_action_uses_heavier_recovery_only_when_model_is_ready(monkeypatch):
    monkeypatch.setattr(validator, "inspect_installed_restoration_model", lambda task: _ready(task))
    metrics = {"faces": _metric("faces", {"face_count": 0, "faces": []})}
    item = _prefer_almaz_restoration_if_available("jpeg_artifacts", _legacy_item("jpeg_artifacts"), metrics)
    assert item.candidate == "almaz_ai_jpeg_recovery_v1"
    assert item.parameters["task"] == "jpeg_recovery"


def test_selected_preview_executes_almaz_denoise_candidate(monkeypatch):
    rgb = np.full((24, 32, 3), 100, dtype=np.uint8)
    calls = []

    def fake_apply(arr, **kwargs):
        calls.append(kwargs)
        return np.clip(arr.astype(np.int16) + 7, 0, 255).astype(np.uint8)

    monkeypatch.setattr(validator, "apply_almaz_restoration", fake_apply)
    item = ValidationItem(
        "noise", True, False, "almaz_ai_denoise_v1", None, None, 0.7, {}, "almaz",
        preview_available=True, auto_eligible=False, technical_passed=True, adjustable=True,
        default_strength=0.6, parameters={"task": "denoise", "face_protection": 0.8, "face_boxes": []},
    ).to_dict()
    out = apply_selected_preview(rgb, [item], {"noise"})
    assert np.array_equal(out, rgb + 7)
    assert calls and calls[0]["task"] == "denoise"


def test_almaz_x1_identity_guard_reduces_model_delta_inside_normalized_face(monkeypatch):
    class FakeBackend:
        def restore(self, rgb):
            return np.clip(rgb.astype(np.int16) + 60, 0, 255).astype(np.uint8)

    monkeypatch.setattr(
        almaz_restoration,
        "load_installed_restoration_backend",
        lambda task: (FakeBackend(), SimpleNamespace(ready=True, detail="ready")),
    )
    rgb = np.full((100, 120, 3), 80, dtype=np.uint8)
    out = almaz_restoration.apply_almaz_restoration(
        rgb, task="deblur", strength=1.0,
        face_boxes=[{"x": 0.25, "y": 0.25, "w": 0.30, "h": 0.35}],
        face_protection=1.0,
    )
    face_delta = float(np.mean(out[30:55, 35:60].astype(np.float32) - rgb[30:55, 35:60]))
    background_delta = float(np.mean(out[0:15, 0:15].astype(np.float32) - rgb[0:15, 0:15]))
    assert face_delta < background_delta * 0.75
    assert background_delta >= 55.0


def test_almaz_x1_missing_model_fails_closed(monkeypatch):
    monkeypatch.setattr(
        almaz_restoration,
        "load_installed_restoration_backend",
        lambda task: (None, SimpleNamespace(ready=False, detail="model missing")),
    )
    with pytest.raises(almaz_restoration.AlmazRestorationUnavailable, match="model missing"):
        almaz_restoration.apply_almaz_restoration(np.zeros((8, 8, 3), dtype=np.uint8), task="denoise", strength=0.5)


def test_almaz_strength_tracks_measured_degradation(monkeypatch):
    monkeypatch.setattr(validator, "inspect_installed_restoration_model", lambda task: _ready(task))
    mild = {
        "faces": _metric("faces", {"face_count": 0, "faces": []}),
        "noise": _metric("noise", {}, 67.0),
    }
    severe = {
        "faces": _metric("faces", {"face_count": 0, "faces": []}),
        "noise": _metric("noise", {}, 38.0),
    }
    mild_item = _prefer_almaz_restoration_if_available("noise", _legacy_item("noise"), mild)
    severe_item = _prefer_almaz_restoration_if_available("noise", _legacy_item("noise"), severe)
    assert mild_item.candidate == "almaz_ai_denoise_v1"
    assert severe_item.candidate == "almaz_ai_denoise_v1"
    assert severe_item.default_strength > mild_item.default_strength
    assert 0.42 <= mild_item.default_strength <= 0.74
    assert 0.42 <= severe_item.default_strength <= 0.74


def test_almaz_faces_reduce_default_restoration_strength(monkeypatch):
    monkeypatch.setattr(validator, "inspect_installed_restoration_model", lambda task: _ready(task))
    base = {"noise": _metric("noise", {}, 45.0)}
    without_faces = dict(base)
    without_faces["faces"] = _metric("faces", {"face_count": 0, "faces": []})
    with_faces = dict(base)
    with_faces["faces"] = _metric("faces", {"face_count": 1, "faces": [{"x": 0.2, "y": 0.2, "w": 0.3, "h": 0.3}]})
    a = _prefer_almaz_restoration_if_available("noise", _legacy_item("noise"), without_faces)
    b = _prefer_almaz_restoration_if_available("noise", _legacy_item("noise"), with_faces)
    assert b.default_strength < a.default_strength
    assert b.parameters["face_protection"] >= 0.80


def test_almaz_x1_modules_have_independent_action_keys(monkeypatch):
    monkeypatch.setattr(validator, "inspect_installed_restoration_model", lambda task: _ready(task))
    metrics = {
        "faces": _metric("faces", {"face_count": 0, "faces": []}),
        "noise": _metric("noise", {}, 55.0),
        "sharpness": _metric("sharpness", {}, 58.0),
        "local_sharpness": _metric("local_sharpness", {}, 58.0),
        "jpeg_artifacts": _metric("jpeg_artifacts", {}, 60.0),
    }
    denoise = validator._standalone_almaz_restoration_item("denoise", metrics)
    deblur = validator._standalone_almaz_restoration_item("deblur", metrics)
    jpeg = validator._standalone_almaz_restoration_item("jpeg_recovery", metrics)
    assert denoise is not None and denoise.action_key == "almaz_denoise"
    assert deblur is not None and deblur.action_key == "almaz_deblur"
    assert jpeg is not None and jpeg.action_key == "almaz_jpeg_recovery"
    assert {denoise.candidate, deblur.candidate, jpeg.candidate} == {
        "almaz_ai_denoise_v1", "almaz_ai_deblur_v1", "almaz_ai_jpeg_recovery_v1"
    }
    assert not denoise.accepted and not deblur.accepted and not jpeg.accepted


@pytest.mark.parametrize(
    ("action_key", "candidate", "task"),
    [
        ("almaz_denoise", "almaz_ai_denoise_v1", "denoise"),
        ("almaz_deblur", "almaz_ai_deblur_v1", "deblur"),
        ("almaz_jpeg_recovery", "almaz_ai_jpeg_recovery_v1", "jpeg_recovery"),
    ],
)
def test_almaz_x1_actions_can_be_previewed_one_at_a_time(monkeypatch, action_key, candidate, task):
    rgb = np.full((18, 24, 3), 100, dtype=np.uint8)
    calls: list[str] = []
    def fake_apply(arr, **kwargs):
        calls.append(str(kwargs.get("task")))
        return np.clip(arr.astype(np.int16) + 4, 0, 255).astype(np.uint8)
    monkeypatch.setattr(validator, "apply_almaz_restoration", fake_apply)
    monkeypatch.setattr(
        validator, "validate_almaz_transition",
        lambda *a, **k: SimpleNamespace(accepted=True, message="PASS"),
    )
    item = ValidationItem(
        action_key, True, False, candidate, None, None, 0.7, {}, "manual",
        preview_available=True, auto_eligible=False, technical_passed=True, adjustable=True,
        default_strength=0.5, parameters={"task": task, "face_protection": 0.8, "face_boxes": []},
    ).to_dict()
    out = apply_selected_preview(rgb, [item], {action_key})
    assert calls == [task]
    assert np.array_equal(out, rgb + 4)


def test_deblur_apply_suppresses_model_green_patch_on_colour_photo(monkeypatch):
    class FakeBackend:
        def restore(self, rgb):
            out = rgb.copy()
            h, w = out.shape[:2]
            out[h//4:h//2, w//3:w//2] = [35, 205, 45]
            return out

    monkeypatch.setattr(
        almaz_restoration, "load_installed_restoration_backend",
        lambda task: (FakeBackend(), SimpleNamespace(ready=True, detail="ready", provider="CPU")),
    )
    y, x = np.indices((96, 128))
    rgb = np.dstack([
        50 + x * 120 // 127,
        70 + y * 110 // 95,
        190 - x * 80 // 127,
    ]).astype(np.uint8)
    out = almaz_restoration.apply_almaz_restoration(rgb, task="deblur", strength=1.0)
    lab0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)
    chroma = np.hypot(lab1[..., 1]-lab0[..., 1], lab1[..., 2]-lab0[..., 2])
    assert float(np.percentile(chroma, 99.0)) <= 2.5
