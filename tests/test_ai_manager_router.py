from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from photodoctor.ai.manager import AIModelManager, ModelSpec, ModelState
from photodoctor.ai.router import AIRouter
from photodoctor.core.ai_support import attach_ai_status
from photodoctor.core.models import AnalysisResult, ImageInfo, MetricResult


def _metric(name, score=None, confidence=0.8, raw=None):
    return MetricResult(name, raw if raw is not None else {}, score, confidence, "test")


def _analysis(metrics):
    return AnalysisResult(
        image=ImageInfo(Path("x.png"), 100, 100, "RGB", "PNG"),
        metrics=metrics,
    )


def test_manager_missing_models_is_normal(tmp_path, monkeypatch):
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: False))
    manager = AIModelManager(tmp_path)
    states = manager.states()
    assert states
    native = [state for state in states if state.spec.provider == "native_numpy"]
    external = [state for state in states if state.spec.provider != "native_numpy"]
    assert len(native) == 1
    assert native[0].status == "ready"
    assert all(state.status == "missing" for state in external)
    summary = manager.summary()
    assert summary["ready_count"] == 1
    assert summary["native_ready_count"] == 1
    assert summary["missing_count"] == len(external)


def test_unpinned_existing_model_is_not_trusted(tmp_path, monkeypatch):
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32))
    (tmp_path / "m.onnx").write_bytes(b"dummy")
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: False))
    state = AIModelManager(tmp_path, [spec]).states()[0]
    assert state.status == "unverified"
    assert state.file_present
    assert not state.ready


def test_manager_blocks_hash_mismatch_even_with_runtime(tmp_path, monkeypatch):
    payload = b"model bytes"
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32), sha256="0" * 64)
    (tmp_path / "m.onnx").write_bytes(payload)
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: True))
    state = AIModelManager(tmp_path, [spec]).states()[0]
    assert state.status == "hash_mismatch"
    assert state.actual_sha256 == hashlib.sha256(payload).hexdigest()
    assert not state.ready


def test_manager_ready_when_file_and_runtime_are_available(tmp_path, monkeypatch):
    payload = b"model bytes"
    sha = hashlib.sha256(payload).hexdigest()
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32), sha256=sha)
    (tmp_path / "m.onnx").write_bytes(payload)
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: True))
    state = AIModelManager(tmp_path, [spec], contract_checker=lambda spec, path: (True, "ok")).states()[0]
    assert state.ready
    assert state.status == "ready"


def test_router_requests_blur_and_surface_refinement_when_ambiguous(tmp_path, monkeypatch):
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: False))
    manager = AIModelManager(tmp_path)
    metrics = {
        "sharpness": _metric("sharpness", 42, 0.8),
        "detail_loss_type": _metric("detail_loss_type", None, 0.55, {"classification": "mixed"}),
        "surface_defects": _metric("surface_defects", 60, 0.45, {"candidate_count": 18}),
        "faces": _metric("faces", 80, 0.9, {"face_count": 0}),
        "eyes": _metric("eyes", None, 0.0, {"eye_count": 0}),
    }
    routes = {route.task: route for route in AIRouter(manager).plan(metrics)}
    assert routes["blur_refinement"].requested
    assert routes["surface_defect_refinement"].requested
    assert not routes["blur_refinement"].eligible_to_run


def test_router_requests_blur_refinement_for_contextual_degradation_like(tmp_path, monkeypatch):
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: False))
    manager = AIModelManager(tmp_path)
    metrics = {
        "sharpness": _metric("sharpness", 48, 0.8),
        "detail_loss_type": _metric("detail_loss_type", None, 0.68, {"classification": "degradation_like"}),
        "surface_defects": _metric("surface_defects", 70, 0.8, {"candidate_count": 0}),
        "faces": _metric("faces", 80, 0.9, {"face_count": 0}),
        "eyes": _metric("eyes", None, 0.0, {"eye_count": 0}),
    }
    routes = {route.task: route for route in AIRouter(manager).plan(metrics)}
    assert routes["blur_refinement"].requested


def test_router_does_not_request_blur_for_clear_sharp_image(tmp_path, monkeypatch):
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: False))
    manager = AIModelManager(tmp_path)
    metrics = {
        "sharpness": _metric("sharpness", 92, 0.9),
        "detail_loss_type": _metric("detail_loss_type", None, 0.9, {"classification": "defocus_like"}),
        "surface_defects": _metric("surface_defects", 100, 0.8, {"candidate_count": 0}),
        "faces": _metric("faces", 90, 0.9, {"face_count": 0}),
        "eyes": _metric("eyes", None, 0.0, {"eye_count": 0}),
    }
    routes = {route.task: route for route in AIRouter(manager).plan(metrics)}
    assert not routes["blur_refinement"].requested
    assert not routes["surface_defect_refinement"].requested


def test_attach_ai_status_does_not_change_existing_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: False))
    metrics = {
        "brightness": _metric("brightness", 35, 0.85),
        "sharpness": _metric("sharpness", 45, 0.8),
        "detail_loss_type": _metric("detail_loss_type", None, 0.55, {"classification": "mixed"}),
        "surface_defects": _metric("surface_defects", 60, 0.45, {"candidate_count": 10}),
    }
    result = _analysis(metrics)
    before = {key: metric.normalized_value for key, metric in result.metrics.items()}
    attach_ai_status(result, manager=AIModelManager(tmp_path))
    after = {key: result.metrics[key].normalized_value for key in before}
    assert before == after
    raw = result.metrics["ai_status"].raw_value
    assert raw["execution_enabled"] is False
    assert raw["automatic_downloads"] is False
    assert raw["ready_count"] == 1
    assert raw["native_ready_count"] == 1


def test_manager_can_write_manifest_without_models(tmp_path):
    manager = AIModelManager(tmp_path)
    path = manager.write_manifest()
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "blur_refiner_v1" in text
    assert "native_surface_verifier_v2" in text


def test_manual_import_pins_hash_and_survives_missing_runtime(tmp_path, monkeypatch):
    from photodoctor.ai.manager import ModelImportError

    source = tmp_path / "incoming.onnx"
    source.write_bytes(b"locally supplied onnx bytes")
    model_dir = tmp_path / "models"
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32))
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: False))
    manager = AIModelManager(model_dir, [spec])
    state = manager.import_model("m", source)
    assert state.status == "runtime_missing"
    assert state.hash_ok is True
    assert state.integrity_source == "local_pin"
    assert (model_dir / "installed_models.json").is_file()
    assert (model_dir / "m.onnx").read_bytes() == source.read_bytes()


def test_manual_import_detects_later_file_change(tmp_path, monkeypatch):
    source = tmp_path / "incoming.onnx"
    source.write_bytes(b"first model")
    model_dir = tmp_path / "models"
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32))
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: True))
    manager = AIModelManager(model_dir, [spec], contract_checker=lambda spec, path: (True, "ok"))
    state = manager.import_model("m", source)
    assert state.ready
    (model_dir / "m.onnx").write_bytes(b"changed model")
    changed = manager.inspect(spec)
    assert changed.status == "hash_mismatch"
    assert changed.hash_ok is False


def test_manual_import_rejects_unknown_id_and_non_onnx(tmp_path):
    from photodoctor.ai.manager import ModelImportError

    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32))
    manager = AIModelManager(tmp_path / "models", [spec])
    txt = tmp_path / "x.txt"
    txt.write_text("x", encoding="utf-8")
    try:
        manager.import_model("unknown", txt)
    except ModelImportError:
        pass
    else:
        raise AssertionError("unknown model id must be rejected")
    try:
        manager.import_model("m", txt)
    except ModelImportError:
        pass
    else:
        raise AssertionError("non-onnx file must be rejected")


def test_forget_pin_makes_existing_file_unverified(tmp_path, monkeypatch):
    source = tmp_path / "incoming.onnx"
    source.write_bytes(b"model")
    model_dir = tmp_path / "models"
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32))
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: True))
    manager = AIModelManager(model_dir, [spec], contract_checker=lambda spec, path: (True, "ok"))
    assert manager.import_model("m", source).ready
    manager.forget_integrity_pin("m")
    state = manager.inspect(spec)
    assert state.status == "unverified"
    assert not state.ready


def test_contract_mismatch_blocks_ready_model_and_is_cached(tmp_path, monkeypatch):
    payload = b"model bytes"
    sha = hashlib.sha256(payload).hexdigest()
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32), sha256=sha)
    (tmp_path / "m.onnx").write_bytes(payload)
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: True))
    calls = []

    def checker(spec, path):
        calls.append(path)
        return False, "bad shape"

    manager = AIModelManager(tmp_path, [spec], contract_checker=checker)
    state = manager.states()[0]
    assert state.status == "contract_mismatch"
    assert "bad shape" in state.detail
    assert len(calls) == 1
    manager.invalidate()
    # Same SHA reuses persisted preflight result instead of calling checker again.
    state2 = manager.states()[0]
    assert state2.status == "contract_mismatch"
    assert len(calls) == 1


def test_successful_contract_preflight_is_persisted_by_sha(tmp_path, monkeypatch):
    payload = b"model bytes"
    sha = hashlib.sha256(payload).hexdigest()
    spec = ModelSpec("m", "blur_refinement", "m.onnx", "1", "x", (32, 32), sha256=sha)
    (tmp_path / "m.onnx").write_bytes(payload)
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: True))
    calls = []

    def checker(spec, path):
        calls.append(path)
        return True, "contract ok"

    manager = AIModelManager(tmp_path, [spec], contract_checker=checker)
    assert manager.states()[0].ready
    assert len(calls) == 1
    manager.invalidate()
    assert manager.states()[0].ready
    assert len(calls) == 1
    manifest = (tmp_path / "installed_models.json").read_text(encoding="utf-8")
    assert "contract_checked_sha256" in manifest
    assert "contract ok" in manifest


def test_router_prefers_ready_model_when_multiple_specs_share_task(tmp_path):
    first = ModelSpec("missing", "blur_refinement", "missing.onnx", "1", "x", (32, 32))
    second = ModelSpec("ready", "blur_refinement", "ready.onnx", "1", "x", (32, 32))
    missing = ModelState(first, tmp_path / "missing.onnx", "missing", False, False, None)
    ready = ModelState(second, tmp_path / "ready.onnx", "ready", True, True, True, "a" * 64, "a" * 64, "local_pin")

    class Manager:
        def states(self):
            return [missing, ready]

    metrics = {
        "sharpness": _metric("sharpness", 40, 0.8),
        "detail_loss_type": _metric("detail_loss_type", None, 0.5, {"classification": "mixed"}),
    }
    route = next(item for item in AIRouter(Manager()).plan(metrics) if item.task == "blur_refinement")
    assert route.requested
    assert route.model_id == "ready"
    assert route.eligible_to_run


def test_contract_preflight_cache_is_invalidated_when_declared_contract_changes(tmp_path, monkeypatch):
    payload = b"same model bytes"
    sha = hashlib.sha256(payload).hexdigest()
    (tmp_path / "m.onnx").write_bytes(payload)
    monkeypatch.setattr(AIModelManager, "runtime_available", staticmethod(lambda: True))
    calls = []

    def checker(spec, path):
        calls.append((spec.contract_id, spec.input_size, spec.labels))
        return True, "contract ok"

    spec_v1 = ModelSpec(
        "m", "blur_refinement", "m.onnx", "1", "x", (32, 32), sha256=sha,
        contract_id="rgb01_nchw_classification_v1", labels=("a", "b"),
    )
    manager_v1 = AIModelManager(tmp_path, [spec_v1], contract_checker=checker)
    assert manager_v1.states()[0].ready
    assert len(calls) == 1

    # Same file SHA, but a changed input contract must force a fresh preflight.
    spec_v2 = ModelSpec(
        "m", "blur_refinement", "m.onnx", "2", "x", (64, 64), sha256=sha,
        contract_id="rgb01_nchw_classification_v1", labels=("a", "b"),
    )
    manager_v2 = AIModelManager(tmp_path, [spec_v2], contract_checker=checker)
    assert manager_v2.states()[0].ready
    assert len(calls) == 2
    assert calls[-1][1] == (64, 64)
