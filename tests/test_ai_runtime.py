from __future__ import annotations

from pathlib import Path

import numpy as np

from photodoctor.ai.executor import AIExecutionError, AIInferenceResult
from photodoctor.ai.manager import ModelSpec, ModelState
from photodoctor.core.ai_runtime import _face_crop, _surface_crop, execute_ai_plan
from photodoctor.core.models import MetricResult


class FakeManager:
    def __init__(self, states):
        self._states = list(states)

    def states(self):
        return list(self._states)


class FakeExecutor:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def run(self, state, rgb):
        self.calls.append((state.spec.task, rgb.copy()))
        if self.fail:
            raise AIExecutionError("synthetic failure")
        return AIInferenceResult(
            model_id=state.spec.model_id,
            task=state.spec.task,
            contract_id=state.spec.contract_id,
            confidence=0.88,
            data={"score_0_1": 0.61, "score_0_100": 61.0},
        )


def _metric(name, score=None, confidence=0.8, raw=None):
    return MetricResult(name, raw if raw is not None else {}, score, confidence, "test")


def _ready(spec):
    return ModelState(spec, Path("x.onnx"), "ready", True, True, True, "a" * 64, "a" * 64, "local_pin")


def test_runtime_runs_ready_requested_face_model_on_face_crop_only():
    spec = ModelSpec(
        "face", "face_quality", "face.onnx", "1", "x", (224, 224),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    manager = FakeManager([_ready(spec)])
    executor = FakeExecutor()
    rgb = np.zeros((200, 300, 3), np.uint8)
    metrics = {
        "faces": _metric("faces", 50, 0.7, {"face_count": 1, "faces": [{"x": 0.2, "y": 0.1, "w": 0.3, "h": 0.4}]}),
        "eyes": _metric("eyes", None, 0.5, {"eye_count": 0}),
    }
    metric = execute_ai_plan(rgb, metrics, manager=manager, executor=executor)
    raw = metric.raw_value
    assert raw["successful_inferences"] == 1
    assert len(executor.calls) == 1
    task, crop = executor.calls[0]
    assert task == "face_quality"
    assert crop.shape[0] < rgb.shape[0]
    assert crop.shape[1] < rgb.shape[1]


def test_runtime_isolates_ai_error_instead_of_raising():
    spec = ModelSpec(
        "face", "face_quality", "face.onnx", "1", "x", (224, 224),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    manager = FakeManager([_ready(spec)])
    executor = FakeExecutor(fail=True)
    metrics = {
        "faces": _metric("faces", 50, 0.7, {"face_count": 1, "faces": [{"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.5}]}),
        "eyes": _metric("eyes", None, 0.5, {"eye_count": 0}),
    }
    metric = execute_ai_plan(np.zeros((100, 100, 3), np.uint8), metrics, manager=manager, executor=executor)
    raw = metric.raw_value
    assert raw["successful_inferences"] == 0
    assert raw["attempted_inferences"] == 1
    assert raw["results"][0]["status"] == "partial_error"
    assert "synthetic failure" in raw["results"][0]["regions"][0]["error"]


def test_runtime_respects_model_run_limit():
    blur = ModelSpec(
        "blur", "blur_refinement", "b.onnx", "1", "x", (32, 32),
        contract_id="rgb01_nchw_classification_v1", output_kind="classification", labels=("a", "b"),
    )
    surface = ModelSpec(
        "surface", "surface_defect_refinement", "s.onnx", "1", "x", (32, 32),
        contract_id="rgb01_nchw_segmentation_v1", output_kind="segmentation",
    )
    face = ModelSpec(
        "face", "face_quality", "f.onnx", "1", "x", (32, 32),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    manager = FakeManager([_ready(blur), _ready(surface), _ready(face)])
    executor = FakeExecutor()
    metrics = {
        "sharpness": _metric("sharpness", 40, 0.8),
        "detail_loss_type": _metric("detail_loss_type", None, 0.5, {"classification": "mixed"}),
        "surface_defects": _metric(
            "surface_defects", 50, 0.4,
            {
                "candidate_count": 30,
                "ai_boxes_norm": [{"x": 0.15, "y": 0.15, "w": 0.2, "h": 0.2, "strength": 0.8, "polarity": "bright"}],
            },
        ),
        "faces": _metric("faces", 50, 0.7, {"face_count": 1, "faces": [{"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4}]}),
        "eyes": _metric("eyes", None, 0.5, {"eye_count": 0}),
    }
    raw = execute_ai_plan(
        np.zeros((100, 100, 3), np.uint8), metrics, manager=manager, executor=executor, max_model_runs=1
    ).raw_value
    assert raw["requested_count"] >= 3
    assert raw["selected_model_count"] == 1
    assert len(executor.calls) == 1
    assert sum(item["status"] == "not_run" for item in raw["results"]) >= 2


def test_runtime_with_no_ready_models_runs_nothing():
    spec = ModelSpec("face", "face_quality", "f.onnx", "1", "x", (32, 32))
    state = ModelState(spec, Path("x"), "missing", False, False, None)
    manager = FakeManager([state])
    executor = FakeExecutor()
    metrics = {
        "faces": _metric("faces", 40, 0.6, {"face_count": 1, "faces": [{"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}]}),
        "eyes": _metric("eyes", None, 0.5, {"eye_count": 0}),
    }
    raw = execute_ai_plan(np.zeros((100, 100, 3), np.uint8), metrics, manager=manager, executor=executor).raw_value
    assert raw["requested_count"] == 1
    assert raw["eligible_count"] == 0
    assert raw["attempted_inferences"] == 0
    assert executor.calls == []


def test_ai_crops_reject_nan_and_inf_without_crashing_optional_ai():
    rgb = np.zeros((100, 120, 3), np.uint8)
    assert _face_crop(rgb, {"x": float("inf"), "y": 0.1, "w": 0.2, "h": 0.2}) is None
    assert _surface_crop(rgb, {"x": float("nan"), "y": 0.1, "w": 0.2, "h": 0.2}) is None
    assert _surface_crop(rgb, {"x": 0.1, "y": 0.1, "w": float("inf"), "h": 0.2}) is None


def test_native_ai_incomplete_batch_is_reported_as_partial_error(monkeypatch):
    from photodoctor.ai.catalog import BUILTIN_MODEL_SPECS
    spec = next(item for item in BUILTIN_MODEL_SPECS if item.task == "surface_defect_refinement")
    state = ModelState(spec, Path("<builtin>"), "ready", True, True, True, None, None, "builtin")
    manager = FakeManager([state])
    boxes = [
        {"x": 0.1, "y": 0.1, "w": 0.15, "h": 0.15, "strength": 0.8, "polarity": "bright"},
        {"x": 0.5, "y": 0.5, "w": 0.15, "h": 0.15, "strength": 0.7, "polarity": "bright"},
    ]
    metrics = {
        "surface_defects": _metric(
            "surface_defects", 50, 0.4,
            {"candidate_count": 2, "ai_boxes_norm": boxes},
        )
    }
    monkeypatch.setattr(
        "photodoctor.core.ai_runtime.predict_native_surface_candidates_v2",
        lambda rgb, boxes, batch_size: [{"label": "defect", "confidence": 0.9, "defect_probability": 0.9}],
    )
    raw = execute_ai_plan(np.zeros((100, 100, 3), np.uint8), metrics, manager=manager).raw_value
    task = raw["results"][0]
    assert task["status"] == "partial_error"
    assert raw["attempted_inferences"] == 2
    assert raw["successful_inferences"] == 1
    assert [r["status"] for r in task["regions"]] == ["ok", "error"]
    assert "неполный пакет" in task["regions"][1]["error"]
