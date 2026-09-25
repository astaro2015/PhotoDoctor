from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from photodoctor.ai.executor import AIExecutionError, ONNXExecutor, prepare_rgb01_nchw
from photodoctor.ai.manager import ModelSpec, ModelState


class _Input:
    name = "image"


class _Session:
    def __init__(self, output):
        self.output = output
        self.last_feed = None

    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, feed):
        self.last_feed = feed
        return [self.output]


def _ready_state(spec: ModelSpec) -> ModelState:
    return ModelState(spec, Path("x.onnx"), "ready", True, True, True, "a" * 64, "a" * 64, "local_pin")


def test_preprocess_is_rgb_float32_nchw_and_letterboxed():
    rgb = np.zeros((40, 80, 3), np.uint8)
    rgb[:, :, 0] = 255
    tensor = prepare_rgb01_nchw(rgb, (64, 64))
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    # Red content remains red; grey letterbox is neutral.
    assert float(tensor[0, 0, 32, 32]) > 0.99
    assert float(tensor[0, 1, 32, 32]) < 0.01
    assert np.allclose(tensor[0, :, 2, 2], 0.5)


def test_classification_contract_accepts_logits_and_softmaxes():
    spec = ModelSpec(
        "m", "blur_refinement", "m.onnx", "1", "x", (32, 32),
        contract_id="rgb01_nchw_classification_v1", output_kind="classification",
        labels=("clean", "defocus", "motion"),
    )
    session = _Session(np.array([[0.1, 2.0, -0.5]], np.float32))
    result = ONNXExecutor(lambda path: session).run(_ready_state(spec), np.zeros((20, 30, 3), np.uint8))
    assert result.data["label"] == "defocus"
    assert result.confidence > 0.7
    assert set(result.data["probabilities"]) == {"clean", "defocus", "motion"}
    assert session.last_feed["image"].shape == (1, 3, 32, 32)


def test_classification_contract_rejects_wrong_vector_size():
    spec = ModelSpec(
        "m", "blur_refinement", "m.onnx", "1", "x", (32, 32),
        contract_id="rgb01_nchw_classification_v1", output_kind="classification",
        labels=("a", "b", "c"),
    )
    with pytest.raises(AIExecutionError):
        ONNXExecutor(lambda path: _Session(np.array([[0.2, 0.8]], np.float32))).run(
            _ready_state(spec), np.zeros((20, 20, 3), np.uint8)
        )


def test_scalar_contract_returns_0_100_score():
    spec = ModelSpec(
        "m", "iqa_refinement", "m.onnx", "1", "x", (16, 16),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    result = ONNXExecutor(lambda path: _Session(np.array([[0.73]], np.float32))).run(
        _ready_state(spec), np.zeros((16, 16, 3), np.uint8)
    )
    assert result.data["score_0_100"] == pytest.approx(73.0, abs=1e-4)
    assert result.confidence == 0.0
    assert result.data["confidence_available"] is False


def test_scalar_contract_rejects_out_of_range_value():
    spec = ModelSpec(
        "m", "iqa_refinement", "m.onnx", "1", "x", (16, 16),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    with pytest.raises(AIExecutionError):
        ONNXExecutor(lambda path: _Session(np.array([1.2], np.float32))).run(
            _ready_state(spec), np.zeros((16, 16, 3), np.uint8)
        )


def test_segmentation_contract_summarizes_mask():
    spec = ModelSpec(
        "m", "surface_defect_refinement", "m.onnx", "1", "x", (8, 8),
        contract_id="rgb01_nchw_segmentation_v1", output_kind="segmentation",
    )
    mask = np.zeros((1, 1, 8, 8), np.float32)
    mask[:, :, :4] = 0.9
    result = ONNXExecutor(lambda path: _Session(mask)).run(
        _ready_state(spec), np.zeros((8, 8, 3), np.uint8)
    )
    assert result.data["coverage_pct"] == pytest.approx(50.0)
    assert result.data["mask"].shape == (8, 8)


def test_executor_refuses_non_ready_model_before_session_creation():
    spec = ModelSpec("m", "x", "m.onnx", "1", "x", (8, 8))
    state = ModelState(spec, Path("x"), "unverified", True, True, None)
    called = False

    def factory(path):
        nonlocal called
        called = True
        return _Session(np.array([0.5], np.float32))

    with pytest.raises(AIExecutionError):
        ONNXExecutor(factory).run(state, np.zeros((8, 8, 3), np.uint8))
    assert called is False


def test_executor_rejects_nan_output():
    spec = ModelSpec(
        "m", "iqa_refinement", "m.onnx", "1", "x", (8, 8),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    with pytest.raises(AIExecutionError):
        ONNXExecutor(lambda path: _Session(np.array([np.nan], np.float32))).run(
            _ready_state(spec), np.zeros((8, 8, 3), np.uint8)
        )


class _Node:
    def __init__(self, shape, type_="tensor(float)", name="x"):
        self.shape = shape
        self.type = type_
        self.name = name


class _MetaSession:
    def __init__(self, input_shape, output_shape, input_type="tensor(float)", output_type="tensor(float)"):
        self.inp = _Node(input_shape, input_type, "image")
        self.out = _Node(output_shape, output_type, "output")

    def get_inputs(self):
        return [self.inp]

    def get_outputs(self):
        return [self.out]


def test_preflight_accepts_classification_contract():
    from photodoctor.ai.executor import preflight_onnx_contract
    spec = ModelSpec(
        "m", "blur_refinement", "m.onnx", "1", "x", (256, 256),
        contract_id="rgb01_nchw_classification_v1", output_kind="classification",
        labels=("a", "b", "c"),
    )
    ok, detail = preflight_onnx_contract(
        "x.onnx", spec, session_factory=lambda path: _MetaSession([1, 3, 256, 256], [1, 3])
    )
    assert ok
    assert "Контракт совместим" in detail


def test_preflight_accepts_dynamic_input_dims():
    from photodoctor.ai.executor import preflight_onnx_contract
    spec = ModelSpec(
        "m", "iqa_refinement", "m.onnx", "1", "x", (224, 224),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    ok, _ = preflight_onnx_contract(
        "x.onnx", spec, session_factory=lambda path: _MetaSession(["N", 3, "H", "W"], [1, 1])
    )
    assert ok


def test_preflight_rejects_wrong_input_channels():
    from photodoctor.ai.executor import preflight_onnx_contract
    spec = ModelSpec(
        "m", "iqa_refinement", "m.onnx", "1", "x", (224, 224),
        contract_id="rgb01_nchw_scalar_v1", output_kind="scalar_0_1",
    )
    ok, detail = preflight_onnx_contract(
        "x.onnx", spec, session_factory=lambda path: _MetaSession([1, 1, 224, 224], [1, 1])
    )
    assert not ok
    assert "не соответствует" in detail


def test_preflight_rejects_wrong_class_count():
    from photodoctor.ai.executor import preflight_onnx_contract
    spec = ModelSpec(
        "m", "blur_refinement", "m.onnx", "1", "x", (64, 64),
        contract_id="rgb01_nchw_classification_v1", output_kind="classification",
        labels=("a", "b", "c"),
    )
    ok, detail = preflight_onnx_contract(
        "x.onnx", spec, session_factory=lambda path: _MetaSession([1, 3, 64, 64], [1, 2])
    )
    assert not ok
    assert "меток=3" in detail


def test_preflight_rejects_bad_segmentation_shape():
    from photodoctor.ai.executor import preflight_onnx_contract
    spec = ModelSpec(
        "m", "surface_defect_refinement", "m.onnx", "1", "x", (64, 64),
        contract_id="rgb01_nchw_segmentation_v1", output_kind="segmentation",
    )
    ok, detail = preflight_onnx_contract(
        "x.onnx", spec, session_factory=lambda path: _MetaSession([1, 3, 64, 64], [1, 64, 64])
    )
    assert not ok
    assert "Выход сегментации" in detail
