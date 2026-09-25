from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from photodoctor.ai.sr_backend import OnnxX2SuperResolutionBackend, SR_CONTRACT_ID


class _FakeSession:
    def __init__(self, *_args, **_kwargs):
        self._input = SimpleNamespace(name="input")
        self._output = SimpleNamespace(name="output")

    def get_inputs(self):
        return [self._input]

    def get_outputs(self):
        return [self._output]

    def run(self, _outputs, feeds):
        tensor = feeds["input"]
        hwc = np.transpose(tensor[0], (1, 2, 0))
        up = cv2.resize(hwc, (hwc.shape[1] * 2, hwc.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
        return [np.transpose(up, (2, 0, 1))[None, ...].astype(np.float32)]


def test_sr_contract_id_is_stable():
    assert SR_CONTRACT_ID == "rgb01_nchw_static256_x2_v1"


def test_onnx_x2_backend_runs_small_frame_with_fake_session():
    rgb = np.zeros((30, 40, 3), dtype=np.uint8)
    rgb[4:20, 7:22] = [20, 120, 230]
    backend = OnnxX2SuperResolutionBackend("fake.onnx", session_factory=_FakeSession)
    out = backend.upscale_x2(rgb, tile_size=64)
    assert out.shape == (60, 80, 3)
    assert np.array_equal(out[8:40:2, 14:44:2], rgb[4:20, 7:22])


def test_onnx_x2_tiling_has_no_uncovered_pixels_or_shape_drift():
    y, x = np.indices((137, 181))
    rgb = np.dstack([
        (x % 256).astype(np.uint8),
        (y % 256).astype(np.uint8),
        ((x + y) % 256).astype(np.uint8),
    ])
    backend = OnnxX2SuperResolutionBackend("fake.onnx", session_factory=_FakeSession)
    tiled = backend.upscale_x2(rgb, tile_size=80, overlap=12)
    reference = cv2.resize(rgb, (362, 274), interpolation=cv2.INTER_NEAREST)
    assert tiled.shape == reference.shape
    assert np.max(np.abs(tiled.astype(np.int16) - reference.astype(np.int16))) <= 1


def test_onnx_x2_backend_reports_tile_progress():
    rgb = np.zeros((137, 181, 3), dtype=np.uint8)
    backend = OnnxX2SuperResolutionBackend("fake.onnx", session_factory=_FakeSession)
    events: list[tuple[int, int]] = []
    out = backend.upscale_x2(rgb, tile_size=80, overlap=12, progress=lambda done, total: events.append((done, total)))
    assert out.shape == (274, 362, 3)
    assert events
    assert events[-1][0] == events[-1][1]
    assert events[0][0] == 1
    assert all(total == events[-1][1] for _, total in events)
