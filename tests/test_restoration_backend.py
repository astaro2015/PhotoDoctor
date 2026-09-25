from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from photodoctor.ai.restoration_backend import OnnxX1RestorationBackend, RESTORATION_CONTRACT_ID


class _IdentitySession:
    def __init__(self, *_args, **_kwargs):
        self._input = SimpleNamespace(name="input")
        self._output = SimpleNamespace(name="output")

    def get_inputs(self):
        return [self._input]

    def get_outputs(self):
        return [self._output]

    def run(self, _outputs, feeds):
        return [feeds["input"].copy()]


def test_restoration_contract_id_is_stable():
    assert RESTORATION_CONTRACT_ID == "rgb01_nchw_static256_x1_v1"


def test_x1_backend_runs_small_frame_with_fake_session():
    rgb = np.zeros((35, 43, 3), dtype=np.uint8)
    rgb[4:24, 9:31] = [20, 121, 232]
    backend = OnnxX1RestorationBackend("fake.onnx", session_factory=_IdentitySession)
    out = backend.restore(rgb, tile_size=64)
    assert out.shape == rgb.shape
    assert np.max(np.abs(out.astype(np.int16) - rgb.astype(np.int16))) <= 1


def test_x1_backend_tiling_has_no_uncovered_pixels_or_shape_drift():
    y, x = np.indices((137, 181))
    rgb = np.dstack([
        (x % 256).astype(np.uint8),
        (y % 256).astype(np.uint8),
        ((x + y) % 256).astype(np.uint8),
    ])
    backend = OnnxX1RestorationBackend("fake.onnx", session_factory=_IdentitySession)
    tiled = backend.restore(rgb, tile_size=80, overlap=12)
    assert tiled.shape == rgb.shape
    assert np.max(np.abs(tiled.astype(np.int16) - rgb.astype(np.int16))) <= 1


def test_x1_backend_reports_tile_progress():
    rgb = np.zeros((137, 181, 3), dtype=np.uint8)
    backend = OnnxX1RestorationBackend("fake.onnx", session_factory=_IdentitySession)
    events: list[tuple[int, int]] = []
    out = backend.restore(rgb, tile_size=80, overlap=12, progress=lambda done, total: events.append((done, total)))
    assert out.shape == rgb.shape
    assert events
    assert events[-1][0] == events[-1][1]
    assert events[0][0] == 1
    assert all(total == events[-1][1] for _, total in events)
