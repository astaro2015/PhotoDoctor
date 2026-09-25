from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from photodoctor.core.analyzer import analyze_classical
from photodoctor.core.loader import load_image
from photodoctor.core.performance import (
    PERFORMANCE_PROFILE_VERSION,
    PerformanceProfile,
    apply_performance_profile,
    get_pipeline_profile,
    hardware_signature,
)


def _assert_semantically_equal(a, b, *, atol: float = 1e-6, path: str = "root") -> None:
    assert type(a) is type(b), f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: different keys"
        for key in a:
            _assert_semantically_equal(a[key], b[key], atol=atol, path=f"{path}.{key}")
        return
    if isinstance(a, list):
        assert len(a) == len(b), f"{path}: different lengths"
        for i, (left, right) in enumerate(zip(a, b)):
            _assert_semantically_equal(left, right, atol=atol, path=f"{path}[{i}]")
        return
    if isinstance(a, float):
        if math.isnan(a) and math.isnan(b):
            return
        assert abs(a - b) <= atol, f"{path}: {a} != {b}"
        return
    assert a == b, f"{path}: {a!r} != {b!r}"


def test_parallel_analysis_preserves_semantic_result(tmp_path: Path):
    # Structured deterministic image exercises spatial maps and artifact analysis
    # without making this regression test depend on an external photo.
    h, w = 360, 480
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.dstack((
        (0.61 * xx + 0.19 * yy + 24.0 * np.sin(xx / 13.0)) % 256,
        (0.17 * xx + 0.53 * yy + 21.0 * np.cos(yy / 17.0)) % 256,
        (0.29 * xx + 0.31 * yy + 18.0 * np.sin((xx + yy) / 19.0)) % 256,
    )).astype(np.uint8)
    path = tmp_path / "parallel-equivalence.png"
    Image.fromarray(rgb, "RGB").save(path)
    loaded = load_image(path)

    old_serial, old_parallel, old_workers = get_pipeline_profile()
    sig = hardware_signature()
    try:
        serial = PerformanceProfile(
            PERFORMANCE_PROFILE_VERSION, sig, 1, 1, 1, 0.0
        )
        parallel = PerformanceProfile(
            PERFORMANCE_PROFILE_VERSION, sig, 1, 1, 2, 0.0
        )
        apply_performance_profile(serial)
        expected = analyze_classical(loaded, precision="fast").to_dict()
        apply_performance_profile(parallel)
        actual = analyze_classical(loaded, precision="fast").to_dict()
        _assert_semantically_equal(expected, actual)
    finally:
        apply_performance_profile(
            PerformanceProfile(
                PERFORMANCE_PROFILE_VERSION,
                sig,
                old_serial,
                old_parallel,
                old_workers,
                0.0,
            )
        )
