from __future__ import annotations

import cv2
import numpy as np

from photodoctor.core.blur_type import analyze_blur_type


def _texture(size: int = 720) -> np.ndarray:
    rng = np.random.default_rng(1234)
    base = rng.normal(127, 42, (size, size)).clip(0, 255).astype(np.uint8)
    # Add many orientations so scene geometry does not dominate.
    for y in range(40, size, 80):
        cv2.line(base, (20, y), (size - 20, y + 23), 235, 3)
    for x in range(40, size, 90):
        cv2.line(base, (x, 15), (x + 30, size - 15), 25, 3)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)


def _motion_blur(rgb: np.ndarray, length: int = 31) -> np.ndarray:
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0 / length
    return cv2.filter2D(rgb, -1, k)


def test_directional_motion_blur_is_detected():
    result = analyze_blur_type(_motion_blur(_texture()))
    assert result.classification == "motion_like"
    assert result.confidence >= 0.55
    assert result.direction_deg is not None


def test_gaussian_blur_is_not_called_motion():
    blurred = cv2.GaussianBlur(_texture(), (0, 0), 4.0)
    result = analyze_blur_type(blurred)
    assert result.classification in {"defocus_like", "mixed_or_degraded"}
    assert result.classification != "motion_like"


def test_tiny_image_is_unknown():
    rgb = np.zeros((40, 40, 3), dtype=np.uint8)
    result = analyze_blur_type(rgb)
    assert result.classification == "unknown"
    assert result.confidence <= 0.2
