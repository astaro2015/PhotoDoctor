import cv2
import numpy as np

from photodoctor.core.rendering_artifacts import analyze_edge_artifacts, analyze_posterization


def photo_like_scene(size: int = 512) -> np.ndarray:
    y, x = np.indices((size, size))
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = np.clip(35 + x * 175 / size + 24 * np.sin(y / 19), 0, 255)
    img[..., 1] = np.clip(55 + y * 145 / size + 18 * np.sin(x / 27), 0, 255)
    img[..., 2] = np.clip(95 + (x + y) * 85 / (2 * size), 0, 255)
    cv2.circle(img, (205, 220), 86, (205, 125, 85), -1)
    cv2.putText(img, "PHOTO", (58, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (238, 238, 238), 4, cv2.LINE_AA)
    return img


def unsharp(rgb: np.ndarray, amount: float) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(f, (0, 0), 1.2)
    return np.clip((f + amount * (f - blur)) * 255.0, 0, 255).astype(np.uint8)


def test_oversharpened_scene_scores_worse_than_original():
    original = analyze_edge_artifacts(photo_like_scene())
    sharpened = analyze_edge_artifacts(unsharp(photo_like_scene(), 1.6))
    assert original.score > sharpened.score + 20
    assert sharpened.halo_likelihood > original.halo_likelihood
    assert sharpened.classification in {"mild_halo_candidate", "halo_candidate", "strong_halo_candidate"}


def test_smooth_blurred_edge_is_not_called_ringing():
    line = np.zeros((384, 384), np.float32) + 0.22
    line[:, 192:] = 0.76
    line = cv2.GaussianBlur(line, (0, 0), 2.0)
    rgb = np.repeat((line * 255).astype(np.uint8)[..., None], 3, axis=2)
    result = analyze_edge_artifacts(rgb)
    assert result.classification == "none"
    assert result.ringing_likelihood < 0.15


def gradient(levels: int, width: int = 512, height: int = 256) -> np.ndarray:
    values = np.tile(np.linspace(0, 255, width, dtype=np.float32), (height, 1))
    if levels < 256:
        step = 255.0 / (levels - 1)
        values = np.round(values / step) * step
    gray = np.clip(values, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_posterized_gradient_scores_worse_than_smooth_gradient():
    smooth = analyze_posterization(gradient(256))
    quantized = analyze_posterization(gradient(16))
    assert smooth.score > 90
    assert quantized.score < 55
    assert quantized.occupied_luma_levels <= 20
    assert quantized.classification in {"moderate", "strong"}


def test_uniform_image_is_unknown_not_posterized():
    rgb = np.full((256, 256, 3), 128, np.uint8)
    result = analyze_posterization(rgb)
    assert result.classification == "unknown"
    assert result.confidence < 0.4


def test_full_analysis_contains_rendering_artifact_metrics(tmp_path):
    from PIL import Image
    from photodoctor.core.service import analyze_file

    path = tmp_path / "artifact_input.png"
    Image.fromarray(photo_like_scene()).save(path)
    metrics = analyze_file(path).metrics
    assert "edge_artifacts" in metrics
    assert "posterization" in metrics
    assert metrics["edge_artifacts"].raw_value["method"] == "edge_halo_ringing_v1"
    assert metrics["posterization"].raw_value["method"] == "posterization_banding_v1"
