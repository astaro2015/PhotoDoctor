from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from photodoctor.core.jpeg_artifacts import analyze_jpeg_artifacts
from photodoctor.core.service import analyze_file


def scene(size=512):
    y, x = np.indices((size, size))
    img = np.zeros((size, size, 3), np.uint8)
    img[..., 0] = np.clip(40 + x * 180 / size + 30 * np.sin(y / 17), 0, 255)
    img[..., 1] = np.clip(60 + y * 150 / size + 20 * np.sin(x / 23), 0, 255)
    img[..., 2] = np.clip(100 + (x + y) * 80 / (2 * size), 0, 255)
    cv2.circle(img, (200, 220), 90, (210, 120, 80), -1)
    cv2.putText(img, "PHOTO", (60, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (240, 240, 240), 4, cv2.LINE_AA)
    return img


def test_heavy_jpeg_has_more_blocking_than_high_quality(tmp_path):
    img = scene()
    high = tmp_path / "high.jpg"
    low = tmp_path / "low.jpg"
    Image.fromarray(img).save(high, quality=95, subsampling=0)
    Image.fromarray(img).save(low, quality=18, subsampling=0)
    a = analyze_file(high).metrics["jpeg_artifacts"]
    b = analyze_file(low).metrics["jpeg_artifacts"]
    assert a.normalized_value > b.normalized_value + 30
    assert b.raw_value["block_ratio"] > a.raw_value["block_ratio"]
    assert b.raw_value["classification"] in {"moderate", "strong"}


def test_uncompressed_scene_is_not_called_strong_jpeg_blocking():
    result = analyze_jpeg_artifacts(scene(), "PNG")
    assert result.classification in {"none", "mild"}
    assert result.score > 80
    assert not result.source_is_jpeg


def test_jpeg_blocking_degrades_monotonically_with_quality(tmp_path):
    img = scene()
    scores = []
    for quality in (95, 80, 50, 30, 15):
        path = tmp_path / f"q{quality}.jpg"
        Image.fromarray(img).save(path, quality=quality, subsampling=0)
        metric = analyze_file(path).metrics["jpeg_artifacts"]
        scores.append(float(metric.normalized_value))
    # Small codec wiggles are tolerated, but the sequence must not meaningfully improve as compression gets harsher.
    assert all(later <= earlier + 3.0 for earlier, later in zip(scores, scores[1:]))
    assert scores[0] > scores[-1] + 50.0
