from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from photodoctor.core.service import analyze_file


def _save(path, arr):
    Image.fromarray(arr.astype(np.uint8), "RGB").save(path)


def test_black_and_white_clipping(tmp_path):
    black = np.zeros((256, 256, 3), np.uint8)
    white = np.full((256, 256, 3), 255, np.uint8)
    p0, p1 = tmp_path / "black.png", tmp_path / "white.png"
    _save(p0, black); _save(p1, white)
    a = analyze_file(p0); b = analyze_file(p1)
    assert a.metrics["shadow_clipping"].raw_value > 99.0
    assert b.metrics["highlight_clipping"].raw_value > 99.0
    assert a.metrics["brightness"].raw_value < b.metrics["brightness"].raw_value


def test_blur_reduces_sharpness(tmp_path):
    y, x = np.indices((512, 512))
    c = (((x // 8 + y // 8) % 2) * 255).astype(np.uint8)
    rgb = np.dstack([c, c, c])
    sharp = Image.fromarray(rgb, "RGB")
    blur = sharp.filter(ImageFilter.GaussianBlur(radius=3.0))
    p0, p1 = tmp_path / "sharp.png", tmp_path / "blur.png"
    sharp.save(p0); blur.save(p1)
    a = analyze_file(p0); b = analyze_file(p1)
    assert a.metrics["laplacian"].raw_value > b.metrics["laplacian"].raw_value
    assert a.metrics["tenengrad"].raw_value > b.metrics["tenengrad"].raw_value


def test_noise_increases_noise_proxy(tmp_path):
    rng = np.random.default_rng(123)
    base = np.full((512, 512, 3), 128, np.int16)
    noisy = np.clip(base + rng.normal(0, 20, base.shape), 0, 255).astype(np.uint8)
    clean = base.astype(np.uint8)
    p0, p1 = tmp_path / "clean.png", tmp_path / "noisy.png"
    _save(p0, clean); _save(p1, noisy)
    a = analyze_file(p0); b = analyze_file(p1)
    assert a.metrics["noise"].raw_value["sigma_luma"] < b.metrics["noise"].raw_value["sigma_luma"]


def test_color_cast_detected(tmp_path):
    neutral = np.full((128, 128, 3), 128, np.uint8)
    warm = np.full((128, 128, 3), [210, 130, 80], np.uint8)
    p0, p1 = tmp_path / "neutral.png", tmp_path / "warm.png"
    _save(p0, neutral); _save(p1, warm)
    a = analyze_file(p0); b = analyze_file(p1)
    assert a.metrics["color_cast"].raw_value["relative_spread"] < b.metrics["color_cast"].raw_value["relative_spread"]


def test_multiple_resolutions_same_source_are_reasonably_consistent(tmp_path):
    y, x = np.indices((1024, 1536))
    c = (((x // 24 + y // 24) % 2) * 180 + 35).astype(np.uint8)
    rgb = np.dstack([c, c, c])
    img = Image.fromarray(rgb, "RGB")
    p0, p1 = tmp_path / "large.png", tmp_path / "small.png"
    img.save(p0)
    img.resize((768, 512), Image.Resampling.LANCZOS).save(p1)
    a = analyze_file(p0); b = analyze_file(p1)
    assert abs(a.metrics["brightness"].raw_value - b.metrics["brightness"].raw_value) < 0.03
    assert abs(a.metrics["contrast"].raw_value - b.metrics["contrast"].raw_value) < 0.08


def test_image_tone_detects_monochrome(tmp_path):
    gray = np.full((160, 200, 3), 142, np.uint8)
    p = tmp_path / "gray.png"
    _save(p, gray)
    result = analyze_file(p)
    raw = result.metrics["image_tone"].raw_value
    assert raw["classification"] == "monochrome"
    assert raw["confidence"] >= 0.70


def test_image_tone_detects_soft_sepia(tmp_path):
    base = np.full((160, 200, 3), [165, 148, 126], np.uint8)
    # Add a luminance gradient while preserving the warm channel order.
    gradient = np.linspace(-35, 35, 200, dtype=np.int16)[None, :, None]
    sepia = np.clip(base.astype(np.int16) + gradient, 0, 255).astype(np.uint8)
    sepia = np.repeat(sepia[:1, :, :], 160, axis=0)
    p = tmp_path / "sepia.png"
    _save(p, sepia)
    result = analyze_file(p)
    raw = result.metrics["image_tone"].raw_value
    assert raw["classification"] == "sepia"
    assert raw["warm_order_fraction"] > 0.90


def test_image_tone_does_not_call_strong_color_sepia(tmp_path):
    orange = np.full((120, 160, 3), [220, 105, 35], np.uint8)
    p = tmp_path / "orange.png"
    _save(p, orange)
    result = analyze_file(p)
    assert result.metrics["image_tone"].raw_value["classification"] == "color"


def test_analysis_contains_local_contrast_metric(tmp_path):
    y, x = np.indices((360, 480))
    c = (((x // 24 + y // 24) % 2) * 160 + 45).astype(np.uint8)
    rgb = np.dstack([c, c, c])
    p = tmp_path / "contrast.png"
    _save(p, rgb)
    result = analyze_file(p)
    metric = result.metrics["local_contrast"]
    assert metric.normalized_value is not None
    assert isinstance(metric.raw_value, dict)
    assert "fading_likelihood" in metric.raw_value
    assert "cells_norm" in metric.raw_value


def test_full_analysis_contains_structured_decision_plan(tmp_path):
    from PIL import Image
    from photodoctor.core.loader import load_image
    from photodoctor.core.analyzer import analyze_classical
    p = tmp_path / "dark.png"
    Image.new("RGB", (180, 140), (45, 45, 45)).save(p)
    result = analyze_classical(load_image(p))
    metric = result.metrics["decision_plan"]
    assert metric.normalized_value is None
    assert isinstance(metric.raw_value, dict)
    assert metric.raw_value["engine_version"]
    assert isinstance(metric.raw_value["items"], list)
    assert metric.raw_value["items"]


def test_analyze_file_adds_recommendation_validation_metric(tmp_path):
    from PIL import Image
    from photodoctor.core.service import analyze_file
    p = tmp_path / "dark_service.png"
    Image.new("RGB", (240, 180), (70, 70, 70)).save(p)
    result = analyze_file(p)
    metric = result.metrics["recommendation_validation"]
    assert isinstance(metric.raw_value, dict)
    assert metric.raw_value["validator_version"]
    assert metric.raw_value["tested_count"] >= 1
    assert any(item.get("action_key") == "exposure" for item in metric.raw_value["items"])


def test_analysis_contains_main_subject_metric(tmp_path):
    p = tmp_path / "flat_subject.png"
    _save(p, np.full((220, 320, 3), 128, np.uint8))
    result = analyze_file(p)
    metric = result.metrics["main_subject"]
    assert isinstance(metric.raw_value, dict)
    assert metric.raw_value["method"] == "main_subject_rules_saliency_v1"
    assert metric.raw_value["subject_kind"] in {"unknown", "visual_region", "person", "people_group"}
    assert "scene_kind" in metric.raw_value
    assert "protection_box_norm" in metric.raw_value


def test_analysis_contains_separate_aesthetic_metric(tmp_path):
    p = tmp_path / "aesthetic.png"
    arr = np.full((260, 360, 3), 120, np.uint8)
    arr[50:220, 105:255] = (185, 145, 115)
    _save(p, arr)
    result = analyze_file(p)
    metric = result.metrics["aesthetic_quality"]
    assert isinstance(metric.raw_value, dict)
    assert metric.raw_value["technical_quality_independent"] is True
    assert metric.raw_value["method"] == "aesthetic_subject_composition_v1"


def test_analysis_contains_reliability_unknown_guard_metric(tmp_path):
    p = tmp_path / "reliability.png"
    y, x = np.indices((320, 480))
    c = (((x // 20 + y // 20) % 2) * 150 + 45).astype(np.uint8)
    _save(p, np.dstack([c, c, c]))
    result = analyze_file(p)
    metric = result.metrics["analysis_reliability"]
    assert isinstance(metric.raw_value, dict)
    assert metric.raw_value["status"] in {"reliable", "caution", "unknown", "ood_candidate"}
    assert metric.raw_value["method"] == "heuristic_analysis_reliability_v1"
    assert metric.raw_value["probability_interpretation"] is False
