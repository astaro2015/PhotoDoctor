from pathlib import Path

import numpy as np
from PIL import Image

from photodoctor.core.analyzer import analyze_classical
from photodoctor.core.loader import load_image
from photodoctor.core.local_correction_planner import (
    apply_local_contrast,
    apply_local_exposure,
    build_local_correction_plan,
)
from photodoctor.core.models import MetricResult
from photodoctor.core.validator import apply_selected_preview, validate_recommendations


def _save(tmp_path: Path, name: str, rgb: np.ndarray) -> Path:
    path = tmp_path / name
    Image.fromarray(rgb).save(path)
    return path


def _dark_patch_image() -> np.ndarray:
    rgb = np.full((600, 800, 3), 170, dtype=np.uint8)
    rgb[180:420, 260:540] = 65
    for y in range(180, 420, 20):
        rgb[y:y + 4, 260:540] = 95
    return rgb


def _low_contrast_patch_image() -> np.ndarray:
    h, w = 600, 800
    yy, xx = np.indices((h, w))
    base = np.where(((xx // 16 + yy // 16) % 2) == 0, 90, 185).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    patch = np.where(
        (((xx[170:430, 240:560] // 10) + (yy[170:430, 240:560] // 10)) % 2) == 0,
        118,
        132,
    ).astype(np.uint8)
    rgb[170:430, 240:560] = patch[..., None]
    return rgb


def test_spatial_exposure_changes_dark_problem_region_not_normal_background(tmp_path):
    rgb = _dark_patch_image()
    loaded = load_image(_save(tmp_path, "local_exposure.png", rgb))
    result = analyze_classical(loaded, precision="precise")
    plan = result.metrics["local_correction_plan"].raw_value
    assert plan["exposure_cells"]
    assert 0.01 < float(plan["exposure_coverage"]) < 0.40

    items = validate_recommendations(loaded, result.metrics, precision="precise")
    exposure = next(item for item in items if item.action_key == "exposure")
    assert exposure.candidate == "spatial_exposure_v1"
    assert exposure.technical_passed

    out = apply_selected_preview(rgb, [exposure.to_dict()], {"exposure"})
    assert float(np.mean(np.abs(out[:100].astype(np.int16) - rgb[:100].astype(np.int16)))) < 0.05
    assert float(np.mean(out[190:250, 280:520])) > float(np.mean(rgb[190:250, 280:520]))


def test_spatial_contrast_changes_flat_region_not_good_background(tmp_path):
    rgb = _low_contrast_patch_image()
    loaded = load_image(_save(tmp_path, "local_contrast.png", rgb))
    result = analyze_classical(loaded, precision="precise")
    plan = result.metrics["local_correction_plan"].raw_value
    assert plan["contrast_cells"]
    assert 0.01 < float(plan["contrast_coverage"]) < 0.40

    items = validate_recommendations(loaded, result.metrics, precision="precise")
    contrast = next(item for item in items if item.action_key == "contrast")
    assert contrast.candidate == "spatial_contrast_v1"
    assert contrast.technical_passed
    out = apply_selected_preview(rgb, [contrast.to_dict()], {"contrast"})
    patch_delta = float(np.mean(np.abs(out[220:380, 300:500].astype(np.int16) - rgb[220:380, 300:500].astype(np.int16))))
    background_delta = float(np.mean(np.abs(out[:100].astype(np.int16) - rgb[:100].astype(np.int16))))
    assert patch_delta > 0.25
    assert background_delta < 0.05


def test_single_tone_tile_is_not_enough_for_spatial_fix():
    metrics = {
        "local_tone": MetricResult(
            "local_tone",
            {"cells_norm": [{
                "x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2,
                "status": "dark", "p25": 0.07, "p50": 0.10, "p75": 0.13, "p95": 0.30,
                "mean": 0.12, "shadow_clip": 0.0, "highlight_clip": 0.0,
            }]},
            None, 0.9, "test",
        ),
        "local_contrast": MetricResult("local_contrast", {"cells_norm": []}, 80.0, 0.8, "test"),
    }
    plan = build_local_correction_plan(metrics)
    assert not plan.exposure_cells
    assert plan.exposure_coverage == 0.0


def test_local_exposure_highlight_guard_prevents_clipped_white_change():
    rgb = np.full((120, 160, 3), 110, dtype=np.uint8)
    rgb[:, 120:] = 255
    plan = {
        "exposure_cells": [{
            "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0,
            "delta_gamma": -0.16, "target_luma": 0.18, "kind": "lift",
        }]
    }
    out = apply_local_exposure(rgb, plan, 1.0)
    assert np.array_equal(out[:, 130:], rgb[:, 130:])
    assert float(out[:, :80].mean()) > float(rgb[:, :80].mean())


def test_manual_region_can_further_limit_automatic_spatial_mask(tmp_path):
    rgb = _dark_patch_image()
    loaded = load_image(_save(tmp_path, "local_manual_limit.png", rgb))
    result = analyze_classical(loaded, precision="precise")
    exposure = next(item for item in validate_recommendations(loaded, result.metrics, precision="precise") if item.action_key == "exposure")
    whole_auto = apply_selected_preview(rgb, [exposure.to_dict()], {"exposure"})
    limited = apply_selected_preview(
        rgb,
        [exposure.to_dict()],
        {"exposure"},
        action_regions={"exposure": {"x": 0.30, "y": 0.30, "w": 0.18, "h": 0.30}},
    )
    assert not np.array_equal(whole_auto, limited)
    assert np.array_equal(limited[:, :100], rgb[:, :100])


def _locally_soft_image() -> np.ndarray:
    import cv2
    h, w = 480, 640
    yy, xx = np.indices((h, w))
    checker = np.where(((xx // 10 + yy // 10) % 2) == 0, 70, 190).astype(np.uint8)
    rgb = np.repeat(checker[..., None], 3, axis=2)
    rgb[140:360, 220:460] = cv2.GaussianBlur(rgb[140:360, 220:460], (0, 0), 3.0)
    return rgb


def _locally_noisy_image() -> np.ndarray:
    rgb = np.full((480, 640, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 8.0, (240, 320, 1))
    rgb[120:360, 160:480] = np.clip(
        rgb[120:360, 160:480].astype(np.float32) + noise, 0, 255
    ).astype(np.uint8)
    return rgb


def test_spatial_sharpness_changes_soft_region_not_normal_background(tmp_path):
    rgb = _locally_soft_image()
    loaded = load_image(_save(tmp_path, "local_sharpness.png", rgb))
    result = analyze_classical(loaded, precision="precise")
    plan = result.metrics["local_correction_plan"].raw_value
    assert plan["sharpness_cells"]
    assert 0.01 < float(plan["sharpness_coverage"]) < 0.40

    item = next(x for x in validate_recommendations(loaded, result.metrics, precision="precise") if x.action_key == "sharpness")
    assert item.candidate == "spatial_sharpness_v1"
    assert item.technical_passed
    out = apply_selected_preview(rgb, [item.to_dict()], {"sharpness"})
    patch_delta = float(np.mean(np.abs(out[180:320, 260:420].astype(np.int16) - rgb[180:320, 260:420].astype(np.int16))))
    background_delta = float(np.mean(np.abs(out[:80].astype(np.int16) - rgb[:80].astype(np.int16))))
    assert patch_delta > 0.10
    assert background_delta < 0.02


def test_spatial_denoise_changes_noisy_region_not_clean_background(tmp_path):
    rgb = _locally_noisy_image()
    loaded = load_image(_save(tmp_path, "local_noise.png", rgb))
    result = analyze_classical(loaded, precision="precise")
    plan = result.metrics["local_correction_plan"].raw_value
    assert plan["noise_cells"]
    assert 0.01 < float(plan["noise_coverage"]) < 0.40

    item = next(x for x in validate_recommendations(loaded, result.metrics, precision="precise") if x.action_key == "noise")
    assert item.candidate == "spatial_denoise_v1"
    assert item.technical_passed
    out = apply_selected_preview(rgb, [item.to_dict()], {"noise"})
    assert float(out[180:300, 220:420, 0].std()) < float(rgb[180:300, 220:420, 0].std())
    assert float(np.mean(np.abs(out[:80].astype(np.int16) - rgb[:80].astype(np.int16)))) < 0.02


def test_single_noisy_tile_is_not_enough_for_spatial_denoise():
    metrics = {
        "noise": MetricResult(
            "noise",
            {"cells_norm": [{
                "x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2,
                "sigma_luma": 7.0, "flat_fraction": 0.7, "confidence": 0.8, "status": "high",
            }]},
            58.0, 0.8, "test",
        ),
        "local_sharpness": MetricResult("local_sharpness", {"cells_norm": []}, 80.0, 0.8, "test"),
    }
    plan = build_local_correction_plan(metrics)
    assert not plan.noise_cells
    assert plan.noise_coverage == 0.0


def test_confirmed_noise_vetoes_sharpening_same_region():
    soft = []
    noisy = []
    for row in range(3):
        for col in range(3):
            x = 0.30 + col * 0.15
            y = 0.20 + row * 0.20
            soft.append({
                "x": x, "y": y, "w": 0.15, "h": 0.20,
                "score": 38.0, "confidence": 0.85, "status": "soft",
            })
            noisy.append({
                "x": x, "y": y, "w": 0.15, "h": 0.20,
                "sigma_luma": 6.0, "flat_fraction": 0.65, "confidence": 0.8, "status": "high",
            })
    metrics = {
        "local_sharpness": MetricResult("local_sharpness", {"cells_norm": soft}, 50.0, 0.85, "test"),
        "noise": MetricResult("noise", {"cells_norm": noisy}, 64.0, 0.8, "test"),
    }
    plan = build_local_correction_plan(metrics)
    assert plan.noise_cells
    assert not plan.sharpness_cells


def test_confirmed_noise_vetoes_local_contrast_same_region():
    low_contrast = []
    noisy = []
    for row in range(3):
        for col in range(3):
            x = 0.30 + col * 0.15
            y = 0.20 + row * 0.20
            low_contrast.append({
                "x": x, "y": y, "w": 0.15, "h": 0.20,
                "score": 18.0, "range": 0.105, "confidence": 0.86,
                "median": 0.50, "status": "low",
            })
            noisy.append({
                "x": x, "y": y, "w": 0.15, "h": 0.20,
                "sigma_luma": 6.0, "flat_fraction": 0.65, "confidence": 0.82,
                "status": "high",
            })
    metrics = {
        "local_contrast": MetricResult("local_contrast", {"cells_norm": low_contrast}, 30.0, 0.85, "test"),
        "noise": MetricResult("noise", {"cells_norm": noisy}, 64.0, 0.82, "test"),
    }
    plan = build_local_correction_plan(metrics)
    assert plan.noise_cells
    assert not plan.contrast_cells
