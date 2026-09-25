from __future__ import annotations

import numpy as np

from photodoctor.core.auto_tone_color import analyze_auto_tone_color, apply_auto_tone_color, measure_neutral_midtone_cast
from photodoctor.core.validator import apply_selected_preview, preview_action_available


def _cool_test_image() -> np.ndarray:
    # Low-contrast cool neutral gradient plus a few genuinely coloured patches.
    h, w = 120, 180
    base = np.tile(np.linspace(35, 205, w, dtype=np.float32), (h, 1))
    rgb = np.stack([base - 5, base, base + 7], axis=2)
    rgb[20:55, 20:60, :] = np.array([145, 78, 58], dtype=np.float32)
    rgb[65:100, 90:135, :] = np.array([45, 95, 145], dtype=np.float32)
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _neutral_cast_strength(rgb: np.ndarray) -> float:
    return analyze_auto_tone_color(rgb).cast_strength


def test_neutral_midtone_analysis_detects_cool_cast():
    rgb = _cool_test_image()
    profile = analyze_auto_tone_color(rgb)
    assert profile.cast_direction == "cool"
    assert profile.neutral_median_b > profile.neutral_median_g > profile.neutral_median_r
    assert profile.cast_strength > 0.03
    assert profile.confidence >= 0.70


def test_reference_auto_trio_reduces_cast_and_preserves_mean_brightness():
    rgb = _cool_test_image()
    before_profile = analyze_auto_tone_color(rgb)
    corrected = apply_auto_tone_color(rgb, 0.8, before_profile)
    before_cast, _bmed, _ = measure_neutral_midtone_cast(rgb, reference_rgb=rgb)
    after_cast, _amed, _ = measure_neutral_midtone_cast(corrected, reference_rgb=rgb)

    before_luma = float(np.mean(0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]))
    after_luma = float(np.mean(0.2126 * corrected[..., 0] + 0.7152 * corrected[..., 1] + 0.0722 * corrected[..., 2]))

    assert after_cast < before_cast
    assert abs(after_luma - before_luma) < 4.0
    assert corrected.dtype == np.uint8
    assert corrected.shape == rgb.shape


def test_reference_auto_trio_strength_zero_is_identity_and_is_deterministic():
    rgb = _cool_test_image()
    assert np.array_equal(apply_auto_tone_color(rgb, 0.0), rgb)
    a = apply_auto_tone_color(rgb, 0.8)
    b = apply_auto_tone_color(rgb, 0.8)
    assert np.array_equal(a, b)


def test_neutral_gray_is_not_given_large_colour_shift():
    g = np.tile(np.arange(32, 224, dtype=np.uint8), (96, 1))
    rgb = np.stack([g, g, g], axis=2)
    profile = analyze_auto_tone_color(rgb)
    corrected = apply_auto_tone_color(rgb, 1.0, profile)
    channel_spread = np.max(np.abs(corrected.astype(np.int16) - corrected[..., :1].astype(np.int16)))
    assert profile.cast_strength < 0.005
    assert channel_spread <= 2


def test_auto_trio_preview_is_executable_and_supersedes_atomic_tone_actions():
    rgb = _cool_test_image()
    items = [
        {
            "action_key": "auto_tone_color",
            "tested": True,
            "preview_available": True,
            "candidate": "reference_auto_tone_contrast_color_v1",
            "default_strength": 0.8,
        },
        {
            "action_key": "exposure",
            "tested": True,
            "preview_available": True,
            "candidate": "gamma=0.90",
            "default_strength": 0.7,
        },
        {
            "action_key": "contrast",
            "tested": True,
            "preview_available": True,
            "candidate": "mild_lab_clahe",
            "default_strength": 0.4,
        },
    ]
    assert preview_action_available(items[0])
    only_auto = apply_selected_preview(rgb, items, {"auto_tone_color"})
    combined = apply_selected_preview(rgb, items, {"auto_tone_color", "exposure", "contrast"})
    assert np.array_equal(only_auto, combined)
