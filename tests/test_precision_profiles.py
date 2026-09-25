from __future__ import annotations

import numpy as np

from photodoctor.core.local_contrast import analyze_local_contrast
from photodoctor.core.local_sharpness import analyze_local_sharpness
from photodoctor.core.local_tone import analyze_local_tone
from photodoctor.core.precision import get_precision, resolution_aware_positions
from photodoctor.core.spatial_windows import adaptive_square_windows


def textured(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    gray = ((xx * 3 + yy * 5 + ((xx // 17 + yy // 23) % 2) * 70) % 256).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_resolution_aware_positions_increase_sublinearly_with_resolution():
    p = get_precision("normal")
    low = resolution_aware_positions(600, p.sharp_base_positions, min_positions=p.min_positions, max_positions=p.max_positions)
    medium = resolution_aware_positions(1200, p.sharp_base_positions, min_positions=p.min_positions, max_positions=p.max_positions)
    high = resolution_aware_positions(2400, p.sharp_base_positions, min_positions=p.min_positions, max_positions=p.max_positions)
    assert low < medium < high
    assert high <= p.max_positions


def test_precise_mode_has_more_sharpness_map_cells_without_changing_scalar_score():
    summary = textured(900, 1350)
    spatial = textured(1800, 2700)
    normal = analyze_local_sharpness(summary, map_rgb=spatial, precision="normal")
    precise = analyze_local_sharpness(summary, map_rgb=spatial, precision="precise")
    assert len(precise.cells_norm) > len(normal.cells_norm)
    assert precise.map_target_positions > normal.map_target_positions
    assert precise.score == normal.score
    assert precise.soft_area_pct == normal.soft_area_pct


def test_higher_resolution_gets_more_samples_in_same_mode():
    summary = textured(700, 1050)
    low_map = textured(700, 1050)
    high_map = textured(1800, 2700)
    low = analyze_local_sharpness(summary, map_rgb=low_map, precision="normal")
    high = analyze_local_sharpness(summary, map_rgb=high_map, precision="normal")
    assert high.map_target_positions > low.map_target_positions
    assert len(high.cells_norm) > len(low.cells_norm)


def test_tone_and_contrast_precision_modes_change_map_density():
    rgb = textured(900, 1350)
    linear = (rgb.astype(np.float32) / 255.0) ** 2.2
    tone_fast = analyze_local_tone(linear, precision="fast")
    tone_precise = analyze_local_tone(linear, precision="precise")
    assert len(tone_precise.cells_norm) > len(tone_fast.cells_norm)

    contrast_fast = analyze_local_contrast(rgb, map_rgb=rgb, precision="fast")
    contrast_precise = analyze_local_contrast(rgb, map_rgb=rgb, precision="precise")
    assert contrast_precise.map_cells > contrast_fast.map_cells
    assert contrast_precise.score == contrast_fast.score
    assert contrast_precise.fading_likelihood == contrast_fast.fading_likelihood


def test_precise_profile_enables_real_deep_analysis_budget():
    normal = get_precision("normal")
    precise = get_precision("precise")
    assert not normal.deep_analysis
    assert precise.deep_analysis
    assert precise.detail_long_edge == 4096
    assert precise.spatial_long_edge == 4096
    assert precise.max_positions > normal.max_positions


def test_native_center_crop_preserves_original_pixels_without_resampling():
    from photodoctor.core.analyzer import _native_center_crop
    rgb = np.arange(10 * 14 * 3, dtype=np.uint8).reshape(10, 14, 3)
    crop, label = _native_center_crop(rgb, long_edge_limit=7)
    assert crop.shape[:2] == (5, 7)
    assert label == "native_crop_7x5"
    y0 = (10 - 5) // 2
    x0 = (14 - 7) // 2
    assert np.array_equal(crop, rgb[y0:y0 + 5, x0:x0 + 7])


def test_maximum_is_fourth_profile_and_materially_deeper_than_precise():
    from photodoctor.core.precision import all_precisions

    profiles = all_precisions()
    assert [profile.key for profile in profiles] == ["fast", "normal", "precise", "maximum"]
    precise = get_precision("precise")
    maximum = get_precision("maximum")
    assert maximum.label == "Максимальный"
    assert maximum.deep_analysis
    assert maximum.spatial_long_edge == 8192
    assert maximum.detail_long_edge == 8192
    assert maximum.max_positions > precise.max_positions
    assert maximum.sharp_base_positions > precise.sharp_base_positions
    assert maximum.tone_base_positions > precise.tone_base_positions
    assert maximum.contrast_base_positions > precise.contrast_base_positions
    # Compatibility with 0.5.9 scripts/settings.
    assert get_precision("ideal") == maximum
    assert get_precision("идеально") == maximum


def test_maximum_builds_much_denser_local_maps_without_changing_scalar_scores():
    summary = textured(900, 1350)
    spatial = textured(1800, 2700)

    precise_sharp = analyze_local_sharpness(summary, map_rgb=spatial, precision="precise")
    maximum_sharp = analyze_local_sharpness(summary, map_rgb=spatial, precision="maximum")
    assert len(maximum_sharp.cells_norm) > 1.8 * len(precise_sharp.cells_norm)
    assert maximum_sharp.map_target_positions > precise_sharp.map_target_positions
    assert maximum_sharp.score == precise_sharp.score
    assert maximum_sharp.soft_area_pct == precise_sharp.soft_area_pct

    linear = (summary.astype(np.float32) / 255.0) ** 2.2
    precise_tone = analyze_local_tone(linear, precision="precise")
    maximum_tone = analyze_local_tone(linear, precision="maximum")
    assert len(maximum_tone.cells_norm) > 1.8 * len(precise_tone.cells_norm)

    precise_contrast = analyze_local_contrast(summary, map_rgb=spatial, precision="precise")
    maximum_contrast = analyze_local_contrast(summary, map_rgb=spatial, precision="maximum")
    assert maximum_contrast.map_cells > 1.8 * precise_contrast.map_cells
    assert maximum_contrast.score == precise_contrast.score
    assert maximum_contrast.fading_likelihood == precise_contrast.fading_likelihood


def test_maximum_4k_spatial_budget_is_about_four_times_old_ideal_and_eight_times_precise():
    h, w = 2160, 3840

    def total_windows(key: str) -> int:
        p = get_precision(key)
        total = 0
        for base, overlap in (
            (p.sharp_base_positions, p.sharp_overlap),
            (p.tone_base_positions, p.tone_overlap),
            (p.contrast_base_positions, p.contrast_overlap),
        ):
            positions = resolution_aware_positions(
                min(h, w), base, min_positions=p.min_positions, max_positions=p.max_positions
            )
            _spec, windows = adaptive_square_windows(
                h, w, target_short_positions=positions, overlap=overlap
            )
            total += len(windows)
        return total

    precise = total_windows("precise")
    maximum = total_windows("maximum")
    ratio = maximum / precise
    assert 8.0 <= ratio <= 9.0
    assert 42000 <= maximum <= 44000

    # The old 0.5.9 fourth mode used 6144 px; Maximum raises the deep ceiling
    # to 8192 px for sources that actually contain that resolution.
    p = get_precision("precise")
    m = get_precision("maximum")
    pixel_budget_ratio = (m.detail_long_edge / p.detail_long_edge) ** 2
    assert pixel_budget_ratio == 4.0
