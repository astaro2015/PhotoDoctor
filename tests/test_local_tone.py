from __future__ import annotations

import numpy as np

from photodoctor.core.local_tone import analyze_local_tone


def test_dark_linear_image_is_described_as_dark_or_shadow():
    img = np.full((500, 700, 3), 0.08, dtype=np.float32)
    result = analyze_local_tone(img)
    assert result.total_cells > 0
    assert result.dark_area_pct + result.deep_shadow_area_pct > 90.0


def test_mid_tone_image_is_not_marked_as_extreme():
    img = np.full((500, 700, 3), 0.38, dtype=np.float32)
    result = analyze_local_tone(img)
    assert result.dark_area_pct < 1.0
    assert result.bright_area_pct < 1.0
    assert result.deep_shadow_area_pct < 1.0
    assert result.highlight_clip_area_pct < 1.0


def test_split_dark_and_bright_image_reports_both_regions():
    img = np.empty((500, 700, 3), dtype=np.float32)
    img[:, :350] = 0.10
    img[:, 350:] = 0.78
    result = analyze_local_tone(img)
    assert result.dark_area_pct > 25.0
    assert result.bright_area_pct > 25.0


def test_tone_map_uses_more_than_ten_samples_across_short_side():
    img = np.full((600, 900, 3), 0.38, dtype=np.float32)
    result = analyze_local_tone(img)
    ys = {round(float(cell["y"]), 6) for cell in result.cells_norm}
    assert 11 <= len(ys) <= 14
    assert max(float(cell["h"]) for cell in result.cells_norm) < 0.16


def test_small_isolated_white_spot_does_not_mark_whole_windows_as_clipped_highlight():
    img = np.full((600, 800, 3), 0.35, dtype=np.float32)
    img[288:312, 388:412] = 1.0
    result = analyze_local_tone(img, precision="precise")
    assert result.highlight_clip_area_pct < 0.5
    assert not any(cell["status"] == "clipped_highlight" for cell in result.cells_norm)


def test_small_isolated_black_spot_does_not_mark_whole_windows_as_deep_shadow():
    img = np.full((600, 800, 3), 0.35, dtype=np.float32)
    img[284:316, 384:416] = 0.0
    result = analyze_local_tone(img, precision="precise")
    assert result.deep_shadow_area_pct < 0.5
    assert not any(cell["status"] == "deep_shadow" for cell in result.cells_norm)


def test_tone_area_percentage_is_not_strongly_biased_by_overlap_position():
    def measure(y0: int, x0: int) -> float:
        img = np.full((600, 800, 3), 0.35, dtype=np.float32)
        img[y0:y0 + 240, x0:x0 + 280] = 0.08
        return analyze_local_tone(img, precision="precise").dark_area_pct

    center = measure(180, 260)
    corner = measure(0, 0)
    edge = measure(180, 0)
    assert max(center, corner, edge) - min(center, corner, edge) < 3.0
    assert 9.0 < center < 18.0


def test_bright_near_white_is_not_called_clipped_without_actual_clipping():
    # 240/255 converts to ~0.871 linear: bright, but it is not saturated/clipped.
    from photodoctor.core.loader import srgb_to_linear
    rgb = np.full((500, 700, 3), 240, dtype=np.uint8)
    result = analyze_local_tone(srgb_to_linear(rgb), precision="normal")
    assert result.bright_area_pct > 99.0
    assert result.highlight_clip_area_pct < 0.1
    assert all(cell["status"] == "bright" for cell in result.cells_norm)


def test_true_white_region_is_still_called_clipped_highlight():
    img = np.ones((500, 700, 3), dtype=np.float32)
    result = analyze_local_tone(img, precision="normal")
    assert result.highlight_clip_area_pct > 99.0
    assert all(cell["status"] == "clipped_highlight" for cell in result.cells_norm)
