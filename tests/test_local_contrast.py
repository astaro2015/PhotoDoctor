import numpy as np

from photodoctor.core.local_contrast import analyze_local_contrast


def checker(low: int, high: int, size: int = 360, block: int = 30):
    yy, xx = np.indices((size, size))
    mask = ((yy // block) + (xx // block)) % 2
    gray = np.where(mask, high, low).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_strong_local_contrast_scores_above_flattened_version():
    strong = analyze_local_contrast(checker(35, 220))
    flat = analyze_local_contrast(checker(105, 145))
    assert strong.score > flat.score + 25
    assert strong.fading_likelihood < flat.fading_likelihood


def test_uniform_image_is_not_called_faded_just_for_being_smooth():
    img = np.full((360, 360, 3), 128, np.uint8)
    result = analyze_local_contrast(img)
    assert result.informative_cells == 0
    assert result.classification == "unknown"


def test_flat_structured_image_becomes_fading_candidate_or_mixed():
    # Real low-contrast structure, not random sensor noise. Random grain belongs to
    # the noise layer and must not manufacture a fading diagnosis.
    img = checker(118, 132, size=360, block=12)
    result = analyze_local_contrast(img)
    assert result.informative_cells > 0
    assert result.classification in {"fading_candidate", "mixed"}
    assert result.score < 55


def test_contrast_map_is_finer_than_old_five_cell_short_side_grid():
    img = checker(40, 210, size=600, block=20)
    result = analyze_local_contrast(img)
    ys = {round(float(cell["y"]), 6) for cell in result.cells_norm}
    assert 12 <= len(ys) <= 15
    assert max(float(cell["h"]) for cell in result.cells_norm) < 0.16


def test_fine_map_does_not_replace_stable_summary_scale():
    img = checker(35, 220, size=600, block=30)
    result = analyze_local_contrast(img)
    # Visualization and scalar summary use independent, fixed scales.  The summary
    # is overlapping now, but it must stay bounded and independent of UI precision.
    assert result.map_cells > result.total_cells
    assert result.total_cells <= 140


def test_random_noise_is_not_called_low_contrast_or_fading():
    rng = np.random.default_rng(43)
    gray = np.clip(128.0 + rng.normal(0.0, 8.0, (600, 800)), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    result = analyze_local_contrast(rgb, map_rgb=rgb, precision="precise")
    assert result.cells_norm
    assert all(cell["status"] == "low_texture" for cell in result.cells_norm)
    assert result.informative_cells == 0
    assert result.low_contrast_area_pct == 0.0
    assert result.classification == "unknown"


def test_structural_low_contrast_texture_remains_detectable_after_noise_guard():
    img = checker(118, 132, size=600, block=12)
    result = analyze_local_contrast(img, map_rgb=img, precision="precise")
    assert result.informative_cells > 0
    assert any(cell["status"] == "low" for cell in result.cells_norm)
    assert result.score < 38.0


def test_low_contrast_area_is_not_strongly_biased_by_grid_position():
    def measure(y0: int, x0: int) -> float:
        h, w = 600, 800
        yy, xx = np.indices((h, w))
        base = np.where(((yy // 12 + xx // 12) % 2) > 0, 205, 50).astype(np.uint8)
        low = np.where(((yy // 12 + xx // 12) % 2) > 0, 145, 110).astype(np.uint8)
        base[y0:y0 + 240, x0:x0 + 280] = low[y0:y0 + 240, x0:x0 + 280]
        rgb = np.repeat(base[..., None], 3, axis=2)
        return analyze_local_contrast(rgb, map_rgb=rgb, precision="precise").low_contrast_area_pct

    values = [measure(180, 260), measure(0, 0), measure(180, 0)]
    assert max(values) - min(values) < 2.5
    assert all(6.0 < value < 14.0 for value in values)


def test_summary_only_score_is_exactly_the_full_analyzer_score():
    from photodoctor.core.local_contrast import local_contrast_summary_score

    images = [
        checker(35, 220),
        checker(105, 145),
        np.full((96, 128, 3), 127, np.uint8),
    ]
    rng = np.random.default_rng(20260924)
    images.append(rng.integers(0, 256, size=(180, 260, 3), dtype=np.uint8))
    for rgb in images:
        assert local_contrast_summary_score(rgb) == analyze_local_contrast(rgb).score
