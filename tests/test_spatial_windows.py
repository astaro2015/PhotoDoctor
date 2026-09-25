from photodoctor.core.spatial_windows import adaptive_square_windows


def test_sharpness_style_grid_has_about_fifteen_samples_on_short_side():
    spec, windows = adaptive_square_windows(
        600, 900, target_short_positions=15, overlap=0.50, min_window=40, max_window=280
    )
    ys = sorted({y0 for y0, _, _, _ in windows})
    assert 14 <= len(ys) <= 16
    assert spec.step < spec.window
    assert 0.45 <= spec.overlap <= 0.55


def test_last_windows_reach_far_edges_without_unanalysed_strip():
    _, windows = adaptive_square_windows(
        613, 877, target_short_positions=13, overlap=0.40, min_window=40, max_window=280
    )
    assert max(y1 for _, y1, _, _ in windows) == 613
    assert max(x1 for _, _, _, x1 in windows) == 877


def test_windows_scale_with_resolution_instead_of_fixed_giant_tiles():
    low_spec, _ = adaptive_square_windows(
        480, 720, target_short_positions=15, overlap=0.50, min_window=40, max_window=280
    )
    high_spec, _ = adaptive_square_windows(
        1600, 2400, target_short_positions=15, overlap=0.50, min_window=40, max_window=280
    )
    assert high_spec.window > low_spec.window
    assert high_spec.window / 1600 < 0.18
    assert low_spec.window / 480 < 0.18
