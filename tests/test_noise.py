import numpy as np

from photodoctor.core.noise import analyze_noise


def smooth_scene(size: int = 480) -> np.ndarray:
    y, x = np.indices((size, size))
    gray = 75.0 + 105.0 * x / size + 25.0 * np.sin(y / 67.0)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def with_noise(rgb: np.ndarray, sigma: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, rgb.shape[:2])[..., None]
    return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def test_noise_estimate_tracks_added_sigma_monotonically():
    base = smooth_scene()
    estimates = [analyze_noise(with_noise(base, sigma)).sigma_luma for sigma in (1, 2, 4, 8)]
    assert all(b > a for a, b in zip(estimates, estimates[1:]))
    assert 0.65 <= estimates[0] <= 1.4
    assert 6.0 <= estimates[-1] <= 9.5


def test_clean_smooth_image_is_not_called_noisy():
    result = analyze_noise(smooth_scene())
    assert result.score > 95
    assert result.classification in {"very_low", "low"}
    assert result.usable_tiles > 0


def test_heavy_noise_scores_worse_than_clean():
    base = smooth_scene()
    clean = analyze_noise(base)
    noisy = analyze_noise(with_noise(base, 10.0))
    assert clean.score > noisy.score + 40
    assert noisy.classification in {"high", "strong"}


def test_tiny_image_is_unknown():
    result = analyze_noise(np.full((24, 24, 3), 128, np.uint8))
    assert result.classification == "unknown"
    assert result.confidence == 0.0


def test_noise_result_keeps_spatial_cells_without_recalibrating_global_score():
    base = smooth_scene()
    clean = analyze_noise(base)
    noisy = base.copy()
    rng = np.random.default_rng(13)
    patch = rng.normal(0.0, 8.0, (192, 192, 1))
    noisy[96:288, 96:288] = np.clip(noisy[96:288, 96:288].astype(np.float32) + patch, 0, 255).astype(np.uint8)
    local = analyze_noise(noisy)
    assert local.cells_norm
    assert any(float(cell["sigma_luma"]) >= 2.7 for cell in local.cells_norm)
    # A local noisy patch is allowed to leave the robust whole-image median nearly clean.
    assert local.score >= 70.0
    assert clean.score >= local.score
