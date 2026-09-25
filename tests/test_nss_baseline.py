import cv2
import numpy as np

from photodoctor.core.nss_baseline import analyze_nss_baseline


def textured_image(size=256):
    y, x = np.indices((size, size))
    base = 110 + 45 * np.sin(x / 9.0) + 30 * np.cos(y / 13.0)
    texture = ((x * 17 + y * 11) % 23) - 11
    gray = np.clip(base + texture, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_nss_baseline_produces_36_brisque_style_features():
    result = analyze_nss_baseline(textured_image())
    assert result.informative
    assert result.feature_count == 36
    assert len(result.features) == 36
    assert all(np.isfinite(v) for v in result.features)


def test_nss_baseline_is_deterministic():
    rgb = textured_image(192)
    a = analyze_nss_baseline(rgb)
    b = analyze_nss_baseline(rgb)
    assert np.allclose(a.features, b.features, rtol=0, atol=1e-12)


def test_uniform_image_is_not_claimed_as_reliable_nss_quality_signal():
    rgb = np.full((160, 160, 3), 128, np.uint8)
    result = analyze_nss_baseline(rgb)
    assert result.feature_count == 36
    assert not result.informative
    assert result.confidence <= 0.35


def test_blur_and_noise_change_feature_vector_without_inventing_quality_score():
    clean = textured_image()
    blur = cv2.GaussianBlur(clean, (0, 0), 2.5)
    rng = np.random.default_rng(123)
    noisy = np.clip(clean.astype(np.int16) + rng.normal(0, 16, clean.shape).astype(np.int16), 0, 255).astype(np.uint8)
    f_clean = np.array(analyze_nss_baseline(clean).features)
    f_blur = np.array(analyze_nss_baseline(blur).features)
    f_noisy = np.array(analyze_nss_baseline(noisy).features)
    assert np.linalg.norm(f_clean - f_blur) > 0.05
    assert np.linalg.norm(f_clean - f_noisy) > 0.05
