import cv2
import numpy as np

from photodoctor.core.local_sharpness import analyze_local_sharpness


def checkerboard(size=640, block=16):
    y, x = np.indices((size, size))
    gray = (((x // block + y // block) % 2) * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def test_blur_reduces_local_sharpness_score_and_increases_soft_area():
    sharp = checkerboard()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 5.0)
    a = analyze_local_sharpness(sharp)
    b = analyze_local_sharpness(blurred)
    assert a.informative_cells > 0
    assert b.informative_cells > 0
    assert a.score > b.score
    assert b.soft_area_pct >= a.soft_area_pct


def test_uniform_image_is_not_declared_blurry_just_for_being_smooth():
    rgb = np.full((600, 800, 3), 128, dtype=np.uint8)
    result = analyze_local_sharpness(rgb)
    assert result.total_cells > 0
    assert result.informative_cells == 0
    assert result.soft_area_pct == 0.0
    assert all(cell["status"] == "low_texture" for cell in result.cells_norm)


def test_cells_use_normalized_coordinates_and_known_statuses():
    result = analyze_local_sharpness(checkerboard(480, 12))
    assert result.cells_norm
    valid = {"low_texture", "soft", "medium", "sharp"}
    for cell in result.cells_norm:
        assert 0.0 <= float(cell["x"]) <= 1.0
        assert 0.0 <= float(cell["y"]) <= 1.0
        assert 0.0 < float(cell["w"]) <= 1.0
        assert 0.0 < float(cell["h"]) <= 1.0
        assert 0.0 <= float(cell["score"]) <= 100.0
        assert 0.0 <= float(cell["confidence"]) <= 1.0
        assert cell["status"] in valid


def test_map_is_fine_grained_and_overlapping_on_typical_portrait():
    result = analyze_local_sharpness(checkerboard(600, 14))
    xs = sorted({round(float(cell["x"]), 6) for cell in result.cells_norm})
    ys = sorted({round(float(cell["y"]), 6) for cell in result.cells_norm})
    assert len(xs) >= 14
    assert len(ys) >= 14
    widths = [float(cell["w"]) for cell in result.cells_norm]
    assert max(widths) < 0.16
    # With 50% overlap, neighbouring x origins are closer than a cell width.
    first_row = sorted((cell for cell in result.cells_norm if abs(float(cell["y"])) < 1e-9), key=lambda c: float(c["x"]))
    assert len(first_row) > 2
    assert float(first_row[1]["x"]) - float(first_row[0]["x"]) < float(first_row[0]["w"])


def test_random_noise_is_not_treated_as_structural_sharpness():
    rng = np.random.default_rng(42)
    gray = np.clip(128.0 + rng.normal(0.0, 12.0, (600, 800)), 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    result = analyze_local_sharpness(rgb, map_rgb=rgb, precision="precise")
    assert result.cells_norm
    assert all(cell["status"] == "low_texture" for cell in result.cells_norm)
    assert result.informative_cells == 0
    assert result.soft_area_pct == 0.0


def test_crisp_low_contrast_structure_is_not_mislabeled_soft_just_for_low_amplitude():
    y, x = np.indices((600, 800))
    mask = ((x // 16 + y // 16) % 2) > 0
    gray = np.where(mask, 132, 118).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    result = analyze_local_sharpness(rgb, map_rgb=rgb, precision="precise")
    informative = [cell for cell in result.cells_norm if cell["status"] != "low_texture"]
    assert informative
    assert sum(cell["status"] == "soft" for cell in informative) <= len(informative) * 0.10
    assert np.median([float(cell["score"]) for cell in informative]) >= 65.0


def test_blur_still_drops_low_contrast_structure_after_acutance_normalization():
    y, x = np.indices((600, 800))
    mask = ((x // 16 + y // 16) % 2) > 0
    gray = np.where(mask, 132, 118).astype(np.uint8)
    sharp = np.repeat(gray[..., None], 3, axis=2)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 1.0)
    a = analyze_local_sharpness(sharp)
    b = analyze_local_sharpness(blurred)
    assert a.score >= b.score + 25.0
    assert b.soft_area_pct >= 80.0
