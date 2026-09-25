from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photodoctor.core.image_formats import HEIF_EXTENSIONS, INPUT_FILE_DIALOG_FILTER, SUPPORTED_INPUT_EXTENSIONS
from photodoctor.core.loader import ImageLoadError, load_image
from photodoctor.core.service import analyze_file


def test_exif_orientation_is_applied(tmp_path):
    # 40x20 physical bitmap, orientation=6 means display rotated 90 CW -> 20x40.
    arr = np.zeros((20, 40, 3), dtype=np.uint8)
    arr[:, :20] = [255, 0, 0]
    img = Image.fromarray(arr, "RGB")
    exif = Image.Exif()
    exif[274] = 6
    p = tmp_path / "orientation.jpg"
    img.save(p, exif=exif, quality=95)
    loaded = load_image(p)
    assert loaded.info.exif_orientation == 6
    assert (loaded.info.width, loaded.info.height) == (20, 40)


def test_gradient_brightness_and_histogram(tmp_path):
    grad = np.linspace(0, 255, 512, dtype=np.uint8)
    arr = np.repeat(grad[None, :, None], 128, axis=0)
    arr = np.repeat(arr, 3, axis=2)
    p = tmp_path / "gradient.png"
    Image.fromarray(arr, "RGB").save(p)
    result = analyze_file(p)
    brightness_metric = result.metrics["brightness"]
    brightness = brightness_metric.raw_value
    hist = result.metrics["histogram"].raw_value["bins"]
    assert 0.25 < brightness < 0.40  # linear-light mean, not gamma-space 0.5
    assert brightness_metric.normalized_value >= 90.0  # balanced gradient must not be called underexposed
    assert len(hist) == 256
    assert abs(sum(hist) - 1.0) < 1e-5


def test_unicode_path(tmp_path):
    folder = tmp_path / "Фото тест_日本語"
    folder.mkdir()
    p = folder / "семья_№1.png"
    Image.fromarray(np.full((32, 48, 3), 100, np.uint8), "RGB").save(p)
    loaded = load_image(p)
    assert loaded.info.width == 48
    assert loaded.info.height == 32


def test_corrupt_supported_extension_is_rejected(tmp_path):
    p = tmp_path / "broken.jpg"
    p.write_bytes(b"this is definitely not jpeg")
    with pytest.raises(ImageLoadError):
        load_image(p)


def test_unsupported_extension_is_rejected(tmp_path):
    p = tmp_path / "x.gif"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8), "RGB").save(p)
    with pytest.raises(ImageLoadError):
        load_image(p)


@pytest.mark.parametrize(
    ("suffix", "pil_format"),
    [
        (".webp", "WEBP"),
        (".tif", "TIFF"),
        (".tiff", "TIFF"),
        (".bmp", "BMP"),
        (".jpe", "JPEG"),
        (".jfif", "JPEG"),
    ],
)
def test_extended_raster_inputs_are_loaded(tmp_path, suffix, pil_format):
    p = tmp_path / f"extended{suffix}"
    rgb = np.zeros((19, 31, 3), np.uint8)
    rgb[..., 0] = 30
    rgb[..., 1] = 120
    rgb[..., 2] = 210
    Image.fromarray(rgb, "RGB").save(p, format=pil_format)
    loaded = load_image(p)
    assert loaded.srgb.shape == (19, 31, 3)
    assert loaded.info.width == 31
    assert loaded.info.height == 19


def test_avif_input_is_loaded_when_pillow_has_decoder(tmp_path):
    from PIL import features

    if not features.check("avif"):
        pytest.skip("Pillow build has no AVIF decoder")
    p = tmp_path / "modern.avif"
    Image.new("RGB", (29, 17), (40, 80, 160)).save(p, format="AVIF", quality=90)
    loaded = load_image(p)
    assert loaded.srgb.shape == (17, 29, 3)
    assert loaded.info.format == "AVIF"


def test_supported_input_extensions_and_dialog_include_modern_photo_formats():
    expected = {
        ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp",
        ".heic", ".heif", ".hif", ".tif", ".tiff", ".bmp", ".avif",
    }
    assert expected == set(SUPPORTED_INPUT_EXTENSIONS)
    assert HEIF_EXTENSIONS == frozenset({".heic", ".heif", ".hif"})
    for suffix in expected:
        assert f"*{suffix}" in INPUT_FILE_DIALOG_FILTER


def test_heic_has_clear_dependency_error_when_decoder_is_unavailable(tmp_path, monkeypatch):
    import photodoctor.core.image_formats as formats

    p = tmp_path / "phone.heic"
    p.write_bytes(b"not-a-real-heic")
    monkeypatch.setattr(formats, "_register_heif_opener", None)
    monkeypatch.setattr(formats, "_heif_plugin_registered", False)
    with pytest.raises(ImageLoadError, match="pillow-heif"):
        load_image(p)


def test_transparent_png_is_safely_converted(tmp_path):
    rgba = np.zeros((20, 30, 4), np.uint8)
    rgba[..., :3] = [10, 20, 30]
    rgba[..., 3] = 120
    p = tmp_path / "alpha.png"
    Image.fromarray(rgba, "RGBA").save(p)
    loaded = load_image(p)
    assert loaded.srgb.shape == (20, 30, 3)
    assert loaded.info.mode == "RGB"


def test_16bit_png_does_not_crash(tmp_path):
    arr = np.linspace(0, 65535, 64 * 64, dtype=np.uint16).reshape(64, 64)
    p = tmp_path / "gray16.png"
    Image.fromarray(arr).save(p)
    loaded = load_image(p)
    assert loaded.srgb.dtype == np.uint8
    assert loaded.srgb.shape == (64, 64, 3)
