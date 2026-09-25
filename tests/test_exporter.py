from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from photodoctor.core.exporter import ExportError, save_rgb_copy


def test_png_copy_is_lossless(tmp_path: Path):
    y, x = np.indices((50, 70))
    rgb = np.stack(((x * 3) % 256, (y * 5) % 256, ((x + y) * 7) % 256), axis=2).astype(np.uint8)
    out = save_rgb_copy(rgb, tmp_path / "copy.png")
    loaded = np.asarray(Image.open(out).convert("RGB"))
    assert np.array_equal(loaded, rgb)


def test_export_refuses_to_overwrite_source(tmp_path: Path):
    source = tmp_path / "original.png"
    rgb = np.full((20, 20, 3), 100, np.uint8)
    Image.fromarray(rgb).save(source)
    with pytest.raises(ExportError, match="не перезаписывает"):
        save_rgb_copy(rgb, source, source_path=source)


def test_missing_extension_defaults_to_png(tmp_path: Path):
    rgb = np.full((12, 16, 3), 140, np.uint8)
    out = save_rgb_copy(rgb, tmp_path / "photo_copy")
    assert out.suffix.lower() == ".png"
    assert out.is_file()





def test_export_remains_png_or_jpeg_only(tmp_path: Path):
    rgb = np.full((12, 16, 3), 140, np.uint8)
    for requested in ("copy.webp", "copy.heic", "copy.tiff", "copy.avif", "copy.bmp"):
        out = save_rgb_copy(rgb, tmp_path / requested)
        assert out.suffix.lower() == ".png"
        assert out.is_file()

def test_export_preserves_basic_exif_but_drops_orientation(tmp_path: Path):
    source = tmp_path / "source.jpg"
    rgb = np.full((24, 32, 3), 120, np.uint8)
    exif = Image.Exif()
    exif[271] = "UnitTestCam"
    exif[272] = "ModelX"
    exif[274] = 6
    Image.fromarray(rgb).save(source, format="JPEG", exif=exif)
    out = save_rgb_copy(rgb, tmp_path / "copy.jpg", source_path=source)
    loaded = Image.open(out)
    loaded_exif = loaded.getexif()
    assert loaded_exif.get(271) == "UnitTestCam"
    assert loaded_exif.get(272) == "ModelX"
    assert loaded_exif.get(274) in (None, 1)


def _inject_jpeg_segment(path: Path, marker: int, payload: bytes) -> None:
    data = path.read_bytes()
    assert data[:2] == b"\xff\xd8"
    segment = bytes((0xFF, marker)) + (len(payload) + 2).to_bytes(2, "big") + payload
    path.write_bytes(data[:2] + segment + data[2:])


def test_export_preserves_exif_datetime_gps_icc_xmp_and_does_not_add_software(tmp_path: Path):
    from PIL import ImageCms
    from photodoctor.core.exporter import audit_metadata_preservation

    source = tmp_path / "rich source.jpg"
    target = tmp_path / "rich copy.jpg"
    rgb = np.full((24, 32, 3), 122, np.uint8)
    exif = Image.Exif()
    exif[271] = "UnitTestCam"
    exif[272] = "ModelX"
    exif[274] = 6
    exif[36867] = "2026:09:18 12:34:56"
    exif[34853] = {1: "N", 2: (52.0, 13.0, 0.0), 3: "E", 4: (21.0, 0.0, 0.0)}
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"/></x:xmpmeta>'
    Image.fromarray(rgb).save(source, format="JPEG", exif=exif, icc_profile=icc, xmp=xmp)
    source_before = source.read_bytes()

    out = save_rgb_copy(rgb, target, source_path=source)
    assert source.read_bytes() == source_before
    with Image.open(out) as loaded:
        exported = loaded.getexif()
        assert exported.get(271) == "UnitTestCam"
        assert exported.get(272) == "ModelX"
        assert exported.get(36867) == "2026:09:18 12:34:56"
        assert exported.get(274) in (None, 1)
        gps = exported.get_ifd(34853)
        assert gps.get(1) == "N"
        assert gps.get(3) == "E"
        assert loaded.info.get("icc_profile") == icc
        assert loaded.info.get("xmp") == xmp
        assert exported.get(305) is None  # Software was not invented by Photo Doctor.
        assert loaded.info.get("comment") is None

    audit = audit_metadata_preservation(source, out)
    assert audit.omitted_classes == ()
    assert {"EXIF", "ICC profile", "XMP"}.issubset(set(audit.preserved_classes))


def test_metadata_audit_warns_when_photoshop_iptc_app13_is_not_preserved(tmp_path: Path):
    from photodoctor.core.exporter import audit_metadata_preservation

    source = tmp_path / "iptc source.jpg"
    target = tmp_path / "iptc copy.jpg"
    rgb = np.full((20, 24, 3), 90, np.uint8)
    Image.fromarray(rgb).save(source, format="JPEG")
    _inject_jpeg_segment(source, 0xED, b"Photoshop 3.0\x00" + b"8BIM" + b"PhotoDoctorMetadataAudit")

    out = save_rgb_copy(rgb, target, source_path=source)
    audit = audit_metadata_preservation(source, out)
    assert "IPTC/Photoshop APP13" in audit.source_classes
    assert "IPTC/Photoshop APP13" in audit.omitted_classes
    assert "IPTC/Photoshop APP13" in audit.warning_text()


def test_export_preserves_all_jpeg_comment_segments_exactly(tmp_path: Path):
    from photodoctor.core.exporter import _raw_jpeg_comments, audit_metadata_preservation

    source = tmp_path / "comment source.jpg"
    target = tmp_path / "comment copy.jpg"
    rgb = np.full((28, 36, 3), 111, np.uint8)
    Image.fromarray(rgb).save(source, format="JPEG")
    expected = (
        b"Photo Doctor source comment",
        b"second comment: \\x00\\xff raw bytes",
    )
    for payload in reversed(expected):
        _inject_jpeg_segment(source, 0xFE, payload)

    out = save_rgb_copy(rgb, target, source_path=source)
    assert _raw_jpeg_comments(source) == expected
    assert _raw_jpeg_comments(out) == expected

    audit = audit_metadata_preservation(source, out)
    assert "JPEG Comment" in audit.source_classes
    assert "JPEG Comment" in audit.preserved_classes
    assert "JPEG Comment" not in audit.omitted_classes


def test_export_updates_existing_exif_dimension_tags_after_x2(tmp_path: Path):
    source = tmp_path / "source_dimensions.jpg"
    src = np.full((24, 32, 3), 105, np.uint8)
    exif = Image.Exif()
    exif[271] = "UnitTestCam"
    exif[40962] = 32
    exif[40963] = 24
    Image.fromarray(src).save(source, format="JPEG", exif=exif)

    enlarged = np.repeat(np.repeat(src, 2, axis=0), 2, axis=1)
    out = save_rgb_copy(enlarged, tmp_path / "copy_dimensions.jpg", source_path=source)
    with Image.open(out) as loaded:
        exported = loaded.getexif()
        assert loaded.size == (64, 48)
        assert exported.get(40962) == 64
        assert exported.get(40963) == 48
        assert exported.get(271) == "UnitTestCam"


def test_export_does_not_invent_exif_dimension_tags_when_source_lacks_them(tmp_path: Path):
    source = tmp_path / "source_no_dimensions.jpg"
    src = np.full((20, 30, 3), 99, np.uint8)
    exif = Image.Exif()
    exif[271] = "UnitTestCam"
    Image.fromarray(src).save(source, format="JPEG", exif=exif)
    enlarged = np.repeat(np.repeat(src, 2, axis=0), 2, axis=1)
    out = save_rgb_copy(enlarged, tmp_path / "copy_no_dimensions.jpg", source_path=source)
    with Image.open(out) as loaded:
        exported = loaded.getexif()
        assert exported.get(40962) is None
        assert exported.get(40963) is None
        assert exported.get(271) == "UnitTestCam"
