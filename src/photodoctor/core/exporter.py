from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from .image_formats import HEIF_EXTENSIONS, ensure_heif_plugin
from .surface_history import HISTORY_KEY, encode_surface_history, is_jpeg_history_comment, jpeg_history_comment


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MetadataPreservationAudit:
    source_classes: tuple[str, ...]
    preserved_classes: tuple[str, ...]
    omitted_classes: tuple[str, ...]
    intentional_changes: tuple[str, ...] = ("Поле ориентации EXIF удалено после физического поворота пикселей",)

    @property
    def ok(self) -> bool:
        return not self.omitted_classes

    def warning_text(self) -> str:
        if self.ok:
            return ""
        return "Не удалось перенести часть метаданных: " + ", ".join(self.omitted_classes) + "."


def _jpeg_segments(path: Path) -> list[tuple[int, bytes]]:
    """Read JPEG metadata segments before SOS without decoding image pixels."""
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return []
    out: list[tuple[int, bytes]] = []
    pos = 2
    while pos + 1 < len(data):
        if data[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            break
        marker = data[pos]
        pos += 1
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:  # Start of Scan: metadata header is finished.
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(data):
            break
        seg_len = int.from_bytes(data[pos:pos + 2], "big")
        if seg_len < 2 or pos + seg_len > len(data):
            break
        payload = data[pos + 2:pos + seg_len]
        out.append((marker, payload))
        pos += seg_len
    return out


def _raw_xmp_from_jpeg(path: Path) -> bytes | None:
    prefix = b"http://ns.adobe.com/xap/1.0/\x00"
    for marker, payload in _jpeg_segments(path):
        if marker == 0xE1 and payload.startswith(prefix):
            value = payload[len(prefix):]
            if value:
                return value
    return None


def _raw_jpeg_comments(path: Path) -> tuple[bytes, ...]:
    """Return all JPEG COM payloads exactly as stored in the source file."""
    return tuple(payload for marker, payload in _jpeg_segments(path) if marker == 0xFE)


def _replace_jpeg_comments(path: Path, comments: tuple[bytes, ...]) -> None:
    """Replace JPEG COM segments without re-encoding image pixels.

    Pillow's JPEG encoder does not preserve/write ``comment`` consistently
    across supported versions.  Metadata preservation must therefore not rely
    on that encoder keyword.  We strip any encoder-created COM segments and
    inject the source payloads directly into the JPEG header.
    """
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ExportError(f"Не удалось перенести JPEG Comment: {path}")

    out = bytearray(data[:2])
    for payload in comments:
        payload = bytes(payload)
        if len(payload) > 65533:
            raise ExportError("JPEG Comment слишком велик для одного COM-блока.")
        out += b"\xff\xfe" + (len(payload) + 2).to_bytes(2, "big") + payload

    pos = 2
    while pos < len(data):
        if data[pos] != 0xFF:
            # Before SOS a valid JPEG consists of marker segments.  If a file
            # is unusual, preserve the remainder rather than risking damage.
            out += data[pos:]
            break
        marker_start = pos
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        if pos >= len(data):
            out += data[marker_start:]
            break
        marker = data[pos]
        pos += 1

        if marker == 0xDA:  # SOS: compressed image data follows unchanged.
            out += data[marker_start:]
            break
        if marker in {0xD8, 0xD9} or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            out += data[marker_start:pos]
            if marker == 0xD9:
                break
            continue
        if pos + 2 > len(data):
            out += data[marker_start:]
            break
        seg_len = int.from_bytes(data[pos:pos + 2], "big")
        if seg_len < 2 or pos + seg_len > len(data):
            out += data[marker_start:]
            break
        seg_end = pos + seg_len
        if marker != 0xFE:  # Existing target comments are replaced by source comments.
            out += data[marker_start:seg_end]
        pos = seg_end

    path.write_bytes(bytes(out))


def _meaningful_exif_present(image: Image.Image) -> bool:
    try:
        exif = image.getexif()
    except Exception:
        return False
    return any(int(tag) != 274 for tag in exif.keys())


def _metadata_classes(path: str | Path | None) -> set[str]:
    if source := (Path(path) if path is not None else None):
        if not source.exists() or not source.is_file():
            return set()
    else:
        return set()

    classes: set[str] = set()
    if source.suffix.lower() in HEIF_EXTENSIONS and not ensure_heif_plugin():
        return classes
    try:
        with Image.open(source) as image:
            if _meaningful_exif_present(image):
                classes.add("EXIF")
            icc = image.info.get("icc_profile")
            if isinstance(icc, (bytes, bytearray)) and icc:
                classes.add("ICC profile")
            xmp = image.info.get("xmp")
            if isinstance(xmp, (bytes, bytearray, str)) and xmp:
                classes.add("XMP")
            dpi = image.info.get("dpi")
            if isinstance(dpi, tuple) and len(dpi) == 2:
                classes.add("DPI")
            text_map = getattr(image, "text", None)
            if isinstance(text_map, dict) and text_map:
                classes.add("PNG text chunks")
            if image.info.get("comment"):
                classes.add("JPEG Comment")
    except OSError:
        pass

    if source.suffix.lower() in {".jpg", ".jpeg"}:
        if _raw_xmp_from_jpeg(source):
            classes.add("XMP")
        generic_appn = False
        for marker, payload in _jpeg_segments(source):
            if marker == 0xED:  # APP13, usually Photoshop/IPTC IRB.
                if payload.startswith(b"Photoshop 3.0") or b"8BIM" in payload[:64]:
                    classes.add("IPTC/Photoshop APP13")
                else:
                    generic_appn = True
            elif 0xE1 <= marker <= 0xEF:
                known = (
                    marker == 0xE1 and (payload.startswith(b"Exif\x00\x00") or payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00"))
                ) or (marker == 0xE2 and payload.startswith(b"ICC_PROFILE\x00"))
                if not known:
                    generic_appn = True
            elif marker == 0xFE:
                classes.add("JPEG Comment")
        if generic_appn:
            classes.add("нестандартные JPEG APPn-блоки")
    return classes


def audit_metadata_preservation(
    source_path: str | Path | None,
    target_path: str | Path,
) -> MetadataPreservationAudit:
    """Compare metadata *classes* after export.

    This deliberately does not claim bit-for-bit preservation of every metadata
    container.  It verifies supported classes after re-encoding and reports any
    unsupported APPn/IPTC classes that disappeared. JPEG COM is preserved by the
    exporter directly rather than delegated to Pillow.
    """
    source = _metadata_classes(source_path)
    target = _metadata_classes(target_path)
    preserved = tuple(sorted(source & target))
    omitted = tuple(sorted(source - target))
    return MetadataPreservationAudit(
        source_classes=tuple(sorted(source)),
        preserved_classes=preserved,
        omitted_classes=omitted,
    )


def _extract_source_metadata(
    source_path: str | Path | None, *, output_size: tuple[int, int] | None = None
) -> dict[str, object]:
    if source_path is None:
        return {}
    path = Path(source_path)
    if not path.exists() or not path.is_file():
        return {}
    meta: dict[str, object] = {}
    pnginfo: PngImagePlugin.PngInfo | None = None
    if path.suffix.lower() in HEIF_EXTENSIONS and not ensure_heif_plugin():
        return {}
    try:
        with Image.open(path) as original:
            icc = original.info.get("icc_profile")
            if isinstance(icc, (bytes, bytearray)) and icc:
                meta["icc_profile"] = bytes(icc)

            # Pixels have already passed exif_transpose(), so carrying the old
            # Orientation value would rotate the exported copy a second time.
            try:
                exif = original.getexif()
                if exif:
                    if 274 in exif:
                        del exif[274]
                    if output_size is not None:
                        out_w, out_h = (int(output_size[0]), int(output_size[1]))
                        # Preserve metadata, but dimension tags that already exist
                        # must describe the pixels we are actually writing. Do not
                        # invent new EXIF fields when the source did not have them.
                        for tag, value in ((256, out_w), (257, out_h), (40962, out_w), (40963, out_h)):
                            if tag in exif:
                                exif[tag] = value
                    exif_bytes = exif.tobytes()
                    if exif_bytes:
                        meta["exif"] = exif_bytes
            except Exception:
                pass

            dpi = original.info.get("dpi")
            if isinstance(dpi, tuple) and len(dpi) == 2:
                meta["dpi"] = dpi

            xmp = original.info.get("xmp")
            if isinstance(xmp, str):
                xmp = xmp.encode("utf-8")
            if not isinstance(xmp, (bytes, bytearray)) or not xmp:
                xmp = _raw_xmp_from_jpeg(path)
            if isinstance(xmp, (bytes, bytearray)) and xmp:
                meta["xmp"] = bytes(xmp)

            # Preserve source PNG text as-is. We do not add Photo Doctor's own
            # Software/Comment fields; an existing source field is not synthetic.
            text_map = getattr(original, "text", None)
            if isinstance(text_map, dict) and text_map:
                pnginfo = PngImagePlugin.PngInfo()
                for key, value in text_map.items():
                    if not isinstance(key, str):
                        continue
                    try:
                        pnginfo.add_text(key, str(value))
                    except Exception:
                        continue
            gamma = original.info.get("gamma")
            if isinstance(gamma, (int, float)):
                meta["gamma"] = float(gamma)

            if path.suffix.lower() in {".jpg", ".jpeg"}:
                comments = _raw_jpeg_comments(path)
                if comments:
                    meta["jpeg_comments"] = comments
    except OSError:
        return {}
    if pnginfo is not None:
        meta["pnginfo"] = pnginfo
    return meta


def save_rgb_copy(
    rgb: np.ndarray,
    target: str | Path,
    *,
    source_path: str | Path | None = None,
    jpeg_quality: int = 95,
    photo_doctor_surface_history: dict[str, object] | None = None,
) -> Path:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ExportError("Ожидается изображение RGB uint8.")
    target_path = Path(target)
    if target_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        target_path = target_path.with_suffix(".png")
    if source_path is not None:
        try:
            if target_path.resolve() == Path(source_path).resolve():
                raise ExportError("Photo Doctor не перезаписывает исходный файл. Выберите имя для копии.")
        except OSError:
            pass
    if not target_path.parent.exists():
        raise ExportError(f"Папка назначения не существует: {target_path.parent}")

    image = Image.fromarray(rgb, mode="RGB")
    source_meta = _extract_source_metadata(source_path, output_size=(rgb.shape[1], rgb.shape[0]))
    save_kwargs: dict[str, object] = {}
    for key in ("icc_profile", "exif", "dpi"):
        if key in source_meta:
            save_kwargs[key] = source_meta[key]
    try:
        if target_path.suffix.lower() == ".png":
            if photo_doctor_surface_history is not None:
                pnginfo = PngImagePlugin.PngInfo()
                try:
                    with Image.open(Path(source_path)) if source_path is not None else Image.new("RGB", (1, 1)) as source_image:
                        source_text = getattr(source_image, "text", None)
                        if isinstance(source_text, dict):
                            for key, value in source_text.items():
                                if isinstance(key, str) and key != HISTORY_KEY:
                                    pnginfo.add_text(key, str(value))
                except Exception:
                    pass
                pnginfo.add_text(HISTORY_KEY, encode_surface_history(photo_doctor_surface_history))
                save_kwargs["pnginfo"] = pnginfo
            elif "pnginfo" in source_meta:
                save_kwargs["pnginfo"] = source_meta["pnginfo"]
            if "gamma" in source_meta:
                save_kwargs["gamma"] = source_meta["gamma"]
            if "xmp" in source_meta:
                save_kwargs["xmp"] = source_meta["xmp"]
            image.save(target_path, format="PNG", optimize=True, **save_kwargs)
        else:
            quality = int(max(70, min(100, jpeg_quality)))
            if "xmp" in source_meta:
                save_kwargs["xmp"] = source_meta["xmp"]
            image.save(target_path, format="JPEG", quality=quality, subsampling=0, optimize=True, **save_kwargs)
            comments = source_meta.get("jpeg_comments")
            comment_list = list(comments) if isinstance(comments, tuple) else []
            if photo_doctor_surface_history is not None:
                comment_list = [comment for comment in comment_list if not is_jpeg_history_comment(comment)]
                comment_list.append(jpeg_history_comment(photo_doctor_surface_history))
            if comment_list:
                _replace_jpeg_comments(target_path, tuple(comment_list))
    except (OSError, TypeError, ValueError) as exc:
        raise ExportError(f"Не удалось сохранить копию: {target_path}") from exc
    return target_path
