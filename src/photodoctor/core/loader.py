from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .image_formats import HEIF_EXTENSIONS, SUPPORTED_INPUT_EXTENSIONS, ensure_heif_plugin
from .models import ImageInfo


class ImageLoadError(RuntimeError):
    pass


@dataclass(slots=True)
class LoadedImage:
    info: ImageInfo
    srgb: np.ndarray
    # Optional for backward compatibility with callers that explicitly provide a
    # native linear-light array. Normal file loading leaves it lazy so a huge
    # float32 copy does not stay resident for the whole analysis.
    linear_rgb: np.ndarray | None = None


def _exif_to_dict(img: Image.Image) -> tuple[dict[str, Any], int | None]:
    result: dict[str, Any] = {}
    orientation: int | None = None
    try:
        exif = img.getexif()
        if exif:
            orientation = exif.get(274)
            tag_map = {
                271: "make",
                272: "model",
                36867: "datetime_original",
                33434: "exposure_time",
                33437: "aperture",
                34855: "iso",
                37386: "focal_length",
                37380: "exposure_compensation",
                37385: "flash",
                42036: "lens_model",
            }
            for tag, name in tag_map.items():
                value = exif.get(tag)
                if value is not None:
                    result[name] = str(value)
    except Exception:
        pass
    return result, orientation


def _to_srgb(img: Image.Image) -> tuple[Image.Image, bool]:
    icc = img.info.get("icc_profile")
    if not icc:
        return img.convert("RGB"), False
    try:
        src = ImageCms.ImageCmsProfile(bytes(icc))
        dst = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(img, src, dst, outputMode="RGB")
        return converted, True
    except Exception:
        return img.convert("RGB"), True


_SRGB8_TO_LINEAR_LUT = (lambda x: np.where(
    x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4
).astype(np.float32))(np.arange(256, dtype=np.float32) / 255.0)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear light without recomputing the transfer curve per pixel.

    Decoded images are uint8, so there are only 256 possible channel values.  A
    256-entry LUT is numerically equivalent to evaluating the sRGB transfer curve
    tens of millions of times and avoids several full-frame temporary float arrays.
    Non-uint8 callers retain the general formula for compatibility.
    """
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return _SRGB8_TO_LINEAR_LUT[arr]
    x = np.clip(arr.astype(np.float32) / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def load_image(path: str | Path) -> LoadedImage:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ImageLoadError(f"Файл не найден: {p}")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        raise ImageLoadError(f"Неподдерживаемый формат: {p.suffix}")
    if suffix in HEIF_EXTENSIONS and not ensure_heif_plugin():
        raise ImageLoadError(
            "Для HEIC/HEIF требуется декодер pillow-heif. "
            "Переустановите Photo Doctor или выполните установку зависимостей."
        )

    try:
        with Image.open(p) as original:
            original.load()
            fmt = (original.format or p.suffix.lstrip(".")).upper()
            exif, orientation = _exif_to_dict(original)
            oriented = ImageOps.exif_transpose(original)
            srgb_img, had_icc = _to_srgb(oriented)
            arr = np.asarray(srgb_img, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Не удалось декодировать изображение: {p}") from exc

    info = ImageInfo(
        path=p,
        width=int(arr.shape[1]),
        height=int(arr.shape[0]),
        mode="RGB",
        format=fmt,
        exif_orientation=orientation,
        icc_present=had_icc,
        exif=exif,
    )
    return LoadedImage(info=info, srgb=arr, linear_rgb=None)
