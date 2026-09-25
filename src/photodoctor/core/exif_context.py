from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

import numpy as np


@dataclass(slots=True)
class ExifContext:
    available: bool
    field_count: int
    exposure_s: float | None
    aperture_f: float | None
    iso: float | None
    focal_length_mm: float | None
    exposure_compensation_ev: float | None
    flash_value: int | None
    make: str | None
    model: str | None
    lens_model: str | None
    datetime_original: str | None
    shutter_risk: str
    shutter_ratio_to_reciprocal: float | None
    iso_noise_risk: str
    confidence: float

    def to_raw(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "field_count": self.field_count,
            "exposure_s": self.exposure_s,
            "aperture_f": self.aperture_f,
            "iso": self.iso,
            "focal_length_mm": self.focal_length_mm,
            "exposure_compensation_ev": self.exposure_compensation_ev,
            "flash_value": self.flash_value,
            "make": self.make,
            "model": self.model,
            "lens_model": self.lens_model,
            "datetime_original": self.datetime_original,
            "shutter_risk": self.shutter_risk,
            "shutter_ratio_to_reciprocal": self.shutter_ratio_to_reciprocal,
            "iso_noise_risk": self.iso_noise_risk,
            "method": "exif_context_v1",
        }


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            out = float(value)
            return out if np.isfinite(out) else None
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        # Pillow rationals usually stringify as either '1/125' or '0.008'.
        out = float(Fraction(text)) if "/" in text else float(text)
    except (ValueError, ZeroDivisionError):
        return None
    return out if np.isfinite(out) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None:
        return None
    try:
        return int(round(number))
    except (TypeError, ValueError, OverflowError):
        return None


def analyze_exif_context(exif: Mapping[str, Any] | None) -> ExifContext:
    exif = exif or {}
    exposure = _number(exif.get("exposure_time"))
    aperture = _number(exif.get("aperture"))
    iso = _number(exif.get("iso"))
    focal = _number(exif.get("focal_length"))
    compensation = _number(exif.get("exposure_compensation"))
    flash = _integer(exif.get("flash"))
    make = str(exif.get("make")).strip() if exif.get("make") is not None else None
    model = str(exif.get("model")).strip() if exif.get("model") is not None else None
    lens_model = str(exif.get("lens_model")).strip() if exif.get("lens_model") is not None else None
    datetime_original = str(exif.get("datetime_original")).strip() if exif.get("datetime_original") is not None else None

    known = [exposure, aperture, iso, focal, compensation, flash]
    field_count = sum(value is not None for value in known)
    available = field_count > 0 or any(k in exif for k in ("make", "model", "lens_model", "datetime_original"))

    shutter_ratio: float | None = None
    shutter_risk = "unknown"
    if exposure is not None and exposure > 0 and focal is not None and focal > 0:
        # Ratio > 1 means slower than the classic full-frame 1/f guideline.
        # Crop factor, stabilization, support and subject motion are unknown, so this is only a hint.
        shutter_ratio = float(exposure * focal)
        if shutter_ratio < 0.55:
            shutter_risk = "low"
        elif shutter_ratio < 1.15:
            shutter_risk = "moderate"
        else:
            shutter_risk = "high"

    iso_noise_risk = "unknown"
    if iso is not None and iso > 0:
        if iso <= 250:
            iso_noise_risk = "low"
        elif iso <= 1000:
            iso_noise_risk = "moderate"
        else:
            iso_noise_risk = "high"

    if not available:
        confidence = 0.0
    else:
        evidence = 0.28
        evidence += 0.15 if exposure is not None else 0.0
        evidence += 0.15 if focal is not None else 0.0
        evidence += 0.12 if iso is not None else 0.0
        evidence += 0.06 if aperture is not None else 0.0
        evidence += 0.04 if compensation is not None else 0.0
        confidence = float(np.clip(evidence, 0.28, 0.72))

    return ExifContext(
        available=available,
        field_count=field_count,
        exposure_s=exposure,
        aperture_f=aperture,
        iso=iso,
        focal_length_mm=focal,
        exposure_compensation_ev=compensation,
        flash_value=flash,
        make=make,
        model=model,
        lens_model=lens_model,
        datetime_original=datetime_original,
        shutter_risk=shutter_risk,
        shutter_ratio_to_reciprocal=shutter_ratio,
        iso_noise_risk=iso_noise_risk,
        confidence=confidence,
    )
