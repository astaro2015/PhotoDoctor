from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class NoiseResult:
    score: float
    confidence: float
    sigma_luma: float
    sigma_p75: float
    classification: str
    flat_area_pct: float
    usable_tiles: int
    total_tiles: int
    cells_norm: list[dict[str, float | str]]


def analyze_noise(rgb: np.ndarray, tile_size: int = 96) -> NoiseResult:
    """Estimate fine luminance noise in weak-gradient regions.

    Uses the classic 3x3 Laplacian-like noise kernel, but only in areas whose
    *smoothed* gradient is weak. Estimates are aggregated per tile, which keeps a
    crack, eye or clothing edge from dominating the whole image. This measures
    digital/high-frequency grain, not scratches or paper texture as semantic
    defects; those belong to the surface-defect layer.
    """
    h, w = rgb.shape[:2]
    if h < 48 or w < 48:
        return NoiseResult(100.0, 0.0, 0.0, 0.0, "unknown", 0.0, 0, 0, [])

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    guide = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)

    # Absolute threshold keeps the amount of eligible area meaningful. On very
    # textured images flat_area_pct will be low instead of always being 40% by
    # construction as in the previous percentile-based MVP estimator.
    flat_mask = gradient <= 12.0
    flat_area_pct = float(100.0 * np.mean(flat_mask))
    # The guide/derivative arrays are no longer needed after the flat-region mask.
    # Releasing them before the high-pass response avoids carrying several full-size
    # float32 images into the next stage on 4K/24-MP precise analysis.
    del guide, gx, gy, gradient

    kernel = np.asarray([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)
    response = cv2.filter2D(gray, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
    np.abs(response, out=response)
    scale = math.sqrt(math.pi / 2.0) / 6.0

    tile_sigmas: list[float] = []
    cells_norm: list[dict[str, float | str]] = []
    total_tiles = 0
    for y0 in range(0, h, tile_size):
        for x0 in range(0, w, tile_size):
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            if (y1 - y0) < 24 or (x1 - x0) < 24:
                continue
            total_tiles += 1
            mask = flat_mask[y0:y1, x0:x1]
            if mask.size == 0 or float(mask.mean()) < 0.20:
                continue
            values = response[y0:y1, x0:x1][mask]
            if values.size < 256:
                continue
            # Trim only extreme high-pass outliers, normally dust, cracks or an
            # edge that slipped through the weak-gradient mask.
            cap = float(np.percentile(values, 98.0))
            trimmed = values[values <= cap]
            if trimmed.size < 128:
                continue
            tile_sigma = float(scale * float(trimmed.mean()))
            tile_sigmas.append(tile_sigma)
            if tile_sigma < 0.8:
                tile_class = "very_low"
            elif tile_sigma < 2.5:
                tile_class = "low"
            elif tile_sigma < 5.0:
                tile_class = "moderate"
            elif tile_sigma < 9.0:
                tile_class = "high"
            else:
                tile_class = "strong"
            flat_fraction = float(mask.mean())
            cell_confidence = float(np.clip(0.28 + 0.54 * min(1.0, flat_fraction / 0.55), 0.0, 0.82))
            cells_norm.append({
                "x": x0 / w, "y": y0 / h, "w": (x1 - x0) / w, "h": (y1 - y0) / h,
                "sigma_luma": tile_sigma, "flat_fraction": flat_fraction,
                "confidence": cell_confidence, "status": tile_class,
            })

    if not tile_sigmas:
        return NoiseResult(100.0, 0.22, 0.0, 0.0, "unknown", flat_area_pct, 0, total_tiles, [])

    sigmas = np.asarray(tile_sigmas, dtype=np.float32)
    sigma = float(np.median(sigmas))
    sigma_p75 = float(np.percentile(sigmas, 75.0))
    score = float(np.clip(100.0 - 6.0 * sigma, 0.0, 100.0))

    if sigma < 0.8:
        classification = "very_low"
    elif sigma < 2.5:
        classification = "low"
    elif sigma < 5.0:
        classification = "moderate"
    elif sigma < 9.0:
        classification = "high"
    else:
        classification = "strong"

    tile_support = float(np.clip(len(tile_sigmas) / max(total_tiles * 0.55, 1.0), 0.0, 1.0))
    flat_support = float(np.clip(flat_area_pct / 35.0, 0.0, 1.0))
    confidence = float(np.clip(0.30 + 0.30 * tile_support + 0.22 * flat_support, 0.0, 0.82))

    return NoiseResult(
        score=score,
        confidence=confidence,
        sigma_luma=sigma,
        sigma_p75=sigma_p75,
        classification=classification,
        flat_area_pct=flat_area_pct,
        usable_tiles=len(tile_sigmas),
        total_tiles=total_tiles,
        cells_norm=cells_norm,
    )
