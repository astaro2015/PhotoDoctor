from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class BlurTypeResult:
    classification: str
    confidence: float
    anisotropy: float
    direction_deg: float | None
    fine_to_coarse_ratio: float
    informative_patches: int
    explanation_code: str


def _angle_distance_180(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def analyze_blur_type(rgb: np.ndarray) -> BlurTypeResult:
    """Conservative blur-type heuristic.

    It intentionally reports *unknown* when the scene itself has strong oriented
    structure or there is not enough textured material. This is a diagnostic
    hint, not proof of the physical blur kernel.
    """
    h, w = rgb.shape[:2]
    if h < 96 or w < 96:
        return BlurTypeResult("unknown", 0.10, 1.0, None, 0.0, 0, "too_small")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Work on a bounded size so thresholds are reasonably stable.
    long_edge = max(h, w)
    if long_edge > 1200:
        scale = 1200.0 / long_edge
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = gray.shape

    # Multi-scale high frequency retention. Defocus and motion both reduce this,
    # but it is useful to decide whether a blur classification is warranted.
    lap_fine = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    coarse = cv2.GaussianBlur(gray, (0, 0), 2.0)
    lap_coarse = cv2.Laplacian(coarse, cv2.CV_32F, ksize=3)
    fine_energy = float(np.mean(lap_fine * lap_fine))
    coarse_energy = float(np.mean(lap_coarse * lap_coarse))
    fine_to_coarse = float(fine_energy / max(coarse_energy, 1e-8))

    patch = max(72, min(144, round(min(h, w) / 5)))
    stride = max(48, patch // 2)
    records: list[tuple[float, float, float]] = []  # anisotropy, direction, weight

    for y in range(0, max(1, h - patch + 1), stride):
        for x in range(0, max(1, w - patch + 1), stride):
            tile = gray[y:min(y + patch, h), x:min(x + patch, w)]
            if tile.shape[0] < 48 or tile.shape[1] < 48:
                continue
            gx = cv2.Sobel(tile, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(tile, cv2.CV_32F, 0, 1, ksize=3)
            mag2 = gx * gx + gy * gy
            grad_energy = float(np.mean(mag2))
            if grad_energy < 0.0012:
                continue

            # Structure tensor. The eigenvalue ratio measures directional
            # concentration; direction is the dominant gradient orientation.
            jxx = float(np.mean(gx * gx))
            jyy = float(np.mean(gy * gy))
            jxy = float(np.mean(gx * gy))
            trace = jxx + jyy
            disc = max((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0) ** 0.5
            l1 = 0.5 * (trace + disc)
            l2 = 0.5 * (trace - disc)
            anis = float(l1 / max(l2, 1e-8))
            # dominant gradient axis, modulo 180 degrees
            theta = 0.5 * np.degrees(np.arctan2(2.0 * jxy, jxx - jyy))
            theta = float(theta % 180.0)
            weight = float(np.clip(grad_energy / 0.02, 0.2, 1.0))
            records.append((anis, theta, weight))

    if len(records) < 3:
        return BlurTypeResult("unknown", 0.20, 1.0, None, fine_to_coarse, len(records), "not_enough_texture")

    anis_values = np.asarray([r[0] for r in records], dtype=np.float64)
    weights = np.asarray([r[2] for r in records], dtype=np.float64)
    median_anis = float(np.median(anis_values))

    # Only strongly anisotropic patches vote for motion. Require direction
    # agreement across several independent patches to avoid mistaking scene
    # geometry for camera motion.
    strong = [r for r in records if r[0] >= 2.2]
    direction: float | None = None
    agreement = 0.0
    if len(strong) >= 3:
        angles = np.asarray([np.deg2rad(r[1] * 2.0) for r in strong])
        sw = np.asarray([r[2] for r in strong])
        cx = float(np.sum(np.cos(angles) * sw))
        sy = float(np.sum(np.sin(angles) * sw))
        resultant = (cx * cx + sy * sy) ** 0.5 / max(float(sw.sum()), 1e-8)
        agreement = float(np.clip(resultant, 0.0, 1.0))
        mean_angle = (0.5 * np.degrees(np.arctan2(sy, cx))) % 180.0
        direction = float(mean_angle)

    strong_fraction = len(strong) / len(records)
    # Very low fine/coarse ratio means globally softened fine detail. The
    # absolute range depends on content, so use broad thresholds and cap conf.
    softness_signal = float(np.clip((18.0 - fine_to_coarse) / 14.0, 0.0, 1.0))

    if len(strong) >= 3 and strong_fraction >= 0.30 and agreement >= 0.68:
        conf = 0.42 + 0.30 * agreement + 0.18 * min(strong_fraction / 0.7, 1.0)
        # If there is abundant fine detail, orientation may simply be scene geometry.
        conf *= 0.72 + 0.28 * softness_signal
        return BlurTypeResult(
            "motion_like",
            float(np.clip(conf, 0.0, 0.86)),
            median_anis,
            direction,
            fine_to_coarse,
            len(records),
            "directional_consistency",
        )

    # Isotropic softening across textured patches suggests defocus-like blur.
    isotropic_fraction = float(np.mean(anis_values < 2.0))
    if isotropic_fraction >= 0.62 and softness_signal >= 0.28:
        conf = 0.38 + 0.30 * isotropic_fraction + 0.22 * softness_signal
        return BlurTypeResult(
            "defocus_like",
            float(np.clip(conf, 0.0, 0.82)),
            median_anis,
            None,
            fine_to_coarse,
            len(records),
            "isotropic_softening",
        )

    # Mixed blur/noise/print texture. A separate contextual layer may label
    # this as degradation for old prints, but the raw classifier stays modest.
    conf = 0.28 + 0.18 * min(len(records) / 12.0, 1.0)
    return BlurTypeResult(
        "mixed_or_degraded",
        float(np.clip(conf, 0.0, 0.52)),
        median_anis,
        direction if agreement >= 0.50 else None,
        fine_to_coarse,
        len(records),
        "mixed_signals",
    )
