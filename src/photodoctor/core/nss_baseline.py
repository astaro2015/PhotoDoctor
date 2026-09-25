from __future__ import annotations

from dataclasses import dataclass
from math import gamma

import cv2
import numpy as np


_ALPHA_GRID = np.arange(0.2, 10.001, 0.01, dtype=np.float64)
_GGD_RATIOS = np.array([
    (gamma(2.0 / a) ** 2) / (gamma(1.0 / a) * gamma(3.0 / a))
    for a in _ALPHA_GRID
], dtype=np.float64)


@dataclass(slots=True)
class NSSBaseline:
    features: list[float]
    feature_count: int
    confidence: float
    informative: bool
    image_std: float
    backend: str = "brisque_style_nss_features_v1"


def _estimate_alpha_from_ratio(ratio: float) -> float:
    if not np.isfinite(ratio):
        return 2.0
    idx = int(np.argmin((_GGD_RATIOS - ratio) ** 2))
    return float(_ALPHA_GRID[idx])


def _ggd_features(values: np.ndarray) -> tuple[float, float]:
    x = values.astype(np.float64, copy=False).ravel()
    second = float(np.mean(x * x))
    if second <= 1e-12:
        return 2.0, 0.0
    first = float(np.mean(np.abs(x)))
    ratio = (first * first) / second
    alpha = _estimate_alpha_from_ratio(ratio)
    return alpha, second


def _aggd_features(values: np.ndarray) -> tuple[float, float, float, float]:
    x = values.astype(np.float64, copy=False).ravel()
    left = x[x < 0]
    right = x[x > 0]
    left_std = float(np.sqrt(np.mean(left * left))) if left.size else 0.0
    right_std = float(np.sqrt(np.mean(right * right))) if right.size else 0.0
    if left_std <= 1e-9 and right_std <= 1e-9:
        return 2.0, 0.0, 0.0, 0.0

    gamma_hat = left_std / max(right_std, 1e-9)
    second = float(np.mean(x * x))
    first = float(np.mean(np.abs(x)))
    r_hat = (first * first) / max(second, 1e-12)
    denom = (gamma_hat * gamma_hat + 1.0) ** 2
    correction = ((gamma_hat ** 3 + 1.0) * (gamma_hat + 1.0)) / max(denom, 1e-12)
    r_hat_norm = r_hat * correction
    alpha = _estimate_alpha_from_ratio(r_hat_norm)

    g1 = gamma(1.0 / alpha)
    g2 = gamma(2.0 / alpha)
    g3 = gamma(3.0 / alpha)
    scale = np.sqrt(g1 / g3)
    mean_param = (right_std - left_std) * (g2 / g1) * scale
    return alpha, float(mean_param), left_std * left_std, right_std * right_std


def _mscn(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32, copy=False)
    mu = cv2.GaussianBlur(gray, (7, 7), 7.0 / 6.0, borderType=cv2.BORDER_REFLECT)
    sq = cv2.GaussianBlur(gray * gray, (7, 7), 7.0 / 6.0, borderType=cv2.BORDER_REFLECT)
    sigma = np.sqrt(np.maximum(sq - mu * mu, 0.0))
    return (gray - mu) / (sigma + (1.0 / 255.0))


def _features_one_scale(gray: np.ndarray) -> list[float]:
    coeff = _mscn(gray)
    alpha, variance = _ggd_features(coeff)
    features: list[float] = [alpha, variance]
    shifts = ((0, 1), (1, 0), (1, 1), (1, -1))
    for dy, dx in shifts:
        if dy == 0 and dx == 1:
            pair = coeff[:, :-1] * coeff[:, 1:]
        elif dy == 1 and dx == 0:
            pair = coeff[:-1, :] * coeff[1:, :]
        elif dy == 1 and dx == 1:
            pair = coeff[:-1, :-1] * coeff[1:, 1:]
        else:
            pair = coeff[1:, :-1] * coeff[:-1, 1:]
        features.extend(_aggd_features(pair))
    return [float(v) for v in features]


def analyze_nss_baseline(rgb: np.ndarray) -> NSSBaseline:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16:
        return NSSBaseline([], 0, 0.0, False, 0.0)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    image_std = float(gray.std())
    if min(gray.shape) < 32:
        return NSSBaseline([], 0, 0.15, False, image_std)

    features = _features_one_scale(gray)
    small = cv2.resize(gray, (max(16, gray.shape[1] // 2), max(16, gray.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    features.extend(_features_one_scale(small))
    finite = all(np.isfinite(v) for v in features)
    informative = bool(finite and len(features) == 36 and image_std >= 0.02)
    if not finite:
        features = [0.0 if not np.isfinite(v) else float(v) for v in features]
    texture_factor = float(np.clip((image_std - 0.01) / 0.12, 0.0, 1.0))
    size_factor = float(np.clip(min(gray.shape) / 256.0, 0.25, 1.0))
    confidence = float(np.clip(0.30 + 0.32 * texture_factor + 0.18 * size_factor, 0.15, 0.80))
    if not informative:
        confidence = min(confidence, 0.35)
    return NSSBaseline(features, len(features), confidence, informative, image_std)
