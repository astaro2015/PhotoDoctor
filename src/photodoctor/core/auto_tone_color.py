from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class AutoToneColorProfile:
    neutral_fraction: float
    neutral_median_r: float
    neutral_median_g: float
    neutral_median_b: float
    cast_strength: float
    cast_direction: str
    black_points: tuple[float, float, float]
    white_points: tuple[float, float, float]
    gammas: tuple[float, float, float]
    confidence: float
    method: str = "neutral_midtone_levels_gamma_v1"

    def to_raw(self) -> dict[str, object]:
        raw = asdict(self)
        raw["black_points"] = list(self.black_points)
        raw["white_points"] = list(self.white_points)
        raw["gammas"] = list(self.gammas)
        return raw


def _neutral_mask(rgb: np.ndarray, spread_limit: float = 18.0) -> np.ndarray:
    # Keep the uint8 source compact.  Expanding the whole RGB image to float32
    # merely to compare channel spread used tens of MiB and amplified allocator
    # pressure late in Precise analysis.
    arr = np.asarray(rgb)
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    luma = np.float32(0.2126) * r + np.float32(0.7152) * g + np.float32(0.0722) * b
    return (mx - mn <= spread_limit) & (luma >= 32.0) & (luma <= 200.0)


def _quantile_u8(values: np.ndarray, q: float) -> float:
    """Exact NumPy-linear quantile for uint8 samples using a 256-bin histogram."""
    flat = np.asarray(values, dtype=np.uint8).ravel()
    n = int(flat.size)
    if n <= 0:
        return 0.0
    hist = np.bincount(flat, minlength=256)
    cumulative = np.cumsum(hist, dtype=np.int64)
    pos = float(np.clip(q, 0.0, 1.0)) * (n - 1)
    lo_i = int(np.floor(pos)); hi_i = int(np.ceil(pos)); frac = pos - lo_i
    lo_v = int(np.searchsorted(cumulative, lo_i + 1, side="left"))
    hi_v = int(np.searchsorted(cumulative, hi_i + 1, side="left"))
    return float(lo_v + (hi_v - lo_v) * frac)


def _channel_medians_u8(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.uint8)
    if mask is None:
        return np.asarray([_quantile_u8(values[..., c], 0.5) for c in range(3)], dtype=np.float64)
    return np.asarray([_quantile_u8(values[..., c][mask], 0.5) for c in range(3)], dtype=np.float64)


def measure_neutral_midtone_cast(
    rgb: np.ndarray,
    *,
    reference_rgb: np.ndarray | None = None,
) -> tuple[float, tuple[float, float, float], float]:
    """Measure channel spread on one stable set of neutral midtone pixels.

    When ``reference_rgb`` is supplied the mask is derived from that image, so a
    before/after comparison measures the same physical pixels instead of letting
    the neutral selection drift after colour correction.
    """
    ref = reference_rgb if reference_rgb is not None else rgb
    mask = _neutral_mask(ref, 18.0)
    total = max(1, ref.shape[0] * ref.shape[1])
    if int(mask.sum()) < max(128, int(total * 0.01)):
        mask = _neutral_mask(ref, 24.0)
    if np.any(mask):
        med = _channel_medians_u8(rgb, mask)
        fraction = float(mask.mean())
    else:
        med = _channel_medians_u8(rgb)
        fraction = 0.0
    target = float(np.median(med))
    dev = (med - target) / max(target, 1.0)
    return float(np.max(np.abs(dev))), (float(med[0]), float(med[1]), float(med[2])), fraction


def analyze_auto_tone_color(rgb: np.ndarray) -> AutoToneColorProfile:
    """Estimate a conservative Photoshop-like auto tone/colour profile.

    The profile is driven by low-chroma midtones rather than whole-image RGB
    means. Whole-image means easily cancel a real cast when a scene contains
    coloured clothes/backgrounds. The transform is intentionally global and
    monotonic: no local relighting or generative colour changes are performed.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3")
    total = max(1, rgb.shape[0] * rgb.shape[1])
    mask = _neutral_mask(rgb, 18.0)
    if int(mask.sum()) < max(128, int(total * 0.01)):
        mask = _neutral_mask(rgb, 24.0)

    if np.any(mask):
        med = _channel_medians_u8(rgb, mask)
        neutral_fraction = float(mask.mean())
    else:
        med = _channel_medians_u8(rgb)
        neutral_fraction = 0.0

    target = float(np.median(med))
    denom = max(target, 1.0)
    dev = (med - target) / denom
    cast_strength = float(np.max(np.abs(dev)))
    if dev[2] - dev[0] > 0.035:
        cast_direction = "cool"
    elif dev[0] - dev[2] > 0.035:
        cast_direction = "warm"
    elif abs(dev[1]) > 0.04:
        cast_direction = "green_magenta"
    else:
        cast_direction = "neutral"

    blacks: list[float] = []
    whites: list[float] = []
    gammas: list[float] = []
    for c in range(3):
        pos = max(float(dev[c]), 0.0)
        neg = max(float(-dev[c]), 0.0)
        # Channel endpoints are slightly more aggressive for the channel that
        # dominates supposedly neutral midtones.  This reproduces the useful
        # behaviour observed in Photoshop Auto Tone/Contrast/Color without
        # hard-coding one photograph's LUT.
        q_low = float(np.clip(0.0008 + pos * 0.010 - neg * 0.001, 0.0003, 0.0025))
        q_high = float(np.clip(0.99945 - pos * 0.078 + neg * 0.002, 0.992, 0.9998))
        lo = _quantile_u8(rgb[..., c], q_low)
        hi = _quantile_u8(rgb[..., c], q_high)
        if hi - lo < 16.0:
            lo, hi = 0.0, 255.0

        x_mid = float(np.clip((med[c] - lo) / max(hi - lo, 1.0), 1e-4, 0.9999))
        t_mid = float(np.clip((target - lo) / max(hi - lo, 1.0), 1e-4, 0.9999))
        gamma = float(np.clip(np.log(t_mid) / np.log(x_mid), 0.88, 1.12))
        blacks.append(lo)
        whites.append(hi)
        gammas.append(gamma)

    coverage_conf = min(1.0, neutral_fraction / 0.12) if neutral_fraction > 0 else 0.0
    confidence = float(np.clip(0.42 + 0.48 * coverage_conf, 0.42, 0.90))
    return AutoToneColorProfile(
        neutral_fraction=neutral_fraction,
        neutral_median_r=float(med[0]),
        neutral_median_g=float(med[1]),
        neutral_median_b=float(med[2]),
        cast_strength=cast_strength,
        cast_direction=cast_direction,
        black_points=(blacks[0], blacks[1], blacks[2]),
        white_points=(whites[0], whites[1], whites[2]),
        gammas=(gammas[0], gammas[1], gammas[2]),
        confidence=confidence,
    )


def _selective_chroma_boost(rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.0:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    aa = lab[..., 1] - 128.0
    bb = lab[..., 2] - 128.0
    chroma = np.sqrt(aa * aa + bb * bb)
    # Leave near-neutral pixels alone; give already-coloured material a bounded
    # lift similar to the reference Photoshop result.
    gate = np.clip((chroma - 8.0) / 28.0, 0.0, 1.0)
    factor = 1.0 + amount * gate
    lab[..., 1] = np.clip(128.0 + aa * factor, 0.0, 255.0)
    lab[..., 2] = np.clip(128.0 + bb * factor, 0.0, 255.0)
    return cv2.cvtColor(np.rint(lab).astype(np.uint8), cv2.COLOR_LAB2RGB)


def apply_auto_tone_color(
    rgb: np.ndarray,
    strength: float = 1.0,
    profile: AutoToneColorProfile | None = None,
) -> np.ndarray:
    """Apply bounded global tone/contrast/colour correction.

    At 100% this is calibrated against a real Photoshop sequence
    Auto Tone -> Auto Contrast -> Auto Color supplied by the user.  It is not a
    claim to reproduce Adobe's proprietary algorithm exactly.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()
    profile = profile or analyze_auto_tone_color(rgb)
    x = rgb.astype(np.float32)
    full = np.empty_like(x)
    for c in range(3):
        lo = profile.black_points[c]
        hi = profile.white_points[c]
        gamma = profile.gammas[c]
        norm = np.clip((x[..., c] - lo) / max(hi - lo, 1.0), 0.0, 1.0)
        full[..., c] = 255.0 * np.power(norm, gamma)

    full_u8 = np.clip(np.rint(full), 0, 255).astype(np.uint8)

    # Preserve the scene's global average brightness: the supplied Photoshop
    # reference changed tonal separation far more than mean brightness.
    old_luma = float(np.mean(0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]))
    f = full_u8.astype(np.float32)
    new_luma = float(np.mean(0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]))
    luma_offset = float(np.clip(old_luma - new_luma, -3.0, 3.0))
    full_u8 = np.clip(np.rint(f + luma_offset), 0, 255).astype(np.uint8)

    # Photoshop reference increased separation of coloured material while the
    # neutral median stayed essentially stable. Keep this bounded and selective.
    full_u8 = _selective_chroma_boost(full_u8, 0.22)

    if strength >= 0.999:
        return full_u8
    mixed = x * (1.0 - strength) + full_u8.astype(np.float32) * strength
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
