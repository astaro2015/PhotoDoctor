from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class AnalysisPrecision:
    key: str
    label: str
    spatial_long_edge: int
    detail_long_edge: int
    deep_analysis: bool
    sharp_base_positions: int
    sharp_overlap: float
    tone_base_positions: int
    tone_overlap: float
    contrast_base_positions: int
    contrast_overlap: float
    min_positions: int
    max_positions: int


_PROFILES = {
    "fast": AnalysisPrecision(
        key="fast",
        label="Быстро",
        spatial_long_edge=1800,
        detail_long_edge=2048,
        deep_analysis=False,
        sharp_base_positions=11,
        sharp_overlap=0.42,
        tone_base_positions=9,
        tone_overlap=0.28,
        contrast_base_positions=10,
        contrast_overlap=0.32,
        min_positions=8,
        max_positions=18,
    ),
    "normal": AnalysisPrecision(
        key="normal",
        label="Нормально",
        spatial_long_edge=3072,
        detail_long_edge=2048,
        deep_analysis=False,
        sharp_base_positions=15,
        sharp_overlap=0.50,
        tone_base_positions=12,
        tone_overlap=0.35,
        contrast_base_positions=13,
        contrast_overlap=0.40,
        min_positions=10,
        max_positions=26,
    ),
    "precise": AnalysisPrecision(
        key="precise",
        label="Точно",
        spatial_long_edge=4096,
        detail_long_edge=4096,
        deep_analysis=True,
        sharp_base_positions=24,
        sharp_overlap=0.62,
        tone_base_positions=20,
        tone_overlap=0.50,
        contrast_base_positions=22,
        contrast_overlap=0.55,
        min_positions=14,
        max_positions=42,
    ),
    "maximum": AnalysisPrecision(
        key="maximum",
        label="Максимальный",
        spatial_long_edge=8192,
        detail_long_edge=8192,
        deep_analysis=True,
        # Maximum doubles the previous fourth-mode spatial workload on 4K: about
        # about 42.9k overlapping map windows on 4K vs ~21.9k in the first Maximum draft and ~10.6k in old Ideal. Stable scalar quality
        # scores remain calibrated on the same summary scale; extra work is used
        # for localisation and deep searches, not weaker quality thresholds.
        sharp_base_positions=72,
        sharp_overlap=0.78,
        tone_base_positions=61,
        tone_overlap=0.72,
        contrast_base_positions=65,
        contrast_overlap=0.74,
        min_positions=32,
        max_positions=150,
    ),
}


def get_precision(value: str | None) -> AnalysisPrecision:
    key = str(value or "normal").strip().lower()
    if key in {"ideal", "идеально"}:
        key = "maximum"
    return _PROFILES.get(key, _PROFILES["normal"])


def all_precisions() -> tuple[AnalysisPrecision, ...]:
    return tuple(_PROFILES[key] for key in ("fast", "normal", "precise", "maximum"))


def resolution_aware_positions(
    short_side: int,
    base_positions: int,
    *,
    min_positions: int,
    max_positions: int,
    reference_short_side: int = 1200,
) -> int:
    """Increase map density sub-linearly as useful image resolution grows.

    Linear growth would turn a 50 MP photo into thousands of noisy tiny tiles.
    Square-root scaling preserves the important property the user expects — a
    higher-resolution source gets more spatial samples — while keeping runtime
    bounded and leaving low-resolution scans usable.
    """
    if short_side <= 0:
        return min_positions
    ratio = max(short_side / max(reference_short_side, 1), 0.10)
    scaled = int(round(base_positions * math.sqrt(ratio)))
    return max(min_positions, min(max_positions, scaled))
