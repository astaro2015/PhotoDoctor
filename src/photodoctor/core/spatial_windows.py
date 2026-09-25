from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WindowSpec:
    window: int
    step: int
    overlap: float
    target_short_positions: int


def _axis_starts(length: int, window: int, step: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, step))
    last = length - window
    if not starts:
        starts = [0]
    if starts[-1] != last:
        starts.append(last)
    return starts


def adaptive_square_windows(
    height: int,
    width: int,
    *,
    target_short_positions: int,
    overlap: float,
    min_window: int = 40,
    max_window: int = 280,
) -> tuple[WindowSpec, list[tuple[int, int, int, int]]]:
    """Return overlapping square windows scaled to the short image side.

    ``target_short_positions`` describes roughly how many measurements should span
    the short side.  The formula accounts for overlap, so increasing overlap does
    not silently explode the physical window size.  The last window on each axis
    is always aligned to the far edge, avoiding an unanalysed strip.
    """
    if height <= 0 or width <= 0:
        return WindowSpec(0, 0, overlap, target_short_positions), []
    if target_short_positions < 2:
        raise ValueError("Число позиций по короткой стороне должно быть не меньше 2")
    if not 0.0 <= overlap < 0.90:
        raise ValueError("перекрытие должно быть в диапазоне [0, 0.90)")

    short_side = min(height, width)
    step_ratio = 1.0 - overlap
    denom = 1.0 + (target_short_positions - 1) * step_ratio
    ideal_window = int(round(short_side / max(denom, 1e-6)))
    window = max(1, min(short_side, max(min_window, min(max_window, ideal_window))))
    step = max(1, int(round(window * step_ratio)))

    ys = _axis_starts(height, window, step)
    xs = _axis_starts(width, window, step)
    windows = [(y, min(y + window, height), x, min(x + window, width)) for y in ys for x in xs]
    return WindowSpec(window, step, overlap, target_short_positions), windows
