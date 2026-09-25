from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from .loader import srgb_to_linear
from .models import MetricResult


MODEL_ID = "local_correction_planner_v3"
FEATURE_SCHEMA = "local_tone_contrast_sharpness_noise_wb_cells_v3"


@dataclass(frozen=True, slots=True)
class LocalCorrectionPlan:
    exposure_cells: tuple[dict[str, float | str], ...]
    contrast_cells: tuple[dict[str, float | str], ...]
    sharpness_cells: tuple[dict[str, float | str], ...]
    noise_cells: tuple[dict[str, float | str], ...]
    white_balance_cells: tuple[dict[str, float | str], ...]
    face_regions: tuple[dict[str, float | str], ...]
    eye_regions: tuple[dict[str, float | str], ...]
    exposure_confidence: float
    contrast_confidence: float
    sharpness_confidence: float
    noise_confidence: float
    white_balance_confidence: float
    white_balance_base_strength: float
    white_balance_kind: str
    white_balance_mixed_score: float
    exposure_coverage: float
    contrast_coverage: float
    sharpness_coverage: float
    noise_coverage: float
    white_balance_coverage: float

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "LocalCorrectionPlan":
        """Rehydrate an already-computed plan without rebuilding spatial consensus.

        Analyzer stores ``to_raw()`` in metrics so Validator/GUI can reuse the exact
        same plan.  This deliberately performs only bounded parsing/clamping and no
        image/map analysis.
        """
        def cells(key: str) -> tuple[dict[str, float | str], ...]:
            values = raw.get(key, []) if isinstance(raw, Mapping) else []
            if not isinstance(values, (list, tuple)):
                return ()
            out: list[dict[str, float | str]] = []
            for value in values:
                if isinstance(value, Mapping):
                    norm = _norm_cell(value)
                    if norm is not None:
                        out.append(norm)
            return tuple(out)

        return cls(
            exposure_cells=cells("exposure_cells"),
            contrast_cells=cells("contrast_cells"),
            sharpness_cells=cells("sharpness_cells"),
            noise_cells=cells("noise_cells"),
            white_balance_cells=cells("white_balance_cells"),
            face_regions=cells("face_regions"),
            eye_regions=cells("eye_regions"),
            exposure_confidence=float(np.clip(_finite(raw.get("exposure_confidence")), 0.0, 1.0)),
            contrast_confidence=float(np.clip(_finite(raw.get("contrast_confidence")), 0.0, 1.0)),
            sharpness_confidence=float(np.clip(_finite(raw.get("sharpness_confidence")), 0.0, 1.0)),
            noise_confidence=float(np.clip(_finite(raw.get("noise_confidence")), 0.0, 1.0)),
            white_balance_confidence=float(np.clip(_finite(raw.get("white_balance_confidence")), 0.0, 1.0)),
            white_balance_base_strength=float(np.clip(_finite(raw.get("white_balance_base_strength")), 0.0, 1.0)),
            white_balance_kind=str(raw.get("white_balance_kind", "uniform")),
            white_balance_mixed_score=float(np.clip(_finite(raw.get("white_balance_mixed_score")), 0.0, 1.0)),
            exposure_coverage=float(np.clip(_finite(raw.get("exposure_coverage")), 0.0, 1.0)),
            contrast_coverage=float(np.clip(_finite(raw.get("contrast_coverage")), 0.0, 1.0)),
            sharpness_coverage=float(np.clip(_finite(raw.get("sharpness_coverage")), 0.0, 1.0)),
            noise_coverage=float(np.clip(_finite(raw.get("noise_coverage")), 0.0, 1.0)),
            white_balance_coverage=float(np.clip(_finite(raw.get("white_balance_coverage")), 0.0, 1.0)),
        )

    def to_raw(self) -> dict[str, Any]:
        return {
            "model_id": MODEL_ID,
            "feature_schema": FEATURE_SCHEMA,
            "exposure_cells": [dict(x) for x in self.exposure_cells],
            "contrast_cells": [dict(x) for x in self.contrast_cells],
            "sharpness_cells": [dict(x) for x in self.sharpness_cells],
            "noise_cells": [dict(x) for x in self.noise_cells],
            "white_balance_cells": [dict(x) for x in self.white_balance_cells],
            "face_regions": [dict(x) for x in self.face_regions],
            "eye_regions": [dict(x) for x in self.eye_regions],
            "exposure_confidence": self.exposure_confidence,
            "contrast_confidence": self.contrast_confidence,
            "sharpness_confidence": self.sharpness_confidence,
            "noise_confidence": self.noise_confidence,
            "white_balance_confidence": self.white_balance_confidence,
            "white_balance_base_strength": self.white_balance_base_strength,
            "white_balance_kind": self.white_balance_kind,
            "white_balance_mixed_score": self.white_balance_mixed_score,
            "exposure_coverage": self.exposure_coverage,
            "contrast_coverage": self.contrast_coverage,
            "sharpness_coverage": self.sharpness_coverage,
            "noise_coverage": self.noise_coverage,
            "white_balance_coverage": self.white_balance_coverage,
        }


def _raw(metrics: Mapping[str, MetricResult], key: str) -> dict[str, Any]:
    metric = metrics.get(key)
    if metric is None or not isinstance(metric.raw_value, dict):
        return {}
    return metric.raw_value


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if np.isfinite(result) else default


def _norm_cell(cell: Mapping[str, Any]) -> dict[str, float | str] | None:
    x = _finite(cell.get("x")); y = _finite(cell.get("y"))
    w = _finite(cell.get("w")); h = _finite(cell.get("h"))
    if w <= 0.0 or h <= 0.0:
        return None
    x0 = float(np.clip(x, 0.0, 1.0)); y0 = float(np.clip(y, 0.0, 1.0))
    x1 = float(np.clip(x + w, x0, 1.0)); y1 = float(np.clip(y + h, y0, 1.0))
    if x1 - x0 <= 1e-4 or y1 - y0 <= 1e-4:
        return None
    out = dict(cell)
    out.update({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
    return out


def _regions(raw: Mapping[str, Any], key: str) -> tuple[dict[str, float | str], ...]:
    values = raw.get(key, []) if isinstance(raw, Mapping) else []
    if not isinstance(values, list):
        return ()
    out: list[dict[str, float | str]] = []
    for value in values:
        if not isinstance(value, Mapping):
            continue
        cell = _norm_cell(value)
        if cell is None:
            continue
        out.append({"x": cell["x"], "y": cell["y"], "w": cell["w"], "h": cell["h"]})
    return tuple(out)


def _center(cell: Mapping[str, Any]) -> tuple[float, float]:
    return (
        _finite(cell.get("x")) + 0.5 * _finite(cell.get("w")),
        _finite(cell.get("y")) + 0.5 * _finite(cell.get("h")),
    )


def _center_inside(cell: Mapping[str, Any], region: Mapping[str, Any], margin: float = 0.0) -> bool:
    cx, cy = _center(cell)
    x = _finite(region.get("x")); y = _finite(region.get("y"))
    w = _finite(region.get("w")); h = _finite(region.get("h"))
    return (x - margin * w) <= cx <= (x + w + margin * w) and (y - margin * h) <= cy <= (y + h + margin * h)


def _supported(cells: list[dict[str, float | str]], index: int, radius_scale: float = 1.10) -> bool:
    """Require several neighbouring windows before an automatic local correction exists."""
    if len(cells) <= 1:
        return False
    cell = cells[index]
    cx, cy = _center(cell)
    rw = max(_finite(cell.get("w")), 0.02)
    rh = max(_finite(cell.get("h")), 0.02)
    support = 0
    for j, other in enumerate(cells):
        if j == index:
            continue
        ox, oy = _center(other)
        if abs(ox - cx) <= rw * radius_scale and abs(oy - cy) <= rh * radius_scale:
            support += 1
            if support >= 2:
                return True
    return False




def _supported_mask(
    cells: list[dict[str, float | str]], radius_scale: float = 1.10, *, chunk_size: int = 512
) -> np.ndarray:
    """Vectorized equivalent of calling :func:`_supported` for every cell.

    The predicate is intentionally identical.  Precise maps can contain roughly
    two thousand windows; the former nested Python loop turned the consensus gate
    into millions of interpreter-level comparisons.  Chunked broadcasting keeps
    memory bounded while moving the exact same comparisons into NumPy.
    """
    count = len(cells)
    if count <= 1:
        return np.zeros(count, dtype=bool)
    centers = np.asarray([_center(cell) for cell in cells], dtype=np.float64)
    rw = np.asarray([max(_finite(cell.get("w")), 0.02) * radius_scale for cell in cells], dtype=np.float64)
    rh = np.asarray([max(_finite(cell.get("h")), 0.02) * radius_scale for cell in cells], dtype=np.float64)
    result = np.zeros(count, dtype=bool)
    for start in range(0, count, max(1, int(chunk_size))):
        stop = min(count, start + max(1, int(chunk_size)))
        dx = np.abs(centers[start:stop, 0, None] - centers[None, :, 0])
        dy = np.abs(centers[start:stop, 1, None] - centers[None, :, 1])
        nearby = (dx <= rw[start:stop, None]) & (dy <= rh[start:stop, None])
        # Every row contains itself, so at least three hits means two neighbours.
        result[start:stop] = np.count_nonzero(nearby, axis=1) >= 3
    return result


def _supported_cells(
    cells: list[dict[str, float | str]], radius_scale: float = 1.10
) -> tuple[dict[str, float | str], ...]:
    mask = _supported_mask(cells, radius_scale)
    return tuple(cell for cell, keep in zip(cells, mask.tolist()) if keep)

def _exclude_cells_inside_regions(
    cells: tuple[dict[str, float | str], ...],
    regions: tuple[dict[str, float | str], ...],
    *,
    margin: float,
    region_filter_key: str | None = None,
    region_filter_min: float = 0.0,
    chunk_size: int = 512,
) -> tuple[dict[str, float | str], ...]:
    """Vectorized equivalent of the old nested ``any(_center_inside(...))`` gate."""
    if not cells or not regions:
        return cells
    filtered_regions = [
        region for region in regions
        if region_filter_key is None or _finite(region.get(region_filter_key), 0.0) >= region_filter_min
    ]
    if not filtered_regions:
        return cells
    centers = np.asarray([_center(cell) for cell in cells], dtype=np.float64)
    boxes = np.asarray([
        [_finite(r.get("x")), _finite(r.get("y")), _finite(r.get("w")), _finite(r.get("h"))]
        for r in filtered_regions
    ], dtype=np.float64)
    x0 = boxes[:, 0] - margin * boxes[:, 2]
    y0 = boxes[:, 1] - margin * boxes[:, 3]
    x1 = boxes[:, 0] + boxes[:, 2] * (1.0 + margin)
    y1 = boxes[:, 1] + boxes[:, 3] * (1.0 + margin)
    keep = np.ones(len(cells), dtype=bool)
    step = max(1, int(chunk_size))
    for start in range(0, len(cells), step):
        stop = min(len(cells), start + step)
        cx = centers[start:stop, 0, None]
        cy = centers[start:stop, 1, None]
        inside = (cx >= x0[None, :]) & (cx <= x1[None, :]) & (cy >= y0[None, :]) & (cy <= y1[None, :])
        keep[start:stop] = ~np.any(inside, axis=1)
    return tuple(cell for cell, flag in zip(cells, keep.tolist()) if flag)


def _cell_window(height: int, width: int) -> np.ndarray:
    if height < 2 or width < 2:
        return np.ones((max(1, height), max(1, width)), dtype=np.float32)
    wy = np.hanning(max(3, height + 2))[1:-1]
    wx = np.hanning(max(3, width + 2))[1:-1]
    if wy.size != height:
        wy = cv2.resize(wy.astype(np.float32)[None, :], (height, 1), interpolation=cv2.INTER_LINEAR).ravel()
    if wx.size != width:
        wx = cv2.resize(wx.astype(np.float32)[None, :], (width, 1), interpolation=cv2.INTER_LINEAR).ravel()
    window = np.outer(wy, wx).astype(np.float32)
    return np.maximum(window, 0.04)


def rasterize_plan_field(
    cells: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    height: int,
    width: int,
    *,
    value_key: str,
    max_long_edge: int = 768,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth value field and support mask from normalized cells."""
    if height <= 0 or width <= 0 or not cells:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width), dtype=np.float32)
    scale = min(1.0, float(max_long_edge) / max(height, width))
    gh = max(32, int(round(height * scale)))
    gw = max(32, int(round(width * scale)))
    accum = np.zeros((gh, gw), dtype=np.float32)
    weight = np.zeros((gh, gw), dtype=np.float32)
    for raw_cell in cells:
        cell = _norm_cell(raw_cell)
        if cell is None:
            continue
        value = _finite(cell.get(value_key), 0.0)
        if abs(value) <= 1e-8:
            continue
        x0 = int(np.floor(_finite(cell["x"]) * gw)); y0 = int(np.floor(_finite(cell["y"]) * gh))
        x1 = int(np.ceil((_finite(cell["x"]) + _finite(cell["w"])) * gw))
        y1 = int(np.ceil((_finite(cell["y"]) + _finite(cell["h"])) * gh))
        x0 = max(0, min(gw - 1, x0)); y0 = max(0, min(gh - 1, y0))
        x1 = max(x0 + 1, min(gw, x1)); y1 = max(y0 + 1, min(gh, y1))
        win = _cell_window(y1 - y0, x1 - x0)
        accum[y0:y1, x0:x1] += win * value
        weight[y0:y1, x0:x1] += win
    field = np.divide(accum, np.maximum(weight, 1e-6), out=np.zeros_like(accum), where=weight > 1e-6)
    nonzero = weight[weight > 0]
    denom = np.percentile(nonzero, 75) if nonzero.size else 1.0
    support = np.clip(weight / np.maximum(denom, 1e-6), 0.0, 1.0)
    sigma = max(1.0, min(gh, gw) / 180.0)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma)
    support = cv2.GaussianBlur(support, (0, 0), sigmaX=sigma * 1.2, sigmaY=sigma * 1.2)
    if gh != height or gw != width:
        field = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)
        support = cv2.resize(support, (width, height), interpolation=cv2.INTER_CUBIC)
    return field.astype(np.float32), np.clip(support, 0.0, 1.0).astype(np.float32)


def rasterize_plan_fields(
    cells: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    height: int,
    width: int,
    *,
    value_keys: tuple[str, ...] | list[str],
    max_long_edge: int = 768,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Rasterize several fields while preserving single-field semantics exactly.

    Geometry and Hann windows are shared, but each value key keeps its own weight
    map because ``rasterize_plan_field`` intentionally excludes cells whose value
    for that key is zero.  The returned support is the support of the *first* key,
    matching the historical spatial-WB caller which kept the first support map.
    """
    keys = tuple(str(key) for key in value_keys)
    if height <= 0 or width <= 0 or not cells or not keys:
        return [np.zeros((height, width), dtype=np.float32) for _ in keys], np.zeros((height, width), dtype=np.float32)
    scale = min(1.0, float(max_long_edge) / max(height, width))
    gh = max(32, int(round(height * scale)))
    gw = max(32, int(round(width * scale)))
    accums = [np.zeros((gh, gw), dtype=np.float32) for _ in keys]
    weights = [np.zeros((gh, gw), dtype=np.float32) for _ in keys]
    for raw_cell in cells:
        cell = _norm_cell(raw_cell)
        if cell is None:
            continue
        values = [_finite(cell.get(key), 0.0) for key in keys]
        if not any(abs(value) > 1e-8 for value in values):
            continue
        x0 = int(np.floor(_finite(cell["x"]) * gw)); y0 = int(np.floor(_finite(cell["y"]) * gh))
        x1 = int(np.ceil((_finite(cell["x"]) + _finite(cell["w"])) * gw))
        y1 = int(np.ceil((_finite(cell["y"]) + _finite(cell["h"])) * gh))
        x0 = max(0, min(gw - 1, x0)); y0 = max(0, min(gh - 1, y0))
        x1 = max(x0 + 1, min(gw, x1)); y1 = max(y0 + 1, min(gh, y1))
        win = _cell_window(y1 - y0, x1 - x0)
        for accum, weight, value in zip(accums, weights, values):
            if abs(value) <= 1e-8:
                continue
            accum[y0:y1, x0:x1] += win * value
            weight[y0:y1, x0:x1] += win

    sigma = max(1.0, min(gh, gw) / 180.0)
    fields: list[np.ndarray] = []
    for accum, weight in zip(accums, weights):
        field = np.divide(accum, np.maximum(weight, 1e-6), out=np.zeros_like(accum), where=weight > 1e-6)
        field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if gh != height or gw != width:
            field = cv2.resize(field, (width, height), interpolation=cv2.INTER_CUBIC)
        fields.append(field.astype(np.float32))

    first_weight = weights[0]
    nonzero = first_weight[first_weight > 0]
    denom = np.percentile(nonzero, 75) if nonzero.size else 1.0
    support = np.clip(first_weight / np.maximum(denom, 1e-6), 0.0, 1.0)
    support = cv2.GaussianBlur(support, (0, 0), sigmaX=sigma * 1.2, sigmaY=sigma * 1.2)
    if gh != height or gw != width:
        support = cv2.resize(support, (width, height), interpolation=cv2.INTER_CUBIC)
    return fields, np.clip(support, 0.0, 1.0).astype(np.float32)


def _coverage(cells: tuple[dict[str, float | str], ...], value_key: str | None = None) -> float:
    if not cells:
        return 0.0
    key = value_key or ("delta_gamma" if "delta_gamma" in cells[0] else "strength")
    _field, support = rasterize_plan_field(cells, 160, 160, value_key=key, max_long_edge=160)
    return float(np.mean(support >= 0.18))


def _semantic_protection(
    plan: Mapping[str, Any], height: int, width: int, *, face_factor: float, eye_factor: float
) -> np.ndarray:
    """Build a soft multiplicative protection map from face/eye boxes."""
    guard = np.ones((height, width), dtype=np.float32)
    specs = (("face_regions", face_factor), ("eye_regions", eye_factor))
    for key, factor in specs:
        regions = plan.get(key) if isinstance(plan, Mapping) else None
        if not isinstance(regions, (list, tuple)):
            continue
        for raw in regions:
            if not isinstance(raw, Mapping):
                continue
            cell = _norm_cell(raw)
            if cell is None:
                continue
            x0 = max(0, min(width - 1, int(np.floor(_finite(cell["x"]) * width))))
            y0 = max(0, min(height - 1, int(np.floor(_finite(cell["y"]) * height))))
            x1 = max(x0 + 1, min(width, int(np.ceil((_finite(cell["x"]) + _finite(cell["w"])) * width))))
            y1 = max(y0 + 1, min(height, int(np.ceil((_finite(cell["y"]) + _finite(cell["h"])) * height))))
            region = np.zeros((height, width), dtype=np.float32)
            region[y0:y1, x0:x1] = 1.0
            feather = max(1.0, min(x1 - x0, y1 - y0) * 0.10)
            region = cv2.GaussianBlur(region, (0, 0), sigmaX=feather, sigmaY=feather)
            guard *= 1.0 - region * (1.0 - float(np.clip(factor, 0.0, 1.0)))
    return np.clip(guard, 0.0, 1.0)


def build_local_correction_plan(metrics: Mapping[str, MetricResult]) -> LocalCorrectionPlan:
    tone_raw = _raw(metrics, "local_tone")
    contrast_raw = _raw(metrics, "local_contrast")
    sharp_raw = _raw(metrics, "local_sharpness")
    noise_raw = _raw(metrics, "noise")
    wb_local_raw = _raw(metrics, "local_white_balance")
    wb_global_raw = _raw(metrics, "white_balance_advisor")

    tone_candidates: list[dict[str, float | str]] = []
    for item in tone_raw.get("cells_norm", []) if isinstance(tone_raw.get("cells_norm"), list) else []:
        if not isinstance(item, Mapping):
            continue
        cell = _norm_cell(item)
        if cell is None:
            continue
        status = str(cell.get("status", ""))
        p25 = _finite(cell.get("p25")); p50 = _finite(cell.get("p50")); p75 = _finite(cell.get("p75")); p95 = _finite(cell.get("p95"))
        mean = _finite(cell.get("mean")); shadow_clip = _finite(cell.get("shadow_clip")); highlight_clip = _finite(cell.get("highlight_clip"))
        delta = 0.0
        target = p50
        kind = ""
        if status in {"dark", "deep_shadow"} and p95 >= 0.18 and p75 >= 0.075 and highlight_clip <= 0.015:
            target = 0.16 if status == "dark" else 0.105
            deficit = max(0.0, target - p50)
            if deficit >= 0.018 and shadow_clip <= 0.35:
                delta = -float(np.clip(0.055 + deficit * 0.65, 0.055, 0.17))
                kind = "lift"
        elif status == "bright" and highlight_clip <= 0.025 and p25 >= 0.22:
            target = 0.64
            excess = max(0.0, p50 - target)
            if excess >= 0.035:
                delta = float(np.clip(0.035 + excess * 0.30, 0.035, 0.085))
                kind = "lower"
        if not kind:
            continue
        tone_candidates.append({
            "x": cell["x"], "y": cell["y"], "w": cell["w"], "h": cell["h"],
            "delta_gamma": delta, "target_luma": target, "kind": kind,
            "source_status": status, "source_p50": p50, "source_mean": mean,
            "shadow_clip": shadow_clip, "highlight_clip": highlight_clip,
        })
    exposure_cells = _supported_cells(tone_candidates)

    contrast_candidates: list[dict[str, float | str]] = []
    for item in contrast_raw.get("cells_norm", []) if isinstance(contrast_raw.get("cells_norm"), list) else []:
        if not isinstance(item, Mapping):
            continue
        cell = _norm_cell(item)
        if cell is None or str(cell.get("status", "")) != "low":
            continue
        score = _finite(cell.get("score")); confidence = _finite(cell.get("confidence")); local_range = _finite(cell.get("range")); median = _finite(cell.get("median"), 0.5)
        if confidence < 0.42 or local_range < 0.05 or median <= 0.035 or median >= 0.94:
            continue
        strength = float(np.clip(0.15 + (38.0 - min(score, 38.0)) / 38.0 * 0.38, 0.15, 0.53))
        contrast_candidates.append({
            "x": cell["x"], "y": cell["y"], "w": cell["w"], "h": cell["h"],
            "strength": strength, "source_score": score, "source_range": local_range,
            "source_confidence": confidence, "source_median": median,
        })
    contrast_cells = _supported_cells(contrast_candidates)

    sharp_candidates: list[dict[str, float | str]] = []
    for item in sharp_raw.get("cells_norm", []) if isinstance(sharp_raw.get("cells_norm"), list) else []:
        if not isinstance(item, Mapping):
            continue
        cell = _norm_cell(item)
        if cell is None or str(cell.get("status", "")) != "soft":
            continue
        score = _finite(cell.get("score")); confidence = _finite(cell.get("confidence"))
        if confidence < 0.50 or score >= 48.0:
            continue
        strength = float(np.clip(0.18 + (48.0 - max(0.0, score)) / 48.0 * 0.30, 0.18, 0.48))
        sharp_candidates.append({
            "x": cell["x"], "y": cell["y"], "w": cell["w"], "h": cell["h"],
            "strength": strength, "source_score": score, "source_confidence": confidence,
        })
    sharpness_cells = _supported_cells(sharp_candidates)

    noise_candidates: list[dict[str, float | str]] = []
    for item in noise_raw.get("cells_norm", []) if isinstance(noise_raw.get("cells_norm"), list) else []:
        if not isinstance(item, Mapping):
            continue
        cell = _norm_cell(item)
        if cell is None:
            continue
        sigma = _finite(cell.get("sigma_luma")); confidence = _finite(cell.get("confidence")); flat_fraction = _finite(cell.get("flat_fraction"))
        status = str(cell.get("status", ""))
        if status not in {"moderate", "high", "strong"} or sigma < 2.7 or confidence < 0.48 or flat_fraction < 0.24:
            continue
        strength = float(np.clip(0.16 + (sigma - 2.5) / 8.0 * 0.44, 0.16, 0.60))
        noise_candidates.append({
            "x": cell["x"], "y": cell["y"], "w": cell["w"], "h": cell["h"],
            "strength": strength, "source_sigma": sigma, "source_confidence": confidence,
            "flat_fraction": flat_fraction, "source_status": status,
        })
    noise_cells = _supported_cells(noise_candidates, radius_scale=1.15)

    # Do not sharpen or boost local contrast where the independent noise estimator
    # already marked genuine grain.  Otherwise one stochastic signal can create
    # mutually hostile actions: denoise plus sharpen/CLAHE on the same pixels.
    if noise_cells and sharpness_cells:
        sharpness_cells = _exclude_cells_inside_regions(
            sharpness_cells, noise_cells, margin=0.65, region_filter_key="source_sigma", region_filter_min=3.0
        )
    if noise_cells and contrast_cells:
        contrast_cells = _exclude_cells_inside_regions(
            contrast_cells, noise_cells, margin=0.65, region_filter_key="source_sigma", region_filter_min=3.0
        )

    wb_candidates: list[dict[str, float | str]] = []
    for item in wb_local_raw.get("cells_norm", []) if isinstance(wb_local_raw.get("cells_norm"), list) else []:
        if not isinstance(item, Mapping):
            continue
        cell = _norm_cell(item)
        if cell is None:
            continue
        confidence = _finite(cell.get("confidence")); neutral_fraction = _finite(cell.get("neutral_fraction"))
        cast_strength = _finite(cell.get("cast_strength")); temp = _finite(cell.get("temperature_shift_k")); tint = _finite(cell.get("tint_shift"))
        if confidence < 0.46 or neutral_fraction < 0.010:
            continue
        if cast_strength < 0.035 and abs(temp) < 220.0 and abs(tint) < 2.0:
            continue
        local_strength = float(np.clip(0.24 + 0.42 * confidence + 0.16 * cast_strength, 0.28, 0.76))
        wb_candidates.append({
            "x": cell["x"], "y": cell["y"], "w": cell["w"], "h": cell["h"],
            "target_log_r": _finite(cell.get("log_gain_r")) * local_strength,
            "target_log_g": _finite(cell.get("log_gain_g")) * local_strength,
            "target_log_b": _finite(cell.get("log_gain_b")) * local_strength,
            "temperature_shift_k": temp, "tint_shift": tint,
            "source_confidence": confidence, "neutral_fraction": neutral_fraction,
            "source_cast_strength": cast_strength, "local_strength": local_strength,
        })
    white_balance_cells = _supported_cells(wb_candidates, radius_scale=1.18)

    wb_kind = str(wb_local_raw.get("classification", "uncertain") or "uncertain")
    wb_mixed_score = float(np.clip(_finite(wb_local_raw.get("mixed_light_score")), 0.0, 1.0))
    global_neutralization = float(np.clip(_finite(wb_global_raw.get("neutralization_strength")), 0.0, 1.0))
    if wb_kind == "mixed_light":
        white_balance_base_strength = min(global_neutralization, 0.34)
    elif wb_kind == "local_cast":
        white_balance_base_strength = min(global_neutralization, 0.48)
    else:
        white_balance_base_strength = min(global_neutralization, 0.62)

    exposure_coverage = _coverage(exposure_cells, "delta_gamma")
    contrast_coverage = _coverage(contrast_cells, "strength")
    sharpness_coverage = _coverage(sharpness_cells, "strength")
    noise_coverage = _coverage(noise_cells, "strength")
    white_balance_coverage = _coverage(white_balance_cells, "local_strength") if white_balance_cells else 0.0

    tone_conf = _finite(getattr(metrics.get("local_tone"), "confidence", 0.0), 0.0)
    contrast_conf = _finite(getattr(metrics.get("local_contrast"), "confidence", 0.0), 0.0)
    sharp_conf = _finite(getattr(metrics.get("local_sharpness"), "confidence", 0.0), 0.0)
    noise_conf = _finite(getattr(metrics.get("noise"), "confidence", 0.0), 0.0)
    wb_conf = _finite(getattr(metrics.get("local_white_balance"), "confidence", 0.0), 0.0)
    local_contrast_support = float(np.mean([_finite(c.get("source_confidence"), 0.0) for c in contrast_cells])) if contrast_cells else 0.0
    local_sharp_support = float(np.mean([_finite(c.get("source_confidence"), 0.0) for c in sharpness_cells])) if sharpness_cells else 0.0
    local_noise_support = float(np.mean([_finite(c.get("source_confidence"), 0.0) for c in noise_cells])) if noise_cells else 0.0
    local_wb_support = float(np.mean([_finite(c.get("source_confidence"), 0.0) for c in white_balance_cells])) if white_balance_cells else 0.0

    exposure_confidence = float(np.clip(tone_conf * min(1.0, len(exposure_cells) / 5.0) * (1.0 if exposure_coverage <= 0.55 else 0.65), 0.0, 0.93))
    contrast_base = max(contrast_conf * 0.75, local_contrast_support * 0.92)
    contrast_confidence = float(np.clip(contrast_base * min(1.0, len(contrast_cells) / 5.0) * (1.0 if contrast_coverage <= 0.62 else 0.70), 0.0, 0.93))
    sharp_base = max(sharp_conf * 0.72, local_sharp_support * 0.95)
    sharpness_confidence = float(np.clip(sharp_base * min(1.0, len(sharpness_cells) / 5.0) * (1.0 if sharpness_coverage <= 0.58 else 0.62), 0.0, 0.92))
    noise_base = max(noise_conf * 0.78, local_noise_support * 0.96)
    noise_confidence = float(np.clip(noise_base * min(1.0, len(noise_cells) / 5.0) * (1.0 if noise_coverage <= 0.65 else 0.66), 0.0, 0.92))
    wb_base = max(wb_conf * 0.78, local_wb_support * 0.94)
    white_balance_confidence = float(np.clip(wb_base * min(1.0, len(white_balance_cells) / 4.0) * (1.0 if 0.08 <= white_balance_coverage <= 0.88 else 0.72), 0.0, 0.92))

    return LocalCorrectionPlan(
        exposure_cells=exposure_cells,
        contrast_cells=contrast_cells,
        sharpness_cells=sharpness_cells,
        noise_cells=noise_cells,
        white_balance_cells=white_balance_cells,
        face_regions=_regions(_raw(metrics, "faces"), "faces"),
        eye_regions=_regions(_raw(metrics, "eyes"), "eyes"),
        exposure_confidence=exposure_confidence,
        contrast_confidence=contrast_confidence,
        sharpness_confidence=sharpness_confidence,
        noise_confidence=noise_confidence,
        white_balance_confidence=white_balance_confidence,
        white_balance_base_strength=white_balance_base_strength,
        white_balance_kind=wb_kind,
        white_balance_mixed_score=wb_mixed_score,
        exposure_coverage=exposure_coverage,
        contrast_coverage=contrast_coverage,
        sharpness_coverage=sharpness_coverage,
        noise_coverage=noise_coverage,
        white_balance_coverage=white_balance_coverage,
    )


def apply_local_exposure(rgb: np.ndarray, plan: Mapping[str, Any], strength: float = 1.0) -> np.ndarray:
    cells = plan.get("exposure_cells") if isinstance(plan, Mapping) else None
    if not isinstance(cells, (list, tuple)) or not cells:
        return rgb.copy()
    h, w = rgb.shape[:2]
    delta, support = rasterize_plan_field(cells, h, w, value_key="delta_gamma")
    strength = float(np.clip(strength, 0.0, 1.0))
    gamma = np.clip(1.0 + delta * support * strength, 0.78, 1.10)
    x = np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)
    corrected = np.power(x, gamma[..., None])
    linear = srgb_to_linear(rgb)
    luma = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    highlight_guard = np.clip((0.97 - luma) / 0.13, 0.0, 1.0)[..., None]
    mix = np.clip(support[..., None] * highlight_guard, 0.0, 1.0)
    out = x * (1.0 - mix) + corrected * mix
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def apply_local_contrast(rgb: np.ndarray, plan: Mapping[str, Any], strength: float = 1.0) -> np.ndarray:
    cells = plan.get("contrast_cells") if isinstance(plan, Mapping) else None
    if not isinstance(cells, (list, tuple)) or not cells:
        return rgb.copy()
    h, w = rgb.shape[:2]
    local_strength, support = rasterize_plan_field(cells, h, w, value_key="strength")
    strength = float(np.clip(strength, 0.0, 1.0))
    mask = np.clip(local_strength * support * strength / 0.53, 0.0, 1.0)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.30, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    clahe_rgb = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB).astype(np.float32)
    alpha = np.clip(0.68 * mask, 0.0, 0.68)[..., None]
    base = rgb.astype(np.float32)
    out = base * (1.0 - alpha) + clahe_rgb * alpha
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def apply_local_sharpness(rgb: np.ndarray, plan: Mapping[str, Any], strength: float = 1.0) -> np.ndarray:
    """Sharpen only consensus-soft regions while protecting noise and semantic detail."""
    cells = plan.get("sharpness_cells") if isinstance(plan, Mapping) else None
    if not isinstance(cells, (list, tuple)) or not cells:
        return rgb.copy()
    h, w = rgb.shape[:2]
    local_strength, support = rasterize_plan_field(cells, h, w, value_key="strength")
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32)
    blur = cv2.GaussianBlur(lf, (0, 0), sigmaX=1.0)
    detail = lf - blur
    guide = cv2.GaussianBlur(lf, (0, 0), sigmaX=1.15)
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    structured_edge = np.clip((cv2.magnitude(gx, gy) - 2.0) / 18.0, 0.0, 1.0)
    detail_gate = np.clip((np.abs(detail) - 0.8) / 5.5, 0.0, 1.0)
    # Real edges survive the smoothed-gradient test; isolated high-frequency noise mostly does not.
    texture_guard = np.clip(0.20 + 0.80 * structured_edge, 0.0, 1.0)
    semantic_guard = _semantic_protection(plan, h, w, face_factor=0.62, eye_factor=0.78)
    mask = np.clip(local_strength * support * strength / 0.48, 0.0, 1.0)
    amount = np.clip(0.58 * mask * detail_gate * texture_guard * semantic_guard, 0.0, 0.58)
    safe_detail = np.clip(detail, -10.0, 10.0)
    l2 = np.clip(lf + amount * safe_detail, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)


def apply_local_denoise(rgb: np.ndarray, plan: Mapping[str, Any], strength: float = 1.0) -> np.ndarray:
    """Denoise only consensus-noisy low-texture regions with edge/face/eye protection."""
    cells = plan.get("noise_cells") if isinstance(plan, Mapping) else None
    if not isinstance(cells, (list, tuple)) or not cells:
        return rgb.copy()
    h, w = rgb.shape[:2]
    local_strength, support = rasterize_plan_field(cells, h, w, value_key="strength")
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return rgb.copy()

    # One conservative filtered candidate is computed, then mixed only where the
    # spatial plan and texture guard agree. This avoids NLM on unrelated regions.
    filtered = cv2.fastNlMeansDenoisingColored(rgb, None, 4.6, 4.0, 7, 21)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    guide = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    gx = cv2.Sobel(guide, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(guide, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    texture_guard = np.clip((20.0 - gradient) / 16.0, 0.0, 1.0)
    semantic_guard = _semantic_protection(plan, h, w, face_factor=0.58, eye_factor=0.25)
    mask = np.clip(local_strength * support * strength / 0.60, 0.0, 1.0)
    alpha = np.clip(0.78 * mask * texture_guard * semantic_guard, 0.0, 0.78)[..., None]
    base = rgb.astype(np.float32)
    out = base * (1.0 - alpha) + filtered.astype(np.float32) * alpha
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
