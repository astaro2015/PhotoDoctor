from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .decisions import ExposureAssessment, assess_exposure
from .models import MetricResult


@dataclass(frozen=True, slots=True)
class HistogramDiagnostic:
    status: str
    headline: str
    summary: str
    shadow_clip_pct: float
    highlight_clip_pct: float
    percentiles: dict[str, float]
    exposure: ExposureAssessment


def _raw_number(metrics: Mapping[str, MetricResult], key: str, default: float = 0.0) -> float:
    metric = metrics.get(key)
    if metric is None or isinstance(metric.raw_value, dict):
        return default
    try:
        value = float(metric.raw_value)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if np.isfinite(value) else default


def _raw_dict(metrics: Mapping[str, MetricResult], key: str) -> dict[str, Any]:
    metric = metrics.get(key)
    return metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}


def build_histogram_diagnostic(metrics: Mapping[str, MetricResult]) -> HistogramDiagnostic:
    exposure = assess_exposure(metrics)
    raw = _raw_dict(metrics, "histogram")

    percentiles: dict[str, float] = {}
    for key in ("p1", "p5", "p50", "p95", "p99"):
        try:
            value = float(raw.get(key))
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(value):
            percentiles[key] = float(np.clip(value, 0.0, 1.0))

    shadow_clip = max(0.0, _raw_number(metrics, "shadow_clipping"))
    total_highlight_clip = max(0.0, _raw_number(metrics, "highlight_clipping"))
    highlight_context = _raw_dict(metrics, "highlight_context")
    try:
        unexplained = float(highlight_context.get("unexplained_clip_pct", total_highlight_clip) or 0.0)
    except (TypeError, ValueError, OverflowError):
        unexplained = total_highlight_clip
    if not np.isfinite(unexplained):
        unexplained = total_highlight_clip
    highlight_clip = max(0.0, unexplained)

    p50 = percentiles.get("p50", exposure.median_luma)
    mean = exposure.mean_luma

    if exposure.decision == "fix":
        status = "dark"
        headline = "Экспозиция: кадр заметно тёмный"
        action = "Допустим мягкий подъём средних тонов с защитой лиц и светов."
    elif exposure.decision == "review":
        status = "review"
        headline = "Экспозиция: тёмный кадр, требуется проверка"
        action = "Глобальное осветление возможно только после предпросмотра."
    elif highlight_clip >= 2.0:
        status = "highlights"
        headline = "Экспозиция: света требуют внимания"
        action = "Глобальное осветление не требуется: сначала защитить яркие области."
    elif mean is not None and mean > 0.55 and p50 > 0.50:
        status = "bright"
        headline = "Экспозиция: светлый кадр / высокий ключ"
        action = "Сам по себе светлый тон не является ошибкой; глобальное осветление не требуется."
    else:
        status = "normal"
        headline = "Экспозиция: нормальная"
        if exposure.subject_well_exposed:
            action = "Главный объект/лица освещены нормально; тёмная одежда или фон не требуют глобального осветления."
        else:
            action = "Глобальное осветление по гистограмме не требуется."

    if shadow_clip < 0.5:
        shadow_text = f"существенного провала теней нет ({shadow_clip:.2f}%)"
    elif shadow_clip < 2.0:
        shadow_text = f"небольшая доля почти чёрного ({shadow_clip:.2f}%)"
    else:
        shadow_text = f"заметная доля проваленных теней ({shadow_clip:.2f}%)"

    if highlight_clip < 0.5:
        highlight_text = f"существенного необъяснённого клиппинга светов нет ({highlight_clip:.2f}%)"
    elif highlight_clip < 2.0:
        highlight_text = f"небольшая доля выбитых светов ({highlight_clip:.2f}%)"
    else:
        highlight_text = f"заметный необъяснённый клиппинг светов ({highlight_clip:.2f}%)"

    summary = f"Тени: {shadow_text}. Света: {highlight_text}. {action}"
    return HistogramDiagnostic(
        status=status,
        headline=headline,
        summary=summary,
        shadow_clip_pct=shadow_clip,
        highlight_clip_pct=highlight_clip,
        percentiles=percentiles,
        exposure=exposure,
    )
