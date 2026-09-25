from __future__ import annotations

from statistics import median
from typing import Any, Mapping

from .models import MetricResult

AI_CROSSCHECK_VERSION = "0.1.2"

_STATUS_RU = {
    "agree": "Согласен",
    "refine": "Уточняет",
    "contradict": "Противоречит",
    "low_confidence": "Недостаточно уверенности",
    "not_comparable": "Не сравнимо",
}


def _raw(metrics: Mapping[str, MetricResult], key: str) -> dict[str, Any]:
    metric = metrics.get(key)
    return metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}


def _score(metrics: Mapping[str, MetricResult], key: str) -> float | None:
    metric = metrics.get(key)
    if metric is None or metric.normalized_value is None:
        return None
    try:
        return float(metric.normalized_value)
    except (TypeError, ValueError):
        return None


def _classical_blur(metrics: Mapping[str, MetricResult]) -> tuple[str, float]:
    metric = metrics.get("detail_loss_type")
    raw = _raw(metrics, "detail_loss_type")
    kind = str(raw.get("classification", "unknown"))
    mapping = {
        "camera_shake_like": "motion",
        "subject_motion_like": "motion",
        "motion_like": "motion",
        "local_subject_softness": "mixed",
        "defocus_like": "defocus",
        "degradation_like": "degradation",
        "mixed_or_degraded": "mixed",
        "mixed": "mixed",
        "unknown": "unknown",
    }
    return mapping.get(kind, "unknown"), float(metric.confidence if metric is not None else 0.0)


def _technical_baseline(metrics: Mapping[str, MetricResult]) -> tuple[float | None, float]:
    keys = (
        "brightness", "shadow_clipping", "highlight_clipping", "contrast",
        "jpeg_artifacts", "edge_artifacts", "posterization", "noise",
    )
    values: list[float] = []
    confs: list[float] = []
    for key in keys:
        metric = metrics.get(key)
        if metric is None or metric.normalized_value is None or metric.confidence < 0.5:
            continue
        try:
            values.append(float(metric.normalized_value))
            confs.append(float(metric.confidence))
        except (TypeError, ValueError):
            pass
    lap = _score(metrics, "laplacian")
    ten = _score(metrics, "tenengrad")
    if lap is not None and ten is not None:
        values.append((lap + ten) / 2.0)
        confs.append(min(float(metrics["laplacian"].confidence), float(metrics["tenengrad"].confidence)))
    if not values:
        return None, 0.0
    return float(median(values)), float(sum(confs) / len(confs)) if confs else 0.0


def _result_record(task: str, model_id: str, region: str, status: str, classical: str,
                   ai: str, confidence: float, detail: str) -> dict[str, Any]:
    return {
        "task": task,
        "model_id": model_id,
        "region": region,
        "status": status,
        "status_label": _STATUS_RU[status],
        "classical": classical,
        "ai": ai,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "detail": detail,
    }


def _compare_blur(metrics, model_id: str, region: str, result: dict[str, Any]):
    ai_label = str(result.get("label", "unknown"))
    ai_conf = float(result.get("confidence", 0.0) or 0.0)
    classical, classical_conf = _classical_blur(metrics)
    if ai_conf < 0.58:
        return _result_record("blur_refinement", model_id, region, "low_confidence", classical, ai_label,
                              ai_conf, "AI-уточнение типа смаза недостаточно уверенно.")
    if classical == "unknown" or classical_conf < 0.55:
        return _result_record("blur_refinement", model_id, region, "refine", classical, ai_label,
                              min(ai_conf, max(0.55, classical_conf + 0.15)), "AI уточняет слабую классическую гипотезу.")
    if classical == ai_label:
        return _result_record("blur_refinement", model_id, region, "agree", classical, ai_label,
                              min(ai_conf, classical_conf), "Оба метода указывают на один тип потери деталей.")
    if classical == "mixed" and ai_label in {"motion", "defocus", "degradation"}:
        return _result_record("blur_refinement", model_id, region, "refine", classical, ai_label,
                              min(ai_conf, 0.82), "AI предлагает более конкретный вариант для смешанного классического результата.")
    if ai_label == "mixed":
        return _result_record("blur_refinement", model_id, region, "refine", classical, ai_label,
                              min(ai_conf, classical_conf), "AI видит смешанный случай вместо более узкой классической гипотезы.")
    if ai_label == "clean":
        lap, ten = _score(metrics, "laplacian"), _score(metrics, "tenengrad")
        sharp = None if lap is None or ten is None else (lap + ten) / 2.0
        if sharp is not None and sharp >= 68:
            return _result_record("blur_refinement", model_id, region, "agree", classical, ai_label,
                                  ai_conf, "AI считает кадр чистым, и классическая резкость не указывает на выраженную потерю деталей.")
    return _result_record("blur_refinement", model_id, region, "contradict", classical, ai_label,
                          min(ai_conf, classical_conf), "Классическая и AI-гипотезы о типе потери деталей расходятся; автоматическое решение запрещено.")


def _compare_surface(metrics, model_id: str, region: str, result: dict[str, Any]):
    raw = _raw(metrics, "surface_defects")
    count = int(raw.get("candidate_count", 0) or 0)
    density = float(raw.get("candidate_density_pct", raw.get("candidate_density", 0.0)) or 0.0)

    # Native patch refiner: every region corresponds to one classical candidate.
    if "label" in result:
        label = str(result.get("label", "unknown"))
        ai_conf = float(result.get("confidence", 0.0) or 0.0)
        defect_probability = float(result.get("defect_probability", 0.0) or 0.0)
        classical = f"классический кандидат {region}; всего {count}"
        ai = f"{label}; P(defect)={defect_probability * 100:.0f}%"
        if ai_conf < 0.58:
            status, detail = "low_confidence", "Встроенный уточнитель недостаточно уверен; кандидат остаётся на ручной проверке."
        elif label == "defect":
            status, detail = "agree", "Surface AI поддерживает этот классический кандидат; итоговое решение всё равно требует согласия Context Verifier."
        elif label == "natural_detail" and ai_conf >= 0.75:
            status, detail = "contradict", "Surface AI считает кандидат вероятной естественной линией/текстурой; автоматически удалять кандидат нельзя."
        else:
            status, detail = "refine", "Surface AI не подтверждает дефект с достаточной уверенностью; кандидат остаётся на проверке контекстом/пользователем."
        return _result_record("surface_defect_refinement", model_id, region, status, classical, ai, ai_conf, detail)

    # Backward-compatible segmentation contract for external/future models.
    coverage = float(result.get("coverage_pct", 0.0) or 0.0)
    ai_conf = float(result.get("confidence", 0.0) or 0.0)
    classical = f"{count} кандид.; {density:.2f}% площади"
    ai = f"маска {coverage:.2f}%"
    if ai_conf < 0.50:
        status = "low_confidence"
        detail = "AI-маска недостаточно уверенная для проверки кандидатов поверхности."
    elif count == 0 and coverage < 0.05:
        status, detail = "agree", "Оба метода не видят заметных дефектов поверхности."
    elif count > 0 and coverage >= 0.05:
        status, detail = "agree", "AI подтверждает наличие локальных дефектов среди классических кандидатов."
    elif count > 0 and coverage < 0.02:
        status, detail = "contradict", "Классика видит кандидатов, а AI почти не подтверждает дефектную область; нужна ручная проверка."
    else:
        status, detail = "refine", "AI уточняет пространственную долю дефектов без прямого совпадения шкал."
    return _result_record("surface_defect_refinement", model_id, region, status, classical, ai, ai_conf, detail)


def _face_classical_score(metrics: Mapping[str, MetricResult], region: str) -> tuple[float | None, float]:
    metric = metrics.get("faces")
    raw = _raw(metrics, "faces")
    faces = raw.get("faces", []) if isinstance(raw, dict) else []
    if region.startswith("face_") and isinstance(faces, list):
        try:
            index = int(region.split("_", 1)[1]) - 1
            face = faces[index]
            if isinstance(face, dict):
                value = face.get("sharpness_score")
                conf = face.get("measurement_confidence", metric.confidence if metric else 0.0)
                return float(value), float(conf)
        except (ValueError, IndexError, TypeError):
            pass
    return _score(metrics, "faces"), float(metric.confidence if metric is not None else 0.0)


def _compare_scalar(task: str, metrics, model_id: str, region: str, result: dict[str, Any]):
    ai_score = float(result.get("score_0_100", 0.0) or 0.0)
    if result.get("confidence_available") is False:
        return _result_record(
            task, model_id, region, "low_confidence", "классическая оценка доступна",
            f"{ai_score:.1f}/100", 0.0,
            "Скалярный контракт v1 выдаёт оценку качества, но не калиброванную уверенность; число показывается только справочно.",
        )
    ai_conf = float(result.get("confidence", 1.0) or 0.0)
    if task == "face_quality":
        classical_score, classical_conf = _face_classical_score(metrics, region)
        label = "локальная резкость лица"
    else:
        classical_score, classical_conf = _technical_baseline(metrics)
        label = "классическая техническая базовая оценка"
    if classical_score is None:
        return _result_record(task, model_id, region, "not_comparable", "нет сопоставимой оценки",
                              f"{ai_score:.1f}/100", ai_conf, "Для этой области нет надёжной классической численной оценки.")
    diff = abs(ai_score - classical_score)
    combined = min(ai_conf, max(classical_conf, 0.5))
    if ai_conf < 0.58:
        status, detail = "low_confidence", "Оценка ИИ недостаточно уверена для содержательной сверки."
    elif classical_conf < 0.55:
        status, detail = "refine", f"AI уточняет {label} с низкой классической уверенностью."
    elif diff <= 12.0:
        status, detail = "agree", f"AI и {label} близки по оценке (разница {diff:.1f} пункта)."
    elif diff >= 25.0:
        status, detail = "contradict", f"AI и {label} расходятся на {diff:.1f} пункта; автоматическое решение запрещено."
    else:
        status, detail = "refine", f"AI заметно сдвигает {label} ({diff:.1f} пункта), но расхождение ещё не критическое."
    return _result_record(task, model_id, region, status, f"{classical_score:.1f}/100", f"{ai_score:.1f}/100", combined, detail)


def _compare_semantic(metrics, model_id: str, region: str, result: dict[str, Any]):
    metric = metrics.get("semantic_context")
    classical = str(_raw(metrics, "semantic_context").get("classification", "unknown"))
    ai_label = str(result.get("label", "unknown"))
    ai_conf = float(result.get("confidence", 0.0) or 0.0)
    classical_conf = float(metric.confidence if metric is not None else 0.0)
    if ai_conf < 0.58:
        status, detail = "low_confidence", "AI-контекст недостаточно уверен."
    elif classical == ai_label:
        status, detail = "agree", "Классический и AI-контекст совпадают."
    elif classical_conf < 0.60:
        status, detail = "refine", "ИИ уточняет контекст при низкой уверенности классической базовой оценки."
    elif classical in {"portrait", "general_photo"} and ai_label in {"group_portrait", "archival_portrait", "portrait"}:
        status, detail = "refine", "AI предлагает более конкретный совместимый контекст."
    else:
        status, detail = "contradict", "Классический и AI-контекст расходятся; менять приоритеты автоматически нельзя."
    return _result_record("semantic_context_refinement", model_id, region, status, classical, ai_label,
                          min(ai_conf, max(classical_conf, 0.5)), detail)


def build_ai_crosscheck_metric(metrics: Mapping[str, MetricResult]) -> MetricResult:
    inference = metrics.get("ai_inference")
    raw = inference.raw_value if inference is not None and isinstance(inference.raw_value, dict) else {}
    task_results = raw.get("results", []) if isinstance(raw, dict) else []
    comparisons: list[dict[str, Any]] = []
    if isinstance(task_results, list):
        for task_result in task_results:
            if not isinstance(task_result, dict) or task_result.get("status") not in {"ok", "partial_error"}:
                continue
            task = str(task_result.get("task", ""))
            model_id = str(task_result.get("model_id", ""))
            regions = task_result.get("regions", [])
            if not isinstance(regions, list):
                continue
            for region in regions:
                if not isinstance(region, dict) or region.get("status") != "ok" or not isinstance(region.get("result"), dict):
                    continue
                region_name = str(region.get("region", "global"))
                result = region["result"]
                if task == "blur_refinement":
                    item = _compare_blur(metrics, model_id, region_name, result)
                elif task == "surface_defect_refinement":
                    item = _compare_surface(metrics, model_id, region_name, result)
                elif task in {"face_quality", "iqa_refinement"}:
                    item = _compare_scalar(task, metrics, model_id, region_name, result)
                elif task == "semantic_context_refinement":
                    item = _compare_semantic(metrics, model_id, region_name, result)
                else:
                    item = _result_record(task, model_id, region_name, "not_comparable", "—", "—", 0.0,
                                          "Для этой AI-задачи ещё нет правила сверки.")
                comparisons.append(item)

    counts = {key: 0 for key in _STATUS_RU}
    for item in comparisons:
        status = str(item.get("status", "not_comparable"))
        counts[status] = counts.get(status, 0) + 1
    meaningful = [item for item in comparisons if item.get("status") != "not_comparable"]
    confidence = (
        sum(float(item.get("confidence", 0.0)) for item in meaningful) / len(meaningful)
        if meaningful else 0.0
    )
    return MetricResult(
        "ai_crosscheck",
        {
            "version": AI_CROSSCHECK_VERSION,
            "advisory_only": True,
            "changes_classical_scores": False,
            "comparison_count": len(comparisons),
            "counts": counts,
            "comparisons": comparisons,
        },
        None,
        float(confidence),
        "ai",
        region="ai",
        diagnostic="Сверка локального AI с классическим анализом: согласие, уточнение, противоречие или недостаточная уверенность; оценки не меняет",
    )
