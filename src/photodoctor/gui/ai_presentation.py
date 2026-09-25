from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from photodoctor.core.models import MetricResult
from photodoctor.gui.localization import localize_model_id, localize_payload, localize_task, localize_value


_TASK_LABELS = {
    "blur_refinement": "Тип смаза / потери деталей",
    "surface_defect_refinement": "Дефекты поверхности",
    "face_quality": "Качество лица",
    "iqa_refinement": "Дополнительная оценка качества",
    "semantic_context_refinement": "Контекст кадра",
}

_STATUS_LABELS = {
    "ok": "Готово",
    "partial_error": "Частично",
    "error": "Ошибка",
    "not_run": "Не запущено",
}


@dataclass(frozen=True, slots=True)
class AIInferenceRow:
    task: str
    model_id: str
    region: str
    result: str
    confidence: float | None
    status: str
    comparison: str = "—"
    comparison_detail: str = ""
    trust: str = "—"
    trust_detail: str = ""
    detail: str = ""


def _fmt_result(result: dict) -> tuple[str, float | None, str]:
    try:
        confidence = float(result.get("confidence")) if result.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    if result.get("confidence_available") is False:
        confidence = None

    if "label" in result:
        label = str(localize_value(result.get("label", "—")))
        probs = result.get("probabilities")
        detail = ""
        if isinstance(probs, dict) and probs:
            ranked = []
            for key, value in probs.items():
                try:
                    ranked.append((str(localize_value(key)), float(value)))
                except (TypeError, ValueError):
                    continue
            ranked.sort(key=lambda item: item[1], reverse=True)
            detail = "; ".join(f"{name}: {value * 100:.0f}%" for name, value in ranked[:3])
        return label, confidence, detail

    if "score_0_100" in result:
        try:
            score = float(result.get("score_0_100"))
            return f"{score:.1f}/100", confidence, ""
        except (TypeError, ValueError):
            pass

    if "coverage_pct" in result:
        try:
            coverage = float(result.get("coverage_pct"))
            shape = result.get("mask_shape")
            detail = f"размер маски {shape}" if isinstance(shape, list) else ""
            return f"Покрытие {coverage:.1f}%", confidence, detail
        except (TypeError, ValueError):
            pass

    # Deliberately concise fallback: technical JSON remains available in the
    # Technical tab, while the AI tab should remain readable.
    keys = [key for key in result if key not in {"model_id", "task", "contract_id", "confidence"}]
    summary = ", ".join(str(k) for k in localize_payload({key: None for key in keys[:4]}).keys()) if keys else "результат получен"
    return summary, confidence, "контракт: " + str(result.get("contract_id", "")) if result.get("contract_id") else ""


def _crosscheck_index(metrics: Mapping[str, MetricResult]) -> dict[tuple[str, str, str], dict]:
    metric = metrics.get("ai_crosscheck")
    raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
    comparisons = raw.get("comparisons", []) if isinstance(raw, dict) else []
    index: dict[tuple[str, str, str], dict] = {}
    if isinstance(comparisons, list):
        for item in comparisons:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("task", "")), str(item.get("model_id", "")), str(item.get("region", "global")))
            index[key] = item
    return index


def _trust_index(metrics: Mapping[str, MetricResult]) -> dict[tuple[str, str, str], dict]:
    metric = metrics.get("ai_trust_gate")
    raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
    items = raw.get("items", []) if isinstance(raw, dict) else []
    index: dict[tuple[str, str, str], dict] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("task", "")), str(item.get("model_id", "")), str(item.get("region", "global")))
            index[key] = item
    return index


def build_ai_inference_rows(metrics: Mapping[str, MetricResult]) -> list[AIInferenceRow]:
    metric = metrics.get("ai_inference")
    raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
    results = raw.get("results", []) if isinstance(raw, dict) else []
    if not isinstance(results, list):
        return []

    rows: list[AIInferenceRow] = []
    crosscheck = _crosscheck_index(metrics)
    trust_index = _trust_index(metrics)
    for task_result in results:
        if not isinstance(task_result, dict):
            continue
        task_key = str(task_result.get("task", ""))
        task = _TASK_LABELS.get(task_key, localize_task(task_key))
        model_id = localize_model_id(task_result.get("model_id") or "—")
        task_status = str(task_result.get("status", "not_run"))
        regions = task_result.get("regions")
        if not isinstance(regions, list) or not regions:
            rows.append(AIInferenceRow(
                task=task,
                model_id=model_id,
                region="—",
                result="—",
                confidence=None,
                status=_STATUS_LABELS.get(task_status, task_status),
                detail=str(task_result.get("reason", "")),
            ))
            continue

        for region in regions:
            if not isinstance(region, dict):
                continue
            region_name_raw = str(region.get("region", "global"))
            region_name = str(localize_value(region_name_raw))
            region_status = str(region.get("status", task_status))
            if region_status == "error":
                rows.append(AIInferenceRow(
                    task=task,
                    model_id=model_id,
                    region=region_name,
                    result="—",
                    confidence=None,
                    status="Ошибка",
                    detail=str(region.get("error", "Ошибка запуска ИИ")),
                ))
                continue
            result = region.get("result")
            if isinstance(result, dict):
                text, confidence, detail = _fmt_result(result)
            else:
                text, confidence, detail = "результат получен", None, ""
            comparison = crosscheck.get((task_key, str(task_result.get("model_id") or "—"), region_name_raw), {})
            comparison_label = str(comparison.get("status_label", "—")) if isinstance(comparison, dict) else "—"
            comparison_detail = str(comparison.get("detail", "")) if isinstance(comparison, dict) else ""
            trust = trust_index.get((task_key, str(task_result.get("model_id") or "—"), region_name_raw), {})
            trust_label = str(trust.get("policy_label", "—")) if isinstance(trust, dict) else "—"
            trust_detail = str(trust.get("detail", "")) if isinstance(trust, dict) else ""
            rows.append(AIInferenceRow(
                task=task,
                model_id=model_id,
                region=region_name,
                result=text,
                confidence=confidence,
                status="Готово" if region_status == "ok" else _STATUS_LABELS.get(region_status, region_status),
                comparison=comparison_label,
                comparison_detail=comparison_detail,
                trust=trust_label,
                trust_detail=trust_detail,
                detail=detail,
            ))
    return rows

_SIMPLE_STATE_TITLES = {
    "USED": "ИИ ИСПОЛЬЗОВАН НА ЭТОМ ФОТО",
    "READY_UNUSED": "ИИ ГОТОВ, НО НЕ ПОТРЕБОВАЛСЯ",
    "CLASSIC_ONLY": "ИИ НЕДОСТУПЕН",
    "PARTIAL_ERROR": "ИИ РАБОТАЕТ ЧАСТИЧНО",
    "ERROR": "ОШИБКА ИИ",
}

_BATCH_LABELS = {
    "hint": "Подсказка",
    "caution": "Осторожно",
    "manual": "Вручную",
    "not_used": "Не использовался",
    "error": "Ошибка",
}


@dataclass(frozen=True, slots=True)
class SimpleAIView:
    state: str
    title: str
    trust: str
    trust_detail: str
    actions: tuple[str, ...]
    batch_key: str
    batch_label: str
    used: bool


def _metric_raw(metrics: Mapping[str, MetricResult], key: str) -> dict:
    metric = metrics.get(key)
    return metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}


def _inference_has_error(inference_raw: dict) -> bool:
    results = inference_raw.get("results", [])
    if not isinstance(results, list):
        return False
    for task in results:
        if not isinstance(task, dict):
            continue
        if str(task.get("status", "")) in {"error", "partial_error"}:
            return True
        regions = task.get("regions", [])
        if isinstance(regions, list) and any(
            isinstance(region, dict) and str(region.get("status", "")) == "error"
            for region in regions
        ):
            return True
    return False


def _inference_has_not_run(inference_raw: dict) -> bool:
    results = inference_raw.get("results", [])
    return bool(isinstance(results, list) and any(
        isinstance(task, dict) and str(task.get("status", "")) == "not_run"
        for task in results
    ))


def _simple_state(metrics: Mapping[str, MetricResult]) -> str:
    status_raw = _metric_raw(metrics, "ai_status")
    inference_raw = _metric_raw(metrics, "ai_inference")
    ready_count = int(status_raw.get("ready_count", 0) or 0)
    native_ready = int(status_raw.get("native_ready_count", 0) or 0)
    requested = int(status_raw.get("requested_count", inference_raw.get("requested_count", 0)) or 0)
    successful = int(inference_raw.get("successful_inferences", 0) or 0)
    has_error = _inference_has_error(inference_raw)
    has_not_run = _inference_has_not_run(inference_raw)

    if successful > 0:
        return "PARTIAL_ERROR" if has_error else "USED"
    if has_error:
        return "PARTIAL_ERROR" if (native_ready > 0 or ready_count > 0) else "ERROR"
    if requested > 0 and has_not_run:
        return "PARTIAL_ERROR" if (native_ready > 0 or ready_count > 0) else "ERROR"
    if ready_count > 0:
        return "READY_UNUSED"
    return "CLASSIC_ONLY"


def _simple_trust(metrics: Mapping[str, MetricResult], state: str) -> tuple[str, str, str]:
    if state in {"READY_UNUSED", "CLASSIC_ONLY", "ERROR"}:
        return "НЕТ ДАННЫХ", "ИИ не дал пригодного результата для этого фото.", "not_used" if state != "ERROR" else "error"

    trust_raw = _metric_raw(metrics, "ai_trust_gate")
    policy = str(trust_raw.get("overall_policy", "unavailable"))
    if policy == "advisory_ok":
        trust = "МОЖНО УЧИТЫВАТЬ"
        batch_key = "hint"
    elif policy == "advisory_caution":
        trust = "С ОСТОРОЖНОСТЬЮ"
        batch_key = "caution"
    elif policy in {"manual_review", "ignore"}:
        trust = "ТОЛЬКО РУЧНАЯ ПРОВЕРКА"
        batch_key = "manual"
    else:
        trust = "НЕТ ДАННЫХ"
        batch_key = "caution" if state in {"USED", "PARTIAL_ERROR"} else "not_used"

    cross_raw = _metric_raw(metrics, "ai_crosscheck")
    counts = cross_raw.get("counts", {}) if isinstance(cross_raw, dict) else {}
    counts = counts if isinstance(counts, dict) else {}
    if int(counts.get("contradict", 0) or 0) > 0:
        detail = "ИИ противоречит классическому анализу"
    elif int(counts.get("refine", 0) or 0) > 0:
        detail = "ИИ уточняет результат классики"
    elif int(counts.get("agree", 0) or 0) > 0:
        detail = "Классический анализ и ИИ согласны"
    elif state == "PARTIAL_ERROR":
        detail = "Часть задач ИИ завершилась с ошибкой; доступные результаты сохранены"
    else:
        detail = "Сопоставимых данных классики и ИИ пока недостаточно"

    if state == "PARTIAL_ERROR" and batch_key == "hint":
        batch_key = "caution"
    return trust, detail, batch_key


def _simple_actions(metrics: Mapping[str, MetricResult], state: str) -> tuple[str, ...]:
    surface = _metric_raw(metrics, "surface_refinement")
    evaluated = int(surface.get("evaluated_count", 0) or 0)
    actions: list[str] = []
    if evaluated > 0:
        actions.append(f"Проверено {evaluated} кандидатов дефектов")
        actions.append(f"Контекст + ИИ подтвердили {int(surface.get('confirmed_defect_count', 0) or 0)} вероятных дефектов")
        actions.append(f"{int(surface.get('likely_natural_count', 0) or 0)} кандидатов похожи на естественные детали")
        uncertain = int(surface.get("uncertain_count", 0) or 0)
        if uncertain:
            actions.append(f"{uncertain} случаев остались неоднозначными")

    if not actions and state in {"USED", "PARTIAL_ERROR"}:
        for row in build_ai_inference_rows(metrics):
            if row.status != "Готово":
                continue
            confidence = f" · уверенность {row.confidence * 100:.0f}%" if row.confidence is not None else ""
            actions.append(f"{row.task}: {row.result}{confidence}")
            if len(actions) >= 4:
                break

    if not actions:
        if state == "READY_UNUSED":
            actions.append("ИИ не использовался: классического анализа было достаточно")
        elif state == "CLASSIC_ONLY":
            actions.append("Классический анализ выполнен без ИИ")
        elif state == "ERROR":
            actions.append("Запрошенная AI-задача не выполнена; классический анализ сохранён")
    return tuple(actions)


def build_simple_ai_view(metrics: Mapping[str, MetricResult]) -> SimpleAIView:
    """Преобразовать внутренние состояния ИИ в компактное пользовательское представление."""
    state = _simple_state(metrics)
    trust, detail, batch_key = _simple_trust(metrics, state)
    if state == "ERROR":
        batch_key = "error"
    elif state in {"READY_UNUSED", "CLASSIC_ONLY"}:
        batch_key = "not_used"
    return SimpleAIView(
        state=state,
        title=_SIMPLE_STATE_TITLES[state],
        trust=trust,
        trust_detail=detail,
        actions=_simple_actions(metrics, state),
        batch_key=batch_key,
        batch_label=_BATCH_LABELS[batch_key],
        used=int(_metric_raw(metrics, "ai_inference").get("successful_inferences", 0) or 0) > 0,
    )
