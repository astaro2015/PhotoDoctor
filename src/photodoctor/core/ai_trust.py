from __future__ import annotations

from typing import Any, Mapping

from .models import MetricResult

AI_TRUST_GATE_VERSION = "0.1.0"

_POLICY_LABELS = {
    "advisory_ok": "Можно как подсказку",
    "advisory_caution": "С осторожностью",
    "manual_review": "Только вручную",
    "ignore": "Игнорировать",
    "unavailable": "Нет данных",
}


def _policy_for(status: str, confidence: float) -> tuple[str, str]:
    if status == "contradict":
        return "manual_review", "AI противоречит классическому анализу; автоматическое использование запрещено."
    if status == "agree":
        if confidence >= 0.62:
            return "advisory_ok", "AI подтверждает классику и может использоваться как дополнительная подсказка."
        return "advisory_caution", "Методы согласны, но суммарная уверенность невысока."
    if status == "refine":
        if confidence >= 0.68:
            return "advisory_caution", "AI даёт полезное уточнение, но не заменяет классический результат."
        return "ignore", "Уточнение слишком неуверенное для практического решения."
    if status == "low_confidence":
        return "ignore", "AI-результат недостаточно уверен."
    return "ignore", "Нет надёжной сопоставимой опоры для использования AI-результата."


def build_ai_trust_gate_metric(metrics: Mapping[str, MetricResult]) -> MetricResult:
    cross = metrics.get("ai_crosscheck")
    raw = cross.raw_value if cross is not None and isinstance(cross.raw_value, dict) else {}
    comparisons = raw.get("comparisons", []) if isinstance(raw, dict) else []
    items: list[dict[str, Any]] = []
    if isinstance(comparisons, list):
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                continue
            status = str(comparison.get("status", "not_comparable"))
            try:
                confidence = float(comparison.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            policy, detail = _policy_for(status, confidence)
            items.append({
                "task": str(comparison.get("task", "")),
                "model_id": str(comparison.get("model_id", "")),
                "region": str(comparison.get("region", "global")),
                "crosscheck_status": status,
                "policy": policy,
                "policy_label": _POLICY_LABELS[policy],
                "confidence": max(0.0, min(1.0, confidence)),
                "detail": detail,
            })

    counts = {key: 0 for key in _POLICY_LABELS}
    for item in items:
        counts[item["policy"]] += 1

    if not items:
        overall = "unavailable"
    elif counts["manual_review"]:
        overall = "manual_review"
    elif counts["advisory_caution"]:
        overall = "advisory_caution"
    elif counts["advisory_ok"]:
        overall = "advisory_ok"
    else:
        overall = "ignore"

    usable = [item for item in items if item["policy"] in {"advisory_ok", "advisory_caution"}]
    confidence = (
        sum(float(item["confidence"]) for item in usable) / len(usable)
        if usable else 0.0
    )
    return MetricResult(
        "ai_trust_gate",
        {
            "version": AI_TRUST_GATE_VERSION,
            "advisory_only": True,
            "changes_classical_scores": False,
            "can_trigger_automatic_edits": False,
            "overall_policy": overall,
            "overall_label": _POLICY_LABELS[overall],
            "counts": counts,
            "items": items,
        },
        None,
        float(confidence),
        "ai",
        region="ai",
        diagnostic="Предохранитель использования AI: разрешает только подсказку, требует ручной проверки при противоречии и не запускает автоматические правки",
    )
