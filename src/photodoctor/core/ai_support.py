from __future__ import annotations

from pathlib import Path
from typing import Mapping

from photodoctor.ai.manager import AIModelManager
from photodoctor.ai.router import AI_ROUTER_VERSION, AIRouter
from photodoctor.ai.resources import resource_snapshot

from .models import AnalysisResult, MetricResult
from .versioning import AI_CATALOG_VERSION


def build_ai_status_metric(
    metrics: Mapping[str, MetricResult],
    *,
    model_dir: str | Path | None = None,
    manager: AIModelManager | None = None,
) -> MetricResult:
    manager = manager or AIModelManager(model_dir=model_dir)
    # Never feed the previous ai_status back into routing heuristics.
    base_metrics = {key: value for key, value in metrics.items() if key != "ai_status"}
    router = AIRouter(manager)
    routes = router.plan(base_metrics)
    summary = manager.summary(router.model_states())
    resources = resource_snapshot()
    requested = [route for route in routes if route.requested]
    eligible = [route for route in routes if route.eligible_to_run]
    return MetricResult(
        "ai_status",
        {
            "catalog_version": AI_CATALOG_VERSION,
            "router_version": AI_ROUTER_VERSION,
            "execution_enabled": "ai_inference" in metrics,
            "privacy": "local_only",
            "automatic_downloads": False,
            "model_dir": summary["model_dir"],
            "runtime_available": summary["runtime_available"],
            "catalog_count": summary["catalog_count"],
            "ready_count": summary["ready_count"],
            "native_count": summary.get("native_count", 0),
            "native_ready_count": summary.get("native_ready_count", 0),
            "external_count": summary.get("external_count", 0),
            "external_ready_count": summary.get("external_ready_count", 0),
            "external_installed_count": summary.get("external_installed_count", 0),
            "missing_count": summary["missing_count"],
            "blocked_count": summary["blocked_count"],
            "requested_count": len(requested),
            "eligible_count": len(eligible),
            "inference": (
                metrics["ai_inference"].raw_value
                if "ai_inference" in metrics and isinstance(metrics["ai_inference"].raw_value, dict)
                else None
            ),
            "routes": [route.to_dict() for route in routes],
            "models": summary["models"],
            "resources": resources,
        },
        None,
        1.0,
        "metadata",
        region="ai",
        diagnostic="Локальный маршрутизатор ИИ и менеджер моделей: показывает, где микро-модель полезна, доступна ли она и запускалась ли как рекомендательное уточнение",
    )


def attach_ai_status(
    result: AnalysisResult,
    *,
    model_dir: str | Path | None = None,
    manager: AIModelManager | None = None,
) -> AnalysisResult:
    result.metrics["ai_status"] = build_ai_status_metric(
        result.metrics, model_dir=model_dir, manager=manager
    )
    return result
