from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from photodoctor.core.models import MetricResult

from .manager import AIModelManager

AI_ROUTER_VERSION = "0.1.2"


@dataclass(slots=True)
class AIRoute:
    task: str
    priority: float
    reason: str
    requested: bool
    model_id: str | None
    model_status: str
    eligible_to_run: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _metric_score(metrics: Mapping[str, MetricResult], name: str) -> float | None:
    metric = metrics.get(name)
    if metric is None or metric.normalized_value is None:
        return None
    try:
        return float(metric.normalized_value)
    except (TypeError, ValueError):
        return None


def _raw_dict(metrics: Mapping[str, MetricResult], name: str) -> dict[str, Any]:
    metric = metrics.get(name)
    return metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}


class AIRouter:
    """Route only ambiguous/high-value cases to optional local micro-models."""

    def __init__(self, manager: AIModelManager):
        self.manager = manager
        self._state_cache: dict[str, object] = {}
        self._states_snapshot: list[object] = []

    def _route(self, task: str, priority: float, reason: str, requested: bool) -> AIRoute:
        state = self._state_cache.get(task)
        return AIRoute(
            task=task,
            priority=float(max(0.0, min(100.0, priority))),
            reason=reason,
            requested=requested,
            model_id=state.spec.model_id if state is not None else None,
            model_status=state.status if state is not None else "not_in_catalog",
            eligible_to_run=bool(requested and state is not None and state.ready),
        )

    def model_state_for_task(self, task: str):
        return self._state_cache.get(task)

    def model_states(self) -> list[object]:
        return list(self._states_snapshot)

    def plan(self, metrics: Mapping[str, MetricResult]) -> list[AIRoute]:
        # Hash/inspect the catalog only once per routing pass. Models can be large;
        # re-hashing every file for every route would defeat the point of a small
        # conditional AI cascade.
        self._state_cache = {}
        self._states_snapshot = list(self.manager.states())
        for state in self._states_snapshot:
            current = self._state_cache.get(state.spec.task)
            if current is None or (not current.ready and state.ready):
                self._state_cache[state.spec.task] = state

        routes: list[AIRoute] = []

        sharp = _metric_score(metrics, "sharpness")
        blur_metric = metrics.get("detail_loss_type")
        blur_raw = _raw_dict(metrics, "detail_loss_type")
        blur_kind = str(blur_raw.get("classification", "unknown"))
        blur_conf = float(blur_metric.confidence) if blur_metric is not None else 0.0
        blur_requested = bool(
            sharp is not None
            and sharp < 68.0
            and (blur_conf < 0.72 or blur_kind in {"mixed", "unknown", "local_subject_softness", "degradation_like"})
        )
        routes.append(self._route(
            "blur_refinement",
            85.0 if blur_requested else 20.0,
            "Классическая оценка видит потерю деталей, но тип причины недостаточно надёжен."
            if blur_requested else "Классической оценки типа потери деталей достаточно.",
            blur_requested,
        ))

        surface_metric = metrics.get("surface_defects")
        surface_raw = _raw_dict(metrics, "surface_defects")
        candidate_count = int(surface_raw.get("candidate_count", 0) or 0)
        surface_conf = float(surface_metric.confidence) if surface_metric is not None else 0.0
        surface_requested = candidate_count > 0
        routes.append(self._route(
            "surface_defect_refinement",
            min(90.0, 45.0 + candidate_count * 1.5) if surface_requested else 15.0,
            "Есть кандидаты на дефекты поверхности; встроенный Surface AI v2 проверяет каждый кандидат перед показом как вероятного дефекта."
            if surface_requested else "Кандидатов на дефекты поверхности нет.",
            surface_requested,
        ))

        faces_raw = _raw_dict(metrics, "faces")
        eyes_raw = _raw_dict(metrics, "eyes")
        face_count = int(faces_raw.get("face_count", 0) or 0)
        eye_count = int(eyes_raw.get("eye_count", 0) or 0)
        faces_metric = metrics.get("faces")
        face_conf = float(faces_metric.confidence) if faces_metric is not None else 0.0
        face_score = _metric_score(metrics, "faces")
        face_requested = face_count > 0 and (
            face_conf < 0.80
            or (face_score is not None and face_score < 65.0)
            or eye_count < max(1, face_count)
        )
        routes.append(self._route(
            "face_quality",
            88.0 if face_requested else 15.0,
            "Портрет содержит лицо с мягкой/неполной локальной диагностикой; полезно уточнение без идентификации личности."
            if face_requested else "Классического локального анализа лиц сейчас достаточно.",
            face_requested,
        ))

        nss = metrics.get("nss_baseline")
        summary_uncertainty = 0.0
        if nss is not None:
            summary_uncertainty = max(0.0, 1.0 - float(nss.confidence))
        low_conf_metrics = [
            float(metric.confidence)
            for key, metric in metrics.items()
            if key not in {"decision_plan", "recommendation_validation", "nss_baseline", "ai_status", "ai_inference"}
            and metric.normalized_value is not None
        ]
        mean_conf = sum(low_conf_metrics) / len(low_conf_metrics) if low_conf_metrics else 1.0
        iqa_requested = bool(nss is not None and (mean_conf < 0.72 or summary_uncertainty > 0.28))
        routes.append(self._route(
            "iqa_refinement",
            60.0 if iqa_requested else 10.0,
            "Несколько классических метрик имеют ограниченную уверенность; независимая локальная оценка качества может помочь."
            if iqa_requested else "Дополнительная модель оценки качества сейчас не нужна.",
            iqa_requested,
        ))

        semantic = metrics.get("semantic_context")
        semantic_raw = _raw_dict(metrics, "semantic_context")
        semantic_kind = str(semantic_raw.get("classification", "general_photo"))
        semantic_conf = float(semantic.confidence) if semantic is not None else 0.0
        semantic_requested = semantic is not None and semantic_conf < 0.70 and semantic_kind in {"general_photo", "portrait"}
        routes.append(self._route(
            "semantic_context_refinement",
            50.0 if semantic_requested else 8.0,
            "Контекст кадра неоднозначен; микро-модель может уточнить приоритеты без идентификации людей."
            if semantic_requested else "Базовой оценки контекста достаточно.",
            semantic_requested,
        ))

        return sorted(routes, key=lambda route: route.priority, reverse=True)
