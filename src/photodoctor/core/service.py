from __future__ import annotations

from pathlib import Path

from .analyzer import analyze_classical
from .ai_support import attach_ai_status
from .ai_runtime import execute_ai_plan
from .ai_crosscheck import build_ai_crosscheck_metric
from .surface_refinement import build_surface_refinement_metric, refine_surface_decision_plan
from .ai_trust import build_ai_trust_gate_metric
from photodoctor.ai.manager import AIModelManager
from photodoctor.ai.sr_runtime import inspect_installed_sr_model
from photodoctor.ai.restoration_runtime import inspect_all_restoration_models
from .loader import LoadedImage, load_image
from .models import AnalysisResult, MetricResult
from .validator import validate_recommendations, refresh_surface_validation_after_refinement
from .versioning import VALIDATOR_VERSION
from .surface_history import read_surface_history


def analyze_file(
    path: str | Path,
    precision: str = "normal",
    manual_face_boxes: list[dict[str, float]] | None = None,
    manual_eye_boxes: list[dict[str, float]] | None = None,
    *,
    loaded_image: LoadedImage | None = None,
) -> AnalysisResult:
    # GUI workers already decode the image for display. Reuse those exact pixels
    # instead of decoding the same file a second time before analysis.
    image = loaded_image if loaded_image is not None else load_image(path)
    result = analyze_classical(
        image, precision=precision, manual_face_boxes=manual_face_boxes, manual_eye_boxes=manual_eye_boxes
    )
    validation = validate_recommendations(
        image, result.metrics, precision=precision, defer_surface_refinement=True
    )
    confidence = (
        sum(item.confidence for item in validation) / len(validation)
        if validation else 0.0
    )
    precision_key = str(precision or "normal").strip().lower()
    validation_scale = (
        "technical_copy_4096px" if precision_key == "maximum" else
        "technical_copy_2048px" if precision_key == "precise" else
        "technical_copy_1024px"
    )
    validation_diagnostic = (
        "Максимально углублённая проверка рекомендаций на копии до 4096 px с удвоенной пространственной плотностью" if precision_key == "maximum" else
        "Углублённая проверка безопасных рекомендаций на копии до 2048 px с повторным измерением целевой метрики и побочных регрессий" if precision_key == "precise" else
        "Проверка безопасных рекомендаций на уменьшенной копии с повторным измерением целевой метрики и побочных регрессий"
    )
    result.metrics["recommendation_validation"] = MetricResult(
        "recommendation_validation",
        {
            "validator_version": VALIDATOR_VERSION,
            "items": [item.to_dict() for item in validation],
            "tested_count": len(validation),
            "accepted_count": sum(item.accepted for item in validation),
            "rejected_count": sum(not item.accepted for item in validation),
        },
        None,
        confidence,
        validation_scale,
        region="decision",
        diagnostic=validation_diagnostic,
    )
    try:
        sr_status = inspect_installed_sr_model().to_dict()
        restoration_statuses = inspect_all_restoration_models()
        all_statuses = [sr_status, *restoration_statuses.values()]
        ready_count = sum(bool(item.get("ready", False)) for item in all_statuses if isinstance(item, dict))
        result.metrics["almaz_runtime"] = MetricResult(
            "almaz_runtime",
            {
                "sr_x2": sr_status,
                "restoration": restoration_statuses,
                "ready_count": ready_count,
                "total_slots": len(all_statuses),
            },
            None,
            1.0,
            "runtime",
            region="ai_backend",
            diagnostic="Состояние локальных ALMAZ-моделей: SHA/contract/provider; это статус runtime, а не оценка качества фотографии",
        )
    except Exception as exc:
        result.metrics["almaz_runtime"] = MetricResult(
            "almaz_runtime",
            {"ready_count": 0, "total_slots": 4, "status": "error", "detail": str(exc)},
            None,
            0.0,
            "runtime",
            region="ai_backend",
            diagnostic="Не удалось прочитать состояние ALMAZ runtime; анализ фотографии продолжен без блокировки",
        )
    ai_manager = AIModelManager()
    result.metrics["ai_inference"] = execute_ai_plan(image.srgb, result.metrics, manager=ai_manager, precision=precision)
    result.metrics["surface_refinement"] = build_surface_refinement_metric(result.metrics)
    surface_history = read_surface_history(image.info.path)
    refresh_surface_validation_after_refinement(image.srgb, result.metrics, surface_history=surface_history)
    refine_surface_decision_plan(result.metrics)
    result.metrics["ai_crosscheck"] = build_ai_crosscheck_metric(result.metrics)
    result.metrics["ai_trust_gate"] = build_ai_trust_gate_metric(result.metrics)
    attach_ai_status(result, manager=ai_manager)
    return result
