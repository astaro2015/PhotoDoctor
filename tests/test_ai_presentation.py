from photodoctor.core.models import MetricResult
from photodoctor.gui.ai_presentation import build_ai_inference_rows


def metric(raw):
    return {"ai_inference": MetricResult("ai_inference", raw, None, 1.0, "ai", region="ai")}


def test_classification_result_is_human_readable():
    rows = build_ai_inference_rows(metric({"results": [{
        "task": "blur_refinement", "model_id": "blur-v1", "status": "ok",
        "regions": [{"region": "global", "status": "ok", "result": {
            "label": "motion", "confidence": 0.82,
            "probabilities": {"motion": 0.82, "defocus": 0.12, "other": 0.06},
        }}],
    }]}))
    assert len(rows) == 1
    assert rows[0].result == "motion"
    assert rows[0].confidence == 0.82
    assert "motion: 82%" in rows[0].detail


def test_scalar_and_segmentation_results_are_condensed():
    rows = build_ai_inference_rows(metric({"results": [
        {"task": "face_quality", "model_id": "face-v1", "status": "ok", "regions": [
            {"region": "face_1", "status": "ok", "result": {"score_0_100": 71.25, "confidence": 1.0}},
        ]},
        {"task": "surface_defect_refinement", "model_id": "surface-v1", "status": "ok", "regions": [
            {"region": "global", "status": "ok", "result": {"coverage_pct": 4.6, "confidence": 0.73, "mask_shape": [256, 256]}},
        ]},
    ]}))
    assert rows[0].result == "71.2/100"
    assert rows[1].result == "Покрытие 4.6%"
    assert "маск" in rows[1].detail


def test_not_run_and_error_have_reason_without_fake_result():
    rows = build_ai_inference_rows(metric({"results": [
        {"task": "face_quality", "model_id": "face-v1", "status": "not_run", "reason": "Нет модели."},
        {"task": "iqa_refinement", "model_id": "iqa-v1", "status": "partial_error", "regions": [
            {"region": "global", "status": "error", "error": "bad tensor"},
        ]},
    ]}))
    assert rows[0].status == "Не запущено"
    assert rows[0].detail == "Нет модели."
    assert rows[1].status == "Ошибка"
    assert rows[1].detail == "bad tensor"


def test_crosscheck_is_shown_next_to_successful_inference():
    metrics = metric({"results": [{
        "task": "blur_refinement", "model_id": "blur-v1", "status": "ok",
        "regions": [{"region": "global", "status": "ok", "result": {"label": "motion", "confidence": 0.9}}],
    }]})
    metrics["ai_crosscheck"] = MetricResult("ai_crosscheck", {
        "comparisons": [{
            "task": "blur_refinement", "model_id": "blur-v1", "region": "global",
            "status": "agree", "status_label": "Согласен", "detail": "Оба метода совпали."
        }]
    }, None, 0.9, "ai", region="ai")
    rows = build_ai_inference_rows(metrics)
    assert rows[0].comparison == "Согласен"
    assert "совпали" in rows[0].comparison_detail


def test_trust_gate_is_shown_next_to_crosscheck():
    metrics = metric({"results": [{
        "task": "face_quality", "model_id": "face-v1", "status": "ok",
        "regions": [{"region": "face_1", "status": "ok", "result": {"score_0_100": 62.0, "confidence": 0.9}}],
    }]})
    metrics["ai_crosscheck"] = MetricResult("ai_crosscheck", {"comparisons": [{
        "task": "face_quality", "model_id": "face-v1", "region": "face_1",
        "status": "agree", "status_label": "Согласен", "detail": "Совпадает."
    }]}, None, 0.8, "ai")
    metrics["ai_trust_gate"] = MetricResult("ai_trust_gate", {"items": [{
        "task": "face_quality", "model_id": "face-v1", "region": "face_1",
        "policy": "advisory_ok", "policy_label": "Можно как подсказку", "detail": "Только подсказка."
    }]}, None, 0.8, "ai")
    row = build_ai_inference_rows(metrics)[0]
    assert row.trust == "Можно как подсказку"
    assert "подсказка" in row.trust_detail.lower()

from photodoctor.gui.ai_presentation import build_simple_ai_view


def m(name, raw, confidence=1.0):
    return MetricResult(name, raw, None, confidence, "ai", region="ai")


def test_simple_ai_ready_without_onnx_is_not_reported_disabled():
    metrics = {
        "ai_status": m("ai_status", {
            "ready_count": 1, "native_ready_count": 1, "requested_count": 0,
            "runtime_available": False,
        }),
        "ai_inference": m("ai_inference", {
            "requested_count": 0, "attempted_inferences": 0, "successful_inferences": 0, "results": [],
        }),
    }
    view = build_simple_ai_view(metrics)
    assert view.state == "READY_UNUSED"
    assert "ГОТОВ" in view.title
    assert view.batch_label == "Не использовался"


def test_simple_ai_used_summarizes_real_surface_work_and_trust():
    metrics = {
        "ai_status": m("ai_status", {"ready_count": 1, "native_ready_count": 1, "requested_count": 1}),
        "ai_inference": m("ai_inference", {
            "requested_count": 1, "attempted_inferences": 28, "successful_inferences": 28,
            "results": [{"task": "surface_defect_refinement", "model_id": "native_surface_refiner_v1", "status": "ok", "regions": []}],
        }),
        "surface_refinement": m("surface_refinement", {
            "evaluated_count": 28, "confirmed_defect_count": 11,
            "likely_natural_count": 9, "uncertain_count": 8,
        }),
        "ai_crosscheck": m("ai_crosscheck", {"counts": {"agree": 1, "refine": 0, "contradict": 0, "low_confidence": 0}}),
        "ai_trust_gate": m("ai_trust_gate", {"overall_policy": "advisory_ok"}),
    }
    view = build_simple_ai_view(metrics)
    assert view.state == "USED"
    assert view.trust == "МОЖНО УЧИТЫВАТЬ"
    assert any("Проверено 28" in line for line in view.actions)
    assert any("11 вероятных дефектов" in line for line in view.actions)
    assert view.batch_label == "Подсказка"


def test_simple_ai_partial_error_keeps_working_native_ai_available():
    metrics = {
        "ai_status": m("ai_status", {"ready_count": 1, "native_ready_count": 1, "requested_count": 1}),
        "ai_inference": m("ai_inference", {
            "requested_count": 1, "attempted_inferences": 1, "successful_inferences": 0,
            "results": [{"task": "face_quality", "model_id": "external", "status": "partial_error", "regions": [
                {"region": "global", "status": "error", "error": "bad tensor"}
            ]}],
        }),
    }
    view = build_simple_ai_view(metrics)
    assert view.state == "PARTIAL_ERROR"
    assert view.state != "ERROR"
    assert view.batch_label in {"Осторожно", "Вручную"}


def test_simple_ai_error_requires_failed_requested_task_without_fallback():
    metrics = {
        "ai_status": m("ai_status", {"ready_count": 0, "native_ready_count": 0, "requested_count": 1}),
        "ai_inference": m("ai_inference", {
            "requested_count": 1, "attempted_inferences": 1, "successful_inferences": 0,
            "results": [{"task": "face_quality", "model_id": "external", "status": "error", "regions": [
                {"region": "global", "status": "error", "error": "runtime failed"}
            ]}],
        }),
    }
    view = build_simple_ai_view(metrics)
    assert view.state == "ERROR"
    assert view.batch_label == "Ошибка"


def test_simple_ai_classic_only_when_no_model_and_no_ai_was_needed():
    metrics = {
        "ai_status": m("ai_status", {"ready_count": 0, "native_ready_count": 0, "requested_count": 0}),
        "ai_inference": m("ai_inference", {"requested_count": 0, "attempted_inferences": 0, "successful_inferences": 0, "results": []}),
    }
    view = build_simple_ai_view(metrics)
    assert view.state == "CLASSIC_ONLY"
    assert view.batch_label == "Не использовался"


def test_requested_unavailable_external_task_is_partial_when_native_fallback_exists():
    metrics = {
        "ai_status": m("ai_status", {"ready_count": 1, "native_ready_count": 1, "requested_count": 1}),
        "ai_inference": m("ai_inference", {
            "requested_count": 1, "attempted_inferences": 0, "successful_inferences": 0,
            "results": [{"task": "face_quality", "model_id": "external", "status": "not_run", "reason": "Модель не готова."}],
        }),
    }
    view = build_simple_ai_view(metrics)
    assert view.state == "PARTIAL_ERROR"
    assert view.used is False


def test_scalar_result_without_confidence_does_not_show_fake_100_percent():
    metrics = {
        "ai_inference": m("ai_inference", {
            "results": [{
                "task": "face_quality", "model_id": "face_quality_v1", "status": "ok",
                "regions": [{
                    "region": "face_1", "status": "ok",
                    "result": {"score_0_100": 71.0, "confidence": 0.0, "confidence_available": False},
                }],
            }]
        }),
    }
    rows = build_ai_inference_rows(metrics)
    assert rows[0].result == "71.0/100"
    assert rows[0].confidence is None
