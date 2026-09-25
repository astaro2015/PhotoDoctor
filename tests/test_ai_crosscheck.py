from photodoctor.core.ai_crosscheck import build_ai_crosscheck_metric
from photodoctor.core.models import MetricResult


def m(name, raw, score=None, confidence=0.8):
    return MetricResult(name, raw, score, confidence, "test")


def inference(task, result, region="global", model_id="model"):
    return m("ai_inference", {"results": [{"task": task, "model_id": model_id, "status": "ok", "regions": [{"region": region, "status": "ok", "result": result}]}]}, None, 1.0)


def test_blur_agreement():
    metrics = {
        "detail_loss_type": m("detail_loss_type", {"classification": "motion_like"}, None, 0.8),
        "laplacian": m("laplacian", 1, 40, 0.8), "tenengrad": m("tenengrad", 1, 45, 0.8),
        "ai_inference": inference("blur_refinement", {"label": "motion", "confidence": 0.9}),
    }
    raw = build_ai_crosscheck_metric(metrics).raw_value
    assert raw["comparisons"][0]["status"] == "agree"


def test_blur_contradiction():
    metrics = {
        "detail_loss_type": m("detail_loss_type", {"classification": "motion_like"}, None, 0.82),
        "laplacian": m("laplacian", 1, 40, 0.8), "tenengrad": m("tenengrad", 1, 45, 0.8),
        "ai_inference": inference("blur_refinement", {"label": "defocus", "confidence": 0.91}),
    }
    item = build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]
    assert item["status"] == "contradict"
    assert "запрещено" in item["detail"].lower()


def test_mixed_blur_is_refined_by_specific_ai():
    metrics = {
        "detail_loss_type": m("detail_loss_type", {"classification": "mixed_or_degraded"}, None, 0.7),
        "ai_inference": inference("blur_refinement", {"label": "degradation", "confidence": 0.85}),
    }
    assert build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]["status"] == "refine"


def test_low_ai_confidence_does_not_claim_agreement():
    metrics = {
        "detail_loss_type": m("detail_loss_type", {"classification": "motion_like"}, None, 0.8),
        "ai_inference": inference("blur_refinement", {"label": "motion", "confidence": 0.4}),
    }
    assert build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]["status"] == "low_confidence"


def test_face_scalar_agrees_with_same_face_region():
    metrics = {
        "faces": m("faces", {"face_count": 1, "faces": [{"sharpness_score": 52.0, "measurement_confidence": 0.8}]}, 52, 0.8),
        "ai_inference": inference("face_quality", {"score_0_100": 58.0, "confidence": 0.9}, region="face_1"),
    }
    item = build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]
    assert item["status"] == "agree"
    assert item["region"] == "face_1"


def test_face_scalar_strong_disagreement_is_contradiction():
    metrics = {
        "faces": m("faces", {"face_count": 1, "faces": [{"sharpness_score": 30.0, "measurement_confidence": 0.85}]}, 30, 0.85),
        "ai_inference": inference("face_quality", {"score_0_100": 82.0, "confidence": 0.95}, region="face_1"),
    }
    assert build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]["status"] == "contradict"


def test_semantic_more_specific_compatible_result_refines():
    metrics = {
        "semantic_context": m("semantic_context", {"classification": "portrait"}, None, 0.75),
        "ai_inference": inference("semantic_context_refinement", {"label": "group_portrait", "confidence": 0.9}),
    }
    assert build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]["status"] == "refine"


def test_surface_presence_agrees_when_ai_mask_present():
    metrics = {
        "surface_defects": m("surface_defects", {"candidate_count": 12, "candidate_density": 0.3}, 70, 0.45),
        "ai_inference": inference("surface_defect_refinement", {"coverage_pct": 0.8, "confidence": 0.8}),
    }
    assert build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]["status"] == "agree"


def test_crosscheck_never_changes_scores():
    metric = build_ai_crosscheck_metric({"ai_inference": m("ai_inference", {"results": []}, None, 1.0)})
    assert metric.normalized_value is None
    assert metric.raw_value["changes_classical_scores"] is False


def test_crosscheck_metric_is_advisory_and_empty_without_successful_ai():
    metrics = {
        "brightness": m("brightness", 0.3, 40, 0.9),
        "ai_inference": m("ai_inference", {"results": [
            {"task": "face_quality", "model_id": "face-v1", "status": "not_run", "reason": "Нет модели."}
        ]}, None, 1.0),
    }
    metric = build_ai_crosscheck_metric(metrics)
    assert metric.raw_value["comparison_count"] == 0
    assert metric.raw_value["advisory_only"] is True
    assert metric.raw_value["changes_classical_scores"] is False
    assert metric.normalized_value is None


def test_native_surface_natural_detail_is_a_contradiction_not_a_refine():
    metrics = {
        "surface_defects": m("surface_defects", {"candidate_count": 5, "candidate_density": 0.2}, 65, 0.45),
        "ai_inference": inference(
            "surface_defect_refinement",
            {"label": "natural_detail", "confidence": 0.91, "defect_probability": 0.09},
            region="surface_1",
            model_id="native_surface_refiner_v1",
        ),
    }
    item = build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]
    assert item["status"] == "contradict"
    assert "естественной" in item["detail"]


def test_scalar_contract_without_confidence_is_never_treated_as_trusted_crosscheck():
    metrics = {
        "faces": m("faces", {"face_count": 1, "faces": [{"sharpness_score": 55.0, "measurement_confidence": 0.9}]}, 55, 0.9),
        "ai_inference": inference(
            "face_quality",
            {"score_0_100": 56.0, "confidence": 0.0, "confidence_available": False},
            region="face_1",
        ),
    }
    item = build_ai_crosscheck_metric(metrics).raw_value["comparisons"][0]
    assert item["status"] == "low_confidence"
    assert item["confidence"] == 0.0
    assert "не калиброванную уверенность" in item["detail"]
