from photodoctor.core.ai_trust import build_ai_trust_gate_metric
from photodoctor.core.models import MetricResult


def cross(items):
    return {"ai_crosscheck": MetricResult("ai_crosscheck", {"comparisons": items}, None, 0.8, "ai")}


def item(status, confidence=0.8, task="blur_refinement"):
    return {"task": task, "model_id": "m", "region": "global", "status": status, "confidence": confidence}


def test_agreement_can_be_used_only_as_advice():
    metric = build_ai_trust_gate_metric(cross([item("agree", 0.8)]))
    raw = metric.raw_value
    assert raw["overall_policy"] == "advisory_ok"
    assert raw["can_trigger_automatic_edits"] is False
    assert metric.normalized_value is None


def test_refinement_is_cautious_advice():
    raw = build_ai_trust_gate_metric(cross([item("refine", 0.8)])).raw_value
    assert raw["overall_policy"] == "advisory_caution"


def test_contradiction_forces_manual_review():
    raw = build_ai_trust_gate_metric(cross([item("agree", 0.9), item("contradict", 0.9)])).raw_value
    assert raw["overall_policy"] == "manual_review"
    assert raw["counts"]["manual_review"] == 1


def test_low_confidence_is_ignored():
    raw = build_ai_trust_gate_metric(cross([item("low_confidence", 0.3)])).raw_value
    assert raw["overall_policy"] == "ignore"


def test_no_ai_crosscheck_is_unavailable_not_error():
    raw = build_ai_trust_gate_metric({}).raw_value
    assert raw["overall_policy"] == "unavailable"
    assert raw["items"] == []
