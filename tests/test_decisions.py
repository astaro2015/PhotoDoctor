from photodoctor.core.decisions import build_decision_plan
from photodoctor.core.models import MetricResult


def m(name, raw, score, conf=0.8):
    return MetricResult(name, raw, score, conf, "test")


def base():
    return {
        "brightness": m("brightness", 0.5, 70, 0.85),
        "highlight_context": m("highlight_context", {"unexplained_clip_pct": 0.0}, None, 0.7),
        "contrast": m("contrast", 0.6, 85, 0.8),
        "local_contrast": m("local_contrast", {"classification": "normal"}, 82, 0.8),
        "laplacian": m("laplacian", 100, 75, 0.72),
        "tenengrad": m("tenengrad", 5000, 78, 0.78),
        "detail_loss_type": m("detail_loss_type", {"classification": "unknown"}, None, 0.5),
        "noise": m("noise", {}, 90, 0.7),
        "jpeg_artifacts": m("jpeg_artifacts", {}, 95, 0.7),
        "edge_artifacts": m("edge_artifacts", {}, 95, 0.7),
        "posterization": m("posterization", {}, 95, 0.7),
        "surface_defects": m("surface_defects", {"candidate_count": 0}, 100, 0.45),
        "image_tone": m("image_tone", {"classification": "color"}, None, 0.8),
        "semantic_context": m("semantic_context", {"classification": "general_photo", "face_count": 0}, None, 0.5),
    }


def test_dark_repairable_image_gets_exposure_fix():
    metrics = base()
    metrics["brightness"] = m("brightness", 0.08, 55, 0.85)
    plan = build_decision_plan(metrics)
    exposure = next(item for item in plan if item.key == "exposure")
    assert exposure.decision == "fix"
    assert exposure.repairability >= 80
    assert exposure.severity > 30



def test_normal_linear_brightness_does_not_trigger_exposure_fix():
    metrics = base()
    metrics["brightness"] = m("brightness", 0.24, 100, 0.85)
    plan = build_decision_plan(metrics)
    assert not any(item.key == "exposure" for item in plan)


def test_bright_main_face_vetoes_global_lift_from_dark_clothing_background():
    metrics = base()
    metrics["brightness"] = m("brightness", 0.12, 73, 0.85)
    metrics["histogram"] = m("histogram", {"p50": 0.09}, None, 0.98)
    metrics["faces"] = m(
        "faces",
        {"face_count": 1, "faces": [{"brightness_linear": 0.30}]},
        80,
        0.80,
    )
    metrics["main_subject"] = m(
        "main_subject",
        {"subject_kind": "person", "confidence": 0.85, "face_indices": [0]},
        None,
        0.85,
    )
    plan = build_decision_plan(metrics)
    assert not any(item.key == "exposure" for item in plan)

def test_archival_portrait_creates_high_priority_preserve_guardrail():
    metrics = base()
    metrics["semantic_context"] = m(
        "semantic_context",
        {"classification": "archival_portrait", "face_count": 2},
        None,
        0.78,
    )
    plan = build_decision_plan(metrics)
    first = plan[0]
    assert first.decision == "preserve"
    assert "оригинал" in first.title.lower() or "оригинал" in first.reason.lower()
    assert "лиц" in first.reason.lower()


def test_soft_archival_face_is_review_not_automatic_fix():
    metrics = base()
    metrics["semantic_context"] = m("semantic_context", {"classification": "archival_portrait", "face_count": 1}, None, 0.76)
    metrics["faces"] = m("faces", {"face_count": 1}, 38, 0.65)
    metrics["eyes"] = m("eyes", {"eye_count": 2}, 42, 0.55)
    metrics["detail_loss_type"] = m("detail_loss_type", {"classification": "degradation_like"}, None, 0.58)
    plan = build_decision_plan(metrics)
    sharp = next(item for item in plan if item.key == "sharpness")
    assert sharp.decision == "review"
    assert sharp.repairability < 60
    assert "лица" in sharp.guardrail.lower()
    assert "генератив" in sharp.guardrail.lower()


def test_clean_image_gets_skip_instead_of_invented_problem():
    plan = build_decision_plan(base())
    assert len(plan) == 1
    assert plan[0].decision == "skip"


def test_unknown_guard_downgrades_automatic_fix_to_manual_review():
    metrics = base()
    metrics["brightness"] = m("brightness", 0.08, 55, 0.85)
    metrics["analysis_reliability"] = m(
        "analysis_reliability",
        {"status": "unknown", "calibrated_confidence": 0.38, "ood_score": 0.20},
        None,
        0.38,
    )
    plan = build_decision_plan(metrics)
    exposure = next(item for item in plan if item.key == "exposure")
    assert exposure.decision == "review"
    assert "надёжность анализа снижена" in exposure.guardrail.lower()
