from __future__ import annotations

from photodoctor.core.analysis_reliability import analyze_analysis_reliability
from photodoctor.core.models import MetricResult


def m(name, confidence=0.8, raw=None, score=80.0):
    return MetricResult(name, raw if raw is not None else {}, score, confidence, "test")


def supported_metrics():
    return {
        "brightness": m("brightness", 0.85),
        "shadow_clipping": m("shadow_clipping", 0.90),
        "highlight_clipping": m("highlight_clipping", 0.90),
        "contrast": m("contrast", 0.80),
        "local_contrast": m("local_contrast", 0.82, {"informative_cells": 80, "total_cells": 100}),
        "local_sharpness": m("local_sharpness", 0.84, {"informative_cells": 75, "total_cells": 100}),
        "noise": m("noise", 0.78, {"usable_tiles": 40, "total_tiles": 50}),
        "jpeg_artifacts": m("jpeg_artifacts", 0.75),
        "edge_artifacts": m("edge_artifacts", 0.72),
        "posterization": m("posterization", 0.72),
        "detail_loss_type": m("detail_loss_type", 0.72, {"classification": "defocus_like"}, None),
        "nss_baseline": m("nss_baseline", 0.70, {"informative": True, "image_std": 0.12}, None),
        "main_subject": m("main_subject", 0.78, {"subject_kind": "person"}, None),
        "semantic_context": m("semantic_context", 0.70, {"classification": "portrait"}, None),
        "analysis_profile": m("analysis_profile", 1.0, {"technical_width": 1600, "technical_height": 1200}, None),
    }


def test_supported_photo_is_reliable_without_ood_signal():
    result = analyze_analysis_reliability(supported_metrics())
    assert result.status == "reliable"
    assert result.calibrated_confidence >= 0.70
    assert result.ood_score < 0.20
    assert not result.unknown_components


def test_multiple_low_information_signals_become_ood_candidate():
    metrics = supported_metrics()
    metrics["nss_baseline"] = m("nss_baseline", 0.20, {"informative": False, "image_std": 0.005}, None)
    metrics["local_sharpness"] = m("local_sharpness", 0.20, {"informative_cells": 2, "total_cells": 100})
    metrics["local_contrast"] = m("local_contrast", 0.20, {"informative_cells": 3, "total_cells": 100})
    metrics["noise"] = m("noise", 0.20, {"usable_tiles": 1, "total_tiles": 50})
    metrics["detail_loss_type"] = m("detail_loss_type", 0.20, {"classification": "unknown"}, None)
    metrics["main_subject"] = m("main_subject", 0.10, {"subject_kind": "unknown"}, None)
    metrics["semantic_context"] = m("semantic_context", 0.20, {"classification": "general_photo"}, None)
    metrics["analysis_profile"] = m("analysis_profile", 1.0, {"technical_width": 80, "technical_height": 80}, None)
    result = analyze_analysis_reliability(metrics)
    assert result.status == "ood_candidate"
    assert result.ood_score >= 0.70
    assert result.calibrated_confidence < 0.45
    assert "detail_loss_type" in result.unknown_components


def test_unknown_blur_alone_does_not_make_normal_photo_ood():
    metrics = supported_metrics()
    metrics["detail_loss_type"] = m("detail_loss_type", 0.38, {"classification": "unknown"}, None)
    result = analyze_analysis_reliability(metrics)
    assert result.status in {"caution", "unknown"}
    assert result.status != "ood_candidate"
    assert result.ood_score < 0.40


def test_raw_payload_explicitly_says_confidence_is_not_probability():
    raw = analyze_analysis_reliability(supported_metrics()).to_raw()
    assert raw["method"] == "heuristic_analysis_reliability_v1"
    assert raw["probability_interpretation"] is False
