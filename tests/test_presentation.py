from photodoctor.core.models import MetricResult
from photodoctor.gui.presentation import (
    build_display_metrics,
    build_photo_profile,
    build_recommendation_groups,
    build_recommendations,
    build_summary,
)


def m(name, raw, score, conf=0.8):
    return MetricResult(name, raw, score, conf, "test", diagnostic="raw diagnostic")


def sample_metrics(with_tone=True):
    metrics = {
        "brightness": m("brightness", 0.2864, 100.0, 0.85),
        "histogram": m("histogram", {"bins": [0.1, 0.9]}, None, 0.98),
        "shadow_clipping": m("shadow_clipping", 1.55, 87.6, 0.9),
        "highlight_clipping": m("highlight_clipping", 2.68, 78.6, 0.9),
        "contrast": m("contrast", 0.844, 100.0, 0.8),
        "laplacian": m("laplacian", 153.6, 43.8, 0.72),
        "tenengrad": m("tenengrad", 3988.0, 50.4, 0.78),
        "noise": m("noise", 1.48, 94.1, 0.55),
        "color_cast": m("color_cast", {"mean_r": 136.7, "mean_g": 121.9, "mean_b": 109.0, "relative_spread": 0.227}, 59.1, 0.45),
    }
    if with_tone:
        metrics["image_tone"] = m(
            "image_tone",
            {
                "classification": "sepia",
                "confidence": 0.82,
                "mean_chroma": 0.18,
                "neutral_fraction": 0.14,
                "warm_order_fraction": 0.87,
            },
            None,
            0.82,
        )
    return metrics


def test_user_metrics_hide_histogram_and_merge_sharpness():
    rows = build_display_metrics(sample_metrics())
    keys = [r.key for r in rows]
    assert "histogram" not in keys
    assert "image_tone" not in keys
    assert "laplacian" not in keys
    assert "tenengrad" not in keys
    assert keys.count("sharpness") == 1


def test_sepia_like_cast_is_not_declared_simple_defect():
    rows = {r.key: r for r in build_display_metrics(sample_metrics())}
    text = rows["color_cast"].diagnosis.lower()
    assert "сеп" in text
    assert rows["color_cast"].status == "Особенность"


def test_photo_profile_uses_explicit_tone_detector():
    profile = build_photo_profile(sample_metrics())
    assert profile.key == "sepia"
    assert profile.confidence == 0.82


def test_summary_has_expected_ranges_and_ignores_sepia_as_issue():
    summary = build_summary(sample_metrics())
    assert 0 <= summary["quality"] <= 100
    assert 0 <= summary["potential"] <= 100
    assert 0 <= summary["confidence"] <= 100
    assert summary["issues"] == 1  # sharpness only; balanced linear brightness is not a defect
    assert "Сеп" in summary["profile"]


def test_recommendations_include_vintage_color_warning():
    recs = build_recommendations(sample_metrics())
    joined = " ".join(recs).lower()
    assert "сеп" in joined
    assert "автомат" in joined


def test_recommendation_groups_separate_safe_caution_and_avoid():
    groups = build_recommendation_groups(sample_metrics())
    assert groups.safe
    assert groups.caution
    assert groups.avoid
    assert any("сеп" in text.lower() for text in groups.avoid)


def test_legacy_metrics_still_get_sepia_fallback():
    profile = build_photo_profile(sample_metrics(with_tone=False))
    assert profile.key == "sepia"
    assert profile.confidence < 0.60


def test_local_sharpness_is_displayed_but_not_double_counted_in_summary():
    base = sample_metrics()
    summary_before = build_summary(base)
    base["local_sharpness"] = m(
        "local_sharpness",
        {"soft_area_pct": 35.0, "informative_cells": 20, "total_cells": 30, "cells_norm": []},
        40.0,
        0.8,
    )
    rows = {r.key: r for r in build_display_metrics(base)}
    assert "local_sharpness" in rows
    assert "мяг" in rows["local_sharpness"].diagnosis.lower()
    # Local map refines where blur is located, but does not count sharpness twice in the global score.
    assert build_summary(base) == summary_before


def test_local_sharpness_can_add_map_recommendation():
    metrics = sample_metrics()
    metrics["local_sharpness"] = m(
        "local_sharpness",
        {"soft_area_pct": 42.0, "informative_cells": 18, "total_cells": 28, "cells_norm": []},
        44.0,
        0.82,
    )
    groups = build_recommendation_groups(metrics)
    assert any("карт" in text.lower() and "резк" in text.lower() for text in groups.caution)


def test_summary_groups_face_and_global_sharpness_as_one_problem():
    metrics = sample_metrics()
    metrics["faces"] = m(
        "faces",
        {
            "detector_available": True,
            "face_count": 1,
            "faces": [{"brightness_linear": 0.20, "sharpness_score": 40.0}],
        },
        40.0,
        0.75,
    )
    summary = build_summary(metrics)
    # Face sharpness refines the existing global sharpness issue instead of adding another one.
    assert summary["issues"] == 1


def test_local_contrast_fading_candidate_has_human_diagnosis_and_recommendation():
    metrics = sample_metrics()
    metrics["local_contrast"] = m(
        "local_contrast",
        {
            "low_contrast_area_pct": 61.0,
            "median_local_range": 0.12,
            "fading_likelihood": 0.74,
            "classification": "fading_candidate",
            "informative_cells": 24,
            "total_cells": 30,
            "cells_norm": [],
        },
        32.0,
        0.78,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["local_contrast"].status in {"Проблема", "Внимание"}
    assert "выцвет" in rows["local_contrast"].diagnosis.lower()
    groups = build_recommendation_groups(metrics)
    joined = " ".join((*groups.safe, *groups.caution)).lower()
    assert "полут" in joined or "выцвет" in joined


def test_global_and_local_contrast_share_one_issue_group():
    metrics = sample_metrics()
    metrics["contrast"] = m("contrast", 0.18, 25.0, 0.80)
    metrics["local_contrast"] = m(
        "local_contrast",
        {"low_contrast_area_pct": 70.0, "median_local_range": 0.10, "fading_likelihood": 0.8, "classification": "fading_candidate"},
        28.0,
        0.78,
    )
    summary = build_summary(metrics)
    # contrast + sharpness; local contrast refines the contrast issue rather than adding another one.
    assert summary["issues"] == 2


def test_strong_jpeg_artifacts_are_explained_and_warn_before_sharpening():
    metrics = sample_metrics()
    metrics["jpeg_artifacts"] = m(
        "jpeg_artifacts",
        {"classification": "strong", "block_ratio": 4.2, "source_is_jpeg": True},
        25.0,
        0.80,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert "jpeg" in rows["jpeg_artifacts"].diagnosis.lower()
    assert rows["jpeg_artifacts"].status == "Проблема"
    groups = build_recommendation_groups(metrics)
    joined = " ".join(groups.caution).lower()
    assert "резк" in joined
    assert "block" in joined or "блок" in joined


def test_edge_artifact_warning_discourages_more_sharpening():
    metrics = sample_metrics()
    metrics["edge_artifacts"] = m(
        "edge_artifacts",
        {"classification": "strong_halo_candidate", "halo_likelihood": 0.82, "ringing_likelihood": 0.25},
        18.0,
        0.72,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["edge_artifacts"].status == "Проблема"
    assert "ореол" in rows["edge_artifacts"].diagnosis.lower() or "перешарп" in rows["edge_artifacts"].diagnosis.lower()
    groups = build_recommendation_groups(metrics)
    assert any("резк" in item.lower() or "sharpen" in item.lower() for item in (*groups.caution, *groups.avoid))


def test_posterization_has_human_diagnosis_and_debanding_advice():
    metrics = sample_metrics()
    metrics["posterization"] = m(
        "posterization",
        {"classification": "strong", "severity": 0.80, "occupied_luma_levels": 16},
        20.0,
        0.74,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["posterization"].status == "Проблема"
    assert "постер" in rows["posterization"].diagnosis.lower() or "ступ" in rows["posterization"].diagnosis.lower()
    groups = build_recommendation_groups(metrics)
    joined = " ".join((*groups.caution, *groups.avoid)).lower()
    assert "deband" in joined or "гради" in joined or "ступ" in joined


def test_processing_artifacts_share_one_issue_group():
    metrics = sample_metrics()
    metrics["edge_artifacts"] = m(
        "edge_artifacts",
        {"classification": "strong_halo_candidate", "halo_likelihood": 0.8},
        20.0,
        0.72,
    )
    metrics["posterization"] = m(
        "posterization",
        {"classification": "strong", "severity": 0.8, "occupied_luma_levels": 16},
        20.0,
        0.72,
    )
    summary = build_summary(metrics)
    # Baseline sample has one sharpness issue. Both processing artifacts add one semantic group.
    assert summary["issues"] == 2


def test_highlight_context_explains_protected_clipping():
    metrics = sample_metrics()
    metrics["highlight_context"] = m(
        "highlight_context",
        {
            "total_clip_pct": 2.5,
            "protected_clip_pct": 2.1,
            "unexplained_clip_pct": 0.4,
            "candidate_count": 2,
            "face_rejected_count": 0,
            "boxes_norm": [],
        },
        None,
        0.72,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["highlight_context"].status == "Инфо"
    assert "2.10" in rows["highlight_context"].diagnosis
    assert "0.40" in rows["highlight_context"].diagnosis
    groups = build_recommendation_groups(metrics)
    assert any("зеркальн" in item.lower() for item in groups.safe)


def test_missing_eyes_do_not_penalize_summary_or_claim_closed_eyes():
    metrics = sample_metrics()
    before = build_summary(metrics)
    metrics["eyes"] = m(
        "eyes",
        {
            "detector_available": True,
            "face_count": 1,
            "eye_count": 0,
            "faces_with_eyes": 0,
            "eyes": [],
        },
        None,
        0.38,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["eyes"].status == "Инфо"
    text = rows["eyes"].diagnosis.lower()
    assert "не удалось" in text
    assert "не означает" in text
    assert "закрыт" in text
    assert build_summary(metrics) == before


def test_soft_detected_eyes_refine_existing_sharpness_issue_instead_of_adding_one():
    metrics = sample_metrics()
    metrics["faces"] = m(
        "faces",
        {
            "detector_available": True,
            "face_count": 1,
            "faces": [{"brightness_linear": 0.30, "sharpness_score": 55.0}],
        },
        55.0,
        0.60,
    )
    metrics["eyes"] = m(
        "eyes",
        {
            "detector_available": True,
            "face_count": 1,
            "eye_count": 2,
            "faces_with_eyes": 1,
            "eyes": [
                {"face_index": 0, "sharpness_score": 39.0},
                {"face_index": 0, "sharpness_score": 47.0},
            ],
        },
        39.0,
        0.58,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["eyes"].status in {"Проблема", "Внимание"}
    assert "глаз" in rows["eyes"].diagnosis.lower()
    summary = build_summary(metrics)
    assert summary["issues"] == 1  # one shared sharpness group
    groups = build_recommendation_groups(metrics)
    joined = " ".join((*groups.caution, *groups.avoid)).lower()
    assert "глаз" in joined


def test_face_and_eye_refinements_do_not_multiply_global_quality_weight():
    metrics = sample_metrics()
    baseline = build_summary(metrics)
    metrics["faces"] = m(
        "faces",
        {"detector_available": True, "face_count": 1, "faces": [{"brightness_linear": 0.30, "sharpness_score": 20.0}]},
        20.0,
        0.72,
    )
    metrics["eyes"] = m(
        "eyes",
        {
            "detector_available": True,
            "face_count": 1,
            "eye_count": 2,
            "faces_with_eyes": 1,
            "eyes": [{"face_index": 0, "sharpness_score": 18.0}, {"face_index": 0, "sharpness_score": 22.0}],
        },
        18.0,
        0.62,
    )
    refined = build_summary(metrics)
    assert refined["quality"] == baseline["quality"]
    assert refined["potential"] == baseline["potential"]
    assert refined["confidence"] == baseline["confidence"]
    assert refined["issues"] == baseline["issues"]  # already one global sharpness issue


def test_exif_context_is_informational_and_non_penalizing():
    metrics = sample_metrics()
    baseline = build_summary(metrics)
    metrics["exif_context"] = m(
        "exif_context",
        {
            "available": True,
            "exposure_s": 1 / 30,
            "focal_length_mm": 50.0,
            "aperture_f": 2.8,
            "iso": 3200.0,
            "shutter_risk": "high",
            "iso_noise_risk": "high",
        },
        None,
        0.70,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["exif_context"].status == "Инфо"
    assert "1/" in rows["exif_context"].diagnosis
    assert build_summary(metrics) == baseline


def test_exif_only_supports_motion_recommendation_when_image_also_says_motion():
    metrics = sample_metrics()
    metrics["exif_context"] = m(
        "exif_context",
        {"available": True, "shutter_risk": "high", "iso_noise_risk": "low", "exposure_s": 1 / 20, "focal_length_mm": 85.0},
        None,
        0.70,
    )
    groups_without_motion = build_recommendation_groups(metrics)
    assert not any("exif поддерживает гипотезу смаза" in text.lower() for text in groups_without_motion.caution)
    metrics["detail_loss_type"] = m(
        "detail_loss_type",
        {"classification": "motion_like", "confidence": 0.72},
        None,
        0.72,
    )
    groups_with_motion = build_recommendation_groups(metrics)
    assert any("exif поддерживает гипотезу смаза" in text.lower() for text in groups_with_motion.caution)


def test_nss_baseline_is_informational_and_never_fake_brisque_score():
    metrics = sample_metrics()
    baseline = build_summary(metrics)
    metrics["nss_baseline"] = m(
        "nss_baseline",
        {
            "backend": "brisque_style_nss_features_v1",
            "feature_count": 36,
            "features": [0.0] * 36,
            "informative": True,
            "score_available": False,
        },
        None,
        0.76,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["nss_baseline"].status == "Инфо"
    text = rows["nss_baseline"].diagnosis.lower()
    assert "36" in text
    assert "оценк" in text
    assert "brisque" in text
    assert build_summary(metrics) == baseline


def test_semantic_context_is_informational_and_preserves_archival_faces():
    metrics = sample_metrics()
    baseline = build_summary(metrics)
    metrics["semantic_context"] = m(
        "semantic_context",
        {
            "classification": "archival_portrait",
            "people_context": "group",
            "face_count": 2,
            "archival_likelihood": 0.74,
            "preservation_priority": "faces_and_original_character",
        },
        None,
        0.74,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert rows["semantic_context"].status == "Инфо"
    assert "архив" in rows["semantic_context"].diagnosis.lower()
    assert "лиц" in rows["semantic_context"].diagnosis.lower()
    assert build_summary(metrics) == baseline
    groups = build_recommendation_groups(metrics)
    joined = " ".join((*groups.safe, *groups.avoid)).lower()
    assert "архив" in joined
    assert "лиц" in joined


def test_rejected_validator_result_downgrades_safe_exposure_recommendation():
    metrics = sample_metrics()
    metrics["recommendation_validation"] = m(
        "recommendation_validation",
        {"items": [{"action_key": "exposure", "tested": True, "accepted": False}]},
        None,
        0.7,
    )
    groups = build_recommendation_groups(metrics)
    assert not any("поднять средние тона" in text.lower() for text in groups.safe)
    assert any("проверка безопасности" in text.lower() and "средн" in text.lower() for text in groups.caution)


def test_manual_review_validation_is_not_misreported_as_validator_rejection():
    metrics = sample_metrics()
    metrics["recommendation_validation"] = m(
        "recommendation_validation",
        {"items": [{
            "action_key": "exposure",
            "tested": True,
            "accepted": False,
            "auto_eligible": False,
            "technical_passed": True,
        }]},
        None,
        0.7,
    )
    groups = build_recommendation_groups(metrics)
    joined = " ".join(groups.caution).lower()
    assert "модуль решений" in joined
    assert "галоч" in joined
    assert "validator не подтвердил" not in joined


def test_aesthetic_quality_is_displayed_but_does_not_change_technical_summary():
    metrics = sample_metrics()
    baseline = build_summary(metrics)
    metrics["aesthetic_quality"] = m(
        "aesthetic_quality",
        {
            "score": 18.0,
            "confidence": 0.78,
            "classification": "review",
            "components": {"placement": 12.0, "prominence": 25.0},
            "method": "aesthetic_subject_composition_v1",
            "technical_quality_independent": True,
        },
        18.0,
        0.78,
    )
    rows = {r.key: r for r in build_display_metrics(metrics)}
    assert "aesthetic_quality" in rows
    assert rows["aesthetic_quality"].status == "Инфо"
    assert build_summary(metrics) == baseline

    # Even a very high composition score must not silently inflate technical quality.
    metrics["aesthetic_quality"] = m(
        "aesthetic_quality",
        {
            "score": 96.0,
            "confidence": 0.78,
            "classification": "strong_composition_signal",
            "components": {"placement": 98.0, "prominence": 94.0},
            "method": "aesthetic_subject_composition_v1",
            "technical_quality_independent": True,
        },
        96.0,
        0.78,
    )
    assert build_summary(metrics) == baseline


def test_analysis_reliability_replaces_summary_confidence_but_not_quality():
    metrics = sample_metrics()
    baseline = build_summary(metrics)
    metrics["analysis_reliability"] = m(
        "analysis_reliability",
        {
            "status": "caution",
            "calibrated_confidence": 0.61,
            "support_score": 0.74,
            "ood_score": 0.20,
            "low_support_metrics": [],
            "unknown_components": ["detail_loss_type"],
        },
        None,
        0.61,
    )
    summary = build_summary(metrics)
    assert summary["quality"] == baseline["quality"]
    assert summary["potential"] == baseline["potential"]
    assert summary["issues"] == baseline["issues"]
    assert summary["confidence"] == 61.0
    rows = {row.key: row for row in build_display_metrics(metrics)}
    assert rows["analysis_reliability"].status == "Внимание"
    assert "не вероятность истины" in rows["analysis_reliability"].diagnosis.lower()


def test_unknown_guard_adds_manual_review_recommendation():
    metrics = sample_metrics()
    metrics["analysis_reliability"] = m(
        "analysis_reliability",
        {"status": "unknown", "calibrated_confidence": 0.39, "ood_score": 0.25},
        None,
        0.39,
    )
    groups = build_recommendation_groups(metrics)
    assert any("защита при неопределённости" in text.lower() for text in groups.avoid)
