from photodoctor.core.models import MetricResult
from photodoctor.core.surface_refinement import build_surface_refinement_metric


def _metric(name, raw, confidence=0.8):
    return MetricResult(name, raw, None, confidence, "test")


def _region(index, label, confidence=0.9):
    defect_probability = 0.9 if label == "defect" else 0.1 if label == "natural_detail" else 0.5
    return {
        "region": f"surface_{index + 1}",
        "status": "ok",
        "result": {
            "label": label,
            "confidence": confidence,
            "defect_probability": defect_probability,
            "model_id": "native_surface_refiner_v1",
            "training_kind": "procedural_precise_candidate_hograwmeta_v1",
        },
    }


def _strong_box(x=0.1):
    return {
        "x": x, "y": 0.1, "w": 0.02, "h": 0.08,
        "polarity": "bright", "candidate_quality": 0.86,
        "context_contrast": 0.92, "side_similarity": 0.90,
        "texture_risk": 0.08, "chroma_risk": 0.05,
    }


def test_surface_refinement_summarizes_full_ai_pool_not_only_visible_twenty():
    boxes = [_strong_box(i / 100.0) for i in range(20)]
    ai_boxes = [_strong_box(i / 100.0) for i in range(30)]
    regions = []
    for i in range(28):
        label = "defect" if i < 12 else "natural_detail" if i < 24 else "uncertain"
        regions.append(_region(i, label))
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 30, "boxes_norm": boxes, "ai_boxes_norm": ai_boxes}),
        "ai_inference": _metric("ai_inference", {
            "results": [{"task": "surface_defect_refinement", "regions": regions}]
        }),
    }
    metric = build_surface_refinement_metric(metrics)
    raw = metric.raw_value
    assert raw["evaluated_count"] == 28
    assert raw["display_evaluated_count"] == 20
    assert raw["confirmed_defect_count"] == 12
    assert raw["likely_natural_count"] == 12
    assert raw["uncertain_count"] == 4
    assert raw["unprocessed_count"] == 2
    assert len(raw["refined_boxes_norm"]) == 20


def test_surface_refinement_overlay_keeps_ai_labels_for_visible_candidates():
    boxes = [_strong_box(0.1), _strong_box(0.3)]
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 4, "boxes_norm": boxes, "ai_boxes_norm": boxes + [_strong_box(0.5), _strong_box(0.7)]}),
        "ai_inference": _metric("ai_inference", {
            "results": [{"task": "surface_defect_refinement", "regions": [
                _region(0, "natural_detail", 0.91), _region(1, "defect", 0.88), _region(2, "defect", 0.95)
            ]}]
        }),
    }
    raw = build_surface_refinement_metric(metrics).raw_value
    refined = raw["refined_boxes_norm"]
    assert refined[0]["ai_label"] == "natural_detail"
    assert refined[1]["ai_label"] == "defect"
    assert raw["evaluated_count"] == 3
    # v6 promotes confirmed defects from below the small classical display pool.
    assert raw["display_evaluated_count"] == 3
    assert raw["promoted_confirmed_count"] == 1
    assert len(refined) == 3
    assert refined[2]["verification_label"] == "defect"
    assert refined[2]["promoted_from_ai_pool"] is True


def test_synthetic_ai_cannot_rescue_low_quality_candidate():
    box = {
        "x": 0.1, "y": 0.1, "w": 0.02, "h": 0.08,
        "polarity": "bright", "candidate_quality": 0.18,
        "context_contrast": 0.10, "side_similarity": 0.30,
        "texture_risk": 0.92, "chroma_risk": 0.10,
    }
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_region(0, "defect", 0.98)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_raw_label"] == "defect"
    assert refined["ai_label"] != "defect"


def test_synthetic_ai_does_not_confirm_dark_line_by_itself():
    box = _strong_box()
    box["polarity"] = "dark"
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_region(0, "defect", 0.98)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_raw_label"] == "defect"
    assert refined["ai_label"] == "uncertain"


def test_context_and_synthetic_ai_can_confirm_strong_bright_candidate():
    box = _strong_box()
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_region(0, "defect", 0.94)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] == "defect"
    assert refined["verification_reason"]


def _v2_region(index, *, p=0.90, meta_p=0.90, meta_support=True):
    return {
        "region": f"surface_{index + 1}",
        "status": "ok",
        "result": {
            "label": "defect" if p >= 0.82 else "uncertain",
            "confidence": p if p >= 0.82 else 0.5,
            "defect_probability": p,
            "model_id": "native_surface_verifier_v2",
            "training_kind": "candidate_pair_group_cv_segmentation_v2",
            "context_meta_model_id": "native_surface_context_meta_v2",
            "context_meta_probability": meta_p,
            "context_meta_threshold": 0.505,
            "context_meta_supports_defect": meta_support,
            "expert_verified": False,
        },
    }


def test_v2_pixel_model_cannot_confirm_when_context_meta_vetoes():
    box = _strong_box()
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.97, meta_p=0.20, meta_support=False)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_raw_label"] == "defect"
    assert refined["ai_label"] != "defect"
    assert refined["context_meta_probability"] == 0.20


def test_v2_requires_pixel_and_context_agreement_for_probable_defect():
    box = _strong_box()
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.94, meta_p=0.91, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] == "defect"
    assert refined["context_meta_supports_defect"] is True
    assert "Context Meta Verifier" in refined["verification_reason"]


def test_v2_borderline_dark_spot_stays_uncertain_even_when_both_models_lean_positive():
    box = _strong_box()
    box.update({
        "polarity": "dark", "candidate_kind": "spot",
        "candidate_quality": 0.711, "context_contrast": 1.0,
        "side_similarity": 0.5, "texture_risk": 0.053, "chroma_risk": 0.0,
    })
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.8214, meta_p=0.6018, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] == "uncertain"


def test_v2_strong_context_can_confirm_when_pixel_model_is_only_uncertain():
    box = _strong_box()
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.55, meta_p=0.90, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_raw_label"] == "uncertain"
    assert refined["ai_label"] == "defect"
    assert "не дал сильного отрицательного" in refined["verification_reason"]


def test_v2_strong_context_is_vetoed_by_strong_pixel_negative():
    box = _strong_box()
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.10, meta_p=0.92, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] != "defect"


def test_v2_pale_low_contrast_bright_line_can_be_rescued_by_strong_context():
    box = _strong_box()
    box.update({
        "detection_branch": "low_contrast_hough",
        "polarity": "bright",
        "candidate_kind": "line",
        "candidate_quality": 0.82,
        "context_contrast": 1.0,
        "side_similarity": 0.88,
        "texture_risk": 0.0,
        "chroma_risk": 0.0,
    })
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.04, meta_p=0.84, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["verification_label"] == "defect"
    assert "известную слабость" in refined["verification_reason"]


def test_v2_bright_spot_in_textured_region_needs_very_strong_pixel_support():
    box = _strong_box()
    box.update({"candidate_kind": "spot", "polarity": "bright", "texture_risk": 0.75})
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.94, meta_p=0.82, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] != "defect"


def test_v2_bright_spot_with_weak_pixel_support_stays_unconfirmed_even_with_strong_meta():
    box = _strong_box()
    box.update({"candidate_kind": "spot", "polarity": "bright", "texture_risk": 0.40})
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.25, meta_p=0.95, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] != "defect"


def test_v2_bright_spot_can_pass_when_context_and_pixel_support_are_both_strong():
    box = _strong_box()
    box.update({"candidate_kind": "spot", "polarity": "bright", "texture_risk": 0.70})
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.98, meta_p=0.90, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] == "defect"


def test_v2_bright_spot_below_precision_first_pixel_floor_stays_unconfirmed():
    box = _strong_box()
    box.update({"candidate_kind": "spot", "polarity": "bright", "texture_risk": 0.40})
    metrics = {
        "surface_defects": _metric("surface_defects", {"candidate_count": 1, "boxes_norm": [box], "ai_boxes_norm": [box]}),
        "ai_inference": _metric("ai_inference", {"results": [{"task": "surface_defect_refinement", "regions": [_v2_region(0, p=0.69, meta_p=0.94, meta_support=True)]}]}),
    }
    refined = build_surface_refinement_metric(metrics).raw_value["refined_boxes_norm"][0]
    assert refined["ai_label"] != "defect"
