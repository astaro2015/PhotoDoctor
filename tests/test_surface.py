from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from photodoctor.core.service import analyze_file
from photodoctor.core.surface import SurfaceDefectDetection, _bounded_hysteresis_mask, _deduplicate_ranked_candidates, _iou, detect_surface_defects, merge_surface_detections
from photodoctor.gui.presentation import build_display_metrics, build_recommendation_groups


def test_clean_flat_image_has_no_surface_candidates():
    rgb = np.full((256, 320, 3), 128, np.uint8)
    result = detect_surface_defects(rgb)
    assert result.candidate_count == 0
    assert result.candidate_density == 0.0
    assert result.score == 100.0


def test_bright_and_dark_thin_lines_are_candidates():
    rgb = np.full((512, 512, 3), 128, np.uint8)
    rgb[100:103, 60:430] = 235
    for y in range(220, 420):
        x = 250 + (y - 220) // 6
        rgb[y:y + 2, x:x + 2] = 25
    result = detect_surface_defects(rgb)
    assert result.candidate_count >= 2
    assert result.candidate_density > 0.1
    assert result.boxes_norm


def test_surface_boxes_are_normalized_and_limited():
    rgb = np.full((480, 640, 3), 128, np.uint8)
    for x in range(30, 610, 20):
        rgb[40:43, x:x + 10] = 230
    result = detect_surface_defects(rgb, max_boxes=7)
    assert len(result.boxes_norm) <= 7
    for box in result.boxes_norm:
        assert 0.0 <= box["x"] <= 1.0
        assert 0.0 <= box["y"] <= 1.0
        assert 0.0 < box["w"] <= 1.0
        assert 0.0 < box["h"] <= 1.0



def test_surface_keeps_larger_ai_candidate_pool_than_display_pool():
    rgb = np.full((900, 1200, 3), 128, np.uint8)
    # Many separated bright scratches so the display pool truncates before the AI pool.
    for y in range(70, 830, 28):
        x0 = 80 + (y % 45)
        rgb[y:y + 2, x0:x0 + 90] = 245
    result = detect_surface_defects(rgb, max_boxes=5, max_ai_boxes=32)
    assert len(result.boxes_norm) <= 5
    assert len(result.ai_boxes_norm) <= 32
    assert len(result.ai_boxes_norm) >= len(result.boxes_norm)
    assert result.ai_boxes_norm[:len(result.boxes_norm)] == result.boxes_norm
    if result.candidate_count > 5:
        assert len(result.ai_boxes_norm) > len(result.boxes_norm)

def test_analyzer_exposes_surface_metric(tmp_path):
    rgb = np.full((300, 420, 3), 120, np.uint8)
    rgb[150:153, 40:380] = 245
    path = tmp_path / "scratch.png"
    Image.fromarray(rgb, "RGB").save(path)
    result = analyze_file(path)
    metric = result.metrics["surface_defects"]
    assert isinstance(metric.raw_value, dict)
    assert metric.raw_value["method"] == "context_aware_morphology_v9_deep_recall_manual_only"
    assert metric.raw_value["shown_count"] <= 24
    assert metric.confidence < 0.50


def test_surface_candidates_are_manual_review_not_confirmed_defect(tmp_path):
    rgb = np.full((300, 420, 3), 120, np.uint8)
    rgb[150:153, 40:380] = 245
    path = tmp_path / "scratch.png"
    Image.fromarray(rgb, "RGB").save(path)
    metrics = analyze_file(path).metrics
    row = {r.key: r for r in build_display_metrics(metrics)}["surface_defects"]
    assert row.status == "Проверить"
    assert "кандидат" in row.diagnosis.lower()
    groups = build_recommendation_groups(metrics)
    assert any("карт" in x.lower() for x in groups.caution)
    assert any("автомат" in x.lower() for x in groups.avoid)


def test_surface_candidates_include_tight_normalized_contours():
    rgb = np.full((420, 620, 3), 128, np.uint8)
    for y in range(80, 330):
        x = 120 + (y - 80) // 3
        rgb[y:y + 2, x:x + 2] = 240
    result = detect_surface_defects(rgb)
    with_contour = [box for box in result.boxes_norm if isinstance(box.get("contour"), list) and len(box["contour"]) >= 2]
    assert with_contour
    for box in with_contour:
        for point in box["contour"]:
            assert 0.0 <= float(point["x"]) <= 1.0
            assert 0.0 <= float(point["y"]) <= 1.0


def test_context_detector_rejects_text_like_natural_detail():
    import cv2
    rgb = np.full((512, 512, 3), 235, np.uint8)
    cv2.putText(rgb, "PHOTO", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (35, 35, 35), 5, cv2.LINE_AA)
    result = detect_surface_defects(rgb)
    assert result.candidate_count == 0


def test_context_detector_keeps_multidirectional_curved_scratch():
    import cv2
    rgb = np.full((512, 512, 3), 115, np.uint8)
    points = np.array([[80, 400], [140, 340], [180, 290], [235, 250], [290, 180], [350, 125], [430, 80]], np.int32)
    cv2.polylines(rgb, [points], False, (238, 238, 238), 2, cv2.LINE_AA)
    result = detect_surface_defects(rgb)
    assert result.candidate_count >= 1
    box = result.boxes_norm[0]
    assert box["candidate_kind"] == "line"
    assert float(box["oriented_aspect"]) > 8.0
    assert float(box["stroke_width_px"]) < 8.5
    assert float(box["candidate_quality"]) > 0.60


def test_context_detector_keeps_small_bright_dust_spot():
    import cv2
    rgb = np.full((512, 512, 3), 128, np.uint8)
    cv2.circle(rgb, (250, 250), 4, (245, 245, 245), -1)
    result = detect_surface_defects(rgb)
    assert result.candidate_count == 1
    assert result.boxes_norm[0]["candidate_kind"] == "spot"
    assert float(result.boxes_norm[0]["candidate_quality"]) > 0.75


def test_semantic_protection_penalizes_dark_line_inside_face_region():
    import cv2
    rgb = np.full((512, 512, 3), (190, 155, 135), np.uint8)
    cv2.line(rgb, (250, 40), (280, 470), (35, 30, 28), 3)
    raw = detect_surface_defects(rgb)
    protected = detect_surface_defects(rgb, protection_boxes=[(180, 20, 180, 470, 0.95)])
    assert raw.candidate_count >= 1
    assert protected.candidate_count >= 1
    assert float(protected.boxes_norm[0]["semantic_risk"]) > 0.80
    assert float(protected.boxes_norm[0]["candidate_quality"]) < float(raw.boxes_norm[0]["candidate_quality"])


def test_context_detector_rejects_repeated_parallel_fabric_lines():
    import cv2
    rgb = np.full((768, 1024, 3), 180, np.uint8)
    # Repeated, almost identical diagonal seams are deliberate scene texture, not
    # independent film scratches. The second-stage repetition check should suppress
    # this hard negative before the synthetic AI sees dozens of tempting lines.
    for x in range(80, 950, 28):
        cv2.line(rgb, (x, 100), (x + 180, 680), (115, 115, 115), 2, cv2.LINE_AA)
    result = detect_surface_defects(rgb)
    assert result.candidate_count == 0


def test_vectorized_surface_iou_dedup_matches_scalar_greedy_reference():
    rng = np.random.default_rng(20260922)
    candidates: list[tuple[float, dict[str, object]]] = []
    for index in range(320):
        # Mix independent boxes with clusters of near-duplicates so both keep/reject
        # branches are exercised heavily. Order is intentionally fixed because the
        # production algorithm is greedy in rank order.
        cluster = index % 17
        base_x = (cluster % 5) * 0.16 + float(rng.normal(0.0, 0.012))
        base_y = (cluster // 5) * 0.21 + float(rng.normal(0.0, 0.012))
        width = float(np.clip(rng.uniform(0.035, 0.16), 0.01, 0.24))
        height = float(np.clip(rng.uniform(0.025, 0.14), 0.01, 0.24))
        box: dict[str, object] = {
            "id": index,
            "x": float(np.clip(base_x, 0.0, 0.92)),
            "y": float(np.clip(base_y, 0.0, 0.92)),
            "w": width,
            "h": height,
        }
        candidates.append((float(1000 - index), box))

    scalar: list[tuple[float, dict[str, object]]] = []
    for item in candidates:
        if any(_iou(item[1], kept[1]) > 0.45 for kept in scalar):
            continue
        scalar.append(item)

    vectorized = _deduplicate_ranked_candidates(candidates, iou_threshold=0.45)
    assert [int(item[1]["id"]) for item in vectorized] == [int(item[1]["id"]) for item in scalar]



def test_bounded_hysteresis_bridges_only_nearby_weak_crack_pixels():
    strong = np.zeros((40, 80), bool)
    weak = np.zeros_like(strong)
    strong[20, 12:18] = True
    weak[20, 12:25] = True
    weak[20, 40:65] = True  # unrelated weak line must not be reached
    mask = _bounded_hysteresis_mask(strong, weak, max_steps=7)
    assert np.all(mask[20, 12:25] > 0)
    assert not np.any(mask[20, 40:65] > 0)


def test_archive_reference_recovers_low_contrast_crack_segments():
    fixture = Path(__file__).parent / "fixtures" / "archive_low_contrast_cracks_user_reference.png"
    rgb = np.asarray(Image.open(fixture).convert("RGB"), dtype=np.uint8)
    result = detect_surface_defects(rgb, max_boxes=64, max_ai_boxes=128)
    low = [box for box in result.boxes_norm if box.get("detection_branch") == "low_contrast_hough"]
    assert low

    def hits(x0: float, y0: float, x1: float, y1: float) -> bool:
        for box in low:
            bx0 = float(box["x"]); by0 = float(box["y"])
            bx1 = bx0 + float(box["w"]); by1 = by0 + float(box["h"])
            if min(bx1, x1) > max(bx0, x0) and min(by1, y1) > max(by0, y0):
                return True
        return False

    # Visible pale crack across the woman's face and its continuation into the
    # light background were the concrete false-negative reported by the user.
    assert hits(0.30, 0.12, 0.54, 0.22)
    assert hits(0.54, 0.16, 0.70, 0.33)
    # Strong lower-body archival crack should still be represented as a line chain.
    assert hits(0.46, 0.78, 0.76, 0.98)


def test_archive_reference_post_ai_repair_reaches_face_and_light_background():
    fixture = Path(__file__).parent / "fixtures" / "archive_low_contrast_cracks_user_reference.png"
    result = analyze_file(fixture, precision="normal")
    validation = result.metrics["recommendation_validation"].raw_value
    item = next(row for row in validation["items"] if row.get("action_key") == "surface_defects")
    assert item["accepted"] is False
    assert item["auto_eligible"] is False
    assert item["technical_passed"] is None
    assert "Автолечение отключено" in str(item.get("message", ""))
    assert 0.90 <= float(item["default_strength"]) <= 1.0
    candidates = item.get("source_candidates", [])
    assert isinstance(candidates, list) and candidates

    def hits(zone):
        x0, y0, x1, y1 = zone
        for box in candidates:
            bx0 = float(box["x"]); by0 = float(box["y"])
            bx1 = bx0 + float(box["w"]); by1 = by0 + float(box["h"])
            if min(bx1, x1) > max(bx0, x0) and min(by1, y1) > max(by0, y0):
                return True
        return False

    assert hits((0.30, 0.12, 0.54, 0.22))  # woman's damaged face
    assert hits((0.54, 0.16, 0.70, 0.33))  # pale continuation over light background
    refinement = result.metrics["surface_refinement"].raw_value
    assert int(refinement.get("promoted_confirmed_count", 0) or 0) >= 1
    assert int(refinement.get("unprocessed_count", 0) or 0) == 0
    regressions = item["regressions"]
    assert float(regressions["repair_trial_deferred_until_manual_selection"]) == 1.0
    assert float(regressions["candidate_count"]) >= 1.0


def test_archive_reference_surface_v9_defers_repair_until_manual_selection():
    fixture = Path(__file__).parent / "fixtures" / "archive_low_contrast_cracks_user_reference.png"
    result = analyze_file(fixture, precision="normal")
    validation = result.metrics["recommendation_validation"].raw_value
    item = next(row for row in validation["items"] if row.get("action_key") == "surface_defects")
    regressions = item["regressions"]
    assert item["accepted"] is False
    assert item["auto_eligible"] is False
    assert item["technical_passed"] is None
    assert float(regressions["repair_trial_deferred_until_manual_selection"]) == 1.0
    assert any(
        box.get("detection_branch") == "broad_bright_ridge"
        for box in item.get("source_candidates", [])
    )


def test_surface_v9_finds_faint_interrupted_bright_scratch_on_gradient():
    h, w = 480, 720
    yy, xx = np.indices((h, w))
    base = (132 + 32 * xx / (w - 1) + 12 * yy / (h - 1)).astype(np.uint8)
    rgb = np.repeat(base[..., None], 3, axis=2)
    for i in range(420):
        if (i // 22) % 5 == 4:
            continue
        x = 100 + i
        y = int(390 - i * 0.55)
        if 1 <= y < h - 1 and 1 <= x < w - 1:
            value = min(255, int(base[y, x]) + 17)
            rgb[y - 1:y + 2, x - 1:x + 2] = value

    result = detect_surface_defects(rgb, max_boxes=64, max_ai_boxes=128, precision="normal")
    assert result.candidate_count >= 3
    # At least one candidate must hit the central portion of the interrupted crack.
    assert any(
        float(box["x"]) < 0.55
        and float(box["x"]) + float(box["w"]) > 0.35
        and float(box["y"]) < 0.68
        and float(box["y"]) + float(box["h"]) > 0.45
        for box in result.ai_boxes_norm
    )


def test_maximum_surface_merge_keeps_distinct_precise_candidate():
    precise_only = {
        "x": 0.10, "y": 0.12, "w": 0.05, "h": 0.25,
        "candidate_quality": 0.72, "strength": 0.66, "oriented_length_px": 160.0,
    }
    maximum_only = {
        "x": 0.72, "y": 0.55, "w": 0.18, "h": 0.03,
        "candidate_quality": 0.80, "strength": 0.70, "oriented_length_px": 210.0,
    }
    precise = SurfaceDefectDetection(0.1, 1, 20.0, 0.45, 90.0, [precise_only], [precise_only])
    maximum = SurfaceDefectDetection(0.2, 1, 18.0, 0.46, 88.0, [maximum_only], [maximum_only])
    merged = merge_surface_detections(maximum, precise, max_boxes=8, max_ai_boxes=16)
    assert merged.candidate_count == 2
    assert len(merged.ai_boxes_norm) == 2
    assert any(abs(float(box["x"]) - 0.10) < 1e-9 for box in merged.ai_boxes_norm)
    assert any(abs(float(box["x"]) - 0.72) < 1e-9 for box in merged.ai_boxes_norm)
