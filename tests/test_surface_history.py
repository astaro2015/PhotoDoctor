from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from photodoctor.core.exporter import save_rgb_copy
from photodoctor.core.models import MetricResult
from photodoctor.core.surface_history import merge_surface_history, read_surface_history
from photodoctor.core.validator import _linked_surface_repair_mask, refresh_surface_validation_after_refinement


def _line(x1: float, x2: float, y: float = 0.5, confidence: float = 0.74) -> dict:
    half = 0.012
    return {
        "x": x1, "y": y - half, "w": x2 - x1, "h": half * 2,
        "contour": [
            {"x": x1, "y": y - half}, {"x": x2, "y": y - half},
            {"x": x2, "y": y + half}, {"x": x1, "y": y + half},
        ],
        "line_endpoints": [{"x": x1, "y": y}, {"x": x2, "y": y}],
        "polarity": "bright", "candidate_kind": "line", "detection_branch": "low_contrast_hough",
        "orientation_deg": 0.0, "stroke_width_px": 2.4, "candidate_quality": 0.68,
        "semantic_risk": 0.0, "verification_label": "defect", "verification_confidence": confidence,
        "ai_label": "defect", "ai_confidence": confidence,
    }


def _metrics(boxes: list[dict]) -> dict[str, MetricResult]:
    return {
        "surface_refinement": MetricResult(
            "surface_refinement", {"refined_boxes_norm": boxes}, None, 0.8, "context_ai_advisory", region="surface"
        ),
        "recommendation_validation": MetricResult(
            "recommendation_validation",
            {"items": [{
                "action_key": "surface_defects", "accepted": False, "auto_eligible": False,
                "technical_passed": False, "source_candidates": [], "preview_available": False,
                "regressions": {}, "candidate": "bounded_surface_heal", "tested": True,
            }]},
            None, 0.7, "technical_copy", region="decision",
        ),
    }


def test_surface_history_roundtrips_png_and_jpeg_and_accumulates(tmp_path: Path):
    rgb = np.full((80, 120, 3), 112, np.uint8)
    for suffix in (".png", ".jpg"):
        source = tmp_path / f"source{suffix}"
        Image.fromarray(rgb).save(source)
        first = merge_surface_history({}, [_line(0.10, 0.35)])
        out1 = save_rgb_copy(rgb, tmp_path / f"pass1{suffix}", source_path=source, photo_doctor_surface_history=first)
        loaded1 = read_surface_history(out1)
        assert loaded1["passes"] == 1
        assert len(loaded1["candidates"]) == 1

        second = merge_surface_history(loaded1, [_line(0.45, 0.70)])
        out2 = save_rgb_copy(rgb, tmp_path / f"pass2{suffix}", source_path=out1, photo_doctor_surface_history=second)
        loaded2 = read_surface_history(out2)
        assert loaded2["passes"] == 2
        assert len(loaded2["candidates"]) == 2


def test_surface_v7_suppresses_previously_repaired_overlap_but_not_adjacent_tail():
    rgb = np.full((100, 100, 3), 90, np.uint8)
    old = _line(0.10, 0.40)
    same = _line(0.11, 0.39)
    tail = _line(0.42, 0.72)
    history = {"version": 1, "passes": 1, "candidates": [old]}
    metrics = _metrics([same, tail])

    refresh_surface_validation_after_refinement(rgb, metrics, surface_history=history)

    refined = metrics["surface_refinement"].raw_value["refined_boxes_norm"]
    assert refined[0]["previously_repaired"] is True
    assert refined[1]["previously_repaired"] is False
    item = metrics["recommendation_validation"].raw_value["items"][0]
    assert len(item["source_candidates"]) == 1
    assert item["source_candidates"][0]["previously_repaired"] is False
    assert item["regressions"]["history_filtered_count"] == 1.0


def test_surface_v7_small_weak_residual_after_prior_pass_becomes_manual_only():
    rgb = np.full((120, 120, 3), 100, np.uint8)
    candidate = _line(0.55, 0.61, y=0.55, confidence=0.68)
    history = {"version": 1, "passes": 1, "candidates": [_line(0.10, 0.25, y=0.2)]}
    metrics = _metrics([candidate])

    refresh_surface_validation_after_refinement(rgb, metrics, surface_history=history)
    item = metrics["recommendation_validation"].raw_value["items"][0]
    assert item["accepted"] is False
    assert item["auto_eligible"] is False
    assert item["regressions"]["residual_manual_only"] == 1.0
    assert "ручной проверки" in item["message"]


def test_surface_v7_bridges_short_bright_gap_between_same_crack_segments():
    rgb = np.full((100, 120, 3), 65, np.uint8)
    cv2.line(rgb, (20, 50), (85, 50), (160, 160, 160), 2, cv2.LINE_AA)
    a = _line(20 / 120, 40 / 120)
    b = _line(45 / 120, 70 / 120)
    a["stroke_width_px"] = 2.0
    b["stroke_width_px"] = 2.0
    mask = _linked_surface_repair_mask(rgb, [a, b])
    assert mask[50, 42] > 0
    assert mask[50, 44] > 0


def test_surface_v7_after_two_passes_three_tiny_residuals_are_manual_only():
    rgb = np.full((160, 160, 3), 100, np.uint8)
    boxes = [
        _line(0.20, 0.215, y=0.45, confidence=0.70),
        _line(0.40, 0.415, y=0.55, confidence=0.72),
        _line(0.62, 0.635, y=0.65, confidence=0.74),
    ]
    for box in boxes:
        y = float(box["line_endpoints"][0]["y"])
        x1 = float(box["line_endpoints"][0]["x"])
        x2 = float(box["line_endpoints"][1]["x"])
        half = 0.003
        box["y"] = y - half
        box["h"] = half * 2
        box["contour"] = [
            {"x": x1, "y": y - half}, {"x": x2, "y": y - half},
            {"x": x2, "y": y + half}, {"x": x1, "y": y + half},
        ]
    history = {"version": 1, "passes": 2, "candidates": [_line(0.05, 0.10, y=0.15)]}
    metrics = _metrics(boxes)
    refresh_surface_validation_after_refinement(rgb, metrics, surface_history=history)
    item = metrics["recommendation_validation"].raw_value["items"][0]
    assert item["accepted"] is False
    assert item["auto_eligible"] is False
    assert item["regressions"]["residual_manual_only"] == 1.0


def test_surface_v8_bridges_long_collinear_gap_when_ridge_continues():
    h, w = 800, 1000
    rgb = np.full((h, w, 3), 92, np.uint8)
    cv2.line(rgb, (120, 390), (820, 390), (165, 165, 165), 3, cv2.LINE_AA)
    a = _line(120 / w, 390 / w, y=390 / h)
    b = _line(445 / w, 720 / w, y=390 / h)
    a["stroke_width_px"] = 3.0
    b["stroke_width_px"] = 3.0
    mask = _linked_surface_repair_mask(rgb, [a, b])
    # v7 stopped at 18 px. v8 must bridge this 55 px gap only because the same
    # two-sided bright ridge is present all the way through it.
    assert mask[390, 415] > 0
    assert mask[390, 435] > 0


def test_surface_v8_adaptive_width_covers_broad_white_crack_but_not_whole_neighbourhood():
    h, w = 220, 280
    rgb = np.full((h, w, 3), 105, np.uint8)
    cv2.line(rgb, (35, 110), (245, 110), (198, 198, 198), 9, cv2.LINE_AA)
    cand = _line(35 / w, 245 / w, y=110 / h)
    cand["stroke_width_px"] = 2.0  # detector seed intentionally underestimates width
    mask = _linked_surface_repair_mask(rgb, [cand])
    assert mask[106, 140] > 0
    assert mask[114, 140] > 0
    assert mask[100, 140] == 0
    assert mask[120, 140] == 0


def test_surface_v8_adaptive_growth_does_not_spread_into_one_sided_brightness_edge():
    h, w = 220, 280
    rgb = np.full((h, w, 3), 75, np.uint8)
    rgb[:, 142:] = 180
    cand = _line(0.49, 0.51, y=0.50)
    cand["line_endpoints"] = [{"x": 0.50, "y": 0.20}, {"x": 0.50, "y": 0.80}]
    cand["orientation_deg"] = 90.0
    cand["x"] = 0.495; cand["y"] = 0.20; cand["w"] = 0.01; cand["h"] = 0.60
    cand["contour"] = [
        {"x": 0.495, "y": 0.20}, {"x": 0.505, "y": 0.20},
        {"x": 0.505, "y": 0.80}, {"x": 0.495, "y": 0.80},
    ]
    mask = _linked_surface_repair_mask(rgb, [cand])
    # Seed stays bounded around the supplied candidate instead of flooding the
    # whole bright half-plane.
    assert np.count_nonzero(mask[:, 160:]) == 0


def test_surface_v8_strong_heal_reduces_bright_crack_contrast():
    from photodoctor.core.validator import _bounded_surface_heal, _surface_repair_effectiveness
    h, w = 260, 320
    rgb = np.full((h, w, 3), 112, np.uint8)
    cv2.line(rgb, (45, 130), (275, 130), (220, 220, 220), 7, cv2.LINE_AA)
    cand = _line(45 / w, 275 / w, y=130 / h, confidence=0.90)
    cand["stroke_width_px"] = 2.0
    mask = _linked_surface_repair_mask(rgb, [cand])
    healed = _bounded_surface_heal(rgb, [cand], 0.90)
    before, after, reduction = _surface_repair_effectiveness(rgb, healed, mask)
    assert before > 8.0
    assert after < before
    assert reduction >= 0.45
