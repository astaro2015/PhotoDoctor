import json
from pathlib import Path

from PIL import Image

from photodoctor.core.database import AnalysisDatabase
from photodoctor.core.feedback_export import _candidate_crop, export_surface_feedback_dataset


def _candidate():
    return {
        "x": 0.25, "y": 0.25, "w": 0.2, "h": 0.08,
        "strength": 0.8, "polarity": "bright",
        "ai_label": "natural_detail", "ai_model_id": "native_surface_refiner_v1", "ai_confidence": 0.84,
        "ai_raw_label": "defect", "ai_raw_confidence": 0.91,
        "verification_label": "natural_detail", "verification_confidence": 0.84,
        "candidate_quality": 0.24, "candidate_kind": "line", "context_support": 0.20, "context_risk": 0.78,
    }


def test_feedback_export_creates_private_96px_dataset(tmp_path):
    source = tmp_path / "family photo.png"
    Image.new("RGB", (320, 240), (130, 120, 110)).save(source)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        db.save_surface_feedback(source, _candidate(), "natural_detail")
    out = tmp_path / "dataset"
    summary = export_surface_feedback_dataset(db_path, out)
    assert summary.exported == 1
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["user_label"] == "natural_detail"
    assert row["ai_label"] == "natural_detail"
    assert row["raw_ai_label"] == "defect"
    assert row["verification_label"] == "natural_detail"
    assert row["candidate_kind"] == "line"
    assert row["ai_model_id"] == "native_surface_refiner_v1"
    assert row["label"] == "N"
    assert row["source_tier"] == "user_feedback"
    assert row["label_origin"] == "user_manual"
    assert row["expert_verified"] is False
    assert row["training_lane"] == "hard_mining"
    assert row["license"] == "private-local-only"
    assert row["use_scope"] == "preference_hard_case_only"
    assert row["source_group_id"] == row["source_quick_hash"]
    assert row["source_name"] == source.name
    manifest_text = (out / "manifest.jsonl").read_text(encoding="utf-8")
    assert str(source.resolve()) not in manifest_text
    patch = Image.open(out / row["patch"])
    assert patch.size == (96, 96)
    context_patch = Image.open(out / row["context_patch"])
    assert context_patch.size == (256, 256)
    assert row["context_feature_schema"] == "rgb256_candidate_context_v1"
    candidate_box = row["candidate_box_in_context"]
    assert 0.0 <= candidate_box["x"] <= 1.0
    assert 0.0 <= candidate_box["y"] <= 1.0
    assert 0.0 < candidate_box["w"] <= 1.0
    assert 0.0 < candidate_box["h"] <= 1.0
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["exported"] == 1


def test_feedback_export_skips_changed_source(tmp_path):
    source = tmp_path / "photo.png"
    Image.new("RGB", (100, 100), (100, 100, 100)).save(source)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        db.save_surface_feedback(source, _candidate(), "defect")
    Image.new("RGB", (100, 100), (180, 180, 180)).save(source)
    summary = export_surface_feedback_dataset(db_path, tmp_path / "out")
    assert summary.exported == 0
    assert summary.skipped_changed == 1


def test_feedback_crop_rejects_nonfinite_geometry():
    import numpy as np
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    assert _candidate_crop(rgb, {"x": float("nan"), "y": 0.1, "w": 0.2, "h": 0.2}) is None
    assert _candidate_crop(rgb, {"x": 0.1, "y": 0.1, "w": float("inf"), "h": 0.2}) is None
