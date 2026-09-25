import json
import zipfile
from pathlib import Path

from PIL import Image

from photodoctor.ai.parameter_recommender import FEATURE_SCHEMA, MODEL_ID
from photodoctor.core.database import AnalysisDatabase
from photodoctor.core.training_export import export_training_package


def _candidate():
    return {
        "x": 0.25, "y": 0.25, "w": 0.2, "h": 0.08,
        "strength": 0.8, "polarity": "bright",
        "ai_label": "defect", "ai_model_id": "native_surface_refiner_v1", "ai_confidence": 0.91,
    }


def test_temp_database_export_keeps_user_feedback_separate_from_expert_ground_truth(tmp_path):
    source = tmp_path / "private_photo.png"
    Image.new("RGB", (320, 240), (120, 110, 100)).save(source)
    db_path = tmp_path / "analysis.sqlite3"
    item = {
        "action_key": "contrast", "tested": True, "accepted": True, "auto_eligible": True,
        "candidate": "mild_lab_clahe", "confidence": 0.83, "adjustable": True, "default_strength": 0.4,
    }
    suggestion = {
        "suggested_strength": 0.31, "confidence": 0.76,
        "model_id": MODEL_ID, "feature_schema": FEATURE_SCHEMA,
        "features": [0.4, 0.83, 1.0, 0.6, 0.7, 0.8, 0.5, 0.3, 1.0, 0.0],
    }
    with AnalysisDatabase(db_path) as db:
        db.record_correction_feedback(
            source, [item], {"contrast"}, saved_format="jpg",
            user_strengths={"contrast": 0.36},
            parameter_suggestions={"contrast": suggestion},
        )
        db.save_surface_feedback(source, _candidate(), "natural_detail")

    out_zip = tmp_path / "training.zip"
    summary = export_training_package(db_path, out_zip)
    assert summary.parameter_exported == 1
    assert summary.surface_exported == 1

    with zipfile.ZipFile(out_zip) as zf:
        parameter = json.loads(zf.read("PhotoDoctor_AI_Training_Data/parameter_feedback/manifest.jsonl").decode("utf-8").splitlines()[0])
        surface = json.loads(zf.read("PhotoDoctor_AI_Training_Data/surface_feedback/manifest.jsonl").decode("utf-8").splitlines()[0])
        names = zf.namelist()

    assert parameter["source_tier"] == "user_feedback"
    assert parameter["training_lane"] == "preference"
    assert parameter["expert_verified"] is False
    assert parameter["feature_schema"] == FEATURE_SCHEMA

    assert surface["label"] == "N"
    assert surface["source_tier"] == "user_feedback"
    assert surface["training_lane"] == "hard_mining"
    assert surface["label_origin"] == "user_manual"
    assert surface["expert_verified"] is False
    assert surface["use_scope"] == "preference_hard_case_only"

    archive_text = "\
".join(names)
    assert str(source.resolve()) not in archive_text
    assert source.name not in archive_text
