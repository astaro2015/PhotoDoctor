import json
import zipfile
from pathlib import Path

from PIL import Image

from photodoctor.core.database import AnalysisDatabase
from photodoctor.ai.parameter_recommender import FEATURE_SCHEMA, MODEL_ID
from photodoctor.core.training_export import (
    PARAMETER_FIRST_EXPERIMENT_MIN,
    MIN_TRAINING_SOURCE_GROUPS,
    collect_training_data_stats,
    export_training_package,
)


def _photo(path: Path, color=(110, 120, 130)) -> None:
    Image.new("RGB", (240, 180), color).save(path)


def _candidate(index: int, *, ai_label="defect") -> dict[str, object]:
    return {
        "x": 0.05 + (index % 8) * 0.1,
        "y": 0.05 + ((index // 8) % 6) * 0.1,
        "w": 0.04,
        "h": 0.03,
        "strength": 0.5 + (index % 3) * 0.1,
        "polarity": "bright" if index % 2 else "dark",
        "ai_label": ai_label,
        "ai_model_id": "native_surface_refiner_v1",
        "ai_confidence": 0.8,
    }


def _record_strength(db: AnalysisDatabase, source: Path, action: str, strength: float) -> None:
    item = {
        "action_key": action,
        "tested": True,
        "accepted": True,
        "auto_eligible": True,
        "candidate": "mild_lab_clahe" if action == "contrast" else "edge_aware_unsharp",
        "confidence": 0.8,
        "adjustable": True,
        "default_strength": 0.3,
    }
    db.record_correction_feedback(
        source,
        [item],
        {action},
        saved_format="jpg",
        user_strengths={action: strength},
        parameter_suggestions={action: {
            "suggested_strength": 0.25,
            "confidence": 0.7,
            "model_id": MODEL_ID,
            "feature_schema": FEATURE_SCHEMA,
            "features": [0.2, 0.4, 0.6, 0.8],
        }},
    )


def test_training_stats_are_empty_for_fresh_database(tmp_path):
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path):
        pass
    stats = collect_training_data_stats(db_path)
    assert stats.meaningful_correction_decisions == 0
    assert stats.parameter_strength_samples == 0
    assert stats.parameter_by_action == {}
    assert stats.parameter_source_groups_by_action == {}
    assert stats.surface_source_groups == 0
    assert stats.surface_defect == 0
    assert stats.surface_natural == 0
    assert stats.surface_uncertain == 0
    assert stats.parameter_ready_actions == ()
    assert stats.surface_ready is False


def test_parameter_readiness_counts_real_saved_strengths_and_independent_sources(tmp_path):
    db_path = tmp_path / "analysis.sqlite3"
    sources = []
    for index in range(PARAMETER_FIRST_EXPERIMENT_MIN):
        source = tmp_path / f"photo_{index:02d}.png"
        _photo(source, (80 + index, 100, 120))
        sources.append(source)
    with AnalysisDatabase(db_path) as db:
        for index, source in enumerate(sources):
            _record_strength(db, source, "contrast", 0.2 + index * 0.001)
        for _ in range(7):
            _record_strength(db, sources[0], "sharpness", 0.3)
    stats = collect_training_data_stats(db_path)
    assert stats.parameter_by_action["contrast"] == PARAMETER_FIRST_EXPERIMENT_MIN
    assert stats.parameter_source_groups_by_action["contrast"] >= MIN_TRAINING_SOURCE_GROUPS
    assert stats.parameter_by_action["sharpness"] == 7
    assert stats.parameter_source_groups_by_action["sharpness"] == 1
    assert stats.parameter_strength_samples == PARAMETER_FIRST_EXPERIMENT_MIN + 7
    assert stats.parameter_ready_actions == ("contrast",)


def test_repeated_saves_of_one_photo_do_not_fake_parameter_readiness(tmp_path):
    source = tmp_path / "same_photo.png"
    _photo(source)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        for index in range(PARAMETER_FIRST_EXPERIMENT_MIN + 10):
            _record_strength(db, source, "contrast", 0.2 + (index % 10) * 0.001)
    stats = collect_training_data_stats(db_path)
    assert stats.parameter_by_action["contrast"] >= PARAMETER_FIRST_EXPERIMENT_MIN
    assert stats.parameter_source_groups_by_action["contrast"] == 1
    assert stats.parameter_max_source_group_share_by_action["contrast"] == 1.0
    assert stats.parameter_ready_actions == ()


def test_training_package_is_privacy_safe_and_contains_both_feedback_types(tmp_path):
    source = tmp_path / "VERY SECRET FAMILY NAME.jpg"
    _photo(source)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        _record_strength(db, source, "contrast", 0.31)
        db.save_surface_feedback(source, _candidate(1), "defect")
        db.save_surface_feedback(source, _candidate(2, ai_label="natural_detail"), "natural_detail")
        db.save_surface_feedback(source, _candidate(3), "uncertain")

    target = tmp_path / "training package.zip"
    summary = export_training_package(db_path, target)
    assert summary.output_zip == target
    assert summary.parameter_exported == 1
    assert summary.surface_exported == 3

    with zipfile.ZipFile(target) as archive:
        names = set(archive.namelist())
        assert "PhotoDoctor_AI_Training_Data/parameter_feedback/manifest.jsonl" in names
        assert "PhotoDoctor_AI_Training_Data/surface_feedback/manifest.jsonl" in names
        assert "PhotoDoctor_AI_Training_Data/summary.json" in names
        assert "PhotoDoctor_AI_Training_Data/dataset_info.json" in names
        parameter_text = archive.read("PhotoDoctor_AI_Training_Data/parameter_feedback/manifest.jsonl").decode("utf-8")
        surface_text = archive.read("PhotoDoctor_AI_Training_Data/surface_feedback/manifest.jsonl").decode("utf-8")
        all_text = parameter_text + surface_text + archive.read("PhotoDoctor_AI_Training_Data/README.txt").decode("utf-8")
        assert str(source.resolve()) not in all_text
        assert source.name not in all_text
        parameter = json.loads(parameter_text.strip())
        assert parameter["action_key"] == "contrast"
        assert parameter["user_strength"] == 0.31
        assert parameter["parameter_features"] == [0.2, 0.4, 0.6, 0.8]
        assert parameter["feature_schema"] == FEATURE_SCHEMA
        assert parameter["training_lane"] == "preference"
        assert parameter["source_tier"] == "user_feedback"
        surface_rows = [json.loads(line) for line in surface_text.splitlines()]
        assert {row["user_label"] for row in surface_rows} == {"defect", "natural_detail", "uncertain"}
        assert all("source_name" not in row for row in surface_rows)
        patch_names = [name for name in names if "/surface_feedback/patches/" in name]
        context_names = [name for name in names if "/surface_feedback/context_patches/" in name]
        assert len(patch_names) == 3
        assert len(context_names) == 3
        assert all(row["context_feature_schema"] == "rgb256_candidate_context_v1" for row in surface_rows)


def test_training_package_skips_surface_patch_if_source_changed_but_keeps_parameter_feedback(tmp_path):
    source = tmp_path / "photo.jpg"
    _photo(source, (70, 80, 90))
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        _record_strength(db, source, "contrast", 0.27)
        db.save_surface_feedback(source, _candidate(4), "defect")
    _photo(source, (180, 180, 180))
    summary = export_training_package(db_path, tmp_path / "training.zip")
    assert summary.parameter_exported == 1
    assert summary.surface_exported == 0
    assert summary.surface_skipped_changed == 1


def test_many_surface_labels_from_one_photo_do_not_fake_surface_training_readiness(tmp_path):
    source = tmp_path / "one_surface_source.png"
    _photo(source)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        for index in range(220):
            candidate = {
                "x": (index % 20) / 25.0,
                "y": ((index // 20) % 11) / 14.0,
                "w": 0.012 + (index % 3) * 0.001,
                "h": 0.010 + (index % 5) * 0.001,
                "strength": 0.4 + (index % 7) * 0.01,
                "polarity": "bright" if index % 2 else "dark",
                "ai_label": "defect",
                "ai_model_id": "native_surface_refiner_v1",
                "ai_confidence": 0.8,
            }
            db.save_surface_feedback(source, candidate, "defect" if index % 2 else "natural_detail")
    stats = collect_training_data_stats(db_path)
    assert stats.surface_defect >= 50
    assert stats.surface_natural >= 50
    assert stats.surface_total >= 200
    assert stats.surface_source_groups == 1
    assert stats.surface_max_source_group_share == 1.0
    assert stats.surface_ready is False
