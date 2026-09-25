from pathlib import Path

from PIL import Image

from photodoctor.core.database import AnalysisDatabase, surface_candidate_signature


def _image(path: Path, value: int = 120):
    Image.new("RGB", (64, 48), (value, value, value)).save(path)


def _candidate():
    return {
        "x": 0.125,
        "y": 0.25,
        "w": 0.08,
        "h": 0.015,
        "strength": 0.74,
        "polarity": "bright",
        "ai_label": "natural_detail",
        "ai_model_id": "native_surface_refiner_v1",
        "ai_confidence": 0.83,
    }


def test_surface_feedback_roundtrip_and_delete(tmp_path):
    path = tmp_path / "photo.png"
    _image(path)
    candidate = _candidate()
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        signature = db.save_surface_feedback(path, candidate, "defect")
        assert signature == surface_candidate_signature(candidate)
    with AnalysisDatabase(db_path) as db:
        rows = db.load_surface_feedback(path)
        assert rows[signature]["user_label"] == "defect"
        assert rows[signature]["ai_label"] == "natural_detail"
        assert abs(float(rows[signature]["ai_confidence"]) - 0.83) < 1e-9
        db.delete_surface_feedback(path, candidate)
        assert db.load_surface_feedback(path) == {}


def test_surface_feedback_is_bound_to_current_file_hash(tmp_path):
    path = tmp_path / "photo.png"
    _image(path, 100)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        db.save_surface_feedback(path, _candidate(), "natural_detail")
        assert db.load_surface_feedback(path)
    _image(path, 180)
    with AnalysisDatabase(db_path) as db:
        assert db.load_surface_feedback(path) == {}


def test_surface_candidate_signature_ignores_ai_answer_but_not_geometry():
    a = _candidate()
    b = dict(a)
    b["ai_label"] = "defect"
    b["ai_confidence"] = 0.99
    assert surface_candidate_signature(a) == surface_candidate_signature(b)
    b["x"] = 0.2
    assert surface_candidate_signature(a) != surface_candidate_signature(b)


def test_surface_feedback_rejects_unknown_label(tmp_path):
    path = tmp_path / "photo.png"
    _image(path)
    with AnalysisDatabase(tmp_path / "analysis.sqlite3") as db:
        try:
            db.save_surface_feedback(path, _candidate(), "maybe")
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported label must be rejected")


def test_surface_feedback_stats_use_human_labels_as_reference(tmp_path):
    path = tmp_path / "photo.png"
    _image(path)
    db_path = tmp_path / "analysis.sqlite3"
    cases = [
        ("defect", "defect", 0.90),
        ("defect", "natural_detail", 0.85),
        ("natural_detail", "defect", 0.95),
        ("natural_detail", "natural_detail", 0.70),
        ("uncertain", "defect", 0.92),
        ("defect", "uncertain", 0.55),
    ]
    with AnalysisDatabase(db_path) as db:
        for i, (user, ai, conf) in enumerate(cases):
            c = _candidate()
            c["x"] = 0.05 + i * 0.1
            c["ai_label"] = ai
            c["ai_confidence"] = conf
            db.save_surface_feedback(path, c, user)
        stats = db.surface_feedback_stats()
    row = next(item for item in stats if item["model_id"] == "native_surface_refiner_v1")
    assert row["labeled"] == 6
    assert row["binary_evaluated"] == 4
    assert row["correct"] == 2
    assert row["wrong"] == 2
    assert row["user_uncertain"] == 1
    assert row["ai_uncertain_or_missing"] == 1
    assert row["tp_defect"] == 1
    assert row["fn_defect_as_natural"] == 1
    assert row["fp_natural_as_defect"] == 1
    assert row["tn_natural"] == 1
    assert row["high_confidence_wrong"] == 2
    assert abs(float(row["agreement_pct"]) - 50.0) < 1e-9
    assert abs(float(row["defect_recall_pct"]) - 50.0) < 1e-9
    assert abs(float(row["natural_recall_pct"]) - 50.0) < 1e-9


def test_surface_feedback_schema_has_context_fields_and_schema_v9(tmp_path):
    with AnalysisDatabase(tmp_path / "analysis.sqlite3") as db:
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(surface_feedback)").fetchall()}
        assert "ai_model_id" in columns
        assert {"raw_ai_label", "raw_ai_confidence", "verification_label", "verification_confidence",
                "candidate_quality", "candidate_kind", "context_support", "context_risk"}.issubset(columns)
        version = db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        assert version == "9"


def test_surface_feedback_stats_measure_raw_model_not_fused_verdict(tmp_path):
    path = tmp_path / "photo.png"
    _image(path)
    candidate = _candidate()
    candidate.update({
        "ai_raw_label": "defect",
        "ai_raw_confidence": 0.94,
        "verification_label": "natural_detail",
        "verification_confidence": 0.88,
        "ai_label": "natural_detail",
        "ai_confidence": 0.88,
        "candidate_quality": 0.22,
        "candidate_kind": "line",
        "context_support": 0.18,
        "context_risk": 0.81,
    })
    with AnalysisDatabase(tmp_path / "analysis.sqlite3") as db:
        db.save_surface_feedback(path, candidate, "natural_detail")
        loaded = next(iter(db.load_surface_feedback(path).values()))
        assert loaded["raw_ai_label"] == "defect"
        assert loaded["verification_label"] == "natural_detail"
        stats = db.surface_feedback_stats()
    row = next(item for item in stats if item["model_id"] == "native_surface_refiner_v1")
    assert row["wrong"] == 1
    assert row["fp_natural_as_defect"] == 1
    assert row["high_confidence_wrong"] == 1
