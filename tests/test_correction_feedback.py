from PIL import Image

from photodoctor.core.database import AnalysisDatabase


def _photo(path):
    Image.new("RGB", (64, 48), (90, 100, 110)).save(path)


def test_successful_save_feedback_records_keep_disable_and_manual_review(tmp_path):
    source = tmp_path / "source.png"
    _photo(source)
    items = [
        {"action_key": "exposure", "tested": True, "accepted": True, "auto_eligible": True, "candidate": "gamma=0.92", "confidence": 0.8},
        {"action_key": "contrast", "tested": True, "accepted": True, "auto_eligible": True, "candidate": "mild_lab_clahe", "confidence": 0.7},
        {"action_key": "red_eye", "tested": True, "accepted": False, "auto_eligible": False, "candidate": "localized_pupil_neutralization", "confidence": 0.6},
    ]
    with AnalysisDatabase(tmp_path / "feedback.sqlite3") as db:
        count = db.record_correction_feedback(
            source, items, {"exposure", "red_eye"}, saved_format="png", context={"app_version": "0.4.1"}
        )
        rows = db.conn.execute(
            "SELECT action_key,decision_event,user_selected,validator_accepted,context_json FROM correction_feedback ORDER BY action_key"
        ).fetchall()
        stats = {row["action_key"]: row for row in db.correction_feedback_stats()}
    assert count == 3
    events = {row["action_key"]: row["decision_event"] for row in rows}
    assert events == {
        "contrast": "user_disabled_saved",
        "exposure": "auto_accepted_saved",
        "red_eye": "user_enabled_review_saved",
    }
    assert stats["exposure"]["selected_pct"] == 100.0
    assert stats["contrast"]["selected_pct"] == 0.0
    assert stats["red_eye"]["user_enabled_review"] == 1
    assert '"app_version":"0.4.1"' in rows[0]["context_json"]


def test_rejected_unselected_action_is_context_not_meaningful_preference(tmp_path):
    source = tmp_path / "source.png"
    _photo(source)
    item = {
        "action_key": "contrast", "tested": True, "accepted": False, "auto_eligible": True,
        "candidate": "mild_lab_clahe", "confidence": 0.5,
    }
    with AnalysisDatabase(tmp_path / "feedback.sqlite3") as db:
        db.record_correction_feedback(source, [item], set(), saved_format="jpg")
        row = db.conn.execute("SELECT decision_event FROM correction_feedback").fetchone()
        stats = db.correction_feedback_stats()[0]
    assert row["decision_event"] == "validator_rejected_unselected"
    assert stats["meaningful_decisions"] == 0
    assert stats["selected_pct"] is None


def test_correction_feedback_stores_ai_and_user_strength_training_pair(tmp_path):
    source = tmp_path / "source_strength.png"
    _photo(source)
    item = {
        "action_key": "contrast", "tested": True, "accepted": True, "auto_eligible": True,
        "candidate": "mild_lab_clahe", "confidence": 0.8, "adjustable": True, "default_strength": 0.4,
    }
    suggestion = {
        "suggested_strength": 0.28, "confidence": 0.74,
        "model_id": "native_parameter_recommender_v2", "feature_schema": "parameter_features_v2", "features": [0.4, 0.8, 1.0, 0.6],
    }
    with AnalysisDatabase(tmp_path / "feedback_strength.sqlite3") as db:
        count = db.record_correction_feedback(
            source, [item], {"contrast"}, saved_format="jpg",
            user_strengths={"contrast": 0.36}, parameter_suggestions={"contrast": suggestion},
        )
        row = db.conn.execute(
            "SELECT ai_suggested_strength,user_strength,ai_parameter_model_id,ai_parameter_confidence,parameter_features_json "
            "FROM correction_feedback WHERE action_key='contrast'"
        ).fetchone()
        samples = db.correction_parameter_training_samples(
            "contrast", model_id="native_parameter_recommender_v2", feature_schema="parameter_features_v2"
        )
        stats = db.correction_feedback_stats()[0]
    assert count == 1
    assert row["ai_suggested_strength"] == 0.28
    assert row["user_strength"] == 0.36
    assert row["ai_parameter_model_id"] == "native_parameter_recommender_v2"
    assert row["ai_parameter_confidence"] == 0.74
    assert samples[0]["user_strength"] == 0.36
    assert samples[0]["features"] == [0.4, 0.8, 1.0, 0.6]
    assert stats["strength_samples"] == 1
    assert stats["ai_strength_kept"] == 0
    assert stats["ai_strength_changed"] == 1
    assert abs(float(stats["mean_strength_delta"]) - 0.08) < 1e-9


def test_unselected_action_does_not_become_strength_training_sample(tmp_path):
    source = tmp_path / "source_disabled.png"
    _photo(source)
    item = {
        "action_key": "sharpness", "tested": True, "accepted": True, "auto_eligible": True,
        "candidate": "edge_aware_unsharp", "confidence": 0.8, "adjustable": True, "default_strength": 0.35,
    }
    suggestion = {
        "suggested_strength": 0.30, "confidence": 0.7,
        "model_id": "native_parameter_recommender_v2", "feature_schema": "parameter_features_v2", "features": [0.35] * 10,
    }
    with AnalysisDatabase(tmp_path / "feedback_disabled.sqlite3") as db:
        db.record_correction_feedback(
            source, [item], set(), user_strengths={"sharpness": 0.55},
            parameter_suggestions={"sharpness": suggestion},
        )
        row = db.conn.execute("SELECT user_strength,decision_event FROM correction_feedback").fetchone()
        samples = db.correction_parameter_training_samples("sharpness", model_id="native_parameter_recommender_v2", feature_schema="parameter_features_v2")
    assert row["decision_event"] == "user_disabled_saved"
    assert row["user_strength"] is None
    assert samples == []


def test_schema_v9_migrates_existing_correction_feedback_without_losing_rows(tmp_path):
    import sqlite3
    db_path = tmp_path / "legacy_feedback.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key,value) VALUES('schema_version','6');
        CREATE TABLE correction_feedback(
            id INTEGER PRIMARY KEY,
            save_event_id TEXT NOT NULL,
            path TEXT NOT NULL,
            quick_hash TEXT NOT NULL,
            action_key TEXT NOT NULL,
            candidate TEXT NOT NULL DEFAULT '',
            validator_accepted INTEGER NOT NULL,
            user_selected INTEGER NOT NULL,
            decision_event TEXT NOT NULL,
            validation_confidence REAL,
            validation_json TEXT NOT NULL DEFAULT '{}',
            context_json TEXT NOT NULL DEFAULT '{}',
            algorithm_version TEXT NOT NULL,
            saved_format TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO correction_feedback(
            save_event_id,path,quick_hash,action_key,candidate,validator_accepted,user_selected,
            decision_event,validation_confidence,validation_json,context_json,algorithm_version,saved_format
        ) VALUES('evt','legacy.jpg','abc','contrast','mild_lab_clahe',1,1,'auto_accepted_saved',0.8,'{}','{}','0.4.4','jpg');
        """
    )
    conn.commit(); conn.close()

    with AnalysisDatabase(db_path) as db:
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(correction_feedback)").fetchall()}
        version = db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        count = db.conn.execute("SELECT COUNT(*) FROM correction_feedback").fetchone()[0]
    assert version == "9"
    assert count == 1
    for name in {
        "ai_suggested_strength", "user_strength", "ai_parameter_model_id",
        "ai_parameter_confidence", "parameter_features_json",
        "parameter_feature_schema", "feedback_kind", "source_tier",
    }:
        assert name in columns


def test_schema_v8_preserves_legacy_slider_as_v1_only(tmp_path):
    import json
    import sqlite3
    from photodoctor.ai.parameter_recommender import FEATURE_SCHEMA, LEGACY_FEATURE_SCHEMA, LEGACY_MODEL_ID, MODEL_ID

    db_path = tmp_path / "legacy_strength.sqlite3"
    context = {
        "metrics": {"contrast": {"normalized_value": 40.0, "confidence": 0.8, "diagnostic": "low"}},
        "decision_plan": [{
            "key": "contrast", "decision": "fix", "severity": 55.0,
            "repairability": 72.0, "confidence": 0.8, "priority": 44.0,
        }],
        "correction_strengths": {"contrast": 0.33},
    }
    validation = {
        "action_key": "contrast", "tested": True, "accepted": True,
        "adjustable": True, "default_strength": 0.40, "confidence": 0.8,
    }
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key,value) VALUES('schema_version','6');
        CREATE TABLE correction_feedback(
            id INTEGER PRIMARY KEY,
            save_event_id TEXT NOT NULL,
            path TEXT NOT NULL,
            quick_hash TEXT NOT NULL,
            action_key TEXT NOT NULL,
            candidate TEXT NOT NULL DEFAULT '',
            validator_accepted INTEGER NOT NULL,
            user_selected INTEGER NOT NULL,
            decision_event TEXT NOT NULL,
            validation_confidence REAL,
            validation_json TEXT NOT NULL DEFAULT '{}',
            context_json TEXT NOT NULL DEFAULT '{}',
            algorithm_version TEXT NOT NULL,
            saved_format TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute(
        """INSERT INTO correction_feedback(
            save_event_id,path,quick_hash,action_key,candidate,validator_accepted,user_selected,
            decision_event,validation_confidence,validation_json,context_json,algorithm_version,saved_format
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            "evt", "legacy.jpg", "abc", "contrast", "mild_lab_clahe", 1, 1,
            "auto_accepted_saved", 0.8, json.dumps(validation), json.dumps(context), "0.4.4", "jpg",
        ),
    )
    conn.commit(); conn.close()

    with AnalysisDatabase(db_path) as db:
        legacy_samples = db.correction_parameter_training_samples(
            "contrast", model_id=LEGACY_MODEL_ID, feature_schema=LEGACY_FEATURE_SCHEMA
        )
        v2_samples = db.correction_parameter_training_samples(
            "contrast", model_id=MODEL_ID, feature_schema=FEATURE_SCHEMA
        )
        row = db.conn.execute(
            "SELECT user_strength,ai_parameter_model_id,parameter_features_json,parameter_feature_schema FROM correction_feedback"
        ).fetchone()
    assert row["user_strength"] == 0.33
    assert row["ai_parameter_model_id"] == LEGACY_MODEL_ID + ":legacy"
    assert row["parameter_feature_schema"] == LEGACY_FEATURE_SCHEMA
    assert row["parameter_features_json"] != "[]"
    assert len(legacy_samples) == 1
    assert legacy_samples[0]["user_strength"] == 0.33
    assert v2_samples == []


def test_correction_feedback_counts_ai_strength_kept_vs_changed(tmp_path):
    source = tmp_path / "source_ai_compare.png"
    _photo(source)
    item = {
        "action_key": "exposure", "tested": True, "accepted": True, "auto_eligible": True,
        "candidate": "gamma=0.92", "confidence": 0.8, "adjustable": True, "default_strength": 0.4,
    }
    suggestion = {
        "suggested_strength": 0.284, "confidence": 0.74,
        "model_id": "native_parameter_recommender_v2", "feature_schema": "parameter_features_v2", "features": [0.4] * 10,
    }
    with AnalysisDatabase(tmp_path / "feedback_compare.sqlite3") as db:
        # Slider shows/returns integer percent: 28% should count as accepting 28.4%.
        db.record_correction_feedback(
            source, [item], {"exposure"}, user_strengths={"exposure": 0.28},
            parameter_suggestions={"exposure": suggestion},
        )
        db.record_correction_feedback(
            source, [item], {"exposure"}, user_strengths={"exposure": 0.36},
            parameter_suggestions={"exposure": suggestion},
        )
        stats = db.correction_feedback_stats()[0]
    assert stats["strength_samples"] == 2
    assert stats["ai_strength_kept"] == 1
    assert stats["ai_strength_changed"] == 1
