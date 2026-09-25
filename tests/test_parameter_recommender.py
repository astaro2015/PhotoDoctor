from photodoctor.ai.parameter_recommender import (
    FEATURE_SCHEMA,
    MODEL_ID,
    MIN_PERSONAL_SAMPLES,
    MIN_PERSONAL_SOURCE_GROUPS,
    build_features,
    recommend_strength,
)


def _context(severity=70.0, brightness=35.0, archival=0.0, diagnostic="Портрет"):
    return {
        "metrics": {
            "brightness": {"normalized_value": brightness, "confidence": 0.85, "diagnostic": "Темновато"},
            "faces": {"normalized_value": 62.0, "confidence": 0.8, "diagnostic": "Найдено лицо"},
            "semantic_context": {
                "normalized_value": 75.0,
                "confidence": 0.7,
                "diagnostic": diagnostic,
                "archival_likelihood": archival,
            },
        },
        "decision_plan": [{
            "key": "exposure", "decision": "fix", "severity": severity,
            "repairability": 88.0, "confidence": 0.85, "priority": 62.0,
        }],
    }


def _validation(default=0.8, accepted=True):
    return {
        "action_key": "exposure", "default_strength": default, "confidence": 0.82,
        "accepted": accepted, "adjustable": True,
    }


def _sample(features, strength, idx):
    return {
        "features": features,
        "user_strength": strength,
        "feature_schema": FEATURE_SCHEMA,
        "model_id": MODEL_ID,
        "quick_hash": f"photo-{idx}",
    }


def test_parameter_recommender_v2_contract():
    assert MODEL_ID == "native_parameter_recommender_v2"
    assert FEATURE_SCHEMA == "parameter_features_v2"


def test_archival_feature_uses_numeric_semantic_value():
    assert build_features("exposure", _validation(), _context(archival=0.0))[-1] == 0.0
    assert build_features("exposure", _validation(), _context(archival=1.0))[-1] == 1.0
    assert build_features("exposure", _validation(), _context(archival=0.37))[-1] == 0.37


def test_archival_text_cannot_leak_into_v2_feature():
    context = _context(archival=0.0, diagnostic="Архивный archival старый отпечаток")
    assert build_features("exposure", _validation(), context)[-1] == 0.0


def test_invalid_archival_values_fall_back_without_crashing():
    for value in (None, "0.9", "archival", float("nan"), float("inf"), float("-inf")):
        assert build_features("exposure", _validation(), _context(archival=value))[-1] == 0.0


def test_parameter_recommender_cold_start_uses_validator_prior():
    result = recommend_strength("exposure", _validation(0.8, True), _context(), [])
    assert result.model_id == MODEL_ID
    assert result.feature_schema == FEATURE_SCHEMA
    assert not result.personalized
    assert result.sample_count == 0
    assert result.suggested_strength == 0.8
    assert 0.49 <= result.confidence < 0.7


def test_parameter_recommender_learns_bounded_preference_from_independent_photos():
    validation = _validation(0.8, True)
    context = _context()
    features = build_features("exposure", validation, context)
    history = [_sample(features, 0.30, i) for i in range(10)]
    result = recommend_strength("exposure", validation, context, history)
    assert len(history) >= MIN_PERSONAL_SAMPLES
    assert MIN_PERSONAL_SOURCE_GROUPS <= 10
    assert result.personalized
    assert result.sample_count == 10
    assert 0.60 <= result.suggested_strength < 0.8
    assert result.suggested_strength <= 0.8
    assert result.confidence > 0.6


def test_repeated_saves_of_one_photo_do_not_enable_personalization():
    validation = _validation(0.8, True)
    context = _context()
    features = build_features("exposure", validation, context)
    history = [
        {
            "features": features,
            "user_strength": 0.30,
            "feature_schema": FEATURE_SCHEMA,
            "model_id": MODEL_ID,
            "quick_hash": "same-photo",
        }
        for _ in range(20)
    ]
    result = recommend_strength("exposure", validation, context, history)
    assert not result.personalized
    assert result.suggested_strength == 0.8


def test_parameter_recommender_ignores_wrong_feature_shape_and_schema():
    features = build_features("exposure", _validation(), _context())
    history = [
        {"features": [1.0, 2.0], "user_strength": 0.1, "feature_schema": FEATURE_SCHEMA, "model_id": MODEL_ID},
        {"features": features, "user_strength": 0.1, "feature_schema": "parameter_features_v1", "model_id": "native_parameter_recommender_v1"},
    ] * 10
    result = recommend_strength("exposure", _validation(0.8, True), _context(), history)
    assert not result.personalized
    assert result.sample_count == 0
    assert result.suggested_strength == 0.8


def test_manual_only_cold_start_can_be_milder_or_stronger_but_stays_bounded():
    validation = _validation(0.45, False)
    result = recommend_strength("exposure", validation, _context(severity=90.0), [])
    assert 0.0 <= result.suggested_strength <= 1.0
    assert result.suggested_strength > 0.0


def test_database_feedback_drives_next_parameter_recommendation(tmp_path):
    from PIL import Image
    from photodoctor.core.database import AnalysisDatabase

    validation = _validation(0.8, True)
    validation.update({"tested": True, "auto_eligible": True, "candidate": "gamma=0.92"})
    context = _context()
    initial = recommend_strength("exposure", validation, context, [])
    suggestion = initial.to_dict()

    db_path = tmp_path / "learn.sqlite3"
    with AnalysisDatabase(db_path) as db:
        for i in range(6):
            source = tmp_path / f"training_photo_{i}.png"
            Image.new("RGB", (80, 60), (80 + i, 80, 80)).save(source)
            db.record_correction_feedback(
                source, [validation], {"exposure"},
                user_strengths={"exposure": 0.32},
                parameter_suggestions={"exposure": suggestion},
                context=context,
            )
        history = db.correction_parameter_training_samples(
            "exposure", model_id=MODEL_ID, feature_schema=FEATURE_SCHEMA
        )
    learned = recommend_strength("exposure", validation, context, history)
    assert learned.personalized
    assert learned.sample_count == 6
    assert learned.suggested_strength < initial.suggested_strength
