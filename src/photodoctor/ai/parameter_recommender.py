from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping

import numpy as np


MODEL_ID = "native_parameter_recommender_v2"
FEATURE_SCHEMA = "parameter_features_v2"
LEGACY_MODEL_ID = "native_parameter_recommender_v1"
LEGACY_FEATURE_SCHEMA = "parameter_features_v1"
MIN_PERSONAL_SAMPLES = 5
MIN_PERSONAL_SOURCE_GROUPS = 3
MAX_NEIGHBORS = 7
MAX_PERSONAL_DELTA = 0.20


@dataclass(slots=True)
class ParameterSuggestion:
    action_key: str
    suggested_strength: float
    baseline_strength: float
    confidence: float
    personalized: bool
    sample_count: int
    model_id: str
    feature_schema: str
    features: list[float]
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return result


def _safe_unit_number(value: Any, default: float = 0.0) -> float:
    """Read a machine numeric feature without parsing localized/human text."""
    if isinstance(value, str) or value is None:
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not np.isfinite(result):
        return float(default)
    return _clip01(result)


def _metric_payload(context: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    metrics = context.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return {}
    payload = metrics.get(name, {})
    return payload if isinstance(payload, Mapping) else {}


def _metric_score(context: Mapping[str, Any], name: str, default: float = 50.0) -> float:
    payload = _metric_payload(context, name)
    value = payload.get("normalized_value")
    if value is None:
        return float(default)
    return float(np.clip(_safe_float(value, default), 0.0, 100.0))


def _decision_payload(context: Mapping[str, Any], action_key: str) -> Mapping[str, Any]:
    items = context.get("decision_plan", [])
    if not isinstance(items, list):
        return {}
    for item in items:
        if isinstance(item, Mapping) and str(item.get("key", "")) == action_key:
            return item
    return {}


def _problem_deficit(action_key: str, context: Mapping[str, Any]) -> float:
    """Return an action-specific 0..1 estimate of how far the metric is from healthy."""
    if action_key == "exposure":
        score = _metric_score(context, "brightness", 50.0)
    elif action_key == "white_balance":
        score = _metric_score(context, "white_balance_advisor", 82.0)
    elif action_key == "auto_tone_color":
        score = min(_metric_score(context, "contrast", 72.0), _metric_score(context, "neutral_balance", 82.0))
    elif action_key == "contrast":
        score = min(_metric_score(context, "contrast", 60.0), _metric_score(context, "local_contrast", 60.0))
    elif action_key == "sharpness":
        scores = [
            _metric_score(context, "laplacian", 65.0),
            _metric_score(context, "tenengrad", 65.0),
            _metric_score(context, "faces", 65.0),
            _metric_score(context, "eyes", 65.0),
        ]
        score = min(scores)
    elif action_key == "noise":
        score = _metric_score(context, "noise", 70.0)
    elif action_key == "jpeg_artifacts":
        score = _metric_score(context, "jpeg_artifacts", 70.0)
    elif action_key == "edge_artifacts":
        score = _metric_score(context, "edge_artifacts", 70.0)
    elif action_key == "posterization":
        score = _metric_score(context, "posterization", 70.0)
    elif action_key == "red_eye":
        score = _metric_score(context, "red_eye", 75.0)
    elif action_key == "surface_defects":
        score = _metric_score(context, "surface_defects", 75.0)
    else:
        score = 50.0
    return _clip01((100.0 - score) / 100.0)


def _common_features(
    action_key: str,
    validation_item: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[float]:
    decision = _decision_payload(context, action_key)
    default_strength = _clip01(_safe_float(validation_item.get("default_strength", 1.0), 1.0))
    validation_conf = _clip01(_safe_float(validation_item.get("confidence", 0.0), 0.0))
    accepted = 1.0 if bool(validation_item.get("accepted", False)) else 0.0
    severity = _clip01(_safe_float(decision.get("severity", 50.0), 50.0) / 100.0)
    repairability = _clip01(_safe_float(decision.get("repairability", 50.0), 50.0) / 100.0)
    decision_conf = _clip01(_safe_float(decision.get("confidence", validation_conf), validation_conf))
    priority = _clip01(_safe_float(decision.get("priority", 50.0), 50.0) / 100.0)
    deficit = _problem_deficit(action_key, context)

    faces = _metric_payload(context, "faces")
    face_score = faces.get("normalized_value")
    has_faces = 1.0 if face_score is not None else 0.0
    return [
        default_strength,
        validation_conf,
        accepted,
        severity,
        repairability,
        decision_conf,
        priority,
        deficit,
        has_faces,
    ]


def build_features(
    action_key: str,
    validation_item: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[float]:
    """Build v2 machine features.

    Critical v2 rule: archival context comes only from the numeric
    ``semantic_context.archival_likelihood`` value supplied by the analyzer/UI
    context. Human-readable diagnostic text is never searched or parsed.
    """
    features = _common_features(action_key, validation_item, context)
    semantic = _metric_payload(context, "semantic_context")
    archival = _safe_unit_number(semantic.get("archival_likelihood"), 0.0)
    features.append(archival)
    return features


def build_features_v1_legacy(
    action_key: str,
    validation_item: Mapping[str, Any],
    context: Mapping[str, Any],
) -> list[float]:
    """Reproduce the old v1 vector only for migration/reading old records.

    Never use this extractor for new recommendations or v2 training.
    """
    features = _common_features(action_key, validation_item, context)
    semantic = _metric_payload(context, "semantic_context")
    semantic_diag = str(semantic.get("diagnostic", "")).lower()
    archival = 1.0 if "архив" in semantic_diag or "archiv" in semantic_diag else 0.0
    features.append(archival)
    return features


def _baseline_strength(action_key: str, validation_item: Mapping[str, Any], features: list[float]) -> float:
    default = _clip01(_safe_float(validation_item.get("default_strength", 1.0), 1.0))
    accepted = bool(validation_item.get("accepted", False))
    severity = features[3] if len(features) > 3 else 0.5
    if accepted:
        return default
    factor = 0.82 + 0.28 * severity
    return _clip01(default * factor)


def _valid_training_sample(sample: Mapping[str, Any], feature_len: int) -> tuple[np.ndarray, float] | None:
    try:
        schema = str(sample.get("feature_schema", "") or "")
        if schema and schema != FEATURE_SCHEMA:
            return None
        model_id = str(sample.get("model_id", "") or "")
        if model_id and model_id != MODEL_ID:
            return None
        features = sample.get("features", [])
        if not isinstance(features, (list, tuple)) or len(features) != feature_len:
            return None
        x = np.asarray([float(v) for v in features], dtype=np.float64)
        if not np.all(np.isfinite(x)):
            return None
        y = float(sample.get("user_strength"))
        if not np.isfinite(y):
            return None
        return x, _clip01(y)
    except (TypeError, ValueError, OverflowError):
        return None


def _sample_group(sample: Mapping[str, Any], index: int) -> str:
    for key in ("source_group_id", "quick_hash", "source_token", "sample_id"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    # API callers without provenance remain usable, while database-backed history
    # always supplies quick_hash and therefore receives real independence checks.
    return f"anonymous:{index}"


def recommend_strength(
    action_key: str,
    validation_item: Mapping[str, Any],
    context: Mapping[str, Any],
    history: Iterable[Mapping[str, Any]] = (),
) -> ParameterSuggestion:
    features = build_features(action_key, validation_item, context)
    current = np.asarray(features, dtype=np.float64)
    baseline = _baseline_strength(action_key, validation_item, features)

    samples: list[tuple[np.ndarray, float, str]] = []
    for index, sample in enumerate(history):
        parsed = _valid_training_sample(sample, len(features))
        if parsed is not None:
            x, y = parsed
            samples.append((x, y, _sample_group(sample, index)))

    n = len(samples)
    groups = len({group for _, _, group in samples})
    if n < MIN_PERSONAL_SAMPLES or groups < MIN_PERSONAL_SOURCE_GROUPS:
        confidence = 0.50 + min(n, MIN_PERSONAL_SAMPLES - 1) * 0.025 + min(groups, 2) * 0.02
        explanation = (
            "Базовая рекомендация по технической пробе проверки безопасности. "
            f"Для персонализации нужно минимум {MIN_PERSONAL_SAMPLES} сохранённых решений "
            f"из {MIN_PERSONAL_SOURCE_GROUPS} разных фото; сейчас {n} решений из {groups} фото."
        )
        return ParameterSuggestion(
            action_key=action_key,
            suggested_strength=baseline,
            baseline_strength=baseline,
            confidence=float(np.clip(confidence, 0.50, 0.66)),
            personalized=False,
            sample_count=n,
            model_id=MODEL_ID,
            feature_schema=FEATURE_SCHEMA,
            features=features,
            explanation=explanation,
        )

    distances: list[tuple[float, float, str]] = []
    for x, y, group in samples:
        delta = x - current
        if delta.size >= 8:
            delta = delta.copy()
            delta[0] *= 1.35
            delta[7] *= 1.25
        dist = float(np.sqrt(np.mean(delta * delta)))
        distances.append((dist, y, group))
    distances.sort(key=lambda item: item[0])

    # Do not let repeated saves of one photo dominate nearest neighbors.
    nearest: list[tuple[float, float]] = []
    used_groups: set[str] = set()
    for dist, y, group in distances:
        if group in used_groups:
            continue
        used_groups.add(group)
        nearest.append((dist, y))
        if len(nearest) >= min(MAX_NEIGHBORS, groups):
            break

    weights = np.asarray([1.0 / (0.06 + d * d) for d, _ in nearest], dtype=np.float64)
    ys = np.asarray([y for _, y in nearest], dtype=np.float64)
    personal = float(np.sum(weights * ys) / max(float(np.sum(weights)), 1e-9))

    # User history expresses preference, not technical truth. Keep it as a bounded
    # adapter around the Validator-tested baseline rather than replacing the base.
    personal_weight = float(np.clip(0.15 + (groups - MIN_PERSONAL_SOURCE_GROUPS) * 0.04, 0.15, 0.45))
    preferred_delta = float(np.clip(personal - baseline, -MAX_PERSONAL_DELTA, MAX_PERSONAL_DELTA))
    prediction = _clip01(baseline + personal_weight * preferred_delta)

    if bool(validation_item.get("accepted", False)):
        prediction = min(prediction, _clip01(_safe_float(validation_item.get("default_strength", 1.0), 1.0)))

    mean_distance = float(np.mean([d for d, _ in nearest])) if nearest else 1.0
    consistency = float(np.clip(1.0 - np.std(ys) * 1.4, 0.0, 1.0)) if len(ys) > 1 else 0.6
    confidence = float(
        np.clip(
            0.56 + min(groups, 12) * 0.018 + 0.10 * consistency - 0.18 * min(mean_distance, 1.0),
            0.54,
            0.88,
        )
    )
    explanation = (
        f"Персональная поправка по {n} сохранённым решениям из {groups} разных фото; "
        f"учтено {len(nearest)} независимых похожих случаев. Базовая техническая рекомендация остаётся главным ограничителем."
    )
    return ParameterSuggestion(
        action_key=action_key,
        suggested_strength=prediction,
        baseline_strength=baseline,
        confidence=confidence,
        personalized=True,
        sample_count=n,
        model_id=MODEL_ID,
        feature_schema=FEATURE_SCHEMA,
        features=features,
        explanation=explanation,
    )
