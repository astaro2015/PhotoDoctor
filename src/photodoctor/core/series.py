from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Callable, Iterable, Mapping

import cv2
import numpy as np
from PIL import Image, ImageOps

from .database import AnalysisDatabase
from .models import MetricResult
from .versioning import SERIES_ALGORITHM_VERSION

PHASH_METHOD = "phash64_dct_v1"


@dataclass(slots=True)
class SeriesMember:
    path: Path
    capture_time: datetime
    time_source: str
    phash: int
    technical_score: float
    mean_luma: float = 0.48
    aspect_ratio: float = 1.0
    camera_key: str = ""
    relative_score: float = 0.0
    rank: int = 0
    reasons: list[str] = field(default_factory=list)
    metrics_available: bool = True


@dataclass(slots=True)
class PhotoSeries:
    series_id: str
    members: list[SeriesMember]
    best_path: Path
    confidence: float
    similarity_mean: float
    time_span_seconds: float
    selection_note: str


@dataclass(slots=True)
class SeriesSummary:
    total_photos: int
    series_count: int
    grouped_photos: int
    singleton_photos: int
    groups: list[PhotoSeries] = field(default_factory=list)
    cancelled: bool = False


def _clip100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _metric_score(metrics: Mapping[str, MetricResult], name: str, default: float = 50.0) -> float:
    metric = metrics.get(name)
    if metric is None or metric.normalized_value is None:
        return default
    try:
        return _clip100(float(metric.normalized_value))
    except (TypeError, ValueError):
        return default


def _raw_dict(metrics: Mapping[str, MetricResult], name: str) -> dict[str, object]:
    metric = metrics.get(name)
    if metric is None or not isinstance(metric.raw_value, dict):
        return {}
    return metric.raw_value


def _brightness_quality(metrics: Mapping[str, MetricResult]) -> float:
    metric = metrics.get("brightness")
    try:
        mean_luma = float(metric.raw_value) if metric is not None else 0.28
    except (TypeError, ValueError):
        mean_luma = 0.28
    # Linear-light average luma is normally much lower than display-space
    # brightness.  Keep a broad plateau so style and dark clothing/background do
    # not dominate best-shot selection.
    if 0.18 <= mean_luma <= 0.45:
        return 100.0
    if mean_luma < 0.18:
        return _clip100(100.0 - (0.18 - mean_luma) * 450.0)
    return _clip100(100.0 - (mean_luma - 0.45) * 180.0)


def _weighted_component_score(components: Mapping[str, float], *, face_context: bool, eye_context: bool) -> float:
    if face_context:
        weights = {
            "sharpness": 0.13,
            "face": 0.21,
            "eyes": 0.11 if eye_context else 0.0,
            "exposure": 0.16,
            "contrast": 0.08,
            "noise": 0.09,
            "artifacts": 0.08,
            "local_contrast": 0.06,
            "red_eye": 0.08,
        }
        if not eye_context:
            weights["face"] += 0.06
            weights["sharpness"] += 0.05
    else:
        weights = {
            "sharpness": 0.24,
            "face": 0.0,
            "eyes": 0.0,
            "exposure": 0.20,
            "contrast": 0.14,
            "noise": 0.12,
            "artifacts": 0.12,
            "local_contrast": 0.10,
            "red_eye": 0.08,
        }
    weight_sum = sum(weights.values()) or 1.0
    return _clip100(sum(float(components.get(key, 50.0)) * weight for key, weight in weights.items()) / weight_sum)


def technical_series_score(metrics: Mapping[str, MetricResult]) -> tuple[float, dict[str, float]]:
    """Build a conservative technical score for *relative* comparison in one series.

    This deliberately does not evaluate expression, pose, composition or aesthetic
    value. Face/eye sharpness receives extra weight only when those measurements
    are actually available.
    """
    sharpness = 0.48 * _metric_score(metrics, "laplacian") + 0.52 * _metric_score(metrics, "tenengrad")
    exposure = (
        0.34 * _metric_score(metrics, "shadow_clipping")
        + 0.42 * _metric_score(metrics, "highlight_clipping")
        + 0.24 * _brightness_quality(metrics)
    )
    contrast = _metric_score(metrics, "contrast")
    noise = _metric_score(metrics, "noise")
    artifacts = np.mean([
        _metric_score(metrics, "jpeg_artifacts"),
        _metric_score(metrics, "edge_artifacts"),
        _metric_score(metrics, "posterization"),
    ])
    local_contrast = _metric_score(metrics, "local_contrast")
    red_eye = _metric_score(metrics, "red_eye", 100.0)

    face_metric = metrics.get("faces")
    eye_metric = metrics.get("eyes")
    face_score = _metric_score(metrics, "faces", sharpness)
    eye_score = _metric_score(metrics, "eyes", face_score)
    faces_raw = _raw_dict(metrics, "faces")
    face_count = int(faces_raw.get("face_count", 0) or 0)
    face_available = face_count > 0 and face_metric is not None and face_metric.normalized_value is not None
    eye_raw = _raw_dict(metrics, "eyes")
    eye_count = int(eye_raw.get("eye_count", 0) or 0)
    eye_available = eye_count > 0 and eye_metric is not None and eye_metric.normalized_value is not None

    components = {
        "_face_context": 1.0 if face_available else 0.0,
        "_eye_context": 1.0 if eye_available else 0.0,
        "sharpness": float(sharpness),
        "face": float(face_score),
        "eyes": float(eye_score),
        "exposure": float(exposure),
        "contrast": float(contrast),
        "noise": float(noise),
        "artifacts": float(artifacts),
        "local_contrast": float(local_contrast),
        "red_eye": float(red_eye),
    }
    score = _weighted_component_score(components, face_context=face_available, eye_context=eye_available)
    return score, components


def _parse_exif_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidates = (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y:%m:%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
    )
    for fmt in candidates:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def capture_time_for(path: Path, metrics: Mapping[str, MetricResult]) -> tuple[datetime, str]:
    exif = _raw_dict(metrics, "exif_context")
    parsed = _parse_exif_datetime(exif.get("datetime_original"))
    if parsed is not None:
        return parsed, "exif"
    try:
        return datetime.fromtimestamp(path.stat().st_mtime), "file_mtime"
    except OSError:
        return datetime.fromtimestamp(0), "unknown"


def perceptual_hash64(path: str | Path) -> int:
    """64-bit DCT perceptual hash using a small EXIF-oriented thumbnail."""
    p = Path(path)
    with Image.open(p) as image:
        image = ImageOps.exif_transpose(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32)
    dct = cv2.dct(arr)
    low = dct[:8, :8].copy()
    flat = low.ravel()
    median = float(np.median(flat[1:])) if flat.size > 1 else float(flat[0])
    bits = flat > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return int(value)


def phash_similarity(a: int, b: int) -> float:
    distance = (int(a) ^ int(b)).bit_count()
    return float(np.clip(1.0 - distance / 64.0, 0.0, 1.0))


def _series_context(metrics: Mapping[str, MetricResult]) -> tuple[float, float, str]:
    brightness = metrics.get("brightness")
    try:
        mean_luma = float(brightness.raw_value) if brightness is not None else 0.48
    except (TypeError, ValueError):
        mean_luma = 0.48
    profile = _raw_dict(metrics, "analysis_profile")
    try:
        width = float(profile.get("source_width", 0.0) or 0.0)
        height = float(profile.get("source_height", 0.0) or 0.0)
        aspect = width / height if width > 0 and height > 0 else 1.0
    except (TypeError, ValueError, ZeroDivisionError):
        aspect = 1.0
    exif = _raw_dict(metrics, "exif_context")
    make = str(exif.get("make") or "").strip().lower()
    model = str(exif.get("model") or "").strip().lower()
    camera_key = "|".join(part for part in (make, model) if part)
    return mean_luma, aspect, camera_key


def _same_series(prev: SeriesMember, current: SeriesMember, anchor: SeriesMember) -> bool:
    gap = max(0.0, (current.capture_time - prev.capture_time).total_seconds())
    pair_similarity = phash_similarity(prev.phash, current.phash)
    anchor_similarity = phash_similarity(anchor.phash, current.phash)
    reliable_time = prev.time_source == "exif" and current.time_source == "exif"
    if prev.camera_key and current.camera_key and prev.camera_key != current.camera_key:
        return False
    if abs(prev.mean_luma - current.mean_luma) > 0.22:
        return False
    aspect_delta = abs(prev.aspect_ratio - current.aspect_ratio) / max(prev.aspect_ratio, current.aspect_ratio, 1e-6)
    if aspect_delta > 0.10:
        return False

    if reliable_time:
        accepted = (
            (gap <= 4.0 and pair_similarity >= 0.62)
            or (gap <= 15.0 and pair_similarity >= 0.72)
            or (gap <= 45.0 and pair_similarity >= 0.86)
        )
        return accepted and anchor_similarity >= 0.58

    # Filesystem timestamps are a weak substitute: require much stronger visual
    # similarity so a folder copied in one operation does not become one "series".
    accepted = (gap <= 3.0 and pair_similarity >= 0.74) or (gap <= 10.0 and pair_similarity >= 0.88)
    return accepted and anchor_similarity >= 0.70


def _selection_reasons(
    best_components: Mapping[str, float],
    median_components: Mapping[str, float],
    *,
    face_context: bool,
) -> list[str]:
    labels = {
        "face": "лучше резкость лица",
        "eyes": "лучше детализация глаз",
        "sharpness": "выше общая резкость",
        "exposure": "лучше сохранены тени/света",
        "noise": "меньше заметного шума",
        "artifacts": "меньше артефактов обработки/сжатия",
        "red_eye": "меньше риска красных глаз",
        "contrast": "лучше технический контраст",
        "local_contrast": "лучше локальное разделение тонов",
    }
    order = ["face", "eyes", "sharpness", "exposure", "noise", "artifacts", "red_eye", "contrast", "local_contrast"]
    if not face_context:
        order = [x for x in order if x not in {"face", "eyes"}]
    deltas = [(key, float(best_components.get(key, 50.0)) - float(median_components.get(key, 50.0))) for key in order]
    deltas.sort(key=lambda item: item[1], reverse=True)
    reasons = [labels[key] for key, delta in deltas if delta >= 4.0][:3]
    if not reasons:
        reasons.append("технические различия между кадрами небольшие")
    return reasons


def _finalize_group(raw_group: list[tuple[SeriesMember, dict[str, float]]], group_index: int) -> PhotoSeries:
    face_votes = sum(float(components.get("_face_context", 0.0)) >= 0.5 for _, components in raw_group)
    eye_votes = sum(float(components.get("_eye_context", 0.0)) >= 0.5 for _, components in raw_group)
    face_context = face_votes >= max(1, (len(raw_group) + 1) // 2)
    eye_context = face_context and eye_votes >= max(1, (len(raw_group) + 1) // 2)
    for member, components in raw_group:
        member.technical_score = _weighted_component_score(
            components, face_context=face_context, eye_context=eye_context
        )
    scores = np.asarray([member.technical_score for member, _ in raw_group], dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    lo = float(scores.min())
    hi = float(scores.max())
    spread = hi - lo

    best_score = hi
    for rank_index, source_index in enumerate(order, start=1):
        member = raw_group[int(source_index)][0]
        member.rank = rank_index
        # Keep close frames visually close on the relative scale. A tiny technical
        # difference must not turn into a misleading 100-vs-0 verdict.
        member.relative_score = _clip100(100.0 - max(0.0, best_score - member.technical_score) * 3.0)

    best_index = int(order[0])
    best, best_components = raw_group[best_index]
    component_keys = [key for key in best_components if not key.startswith("_")]
    median_components = {
        key: float(np.median([components.get(key, 50.0) for _, components in raw_group]))
        for key in component_keys
    }
    best.reasons = _selection_reasons(best_components, median_components, face_context=face_context)

    similarities = [phash_similarity(raw_group[i - 1][0].phash, raw_group[i][0].phash) for i in range(1, len(raw_group))]
    similarity_mean = float(np.mean(similarities)) if similarities else 1.0
    start = min(member.capture_time for member, _ in raw_group)
    end = max(member.capture_time for member, _ in raw_group)
    span = max(0.0, (end - start).total_seconds())
    exif_fraction = sum(member.time_source == "exif" for member, _ in raw_group) / len(raw_group)
    score_separation = 0.0
    if len(scores) >= 2:
        sorted_scores = np.sort(scores)[::-1]
        score_separation = float(sorted_scores[0] - sorted_scores[1])
    confidence = float(np.clip(
        0.26 + 0.32 * similarity_mean + 0.18 * exif_fraction + min(score_separation / 12.0, 0.18),
        0.0,
        0.90,
    ))
    if score_separation < 1.5:
        note = "Кадры технически очень близки; финальный выбор лучше проверить вручную."
        confidence = min(confidence, 0.58)
    elif score_separation < 4.0:
        note = "Есть небольшой технический перевес; выражение лица и удачность момента не оцениваются."
    else:
        note = "Есть заметный технический перевес; художественный/эмоциональный выбор остаётся за пользователем."

    members = [raw_group[int(i)][0] for i in order]
    return PhotoSeries(
        series_id=f"S{group_index:04d}",
        members=members,
        best_path=best.path,
        confidence=confidence,
        similarity_mean=similarity_mean,
        time_span_seconds=span,
        selection_note=note,
    )


def group_photo_series(
    paths: Iterable[str | Path],
    database: AnalysisDatabase,
    *,
    precision: str = "normal",
    progress: Callable[[int, int, Path], None] | None = None,
    cancel_event: Event | None = None,
) -> SeriesSummary:
    files = [Path(p) for p in paths if Path(p).is_file()]
    prepared: list[tuple[SeriesMember, dict[str, float]]] = []
    total = len(files)
    cancel_event = cancel_event or Event()

    for index, path in enumerate(files, start=1):
        if cancel_event.is_set():
            return SeriesSummary(len(prepared), 0, 0, len(prepared), [], cancelled=True)
        if not database.is_current(path, precision=precision):
            if progress:
                progress(index, total, path)
            continue
        metrics = database.load_metrics(path, precision=precision)
        if not metrics:
            if progress:
                progress(index, total, path)
            continue
        capture_time, time_source = capture_time_for(path, metrics)
        try:
            cached_hash = database.load_file_fingerprint(path, PHASH_METHOD)
            if cached_hash is None:
                phash = perceptual_hash64(path)
                database.save_file_fingerprint(path, PHASH_METHOD, f"{phash:016x}")
            else:
                try:
                    phash = int(cached_hash, 16)
                except ValueError:
                    phash = perceptual_hash64(path)
                    database.save_file_fingerprint(path, PHASH_METHOD, f"{phash:016x}")
        except (OSError, ValueError, RuntimeError):
            if progress:
                progress(index, total, path)
            continue
        score, components = technical_series_score(metrics)
        mean_luma, aspect_ratio, camera_key = _series_context(metrics)
        prepared.append((SeriesMember(
            path, capture_time, time_source, phash, score,
            mean_luma=mean_luma, aspect_ratio=aspect_ratio, camera_key=camera_key,
        ), components))
        if progress:
            progress(index, total, path)

    prepared.sort(key=lambda item: (item[0].capture_time, str(item[0].path).lower()))
    raw_groups: list[list[tuple[SeriesMember, dict[str, float]]]] = []
    current: list[tuple[SeriesMember, dict[str, float]]] = []
    for item in prepared:
        if not current:
            current = [item]
            continue
        if _same_series(current[-1][0], item[0], current[0][0]):
            current.append(item)
        else:
            raw_groups.append(current)
            current = [item]
    if current:
        raw_groups.append(current)

    grouped = [group for group in raw_groups if len(group) >= 2]
    groups = [_finalize_group(group, i + 1) for i, group in enumerate(grouped)]
    grouped_photos = sum(len(group.members) for group in groups)
    return SeriesSummary(
        total_photos=len(prepared),
        series_count=len(groups),
        grouped_photos=grouped_photos,
        singleton_photos=max(0, len(prepared) - grouped_photos),
        groups=groups,
    )
