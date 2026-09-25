from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Any

import numpy as np

from .local_correction_planner import LocalCorrectionPlan, build_local_correction_plan

from .models import MetricResult




@dataclass(frozen=True, slots=True)
class ExposureAssessment:
    mean_luma: float | None
    median_luma: float
    subject_luma: float | None
    subject_well_exposed: bool
    strong_under: bool
    moderate_under: bool
    reference_luma: float
    target_luma: float
    decision: str  # none | fix | review
    repairability: float
    severity: float
    unexplained_highlight_clip_pct: float
    reason: str

@dataclass(slots=True)
class DecisionItem:
    key: str
    title: str
    decision: str  # fix | review | preserve | skip
    severity: float
    repairability: float
    confidence: float
    priority: float
    reason: str
    guardrail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip100(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def _score(metrics: Mapping[str, MetricResult], key: str) -> float | None:
    metric = metrics.get(key)
    if metric is None or metric.normalized_value is None:
        return None
    return float(metric.normalized_value)


def _confidence(metrics: Mapping[str, MetricResult], key: str, default: float = 0.4) -> float:
    metric = metrics.get(key)
    return float(metric.confidence) if metric is not None else default


def _raw_dict(metrics: Mapping[str, MetricResult], key: str) -> dict[str, Any]:
    metric = metrics.get(key)
    return metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}




def _raw_number(metrics: Mapping[str, MetricResult], key: str, default: float | None = None) -> float | None:
    metric = metrics.get(key)
    if metric is None or isinstance(metric.raw_value, dict):
        return default
    try:
        value = float(metric.raw_value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(value):
        return default
    return value


def _subject_face_luma(metrics: Mapping[str, MetricResult]) -> float | None:
    raw = _raw_dict(metrics, "faces")
    faces = raw.get("faces", []) if isinstance(raw, dict) else []
    if not isinstance(faces, list) or not faces:
        return None

    main = _raw_dict(metrics, "main_subject")
    indices = main.get("face_indices", []) if isinstance(main, dict) else []
    selected: list[float] = []
    if isinstance(indices, list):
        for index in indices:
            try:
                face = faces[int(index)]
            except (TypeError, ValueError, IndexError):
                continue
            if isinstance(face, dict):
                try:
                    value = float(face.get("brightness_linear"))
                except (TypeError, ValueError, OverflowError):
                    continue
                if np.isfinite(value):
                    selected.append(value)

    if not selected:
        for face in faces:
            if not isinstance(face, dict):
                continue
            try:
                value = float(face.get("brightness_linear"))
            except (TypeError, ValueError, OverflowError):
                continue
            if np.isfinite(value):
                selected.append(value)
    if not selected:
        return None
    return float(np.median(np.asarray(selected, dtype=np.float64)))

def assess_exposure(metrics: Mapping[str, MetricResult]) -> ExposureAssessment:
    """Evaluate global exposure using the same linear-light rules as Decision Engine.

    The function is intentionally public so GUI explanations and the actual
    correction plan cannot drift into two different definitions of "too dark".
    """
    mean_luma = _raw_number(metrics, "brightness")
    histogram = _raw_dict(metrics, "histogram")
    try:
        median_luma = float(histogram.get("p50", mean_luma if mean_luma is not None else 0.0))
    except (TypeError, ValueError, OverflowError):
        median_luma = float(mean_luma or 0.0)
    if not np.isfinite(median_luma):
        median_luma = float(mean_luma or 0.0)

    subject_luma = _subject_face_luma(metrics)
    subject_well_exposed = subject_luma is not None and subject_luma >= 0.20

    if mean_luma is None:
        return ExposureAssessment(
            None, median_luma, subject_luma, subject_well_exposed, False, False,
            median_luma, 0.18, "none", 0.0, 0.0, 0.0,
            "Недостаточно данных для оценки глобальной экспозиции.",
        )

    if subject_luma is None:
        strong_under = mean_luma < 0.10 and median_luma < 0.085
        moderate_under = mean_luma < 0.16 and median_luma < 0.135
        reference = max(mean_luma, median_luma)
        target = 0.18
    else:
        strong_under = subject_luma < 0.13 and mean_luma < 0.16
        moderate_under = subject_luma < 0.18 and mean_luma < 0.20
        reference = subject_luma
        target = 0.20

    highlight = _raw_dict(metrics, "highlight_context")
    try:
        unexplained = float(highlight.get("unexplained_clip_pct", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        unexplained = 0.0
    if not np.isfinite(unexplained):
        unexplained = 0.0

    if subject_well_exposed or not (strong_under or moderate_under):
        reason = (
            "Главный объект/лица уже находятся в безопасном диапазоне; тёмная одежда или фон не требуют глобального осветления."
            if subject_well_exposed
            else "Средняя и медианная линейная яркость не требуют глобального подъёма средних тонов."
        )
        return ExposureAssessment(
            mean_luma, median_luma, subject_luma, subject_well_exposed, strong_under,
            moderate_under, reference, target, "none", 0.0, 0.0, unexplained, reason,
        )

    repair = 88.0 if unexplained < 2.0 else 68.0
    decision = (
        "fix"
        if strong_under and _confidence(metrics, "brightness") >= 0.65 and repair >= 75
        else "review"
    )
    severity = _clip100(max(10.0, (target - reference) / max(target, 1e-6) * 100.0))
    reason = (
        "Главный объект/лица заметно темнее безопасного диапазона; допустим только мягкий подъём средних тонов."
        if subject_luma is not None
        else "И средняя, и медианная линейная яркость кадра низкие; возможен мягкий подъём средних тонов."
    )
    return ExposureAssessment(
        mean_luma, median_luma, subject_luma, subject_well_exposed, strong_under,
        moderate_under, reference, target, decision, repair, severity, unexplained, reason,
    )


def _priority(severity: float, repairability: float, confidence: float, decision: str) -> float:
    if decision == "preserve":
        return _clip100(72.0 + 24.0 * confidence)
    if decision == "skip":
        return 0.0
    action_factor = 0.42 + 0.58 * (repairability / 100.0)
    review_factor = 0.88 if decision == "review" else 1.0
    return _clip100(severity * action_factor * confidence * review_factor)


def _make(
    key: str,
    title: str,
    decision: str,
    severity: float,
    repairability: float,
    confidence: float,
    reason: str,
    guardrail: str = "",
) -> DecisionItem:
    severity = _clip100(severity)
    repairability = _clip100(repairability)
    confidence = float(np.clip(confidence, 0.0, 1.0))
    return DecisionItem(
        key=key,
        title=title,
        decision=decision,
        severity=severity,
        repairability=repairability,
        confidence=confidence,
        priority=_priority(severity, repairability, confidence, decision),
        reason=reason,
        guardrail=guardrail,
    )


def build_decision_plan(
    metrics: Mapping[str, MetricResult], *, local_plan: LocalCorrectionPlan | None = None
) -> list[DecisionItem]:
    items: list[DecisionItem] = []
    semantic = _raw_dict(metrics, "semantic_context")
    semantic_kind = str(semantic.get("classification", "unknown"))
    archival = semantic_kind == "archival_portrait"
    has_faces = int(semantic.get("face_count", 0) or 0) > 0 or int(_raw_dict(metrics, "faces").get("face_count", 0) or 0) > 0
    subject = _raw_dict(metrics, "main_subject")
    subject_kind = str(subject.get("subject_kind", "unknown"))
    subject_confidence = float(subject.get("confidence", 0.0) or 0.0)
    has_reliable_subject = subject_kind != "unknown" and subject_confidence >= 0.50

    if archival:
        items.append(_make(
            "preserve_archival_character",
            "Сохранить характер оригинала",
            "preserve",
            0.0,
            0.0,
            _confidence(metrics, "semantic_context", 0.6),
            "Кадр похож на архивный портрет: тон отпечатка и реальные черты лиц являются частью исходника.",
            "Не нейтрализовать сепию и не дорисовывать лица автоматически.",
        ))

    if local_plan is None:
        local_plan = build_local_correction_plan(metrics)
    exposure = assess_exposure(metrics)
    local_exposure_ready = bool(
        len(local_plan.exposure_cells) >= 2
        and local_plan.exposure_confidence >= 0.48
        and 0.005 <= local_plan.exposure_coverage <= 0.55
    )
    if (exposure.mean_luma is not None and exposure.decision in {"fix", "review"}) or local_exposure_ready:
        if local_exposure_ready:
            local_pct = local_plan.exposure_coverage * 100.0
            reason = (
                f"Локальная карта тонов нашла согласованные проблемные области примерно на {local_pct:.0f}% кадра; "
                "остальная часть изображения не требует той же поправки."
            )
            decision = "fix" if local_plan.exposure_confidence >= 0.62 else "review"
            severity = max(8.0, min(52.0, local_pct * 0.75))
            confidence = max(local_plan.exposure_confidence, _confidence(metrics, "brightness", 0.0))
        else:
            reason = exposure.reason
            decision = exposure.decision
            severity = exposure.severity
            confidence = _confidence(metrics, "brightness")
        items.append(_make(
            "exposure",
            "Средние тона / яркость",
            decision,
            severity,
            exposure.repairability if exposure.mean_luma is not None else 82.0,
            confidence,
            reason,
            "По умолчанию использовать пространственную маску: нормальные области и света не менять; глобальную коррекцию оставлять только как fallback.",
        ))

    contrast_scores = [v for v in (_score(metrics, "contrast"), _score(metrics, "local_contrast")) if v is not None]
    local_contrast_ready = bool(
        len(local_plan.contrast_cells) >= 2
        and local_plan.contrast_confidence >= 0.46
        and 0.005 <= local_plan.contrast_coverage <= 0.62
    )
    if (contrast_scores and min(contrast_scores) < 58.0) or local_contrast_ready:
        worst = min(contrast_scores) if contrast_scores else 58.0
        local_raw = _raw_dict(metrics, "local_contrast")
        fading = str(local_raw.get("classification", "unknown")) == "fading_candidate"
        if local_contrast_ready:
            local_pct = local_plan.contrast_coverage * 100.0
            decision = "fix" if local_plan.contrast_confidence >= 0.62 else "review"
            reason = f"Карта локального контраста нашла согласованные слабоконтрастные области примерно на {local_pct:.0f}% кадра."
            confidence = max(local_plan.contrast_confidence, _confidence(metrics, "local_contrast", 0.0))
            severity = max(8.0, min(55.0, local_pct * 0.80))
        else:
            decision = "fix" if _confidence(metrics, "local_contrast", _confidence(metrics, "contrast")) >= 0.62 else "review"
            reason = "Локальное тональное разделение ослаблено." if fading else "Контраст ниже желательного диапазона."
            confidence = max(_confidence(metrics, "contrast"), _confidence(metrics, "local_contrast", 0.0))
            severity = (58.0 - worst) * 1.7
        items.append(_make(
            "contrast",
            "Контраст / полутона",
            decision,
            severity,
            82.0 if local_contrast_ready else (78.0 if fading else 72.0),
            confidence,
            reason,
            "По умолчанию усиливать только слабоконтрастные области плавной маской; гладкие и уже нормальные зоны не трогать." +
            (" Главный объект защищать отдельно." if has_reliable_subject else ""),
        ))

    tone_kind = str(_raw_dict(metrics, "image_tone").get("classification", "unknown"))
    wb_raw = _raw_dict(metrics, "white_balance_advisor")
    wb_local_raw = _raw_dict(metrics, "local_white_balance")
    wb_needed = bool(wb_raw.get("correction_needed", False))
    wb_kind = str(wb_local_raw.get("classification", "uncertain") or "uncertain")
    wb_local_conf = _confidence(metrics, "local_white_balance", 0.0)
    wb_local_supported = int(float(wb_local_raw.get("supported_cells", 0) or 0))
    wb_local_coverage = float(np.clip(float(wb_local_raw.get("coverage", 0.0) or 0.0), 0.0, 1.0))
    wb_spatial_ready = bool(
        wb_kind in {"mixed_light", "local_cast"}
        and wb_local_conf >= 0.52
        and wb_local_supported >= 3
        and wb_local_coverage >= 0.10
    )
    wb_action_needed = wb_needed or wb_spatial_ready
    if wb_action_needed and not archival and tone_kind == "color":
        cast_strength = float(np.clip(float(wb_raw.get("cast_strength", 0.0) or 0.0), 0.0, 1.0))
        temp = float(wb_raw.get("recommended_temperature_shift_k", 0.0) or 0.0)
        tint = float(wb_raw.get("recommended_tint_shift", 0.0) or 0.0)
        neutralization = float(np.clip(float(wb_raw.get("neutralization_strength", 0.0) or 0.0), 0.0, 1.0))
        direction = str(wb_raw.get("cast_direction", "neutral"))
        names = {"warm": "тёплый", "cool": "холодный", "green": "зелёный", "magenta": "пурпурный", "neutral": "нейтральный"}
        if wb_spatial_ready:
            mixed_score = float(np.clip(float(wb_local_raw.get("mixed_light_score", 0.0) or 0.0), 0.0, 1.0))
            reason = (
                f"Карта освещения обнаружила неоднородный цветовой свет примерно на {wb_local_coverage * 100:.0f}% кадра "
                f"(mixed-light {mixed_score * 100:.0f}%). Нужна общая базовая поправка и плавные локальные temperature/tint-маски, а не один WB на весь кадр."
            )
            confidence = max(wb_local_conf, _confidence(metrics, "white_balance_advisor", 0.0) * 0.72)
            severity = max(10.0, min(62.0, mixed_score * 58.0))
        else:
            reason = f"Советник видит {names.get(direction, direction)} сдвиг: эквивалентная поправка температуры {temp:+.0f} K, tint {tint:+.1f}; рекомендуемая нейтрализация {neutralization * 100:.0f}%."
            confidence = _confidence(metrics, "white_balance_advisor", 0.55)
            severity = max(8.0, cast_strength * 72.0)
        items.append(_make(
            "white_balance",
            "Баланс белого / температура",
            "review",
            severity,
            88.0 if not wb_spatial_ready else 82.0,
            confidence,
            reason,
            "Сохранять атмосферу сцены и цвет кожи; не нейтрализовать сепию/архивный тон. Локальный WB разрешён только по согласованным нейтральным опорам и проходит Validator.",
        ))

    # Compound reference auto-correction learned from a real Photoshop
    # Auto Tone -> Auto Contrast -> Auto Color before/after pair.  It stays a
    # manual Review action until a larger golden set validates automatic use.
    neutral_score = _score(metrics, "neutral_balance")
    neutral_raw = _raw_dict(metrics, "neutral_balance")
    contrast_need = bool(contrast_scores and min(contrast_scores) < 72.0)
    color_need = (
        neutral_score is not None
        and neutral_score < 82.0
        and _confidence(metrics, "neutral_balance", 0.0) >= 0.55
        and not wb_action_needed
    )
    if not archival and tone_kind == "color" and (contrast_need or color_need):
        contrast_deficit = max(0.0, 72.0 - min(contrast_scores)) if contrast_scores else 0.0
        color_deficit = max(0.0, 82.0 - float(neutral_score)) if neutral_score is not None else 0.0
        cast_direction = str(neutral_raw.get("cast_direction", "neutral"))
        reason_parts = []
        if contrast_need:
            reason_parts.append("тональный диапазон можно аккуратно раздвинуть")
        if color_need:
            reason_parts.append(f"нейтральные полутона показывают цветовой сдвиг ({cast_direction})")
        items.append(_make(
            "auto_tone_color",
            "Автотон + автоконтраст + автоцвет",
            "review",
            max(8.0, contrast_deficit * 1.2 + color_deficit * 0.8),
            82.0,
            max(_confidence(metrics, "neutral_balance", 0.0), _confidence(metrics, "contrast", 0.0)),
            "; ".join(reason_parts).capitalize() + ".",
            "Комбинированная глобальная альтернатива отдельным правкам яркости/контраста. Не совмещать с ними в одном предпросмотре; архивную сепию автоматически не нейтрализовать.",
        ))

    sharp_parts = [v for v in (_score(metrics, "laplacian"), _score(metrics, "tenengrad")) if v is not None]
    global_sharp = sum(sharp_parts) / len(sharp_parts) if sharp_parts else None
    face_score = _score(metrics, "faces")
    eye_score = _score(metrics, "eyes")
    sharp_candidates = [v for v in (global_sharp, face_score, eye_score) if v is not None]
    local_sharpness_ready = bool(
        len(local_plan.sharpness_cells) >= 2
        and local_plan.sharpness_confidence >= 0.48
        and 0.005 <= local_plan.sharpness_coverage <= 0.58
    )
    if (sharp_candidates and min(sharp_candidates) < 65.0) or local_sharpness_ready:
        worst = min(sharp_candidates) if sharp_candidates else 65.0
        detail = _raw_dict(metrics, "detail_loss_type")
        kind = str(detail.get("classification", "unknown"))
        repair_by_kind = {
            "camera_shake_like": 30.0,
            "subject_motion_like": 32.0,
            "motion_like": 34.0,
            "defocus_like": 30.0,
            "local_subject_softness": 42.0,
            "degradation_like": 48.0,
            "mixed": 36.0,
            "mixed_or_degraded": 38.0,
            "unknown": 28.0,
        }
        repair = repair_by_kind.get(kind, 32.0)
        confs = [_confidence(metrics, "laplacian"), _confidence(metrics, "tenengrad")]
        if face_score is not None:
            confs.append(_confidence(metrics, "faces"))
        if eye_score is not None:
            confs.append(_confidence(metrics, "eyes"))
        guard = "Не дорисовывать отсутствующие детали генеративно автоматически."
        if has_faces:
            guard += " Лица и глаза защищены маской, но результат всё равно проверять в 100%."
        elif has_reliable_subject:
            guard += " Главный объект усиливать только в подтверждённых мягких областях."
        if local_sharpness_ready:
            local_pct = local_plan.sharpness_coverage * 100.0
            decision = "fix" if (local_plan.sharpness_confidence >= 0.64 and not has_faces and not archival) else "review"
            reason = f"Карта резкости нашла согласованные мягкие области примерно на {local_pct:.0f}% кадра; резкость вне них не требуется."
            confidence = max(local_plan.sharpness_confidence, min(confs) if confs else 0.0)
            severity = max(8.0, min(54.0, local_pct * 0.80))
            repair = max(repair, 62.0)
        else:
            decision = "review"
            reason = f"Найдена локальная или глобальная мягкость; предполагаемый тип: {kind}."
            confidence = min(confs) if confs else 0.5
            severity = (65.0 - worst) * 1.55
        items.append(_make(
            "sharpness",
            "Резкость / потеря деталей",
            decision,
            severity,
            repair,
            confidence,
            reason,
            guard,
        ))

    sr = _raw_dict(metrics, "super_resolution")
    sr_recommend = bool(sr.get("recommend", False))
    if sr_recommend:
        sr_score = float(sr.get("need_score", 0.0) or 0.0)
        sr_conf = max(0.50, float(sr.get("confidence", _confidence(metrics, "super_resolution", 0.55)) or 0.55))
        reasoning = sr.get("reasoning", [])
        if isinstance(reasoning, list):
            summary = " ".join(str(part) for part in reasoning[:2] if str(part).strip())
        else:
            summary = ""
        has_faces_sr = int(_raw_dict(metrics, "faces").get("face_count", 0) or 0) > 0
        guard = (
            "Увеличивать только x2, без агрессивной дорисовки новых черт и текстур. "
            "После увеличения обязательно проверить лица, волосы и текст на 100%."
        )
        if has_faces_sr:
            guard += " Лица защищать identity-guard маской и не менять геометрию глаз, носа, рта."
        items.append(_make(
            "super_resolution",
            "Восстановление разрешения x2",
            "review",
            max(16.0, min(72.0, sr_score)),
            60.0 if has_faces_sr else 68.0,
            sr_conf,
            summary or "Небольшой исходник может выиграть от осторожного x2-восстановления.",
            guard,
        ))

    noise = _score(metrics, "noise")
    local_noise_ready = bool(
        len(local_plan.noise_cells) >= 2
        and local_plan.noise_confidence >= 0.48
        and 0.005 <= local_plan.noise_coverage <= 0.65
    )
    if (noise is not None and noise < 70.0) or local_noise_ready:
        if local_noise_ready:
            local_pct = local_plan.noise_coverage * 100.0
            decision = "fix" if (local_plan.noise_confidence >= 0.64 and not has_faces) else "review"
            reason = f"Карта шума нашла согласованные шумные слаботекстурные области примерно на {local_pct:.0f}% кадра; остальная фактура защищена."
            severity = max(8.0, min(58.0, local_pct * 0.75 + max(0.0, 70.0 - float(noise or 70.0)) * 0.7))
            confidence = max(local_plan.noise_confidence, _confidence(metrics, "noise"))
            repairability = 84.0
        else:
            decision = "review" if has_faces else ("fix" if _confidence(metrics, "noise") >= 0.65 else "review")
            reason = "Мелкий яркостный шум выше желательного уровня."
            severity = (70.0 - float(noise)) * 1.6
            confidence = _confidence(metrics, "noise")
            repairability = 72.0
        items.append(_make(
            "noise",
            "Шум",
            decision,
            severity,
            repairability,
            confidence,
            reason,
            "Сохранять глаза, волосы, ткань и фактуру; физические дефекты отпечатка не лечить шумодавом." +
            (" Главный объект защищать от чрезмерного сглаживания." if has_reliable_subject else ""),
        ))

    for key, title, repair in (
        ("jpeg_artifacts", "JPEG-компрессия", 48.0),
        ("edge_artifacts", "Ореолы / перешарп", 46.0),
        ("posterization", "Постеризация / полосатость градаций", 38.0),
    ):
        score = _score(metrics, key)
        if score is not None and score < 70.0:
            items.append(_make(
                key,
                title,
                "review",
                (70.0 - score) * 1.7,
                repair,
                _confidence(metrics, key),
                f"Метрика «{title}» ниже безопасного диапазона.",
                "Корректировать на копии и перепроверять 100%; не усиливать артефакт последующей резкостью/контрастом.",
            ))

    surface = _raw_dict(metrics, "surface_defects")
    surface_count = int(surface.get("candidate_count", 0) or 0)
    if surface_count > 0:
        surface_score = _score(metrics, "surface_defects") or 100.0
        items.append(_make(
            "surface_defects",
            "Царапины / дефекты поверхности",
            "review",
            max(8.0, (100.0 - surface_score) * 1.4),
            56.0,
            _confidence(metrics, "surface_defects"),
            f"Найдено кандидатов на дефекты поверхности: {surface_count}; это ещё не подтверждённые трещины.",
            "Не удалять линии автоматически: волосы, швы и контуры могут выглядеть как царапины.",
        ))

    red_eye = _raw_dict(metrics, "red_eye")
    suspicious_eye_count = int(red_eye.get("suspicious_eye_count", 0) or 0)
    if suspicious_eye_count > 0:
        red_eye_score = _score(metrics, "red_eye") or max(0.0, 100.0 - suspicious_eye_count * 24.0)
        raw_candidates = red_eye.get("candidates", []) if isinstance(red_eye, dict) else []
        suspicious_candidates = [
            candidate for candidate in raw_candidates
            if isinstance(candidate, dict) and bool(candidate.get("suspicious", False))
        ] if isinstance(raw_candidates, list) else []
        # A relaxed red-eye pass is intentionally sensitive and therefore never
        # deserves an automatic "fix" decision by itself. Only a strong,
        # non-relaxed candidate with solid analysis confidence may be promoted.
        has_strong_candidate = any(not bool(candidate.get("relaxed_detection", False)) for candidate in suspicious_candidates)
        decision = "fix" if has_strong_candidate and _confidence(metrics, "red_eye", 0.5) >= 0.68 else "review"
        items.append(_make(
            "red_eye",
            "Красные глаза",
            decision,
            max(12.0, (100.0 - red_eye_score) * 1.15 + suspicious_eye_count * 7.0),
            92.0,
            _confidence(metrics, "red_eye", 0.5),
            f"Найдены вероятные красные зрачки: {suspicious_eye_count}. Это обычно локально и хорошо чинится без правки всего лица.",
            "Корректировать только зрачок/рефлекс, сохранить блики и не затемнять глаз целиком.",
        ))

    tone = _raw_dict(metrics, "image_tone")
    if str(tone.get("classification", "unknown")) in {"sepia", "monochrome"} and not archival:
        items.append(_make(
            "preserve_tone",
            "Сохранить исходный тон",
            "preserve",
            0.0,
            0.0,
            _confidence(metrics, "image_tone", 0.5),
            "Сепия/монохромность может быть свойством оригинала, а не ошибкой баланса белого.",
            "Не нейтрализовать цвет автоматически.",
        ))

    if not items:
        items.append(_make(
            "no_action",
            "Критические вмешательства не требуются",
            "skip",
            0.0,
            0.0,
            0.75,
            "По текущим метрикам нет достаточно уверенной причины для автоматической коррекции.",
            "Сохранять исходник неизменным; любые правки делать на копии.",
        ))

    reliability = _raw_dict(metrics, "analysis_reliability")
    reliability_status = str(reliability.get("status", "reliable"))
    if reliability_status in {"unknown", "ood_candidate"}:
        guard_text = (
            "Общая надёжность анализа снижена: автоматическое применение отключено до ручного предпросмотра."
            if reliability_status == "unknown" else
            "Вход выглядит нетипичным для текущей калибровки: автоматическое применение отключено до ручной проверки."
        )
        for item in items:
            if item.decision == "fix":
                item.decision = "review"
            if item.decision in {"fix", "review"}:
                item.guardrail = (item.guardrail + " " + guard_text).strip()
                item.priority = _priority(item.severity, item.repairability, item.confidence, item.decision)

    decision_order = {"preserve": 0, "fix": 1, "review": 2, "skip": 3}
    items.sort(key=lambda item: (decision_order.get(item.decision, 9), -item.priority, item.key))
    return items
