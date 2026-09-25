from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from photodoctor.core.models import MetricResult


@dataclass(frozen=True, slots=True)
class DisplayMetric:
    key: str
    label: str
    score: float | None
    confidence: float
    diagnosis: str
    status: str


@dataclass(frozen=True, slots=True)
class PhotoProfile:
    key: str
    label: str
    confidence: float
    explanation: str


@dataclass(frozen=True, slots=True)
class RecommendationGroups:
    safe: tuple[str, ...]
    caution: tuple[str, ...]
    avoid: tuple[str, ...]


_LABELS = {
    "brightness": "Яркость",
    "shadow_clipping": "Тени",
    "highlight_clipping": "Света",
    "highlight_context": "Блики / источники света",
    "contrast": "Контраст",
    "noise": "Шум / чистота",
    "jpeg_artifacts": "JPEG-компрессия",
    "edge_artifacts": "Ореолы / перешарп",
    "posterization": "Постеризация / полосатость градаций",
    "color_cast": "Баланс цвета",
    "neutral_balance": "Баланс нейтральных полутонов",
    "white_balance_advisor": "Советник баланса белого",
    "local_white_balance": "Карта цветового освещения",
    "surface_defects": "Дефекты поверхности",
    "faces": "Лица",
    "eyes": "Глаза",
    "red_eye": "Красные глаза",
    "local_sharpness": "Локальная резкость",
    "local_tone": "Локальные тона",
    "local_contrast": "Локальный контраст",
    "detail_loss_type": "Тип потери деталей",
    "exif_context": "EXIF-контекст",
    "nss_baseline": "Базовая NSS-оценка",
    "semantic_context": "Сюжетный контекст",
    "main_subject": "Главный объект / сцена",
    "aesthetic_quality": "Эстетическая / композиционная оценка",
    "analysis_reliability": "Надёжность анализа",
}


def _f(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_photo_profile(metrics: Mapping[str, MetricResult]) -> PhotoProfile:
    metric = metrics.get("image_tone")
    if metric is not None and isinstance(metric.raw_value, dict):
        raw = metric.raw_value
        kind = str(raw.get("classification", "unknown"))
        confidence = _f(raw.get("confidence"), metric.confidence)
        if kind == "monochrome":
            return PhotoProfile(
                "monochrome",
                "Почти чёрно-белое",
                confidence,
                "Цветовая насыщенность очень мала; цветовой баланс не следует оценивать как у обычного цветного кадра.",
            )
        if kind == "sepia":
            return PhotoProfile(
                "sepia",
                "Сепия / старый отпечаток",
                confidence,
                "Преобладает устойчивый тёплый порядок каналов R>G>B. Это похоже на сепию или старение отпечатка, а не на обычную ошибку баланса белого.",
            )
        if kind == "color":
            return PhotoProfile(
                "color",
                "Цветное фото",
                confidence,
                "Кадр содержит достаточно цветовой информации для обычной оценки глобального цветового сдвига.",
            )

    # Fallback for cached/legacy analyses without image_tone.
    cast = metrics.get("color_cast")
    if cast is not None and isinstance(cast.raw_value, dict):
        raw = cast.raw_value
        spread = _f(raw.get("relative_spread"))
        r = _f(raw.get("mean_r"))
        g = _f(raw.get("mean_g"))
        b = _f(raw.get("mean_b"))
        if spread < 0.05:
            return PhotoProfile("monochrome", "Почти чёрно-белое", 0.55, "Профиль определён по старым метрикам без отдельного детектора тона.")
        if r > g > b and spread > 0.08:
            return PhotoProfile("sepia", "Сепия / тёплый старый отпечаток", 0.52, "Предварительный профиль по средним каналам; новый анализ даст более надёжную оценку.")
    return PhotoProfile("unknown", "Не определён", 0.35, "Недостаточно данных для надёжного определения типа цветопередачи.")


def _diagnosis(key: str, metric: MetricResult, profile: PhotoProfile) -> str:
    raw = metric.raw_value
    score = metric.normalized_value

    if key == "brightness":
        value = _f(raw)
        if value < 0.07:
            return "Кадр очень тёмный; детали в средних тонах действительно могут быть потеряны."
        if value < 0.12:
            return "Кадр заметно тёмный; осветление допустимо только после проверки главного объекта и светов."
        if value < 0.18:
            return "Кадр немного тёмный, но это может быть нормальным для сцены; обязательной коррекции нет."
        if value > 0.65:
            return "Кадр очень светлый; стоит проверить сохранность деталей в светах."
        if value > 0.50:
            return "Кадр светлый, но без явного признака ошибки экспозиции."
        return "Средняя линейная яркость находится в широком нормальном диапазоне."

    if key == "shadow_clipping":
        pct = _f(raw)
        if pct < 0.5:
            return "Проваленных теней практически нет."
        if pct < 2.0:
            return f"Небольшая доля глубоких теней: около {pct:.1f}% кадра."
        return f"Заметная доля почти чёрных областей: около {pct:.1f}% кадра."

    if key == "highlight_clipping":
        pct = _f(raw)
        if pct < 0.5:
            return "Выбитых светов практически нет."
        if pct < 3.0:
            return f"Небольшая доля очень светлых областей: около {pct:.1f}% кадра."
        return f"Есть заметные выбитые света: около {pct:.1f}% кадра."

    if key == "highlight_context":
        if not isinstance(raw, dict):
            return "Недостаточно данных для контекстной оценки ярких областей."
        total = _f(raw.get("total_clip_pct"))
        protected = _f(raw.get("protected_clip_pct"))
        unexplained = _f(raw.get("unexplained_clip_pct"))
        count = int(_f(raw.get("candidate_count")))
        face_rejected = int(_f(raw.get("face_rejected_count")))
        if total <= 0.01:
            return "Почти полностью выбитых светов не найдено; контекстная проверка бликов не требуется."
        if count > 0:
            extra = f" Ярких областей внутри лица не защищено автоматически: {face_rejected}." if face_rejected else ""
            return (
                f"Найдено кандидатов на допустимый блик/источник света: {count}. "
                f"Из общей доли потерь в светах {total:.2f}% условно объяснено {protected:.2f}%, необъяснённым остаётся {unexplained:.2f}%."
                f"{extra} Кандидаты лучше проверить по карте."
            )
        return f"Из {total:.2f}% потерь в светах безопасные компактные блики не подтверждены; необъяснённым остаётся {unexplained:.2f}%."

    if key == "contrast":
        value = _f(raw)
        if value < 0.28:
            return "Контраст низкий; изображение может выглядеть выцветшим или плоским."
        if value < 0.50:
            return "Контраст умеренный."
        return "Глобальный контраст хороший; локальный контраст будет оцениваться отдельно."

    if key == "noise":
        if isinstance(raw, dict):
            sigma = _f(raw.get("sigma_luma"))
            classification = str(raw.get("classification", "unknown"))
            flat_pct = _f(raw.get("flat_area_pct"))
            usable = int(_f(raw.get("usable_tiles")))
            total = int(_f(raw.get("total_tiles")))
            if classification == "unknown":
                return "Недостаточно спокойных участков для надёжной оценки мелкого цифрового шума."
            support = f" Оценено {usable} из {total} плиток; слабоградиентная площадь около {flat_pct:.0f}%." if total > 0 else ""
            if classification == "very_low":
                return f"Мелкий яркостный шум очень низкий (σ≈{sigma:.2f}). Фактура бумаги и царапины считаются отдельно.{support}"
            if classification == "low":
                return f"Мелкий шум невелик (σ≈{sigma:.2f}); сильное шумоподавление не требуется.{support}"
            if classification == "moderate":
                return f"Есть умеренный мелкий яркостный шум (σ≈{sigma:.2f}); подавлять его стоит только с защитой деталей.{support}"
            return f"Шум заметный (σ≈{sigma:.2f}); перед подавлением проверить лица и мелкую фактуру.{support}"
        sigma = _f(raw)
        if sigma < 2.0:
            return "Высокочастотный шум по старой оценке невелик."
        if sigma < 5.0:
            return "Есть умеренный мелкий шум; сильное шумоподавление пока не требуется."
        return "Шум заметный; нужна более точная оценка по областям перед подавлением."

    if key == "exif_context":
        if not isinstance(raw, dict) or not bool(raw.get("available", False)):
            return "Параметры съёмки EXIF отсутствуют или недостаточны; выводы делаются только по содержимому изображения."
        exposure = raw.get("exposure_s")
        focal = raw.get("focal_length_mm")
        iso = raw.get("iso")
        aperture = raw.get("aperture_f")
        shutter_risk = str(raw.get("shutter_risk", "unknown"))
        iso_risk = str(raw.get("iso_noise_risk", "unknown"))
        parts: list[str] = []
        if exposure is not None:
            exp = _f(exposure)
            if 0 < exp < 1:
                reciprocal = round(1.0 / exp)
                parts.append(f"выдержка ≈1/{reciprocal} с")
            elif exp > 0:
                parts.append(f"выдержка {exp:.2f} с")
        if focal is not None:
            parts.append(f"фокусное {_f(focal):.0f} мм")
        if aperture is not None and _f(aperture) > 0:
            parts.append(f"f/{_f(aperture):.1f}")
        if iso is not None and _f(iso) > 0:
            parts.append(f"ISO {_f(iso):.0f}")
        prefix = ", ".join(parts) if parts else "Есть EXIF, но ключевые параметры съёмки заполнены частично"
        hints: list[str] = []
        if shutter_risk == "high":
            hints.append("выдержка медленнее осторожного ориентира 1/фокусное и может поддерживать гипотезу смаза камеры")
        elif shutter_risk == "moderate":
            hints.append("выдержка близка к ориентиру 1/фокусное; риск смаза зависит от стабилизации и техники съёмки")
        if iso_risk == "high":
            hints.append("высокое ISO может поддерживать гипотезу цифрового шума, если он виден в самом изображении")
        elif iso_risk == "moderate":
            hints.append("ISO умеренное; возможный шум нужно подтверждать по пикселям")
        if hints:
            return prefix + ". " + "; ".join(hints) + ". EXIF используется только как контекст."
        return prefix + ". Явного дополнительного риска по EXIF не видно; метаданные не заменяют анализ изображения."

    if key == "nss_baseline":
        if not isinstance(raw, dict):
            return "Базовая NSS-оценка недоступна для этого изображения."
        count = int(_f(raw.get("feature_count")))
        informative = bool(raw.get("informative", False))
        if count != 36:
            return "Базовую NSS-оценку не удалось сформировать полностью; эта метрика не влияет на итоговую оценку."
        if not informative:
            return "Собран NSS-вектор из 36 признаков в стиле BRISQUE, но кадр слишком однородный или мал для уверенной интерпретации. Оценка качества из него не вычисляется."
        return "Собран NSS-вектор из 36 признаков в стиле BRISQUE для независимой оценки без эталона. Предобученной модели BRISQUE/NIQE нет, поэтому фиктивная оценка качества не вычисляется."

    if key == "aesthetic_quality":
        if not isinstance(raw, dict):
            return "Эстетическая оценка недоступна."
        value = raw.get("score")
        if value is None:
            return "Главный объект определён недостаточно уверенно; эстетический балл не выставляется."
        components = raw.get("components", {}) if isinstance(raw.get("components"), dict) else {}
        weakest = min(components, key=components.get) if components else None
        labels = {"placement": "положение объекта", "prominence": "масштаб объекта", "edge_safety": "кадрирование у края", "background_calmness": "спокойствие фона", "tonal_separation": "отделение от фона"}
        tail = f" Слабейший компонент: {labels.get(weakest, weakest)}." if weakest else ""
        return f"Композиционный ориентир: {_f(value):.0f}/100. Это отдельная эвристика, не техническое качество.{tail}"

    if key == "main_subject":
        if not isinstance(raw, dict):
            return "Главный объект не определён."
        kind = str(raw.get("subject_kind", "unknown"))
        scene = str(raw.get("scene_kind", "unknown_scene"))
        confidence = _f(raw.get("confidence"), metric.confidence)
        area = _f(raw.get("subject_area_pct"))
        kind_labels = {
            "person": "главный человек",
            "people_group": "группа людей",
            "visual_region": "визуально выделяющаяся область",
            "unknown": "не определён",
        }
        scene_labels = {
            "archival_people": "архивный кадр с людьми",
            "people_group": "групповой портретный кадр",
            "people_portrait": "портретный кадр",
            "general_scene": "общая сцена",
            "unknown_scene": "сцена не определена",
        }
        if kind == "unknown":
            return "Надёжный главный объект не определён; программа не будет навязывать локальную область автоматически."
        return f"Определён {kind_labels.get(kind, kind)}; {scene_labels.get(scene, scene)}. Область около {area:.0f}% кадра, уверенность {confidence * 100:.0f}%."

    if key == "semantic_context":
        if not isinstance(raw, dict):
            return "Сюжетный контекст не определён."
        kind = str(raw.get("classification", "unknown"))
        people = str(raw.get("people_context", "none"))
        face_count = int(_f(raw.get("face_count")))
        archival = _f(raw.get("archival_likelihood"))
        if kind == "archival_portrait":
            group_note = "несколько лиц" if people == "group" else "одно лицо"
            return f"Архивный портретный контекст ({group_note}); найдено лиц: {face_count}. Приоритет — сохранить реальные черты лиц и характер оригинального отпечатка. Признак архивности около {archival * 100:.0f}%."
        if kind == "group_portrait":
            return f"Портретный кадр с несколькими лицами ({face_count}). Приоритет локальных решений — лица, особенно глаза и контуры."
        if kind == "portrait":
            return "Портретный кадр с одним уверенно найденным лицом. Приоритет локальных решений — лицо и глаза."
        if kind == "general_photo":
            return "Уверенного портретного контекста не найдено. Это не означает, что людей в кадре точно нет; базовая оценка контекста остаётся консервативной."
        return "Сюжетный контекст определён недостаточно уверенно; на технические оценки это не влияет."

    if key == "faces":
        if not isinstance(raw, dict):
            return "Недостаточно данных для локальной оценки лиц."
        if not bool(raw.get("detector_available", True)):
            return "Локальный детектор лиц недоступен; этот пункт не влияет на итоговую оценку."
        faces = raw.get("faces", [])
        count = int(_f(raw.get("face_count")))
        if count <= 0 or not isinstance(faces, list) or not faces:
            return "Лица не найдены. Это не дефект фотографии; локальная оценка лиц просто пропущена."
        brightness = [_f(face.get("brightness_linear")) for face in faces if isinstance(face, dict)]
        worst_brightness = min(brightness) if brightness else 0.0
        if score is not None and score < 45:
            note = " Одно из лиц также заметно тёмное." if worst_brightness < 0.22 else ""
            return f"Найдено лиц: {count}. По крайней мере одно лицо заметно мягче по высокочастотным признакам; нужна локальная проверка.{note}"
        if score is not None and score < 65:
            return f"Найдено лиц: {count}. Резкость хотя бы одного лица умеренная; перед усилением резкости желательно проверить глаза и контуры."
        return f"Найдено лиц: {count}. По базовым локальным признакам критичной потери резкости на лицах не видно."

    if key == "eyes":
        if not isinstance(raw, dict):
            return "Недостаточно данных для локальной оценки глаз."
        if not bool(raw.get("detector_available", True)):
            return "Детектор глаз недоступен; этот пункт не влияет на итоговую оценку."
        face_count = int(_f(raw.get("face_count")))
        eye_count = int(_f(raw.get("eye_count")))
        faces_with_eyes = int(_f(raw.get("faces_with_eyes")))
        if face_count <= 0:
            return "Лица не найдены, поэтому локальная оценка глаз пропущена."
        if eye_count <= 0:
            return "Глаза не удалось уверенно локализовать внутри найденных лиц. Это не означает, что глаза закрыты или плохого качества."
        if score is not None and score < 45:
            return f"Найдено глаз: {eye_count} в {faces_with_eyes} лицах. По крайней мере один глаз выглядит заметно мягким; проверить в 100% обязательно."
        if score is not None and score < 65:
            return f"Найдено глаз: {eye_count} в {faces_with_eyes} лицах. Резкость худшего глаза умеренная; локальное повышение резкости применять только после визуальной проверки."
        return f"Найдено глаз: {eye_count} в {faces_with_eyes} лицах. Критичной потери резкости по базовой локальной оценке не видно."

    if key == "detail_loss_type":
        if not isinstance(raw, dict):
            return "Недостаточно данных для классификации причины потери деталей."
        kind = str(raw.get("classification", "unknown"))
        conf = _f(raw.get("confidence"), metric.confidence)
        direction = raw.get("direction_deg")
        if kind in {"camera_shake_like", "subject_motion_like", "motion_like"}:
            direction_note = ""
            if direction is not None:
                try:
                    direction_note = f" Доминирующее направление градиентных признаков около {float(direction):.0f}°."
                except (TypeError, ValueError):
                    pass
            if kind == "camera_shake_like":
                return f"Направленный смаз по кадру вместе с EXIF выдержки больше похож на дрожание камеры, но это не доказательство причины ({conf * 100:.0f}% уверенности).{direction_note}"
            if kind == "subject_motion_like":
                return f"Главный объект заметно мягче фона и внутри него есть направленные признаки: вероятнее движение объекта, а не общий смаз кадра ({conf * 100:.0f}% уверенности).{direction_note}"
            return f"Есть направленный смаз, но по одному кадру нельзя надёжно отделить движение объекта от камеры ({conf * 100:.0f}% уверенности).{direction_note}"
        if kind == "local_subject_softness":
            return f"Главный объект заметно мягче информативного фона; проблема выглядит локальной, но её физическая причина пока не доказана ({conf * 100:.0f}% уверенности)."
        if kind == "defocus_like":
            return f"Потеря мелких деталей выглядит преимущественно изотропной по кадру, что больше похоже на дефокус или оптическую мягкость ({conf * 100:.0f}% уверенности)."
        if kind == "degradation_like":
            return f"Смешанные признаки вместе с профилем старого отпечатка больше похожи на деградацию/мягкость оригинала, а не на чистый дефокус или один направленный смаз ({conf * 100:.0f}% уверенности)."
        if kind in {"mixed_or_degraded", "mixed"}:
            return f"Признаки смешанные: уверенно отделить дефокус, движение и деградацию оригинала пока нельзя ({conf * 100:.0f}% уверенности)."
        return "Тип потери деталей пока не определён: недостаточно устойчивых признаков."

    if key == "analysis_reliability":
        if not isinstance(raw, dict):
            return "Недостаточно данных для оценки надёжности анализа."
        status = str(raw.get("status", "unknown"))
        calibrated = _f(raw.get("calibrated_confidence"), metric.confidence)
        ood = _f(raw.get("ood_score"), 0.0)
        labels = {
            "reliable": "Поддержка измерений согласованная",
            "caution": "Часть признаков поддержана слабее обычного",
            "unknown": "Недостаточно согласованных признаков для уверенного автоматического вывода",
            "ood_candidate": "Вход нетипичен для текущей эвристической калибровки",
        }
        note = labels.get(status, labels["unknown"])
        return f"{note}; калиброванная поддержка {calibrated * 100:.0f}%, сигнал нетипичного входа {ood * 100:.0f}%. Это не вероятность истины."

    if key == "local_tone":
        if not isinstance(raw, dict):
            return "Недостаточно данных для локальной карты тонов."
        dark = _f(raw.get("dark_area_pct"))
        deep = _f(raw.get("deep_shadow_area_pct"))
        bright = _f(raw.get("bright_area_pct"))
        clipped = _f(raw.get("highlight_clip_area_pct"))
        parts: list[str] = []
        if deep >= 4.0:
            parts.append(f"глубокие тени около {deep:.0f}% площади")
        if dark >= 8.0:
            parts.append(f"тёмные зоны около {dark:.0f}%")
        if bright >= 8.0:
            parts.append(f"светлые зоны около {bright:.0f}%")
        if clipped >= 3.0:
            parts.append(f"локально почти выбитые света около {clipped:.0f}%")
        if not parts:
            return "Локальное распределение тонов без выраженных крайностей. Карта носит информационный характер."
        return "Локальная карта: " + "; ".join(parts) + ". Это распределение сцены, а не автоматический список дефектов."

    if key == "local_sharpness":
        if not isinstance(raw, dict):
            return "Недостаточно данных для локальной карты резкости."
        soft_pct = _f(raw.get("soft_area_pct"))
        informative = int(_f(raw.get("informative_cells")))
        total = int(_f(raw.get("total_cells")))
        if informative <= 0:
            return "В кадре недостаточно текстурных областей для надёжной локальной оценки резкости."
        if soft_pct < 8.0:
            return f"Выраженных мягких зон среди информативных областей почти нет; оценено {informative} из {total} участков."
        if soft_pct < 25.0:
            return f"Есть отдельные мягкие зоны: около {soft_pct:.0f}% информативной площади. Их лучше проверить по карте резкости."
        return f"Заметная часть информативных областей выглядит мягкой: около {soft_pct:.0f}% площади. Карта поможет отделить локальный смаз от гладкого фона."

    if key == "local_contrast":
        if not isinstance(raw, dict):
            return "Недостаточно данных для локальной оценки контраста."
        low = _f(raw.get("low_contrast_area_pct"))
        median_range = _f(raw.get("median_local_range"))
        fade = _f(raw.get("fading_likelihood"))
        classification = str(raw.get("classification", "unknown"))
        if classification == "unknown":
            return "Недостаточно информативных участков, чтобы уверенно оценить локальный контраст и выцветание."
        if classification == "fading_candidate":
            return (
                f"Во многих информативных участках полутона сжаты: около {low:.0f}% площади имеет низкий локальный контраст. "
                f"Это похоже на выцветание или деградацию тонального разделения (признак {fade * 100:.0f}%, не окончательный диагноз)."
            )
        if classification == "mixed":
            return (
                f"Локальный контраст неоднородный: около {low:.0f}% информативной площади выглядит плоско. "
                f"Возможна частичная потеря полутонов; медианный локальный диапазон {median_range:.3f}."
            )
        return (
            f"Локальное тональное разделение в целом сохранено; участков с низким контрастом около {low:.0f}% "
            f"информативной площади."
        )

    if key == "surface_defects":
        if not isinstance(raw, dict):
            return "Недостаточно данных для оценки локальных дефектов поверхности."
        count = int(_f(raw.get("candidate_count")))
        shown = int(_f(raw.get("shown_count"), count))
        density = _f(raw.get("candidate_density_pct"))
        if count == 0:
            return "Явных кандидатов на тонкие царапины, трещины или пыль базовый детектор не нашёл."
        if count <= 4 and density < 0.12:
            return f"Найдено {count} слабых кандидата на локальные дефекты; нужна визуальная проверка карты."
        shown_note = f" На карте показаны {shown} наиболее заметных." if shown < count else ""
        return f"Найдено {count} кандидатов на локальные дефекты поверхности (около {density:.2f}% площади).{shown_note} Это не подтверждённые трещины: естественные линии могут давать ложные срабатывания."

    if key == "jpeg_artifacts":
        if not isinstance(raw, dict):
            return "Недостаточно данных для оценки JPEG-блочности."
        classification = str(raw.get("classification", "unknown"))
        ratio = _f(raw.get("block_ratio"), 1.0)
        source_is_jpeg = bool(raw.get("source_is_jpeg", False))
        source_note = " Исходный файл JPEG." if source_is_jpeg else " Артефакты могли быть унаследованы от более раннего JPEG-кодирования."
        if classification == "none":
            return "Характерной 8×8 блочности JPEG практически не видно."
        if classification == "mild":
            return f"Есть слабый признак JPEG-блочности (отношение границ блоков {ratio:.2f}).{source_note}"
        if classification == "moderate":
            return f"JPEG-блочность заметна и может маскироваться под шум или потерю деталей (отношение {ratio:.2f}).{source_note}"
        if classification == "strong":
            return f"Выраженная 8×8 блочность: изображение, вероятно, заметно пострадало от JPEG-компрессии (отношение {ratio:.2f}).{source_note}"
        return "Признаки JPEG-компрессии неоднозначны; нужна визуальная проверка в масштабе 100%."

    if key == "edge_artifacts":
        if not isinstance(raw, dict):
            return "Недостаточно данных для оценки ореолов и перешарпа."
        classification = str(raw.get("classification", "unknown"))
        halo = _f(raw.get("halo_likelihood"))
        ringing = _f(raw.get("ringing_likelihood"))
        if classification == "none":
            return "Характерных ореолов или звона вокруг сильных границ по текущей оценке не видно."
        if classification == "ringing_candidate":
            return (
                f"Есть кандидат на звон вокруг границ: колебания яркости продолжаются за пределами основной границы "
                f"(признак {ringing * 100:.0f}%). Это эвристика; контрастная фактура может имитировать эффект."
            )
        if classification == "mild_halo_candidate":
            return f"Есть слабый кандидат на ореолы/перешарп вокруг контрастных границ (признак {halo * 100:.0f}%)."
        if classification == "halo_candidate":
            return f"Ореолы вокруг сильных границ заметны по высокочастотному профилю (признак {halo * 100:.0f}%); перед повышением резкости нужна проверка 100%."
        if classification == "strong_halo_candidate":
            return f"Сильный кандидат на перешарп/ореолы (признак {halo * 100:.0f}%); дополнительная резкость может заметно ухудшить изображение."
        return "Признаки ореолов неоднозначны; нужна визуальная проверка контрастных границ в масштабе 100%."

    if key == "posterization":
        if not isinstance(raw, dict):
            return "Недостаточно данных для оценки постеризации."
        classification = str(raw.get("classification", "unknown"))
        occupied = int(_f(raw.get("occupied_luma_levels")))
        severity = _f(raw.get("severity"))
        if classification == "unknown":
            return "Недостаточно тонального диапазона или плавных переходов для надёжной проверки постеризации."
        if classification == "none":
            return "Явной постеризации или ступенчатых тональных переходов не обнаружено."
        if classification == "mild":
            return f"Есть слабый признак ступенчатых тональных переходов; занято около {occupied} уровней яркости."
        if classification == "moderate":
            return f"Постеризация/бэндинг заметны по структуре уровней (признак {severity * 100:.0f}%, около {occupied} уровней яркости)."
        if classification == "strong":
            return f"Выраженная постеризация: плавные градации разбиты на ступени (признак {severity * 100:.0f}%, около {occupied} уровней яркости)."
        return "Признаки постеризации неоднозначны; нужна визуальная проверка плавных градиентов."

    if key == "neutral_balance":
        if profile.key in {"sepia", "monochrome"}:
            return "Для сепии/монохромного кадра нейтрализация нейтральных полутонов не используется как автоматическая цель."
        if isinstance(raw, dict):
            cast = str(raw.get("cast_direction", "neutral"))
            strength = _f(raw.get("cast_strength"))
            r = _f(raw.get("neutral_median_r")); g = _f(raw.get("neutral_median_g")); b = _f(raw.get("neutral_median_b"))
            names = {"cool": "холодный", "warm": "тёплый", "green_magenta": "зелёно-пурпурный", "neutral": "нейтральный"}
            if strength < 0.025:
                return f"Нейтральные полутона сбалансированы: R/G/B ≈ {r:.0f}/{g:.0f}/{b:.0f}."
            return f"В нейтральных полутонах заметен {names.get(cast, cast)} сдвиг: R/G/B ≈ {r:.0f}/{g:.0f}/{b:.0f}; глобальные средние каналы могли его скрыть."
        return "Недостаточно нейтральных полутонов для устойчивой оценки цветового баланса."

    if key == "local_white_balance":
        if profile.key in {"sepia", "monochrome"}:
            return "Локальная карта цветового освещения отключена для сепии/монохромного кадра."
        if isinstance(raw, dict):
            kind = str(raw.get("classification", "uncertain"))
            conf = _f(raw.get("confidence"), metric.confidence) * 100.0
            coverage = _f(raw.get("coverage")) * 100.0
            mixed = _f(raw.get("mixed_light_score")) * 100.0
            supported = int(_f(raw.get("supported_cells")))
            if kind == "mixed_light":
                return f"Обнаружено неоднородное освещение: mixed-light {mixed:.0f}%, надёжные опоры покрывают около {coverage:.0f}% кадра ({supported} окон). Уверенность {conf:.0f}%."
            if kind == "local_cast":
                return f"Есть локальные различия цветового освещения, но они слабее порога mixed-light. Покрытие {coverage:.0f}%, уверенность {conf:.0f}%."
            if kind == "uniform":
                return f"Надёжные локальные опоры согласуются с единым источником света. Покрытие {coverage:.0f}%, уверенность {conf:.0f}%."
            return "Недостаточно надёжных нейтральных опор для локальной карты баланса белого; коррекция по клеткам не угадывается."
        return "Недостаточно данных для карты локального цветового освещения."

    if key == "white_balance_advisor":
        if profile.key in {"sepia", "monochrome"}:
            return "Советник баланса белого отключён для сепии/монохромного кадра, чтобы не уничтожить характер исходника."
        if isinstance(raw, dict):
            needed = bool(raw.get("correction_needed", False))
            temp = _f(raw.get("recommended_temperature_shift_k"))
            tint = _f(raw.get("recommended_tint_shift"))
            neutral = _f(raw.get("neutralization_strength")) * 100.0
            atmosphere = _f(raw.get("atmosphere_preservation")) * 100.0
            conf = _f(raw.get("confidence")) * 100.0
            if not needed:
                return f"Баланс белого близок к нейтральному; существенная коррекция не требуется. Уверенность {conf:.0f}%."
            direction = "охладить" if temp < 0 else "согреть"
            return (
                f"Рекомендуется {direction} кадр примерно на {abs(temp):.0f} K, tint {tint:+.1f}; "
                f"нейтрализация {neutral:.0f}% с сохранением {atmosphere:.0f}% атмосферы исходного света. "
                f"Уверенность {conf:.0f}%."
            )
        return "Недостаточно данных для совета по балансу белого."

    if key == "color_cast":
        if profile.key == "sepia":
            return "Тёплый сдвиг согласуется с профилем сепии/старого отпечатка. Не считаем его дефектом без дополнительного подтверждения."
        if profile.key == "monochrome":
            return "Кадр почти монохромный; обычная оценка баланса белого здесь малоинформативна."
        if isinstance(raw, dict):
            spread = _f(raw.get("relative_spread"))
            r = _f(raw.get("mean_r"))
            g = _f(raw.get("mean_g"))
            b = _f(raw.get("mean_b"))
            if spread < 0.07:
                return "Средние каналы близки друг к другу; выраженного цветового сдвига не видно."
            if b > g and b > r:
                return "Есть холодный цветовой сдвиг; перед коррекцией нужно учитывать сцену и источник света."
        if score is not None and score < 70:
            return "Есть цветовой сдвиг; автоматическая коррекция без учёта сцены пока не рекомендуется."
        return "Сильного глобального цветового сдвига по базовой оценке не обнаружено."

    return metric.diagnostic


def _status_for(key: str, score: float | None, confidence: float, profile: PhotoProfile) -> str:
    if key in {"color_cast", "neutral_balance", "white_balance_advisor", "local_white_balance"} and profile.key in {"sepia", "monochrome"}:
        return "Особенность"
    if key == "local_white_balance":
        if confidence < 0.40:
            return "Проверить"
        return "Внимание" if score is not None and score < 58 else "Инфо"
    if key == "white_balance_advisor":
        # The numeric score already expresses remaining cast; the advisor's raw
        # correction_needed flag decides whether user attention is useful.
        if score is None or confidence < 0.45:
            return "Проверить"
        return "Внимание" if score < 82 else "Норма"
    if key == "detail_loss_type":
        return "Инфо" if confidence >= 0.58 else "Проверить"
    if key in {"local_tone", "highlight_context", "exif_context", "nss_baseline", "semantic_context", "main_subject", "aesthetic_quality"}:
        return "Инфо"
    if key == "local_contrast":
        if score is None or confidence < 0.45:
            return "Проверить"
        if score < 35:
            return "Проблема"
        if score < 58:
            return "Внимание"
        if score < 78:
            return "Норма"
        return "Хорошо"
    if key == "local_sharpness":
        if score is None:
            return "Инфо"
        if confidence < 0.45:
            return "Проверить"
        if score < 45:
            return "Проблема"
        if score < 65:
            return "Внимание"
        return "Норма"
    if key == "surface_defects":
        return "Проверить" if score is not None and score < 99.5 else "Хорошо"
    if key in {"edge_artifacts", "posterization"} and confidence < 0.58:
        return "Проверить"
    if score is None:
        return "Инфо"
    if confidence < 0.50:
        return "Проверить"
    if score < 40:
        return "Проблема"
    if score < 65:
        return "Внимание"
    if score < 80:
        return "Норма"
    return "Хорошо"


def _sharpness_row(metrics: Mapping[str, MetricResult], profile: PhotoProfile) -> DisplayMetric | None:
    parts = [metrics[k] for k in ("laplacian", "tenengrad") if k in metrics]
    if not parts:
        return None
    scores = [m.normalized_value for m in parts if m.normalized_value is not None]
    score = sum(scores) / len(scores) if scores else None
    confidence = sum(m.confidence for m in parts) / len(parts)
    if score is None:
        diagnosis = "Недостаточно данных для итоговой оценки резкости."
    elif score < 35:
        diagnosis = "Резкость низкая; возможен смаз или сильная потеря мелких деталей."
    elif score < 55:
        diagnosis = "Резкость ниже средней; тип потери деталей ещё нужно определить."
    elif score < 75:
        diagnosis = "Резкость умеренная; мелкие детали частично сохранены."
    else:
        diagnosis = "По базовым высокочастотным признакам резкость хорошая."
    return DisplayMetric("sharpness", "Резкость", score, confidence, diagnosis, _status_for("sharpness", score, confidence, profile))


def build_display_metrics(metrics: Mapping[str, MetricResult]) -> list[DisplayMetric]:
    rows: list[DisplayMetric] = []
    profile = build_photo_profile(metrics)
    order = ["brightness", "shadow_clipping", "highlight_clipping", "highlight_context", "contrast", "neutral_balance", "white_balance_advisor", "local_white_balance"]
    for key in order:
        metric = metrics.get(key)
        if metric is not None:
            rows.append(DisplayMetric(
                key,
                _LABELS[key],
                metric.normalized_value,
                metric.confidence,
                _diagnosis(key, metric, profile),
                _status_for(key, metric.normalized_value, metric.confidence, profile),
            ))

    sharp = _sharpness_row(metrics, profile)
    if sharp is not None:
        rows.append(sharp)

    exif_metric = metrics.get("exif_context")
    if exif_metric is not None:
        rows.append(DisplayMetric(
            "exif_context",
            _LABELS["exif_context"],
            None,
            exif_metric.confidence,
            _diagnosis("exif_context", exif_metric, profile),
            _status_for("exif_context", None, exif_metric.confidence, profile),
        ))

    nss_metric = metrics.get("nss_baseline")
    if nss_metric is not None:
        rows.append(DisplayMetric(
            "nss_baseline",
            _LABELS["nss_baseline"],
            None,
            nss_metric.confidence,
            _diagnosis("nss_baseline", nss_metric, profile),
            _status_for("nss_baseline", None, nss_metric.confidence, profile),
        ))

    semantic_metric = metrics.get("semantic_context")
    if semantic_metric is not None:
        rows.append(DisplayMetric(
            "semantic_context",
            _LABELS["semantic_context"],
            None,
            semantic_metric.confidence,
            _diagnosis("semantic_context", semantic_metric, profile),
            _status_for("semantic_context", None, semantic_metric.confidence, profile),
        ))

    subject_metric = metrics.get("main_subject")
    if subject_metric is not None:
        rows.append(DisplayMetric(
            "main_subject",
            _LABELS["main_subject"],
            None,
            subject_metric.confidence,
            _diagnosis("main_subject", subject_metric, profile),
            _status_for("main_subject", None, subject_metric.confidence, profile),
        ))

    aesthetic_metric = metrics.get("aesthetic_quality")
    if aesthetic_metric is not None:
        rows.append(DisplayMetric(
            "aesthetic_quality",
            _LABELS["aesthetic_quality"],
            aesthetic_metric.normalized_value,
            aesthetic_metric.confidence,
            _diagnosis("aesthetic_quality", aesthetic_metric, profile),
            _status_for("aesthetic_quality", aesthetic_metric.normalized_value, aesthetic_metric.confidence, profile),
        ))

    reliability_metric = metrics.get("analysis_reliability")
    if reliability_metric is not None:
        rows.append(DisplayMetric(
            "analysis_reliability",
            _LABELS["analysis_reliability"],
            None,
            reliability_metric.confidence,
            _diagnosis("analysis_reliability", reliability_metric, profile),
            (
                "Норма" if str(reliability_metric.raw_value.get("status", "unknown")) == "reliable"
                else "Внимание" if str(reliability_metric.raw_value.get("status", "unknown")) == "caution"
                else "Проверить"
            ) if isinstance(reliability_metric.raw_value, dict) else "Проверить",
        ))

    detail_metric = metrics.get("detail_loss_type")
    if detail_metric is not None:
        rows.append(DisplayMetric(
            "detail_loss_type",
            _LABELS["detail_loss_type"],
            None,
            detail_metric.confidence,
            _diagnosis("detail_loss_type", detail_metric, profile),
            _status_for("detail_loss_type", None, detail_metric.confidence, profile),
        ))

    tone_metric = metrics.get("local_tone")
    if tone_metric is not None:
        rows.append(DisplayMetric(
            "local_tone",
            _LABELS["local_tone"],
            None,
            tone_metric.confidence,
            _diagnosis("local_tone", tone_metric, profile),
            _status_for("local_tone", None, tone_metric.confidence, profile),
        ))

    contrast_metric = metrics.get("local_contrast")
    if contrast_metric is not None:
        rows.append(DisplayMetric(
            "local_contrast",
            _LABELS["local_contrast"],
            contrast_metric.normalized_value,
            contrast_metric.confidence,
            _diagnosis("local_contrast", contrast_metric, profile),
            _status_for("local_contrast", contrast_metric.normalized_value, contrast_metric.confidence, profile),
        ))

    local_metric = metrics.get("local_sharpness")
    if local_metric is not None:
        rows.append(DisplayMetric(
            "local_sharpness",
            _LABELS["local_sharpness"],
            local_metric.normalized_value,
            local_metric.confidence,
            _diagnosis("local_sharpness", local_metric, profile),
            _status_for("local_sharpness", local_metric.normalized_value, local_metric.confidence, profile),
        ))

    face_metric = metrics.get("faces")
    if face_metric is not None:
        rows.append(DisplayMetric(
            "faces",
            _LABELS["faces"],
            face_metric.normalized_value,
            face_metric.confidence,
            _diagnosis("faces", face_metric, profile),
            _status_for("faces", face_metric.normalized_value, face_metric.confidence, profile),
        ))

    eye_metric = metrics.get("eyes")
    if eye_metric is not None:
        rows.append(DisplayMetric(
            "eyes",
            _LABELS["eyes"],
            eye_metric.normalized_value,
            eye_metric.confidence,
            _diagnosis("eyes", eye_metric, profile),
            _status_for("eyes", eye_metric.normalized_value, eye_metric.confidence, profile),
        ))

    for key in ("noise", "jpeg_artifacts", "edge_artifacts", "posterization", "surface_defects", "color_cast"):
        metric = metrics.get(key)
        if metric is not None:
            rows.append(DisplayMetric(
                key,
                _LABELS[key],
                metric.normalized_value,
                metric.confidence,
                _diagnosis(key, metric, profile),
                _status_for(key, metric.normalized_value, metric.confidence, profile),
            ))
    return rows


def build_summary(metrics: Mapping[str, MetricResult]) -> dict[str, float | int | str]:
    rows = build_display_metrics(metrics)
    profile = build_photo_profile(metrics)
    # Style-dependent color balance must not lower technical quality for sepia/monochrome photos.
    # Local sharpness, faces and eyes refine *where* the sharpness problem is.
    # They must not multiply the same semantic defect inside the global quality score.
    refinement_keys = {"local_sharpness", "faces", "eyes", "aesthetic_quality"}
    quality_rows = [
        r for r in rows
        if r.score is not None
        and r.confidence >= 0.50
        and r.key not in refinement_keys
        and not (r.key == "color_cast" and profile.key in {"sepia", "monochrome"})
    ]
    if not quality_rows:
        return {
            "quality": 0.0,
            "potential": 0.0,
            "confidence": 0.0,
            "issues": 0,
            "profile": profile.label,
            "profile_confidence": round(profile.confidence * 100.0, 1),
        }
    weight_sum = sum(max(r.confidence, 0.05) for r in quality_rows)
    quality = sum(float(r.score) * max(r.confidence, 0.05) for r in quality_rows) / weight_sum
    confidence = sum(r.confidence for r in quality_rows) / len(quality_rows) * 100.0
    reliability_metric = metrics.get("analysis_reliability")
    if reliability_metric is not None and isinstance(reliability_metric.raw_value, dict):
        confidence = 100.0 * _f(
            reliability_metric.raw_value.get("calibrated_confidence"),
            reliability_metric.confidence,
        )
    issue_groups: set[str] = set()
    issue_group_for_key = {
        "brightness": "exposure",
        "shadow_clipping": "exposure",
        "highlight_clipping": "exposure",
        "contrast": "contrast",
        "local_contrast": "contrast",
        "sharpness": "sharpness",
        "local_sharpness": "sharpness",
        "faces": "sharpness",
        "eyes": "sharpness",
        "noise": "noise",
        "jpeg_artifacts": "compression",
        "edge_artifacts": "processing_artifacts",
        "posterization": "processing_artifacts",
        "color_cast": "color",
        "surface_defects": "surface",
    }
    issue_rows = [r for r in rows if r.score is not None and r.confidence >= 0.50]
    for row in issue_rows:
        if row.key == "color_cast" and profile.key in {"sepia", "monochrome"}:
            continue
        if row.status in {"Проблема", "Внимание"}:
            issue_groups.add(issue_group_for_key.get(row.key, row.key))
    issues = len(issue_groups)
    potential = min(100.0, max(0.0, (100.0 - quality) * 1.15))
    return {
        "quality": round(quality, 1),
        "potential": round(potential, 1),
        "confidence": round(confidence, 1),
        "issues": issues,
        "profile": profile.label,
        "profile_confidence": round(profile.confidence * 100.0, 1),
    }


def build_recommendation_groups(metrics: Mapping[str, MetricResult]) -> RecommendationGroups:
    rows = {r.key: r for r in build_display_metrics(metrics)}
    profile = build_photo_profile(metrics)
    validation_metric = metrics.get("recommendation_validation")
    validation_raw = validation_metric.raw_value if validation_metric is not None and isinstance(validation_metric.raw_value, dict) else {}
    validation_items = validation_raw.get("items", []) if isinstance(validation_raw, dict) else []
    validation_by_key: dict[str, dict[str, object]] = {}
    if isinstance(validation_items, list):
        for item in validation_items:
            if isinstance(item, dict) and item.get("tested"):
                validation_by_key[str(item.get("action_key", ""))] = item
    safe: list[str] = []
    caution: list[str] = []
    avoid: list[str] = []

    reliability_metric = metrics.get("analysis_reliability")
    reliability_raw = reliability_metric.raw_value if reliability_metric is not None and isinstance(reliability_metric.raw_value, dict) else {}
    reliability_status = str(reliability_raw.get("status", "reliable"))
    if reliability_status == "caution":
        caution.append("Общая поддержка анализа ниже обычной: перед сохранением внимательно сравнить предпросмотр с исходником.")
    elif reliability_status == "unknown":
        caution.append("Недостаточно согласованных признаков для уверенного автоматического вывода; коррекции оставлены для ручной проверки.")
        avoid.append("Не применять коррекции автоматически без предпросмотра: активна общая защита при неопределённости.")
    elif reliability_status == "ood_candidate":
        caution.append("Изображение выглядит нетипичным для текущей эвристической калибровки Photo Doctor; результаты использовать только как подсказку.")
        avoid.append("Не применять автоматические коррекции без ручной проверки: активна защита при нетипичном входе.")

    if (row := rows.get("brightness")) and row.score is not None and (row.score < 65 or "exposure" in validation_by_key):
        highlight_metric = metrics.get("highlight_context")
        tone_metric = metrics.get("local_tone")
        clipped = 0.0
        if highlight_metric is not None and isinstance(highlight_metric.raw_value, dict):
            clipped = _f(highlight_metric.raw_value.get("unexplained_clip_pct"))
        elif tone_metric is not None and isinstance(tone_metric.raw_value, dict):
            clipped = _f(tone_metric.raw_value.get("highlight_clip_area_pct"))
        if clipped < 3.0:
            exposure_validation = validation_by_key.get("exposure")
            if exposure_validation is not None and not bool(exposure_validation.get("accepted", False)):
                if not bool(exposure_validation.get("auto_eligible", True)):
                    caution.append("Модуль решений оставил подъём средних тонов для ручной проверки; мягкий пробный вариант можно включить галочкой в «Исправлениях».")
                else:
                    caution.append("Проверка безопасности рекомендаций не подтвердила автоматический подъём средних тонов на пробной копии; мягкий вариант можно проверить вручную.")
            else:
                safe.append("Аккуратно поднять средние тона: локальная карта не показывает большой площади почти выбитых светов.")
        else:
            caution.append("Кадр тёмный, но локально есть очень светлые участки: поднимать средние тона только с защитой светов.")
    if (row := rows.get("contrast")) and row.score is not None and row.score < 58:
        contrast_validation = validation_by_key.get("contrast")
        if contrast_validation is not None and not bool(contrast_validation.get("accepted", False)):
            if not bool(contrast_validation.get("auto_eligible", True)):
                caution.append("Модуль решений оставил контраст для ручной проверки; мягкую пробную коррекцию можно включить галочкой в «Исправлениях».")
            else:
                caution.append("Проверка безопасности рекомендаций не подтвердила автоматическое усиление контраста на пробной копии; мягкий вариант можно проверить вручную.")
        else:
            safe.append("Проверить мягкое восстановление глобального и локального контраста без пережима чёрной точки.")
    if (row := rows.get("local_contrast")) and row.score is not None and row.score < 58:
        metric = metrics.get("local_contrast")
        fade = 0.0
        classification = "unknown"
        if metric is not None and isinstance(metric.raw_value, dict):
            fade = _f(metric.raw_value.get("fading_likelihood"))
            classification = str(metric.raw_value.get("classification", "unknown"))
        if classification == "fading_candidate":
            safe.append("Проверить мягкое восстановление локального контраста и полутонов; признаки похожи на выцветание, но коррекцию лучше ограничить плоскими областями.")
            caution.append("Не лечить выцветание одной глобальной кривой: чёрная точка и белые детали могут быть уже нормальными, а потеря находится внутри полутонов.")
        elif fade >= 0.36:
            caution.append("Часть областей выглядит плоско: локальный контраст повышать выборочно, сверяясь с лицами и фактурой одежды.")
    if (row := rows.get("sharpness")) and row.score is not None and row.score < 58:
        detail_metric = metrics.get("detail_loss_type")
        kind = "unknown"
        conf = 0.0
        if detail_metric is not None and isinstance(detail_metric.raw_value, dict):
            kind = str(detail_metric.raw_value.get("classification", "unknown"))
            conf = _f(detail_metric.raw_value.get("confidence"), detail_metric.confidence)
        if kind == "camera_shake_like" and conf >= 0.58:
            caution.append("Есть признаки общего направленного смаза, согласующиеся с дрожанием камеры: обычное повышение резкости ограничено; устранение смаза пробовать только на копии и оценивать лица/контуры в 100%.")
        elif kind == "subject_motion_like" and conf >= 0.58:
            caution.append("Главный объект похож на движущийся относительно более резкого фона: не применять устранение смаза ко всему кадру; пробовать только локально на главном объекте.")
        elif kind == "motion_like" and conf >= 0.58:
            caution.append("Есть направленный смаз, но источник не доказан: обычное усиление резкости ограничено; сначала проверить локальное устранение смаза на копии.")
        elif kind == "local_subject_softness" and conf >= 0.52:
            caution.append("Главный объект мягче фона без надёжно установленной причины: глобальную резкость не усиливать; корректировать только область главного объекта после проверки в 100%.")
        elif kind == "defocus_like" and conf >= 0.58:
            caution.append("Потеря деталей похожа на дефокус/оптическую мягкость: применять умеренное локальное повышение резкости без попытки дорисовать отсутствующие детали.")
        elif kind == "degradation_like":
            caution.append("Для старого отпечатка признаки больше похожи на деградацию оригинала: сначала восстанавливать тон и локальный контраст, а резкость усиливать очень умеренно.")
        else:
            caution.append("Тип потери деталей определён недостаточно уверенно; не выбирать устранение смаза или коррекцию дефокуса автоматически.")
    if (row := rows.get("local_sharpness")) and row.score is not None and row.score < 65:
        caution.append("Открыть карту резкости и проверить мягкие области отдельно: гладкий фон алгоритм старается не считать смазом.")
    if (row := rows.get("faces")) and row.score is not None and row.score < 60:
        caution.append("Проверить лица в масштабе 100% отдельно от общей резкости кадра; локальное усиление допустимо только после проверки глаз, рта и контуров.")
        avoid.append("Не применять агрессивное или генеративное восстановление лица автоматически: оно может изменить реальные черты и дорисовать детали.")
    if (row := rows.get("eyes")) and row.score is not None and row.score < 60:
        caution.append("Проверить найденные глаза в масштабе 100%: глазная резкость важнее фоновой, но маленький кандидат детектора Хаара может быть неточным.")
        avoid.append("Не дорисовывать глаза генеративно автоматически: при низкой исходной детализации это меняет реальные черты лица.")
    if (row := rows.get("noise")) and row.score is not None and row.score < 70:
        caution.append("Оценить шум по лицам и ровным областям; применять шумоподавление только с защитой мелких деталей.")
    if (row := rows.get("jpeg_artifacts")) and row.score is not None and row.score < 70:
        caution.append("Перед усилением резкости проверить JPEG-блочность в 100%: повышение резкости может подчеркнуть границы блоков 8×8 и звон.")
        if row.score < 45:
            caution.append("При выраженной JPEG-компрессии сначала пробовать мягкое смягчение блоков JPEG на копии, а уже потом оценивать необходимость резкости/шумоподавления.")
    if (row := rows.get("edge_artifacts")) and row.score is not None and row.score < 70:
        caution.append("Перед любым дополнительным повышением резкости проверить контрастные границы в 100%: обнаружены кандидаты на ореолы/перешарп.")
        if row.score < 45:
            avoid.append("Не усиливать резкость глобально при сильном кандидате на ореолы: сначала уменьшить звон/ореолы на копии или оставить исходную резкость.")
    if (row := rows.get("posterization")) and row.score is not None and row.score < 70:
        caution.append("Проверить плавные градиенты в 100%: при постеризации сильный контраст, насыщенность и резкость могут сделать ступени заметнее.")
        if row.score < 45:
            caution.append("При выраженной постеризации сначала пробовать очень мягкое локальное сглаживание полосатости с лёгким дизерингом на копии, не размывая лица и фактуру.")
    if (row := rows.get("shadow_clipping")) and row.score is not None and row.score < 80:
        caution.append("Проверить глубокие тени локально: глобальное осветление может ухудшить фон и контраст.")
    if (row := rows.get("highlight_clipping")) and row.score is not None and row.score < 80:
        caution.append("Проверить выбитые света локально; восстановление возможно только там, где исходные данные ещё сохранились.")

    highlight_metric = metrics.get("highlight_context")
    if highlight_metric is not None and isinstance(highlight_metric.raw_value, dict):
        candidate_count = int(_f(highlight_metric.raw_value.get("candidate_count")))
        face_rejected = int(_f(highlight_metric.raw_value.get("face_rejected_count")))
        if candidate_count > 0:
            safe.append("Компактные кандидаты на зеркальные блики/источники света не штрафовать как обычный пересвет до визуальной проверки карты бликов.")
        if face_rejected > 0:
            caution.append("Есть очень яркие области внутри найденного лица: они специально не исключены из потерь в светах и требуют ручной проверки кожи/деталей.")

    exif_metric = metrics.get("exif_context")
    exif_raw = exif_metric.raw_value if exif_metric is not None else None
    if isinstance(exif_raw, dict) and bool(exif_raw.get("available", False)):
        detail_metric = metrics.get("detail_loss_type")
        detail_raw = detail_metric.raw_value if detail_metric is not None else None
        detail_kind = str(detail_raw.get("classification", "unknown")) if isinstance(detail_raw, dict) else "unknown"
        if detail_kind in {"motion_like", "camera_shake_like"} and str(exif_raw.get("shutter_risk", "unknown")) == "high":
            caution.append("EXIF поддерживает гипотезу смаза камеры: выдержка медленнее ориентира 1/фокусное. Это не доказательство — стабилизация, опора и движение объекта неизвестны.")
        noise_row = rows.get("noise")
        if noise_row is not None and noise_row.score is not None and noise_row.score < 70 and str(exif_raw.get("iso_noise_risk", "unknown")) == "high":
            caution.append("Высокое ISO в EXIF согласуется с найденным шумом; шумоподавление всё равно ограничивать по лицам и мелкой фактуре.")

    semantic_metric = metrics.get("semantic_context")
    semantic_raw = semantic_metric.raw_value if semantic_metric is not None else None
    if isinstance(semantic_raw, dict):
        semantic_kind = str(semantic_raw.get("classification", "unknown"))
        if semantic_kind == "archival_portrait":
            safe.append("Для архивного портрета сохранять характер исходного отпечатка и реальные пиксели лиц; глобальные коррекции делать обратимыми и умеренными.")
            avoid.append("Не использовать автоматическое генеративное восстановление лиц в архивном портрете без отдельного ручного подтверждения результата.")
        elif semantic_kind in {"portrait", "group_portrait"}:
            caution.append("Портретный контекст: перед глобальной резкостью, шумоподавлением и контрастом отдельно проверить лица и глаза в 100%.")

    surface_metric = metrics.get("surface_defects")
    if surface_metric is not None and isinstance(surface_metric.raw_value, dict):
        surface_count = int(_f(surface_metric.raw_value.get("candidate_count")))
        if surface_count > 0:
            caution.append("Открыть карту кандидатов на дефекты и визуально проверить отмеченные области перед любой ретушью.")
            avoid.append("Не удалять найденные линии автоматически: простой детектор может спутать трещину с волосами, швами одежды или контуром объекта.")

    if profile.key == "sepia":
        avoid.append("Не нейтрализовать сепию автоматически: тёплый тон, вероятно, относится к характеру исходного отпечатка.")
    elif profile.key == "monochrome":
        avoid.append("Не пытаться автоматически «исправлять баланс белого» почти монохромного изображения.")
    elif (row := rows.get("color_cast")) and row.score is not None and row.score < 72:
        caution.append("Перед цветокоррекцией проверить нейтральные области и источник освещения; глобальная нейтрализация может исказить сцену.")

    avoid.append("Не применять сильную резкость, шумоподавление или восстановление лиц без локальной проверки: эти операции могут дорисовать несуществующие детали.")

    if not safe:
        safe.append("Явных безопасных глобальных коррекций по текущим базовым метрикам не требуется.")
    if not caution:
        caution.append("Следующий полезный шаг — локальный анализ лиц, смаза и дефектов поверхности.")

    return RecommendationGroups(tuple(safe), tuple(caution), tuple(avoid))


def build_recommendations(metrics: Mapping[str, MetricResult]) -> list[str]:
    """Backward-compatible flat recommendation list for CLI/tests."""
    groups = build_recommendation_groups(metrics)
    out = [*groups.safe, *groups.caution, *groups.avoid]
    out.append("Автоматическое изменение файла не выполняется: это диагностический этап текущей версии.")
    return out
