from __future__ import annotations

from typing import Any

# Display-only localization. Internal keys, database values and model contracts
# intentionally remain unchanged for backwards compatibility.

_KEY_LABELS = {
    "model_id": "модель",
    "task": "задача",
    "contract_id": "контракт",
    "confidence": "уверенность",
    "classification": "классификация",
    "label": "метка",
    "probabilities": "вероятности",
    "status": "состояние",
    "status_label": "состояние",
    "detail": "подробности",
    "reason": "причина",
    "method": "метод",
    "region": "область",
    "source": "источник",
    "score": "оценка",
    "score_0_100": "оценка_0_100",
    "score_0_1": "оценка_0_1",
    "coverage_pct": "покрытие_проц",
    "mask_shape": "размер_маски",
    "candidate_count": "число_кандидатов",
    "candidate_density_pct": "плотность_кандидатов_проц",
    "evaluated_count": "проверено_кандидатов",
    "confirmed_defect_count": "подтверждено_дефектов",
    "likely_natural_count": "естественных_деталей",
    "uncertain_count": "неоднозначных",
    "face_count": "число_лиц",
    "eye_count": "число_глаз",
    "manual_face_count": "лиц_вручную",
    "manual_eye_count": "глаз_вручную",
    "runtime_available": "среда_внешних_моделей_доступна",
    "runtime_version": "версия_среды_ИИ",
    "router_version": "версия_маршрутизатора_ИИ",
    "catalog_version": "версия_каталога_моделей",
    "ready_count": "готовых_моделей",
    "requested_count": "запрошено_задач",
    "successful_inferences": "успешных_запусков_ИИ",
    "attempted_inferences": "попыток_запуска_ИИ",
    "native_ready_count": "готовых_встроенных_моделей",
    "external_ready_count": "готовых_внешних_моделей",
    "external_count": "внешних_моделей",
    "external_installed_count": "установленных_внешних_моделей",
    "catalog_count": "моделей_в_каталоге",
    "blocked_count": "заблокировано",
    "resources": "ресурсы",
    "plan": "план",
    "patch_batch_size": "размер_пакета_фрагментов",
    "working_memory_budget_gib": "бюджет_памяти_ГБ",
    "max_surface_candidates": "максимум_кандидатов_дефектов",
    "logical_cpus": "логических_потоков_процессора",
    "total_ram_gib": "оперативная_память_ГБ",
    "available_ram_gib": "свободная_оперативная_память_ГБ",
    "raw_classification": "сырая_классификация",
    "raw_confidence": "сырая_уверенность",
    "subject_kind": "тип_главного_объекта",
    "scene_kind": "тип_сцены",
    "subject_area_pct": "площадь_главного_объекта_проц",
    "box_norm": "рамка_объекта",
    "protection_box_norm": "защитная_рамка",
    "face_indices": "индексы_лиц",
    "reasons": "причины",
    "components": "компоненты",
    "people_context": "контекст_людей",
    "archival_likelihood": "вероятность_архивного_контекста",
    "preservation_priority": "приоритет_сохранения_характера",
    "calibrated_confidence": "калиброванная_поддержка",
    "ood_score": "оценка_нетипичности",
    "ood_candidate": "кандидат_нетипичного_входа",
    "quality": "техническое_качество",
    "potential": "потенциал",
    "issues": "число_проблем",
    "profile": "профиль",
    "profile_confidence": "уверенность_профиля",
    "decision": "решение",
    "accepted": "подтверждено",
    "adjustable": "регулируется",
    "default_strength": "сила_по_умолчанию",
    "action_key": "тип_коррекции",
    "target_before": "целевая_метрика_до",
    "target_after": "целевая_метрика_после",
    "message": "сообщение",
    "validator_version": "версия_проверки_безопасности",
    "tested_count": "проверено",
    "accepted_count": "подтверждено",
    "rejected_count": "отклонено",
    "algorithm_version": "версия_алгоритма",
    "series_algorithm_version": "версия_алгоритма_серий",
    "created_at": "создано",
    "saved_format": "формат_сохранения",
    "user_selected": "выбрано_пользователем",
    "validator_accepted": "подтверждено_проверкой_безопасности",
    "validation_confidence": "уверенность_проверки_безопасности",
    "ai_suggested_strength": "сила_предложенная_ИИ",
    "user_strength": "сила_выбранная_пользователем",
    "ai_parameter_confidence": "уверенность_рекомендателя_ИИ",
    "parameter_features": "признаки_для_рекомендателя",
    "source_token": "идентификатор_источника",
    "save_event_token": "идентификатор_сохранения",
    "decision_event": "событие_решения",
    "photo_doctor_version": "версия_программы",
    "privacy": "приватность",
    "readiness": "готовность_данных",
    "format": "формат",
    "rows": "строк",
    "strength_samples": "примеров_силы",
    "by_action": "по_типам_коррекции",
    "ready_actions": "готовые_для_эксперимента_коррекции",
    "model_family": "семейство_модели",
    "note": "примечание",
}

_VALUE_LABELS = {
    "global": "всё изображение",
    "face": "лицо",
    "faces": "лица",
    "eye": "глаз",
    "eyes": "глаза",
    "manual": "вручную",
    "ready": "готово",
    "ok": "готово",
    "cached": "из кэша",
    "error": "ошибка",
    "partial_error": "частичная ошибка",
    "not_run": "не запускалось",
    "missing": "отсутствует",
    "runtime_missing": "нет среды внешних моделей",
    "unverified": "не проверено",
    "hash_mismatch": "контрольная сумма не совпала",
    "contract_mismatch": "контракт несовместим",
    "unreadable": "файл не читается",
    "not_in_catalog": "нет в каталоге",
    "unknown": "не определено",
    "mixed": "смешанный случай",
    "reliable": "надёжно",
    "caution": "с осторожностью",
    "ood_candidate": "нетипичный вход",
    "advisory_ok": "можно учитывать",
    "advisory_caution": "с осторожностью",
    "manual_review": "только вручную",
    "ignore": "игнорировать",
    "agree": "согласуется",
    "refine": "уточняет",
    "contradict": "противоречит",
    "low_confidence": "низкая уверенность",
    "not_comparable": "нечего сравнивать",
    "native_numpy": "встроенный локальный исполнитель",
    "onnx": "внешняя модель ONNX",
    "person": "человек",
    "people_group": "группа людей",
    "visual_region": "визуально значимая область",
    "archival_people": "люди на архивном фото",
    "people_portrait": "портрет людей",
    "general_scene": "общая сцена",
    "unknown_scene": "сцена не определена",
    "archival_portrait": "архивный портрет",
    "group_portrait": "групповой портрет",
    "portrait": "портрет",
    "general_photo": "обычная фотография",
    "monochrome": "монохром",
    "sepia": "сепия",
    "color": "цветное фото",
    "camera_shake_like": "похоже на дрожание камеры",
    "subject_motion_like": "похоже на движение главного объекта",
    "motion_like": "направленный смаз",
    "local_subject_softness": "главный объект мягче фона",
    "defocus_like": "похоже на дефокус или оптическую мягкость",
    "degradation_like": "похоже на деградацию оригинала",
    "mixed_or_degraded": "смешанная потеря деталей или деградация",
    "none": "нет",
    "mild": "слабо",
    "moderate": "умеренно",
    "strong": "сильно",
    "very_low": "очень низко",
    "low": "низко",
    "high": "высоко",
    "ringing_candidate": "кандидат на звон вокруг границ",
    "mild_halo_candidate": "слабый кандидат на ореол",
    "halo_candidate": "кандидат на ореол",
    "strong_halo_candidate": "сильный кандидат на ореол",
    "fading_candidate": "кандидат на выцветание",
    "defect": "дефект",
    "natural_detail": "естественная деталь",
    "uncertain": "не уверен",
    "true": "да",
    "false": "нет",
}

_MODEL_LABELS = {
    "native_surface_verifier_v2": "Surface AI v2 — проверка дефектов поверхности",
    "native_surface_refiner_v1": "Уточнитель дефектов поверхности v1 (устаревший)",
    "surface_refiner_v1": "Уточнитель дефектов поверхности v1 (устаревший)",
    "parameter_recommender_v2": "Рекомендатель параметров ИИ v2",
    "native_white_balance_advisor_v1": "Советник баланса белого v1",
    "face_quality_v1": "Оценка качества лица v1",
}

_TASK_LABELS = {
    "blur_refinement": "Уточнение типа потери деталей",
    "surface_defect_refinement": "Уточнение дефектов поверхности",
    "face_quality": "Качество лица",
    "iqa_refinement": "Дополнительная оценка качества",
    "semantic_context_refinement": "Уточнение контекста кадра",
}


def localize_model_id(value: Any) -> str:
    text = str(value or "—")
    if text in _MODEL_LABELS:
        return _MODEL_LABELS[text]
    if text.startswith("custom_"):
        suffix = text.removeprefix("custom_").replace("_", " ").strip()
        return f"Пользовательская модель {suffix}".strip()
    return text


def localize_task(value: Any) -> str:
    text = str(value or "—")
    return _TASK_LABELS.get(text, _VALUE_LABELS.get(text, text))


def localize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if value is None:
        return None
    if isinstance(value, str):
        if value in _MODEL_LABELS:
            return _MODEL_LABELS[value]
        if value in _TASK_LABELS:
            return _TASK_LABELS[value]
        return _VALUE_LABELS.get(value, value)
    return value


def localize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            display_key = _KEY_LABELS.get(str(key), str(key))
            if str(key) == "model_id":
                out[display_key] = localize_model_id(item)
            elif str(key) == "task":
                out[display_key] = localize_task(item)
            else:
                out[display_key] = localize_payload(item)
        return out
    if isinstance(value, list):
        return [localize_payload(item) for item in value]
    if isinstance(value, tuple):
        return [localize_payload(item) for item in value]
    return localize_value(value)


def localize_metric_name(name: str) -> str:
    mapping = {
        "brightness": "Яркость",
        "histogram": "Гистограмма",
        "shadow_clipping": "Провалы в тенях",
        "highlight_clipping": "Потери в светах",
        "highlight_context": "Контекст бликов",
        "contrast": "Глобальный контраст",
        "sharpness": "Резкость",
        "laplacian": "Резкость по Лапласиану",
        "tenengrad": "Резкость по градиенту",
        "noise": "Шум",
        "jpeg_artifacts": "Артефакты JPEG",
        "edge_artifacts": "Ореолы и звон границ",
        "posterization": "Постеризация и полосатость градаций",
        "color_cast": "Цветовой сдвиг",
        "surface_defects": "Дефекты поверхности",
        "faces": "Лица",
        "eyes": "Глаза",
        "red_eye": "Красные глаза",
        "local_sharpness": "Локальная резкость",
        "local_tone": "Локальные тона",
        "local_contrast": "Локальный контраст",
        "detail_loss_type": "Тип потери деталей",
        "exif_context": "Контекст EXIF",
        "nss_baseline": "Базовая NSS-оценка",
        "semantic_context": "Контекст кадра",
        "main_subject": "Главный объект",
        "aesthetic_quality": "Эстетическая и композиционная оценка",
        "analysis_reliability": "Надёжность анализа",
        "decision_plan": "План решений",
        "recommendation_validation": "Проверка безопасности рекомендаций",
        "analysis_profile": "Профиль анализа",
        "image_tone": "Характер тона изображения",
        "ai_status": "Состояние ИИ",
        "almaz_runtime": "ALMAZ — модели / ускоритель",
        "ai_inference": "Результаты ИИ",
        "ai_crosscheck": "Сверка ИИ с классическим анализом",
        "ai_trust_gate": "Проверка доверия к ИИ",
        "surface_refinement": "Уточнение дефектов поверхности",
    }
    return mapping.get(name, name)
