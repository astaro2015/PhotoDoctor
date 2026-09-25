# Аудит интеграции labeling/provenance — 20.09.2026

## Что проверено

- Исходный `SHA256SUMS.json`: 202/202 записей совпали до внесения reconciliation-правок.
- Golden/review: 60 исходных фото, 480 кандидатов, ручных D/N/U/Skip пока нет.
- Stage8 review manifest: 480 строк, 60 source groups, все `golden_test`, admission=0 ожидаемо.
- FILM-AA: 12 135 строк D, 10 source groups, все остаются в quarantine; train/validation/golden/user_feedback/review пусты.
- Никакое обучение не запускалось, predictions не превращались в ground truth.

## Найденные несовместимости

1. `parameter_evaluation.environment()` безусловно читал production `.npz`, хотя веса намеренно исключены из handoff. Reconciled-копия теперь фиксирует отсутствующие reference-файлы в `production_missing`, а не падает.
2. ONNX-тест требует `skl2onnx`/`onnxruntime` из training requirements. В окружении без этих optional training dependencies тест явно SKIP, а не ложный FAIL.
3. Intake provenance использует `source_tier=expert/synthetic`, а Stage8 gate принимает `licensed_pro/synthetic_ground_truth`. Добавлен `provenance_normalization.py`, сохраняющий исходный tier в `provenance_source_tier` и нормализующий Stage8-представление.
4. В Photo Doctor 0.5.3.10 surface feedback exporter не выдавал явные provenance-поля, хотя correction feedback уже был маркирован. Это исправлено в Photo Doctor 0.5.3.11.

## Тесты

- Исходный handoff в текущем окружении: 98 PASS / 2 FAIL. Один FAIL из-за намеренно отсутствующего `.npz`, второй из-за неустановленного `skl2onnx`.
- Reconciled handoff: 103 PASS / 1 SKIP.
- Photo Doctor 0.5.3.11: 416/416 PASS.

## Что изменено в Photo Doctor 0.5.3.11

Surface D/N/U export теперь содержит:

- `label`: D/N/U;
- `source_tier=user_feedback`;
- `label_origin=user_manual`;
- `expert_verified=false`;
- `training_lane=hard_mining`;
- `license=private-local-only`;
- `use_scope=preference_hard_case_only`;
- `source_group_id`;
- `feature_schema=native_gray96_hograwmeta_mlp_binary_v1`.

Parameter feedback остаётся в lane `preference`. Пользовательская разметка не повышается до expert/base автоматически.

Добавлен интеграционный тест на временной SQLite: correction feedback + surface feedback -> training ZIP. Проверяется provenance и отсутствие пути/имени исходной фотографии.

## Ограничение проверки GUI

В текущем Linux-контейнере нет PySide6, поэтому полностью автоматизированный Qt-сценарий реального нажатия `Сохранить исправленную копию` не запускался. Core persistence/export path проверен интеграционным тестом, а исходный GUI-код по-прежнему вызывает запись feedback только после успешного Save. Финальный пользовательский GUI-smoke следует выполнить на Windows-сборке с временной тестовой БД, не на production DB.
