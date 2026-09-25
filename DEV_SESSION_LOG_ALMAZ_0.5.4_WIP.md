# Photo Doctor — DEV SESSION LOG ALMAZ 0.5.4 WIP

Дата: 2026-09-22
Статус: WORK IN PROGRESS, НЕ РЕЛИЗ
База: Photo Doctor 0.5.3.26 FINAL.
Версия внутри дерева намеренно пока остаётся 0.5.3.26 до полного закрытия ALMAZ-ветки.

## Цель

Добавить консервативный блок восстановления «ALMAZ» для старых цифровых фото. Первый этап: только x2, без x4, с приоритетом сохранения реальной информации и идентичности лица.

## Реализовано в этой сессии

- Новый `core/super_resolution.py`:
  - Low Resolution Detector;
  - рекомендация x2 только для действительно небольших/недетализированных кадров;
  - low-information gate: почти однотонные кадры не получают бессмысленную SR-рекомендацию;
  - мягкая pre-clean цепочка JPEG/noise;
  - conservative x2 fallback;
  - Face / Identity Guard;
  - поддержка нормализованных координат лиц 0..1.
- `analyzer.py`: метрика `super_resolution` добавлена в обычный анализ без выполнения тяжёлого x2.
- `decisions.py`: пункт `Восстановление разрешения x2`, только `review`, не авто-применение.
- `validator.py`:
  - ALMAZ candidate является LAZY: реальный x2 не считается во время `analyze_file()`;
  - preview manual-only / auto_eligible=False;
  - ALMAZ выполняется последним в correction chain;
  - локальная пользовательская область к ALMAZ не применяется, т.к. операция меняет размер всего кадра.
- GUI:
  - подпись `ALMAZ — x2 восстановление с защитой лица`;
  - `CorrectionPreviewWorker(QThread)` для ALMAZ;
  - тяжёлый preview больше не блокирует GUI;
  - progress bar остаётся в режиме «Выполняется…»;
  - устаревший результат отбрасывается, если рецепт изменился во время расчёта;
  - save-copy для ALMAZ использует уже готовый preview cache и не пересчитывает SR;
  - закрытие окна не убивает работающий ALMAZ-worker.
- ONNX backend architecture:
  - `ai/sr_backend.py`;
  - фиксированный контракт `rgb01_nchw_static256_x2_v1`;
  - static tile 256x256 -> 512x512;
  - overlap tiling + blend;
  - providers: CUDA, DirectML, OpenVINO, CPU; QNN определяется отдельно;
  - QNN/NPU не считается совместимым с обычным float-моделем автоматически.
- Model runtime security:
  - `ai/sr_runtime.py`;
  - модель ищется в `~/.photodoctor/models` / `PHOTODOCTOR_MODEL_DIR`;
  - обязательны `.onnx` + `.onnx.json` manifest;
  - проверяются model_id, contract_id и SHA-256;
  - mismatch блокирует запуск;
  - успешная ONNX-session кэшируется по path+mtime+size+SHA;
  - при любой ошибке ONNX/inference выполняется безопасный classical fallback.
- Первый кандидат реальной модели:
  - SwinIR-S Classical x2 / lightweight;
  - upstream asset `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`;
  - Apache-2.0;
  - non-GAN / non-generative;
  - planned ONNX: `swinir_s_classical_x2_256.onnx`;
  - отдельный QDQ/quantized artifact потребуется для QNN NPU.
- Добавлен `tools/export_swinir_s_x2_onnx.py`:
  - требует официальный checkout JingyunLiang/SwinIR и официальный `.pth`;
  - конфигурация official lightweight SR x2;
  - export static 256->512, opset 18;
  - `onnx.checker`;
  - генерирует manifest с SHA исходного `.pth` и ONNX.

## Проверки

- ALMAZ/Validator/Decisions первоначальный пакет: 31/31 PASS после lazy-preview fix.
- ALMAZ + backend + runtime + catalog + Validator + Decisions + preview-cache + GUI startup + packaging: 74/74 PASS.
- `tests/test_core.py` прогнан по одному test node на отдельный процесс с watchdog 120 s: 14/14 PASS, timeout 0. Полные времена в `ALMAZ_CORE_TEST_STATUS_2026-09-22.tsv`.
- `pytest --collect-only`: 579 tests collected.
- `python -m compileall -q src tests`: PASS.

## Зафиксированные зависоны / ложные зависоны

1. Полный `pytest tests/test_core.py` в одном долгоживущем процессе неоднократно упирался во внешний timeout оболочки после 2 тестов.
2. Тот же файл, прогнанный по одному node в отдельных процессах: 14/14 PASS, timeout 0.
3. Вывод: это не подтверждённый product regression; для тяжёлых проверок продолжать использовать separate-process runner + START/END + watchdog.
4. В начале ALMAZ Validator ошибочно строил x2 прямо во время анализа, из-за чего тесты резко замедлились. Исправлено архитектурно: x2 теперь lazy и запускается только по явному preview/save path.

## Внешняя модель

Официальный SwinIR-S x2 asset найден в GitHub release, но текущая контейнерная среда не имеет рабочего DNS/скачивания release binary. Случайные зеркала не использовать. Конвертер готов; фактический `.pth -> ONNX` запуск выполнить в среде с доступом к официальному asset и пакетом `onnx`.

## Следующая точка

1. ALMAZ Phase 2: отдельные рекомендации/preview для AI Denoise и Deblur/Recover Detail с теми же safety gates.
2. Добавить отображение активного SR backend/provider в AI/technical UI.
3. После получения официального SwinIR-S x2: экспорт ONNX, SHA pin, parity A/B PyTorch vs ONNX, CPU timing, затем CUDA timing на Windows.
4. Только после parity решать QNN quantization; обязательно сравнить качество лиц/тонких деталей до включения NPU.
5. Полный suite 579+ после Phase 2, затем version bump 0.5.4.x и release verification.

## 2026-09-22 — Windows cp1251 model preparation crash fixed
User reproduced ALMAZ AI Denoise preparation failing with exit code 1. Traceback showed UnicodeEncodeError in cp1251 on U+2713 CHECK MARK in tools/almaz_prepare_env.py before model preparation could continue.

Fix:
- preparation QProcess environment forces PYTHONUTF8=1 / PYTHONIOENCODING=utf-8;
- prepare helper propagates UTF-8 environment to child pip/export processes;
- console decorative glyphs replaced by ASCII-safe [OK] / ... / <->;
- regression covers cp1251 encodability of preparation output.

Verification: focused ALMAZ/model-prep 93/93 PASS; GUI/startup/packaging 106/106 PASS; input/cache/WB 46/46 PASS; compileall PASS.

## 2026-09-23 — ALMAZ Deblur Color Fix
- Пользователь локализовал зелёные пятна: виновник AI Deblur на любых фото.
- Проверены checkpoint SHA, официальный NAFNet-GoPro config и RGB runtime contract — несоответствий не найдено.
- Причина: domain-specific GoPro deblur + слишком мягкий local chroma gate для цветных фото.
- Добавлен universal Deblur Chroma Guard, local p99 threshold 6 Lab, снижена базовая сила GoPro Deblur.
- Regression: ordinary colourful photo + synthetic green islands; raw BLOCK, guarded output suppresses chroma while preserving luminance detail.
- Full suite: 672/672 PASS, 82/82 files.
