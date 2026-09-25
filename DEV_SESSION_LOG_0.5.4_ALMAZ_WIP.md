# Photo Doctor — DEV SESSION LOG 0.5.4 ALMAZ WIP

Дата: 2026-09-22
Статус: WORK IN PROGRESS — НЕ РЕЛИЗ
База: Photo Doctor 0.5.3.26 FINAL

## 1. Цель ALMAZ

Консервативное восстановление старых/маленьких фото без генеративной подмены деталей:
- Super Resolution только x2;
- JPEG/compression recovery;
- AI Denoise;
- AI Deblur;
- Face/Identity Guard;
- post-inference safety validator;
- тяжёлый inference только по явному preview/save, не во время обычного анализа.

## 2. Что уже реализовано

### Low Resolution / SR
- `core/super_resolution.py`.
- Решение о необходимости x2 использует реальные размеры исходника, а не bounded technical copy Analyzer.
- Почти пустые/однотонные кадры не получают x2-рекомендацию только из-за малого размера.
- x2 candidate всегда manual/review, auto apply запрещён.
- При отсутствии проверенной ONNX-модели остаётся классический консервативный x2 fallback.
- Face Identity Guard ослабляет изменение внутри лиц.

### Lazy preview
- Тяжёлый ALMAZ inference не выполняется внутри обычного Validator analysis.
- x2 / AI restoration запускаются только при явном предпросмотре или save-copy.
- GUI использует background preview worker; основной интерфейс не должен замерзать на inference.

### ONNX runtime layer
- SR: `ai/sr_backend.py`, `sr_catalog.py`, `sr_runtime.py`.
- x1 restoration: `ai/restoration_backend.py`, `restoration_catalog.py`, `restoration_runtime.py`.
- Static 256x256 tiled contracts с overlap blending.
- Runtime требует manifest, matching SHA256 и `parity_status=PASS`; неподтверждённая модель fail-closed.
- Execution Provider discovery: CUDA / DirectML / OpenVINO / CPU; QNN заложен архитектурно, но не объявляется универсально готовым.

### Candidate models / exporters
- SwinIR-S lightweight x2 — кандидат для SR.
- NAFNet-SIDD — Denoise.
- NAFNet-GoPro — Deblur.
- NAFNet-REDS — Blur/JPEG recovery.
- `tools/export_almaz_swinir.py`, `export_almaz_nafnet.py`, `install_almaz_onnx.py`.
- Экспорт считается пригодным только после PyTorch↔ONNX parity, manifest и SHA.
- Реальные веса в текущем контейнере НЕ встроены: загрузка release assets из внешней сети недоступна. Не заявлять AI model bundled/verified until actual artifact is installed and checked.

### ALMAZ status / runtime setup
- AI UI показывает состояние x2, Denoise, Deblur, JPEG Recovery и доступный provider.
- ORT installer разделён по runtime-профилям; не устанавливает несколько конфликтующих ORT-пакетов одновременно.

### ALMAZ safety
- `core/almaz_safety.py` валидирует каждую ALMAZ-трансформацию сразу до/после конкретного AI-действия, а не весь correction recipe.
- Проверяет размерный контракт, среднюю дельту, лица, Lab chroma, clipping и edge-energy.
- Для почти плоского источника edge ratio не используется бессмысленно; вместо этого блокируется появление новых сильных контуров.
- При safety BLOCK preview/save fail-closed.

### Глобальная область AI restoration
- ALMAZ x2, AI Denoise, AI Deblur и AI JPEG Recovery сейчас только full-frame.
- GUI показывает `Всё фото ×2` / `Всё фото · ALMAZ`.
- Локальный AI-mask intentionally disabled до отдельной проверки mask/tiling seams.

### Adaptive strength
- Denoise/Deblur/JPEG Recovery больше не используют одну магическую силу для всех кадров.
- default strength зависит от измеренной деградации + Decision Engine severity.
- На кадрах с лицами сила снижается, face protection повышена.
- Всё равно manual/review only до калибровки на реальных фото.

## 3. Проверки текущего WIP

Завершённые блоки:
- ALMAZ dedicated backend/catalog/runtime/integration suite: 45/45 PASS до adaptive-strength change.
- safety + SR + restoration + GUI global-area suite: 23/23 PASS.
- adaptive restoration + safety: 13/13 PASS.
- GUI startup + packaging ранее: 24/24 PASS.
- runtime/profile/packaging/status ранее: 27/27 PASS.
- `compileall src/tools`: PASS на предыдущем checkpoint; повторить после текущих изменений.

`test_core.py`:
- полный foreground вызов несколько раз терялся по внешнему TransportTimeout оболочки;
- процесс при наблюдении был жив и активно грузил CPU, то есть это не подтверждённый hang PhotoDoctor;
- отдельно `test_noise_increases_noise_proxy`: 1/1 PASS (~16.4 s).
- Для финального suite использовать file-by-file runner с START/END/PID/timeout, а не длинный foreground вызов.

## 4. Зафиксированные принципы

- x2 only, x4 не делать.
- Не снижать качество/разрешение ради скорости.
- Не обещать «AI работает», если verified ONNX model отсутствует.
- Лица: чуть мягче лучше, чем изменённая идентичность.
- Автоматическое применение ALMAZ пока запрещено; только рекомендация + preview + safety.
- Не смешивать timeout оболочки/runner с реальным hang продукта.

## 5. Следующие шаги

1. Повторить весь ALMAZ suite после adaptive-strength изменения.
2. `compileall src tests tools`.
3. Проверить обычные Analyzer/Validator regression groups, особенно preview/cache/packaging.
4. Запустить полный suite file-by-file под watchdog с журналом START/END.
5. После зелёного suite собрать WIP checkpoint, но не называть релизом без реальной ONNX-модели и проверки на контрольных старых фото.
6. На Windows установить/экспортировать первый настоящий SwinIR x2 ONNX, проверить CUDA/DirectML/CPU parity и реальный preview.

## 6. Full-suite checkpoint — 2026-09-22

Фактический полный прогон после ALMAZ Phase 1/1.5 изменений:
- `pytest --collect-only`: **614 tests collected**;
- file-by-file runner: **73/73 test-files PASS**;
- `almaz_full_suite.exit`: **0**;
- watchdog timeouts: **0**;
- тяжёлый `tests/test_core.py`: **14/14 PASS**, ~81.8 s pytest / 87 s wall;
- `tests/test_validator.py`: **19/19 PASS**;
- `tests/test_white_balance.py`: **18/18 PASS**;
- `tests/test_white_balance_integration.py`: **4/4 PASS**.

Это подтверждает, что ALMAZ WIP не ломает существующие Analyzer/Validator/GUI/input-format/Surface/WB контуры. Версия намеренно остаётся `0.5.3.26`; ветку пока НЕ называть 0.5.4 release до фактической установки и проверки реальной ONNX-модели на контрольных фотографиях.

## 7. Recovery + final checkpoint continuation — 2026-09-22

Во время длительного полного прогона рабочая подпапка неожиданно исчезла из `/mnt/data`; сохранённый checkpoint ZIP остался цел. Ветка восстановлена строго из `PhotoDoctor_0.5.4_ALMAZ_WIP_CHECKPOINT_2026-09-22.zip`, после чего поздние изменения были воспроизведены и заново проверены.

Восстановлены поздние изменения после предыдущего checkpoint:
- единый `tools/almaz_download.py`: mirror fallback, `.part`, exact size + SHA-256, atomic commit only after verification;
- SwinIR-S x2 закреплён на official checkpoint SHA-256 `193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7`, размер 17147989, source commit `33f616625268d08ba600f8db89388eec0328edb1`;
- NAFNet SIDD/GoPro/REDS закреплены на конкретные checkpoint SHA/size и source commit `2b4af71ebe098a92a75910c233a3965a3e93ede4`;
- exporters проверяют PTH identity ДО `torch.load`;
- ONNX runtime и installer fail-closed проверяют `source_checkpoint_sha256`, `source_checkpoint_size`, `source_commit` вместе с parity + ONNX SHA;
- `prepare_almaz_swinir.py` и `prepare_almaz_nafnet.py` выполняют verified download → pinned source → export → parity → install;
- GUI получил диагностическую кнопку `Подготовить ALMAZ x2` для локального `.venv`; в frozen EXE этот путь намеренно не используется;
- correction preview order стал candidate-aware: ALMAZ Denoise/JPEG/Deblur идут до x2, обычная финальная резкость — после x2;
- добавлены regressions downloader/order/source-identity.

Финальная проверка восстановленного дерева:
- `pytest --collect-only`: **620 tests collected**;
- file-by-file runner: **75/75 test-files PASS**;
- runner exit: **0**;
- watchdog timeouts: **0**;
- `tests/test_core.py`: **14/14 PASS**;
- `tests/test_validator.py`: **19/19 PASS**;
- `tests/test_white_balance.py`: **18/18 PASS**;
- `tests/test_white_balance_integration.py`: **4/4 PASS**;
- broad restored ALMAZ/GUI/packaging regression: **180/180 PASS**;
- `compileall src tests tools`: PASS.

Версия внутри дерева намеренно остаётся `0.5.3.26`: это ALMAZ WIP checkpoint, НЕ 0.5.4 release. Реальные SwinIR/NAFNet ONNX weights в архив не включены, потому что текущая среда не смогла скачать внешние binary assets. Runtime честно остаётся на безопасном fallback до verified ONNX install.

## 2026-09-22 — ALMAZ UI visibility fix
- User confirmed the checkpoint launches, but ALMAZ was effectively undiscoverable.
- Root cause: ALMAZ setup lived under `ИИ -> Расширенно -> Диагностика`; ALMAZ corrections appeared in Plan only after an eligible photo was analyzed.
- Added a permanent top-level `ALMAZ` tab immediately after `План`.
- The tab exposes four modules explicitly: Super Resolution x2, AI Denoise, AI Deblur, JPEG Recovery.
- Added visible runtime/provider state, per-model state, per-model preparation buttons, a combined restoration preparation button, link back to Plan, current-photo ALMAZ recommendation state, and preparation log.
- Added GUI wiring for `prepare_almaz_nafnet.py --task denoise|deblur|jpeg_recovery|all`.
- Existing hidden advanced diagnostics remain available for technical inspection.
- Verification after UI fix: targeted ALMAZ/Validator/packaging/GUI suite 170/170 PASS; GUI source/layout/startup subset 88/88 PASS; `compileall src tests tools` PASS.
- A monolithic full `pytest -q` was attempted but the outer execution wrapper timed out around 46%; no test failure was reported before the wrapper timeout. The functional ALMAZ algorithms were unchanged by this UI-only fix.

## 2026-09-22 — WB preview cache hotfix
- User reproduced stale preview after changing White Balance strength several times.
- Root weakness: preview cache key depended on an explicit recipe revision and selected action keys; effective strengths/regions were not themselves part of the key.
- Added `core/preview_cache.py` with a hashable effective-recipe signature including selected actions, effective strengths, candidates and regions.
- `MainWindow._preview_key()` now embeds that recipe signature, so WB 35% and WB 60% cannot reuse the same cache entry even if a future UI call site forgets explicit invalidation.
- Cache status text now explicitly says cached preview belongs to current settings.
- Added regression tests proving WB 35% vs 60% produces different cache keys and different preview pixels.
- Verification after fix: targeted WB/cache/Validator 54/54 PASS; broader GUI/packaging/WB/cache 160/160 PASS; compileall src/tests/tools PASS.

## 2026-09-22 — ALMAZ rename + model preparation dependency fix
- Renamed the restoration codename to ALMAZ across UI, modules, tools, tests, logs, candidate IDs and filenames.
- Added `tools/almaz_prepare_env.py`: detects missing export dependencies and installs only missing packages into the active Python/.venv.
- SwinIR prep needs torch + timm + onnx + an importable onnxruntime provider.
- NAFNet prep needs torch + onnx + an importable onnxruntime provider.
- Existing ORT GPU/DirectML/OpenVINO package is preserved if `onnxruntime` imports successfully.
- ALMAZ preparation failures now show the last real log lines in a critical dialog instead of only exit code 1.
- Full current suite: 634 tests; 77/77 files PASS; exit 0; timeouts 0.

## 2026-09-22 — ALMAZ progress + safety recovery
- ALMAZ preview worker now reports real pipeline stages with elapsed time.
- ONNX x1/x2 tiled backends report tile progress N/M; GUI displays current stage and elapsed seconds.
- Model preparation status now follows the latest real subprocess stage and elapsed time.
- Safety guard no longer hard-fails immediately: unsafe ALMAZ output is blended toward a safe baseline at 75%, 50%, then 35% of the calculated model effect and revalidated after each attempt.
- If a softened variant passes, preview continues and reports the effective safety reduction. If every softened variant fails, ALMAZ still blocks the correction and preserves the source.
- Safety rejection is shown as a warning/protection event rather than an application crash.
- Regression: 641 tests collected; canonical file-by-file suite 77/77 PASS, exit 0, watchdog timeouts 0.

## 2026-09-23 — ALMAZ split actions
- ALMAZ AI Denoise / AI Deblur / JPEG Recovery are now independent manual actions.
- They no longer replace the ordinary Noise / Sharpness / JPEG correction rows.
- Plan UI adds explicit ALMAZ rows when the corresponding verified model is ready.
- Each module has an independent checkbox and strength value, enabling one-at-a-time artifact diagnosis.
- Focused regression: 46/46 PASS; wider GUI/cache/packaging/decision regression: 127/127 PASS.

## 2026-09-23 — Manual Preview Refresh FINAL edge audit
- Продолжение после `ALMAZ_MANUAL_PREVIEW_REFRESH`.
- Найден edge-case: снятие последней выбранной коррекции могло оставить на экране старую исправленную картинку при уже пустом выборе.
- `_mark_preview_stale_after_recipe_change()` теперь при пустом выборе отжимает preview, возвращает `current_rgb`, сбрасывает stale-state и пишет, что показан исходник.
- `_almaz_preview_completed()` различает изменённый рецепт и полностью пустой выбор: при пустом выборе старый результат отбрасывается без ложного `Обновить предпросмотр`.
- `_almaz_preview_failed()` теперь сверяет ключ рецепта; ошибка устаревшего расчёта не показывается пользователю как ошибка актуальных настроек.
- Добавлены регрессии на все три edge-case.
- Итог: 677 тестов собрано; 82/82 test-files PASS; compileall PASS.

## 2026-09-23 — Archive low-contrast crack detector v5

Пользовательский референс с сильно повреждённым архивным снимком выявил системный false-negative: старая морфологическая ветка хорошо видела яркие дефекты на тёмной одежде, но дробила или пропускала слабоконтрастные трещины на лице и светлом фоне.

Сделано:
- surface detector переведён на `context_aware_morphology_v5_low_contrast_hysteresis`;
- добавлен ограниченный hysteresis: слабая линия может продолжить подтверждённую сильную только на ограниченное расстояние, без бесконтрольного разрастания по текстуре;
- добавлена low-contrast Hough-ветка для длинных тонких линий с проверкой локального профиля яркости по обе стороны сегмента;
- введены фильтры краёв сканирования/рамки, повторяющихся параллельных структур и текстурного риска;
- Hough-кандидаты сохраняются узкими повёрнутыми контурами, а не широкими прямоугольниками, чтобы ручное восстановление затрагивало минимальную область;
- low-contrast Hough-кандидаты никогда не получают `Лечить` автоматически: они показываются и могут оцениваться AI, но требуют ручной галки;
- дополнительно кандидаты с `semantic_risk >= 0.35` не выбираются для лечения автоматически;
- на пользовательском архивном референсе старый Haar-детектор лиц вернул 0 лиц, поэтому review-first применяется независимо от распознавания лица;
- пользовательский референс добавлен в `tests/fixtures/archive_low_contrast_cracks_user_reference.png`;
- добавлены regression-тесты bounded hysteresis, реального low-contrast reference и запрета auto-repair для чувствительной ветки.

Проверки рабочего дерева перед упаковкой:
- `pytest --collect-only -q`: 680 тестов;
- 82/82 test-файла PASS (длинные наборы запускались частями из-за внешнего лимита вызова оболочки, без test failures);
- targeted surface/UI regression: 97/97 PASS.


## 2026-09-23 — 0.5.5 Surface v6 cache/face/light-background repair
- Root cause после V5: app/algorithm version не были подняты, поэтому DB cache мог отдавать старую surface map.
- Найден второй разрыв: validation выполнялся до Surface AI, GUI seed использовал pre-AI first-8 fallback, low_contrast_hough и semantic face candidates были manual-only/hidden by default.
- 0.5.5: cache invalidation, 48 Surface AI candidates normal mode, promotion confirmed AI-pool candidates, narrow low-contrast context rescue, shared face-safe auto-repair guardrail, post-AI validator synchronization.
- User archive fixture: 45/45 candidates evaluated; confirmed repair mask reaches face and light background; changed-pixel coverage remains <3%.

## 2026-09-23 — 0.5.6 Surface V7 convergence/history
- User clarified the large 4->5 image change likely included other enabled corrections; excluded that transition from Surface-stability conclusions.
- Added persistent Surface repair history inside saved PNG/JPEG metadata.
- Added overlap-based previously-repaired suppression without blocking adjacent crack tails.
- Added conservative ridge-supported short-gap linking for fragmented cracks.
- Added residual convergence rule so repeated Surface-only passes stop auto-proposing tiny weak leftovers.
- GUI auto-selection now respects residual_manual_only.
- Real archival-photo Surface-only validation converged 9 -> 5 -> 3 residual candidates; the third set is manual-only.
- Version 0.5.6; algorithm cache key 0.5.6-surface-v7.
