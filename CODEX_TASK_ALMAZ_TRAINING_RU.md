# CODEX TASK: довести ALMAZ до полноценного fine-tuning и провести большой учебный прогон

## Цель
Не делать игрушечное обучение. Подготовить и выполнить полноценный fine-tuning существующих ALMAZ-моделей Photo Doctor на разнообразных данных, с закрытым экзаменом и жёстким safety-gate. Smoke-набор разрешён только для проверки кода и никогда не может породить релизный checkpoint.

## Неприкосновенные правила
1. Не обучать модели с нуля без отдельного доказательства необходимости. Стартовать с текущих pinned SwinIR x2 / NAFNet task-specific checkpoints.
2. `train`, `val`, `exam` разделять по исходному изображению/source_group. Никаких кропов одного исходника в разных split.
3. `exam` никогда не используется для подбора параметров, early stopping, augmentation или loss weights.
4. Пользовательские D/N/U и сохранённые силы Photo Doctor = hard-mining / preference data, не экспертный ground truth.
5. Новая модель не принимается только за лучший PSNR/SSIM. Обязателен `tools/almaz_exam_gate.py` и ручной визуальный экзамен.
6. На почти монохромных/сепийных фото запрещено создавать новые локальные зелёные/пурпурные/синие chroma-islands.
7. Лица: сохранение идентичности важнее дополнительной резкости. Мягче допустимо, выдуманные глаза/рот/нос/морщины недопустимы.
8. Каждая внешняя картинка должна иметь происхождение, license id/rights statement, URL и checksum, если он доступен. Источники с NC/ND/research-only/неясными правами не входят в release training без явного ручного разрешения.
9. Не ослаблять Validator/Safety, пороги или экзамен ради прохождения модели.
10. Все длительные операции печатают текущий этап, N/M или %, elapsed time и ETA, если ETA можно оценить честно.

## Уже подготовлено в репозитории
- `training/almaz/TRAINING_CONTRACT.json` — минимальные объёмы, split-policy и release gate.
- `training/almaz/SOURCES.json` — начальный реестр источников и лицензионная политика.
- `tools/almaz_training_contract.py` — dataset gate.
- `tools/almaz_training_coach.py` — preflight/оркестратор; не разрешает релизное обучение на малом датасете.
- `tools/almaz_exam_gate.py` — финальный safety gate checkpoint.
- существующие `prepare/export/install` ALMAZ scripts и pinned upstream architectures/checkpoints.

## Обязательный объём первого полноценного прогона
Соблюсти `TRAINING_CONTRACT.json`; минимум:
- 100 000 train pair/crops суммарно;
- >= 2 500 независимых train source groups;
- >= 250 val source groups;
- >= 500 закрытых exam source groups;
- SR x2 >= 25k пар;
- Denoise >= 25k;
- Deblur >= 20k;
- JPEG Recovery >= 20k;
- Archive Restore >= 10k;
- exam должен содержать modern/archive/faces/near-monochrome домены согласно contract.
Это минимумы, а не цели. Если лицензированного материала хватает, увеличивать разнообразие/source groups предпочтительнее многократного размножения одного фото.

## Источники первого прохода
### Разрешены по умолчанию для release-training
- SIDD full paired sRGB: real denoise lane. Официальная страница указывает MIT.
- GOPRO_Large: deblur lane. Официальная страница указывает CC BY 4.0; сохранить attribution metadata.
- Public Domain / CC0 / явно допустимые CC BY изображения из Library of Congress Free to Use and Reuse, только после item-level rights filter.

### Только после проверки прав
- REDS: текущая pretrained NAFNet JPEG-модель может быть REDS-trained, но прежде чем использовать сами данные в новом release fine-tuning, зафиксировать реальные условия датасета в `SOURCES.lock.json`.
- DIV2K: не включать в release-training автоматически, пока права для конкретного использования не очищены.
- FFHQ: не использовать для release-training по умолчанию; искать permissive/public-domain лица для экзамена.

## Что реализовать
### A. Dataset acquisition
1. `tools/almaz_fetch_datasets.py`:
   - resume downloads;
   - SHA256/size verification when known;
   - retries/mirrors;
   - `source_ledger.jsonl` с URL, retrieved_at, license, attribution, checksum;
   - не скачивать blocked source без `--allow-source <id>`.
2. Для LOC: item-level rights filter. Автоматически принимать только явно разрешённые contract policy статусы. Всё неоднозначное в quarantine manifest, не в train.
3. Уметь продолжать после прерывания без повторного скачивания целых архивов.

### B. Pair/crop builder
Сделать `tools/almaz_build_dataset.py` и воспроизводимые деградации (seed фиксируется в manifest):
- SR x2: downsampling kernels, mild preblur, ringing, JPEG/chroma-subsampling combinations; target остаётся clean.
- Denoise: приоритет реальным SIDD pairs; synthetic Gaussian/Poisson/chroma/read noise как дополнительный curriculum, не замена реальному шуму.
- Deblur: реальные GoPro pairs + синтетический motion/defocus только как дополнение.
- JPEG Recovery: quality range, 4:4:4/4:2:2/4:2:0 where implementation permits, ringing/resize history; clean target.
- Archive Restore: synthetic scratches/cracks/dust/fading/stains/uneven illumination/sepia on clean permissive/public-domain scans. Не превращать archive task в colorization.
- near-monochrome subset обязательно маркировать.
- manifest row обязан иметь `source_group_id`, `task`, `split`, `source_id`, `license`, `degradation_recipe`, `seed`, input/target paths.
- deduplicate perceptually/cryptographically enough to avoid train/exam leakage.

### C. Trainers
Реализовать task-specific fine-tune, не ломая текущие архитектуры:
- NAFNet denoise: старт с pinned SIDD width32 checkpoint.
- NAFNet deblur: старт с pinned GoPro width32.
- NAFNet jpeg_recovery: старт с pinned REDS width64 checkpoint, но dataset terms must be cleared before using REDS training data.
- SwinIR x2: старт с текущего pinned lightweight x2 checkpoint.
- Archive lane: сначала не заводить пятую огромную модель без необходимости. Проверить fine-tune/mixture на наиболее подходящей x1 architecture; если отдельная модель объективно нужна, обосновать benchmark'ом.
- AMP при CUDA; gradient accumulation для 12 GB VRAM; deterministic seed; resume checkpoints.
- gradient clipping, periodic validation, NaN/Inf fail-fast.
- сохранять optimizer/scheduler/RNG state для честного resume.

### D. Loss / safety curriculum
Обычная reconstruction loss недостаточна. Добавить/исследовать:
- L1/Charbonnier/PSNR-style reconstruction;
- edge/detail term умеренного веса;
- chroma-preservation loss для near-monochrome/sepia;
- identity-preservation term/mask для лиц без генеративного «улучшательства»;
- clipping penalty;
- hallucinated-edge penalty.
Не добавлять perceptual/GAN-loss только ради «красивой резкости» без отдельного доказательства, что лица/архивы не страдают.

### E. Closed exam
Создать `tools/almaz_eval_checkpoint.py`, который сравнивает candidate с текущим baseline на одном и том же immutable exam manifest и пишет JSON, совместимый с `tools/almaz_exam_gate.py`:
- PSNR/SSIM delta;
- face identity drift delta;
- near-monochrome chroma artifact delta;
- clipping delta pp;
- false-edge ratio delta;
- task_quality_delta per task;
- примеры худших 25 regressions как контакт-листы/пути;
- ручной флаг `manual_exam_passed` нельзя выставлять автоматически.

### F. Chroma exam, специально из-за найденного бага
Обязательный тест:
- взять ч/б/сепийные архивные фото;
- измерить локальную chroma исходника и результата в Lab;
- считать новым chroma-island область, где исходная chroma низкая, а candidate создаёт статистически/визуально значимое локальное отклонение;
- отдельно считать зелёные/пурпурные выбросы;
- candidate не может пройти release gate, если таких артефактов больше baseline.

### G. Training Coach integration
Расширить `tools/almaz_training_coach.py`:
- `prepare -> audit -> build -> train -> validate -> exam -> gate -> export ONNX -> parity -> install-candidate`;
- каждый этап пишет progress + elapsed + ETA при наличии устойчивой оценки;
- state file позволяет продолжить после перезапуска;
- не запускать тяжёлый train, если dataset gate не пройден;
- не заменять рабочие ONNX автоматически: candidate устанавливается только после PASS gate + ручного exam flag.

## Железо
Определять автоматически. Если доступна CUDA GPU, использовать её. Для 12 GB VRAM подобрать crop/batch/gradient accumulation без OOM. CPU разрешён для подготовки данных/оценки небольших метрик, но не запускать полноценный release fine-tuning на CPU молча. Если GPU нет, закончить acquisition/build/audit и вывести точную команду продолжения для CUDA-машины.

## Отчётность во время работы
Не молчать долгими периодами. В лог писать, например:
`[ALMAZ TRAIN] SIDD: распаковываю 18432/30000 пар | 61% | 00:18:43 | ETA 00:11:52`
`[ALMAZ TRAIN] denoise epoch 7/40 step 1840/5200 | loss ... | val ... | GPU ... | elapsed ...`
`[ALMAZ EXAM] near-monochrome 73/100 | chroma regressions=0 | ...`

## Definition of Done
Задача НЕ завершена фразой «trainer написан».
Готово только когда:
1. dataset gate PASS на нормальном, не smoke объёме;
2. хотя бы один полный fine-tune каждой выбранной задачи реально выполнен либо документирован конкретный внешний ресурсный блокер;
3. все checkpoints сравнены с baseline на закрытом exam;
4. ни один провал safety не замаскирован средним PSNR;
5. лучший допустимый checkpoint экспортирован в ONNX;
6. PyTorch<->ONNX parity PASS;
7. candidate-модели лежат отдельно от рабочих, пока ручной exam не подтверждён;
8. создан `ALMAZ_TRAINING_REPORT.md` с данными, лицензиями, железом, временем, метриками, rejected checkpoints и причиной выбора финального кандидата;
9. существующие Photo Doctor tests + новые training tests PASS.

## Важное поведение при нехватке ресурсов
Если среда Codex не предоставляет CUDA/достаточный диск/длительный runtime, НЕ уменьшать датасет до игрушечного ради галочки. Выполнить acquisition/build/audit, подготовить resume-safe training state и дать точный handoff/команду для CUDA-машины. Маленький прогон допустим только с маркировкой `SMOKE_ONLY`, он не может быть выбран как release candidate.
