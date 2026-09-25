# ALMAZ training workspace

Это не папка для пользовательских фотографий и не рабочие модели Photo Doctor.

- `TRAINING_CONTRACT.json` — минимальный объём, split policy, release gate.
- `SOURCES.json` — начальный реестр источников/лицензий.
- `dataset_stats.example.json` — формат статистики, которую должен создать Dataset Builder.
- `../../CODEX_TASK_ALMAZ_TRAINING_RU.md` — основное ТЗ для Codex.

Smoke-набор разрешён только для проверки кода. Если dataset gate не пройден, полноценный release fine-tune запускать нельзя.
