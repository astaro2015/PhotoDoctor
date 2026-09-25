# Photo Doctor 0.5.16 — session log

## Причина изменения
В 0.5.15 кнопка «Открыть» оставалась QAction внутри QToolBar, тогда как «Сохранить» и «Предпросмотр» были QPushButton. На Windows QAction отображался как icon-only tool button, поэтому подпись «Открыть» пропадала и внешний вид отличался от соседней кнопки.

## Что изменено
- «Открыть» переведена на QPushButton, как «Сохранить» и «Предпросмотр».
- Сохранены системная иконка открытия и короткая подпись «Открыть».
- «Открыть» и «Сохранить» остаются рядом слева.
- «Предпросмотр» остаётся справа после expanding spacer.
- Все состояния busy/analysis/ALMAZ обновлены на новый open_file_btn.

## Не менялось
- analysis/correction pipeline;
- ALMAZ inference и модели;
- Validator и safety gates;
- ALGORITHM_VERSION = 0.5.10-surface-v9-maximum, кэш анализа не инвалидируется.

## Проверки рабочего дерева
- toolbar/preview/packaging regression: 128/128 PASS;
- полный suite независимыми пакетами: 722/722 PASS;
- compileall src + tests: PASS.
