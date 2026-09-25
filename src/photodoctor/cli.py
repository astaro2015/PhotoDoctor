from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from photodoctor.core.batch import analyze_batch, collect_images
from photodoctor.core.database import AnalysisDatabase
from photodoctor.core.service import analyze_file
from photodoctor.gui.localization import localize_metric_name, localize_payload, localize_value
from photodoctor.utils.logging_setup import configure_logging


_COMPAT_ARGS = {
    "analyze": "анализ",
    "batch": "папка",
    "--no-recursive": "--без-подпапок",
    "--db": "--база",
    "--precision": "--точность",
    "fast": "быстро",
    "normal": "нормально",
    "precise": "точно",
    "ideal": "максимально",
    "maximum": "максимально",
}
_PRECISION_KEYS = {"быстро": "fast", "нормально": "normal", "точно": "precise", "максимально": "maximum", "идеально": "maximum"}


class _RussianArgumentParser(argparse.ArgumentParser):
    """ArgumentParser с русскими стандартными заголовками и ошибками."""

    def __init__(self, *args, **kwargs):
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self._positionals.title = "позиционные аргументы"
        self._optionals.title = "параметры"
        self.add_argument("-h", "--help", action="help", help="показать эту справку и выйти")

    @staticmethod
    def _ru_message(message: str) -> str:
        replacements = (
            ("the following arguments are required:", "обязательные аргументы:"),
            ("unrecognized arguments:", "неизвестные аргументы:"),
            ("invalid choice:", "недопустимое значение:"),
            ("choose from", "допустимые значения:"),
            ("expected one argument", "ожидается одно значение"),
            ("expected at least one argument", "ожидается хотя бы одно значение"),
        )
        text = message
        for old, new in replacements:
            text = text.replace(old, new)
        if text.startswith("argument "):
            text = "аргумент " + text[len("argument "):]
        return text

    def format_help(self) -> str:
        return super().format_help().replace("usage:", "использование:", 1)

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "использование:", 1)

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: ошибка: {self._ru_message(message)}\n")


def _normalize_compat_argv(argv: list[str] | None) -> list[str]:
    source = list(sys.argv[1:]) if argv is None else list(argv)
    return [_COMPAT_ARGS.get(arg, arg) for arg in source]


def main(argv: list[str] | None = None) -> int:
    argv = _normalize_compat_argv(argv)
    parser = _RussianArgumentParser(
        prog="Photo Doctor",
        description="Локальный анализ качества и безопасное улучшение фотографий.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, title="команды")

    p_analyze = sub.add_parser("анализ", help="Проанализировать один поддерживаемый файл")
    p_analyze.add_argument("path", metavar="ФАЙЛ", help="Путь к фотографии")
    p_analyze.add_argument("--json", action="store_true", dest="as_json", help="Вывести машинные данные JSON")
    p_analyze.add_argument(
        "--точность", choices=("быстро", "нормально", "точно", "максимально", "идеально"), default="нормально",
        help="Глубина анализа",
    )

    p_batch = sub.add_parser("папка", help="Проанализировать папку с фотографиями")
    p_batch.add_argument("path", metavar="ПАПКА", help="Путь к папке")
    p_batch.add_argument("--без-подпапок", action="store_true", dest="no_recursive", help="Не обходить подпапки")
    p_batch.add_argument(
        "--база", dest="db", default=str(Path.home() / ".photodoctor" / "analysis.sqlite3"),
        help="Путь к локальной базе результатов",
    )
    p_batch.add_argument(
        "--точность", choices=("быстро", "нормально", "точно", "максимально", "идеально"), default="нормально",
        help="Глубина анализа",
    )

    args = parser.parse_args(argv)
    configure_logging()
    precision = _PRECISION_KEYS[args.точность]

    if args.cmd == "анализ":
        result = analyze_file(args.path, precision=precision)
        if args.as_json:
            # JSON is intentionally a machine-readable contract; display-only
            # localization is applied to values, not to persisted/internal keys.
            print(json.dumps(localize_payload(result.to_dict()), ensure_ascii=False, indent=2))
        else:
            print(f"{result.image.path} | {result.image.width}×{result.image.height} | {result.image.format}")
            for metric in result.metrics.values():
                score = "—" if metric.normalized_value is None else f"{metric.normalized_value:.1f}"
                raw = localize_payload(metric.raw_value) if isinstance(metric.raw_value, (dict, list)) else localize_value(metric.raw_value)
                print(
                    f"{localize_metric_name(metric.name):28} "
                    f"оценка={score:>6} уверенность={metric.confidence:.2f} исходное={raw}"
                )
        return 0

    files = collect_images(args.path, recursive=not args.no_recursive)
    with AnalysisDatabase(args.db) as db:
        summary = analyze_batch(files, db, precision=precision)
    print(f"Всего: {summary.total}")
    print(f"Обработано: {summary.processed}")
    print(f"Успешно: {summary.succeeded}")
    print(f"Ошибок: {summary.failed}")
    print(f"Из кэша: {summary.cached}")
    print(f"Отменено: {'да' if summary.cancelled else 'нет'}")
    return 0 if summary.failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
