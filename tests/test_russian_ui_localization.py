from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Ordinary English UI terminology that has Russian equivalents. Technical
# standards/acronyms (JPEG/EXIF/ONNX/SHA-256/RGB/NSS/ISO) are intentionally not
# forbidden because these are canonical format/technology names.
FORBIDDEN = {
    "Validator",
    "Decision Engine",
    "Parameter Recommender",
    "Surface Refiner",
    "Preview",
    "ground truth",
    "benchmark accuracy",
    "Model ID",
    "Raw",
    "Score",
    "Confidence",
    "Scale",
    "Region",
    "classical candidate",
    "classical candidates",
    "advisory-only",
    "Unknown Guard",
    "OOD-кандидат",
}

FILES = [
    ROOT / "src/photodoctor/gui/main_window.py",
    ROOT / "src/photodoctor/gui/presentation.py",
    ROOT / "src/photodoctor/gui/ai_presentation.py",
    ROOT / "src/photodoctor/cli.py",
    ROOT / "src/photodoctor/__main__.py",
]


def _string_literals(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_user_facing_sources_do_not_reintroduce_old_english_ui_terms():
    offenders: list[str] = []
    for path in FILES:
        for text in _string_literals(path):
            for forbidden in FORBIDDEN:
                if forbidden in text:
                    offenders.append(f"{path.name}: {forbidden!r} in {text!r}")
    assert not offenders, "\n".join(offenders)


def test_main_window_uses_russian_core_controls_and_headers():
    text = (ROOT / "src/photodoctor/gui/main_window.py").read_text(encoding="utf-8")
    required = (
        'QPushButton("По окну")',
        'QPushButton("Выбрать рекомендуемые")',
        '["Метрика", "Исходное значение", "Оценка", "Уверенность", "Масштаб", "Область"]',
        '"Контекст", "Surface AI v2", "Meta-контекст", "Итог"',
        '"Архив ZIP (*.zip)"',
    )
    for value in required:
        assert value in text


def test_cli_help_is_russian_while_legacy_commands_stay_compatible(capsys):
    from photodoctor.cli import main

    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "команды" in out
    assert "Проанализировать один поддерживаемый файл" in out
    assert "Проанализировать папку" in out
    assert "Analyze one" not in out
    assert "Analyze a folder" not in out
    assert "options:" not in out
    assert "show this help message" not in out
    assert "параметры:" in out
    assert "показать эту справку и выйти" in out
