from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True, slots=True)
class PrepareDependency:
    module: str
    package: str
    label: str


COMMON_EXPORT_STACK = (
    PrepareDependency("torch", "torch", "PyTorch"),
    PrepareDependency("onnx", "onnx>=1.16,<2", "ONNX"),
    # Any ORT package exposes the same ``onnxruntime`` module.  Therefore a
    # previously selected GPU/DirectML/OpenVINO runtime is preserved and we do
    # not replace it with the CPU wheel merely because the package name differs.
    PrepareDependency("onnxruntime", "onnxruntime>=1.20,<2", "ONNX Runtime"),
)

SWINIR_EXTRA_STACK = (
    PrepareDependency("timm", "timm>=1.0,<2", "timm"),
)


def configure_prepare_process_io() -> None:
    """Make ALMAZ preparation output deterministic and safe on Windows pipes.

    Photo Doctor reads QProcess output as UTF-8.  Windows Python may otherwise
    inherit cp1251/cp866 from the parent console and either emit mojibake or
    raise UnicodeEncodeError on decorative symbols.  Exporter/pip subprocesses
    inherit these variables as well.
    """
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    if os.name == "nt":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except (OSError, ValueError):
                    pass


def missing_dependencies(requirements: Iterable[PrepareDependency]) -> list[PrepareDependency]:
    return [dep for dep in requirements if importlib.util.find_spec(dep.module) is None]


def _pip_install(package: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", package]
    print("      " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_export_stack(*, need_timm: bool, auto_install: bool = True) -> None:
    configure_prepare_process_io()
    requirements = list(COMMON_EXPORT_STACK)
    if need_timm:
        requirements.extend(SWINIR_EXTRA_STACK)

    missing = missing_dependencies(requirements)
    if not missing:
        print("[0/4] Экспортные зависимости ALMAZ уже установлены.", flush=True)
        return

    names = ", ".join(dep.label for dep in missing)
    print(f"[0/4] Не хватает экспортных зависимостей ALMAZ: {names}", flush=True)
    if not auto_install:
        packages = " ".join(dep.package for dep in missing)
        raise RuntimeError(
            "Отсутствуют зависимости подготовки модели. "
            f"Установите их этим Python: {sys.executable} -m pip install {packages}"
        )

    print("      Устанавливаю недостающие пакеты в текущее окружение...", flush=True)
    for dep in missing:
        try:
            _pip_install(dep.package)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Не удалось установить {dep.label} ({dep.package}) в {sys.executable}. "
                "Проверьте доступ к интернету и права записи в .venv."
            ) from exc
        importlib.invalidate_caches()
        if importlib.util.find_spec(dep.module) is None:
            raise RuntimeError(
                f"Пакет {dep.label} установщик завершил без ошибки, но модуль {dep.module!r} всё ещё не импортируется."
            )
        print(f"      [OK] {dep.label}", flush=True)

    print("      Экспортный стек ALMAZ готов.", flush=True)
