from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import almaz_prepare_env as env


def test_prepare_env_does_nothing_when_stack_is_present(monkeypatch):
    monkeypatch.setattr(env.importlib.util, "find_spec", lambda _name: object())
    calls: list[list[str]] = []
    monkeypatch.setattr(env.subprocess, "run", lambda cmd, check: calls.append(list(cmd)))
    env.ensure_export_stack(need_timm=True)
    assert calls == []


def test_prepare_env_installs_only_missing_modules_with_current_python(monkeypatch):
    installed: set[str] = {"torch", "onnxruntime"}

    def fake_find_spec(name: str):
        return object() if name in installed else None

    calls: list[list[str]] = []

    def fake_run(cmd, check):
        assert check is True
        calls.append(list(cmd))
        package = str(cmd[-1])
        if package.startswith("onnx"):
            installed.add("onnx")
        elif package.startswith("timm"):
            installed.add("timm")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(env.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(env.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(env.subprocess, "run", fake_run)

    env.ensure_export_stack(need_timm=True)

    assert len(calls) == 2
    assert all(call[:3] == [sys.executable, "-m", "pip"] for call in calls)
    assert any(str(call[-1]).startswith("onnx") for call in calls)
    assert any(str(call[-1]).startswith("timm") for call in calls)
    assert not any(str(call[-1]) == "torch" for call in calls)


def test_prepare_env_reports_exact_dependency_when_pip_fails(monkeypatch):
    monkeypatch.setattr(env.importlib.util, "find_spec", lambda name: None if name == "onnx" else object())

    def fail(cmd, check):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(env.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="ONNX"):
        env.ensure_export_stack(need_timm=False)


def test_prepare_env_can_be_fail_closed_without_auto_install(monkeypatch):
    monkeypatch.setattr(env.importlib.util, "find_spec", lambda name: None if name == "torch" else object())
    with pytest.raises(RuntimeError, match="pip install torch"):
        env.ensure_export_stack(need_timm=False, auto_install=False)


def test_prepare_env_console_messages_are_cp1251_safe(monkeypatch, capsys):
    installed: set[str] = {"torch", "onnxruntime"}

    def fake_find_spec(name: str):
        return object() if name in installed else None

    def fake_run(cmd, check):
        package = str(cmd[-1])
        if package.startswith("onnx"):
            installed.add("onnx")
        elif package.startswith("timm"):
            installed.add("timm")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(env.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(env.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(env.subprocess, "run", fake_run)
    env.ensure_export_stack(need_timm=True)
    output = capsys.readouterr().out
    # Regression for Windows cp1251 crash caused by a decorative check-mark.
    output.encode("cp1251")
    assert "[OK] ONNX" in output
    assert "[OK] timm" in output
    assert "✓" not in output
