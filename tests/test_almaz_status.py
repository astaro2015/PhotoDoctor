from __future__ import annotations

from photodoctor.ai.almaz_status import inspect_almaz_status
from photodoctor.ai.sr_backend import SRProviderStatus


def test_almaz_status_is_safe_when_models_or_runtime_are_missing(monkeypatch):
    monkeypatch.setattr(
        "photodoctor.ai.almaz_status.inspect_installed_sr_model",
        lambda: type("S", (), {"to_dict": lambda self: {"ready": False, "status": "missing", "provider": None}})(),
    )
    monkeypatch.setattr(
        "photodoctor.ai.almaz_status.inspect_all_restoration_models",
        lambda: {
            "denoise": {"ready": False, "status": "missing"},
            "deblur": {"ready": False, "status": "missing"},
            "jpeg_recovery": {"ready": False, "status": "missing"},
        },
    )
    monkeypatch.setattr("photodoctor.ai.almaz_status.discover_sr_providers", lambda: [])
    status = inspect_almaz_status()
    assert status.ready_models == 0
    assert status.total_models == 4
    assert status.best_provider is None


def test_almaz_status_reports_best_available_provider(monkeypatch):
    monkeypatch.setattr(
        "photodoctor.ai.almaz_status.inspect_installed_sr_model",
        lambda: type("S", (), {"to_dict": lambda self: {"ready": True, "status": "ready", "provider": "CUDAExecutionProvider"}})(),
    )
    monkeypatch.setattr(
        "photodoctor.ai.almaz_status.inspect_all_restoration_models",
        lambda: {
            "denoise": {"ready": True, "status": "ready"},
            "deblur": {"ready": False, "status": "missing"},
            "jpeg_recovery": {"ready": False, "status": "missing"},
        },
    )
    fake = SRProviderStatus("CUDAExecutionProvider", "NVIDIA CUDA", "gpu", True, 90, "")
    monkeypatch.setattr("photodoctor.ai.almaz_status.discover_sr_providers", lambda: [fake])
    status = inspect_almaz_status()
    assert status.ready_models == 2
    assert status.best_provider == "CUDAExecutionProvider"
    assert status.best_provider_label == "NVIDIA CUDA"
