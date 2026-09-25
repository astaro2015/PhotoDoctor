from __future__ import annotations

import hashlib
import json
from pathlib import Path

from photodoctor.ai import sr_runtime
from photodoctor.ai.sr_backend import SRProviderStatus
from photodoctor.ai.sr_catalog import preferred_sr_candidate


def _write_model_and_manifest(root: Path, payload: bytes, *, sha_override: str | None = None) -> tuple[Path, Path]:
    spec = preferred_sr_candidate()
    model = root / spec.planned_onnx_filename
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(payload)
    sha = sha_override or hashlib.sha256(payload).hexdigest()
    manifest = model.with_suffix(model.suffix + ".json")
    manifest.write_text(json.dumps({
        "model_id": spec.model_id,
        "contract_id": spec.contract_id,
        "onnx_sha256": sha,
        "parity_status": "PASS",
        "source_checkpoint_sha256": spec.source_checkpoint_sha256,
        "source_checkpoint_size": spec.source_checkpoint_size,
        "source_commit": spec.source_commit,
    }), encoding="utf-8")
    return model, manifest


def test_sr_runtime_missing_is_safe_fallback(tmp_path):
    status = sr_runtime.inspect_installed_sr_model(tmp_path)
    assert status.ready is False
    assert status.status == "missing"


def test_sr_runtime_blocks_hash_mismatch(tmp_path):
    _write_model_and_manifest(tmp_path, b"model-bytes", sha_override="0" * 64)
    status = sr_runtime.inspect_installed_sr_model(tmp_path)
    assert status.ready is False
    assert status.status == "hash_mismatch"


def test_sr_runtime_ready_only_with_hash_and_provider(tmp_path, monkeypatch):
    _write_model_and_manifest(tmp_path, b"model-bytes")
    monkeypatch.setattr(
        sr_runtime,
        "choose_sr_provider",
        lambda _compat: SRProviderStatus("CPUExecutionProvider", "ONNX Runtime CPU", "cpu", True, 30, ""),
    )
    status = sr_runtime.inspect_installed_sr_model(tmp_path)
    assert status.ready is True
    assert status.status == "ready"
    assert status.provider == "CPUExecutionProvider"
    assert status.actual_sha256 == status.expected_sha256


def test_sr_runtime_blocks_unverified_parity(tmp_path):
    model, manifest = _write_model_and_manifest(tmp_path, b"model-bytes")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["parity_status"] = "PENDING"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    status = sr_runtime.inspect_installed_sr_model(tmp_path)
    assert status.ready is False
    assert status.status == "parity_pending"


def test_sr_runtime_caches_verified_hash_by_file_state(tmp_path, monkeypatch):
    _write_model_and_manifest(tmp_path, b"model-bytes")
    monkeypatch.setattr(
        sr_runtime,
        "choose_sr_provider",
        lambda _compat: SRProviderStatus("CPUExecutionProvider", "ONNX Runtime CPU", "cpu", True, 30, ""),
    )
    sr_runtime._STATUS_CACHE.clear()
    calls = {"n": 0}
    original = sr_runtime._sha256
    def counted(path):
        calls["n"] += 1
        return original(path)
    monkeypatch.setattr(sr_runtime, "_sha256", counted)
    first = sr_runtime.inspect_installed_sr_model(tmp_path)
    second = sr_runtime.inspect_installed_sr_model(tmp_path)
    assert first.ready and second.ready
    assert calls["n"] == 1


def test_sr_runtime_blocks_wrong_source_checkpoint(tmp_path):
    _model, manifest = _write_model_and_manifest(tmp_path, b"model-bytes")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_checkpoint_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    status = sr_runtime.inspect_installed_sr_model(tmp_path)
    assert status.ready is False
    assert status.status == "source_mismatch"
