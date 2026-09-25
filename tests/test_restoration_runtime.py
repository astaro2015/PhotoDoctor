from __future__ import annotations

import hashlib
import json
from pathlib import Path

from photodoctor.ai import restoration_runtime
from photodoctor.ai.sr_backend import SRProviderStatus
from photodoctor.ai.restoration_catalog import preferred_restoration_candidate


def _write_model_and_manifest(root: Path, task: str, payload: bytes, *, sha_override: str | None = None) -> tuple[Path, Path]:
    spec = preferred_restoration_candidate(task)
    model = root / spec.planned_onnx_filename
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(payload)
    sha = sha_override or hashlib.sha256(payload).hexdigest()
    manifest = model.with_suffix(model.suffix + ".json")
    manifest.write_text(json.dumps({
        "task": spec.task,
        "model_id": spec.model_id,
        "contract_id": spec.contract_id,
        "onnx_sha256": sha,
        "parity_status": "PASS",
        "source_checkpoint_sha256": spec.source_checkpoint_sha256,
        "source_checkpoint_size": spec.source_checkpoint_size,
        "source_commit": spec.source_commit,
    }), encoding="utf-8")
    return model, manifest


def test_restoration_runtime_missing_keeps_legacy_path(tmp_path):
    status = restoration_runtime.inspect_installed_restoration_model("denoise", tmp_path)
    assert status.ready is False
    assert status.status == "missing"


def test_restoration_runtime_blocks_hash_mismatch(tmp_path):
    _write_model_and_manifest(tmp_path, "deblur", b"model-bytes", sha_override="0" * 64)
    status = restoration_runtime.inspect_installed_restoration_model("deblur", tmp_path)
    assert status.ready is False
    assert status.status == "hash_mismatch"


def test_restoration_runtime_ready_only_with_hash_and_provider(tmp_path, monkeypatch):
    _write_model_and_manifest(tmp_path, "jpeg_recovery", b"model-bytes")
    monkeypatch.setattr(
        restoration_runtime,
        "choose_sr_provider",
        lambda _compat: SRProviderStatus("CPUExecutionProvider", "ONNX Runtime CPU", "cpu", True, 30, ""),
    )
    status = restoration_runtime.inspect_installed_restoration_model("jpeg_recovery", tmp_path)
    assert status.ready is True
    assert status.status == "ready"
    assert status.provider == "CPUExecutionProvider"
    assert status.actual_sha256 == status.expected_sha256


def test_restoration_runtime_blocks_unverified_parity(tmp_path):
    model, manifest = _write_model_and_manifest(tmp_path, "denoise", b"model-bytes")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["parity_status"] = "PENDING"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    status = restoration_runtime.inspect_installed_restoration_model("denoise", tmp_path)
    assert status.ready is False
    assert status.status == "parity_pending"


def test_restoration_runtime_caches_verified_hash_by_file_state(tmp_path, monkeypatch):
    _write_model_and_manifest(tmp_path, "denoise", b"model-bytes")
    monkeypatch.setattr(
        restoration_runtime,
        "choose_sr_provider",
        lambda _compat: SRProviderStatus("CPUExecutionProvider", "ONNX Runtime CPU", "cpu", True, 30, ""),
    )
    restoration_runtime._STATUS_CACHE.clear()
    calls = {"n": 0}
    original = restoration_runtime._sha256
    def counted(path):
        calls["n"] += 1
        return original(path)
    monkeypatch.setattr(restoration_runtime, "_sha256", counted)
    first = restoration_runtime.inspect_installed_restoration_model("denoise", tmp_path)
    second = restoration_runtime.inspect_installed_restoration_model("denoise", tmp_path)
    assert first.ready and second.ready
    assert calls["n"] == 1


def test_restoration_runtime_blocks_wrong_source_checkpoint(tmp_path):
    _model, manifest = _write_model_and_manifest(tmp_path, "denoise", b"model-bytes")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_commit"] = "deadbeef"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    status = restoration_runtime.inspect_installed_restoration_model("denoise", tmp_path)
    assert status.ready is False
    assert status.status == "source_mismatch"
