from __future__ import annotations

import hashlib
import json
from pathlib import Path

from photodoctor.ai.almaz_model_promotion import promote_finetuned_model, rollback_almaz_model
from photodoctor.ai.almaz_release_gate import TRAINING_CONTRACT_SHA256
from photodoctor.ai import restoration_runtime, sr_runtime
from photodoctor.ai.restoration_catalog import preferred_restoration_candidate
from photodoctor.ai.sr_backend import SRProviderStatus
from photodoctor.ai.sr_catalog import preferred_sr_candidate


def _exam_report(*, chroma_delta: float = 0.0, manual: bool = True) -> dict:
    return {
        "metrics": {
            "psnr_delta_db": 0.12,
            "ssim_delta": 0.002,
            "face_identity_drift_delta": -0.01,
            "near_monochrome_chroma_artifact_delta": chroma_delta,
            "clipping_delta_pp": -0.02,
            "false_edge_ratio_delta": -0.01,
        },
        "task_quality_delta": {
            "denoise": 0.01,
            "deblur": 0.01,
            "jpeg_recovery": 0.01,
            "sr_x2": 0.01,
            "archive_restore": 0.01,
        },
        "manual_exam_passed": manual,
    }


def _candidate(root: Path, task: str, *, chroma_delta: float = 0.0) -> tuple[Path, Path, Path, Path]:
    if task == "sr_x2":
        spec = preferred_sr_candidate()
        canonical_task = "sr_x2"
    else:
        spec = preferred_restoration_candidate(task)
        canonical_task = spec.task
    root.mkdir(parents=True, exist_ok=True)
    onnx = root / f"candidate_{canonical_task}.onnx"
    onnx.write_bytes((f"trained-{canonical_task}-onnx").encode("ascii"))
    exam = root / f"candidate_{canonical_task}.exam.json"
    exam.write_text(json.dumps(_exam_report(chroma_delta=chroma_delta)), encoding="utf-8")
    parity = root / f"candidate_{canonical_task}.parity.json"
    parity.write_text(json.dumps({"status": "PASS", "max_abs_error": 1e-5}), encoding="utf-8")
    manifest = root / f"candidate_{canonical_task}.manifest.json"
    payload = {
        "variant_kind": "finetuned",
        "variant_id": f"almaz_{canonical_task}_ft_test",
        "task": canonical_task,
        "model_id": spec.model_id,
        "contract_id": spec.contract_id,
        "base_source_checkpoint_sha256": spec.source_checkpoint_sha256,
        "base_source_checkpoint_size": spec.source_checkpoint_size,
        "base_source_commit": spec.source_commit,
        "training_checkpoint_sha256": hashlib.sha256(b"training-checkpoint").hexdigest(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "parity_status": "PASS",
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
        "exam_report_sha256": hashlib.sha256(exam.read_bytes()).hexdigest(),
        "parity_report_sha256": hashlib.sha256(parity.read_bytes()).hexdigest(),
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return onnx, manifest, exam, parity


def _cpu_provider(_compat):
    return SRProviderStatus("CPUExecutionProvider", "ONNX Runtime CPU", "cpu", True, 30, "")


def test_good_finetuned_denoise_is_promoted_and_runtime_accepts_it(tmp_path, monkeypatch):
    onnx, manifest, exam, parity = _candidate(tmp_path / "candidate", "denoise")
    model_dir = tmp_path / "models"
    result = promote_finetuned_model("denoise", onnx, manifest, exam, parity, model_dir=model_dir)
    assert result.accepted is True
    assert Path(result.installed_model_path).is_file()
    monkeypatch.setattr(restoration_runtime, "choose_sr_provider", _cpu_provider)
    restoration_runtime._STATUS_CACHE.clear()
    status = restoration_runtime.inspect_installed_restoration_model("denoise", model_dir)
    assert status.ready is True
    assert status.status == "ready"
    assert "Fine-tune" in status.detail


def test_good_finetuned_sr_is_promoted_and_runtime_accepts_it(tmp_path, monkeypatch):
    onnx, manifest, exam, parity = _candidate(tmp_path / "candidate", "sr_x2")
    model_dir = tmp_path / "models"
    result = promote_finetuned_model("sr_x2", onnx, manifest, exam, parity, model_dir=model_dir)
    assert result.accepted is True
    monkeypatch.setattr(sr_runtime, "choose_sr_provider", _cpu_provider)
    sr_runtime._STATUS_CACHE.clear()
    status = sr_runtime.inspect_installed_sr_model(model_dir)
    assert status.ready is True
    assert status.status == "ready"
    assert "Fine-tune" in status.detail


def test_candidate_with_better_psnr_but_new_chroma_artifacts_is_rejected(tmp_path):
    onnx, manifest, exam, parity = _candidate(tmp_path / "candidate", "denoise", chroma_delta=0.01)
    model_dir = tmp_path / "models"
    result = promote_finetuned_model("denoise", onnx, manifest, exam, parity, model_dir=model_dir)
    assert result.accepted is False
    assert "near_monochrome_chroma_artifact_delta" in result.detail
    assert not model_dir.exists() or not any(model_dir.glob("*.onnx"))


def test_candidate_with_tampered_onnx_is_rejected(tmp_path):
    onnx, manifest, exam, parity = _candidate(tmp_path / "candidate", "deblur")
    onnx.write_bytes(onnx.read_bytes() + b"tampered")
    result = promote_finetuned_model("deblur", onnx, manifest, exam, parity, model_dir=tmp_path / "models")
    assert result.accepted is False
    assert "ONNX SHA mismatch" in result.detail


def test_promotion_backs_up_previous_active_model(tmp_path):
    model_dir = tmp_path / "models"
    one = _candidate(tmp_path / "candidate1", "jpeg_recovery")
    first = promote_finetuned_model("jpeg_recovery", *one, model_dir=model_dir)
    assert first.accepted is True
    two = _candidate(tmp_path / "candidate2", "jpeg_recovery")
    # Give the second candidate distinct bytes/ID and refresh its hashes.
    onnx, manifest, exam, parity = two
    onnx.write_bytes(onnx.read_bytes() + b"-v2")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["variant_id"] = "almaz_jpeg_recovery_ft_test_v2"
    payload["onnx_sha256"] = hashlib.sha256(onnx.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    second = promote_finetuned_model("jpeg_recovery", onnx, manifest, exam, parity, model_dir=model_dir)
    assert second.accepted is True
    assert second.backup_dir is not None
    backup = Path(second.backup_dir)
    assert backup.is_dir()
    assert any(backup.glob("*.onnx"))


def test_rollback_restores_previous_verified_model(tmp_path):
    model_dir = tmp_path / "models"
    one = _candidate(tmp_path / "candidate1", "denoise")
    first = promote_finetuned_model("denoise", *one, model_dir=model_dir)
    assert first.accepted
    first_bytes = Path(first.installed_model_path).read_bytes()

    onnx, manifest, exam, parity = _candidate(tmp_path / "candidate2", "denoise")
    onnx.write_bytes(onnx.read_bytes() + b"-v2")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["variant_id"] = "almaz_denoise_ft_test_v2"
    payload["onnx_sha256"] = hashlib.sha256(onnx.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    second = promote_finetuned_model("denoise", onnx, manifest, exam, parity, model_dir=model_dir)
    assert second.accepted and second.backup_dir
    assert Path(second.installed_model_path).read_bytes() != first_bytes

    rolled = rollback_almaz_model("denoise", backup_dir=second.backup_dir, model_dir=model_dir)
    assert rolled.accepted is True
    assert Path(rolled.installed_model_path).read_bytes() == first_bytes
    pre = list((model_dir / "backups" / "denoise").glob("pre_rollback_*"))
    assert pre


def test_rollback_rejects_tampered_backup(tmp_path):
    model_dir = tmp_path / "models"
    one = _candidate(tmp_path / "candidate1", "deblur")
    first = promote_finetuned_model("deblur", *one, model_dir=model_dir)
    assert first.accepted
    two = _candidate(tmp_path / "candidate2", "deblur")
    onnx, manifest, exam, parity = two
    onnx.write_bytes(onnx.read_bytes() + b"-v2")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["variant_id"] = "almaz_deblur_ft_test_v2"
    payload["onnx_sha256"] = hashlib.sha256(onnx.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    second = promote_finetuned_model("deblur", onnx, manifest, exam, parity, model_dir=model_dir)
    assert second.accepted and second.backup_dir
    backup_model = next(Path(second.backup_dir).glob("*.onnx"))
    backup_model.write_bytes(backup_model.read_bytes() + b"tampered")
    rolled = rollback_almaz_model("deblur", backup_dir=second.backup_dir, model_dir=model_dir)
    assert rolled.accepted is False
    assert "ONNX SHA mismatch" in rolled.detail
