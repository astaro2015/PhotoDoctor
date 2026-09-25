from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
import pytest

from photodoctor.ai.almaz_release_bundle import create_release_bundle, install_release_bundle
from photodoctor.ai.almaz_release_gate import TRAINING_CONTRACT_SHA256
from photodoctor.ai.restoration_catalog import preferred_restoration_candidate


def _valid_candidate(root: Path, task: str = "denoise"):
    spec = preferred_restoration_candidate(task)
    root.mkdir(parents=True, exist_ok=True)
    onnx = root / "candidate.onnx"; onnx.write_bytes(b"candidate-onnx")
    exam = root / "exam.json"
    exam.write_text(json.dumps({
        "metrics": {
            "psnr_delta_db": 0.1,
            "ssim_delta": 0.001,
            "face_identity_drift_delta": -0.01,
            "near_monochrome_chroma_artifact_delta": 0.0,
            "clipping_delta_pp": -0.01,
            "false_edge_ratio_delta": -0.01,
        },
        "task_quality_delta": {task: 0.01},
        "manual_exam_passed": True,
    }), encoding="utf-8")
    parity = root / "parity.json"; parity.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "variant_kind": "finetuned",
        "variant_id": f"almaz_{task}_bundle_test",
        "task": task,
        "model_id": spec.model_id,
        "contract_id": spec.contract_id,
        "base_source_checkpoint_sha256": spec.source_checkpoint_sha256,
        "base_source_checkpoint_size": spec.source_checkpoint_size,
        "base_source_commit": spec.source_commit,
        "training_checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "parity_status": "PASS",
        "parity_report_sha256": hashlib.sha256(parity.read_bytes()).hexdigest(),
        "exam_report_sha256": hashlib.sha256(exam.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(onnx.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    return onnx, manifest, exam, parity


def test_release_bundle_roundtrip_installs_verified_model(tmp_path):
    onnx, manifest, exam, parity = _valid_candidate(tmp_path / "candidate")
    bundle = tmp_path / "release.almaz.zip"
    built = create_release_bundle("denoise", onnx, manifest, exam, parity, bundle)
    assert built.accepted is True
    assert bundle.is_file()
    installed = install_release_bundle(bundle, model_dir=tmp_path / "models")
    assert installed.accepted is True
    assert Path(installed.installed_model_path).is_file()


def test_release_bundle_rejects_tampered_model_inside_zip(tmp_path):
    onnx, manifest, exam, parity = _valid_candidate(tmp_path / "candidate")
    good = tmp_path / "good.zip"
    assert create_release_bundle("denoise", onnx, manifest, exam, parity, good).accepted
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(bad, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "model.onnx":
                data += b"tampered"
            dst.writestr(info.filename, data)
    result = install_release_bundle(bad, model_dir=tmp_path / "models")
    assert result.accepted is False
    assert "SHA bundle-файла" in result.detail


def test_release_bundle_rejects_extra_member(tmp_path):
    onnx, manifest, exam, parity = _valid_candidate(tmp_path / "candidate")
    good = tmp_path / "good.zip"
    assert create_release_bundle("denoise", onnx, manifest, exam, parity, good).accepted
    bad = tmp_path / "bad-extra.zip"
    with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(bad, "w", compression=zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():
            dst.writestr(info.filename, src.read(info.filename))
        dst.writestr("evil.txt", "x")
    result = install_release_bundle(bad, model_dir=tmp_path / "models")
    assert result.accepted is False
    assert "Неверный состав" in result.detail


def test_release_bundle_rejects_duplicate_member(tmp_path):
    onnx, manifest, exam, parity = _valid_candidate(tmp_path / "candidate")
    good = tmp_path / "good.zip"
    assert create_release_bundle("denoise", onnx, manifest, exam, parity, good).accepted
    bad = tmp_path / "bad-dup.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(good, "r") as src, zipfile.ZipFile(bad, "w", compression=zipfile.ZIP_DEFLATED) as dst:
            for info in src.infolist():
                dst.writestr(info.filename, src.read(info.filename))
            dst.writestr("manifest.json", b"{}")
    result = install_release_bundle(bad, model_dir=tmp_path / "models")
    assert result.accepted is False
    assert "дублирующиеся" in result.detail
