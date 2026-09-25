from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

from photodoctor.ai.almaz_release_gate import TRAINING_CONTRACT_SHA256
from photodoctor.ai.restoration_catalog import preferred_restoration_candidate


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_almaz_release_manifest.py"


def _exam(chroma: float = 0.0) -> dict:
    return {
        "metrics": {
            "psnr_delta_db": 0.1,
            "ssim_delta": 0.001,
            "face_identity_drift_delta": -0.01,
            "near_monochrome_chroma_artifact_delta": chroma,
            "clipping_delta_pp": -0.01,
            "false_edge_ratio_delta": -0.01,
        },
        "task_quality_delta": {"denoise": 0.01},
        "manual_exam_passed": True,
    }


def test_manifest_builder_creates_bound_release_manifest(tmp_path):
    onnx = tmp_path / "model.onnx"; onnx.write_bytes(b"onnx")
    ckpt = tmp_path / "model.pth"; ckpt.write_bytes(b"checkpoint")
    exam = tmp_path / "exam.json"; exam.write_text(json.dumps(_exam()), encoding="utf-8")
    parity = tmp_path / "parity.json"; parity.write_text(json.dumps({"status":"PASS"}), encoding="utf-8")
    out = tmp_path / "release.json"
    p = subprocess.run([
        sys.executable, str(TOOL), "--task", "denoise", "--onnx", str(onnx),
        "--training-checkpoint", str(ckpt), "--exam", str(exam), "--parity", str(parity),
        "--variant-id", "almaz_denoise_gold_test", "--output", str(out),
    ], cwd=str(ROOT), capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    raw = json.loads(out.read_text(encoding="utf-8"))
    spec = preferred_restoration_candidate("denoise")
    assert raw["variant_kind"] == "finetuned"
    assert raw["model_id"] == spec.model_id
    assert raw["training_contract_sha256"] == TRAINING_CONTRACT_SHA256
    assert raw["onnx_sha256"] == hashlib.sha256(b"onnx").hexdigest()
    assert raw["training_checkpoint_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()


def test_manifest_builder_refuses_exam_with_new_chroma_artifacts(tmp_path):
    onnx = tmp_path / "model.onnx"; onnx.write_bytes(b"onnx")
    ckpt = tmp_path / "model.pth"; ckpt.write_bytes(b"checkpoint")
    exam = tmp_path / "exam.json"; exam.write_text(json.dumps(_exam(0.02)), encoding="utf-8")
    parity = tmp_path / "parity.json"; parity.write_text(json.dumps({"status":"PASS"}), encoding="utf-8")
    p = subprocess.run([
        sys.executable, str(TOOL), "--task", "denoise", "--onnx", str(onnx),
        "--training-checkpoint", str(ckpt), "--exam", str(exam), "--parity", str(parity),
    ], cwd=str(ROOT), capture_output=True, text=True)
    assert p.returncode == 2
    assert "near_monochrome_chroma_artifact_delta" in p.stdout
