from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TRAINING_CONTRACT_SHA256 = "d72d3a55388c6fdfa3d5217ffa752ea8a14792c3775bb6dc7342c81d86ec39b5"


@dataclass(frozen=True, slots=True)
class ReleaseGateResult:
    accepted: bool
    failures: tuple[str, ...]


_ACCEPTANCE_LIMITS = {
    "min_psnr_delta_db": -0.05,
    "min_ssim_delta": -0.001,
    "max_face_identity_drift_delta": 0.0,
    "max_near_monochrome_chroma_artifact_delta": 0.0,
    "max_clipping_delta_pp": 0.0,
    "max_false_edge_ratio_delta": 0.0,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_exam_report(report: Mapping[str, Any]) -> ReleaseGateResult:
    failures: list[str] = []
    metrics = dict(report.get("metrics", {})) if isinstance(report.get("metrics", {}), Mapping) else {}
    comparisons = {
        "psnr_delta_db": (">=", _ACCEPTANCE_LIMITS["min_psnr_delta_db"]),
        "ssim_delta": (">=", _ACCEPTANCE_LIMITS["min_ssim_delta"]),
        "face_identity_drift_delta": ("<=", _ACCEPTANCE_LIMITS["max_face_identity_drift_delta"]),
        "near_monochrome_chroma_artifact_delta": ("<=", _ACCEPTANCE_LIMITS["max_near_monochrome_chroma_artifact_delta"]),
        "clipping_delta_pp": ("<=", _ACCEPTANCE_LIMITS["max_clipping_delta_pp"]),
        "false_edge_ratio_delta": ("<=", _ACCEPTANCE_LIMITS["max_false_edge_ratio_delta"]),
    }
    for key, (op, limit) in comparisons.items():
        if key not in metrics:
            failures.append(f"missing metric: {key}")
            continue
        try:
            got = float(metrics[key])
        except (TypeError, ValueError, OverflowError):
            failures.append(f"invalid metric: {key}")
            continue
        ok = got >= limit if op == ">=" else got <= limit
        if not ok:
            failures.append(f"{key}: {got:g} {op} {limit:g} required")

    task_deltas = report.get("task_quality_delta", {})
    if isinstance(task_deltas, Mapping):
        for task, delta in task_deltas.items():
            try:
                value = float(delta)
            except (TypeError, ValueError, OverflowError):
                failures.append(f"invalid task delta: {task}")
                continue
            if value < -0.000001:
                failures.append(f"task regression: {task} delta={value:g}")
    else:
        failures.append("task_quality_delta missing or invalid")

    if not bool(report.get("manual_exam_passed", False)):
        failures.append("manual exam not passed")
    return ReleaseGateResult(not failures, tuple(failures))


def load_exam_report(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ALMAZ exam report must be a JSON object")
    return raw


def validate_finetuned_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_task: str,
    expected_model_id: str,
    expected_contract_id: str,
    expected_base_sha256: str,
    expected_base_size: int,
    expected_base_commit: str,
    model_path: str | Path,
    exam_report_path: str | Path,
    parity_report_path: str | Path,
) -> ReleaseGateResult:
    failures: list[str] = []
    if str(manifest.get("variant_kind", "")) != "finetuned":
        failures.append("variant_kind must be finetuned")
    if str(manifest.get("task", "")) != expected_task:
        failures.append("task mismatch")
    if str(manifest.get("model_id", "")) != expected_model_id:
        failures.append("model_id mismatch")
    if str(manifest.get("contract_id", "")) != expected_contract_id:
        failures.append("contract_id mismatch")
    if str(manifest.get("base_source_checkpoint_sha256", "")).lower() != expected_base_sha256.lower():
        failures.append("base checkpoint SHA mismatch")
    try:
        base_size = int(manifest.get("base_source_checkpoint_size", -1))
    except (TypeError, ValueError, OverflowError):
        base_size = -1
    if base_size != int(expected_base_size):
        failures.append("base checkpoint size mismatch")
    if str(manifest.get("base_source_commit", "")) != expected_base_commit:
        failures.append("base source commit mismatch")
    if str(manifest.get("training_contract_sha256", "")).lower() != TRAINING_CONTRACT_SHA256:
        failures.append("training contract SHA mismatch")
    if str(manifest.get("parity_status", "")).upper() != "PASS":
        failures.append("PyTorch/ONNX parity is not PASS")
    checkpoint_sha = str(manifest.get("training_checkpoint_sha256", "")).lower()
    if len(checkpoint_sha) != 64:
        failures.append("training checkpoint SHA missing")

    model = Path(model_path)
    exam = Path(exam_report_path)
    parity = Path(parity_report_path)
    if not model.is_file():
        failures.append("ONNX artifact missing")
    else:
        actual = sha256_file(model)
        expected = str(manifest.get("onnx_sha256", "")).lower()
        if len(expected) != 64 or actual != expected:
            failures.append("ONNX SHA mismatch")
    if not parity.is_file():
        failures.append("parity report missing")
    else:
        actual_parity = sha256_file(parity)
        expected_parity = str(manifest.get("parity_report_sha256", "")).lower()
        if len(expected_parity) != 64 or actual_parity != expected_parity:
            failures.append("parity report SHA mismatch")
        else:
            try:
                parity_raw = json.loads(parity.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"parity report invalid: {exc}")
            else:
                if not isinstance(parity_raw, dict) or str(parity_raw.get("status", parity_raw.get("parity_status", ""))).upper() != "PASS":
                    failures.append("parity report is not PASS")

    if not exam.is_file():
        failures.append("exam report missing")
    else:
        actual_exam = sha256_file(exam)
        expected_exam = str(manifest.get("exam_report_sha256", "")).lower()
        if len(expected_exam) != 64 or actual_exam != expected_exam:
            failures.append("exam report SHA mismatch")
        else:
            try:
                gate = evaluate_exam_report(load_exam_report(exam))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"exam report invalid: {exc}")
            else:
                failures.extend(gate.failures)
    return ReleaseGateResult(not failures, tuple(failures))
