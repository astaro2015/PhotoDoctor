from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable

from .almaz_release_gate import validate_finetuned_manifest
from .restoration_catalog import preferred_restoration_candidate
from .restoration_runtime import default_model_dir
from .sr_catalog import preferred_sr_candidate


@dataclass(frozen=True, slots=True)
class PromotionResult:
    accepted: bool
    task: str
    installed_model_path: str
    backup_dir: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _descriptor(task: str) -> dict[str, Any]:
    key = str(task).strip().lower()
    if key in {"sr", "sr_x2", "x2", "super_resolution"}:
        spec = preferred_sr_candidate()
        return {
            "task": "sr_x2", "model_id": spec.model_id, "contract_id": spec.contract_id,
            "base_sha": spec.source_checkpoint_sha256, "base_size": spec.source_checkpoint_size,
            "base_commit": spec.source_commit, "filename": spec.planned_onnx_filename,
        }
    spec = preferred_restoration_candidate(key)
    return {
        "task": spec.task, "model_id": spec.model_id, "contract_id": spec.contract_id,
        "base_sha": spec.source_checkpoint_sha256, "base_size": spec.source_checkpoint_size,
        "base_commit": spec.source_commit, "filename": spec.planned_onnx_filename,
    }


def promote_finetuned_model(
    task: str,
    onnx_path: str | Path,
    manifest_path: str | Path,
    exam_report_path: str | Path,
    parity_report_path: str | Path,
    *,
    model_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PromotionResult:
    def emit(text: str) -> None:
        if progress is not None:
            progress(text)
    desc = _descriptor(task)
    emit("читаю manifest обученной модели")
    onnx = Path(onnx_path)
    manifest_file = Path(manifest_path)
    exam_file = Path(exam_report_path)
    parity_file = Path(parity_report_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return PromotionResult(False, desc["task"], "", None, f"Manifest кандидата не читается: {exc}")
    if not isinstance(manifest, dict):
        return PromotionResult(False, desc["task"], "", None, "Manifest кандидата имеет неверную структуру.")
    emit("проверяю SHA, parity и закрытый экзамен")
    gate = validate_finetuned_manifest(
        manifest,
        expected_task=desc["task"], expected_model_id=desc["model_id"],
        expected_contract_id=desc["contract_id"], expected_base_sha256=desc["base_sha"],
        expected_base_size=desc["base_size"], expected_base_commit=desc["base_commit"],
        model_path=onnx, exam_report_path=exam_file, parity_report_path=parity_file,
    )
    if not gate.accepted:
        return PromotionResult(False, desc["task"], "", None, "Кандидат ALMAZ отклонён: " + "; ".join(gate.failures))

    root = Path(model_dir).expanduser() if model_dir is not None else default_model_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / desc["filename"]
    target_manifest = target.with_suffix(target.suffix + ".json")
    target_exam = target.with_suffix(target.suffix + ".exam.json")
    target_parity = target.with_suffix(target.suffix + ".parity.json")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup: Path | None = None
    existing = [p for p in (target, target_manifest, target_exam, target_parity) if p.exists()]
    if existing:
        emit("сохраняю резервную копию текущей модели")
        backup = root / "backups" / desc["task"] / stamp
        backup.mkdir(parents=True, exist_ok=True)
        for p in existing:
            shutil.copy2(p, backup / p.name)

    runtime_manifest = dict(manifest)
    runtime_manifest["task"] = desc["task"] if desc["task"] != "sr_x2" else runtime_manifest.get("task", "sr_x2")
    runtime_manifest["model_id"] = desc["model_id"]
    runtime_manifest["contract_id"] = desc["contract_id"]
    runtime_manifest["release_gate_status"] = "PASS"
    runtime_manifest["promoted_at_utc"] = stamp

    tmp_model = target.with_suffix(target.suffix + ".tmp")
    tmp_manifest = target_manifest.with_suffix(target_manifest.suffix + ".tmp")
    tmp_exam = target_exam.with_suffix(target_exam.suffix + ".tmp")
    tmp_parity = target_parity.with_suffix(target_parity.suffix + ".tmp")
    emit("устанавливаю проверенную ONNX-модель и служебные отчёты")
    try:
        shutil.copy2(onnx, tmp_model)
        tmp_manifest.write_text(json.dumps(runtime_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(exam_file, tmp_exam)
        shutil.copy2(parity_file, tmp_parity)
        os.replace(tmp_model, target)
        os.replace(tmp_manifest, target_manifest)
        os.replace(tmp_exam, target_exam)
        os.replace(tmp_parity, target_parity)
    finally:
        for p in (tmp_model, tmp_manifest, tmp_exam, tmp_parity):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    history = root / "ALMAZ_PROMOTION_HISTORY.jsonl"
    event = {
        "at_utc": stamp, "task": desc["task"], "variant_id": str(manifest.get("variant_id", "")),
        "onnx_sha256": str(manifest.get("onnx_sha256", "")), "backup_dir": str(backup) if backup else None,
    }
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    emit("модель установлена; обновляю историю ALMAZ")
    return PromotionResult(True, desc["task"], str(target), str(backup) if backup else None, "Проверенная fine-tune модель ALMAZ установлена.")


def _validate_backup_package(desc: dict[str, Any], backup: Path) -> tuple[bool, str]:
    model = backup / desc["filename"]
    manifest_path = model.with_suffix(model.suffix + ".json")
    if not model.is_file() or not manifest_path.is_file():
        return False, "В backup нет модели или manifest."
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Backup manifest повреждён: {exc}"
    if not isinstance(manifest, dict):
        return False, "Backup manifest имеет неверную структуру."
    if str(manifest.get("model_id", "")) != desc["model_id"] or str(manifest.get("contract_id", "")) != desc["contract_id"]:
        return False, "Backup model ID/contract не совпадает."
    if str(manifest.get("variant_kind", "")).lower() == "finetuned":
        exam = model.with_suffix(model.suffix + ".exam.json")
        parity = model.with_suffix(model.suffix + ".parity.json")
        gate = validate_finetuned_manifest(
            manifest, expected_task=desc["task"], expected_model_id=desc["model_id"],
            expected_contract_id=desc["contract_id"], expected_base_sha256=desc["base_sha"],
            expected_base_size=desc["base_size"], expected_base_commit=desc["base_commit"],
            model_path=model, exam_report_path=exam, parity_report_path=parity,
        )
        return gate.accepted, ("" if gate.accepted else "; ".join(gate.failures))
    if str(manifest.get("parity_status", "")).upper() != "PASS":
        return False, "Backup parity не PASS."
    if (
        str(manifest.get("source_checkpoint_sha256", "")).lower() != str(desc["base_sha"]).lower()
        or int(manifest.get("source_checkpoint_size", -1) or -1) != int(desc["base_size"])
        or str(manifest.get("source_commit", "")) != desc["base_commit"]
    ):
        return False, "Backup относится не к закреплённому базовому checkpoint."
    expected = str(manifest.get("onnx_sha256", "")).lower()
    if len(expected) != 64:
        return False, "В backup manifest отсутствует SHA ONNX."
    from .almaz_release_gate import sha256_file
    if sha256_file(model) != expected:
        return False, "SHA ONNX в backup не совпадает."
    return True, ""


def rollback_almaz_model(
    task: str,
    *,
    backup_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PromotionResult:
    def emit(text: str) -> None:
        if progress is not None:
            progress(text)
    desc = _descriptor(task)
    root = Path(model_dir).expanduser() if model_dir is not None else default_model_dir()
    emit("ищу и проверяю резервную копию модели")
    backups_root = root / "backups" / desc["task"]
    if backup_dir is not None:
        backup = Path(backup_dir).expanduser()
    else:
        candidates = sorted((p for p in backups_root.iterdir() if p.is_dir()), reverse=True) if backups_root.is_dir() else []
        if not candidates:
            return PromotionResult(False, desc["task"], "", None, "Для этой модели нет backup для отката.")
        backup = candidates[0]
    ok, detail = _validate_backup_package(desc, backup)
    if not ok:
        return PromotionResult(False, desc["task"], "", str(backup), "Backup ALMAZ отклонён: " + detail)

    target = root / desc["filename"]
    target_manifest = target.with_suffix(target.suffix + ".json")
    target_exam = target.with_suffix(target.suffix + ".exam.json")
    target_parity = target.with_suffix(target.suffix + ".parity.json")
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    current = [p for p in (target, target_manifest, target_exam, target_parity) if p.exists()]
    if current:
        emit("сохраняю текущую модель перед откатом")
        safety_backup = root / "backups" / desc["task"] / f"pre_rollback_{stamp}"
        safety_backup.mkdir(parents=True, exist_ok=True)
        for src in current:
            shutil.copy2(src, safety_backup / src.name)

    source_model = backup / desc["filename"]
    source_manifest = source_model.with_suffix(source_model.suffix + ".json")
    source_exam = source_model.with_suffix(source_model.suffix + ".exam.json")
    source_parity = source_model.with_suffix(source_model.suffix + ".parity.json")
    pairs = [(source_model, target), (source_manifest, target_manifest)]
    if source_exam.is_file():
        pairs.append((source_exam, target_exam))
    if source_parity.is_file():
        pairs.append((source_parity, target_parity))
    for stale in (target_exam, target_parity):
        if stale.exists() and not (backup / stale.name).is_file():
            stale.unlink()
    emit("восстанавливаю проверенную модель из backup")
    for src, dst in pairs:
        tmp = dst.with_suffix(dst.suffix + ".rollback_tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)

    history = root / "ALMAZ_PROMOTION_HISTORY.jsonl"
    event = {"at_utc": stamp, "task": desc["task"], "event": "rollback", "restored_from": str(backup)}
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    emit("откат завершён")
    return PromotionResult(True, desc["task"], str(target), str(backup), "ALMAZ откатил модель на проверенный backup.")
