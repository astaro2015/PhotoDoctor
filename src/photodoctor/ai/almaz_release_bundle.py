from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import tempfile
from typing import Any, Callable
import zipfile

from .almaz_model_promotion import promote_finetuned_model, PromotionResult
from .almaz_release_gate import sha256_file, validate_finetuned_manifest
from .restoration_catalog import preferred_restoration_candidate
from .sr_catalog import preferred_sr_candidate


BUNDLE_FORMAT = 1
_ALLOWED = {"bundle.json", "model.onnx", "manifest.json", "exam.json", "parity.json"}
_MAX_MODEL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BundleResult:
    accepted: bool
    task: str
    bundle_path: str
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
            "base_commit": spec.source_commit,
        }
    spec = preferred_restoration_candidate(key)
    return {
        "task": spec.task, "model_id": spec.model_id, "contract_id": spec.contract_id,
        "base_sha": spec.source_checkpoint_sha256, "base_size": spec.source_checkpoint_size,
        "base_commit": spec.source_commit,
    }


def create_release_bundle(
    task: str,
    onnx_path: str | Path,
    manifest_path: str | Path,
    exam_report_path: str | Path,
    parity_report_path: str | Path,
    output_path: str | Path,
) -> BundleResult:
    desc = _descriptor(task)
    onnx = Path(onnx_path); manifest_file = Path(manifest_path)
    exam = Path(exam_report_path); parity = Path(parity_report_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BundleResult(False, desc["task"], "", f"Manifest не читается: {exc}")
    if not isinstance(manifest, dict):
        return BundleResult(False, desc["task"], "", "Manifest имеет неверную структуру.")
    gate = validate_finetuned_manifest(
        manifest,
        expected_task=desc["task"], expected_model_id=desc["model_id"], expected_contract_id=desc["contract_id"],
        expected_base_sha256=desc["base_sha"], expected_base_size=desc["base_size"], expected_base_commit=desc["base_commit"],
        model_path=onnx, exam_report_path=exam, parity_report_path=parity,
    )
    if not gate.accepted:
        return BundleResult(False, desc["task"], "", "Release bundle отклонён: " + "; ".join(gate.failures))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": BUNDLE_FORMAT,
        "kind": "photodoctor_almaz_release_bundle",
        "task": desc["task"],
        "variant_id": str(manifest.get("variant_id", "")),
        "files": {
            "model.onnx": sha256_file(onnx),
            "manifest.json": sha256_file(manifest_file),
            "exam.json": sha256_file(exam),
            "parity.json": sha256_file(parity),
        },
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        zf.writestr("bundle.json", json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        zf.write(onnx, "model.onnx")
        zf.write(manifest_file, "manifest.json")
        zf.write(exam, "exam.json")
        zf.write(parity, "parity.json")
    return BundleResult(True, desc["task"], str(out), "Проверенный ALMAZ release-bundle создан.")


def _safe_extract_bundle(
    bundle: Path, root: Path, progress: Callable[[str], None] | None = None
) -> tuple[dict[str, Any], dict[str, Path]]:
    def emit(text: str) -> None:
        if progress is not None:
            progress(text)
    emit("открываю ALMAZ release-bundle и проверяю состав")
    with zipfile.ZipFile(bundle, "r") as zf:
        raw_names = zf.namelist()
        names = set(raw_names)
        if len(raw_names) != len(names):
            raise ValueError("В ALMAZ bundle запрещены дублирующиеся имена файлов")
        if names != _ALLOWED:
            raise ValueError(f"Неверный состав ALMAZ bundle: {sorted(names)}")
        for info in zf.infolist():
            if info.filename == "model.onnx":
                if info.file_size <= 0 or info.file_size > _MAX_MODEL_BYTES:
                    raise ValueError("Размер model.onnx вне допустимого диапазона")
            elif info.file_size > _MAX_JSON_BYTES:
                raise ValueError(f"JSON в bundle слишком большой: {info.filename}")
            if Path(info.filename).name != info.filename:
                raise ValueError("В bundle запрещены пути/подкаталоги")
        zf.extractall(root)
    meta = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or int(meta.get("format", -1)) != BUNDLE_FORMAT or meta.get("kind") != "photodoctor_almaz_release_bundle":
        raise ValueError("bundle.json имеет неизвестный формат")
    files = {name: root / name for name in _ALLOWED if name != "bundle.json"}
    expected = meta.get("files", {})
    if not isinstance(expected, dict):
        raise ValueError("bundle.json: отсутствует таблица SHA")
    for name, path in files.items():
        emit(f"проверяю SHA: {name}")
        want = str(expected.get(name, "")).lower()
        if len(want) != 64 or sha256_file(path) != want:
            raise ValueError(f"SHA bundle-файла не совпадает: {name}")
    return meta, files


def install_release_bundle(
    bundle_path: str | Path, *, model_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> PromotionResult:
    bundle = Path(bundle_path)
    if not bundle.is_file():
        return PromotionResult(False, "", "", None, f"ALMAZ bundle не найден: {bundle}")
    try:
        with tempfile.TemporaryDirectory(prefix="photodoctor_almaz_release_") as tmp:
            meta, files = _safe_extract_bundle(bundle, Path(tmp), progress=progress)
            task = str(meta.get("task", ""))
            if progress is not None:
                progress("release-bundle цел; запускаю release gate и установку")
            return promote_finetuned_model(
                task, files["model.onnx"], files["manifest.json"], files["exam.json"], files["parity.json"],
                model_dir=model_dir, progress=progress,
            )
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        return PromotionResult(False, "", "", None, f"ALMAZ bundle отклонён: {exc}")
