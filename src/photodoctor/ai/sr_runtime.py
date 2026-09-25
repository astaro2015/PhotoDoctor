from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .sr_backend import OnnxX2SuperResolutionBackend, SRProviderStatus, choose_sr_provider
from .sr_catalog import preferred_sr_candidate
from .almaz_release_gate import validate_finetuned_manifest
from .model_fingerprint import sampled_file_digest


@dataclass(slots=True)
class SRInstalledStatus:
    ready: bool
    status: str
    model_path: str
    manifest_path: str
    model_id: str
    contract_id: str
    provider: str | None
    provider_label: str | None
    expected_sha256: str | None
    actual_sha256: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_model_dir() -> Path:
    override = os.environ.get("PHOTODOCTOR_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".photodoctor" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_installed_sr_model_uncached(model_dir: str | Path | None = None) -> SRInstalledStatus:
    spec = preferred_sr_candidate()
    root = Path(model_dir).expanduser() if model_dir is not None else _default_model_dir()
    model_path = root / spec.planned_onnx_filename
    manifest_path = model_path.with_suffix(model_path.suffix + ".json")
    base = dict(
        model_path=str(model_path),
        manifest_path=str(manifest_path),
        model_id=spec.model_id,
        contract_id=spec.contract_id,
        provider=None,
        provider_label=None,
        expected_sha256=None,
        actual_sha256=None,
    )
    if not model_path.is_file():
        return SRInstalledStatus(False, "missing", detail="ALMAZ ONNX-модель не установлена; используется безопасный fallback.", **base)
    if not manifest_path.is_file():
        return SRInstalledStatus(False, "manifest_missing", detail="ONNX найден, но отсутствует manifest с SHA/контрактом; запуск заблокирован.", **base)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return SRInstalledStatus(False, "manifest_invalid", detail=f"Manifest ALMAZ повреждён: {exc}", **base)
    if not isinstance(manifest, dict):
        return SRInstalledStatus(False, "manifest_invalid", detail="Manifest ALMAZ имеет неверную структуру.", **base)
    if str(manifest.get("model_id", "")) != spec.model_id or str(manifest.get("contract_id", "")) != spec.contract_id:
        return SRInstalledStatus(False, "contract_mismatch", detail="Model ID/contract не совпадает с каталогом Photo Doctor.", **base)
    if str(manifest.get("parity_status", "")).strip().upper() != "PASS":
        return SRInstalledStatus(False, "parity_pending", detail="ONNX найден, но PyTorch↔ONNX parity ещё не имеет статуса PASS; запуск заблокирован.", **base)
    if str(manifest.get("variant_kind", "")).strip().lower() == "finetuned":
        exam_path = model_path.with_suffix(model_path.suffix + ".exam.json")
        release_gate = validate_finetuned_manifest(
            manifest,
            expected_task="sr_x2", expected_model_id=spec.model_id, expected_contract_id=spec.contract_id,
            expected_base_sha256=spec.source_checkpoint_sha256, expected_base_size=spec.source_checkpoint_size,
            expected_base_commit=spec.source_commit, model_path=model_path, exam_report_path=exam_path, parity_report_path=model_path.with_suffix(model_path.suffix + ".parity.json"),
        )
        if not release_gate.accepted:
            return SRInstalledStatus(
                False, "release_gate_failed", detail="Fine-tune ALMAZ x2 не прошёл release gate: " + "; ".join(release_gate.failures), **base
            )
    elif (
        str(manifest.get("source_checkpoint_sha256", "")).strip().lower() != spec.source_checkpoint_sha256
        or int(manifest.get("source_checkpoint_size", -1) or -1) != spec.source_checkpoint_size
        or str(manifest.get("source_commit", "")).strip() != spec.source_commit
    ):
        return SRInstalledStatus(False, "source_mismatch", detail="Manifest относится не к закреплённому исходному SwinIR checkpoint; запуск заблокирован.", **base)
    expected = str(manifest.get("onnx_sha256", "")).strip().lower()
    if len(expected) != 64:
        return SRInstalledStatus(False, "hash_missing", detail="В manifest нет корректного SHA-256 ONNX; запуск заблокирован.", **base)
    try:
        actual = _sha256(model_path)
    except OSError as exc:
        return SRInstalledStatus(False, "unreadable", detail=f"Не удалось прочитать ONNX: {exc}", expected_sha256=expected, **{k:v for k,v in base.items() if k!='expected_sha256'})
    if actual.lower() != expected:
        return SRInstalledStatus(
            False, "hash_mismatch", detail="SHA-256 ALMAZ-модели не совпадает с manifest; запуск заблокирован.",
            expected_sha256=expected, actual_sha256=actual,
            **{k:v for k,v in base.items() if k not in {'expected_sha256','actual_sha256'}}
        )
    provider = choose_sr_provider(spec.compatible_providers)
    if provider is None:
        return SRInstalledStatus(
            False, "runtime_missing", detail="Модель целостна, но нет совместимого ONNX execution provider.",
            expected_sha256=expected, actual_sha256=actual,
            **{k:v for k,v in base.items() if k not in {'expected_sha256','actual_sha256'}}
        )
    return SRInstalledStatus(
        True, "ready", provider=provider.provider, provider_label=provider.label,
        expected_sha256=expected, actual_sha256=actual,
        detail=(f"ALMAZ ONNX готов; provider: {provider.label}." + (f" Fine-tune: {manifest.get('variant_id', 'без ID')}." if str(manifest.get("variant_kind", "")).lower() == "finetuned" else "")),
        **{k:v for k,v in base.items() if k not in {'provider','provider_label','expected_sha256','actual_sha256'}}
    )


_STATUS_CACHE: dict[tuple[str, int, int, str, int, int, str], SRInstalledStatus] = {}


def inspect_installed_sr_model(model_dir: str | Path | None = None) -> SRInstalledStatus:
    spec = preferred_sr_candidate()
    root = Path(model_dir).expanduser() if model_dir is not None else _default_model_dir()
    model_path = root / spec.planned_onnx_filename
    manifest_path = model_path.with_suffix(model_path.suffix + ".json")
    if not model_path.is_file() or not manifest_path.is_file():
        return _inspect_installed_sr_model_uncached(model_dir)
    try:
        ms = model_path.stat(); js = manifest_path.stat()
    except OSError:
        return _inspect_installed_sr_model_uncached(model_dir)
    try:
        model_sample = sampled_file_digest(model_path)
        manifest_sample = sampled_file_digest(manifest_path)
    except OSError:
        return _inspect_installed_sr_model_uncached(model_dir)
    key = (
        str(model_path.resolve()), int(ms.st_mtime_ns), int(ms.st_size), model_sample,
        int(js.st_mtime_ns), int(js.st_size), manifest_sample,
    )
    cached = _STATUS_CACHE.get(key)
    if cached is not None:
        return cached
    status = _inspect_installed_sr_model_uncached(model_dir)
    stale = [old for old in _STATUS_CACHE if old[0] == key[0] and old != key]
    for old in stale:
        _STATUS_CACHE.pop(old, None)
    _STATUS_CACHE[key] = status
    return status


_BACKEND_CACHE: dict[tuple[str, int, int, str, str], OnnxX2SuperResolutionBackend] = {}


def load_installed_sr_backend(model_dir: str | Path | None = None) -> tuple[OnnxX2SuperResolutionBackend | None, SRInstalledStatus]:
    status = inspect_installed_sr_model(model_dir)
    if not status.ready or not status.provider or not status.expected_sha256:
        return None, status
    path = Path(status.model_path)
    stat = path.stat()
    key = (
        str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size),
        sampled_file_digest(path), status.expected_sha256,
    )
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached, status
    # Drop stale versions of the same model path before constructing a new session.
    stale = [old for old in _BACKEND_CACHE if old[0] == key[0] and old != key]
    for old in stale:
        _BACKEND_CACHE.pop(old, None)
    spec = preferred_sr_candidate()
    backend = OnnxX2SuperResolutionBackend(
        path,
        compatible_providers=spec.compatible_providers,
        preferred_provider=status.provider,
        input_size=256,
    )
    _BACKEND_CACHE[key] = backend
    return backend, status
