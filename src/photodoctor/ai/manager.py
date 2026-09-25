from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from .manager_types import ModelSpec, ModelState


def _default_model_dir() -> Path:
    override = os.environ.get("PHOTODOCTOR_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".photodoctor" / "models"


def _sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class ModelImportError(ValueError):
    pass


class AIModelManager:
    """Inspect and manage optional local ONNX models.

    The manager never downloads a model. Manual imports are pinned by SHA-256
    in a local manifest, which protects later runs from silent file changes.
    The pin is an integrity check, not a claim about the model's provenance.
    """

    CATALOG_MANIFEST_NAME = "catalog.json"
    INSTALLED_MANIFEST_NAME = "installed_models.json"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        specs: Iterable[ModelSpec] | None = None,
        contract_checker: Callable[[ModelSpec, Path], tuple[bool, str]] | None = None,
    ):
        if specs is None:
            from .catalog import BUILTIN_MODEL_SPECS

            specs = BUILTIN_MODEL_SPECS
        self.model_dir = Path(model_dir).expanduser() if model_dir is not None else _default_model_dir()
        self.specs = tuple(specs)
        self._specs_by_id = {spec.model_id: spec for spec in self.specs}
        self._states_cache: list[ModelState] | None = None
        self._contract_checker = contract_checker

    @staticmethod
    def runtime_available() -> bool:
        return importlib.util.find_spec("onnxruntime") is not None

    @property
    def installed_manifest_path(self) -> Path:
        return self.model_dir / self.INSTALLED_MANIFEST_NAME

    def _load_installed_manifest(self) -> dict[str, dict[str, object]]:
        path = self.installed_manifest_path
        if not path.is_file():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        entries = payload.get("models", {}) if isinstance(payload, dict) else {}
        if not isinstance(entries, dict):
            return {}
        return {str(key): value for key, value in entries.items() if isinstance(value, dict)}

    def _write_installed_manifest(self, entries: dict[str, dict[str, object]]) -> Path:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        target = self.installed_manifest_path
        temp = target.with_suffix(target.suffix + ".tmp")
        payload = {"format": 1, "models": entries}
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, target)
        return target

    def _expected_integrity(self, spec: ModelSpec) -> tuple[str | None, str]:
        if spec.sha256:
            return spec.sha256.lower(), "catalog"
        record = self._load_installed_manifest().get(spec.model_id, {})
        pinned = str(record.get("sha256", "")).strip().lower()
        if len(pinned) == 64:
            return pinned, "local_pin"
        return None, "none"


    @staticmethod
    def _contract_fingerprint(spec: ModelSpec) -> str:
        payload = {
            "contract_id": spec.contract_id,
            "input_size": list(spec.input_size),
            "output_kind": spec.output_kind,
            "labels": list(spec.labels),
            "provider": spec.provider,
        }
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _check_contract(self, spec: ModelSpec, path: Path) -> tuple[bool, str]:
        if self._contract_checker is not None:
            return self._contract_checker(spec, path)
        from .executor import preflight_onnx_contract

        return preflight_onnx_contract(str(path), spec)

    def _preflight_record(self, spec: ModelSpec, actual_hash: str) -> dict[str, object]:
        record = self._load_installed_manifest().get(spec.model_id, {})
        if str(record.get("contract_checked_sha256", "")).lower() != actual_hash.lower():
            return {}
        if str(record.get("contract_fingerprint", "")).lower() != self._contract_fingerprint(spec):
            return {}
        return record

    def _store_preflight(
        self, spec: ModelSpec, actual_hash: str, ok: bool, detail: str
    ) -> None:
        entries = self._load_installed_manifest()
        record = dict(entries.get(spec.model_id, {}))
        record.setdefault("filename", spec.filename)
        record.setdefault("sha256", actual_hash)
        record["contract_checked_sha256"] = actual_hash
        record["contract_fingerprint"] = self._contract_fingerprint(spec)
        record["contract_ok"] = bool(ok)
        record["contract_detail"] = str(detail)
        record["contract_checked_at"] = datetime.now(timezone.utc).isoformat()
        entries[spec.model_id] = record
        self._write_installed_manifest(entries)

    def inspect(self, spec: ModelSpec) -> ModelState:
        if spec.provider == "native_numpy":
            return ModelState(
                spec=spec,
                path=Path("<builtin>") / spec.model_id,
                status="ready",
                runtime_available=True,
                file_present=True,
                hash_ok=True,
                actual_sha256=None,
                expected_sha256=None,
                integrity_source="builtin",
                detail="Встроенная модель Photo Doctor; ONNX Runtime не требуется.",
            )
        path = self.model_dir / spec.filename
        runtime = self.runtime_available()
        expected_hash, integrity_source = self._expected_integrity(spec)
        if not path.is_file():
            return ModelState(
                spec=spec,
                path=path,
                status="missing",
                runtime_available=runtime,
                file_present=False,
                hash_ok=None,
                expected_sha256=expected_hash,
                integrity_source=integrity_source,
                detail="Файл локальной модели не установлен.",
            )
        try:
            actual_hash = _sha256(path)
        except OSError as exc:
            return ModelState(
                spec=spec,
                path=path,
                status="unreadable",
                runtime_available=runtime,
                file_present=True,
                hash_ok=None,
                expected_sha256=expected_hash,
                integrity_source=integrity_source,
                detail=f"Не удалось прочитать файл модели: {exc}",
            )
        if expected_hash and actual_hash.lower() != expected_hash:
            return ModelState(
                spec=spec,
                path=path,
                status="hash_mismatch",
                runtime_available=runtime,
                file_present=True,
                hash_ok=False,
                actual_sha256=actual_hash,
                expected_sha256=expected_hash,
                integrity_source=integrity_source,
                detail="SHA-256 модели не совпадает с зафиксированным значением; запуск заблокирован.",
            )
        if expected_hash is None:
            return ModelState(
                spec=spec,
                path=path,
                status="unverified",
                runtime_available=runtime,
                file_present=True,
                hash_ok=None,
                actual_sha256=actual_hash,
                expected_sha256=None,
                integrity_source="none",
                detail="Файл найден, но его SHA-256 не закреплён в каталоге/локальном файле описания; запуск заблокирован.",
            )
        if not runtime:
            return ModelState(
                spec=spec,
                path=path,
                status="runtime_missing",
                runtime_available=False,
                file_present=True,
                hash_ok=True,
                actual_sha256=actual_hash,
                expected_sha256=expected_hash,
                integrity_source=integrity_source,
                detail="Модель целостна, но ONNX Runtime не установлен; контракт входов/выходов пока не проверен.",
            )

        preflight = self._preflight_record(spec, actual_hash)
        if preflight:
            contract_ok = bool(preflight.get("contract_ok", False))
            contract_detail = str(preflight.get("contract_detail", ""))
        else:
            contract_ok, contract_detail = self._check_contract(spec, path)
            self._store_preflight(spec, actual_hash, contract_ok, contract_detail)
        if not contract_ok:
            return ModelState(
                spec=spec,
                path=path,
                status="contract_mismatch",
                runtime_available=True,
                file_present=True,
                hash_ok=True,
                actual_sha256=actual_hash,
                expected_sha256=expected_hash,
                integrity_source=integrity_source,
                detail=contract_detail or "Контракт входов/выходов ONNX не соответствует ожидаемому.",
            )
        return ModelState(
            spec=spec,
            path=path,
            status="ready",
            runtime_available=True,
            file_present=True,
            hash_ok=True,
            actual_sha256=actual_hash,
            expected_sha256=expected_hash,
            integrity_source=integrity_source,
            detail=contract_detail or "Локальная модель прошла проверку SHA-256 и контракта входов/выходов.",
        )

    def states(self) -> list[ModelState]:
        if self._states_cache is None:
            self._states_cache = [self.inspect(spec) for spec in self.specs]
        return list(self._states_cache)

    def invalidate(self) -> None:
        self._states_cache = None

    def state_for_task(self, task: str) -> ModelState | None:
        candidates = [state for state in self.states() if state.spec.task == task]
        if not candidates:
            return None
        return next((state for state in candidates if state.ready), candidates[0])

    def summary(self, states: list[ModelState] | None = None) -> dict[str, object]:
        states = list(states) if states is not None else self.states()
        native = [state for state in states if state.spec.provider == "native_numpy"]
        external = [state for state in states if state.spec.provider != "native_numpy"]
        return {
            "model_dir": str(self.model_dir),
            "runtime_available": self.runtime_available(),
            "catalog_count": len(states),
            "ready_count": sum(state.ready for state in states),
            "native_count": len(native),
            "native_ready_count": sum(state.ready for state in native),
            "external_count": len(external),
            "external_ready_count": sum(state.ready for state in external),
            "external_installed_count": sum(state.file_present for state in external),
            "missing_count": sum(state.status == "missing" for state in external),
            "blocked_count": sum(state.status in {"hash_mismatch", "unreadable", "unverified", "contract_mismatch"} for state in external),
            "models": [state.to_dict() for state in states],
        }

    def write_catalog_manifest(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.model_dir / self.CATALOG_MANIFEST_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 1,
            "models": [
                {
                    "model_id": spec.model_id,
                    "task": spec.task,
                    "filename": spec.filename,
                    "version": spec.version,
                    "description": spec.description,
                    "input_size": list(spec.input_size),
                    "sha256": spec.sha256,
                    "contract_id": spec.contract_id,
                    "output_kind": spec.output_kind,
                    "labels": list(spec.labels),
                    "provider": spec.provider,
                    "bundled": spec.bundled,
                    "provenance": spec.provenance,
                }
                for spec in self.specs
            ],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    # Backward-compatible alias from 0.2.0.
    def write_manifest(self, path: str | Path | None = None) -> Path:
        return self.write_catalog_manifest(path)

    def import_model(self, model_id: str, source_path: str | Path) -> ModelState:
        spec = self._specs_by_id.get(model_id)
        if spec is None:
            raise ModelImportError(f"Неизвестная модель: {model_id}")
        if spec.provider != "onnx":
            raise ModelImportError("Эта модель встроена в Photo Doctor и не импортируется как ONNX.")
        source = Path(source_path).expanduser()
        if not source.is_file():
            raise ModelImportError("Файл модели не найден.")
        if source.suffix.lower() != ".onnx":
            raise ModelImportError("Ожидается локальный файл модели с расширением .onnx.")
        if source.stat().st_size <= 0:
            raise ModelImportError("Файл модели пуст.")

        self.model_dir.mkdir(parents=True, exist_ok=True)
        target = self.model_dir / spec.filename
        source_resolved = source.resolve()
        target_resolved = target.resolve(strict=False)
        if source_resolved != target_resolved:
            temp = target.with_suffix(target.suffix + ".importing")
            try:
                shutil.copyfile(source, temp)
                os.replace(temp, target)
            finally:
                if temp.exists():
                    temp.unlink(missing_ok=True)

        actual_hash = _sha256(target)
        entries = self._load_installed_manifest()
        entries[spec.model_id] = {
            "filename": spec.filename,
            "sha256": actual_hash,
            "size": target.stat().st_size,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source": "manual_local_import",
            "note": "SHA-256 контролирует только целостность файла и не подтверждает происхождение или качество модели.",
        }
        self._write_installed_manifest(entries)
        self.invalidate()
        return self.inspect(spec)

    def forget_integrity_pin(self, model_id: str) -> None:
        if model_id not in self._specs_by_id:
            raise ModelImportError(f"Неизвестная модель: {model_id}")
        entries = self._load_installed_manifest()
        if model_id in entries:
            del entries[model_id]
            self._write_installed_manifest(entries)
            self.invalidate()


from .manager_types import ModelSpec, ModelState  # noqa: E402,F401
