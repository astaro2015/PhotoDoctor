from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    task: str
    filename: str
    version: str
    description: str
    input_size: tuple[int, int]
    sha256: str | None = None
    contract_id: str = "rgb01_nchw_v1"
    output_kind: str = "classification"
    labels: tuple[str, ...] = ()
    provider: str = "onnx"
    bundled: bool = False
    provenance: str = "user_supplied"


@dataclass(slots=True)
class ModelState:
    spec: ModelSpec
    path: Path
    status: str
    runtime_available: bool
    file_present: bool
    hash_ok: bool | None
    actual_sha256: str | None = None
    expected_sha256: str | None = None
    integrity_source: str = "none"
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["path"] = str(self.path)
        data["ready"] = self.ready
        return data
