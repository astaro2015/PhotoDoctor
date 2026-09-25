from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .restoration_runtime import inspect_all_restoration_models
from .sr_backend import discover_sr_providers
from .sr_runtime import inspect_installed_sr_model


@dataclass(frozen=True, slots=True)
class AlmazStatus:
    sr: dict[str, Any]
    restoration: dict[str, dict[str, Any]]
    providers: list[dict[str, Any]]
    ready_models: int
    total_models: int
    best_provider: str | None
    best_provider_label: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_almaz_status() -> AlmazStatus:
    sr = inspect_installed_sr_model().to_dict()
    restoration = inspect_all_restoration_models()
    providers = [asdict(status) for status in discover_sr_providers()]
    available = [item for item in providers if bool(item.get("available", False))]
    best = available[0] if available else None
    ready = int(bool(sr.get("ready", False))) + sum(
        1 for item in restoration.values() if bool(item.get("ready", False))
    )
    return AlmazStatus(
        sr=sr,
        restoration=restoration,
        providers=providers,
        ready_models=ready,
        total_models=1 + len(restoration),
        best_provider=(str(best.get("provider")) if best else None),
        best_provider_label=(str(best.get("label")) if best else None),
    )
