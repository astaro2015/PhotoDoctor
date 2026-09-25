from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class MetricResult:
    name: str
    raw_value: float | int | str | dict[str, Any]
    normalized_value: float | None
    confidence: float
    scale: str
    region: str = "global"
    diagnostic: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImageInfo:
    path: Path
    width: int
    height: int
    mode: str
    format: str
    exif_orientation: int | None = None
    icc_present: bool = False
    exif: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisResult:
    image: ImageInfo
    metrics: dict[str, MetricResult]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": {
                "path": str(self.image.path),
                "width": self.image.width,
                "height": self.image.height,
                "mode": self.image.mode,
                "format": self.image.format,
                "exif_orientation": self.image.exif_orientation,
                "icc_present": self.image.icc_present,
                "exif": self.image.exif,
            },
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "warnings": list(self.warnings),
        }
