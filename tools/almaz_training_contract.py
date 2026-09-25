from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "training" / "almaz" / "TRAINING_CONTRACT.json"


@dataclass(frozen=True, slots=True)
class DatasetStats:
    train_pairs_total: int
    train_source_groups_total: int
    val_source_groups_total: int
    exam_source_groups_total: int
    train_pairs_by_task: dict[str, int]
    exam_source_groups_by_domain: dict[str, int]


def load_contract(path: str | Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_dataset_stats(stats: DatasetStats, contract: dict[str, Any] | None = None) -> list[str]:
    c = contract or load_contract()
    m = c["minimums"]
    failures: list[str] = []
    for field in ("train_pairs_total", "train_source_groups_total", "val_source_groups_total", "exam_source_groups_total"):
        got = int(getattr(stats, field))
        need = int(m[field])
        if got < need:
            failures.append(f"{field}: {got} < {need}")
    for task, need in m["train_pairs_by_task"].items():
        got = int(stats.train_pairs_by_task.get(task, 0))
        if got < int(need):
            failures.append(f"train_pairs_by_task.{task}: {got} < {need}")
    for domain, need in m["exam_source_groups_by_domain"].items():
        got = int(stats.exam_source_groups_by_domain.get(domain, 0))
        if got < int(need):
            failures.append(f"exam_source_groups_by_domain.{domain}: {got} < {need}")
    return failures


def load_dataset_stats(path: str | Path) -> DatasetStats:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return DatasetStats(
        train_pairs_total=int(raw.get("train_pairs_total", 0)),
        train_source_groups_total=int(raw.get("train_source_groups_total", 0)),
        val_source_groups_total=int(raw.get("val_source_groups_total", 0)),
        exam_source_groups_total=int(raw.get("exam_source_groups_total", 0)),
        train_pairs_by_task={str(k): int(v) for k, v in dict(raw.get("train_pairs_by_task", {})).items()},
        exam_source_groups_by_domain={str(k): int(v) for k, v in dict(raw.get("exam_source_groups_by_domain", {})).items()},
    )
