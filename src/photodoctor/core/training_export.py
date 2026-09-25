from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from photodoctor import __version__
from photodoctor.ai.parameter_recommender import (
    FEATURE_SCHEMA as PARAMETER_FEATURE_SCHEMA,
    MODEL_ID as PARAMETER_MODEL_ID,
)

from .database import AnalysisDatabase
from .feedback_export import export_surface_feedback_dataset
from .versioning import ALGORITHM_VERSION, SERIES_ALGORITHM_VERSION

PARAMETER_FIRST_EXPERIMENT_MIN = 30
MIN_TRAINING_SOURCE_GROUPS = 5
MAX_TRAINING_SOURCE_GROUP_SHARE = 0.25
SURFACE_CLASS_MIN = 50
SURFACE_BINARY_TOTAL_MIN = 200


@dataclass(frozen=True, slots=True)
class TrainingDataStats:
    correction_rows: int
    meaningful_correction_decisions: int
    parameter_strength_samples: int
    parameter_by_action: dict[str, int]
    parameter_source_groups_by_action: dict[str, int]
    parameter_max_source_group_share_by_action: dict[str, float]
    surface_total: int
    surface_source_groups: int
    surface_max_source_group_share: float
    surface_defect: int
    surface_natural: int
    surface_uncertain: int
    parameter_ready_actions: tuple[str, ...]
    surface_ready: bool


@dataclass(frozen=True, slots=True)
class TrainingPackageSummary:
    output_zip: Path
    stats: TrainingDataStats
    parameter_exported: int
    surface_exported: int
    surface_skipped_missing: int
    surface_skipped_changed: int
    surface_skipped_invalid: int


def collect_training_data_stats(db_path: str | Path) -> TrainingDataStats:
    with AnalysisDatabase(db_path) as db:
        correction_rows = db.all_correction_feedback()
        surface_rows = db.all_surface_feedback()

    parameter_counts: Counter[str] = Counter()
    parameter_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    meaningful = 0
    strength_samples = 0
    for row in correction_rows:
        event = str(row.get("decision_event", ""))
        if event in {
            "auto_accepted_saved", "user_disabled_saved", "user_overrode_validator_saved",
            "user_enabled_review_saved",
        }:
            meaningful += 1
        if int(row.get("user_selected", 0) or 0) and row.get("user_strength") is not None:
            try:
                features = json.loads(str(row.get("parameter_features_json") or "[]"))
            except json.JSONDecodeError:
                features = []
            current_schema = str(row.get("parameter_feature_schema", "") or "") == PARAMETER_FEATURE_SCHEMA
            current_model = str(row.get("ai_parameter_model_id", "") or "") == PARAMETER_MODEL_ID
            if isinstance(features, list) and features and current_schema and current_model:
                action = str(row.get("action_key", "")).strip()
                if action:
                    parameter_counts[action] += 1
                    source_group = str(row.get("quick_hash", "") or "").strip() or "missing-source-group"
                    parameter_group_counts[action][source_group] += 1
                    strength_samples += 1

    surface_counts: Counter[str] = Counter(str(row.get("user_label", "")) for row in surface_rows)
    surface_group_counts: Counter[str] = Counter(
        str(row.get("quick_hash", "") or "").strip() or "missing-source-group"
        for row in surface_rows
        if str(row.get("user_label", "")) in {"defect", "natural_detail", "uncertain"}
    )
    defect = int(surface_counts.get("defect", 0))
    natural = int(surface_counts.get("natural_detail", 0))
    uncertain = int(surface_counts.get("uncertain", 0))
    surface_total = defect + natural + uncertain
    binary_total = defect + natural
    surface_source_groups = len(surface_group_counts)
    surface_max_source_group_share = (
        max(surface_group_counts.values()) / surface_total if surface_total and surface_group_counts else 0.0
    )
    surface_ready = (
        defect >= SURFACE_CLASS_MIN
        and natural >= SURFACE_CLASS_MIN
        and binary_total >= SURFACE_BINARY_TOTAL_MIN
        and surface_source_groups >= MIN_TRAINING_SOURCE_GROUPS
        and surface_max_source_group_share <= MAX_TRAINING_SOURCE_GROUP_SHARE
    )
    parameter_source_groups = {action: len(groups) for action, groups in parameter_group_counts.items()}
    parameter_max_group_share = {
        action: (max(groups.values()) / parameter_counts[action] if parameter_counts[action] else 0.0)
        for action, groups in parameter_group_counts.items()
    }
    ready_actions = tuple(sorted(
        action for action, count in parameter_counts.items()
        if count >= PARAMETER_FIRST_EXPERIMENT_MIN
        and parameter_source_groups.get(action, 0) >= MIN_TRAINING_SOURCE_GROUPS
        and parameter_max_group_share.get(action, 1.0) <= MAX_TRAINING_SOURCE_GROUP_SHARE
    ))

    return TrainingDataStats(
        correction_rows=len(correction_rows),
        meaningful_correction_decisions=meaningful,
        parameter_strength_samples=strength_samples,
        parameter_by_action=dict(sorted(parameter_counts.items())),
        parameter_source_groups_by_action=dict(sorted(parameter_source_groups.items())),
        parameter_max_source_group_share_by_action=dict(sorted(parameter_max_group_share.items())),
        surface_total=surface_total,
        surface_source_groups=surface_source_groups,
        surface_max_source_group_share=float(surface_max_source_group_share),
        surface_defect=defect,
        surface_natural=natural,
        surface_uncertain=uncertain,
        parameter_ready_actions=ready_actions,
        surface_ready=surface_ready,
    )


def _parameter_manifest_rows(db_path: str | Path) -> list[dict[str, object]]:
    with AnalysisDatabase(db_path) as db:
        rows = db.all_correction_feedback()

    exported: list[dict[str, object]] = []
    for row in rows:
        try:
            features = json.loads(str(row.get("parameter_features_json") or "[]"))
        except json.JSONDecodeError:
            features = []
        if not isinstance(features, list):
            features = []
        exported.append({
            "format": 1,
            "source_token": str(row.get("quick_hash", ""))[:16],
            "save_event_token": str(row.get("save_event_id", ""))[:16],
            "action_key": str(row.get("action_key", "")),
            "decision_event": str(row.get("decision_event", "")),
            "validator_accepted": bool(row.get("validator_accepted", 0)),
            "user_selected": bool(row.get("user_selected", 0)),
            "validation_confidence": row.get("validation_confidence"),
            "ai_suggested_strength": row.get("ai_suggested_strength"),
            "user_strength": row.get("user_strength"),
            "ai_parameter_model_id": str(row.get("ai_parameter_model_id", "")),
            "ai_parameter_confidence": row.get("ai_parameter_confidence"),
            "feature_schema": str(row.get("parameter_feature_schema", "") or "parameter_features_v1"),
            "feedback_kind": str(row.get("feedback_kind", "") or "preference"),
            "source_tier": str(row.get("source_tier", "") or "user_feedback"),
            "label_origin": "user",
            "training_lane": "preference",
            "expert_verified": False,
            "source_group_id": str(row.get("quick_hash", ""))[:16],
            "parameter_features": features,
            "algorithm_version": str(row.get("algorithm_version", "")),
            "saved_format": str(row.get("saved_format", "")),
            "created_at": str(row.get("created_at", "")),
        })
    return exported


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def export_training_package(db_path: str | Path, output_zip: str | Path) -> TrainingPackageSummary:
    output = Path(output_zip)
    if output.suffix.lower() != ".zip":
        output = output.with_suffix(".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = collect_training_data_stats(db_path)

    with tempfile.TemporaryDirectory(prefix="photodoctor_training_") as tmp_name:
        root = Path(tmp_name) / "PhotoDoctor_AI_Training_Data"
        parameter_dir = root / "parameter_feedback"
        surface_dir = root / "surface_feedback"
        parameter_rows = _parameter_manifest_rows(db_path)
        _write_jsonl(parameter_dir / "manifest.jsonl", parameter_rows)

        parameter_summary = {
            "format": 1,
            "rows": len(parameter_rows),
            "strength_samples": stats.parameter_strength_samples,
            "by_action": stats.parameter_by_action,
            "first_experiment_min_per_action": PARAMETER_FIRST_EXPERIMENT_MIN,
            "min_source_groups": MIN_TRAINING_SOURCE_GROUPS,
            "max_source_group_share": MAX_TRAINING_SOURCE_GROUP_SHARE,
            "source_groups_by_action": stats.parameter_source_groups_by_action,
            "max_source_group_share_by_action": stats.parameter_max_source_group_share_by_action,
            "ready_actions": list(stats.parameter_ready_actions),
            "model_family": PARAMETER_MODEL_ID,
            "feature_schema": PARAMETER_FEATURE_SCHEMA,
            "note": "Готовность считается только по текущей v2-схеме; legacy/v1 строки экспортируются только с явной маркировкой и не смешиваются с v2.",
        }
        (parameter_dir / "summary.json").write_text(
            json.dumps(parameter_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        surface = export_surface_feedback_dataset(db_path, surface_dir, privacy_mode=True)

        dataset_info = {
            "format": 1,
            "photo_doctor_version": __version__,
            "algorithm_version": ALGORITHM_VERSION,
            "series_algorithm_version": SERIES_ALGORITHM_VERSION,
            "parameter_model_id": PARAMETER_MODEL_ID,
            "parameter_feature_schema": PARAMETER_FEATURE_SCHEMA,
            "privacy": {
                "full_source_paths_exported": False,
                "source_file_names_exported": False,
                "photos_exported": False,
                "surface_patches_exported": True,
            },
            "readiness": {
                "parameter_first_experiment_min_per_action": PARAMETER_FIRST_EXPERIMENT_MIN,
                "min_source_groups": MIN_TRAINING_SOURCE_GROUPS,
                "max_source_group_share": MAX_TRAINING_SOURCE_GROUP_SHARE,
                "surface_min_defect": SURFACE_CLASS_MIN,
                "surface_min_natural_detail": SURFACE_CLASS_MIN,
                "surface_min_binary_total": SURFACE_BINARY_TOTAL_MIN,
                "note": "Порог означает достаточный объём данных для первого эксперимента, а не гарантированное качество рабочей модели.",
            },
        }
        (root / "dataset_info.json").write_text(
            json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "summary.json").write_text(
            json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "README.txt").write_text(
            "Photo Doctor AI training export\n"
            "Parameter feedback contains saved user decisions and slider targets.\n"
            "Surface feedback contains 96x96 compatibility patches plus 256x256 context patches for D/N/U hard-mining labels.\n"
            "Full source paths, source file names and complete photographs are intentionally omitted.\n",
            encoding="utf-8",
        )

        temp_zip = output.with_suffix(output.suffix + ".tmp")
        if temp_zip.exists():
            temp_zip.unlink()
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(root.rglob("*")):
                if file.is_file():
                    archive.write(file, file.relative_to(root.parent).as_posix())
        temp_zip.replace(output)

    return TrainingPackageSummary(
        output_zip=output,
        stats=stats,
        parameter_exported=len(parameter_rows),
        surface_exported=surface.exported,
        surface_skipped_missing=surface.skipped_missing,
        surface_skipped_changed=surface.skipped_changed,
        surface_skipped_invalid=surface.skipped_invalid,
    )
