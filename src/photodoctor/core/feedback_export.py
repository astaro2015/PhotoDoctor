from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .database import AnalysisDatabase
from .loader import load_image


@dataclass(frozen=True, slots=True)
class FeedbackExportSummary:
    exported: int
    skipped_missing: int
    skipped_changed: int
    skipped_invalid: int
    output_dir: Path


def _candidate_crop_geometry(
    rgb: np.ndarray, row: dict[str, object], margin: float = 1.55, *, min_fraction: float = 0.035
) -> tuple[np.ndarray, dict[str, float]] | None:
    h, w = rgb.shape[:2]
    try:
        x = float(row.get("x", 0.0) or 0.0) * w
        y = float(row.get("y", 0.0) or 0.0) * h
        bw = float(row.get("w", 0.0) or 0.0) * w
        bh = float(row.get("h", 0.0) or 0.0) * h
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite([x, y, bw, bh]).all() or bw <= 0 or bh <= 0:
        return None
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(20.0, max(bw, bh) * margin, min(w, h) * float(min_fraction))
    try:
        x0 = max(0, int(round(cx - side / 2.0)))
        y0 = max(0, int(round(cy - side / 2.0)))
        x1 = min(w, int(round(cx + side / 2.0)))
        y1 = min(h, int(round(cy + side / 2.0)))
    except (TypeError, ValueError, OverflowError):
        return None
    cw, ch = x1 - x0, y1 - y0
    if cw < 8 or ch < 8:
        return None
    geometry = {
        "x": float(np.clip((x - x0) / cw, 0.0, 1.0)),
        "y": float(np.clip((y - y0) / ch, 0.0, 1.0)),
        "w": float(np.clip(bw / cw, 0.0, 1.0)),
        "h": float(np.clip(bh / ch, 0.0, 1.0)),
    }
    return rgb[y0:y1, x0:x1], geometry


def _candidate_crop(rgb: np.ndarray, row: dict[str, object], margin: float = 1.55) -> np.ndarray | None:
    result = _candidate_crop_geometry(rgb, row, margin)
    return None if result is None else result[0]


def export_surface_feedback_dataset(db_path: str | Path, output_dir: str | Path, *, privacy_mode: bool = False) -> FeedbackExportSummary:
    output = Path(output_dir)
    patches_dir = output / "patches"
    context_dir = output / "context_patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    context_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"

    exported = skipped_missing = skipped_changed = skipped_invalid = 0
    manifest_rows: list[dict[str, object]] = []
    image_cache: dict[str, np.ndarray] = {}

    with AnalysisDatabase(db_path) as db:
        rows = db.all_surface_feedback()
        for row in rows:
            source = Path(str(row.get("path", "")))
            if not source.is_file():
                skipped_missing += 1
                continue
            try:
                current_hash = db.quick_hash(source)
            except OSError:
                skipped_missing += 1
                continue
            if current_hash != str(row.get("quick_hash", "")):
                skipped_changed += 1
                continue
            key = str(source.resolve())
            rgb = image_cache.get(key)
            if rgb is None:
                try:
                    rgb = load_image(source).srgb
                except Exception:
                    skipped_invalid += 1
                    continue
                image_cache[key] = rgb
            crop_result = _candidate_crop_geometry(rgb, row, 1.55, min_fraction=0.035)
            context_result = _candidate_crop_geometry(rgb, row, 4.8, min_fraction=0.12)
            if crop_result is None or context_result is None:
                skipped_invalid += 1
                continue
            crop, _ = crop_result
            context_crop, context_geometry = context_result
            patch = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA if max(crop.shape[:2]) > 96 else cv2.INTER_CUBIC)
            context_patch = cv2.resize(
                context_crop, (256, 256),
                interpolation=cv2.INTER_AREA if max(context_crop.shape[:2]) > 256 else cv2.INTER_CUBIC,
            )
            signature = str(row.get("candidate_signature", "unknown"))
            label = str(row.get("user_label", "uncertain"))
            source_token = str(row.get("quick_hash", ""))[:12]
            filename = f"{label}_{source_token}_{signature}.png"
            patch_path = patches_dir / filename
            context_path = context_dir / filename
            Image.fromarray(patch.astype(np.uint8), "RGB").save(patch_path)
            Image.fromarray(context_patch.astype(np.uint8), "RGB").save(context_path)
            dnu_label = {
                "defect": "D",
                "natural_detail": "N",
                "uncertain": "U",
            }.get(label)
            manifest_row = {
                "format": 2,
                "task": "surface_refiner",
                "patch": f"patches/{filename}",
                "context_patch": f"context_patches/{filename}",
                "source_token": str(row.get("quick_hash", ""))[:16],
                "source_quick_hash": str(row.get("quick_hash", "")),
                "source_group_id": str(row.get("quick_hash", "")),
                "candidate_signature": signature,
                "user_label": label,
                "label": dnu_label,
                "source_tier": "user_feedback",
                "label_origin": "user_manual",
                "expert_verified": False,
                "training_lane": "hard_mining",
                "license": "private-local-only",
                "consent_or_rights_confirmed": False,
                "use_scope": "preference_hard_case_only",
                "feature_schema": "native_gray96_hograwmeta_mlp_binary_v1",
                "context_feature_schema": "rgb256_candidate_context_v1",
                "ai_label": str(row.get("ai_label", "")),
                "ai_model_id": str(row.get("ai_model_id", "")),
                "ai_confidence": row.get("ai_confidence"),
                "raw_ai_label": str(row.get("raw_ai_label", "") or row.get("ai_label", "")),
                "raw_ai_confidence": row.get("raw_ai_confidence") if row.get("raw_ai_confidence") is not None else row.get("ai_confidence"),
                "verification_label": str(row.get("verification_label", "") or row.get("ai_label", "")),
                "verification_confidence": row.get("verification_confidence") if row.get("verification_confidence") is not None else row.get("ai_confidence"),
                "candidate_quality": row.get("candidate_quality"),
                "candidate_kind": str(row.get("candidate_kind", "")),
                "context_support": row.get("context_support"),
                "context_risk": row.get("context_risk"),
                "algorithm_version": str(row.get("algorithm_version", "")),
                "image_valid": True,
                "width": 96,
                "height": 96,
                "context_width": 256,
                "context_height": 256,
                "candidate_box_in_context": context_geometry,
                "geometry": {
                    "x": row.get("x"), "y": row.get("y"), "w": row.get("w"), "h": row.get("h"),
                    "polarity": row.get("polarity"), "strength": row.get("strength"),
                    "candidate_kind": row.get("candidate_kind"),
                },
            }
            if not privacy_mode:
                manifest_row["source_name"] = source.name
            manifest_rows.append(manifest_row)
            exported += 1

    with manifest_path.open("w", encoding="utf-8", newline="\n") as f:
        for item in manifest_rows:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "format": 2,
        "exported": exported,
        "skipped_missing": skipped_missing,
        "skipped_changed": skipped_changed,
        "skipped_invalid": skipped_invalid,
        "privacy_mode": bool(privacy_mode),
        "note": "Ручные D/N/U-метки Photo Doctor экспортируются только как user_feedback/hard_mining; они не являются expert ground truth. Для совместимости сохраняется patch 96x96, а для будущего контекстного verifier — context_patch 256x256 с положением кандидата. Полные пути к исходникам намеренно не экспортируются." + (" Имена исходных файлов также не экспортируются." if privacy_mode else ""),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return FeedbackExportSummary(exported, skipped_missing, skipped_changed, skipped_invalid, output)
