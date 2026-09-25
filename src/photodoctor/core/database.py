from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Mapping

import numpy as np

from photodoctor.ai.model_fingerprint import sampled_file_digest

from .models import AnalysisResult, ImageInfo, MetricResult
from .versioning import ALGORITHM_VERSION, NORMALIZATION_VERSION
from .precision import get_precision


SCHEMA_VERSION = 9


SURFACE_FEEDBACK_LABELS = {"defect", "natural_detail", "uncertain"}


def surface_candidate_signature(candidate: dict[str, object]) -> str:
    """Stable-enough fingerprint for one visible surface candidate.

    Deliberately excludes AI output and contour point density. If detector geometry
    changes materially in a future release, the signature changes rather than
    silently attaching a human label to the wrong scratch.
    """
    payload = {
        "x": round(float(candidate.get("x", 0.0) or 0.0), 6),
        "y": round(float(candidate.get("y", 0.0) or 0.0), 6),
        "w": round(float(candidate.get("w", 0.0) or 0.0), 6),
        "h": round(float(candidate.get("h", 0.0) or 0.0), 6),
        "polarity": str(candidate.get("polarity", "")),
        "strength": round(float(candidate.get("strength", 0.0) or 0.0), 4),
    }
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.blake2b(blob, digest_size=12).hexdigest()


def _analysis_model_state_signature() -> str:
    """Cheap fingerprint of optional external AI/restoration model state.

    Cached analysis can depend on which ONNX/model manifests are installed.  Do
    not hash large model payloads on every cache lookup; model installation and
    replacement already changes file metadata.  Small manifests are covered by
    the same stable name/size/mtime inventory.  False invalidation is acceptable
    here; stale AI decisions are not.
    """
    override = os.environ.get("PHOTODOCTOR_MODEL_DIR", "").strip()
    root = Path(override).expanduser() if override else (Path.home() / ".photodoctor" / "models")
    payload: list[str] = [str(root.absolute())]
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name.casefold()) if root.is_dir() else []
    except OSError:
        entries = []
    for item in entries:
        try:
            if not item.is_file():
                continue
            # The model directory is deliberately narrow in scope.  Include every
            # regular top-level file so future model formats/manifests invalidate
            # old analysis without needing another cache-key migration.
            stat = item.stat()
            sample = sampled_file_digest(item)
            payload.append(f"{item.name}|{stat.st_size}|{stat.st_mtime_ns}|{sample}")
        except OSError:
            # A model being replaced concurrently is itself a reason not to trust
            # an old cached AI decision.
            payload.append(f"{item.name}|unstable")
    encoded = "\n".join(payload).encode("utf-8", errors="surrogatepass")
    return hashlib.blake2b(encoded, digest_size=10).hexdigest()


def _cache_algorithm_version(precision: str = "normal") -> str:
    return (
        f"{ALGORITHM_VERSION}|precision={get_precision(precision).key}"
        f"|models={_analysis_model_state_signature()}"
    )


class AnalysisDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS photo(
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    size INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    quick_hash TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    format TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    algorithm_version TEXT NOT NULL,
                    normalization_version TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metric(
                    photo_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    normalized_value REAL,
                    confidence REAL NOT NULL,
                    scale TEXT NOT NULL,
                    region TEXT NOT NULL,
                    diagnostic TEXT NOT NULL,
                    PRIMARY KEY(photo_id, name),
                    FOREIGN KEY(photo_id) REFERENCES photo(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_photo_cache
                    ON photo(path, size, modified_ns, quick_hash, algorithm_version, normalization_version);
                CREATE TABLE IF NOT EXISTS surface_feedback(
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL,
                    quick_hash TEXT NOT NULL,
                    candidate_signature TEXT NOT NULL,
                    algorithm_version TEXT NOT NULL,
                    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
                    polarity TEXT NOT NULL DEFAULT '',
                    strength REAL NOT NULL DEFAULT 0,
                    ai_label TEXT NOT NULL DEFAULT '',
                    ai_model_id TEXT NOT NULL DEFAULT '',
                    ai_confidence REAL,
                    raw_ai_label TEXT NOT NULL DEFAULT '',
                    raw_ai_confidence REAL,
                    verification_label TEXT NOT NULL DEFAULT '',
                    verification_confidence REAL,
                    candidate_quality REAL,
                    candidate_kind TEXT NOT NULL DEFAULT '',
                    context_support REAL,
                    context_risk REAL,
                    user_label TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(path, quick_hash, candidate_signature)
                );
                CREATE INDEX IF NOT EXISTS idx_surface_feedback_file
                    ON surface_feedback(path, quick_hash);
                CREATE TABLE IF NOT EXISTS file_fingerprint(
                    path TEXT NOT NULL,
                    quick_hash TEXT NOT NULL,
                    method TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(path, quick_hash, method)
                );
                CREATE INDEX IF NOT EXISTS idx_file_fingerprint_path
                    ON file_fingerprint(path, method);
                CREATE TABLE IF NOT EXISTS correction_feedback(
                    id INTEGER PRIMARY KEY,
                    save_event_id TEXT NOT NULL,
                    path TEXT NOT NULL,
                    quick_hash TEXT NOT NULL,
                    action_key TEXT NOT NULL,
                    candidate TEXT NOT NULL DEFAULT '',
                    validator_accepted INTEGER NOT NULL,
                    user_selected INTEGER NOT NULL,
                    decision_event TEXT NOT NULL,
                    validation_confidence REAL,
                    ai_suggested_strength REAL,
                    user_strength REAL,
                    ai_parameter_model_id TEXT NOT NULL DEFAULT '',
                    ai_parameter_confidence REAL,
                    parameter_features_json TEXT NOT NULL DEFAULT '[]',
                    parameter_feature_schema TEXT NOT NULL DEFAULT 'parameter_features_v1',
                    feedback_kind TEXT NOT NULL DEFAULT 'preference',
                    source_tier TEXT NOT NULL DEFAULT 'user_feedback',
                    validation_json TEXT NOT NULL DEFAULT '{}',
                    context_json TEXT NOT NULL DEFAULT '{}',
                    algorithm_version TEXT NOT NULL,
                    saved_format TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_correction_feedback_action
                    ON correction_feedback(action_key, created_at);
                CREATE INDEX IF NOT EXISTS idx_correction_feedback_file
                    ON correction_feedback(path, quick_hash, created_at);
                CREATE TABLE IF NOT EXISTS manual_vision_override(
                    path TEXT NOT NULL,
                    quick_hash TEXT NOT NULL,
                    faces_json TEXT NOT NULL DEFAULT '[]',
                    eyes_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(path, quick_hash)
                );
                CREATE INDEX IF NOT EXISTS idx_manual_vision_override_path
                    ON manual_vision_override(path);
                """
            )
            columns = {row[1] for row in self.conn.execute("PRAGMA table_info(surface_feedback)").fetchall()}
            surface_migrations = {
                "ai_model_id": "TEXT NOT NULL DEFAULT ''",
                "raw_ai_label": "TEXT NOT NULL DEFAULT ''",
                "raw_ai_confidence": "REAL",
                "verification_label": "TEXT NOT NULL DEFAULT ''",
                "verification_confidence": "REAL",
                "candidate_quality": "REAL",
                "candidate_kind": "TEXT NOT NULL DEFAULT ''",
                "context_support": "REAL",
                "context_risk": "REAL",
            }
            for name, declaration in surface_migrations.items():
                if name not in columns:
                    self.conn.execute(f"ALTER TABLE surface_feedback ADD COLUMN {name} {declaration}")
            correction_columns = {row[1] for row in self.conn.execute("PRAGMA table_info(correction_feedback)").fetchall()}
            correction_migrations = {
                "ai_suggested_strength": "REAL",
                "user_strength": "REAL",
                "ai_parameter_model_id": "TEXT NOT NULL DEFAULT ''",
                "ai_parameter_confidence": "REAL",
                "parameter_features_json": "TEXT NOT NULL DEFAULT '[]'",
                "parameter_feature_schema": "TEXT NOT NULL DEFAULT 'parameter_features_v1'",
                "feedback_kind": "TEXT NOT NULL DEFAULT 'preference'",
                "source_tier": "TEXT NOT NULL DEFAULT 'user_feedback'",
            }
            for name, declaration in correction_migrations.items():
                if name not in correction_columns:
                    self.conn.execute(f"ALTER TABLE correction_feedback ADD COLUMN {name} {declaration}")
            self._backfill_legacy_parameter_feedback()
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))

    def _backfill_legacy_parameter_feedback(self) -> None:
        """Reuse 0.4.3-0.4.5 saved slider values as v1 personal-training examples.

        Older releases already stored ``correction_strengths`` inside context_json
        but did not have explicit parameter-learning columns.  We recover only
        selected/saved actions; rejected or disabled actions remain selection
        feedback rather than fake ``strength=0`` samples.
        """
        try:
            rows = self.conn.execute(
                """SELECT id,action_key,user_selected,context_json,validation_json
                   FROM correction_feedback
                   WHERE user_selected=1 AND user_strength IS NULL"""
            ).fetchall()
        except sqlite3.OperationalError:
            return
        if not rows:
            return
        from photodoctor.ai.parameter_recommender import (
            LEGACY_FEATURE_SCHEMA, LEGACY_MODEL_ID, build_features_v1_legacy,
        )

        updates: list[tuple[float, str, str, str, int]] = []
        for row in rows:
            try:
                context = json.loads(row["context_json"] or "{}")
                validation = json.loads(row["validation_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(context, dict) or not isinstance(validation, dict):
                continue
            strengths = context.get("correction_strengths", {})
            if not isinstance(strengths, dict):
                continue
            action_key = str(row["action_key"] or "")
            if action_key not in strengths:
                continue
            try:
                user_strength = float(np.clip(float(strengths[action_key]), 0.0, 1.0))
                features = build_features_v1_legacy(action_key, validation, context)
            except (TypeError, ValueError):
                continue
            feature_json = json.dumps(features, ensure_ascii=True, separators=(",", ":"), default=float)
            updates.append((user_strength, LEGACY_MODEL_ID + ":legacy", feature_json, LEGACY_FEATURE_SCHEMA, int(row["id"])))
        if updates:
            self.conn.executemany(
                "UPDATE correction_feedback SET user_strength=?, ai_parameter_model_id=?, parameter_features_json=?, parameter_feature_schema=? WHERE id=?",
                updates,
            )

    @staticmethod
    def quick_hash(path: str | Path, block_size: int = 65536) -> str:
        p = Path(path)
        h = hashlib.blake2b(digest_size=16)
        size = p.stat().st_size
        with p.open("rb") as f:
            h.update(f.read(block_size))
            if size > block_size:
                f.seek(max(0, size - block_size))
                h.update(f.read(block_size))
        h.update(str(size).encode())
        return h.hexdigest()

    def is_current(
        self, path: str | Path, precision: str = "normal", *, quick_hash: str | None = None
    ) -> bool:
        p = Path(path)
        try:
            st = p.stat()
            resolved = str(p.resolve())
        except OSError:
            return False
        # First reject paths that cannot possibly be current without touching the
        # file contents.  This matters for large first-time batches and slower
        # disks/network shares.  If metadata matches, the quick hash is still
        # mandatory, so replacing contents while preserving size/mtime cannot
        # silently reuse stale analysis.
        row = self.conn.execute(
            "SELECT quick_hash FROM photo WHERE path=? AND size=? AND modified_ns=? "
            "AND algorithm_version=? AND normalization_version=?",
            (resolved, st.st_size, st.st_mtime_ns, _cache_algorithm_version(precision), NORMALIZATION_VERSION),
        ).fetchone()
        if row is None:
            return False
        if quick_hash is None:
            try:
                quick_hash = self.quick_hash(p)
            except OSError:
                return False
        return str(row["quick_hash"]) == str(quick_hash)


    def load_current_result(
        self, image: ImageInfo, precision: str = "normal", *, quick_hash: str | None = None
    ) -> AnalysisResult | None:
        """Return a persistent cached analysis only when the source fingerprint is current.

        The GUI already has to decode the source image for display.  Reusing the cached
        metrics here avoids repeating the expensive analysis when reopening an unchanged
        photo or opening a photo that batch analysis has already processed.  Manual vision
        edits invalidate the same photo cache, so this path cannot bypass them.
        """
        if not self.is_current(image.path, precision=precision, quick_hash=quick_hash):
            return None
        metrics = self.load_metrics(image.path, precision=precision)
        if metrics is None:
            return None
        return AnalysisResult(image=image, metrics=metrics)


    def load_metrics(self, path: str | Path, precision: str = "normal") -> dict[str, MetricResult] | None:
        """Load cached metrics for *path* when they match the current algorithm versions."""
        p = Path(path)
        try:
            resolved = str(p.resolve())
        except OSError:
            resolved = str(p.absolute())
        photo = self.conn.execute(
            "SELECT id FROM photo WHERE path=? AND algorithm_version=? AND normalization_version=?",
            (resolved, _cache_algorithm_version(precision), NORMALIZATION_VERSION),
        ).fetchone()
        if photo is None:
            return None
        rows = self.conn.execute(
            "SELECT name,raw_json,normalized_value,confidence,scale,region,diagnostic "
            "FROM metric WHERE photo_id=? ORDER BY rowid",
            (photo[0],),
        ).fetchall()
        metrics: dict[str, MetricResult] = {}
        for row in rows:
            try:
                raw = json.loads(row["raw_json"])
            except (TypeError, json.JSONDecodeError):
                raw = row["raw_json"]
            metric = MetricResult(
                name=row["name"],
                raw_value=raw,
                normalized_value=row["normalized_value"],
                confidence=float(row["confidence"]),
                scale=row["scale"],
                region=row["region"],
                diagnostic=row["diagnostic"],
            )
            metrics[metric.name] = metric
        return metrics

    def load_file_fingerprint(self, path: str | Path, method: str) -> str | None:
        p = Path(path)
        try:
            resolved = str(p.resolve())
            qh = self.quick_hash(p)
        except OSError:
            return None
        row = self.conn.execute(
            "SELECT value FROM file_fingerprint WHERE path=? AND quick_hash=? AND method=?",
            (resolved, qh, str(method)),
        ).fetchone()
        return None if row is None else str(row["value"])

    def save_file_fingerprint(self, path: str | Path, method: str, value: str) -> None:
        p = Path(path)
        resolved = str(p.resolve())
        qh = self.quick_hash(p)
        with self.conn:
            self.conn.execute(
                "DELETE FROM file_fingerprint WHERE path=? AND method=? AND quick_hash<>?",
                (resolved, str(method), qh),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO file_fingerprint(path,quick_hash,method,value,updated_at) "
                "VALUES(?,?,?,?,CURRENT_TIMESTAMP)",
                (resolved, qh, str(method), str(value)),
            )

    def _invalidate_cached_analysis_resolved(self, resolved: str) -> None:
        rows = self.conn.execute("SELECT id FROM photo WHERE path=?", (resolved,)).fetchall()
        for row in rows:
            self.conn.execute("DELETE FROM metric WHERE photo_id=?", (int(row[0]),))
        self.conn.execute("DELETE FROM photo WHERE path=?", (resolved,))

    def load_manual_vision_overrides(
        self, path: str | Path, *, quick_hash: str | None = None
    ) -> dict[str, list[dict[str, float]]]:
        p = Path(path)
        try:
            resolved = str(p.resolve())
        except OSError:
            return {"faces": [], "eyes": []}
        # Most photos never receive manual face/eye edits. Check for any row by
        # path before reading file contents for the hash; only existing overrides
        # need content binding.
        exists = self.conn.execute(
            "SELECT 1 FROM manual_vision_override WHERE path=? LIMIT 1", (resolved,)
        ).fetchone()
        if exists is None:
            return {"faces": [], "eyes": []}
        try:
            qh = str(quick_hash) if quick_hash is not None else self.quick_hash(p)
        except OSError:
            return {"faces": [], "eyes": []}
        row = self.conn.execute(
            "SELECT faces_json,eyes_json FROM manual_vision_override WHERE path=? AND quick_hash=?",
            (resolved, qh),
        ).fetchone()
        if row is None:
            return {"faces": [], "eyes": []}
        out: dict[str, list[dict[str, float]]] = {"faces": [], "eyes": []}
        for key, column in (("faces", "faces_json"), ("eyes", "eyes_json")):
            try:
                raw = json.loads(row[column])
            except (TypeError, json.JSONDecodeError):
                raw = []
            if isinstance(raw, list):
                clean: list[dict[str, float]] = []
                for box in raw:
                    if not isinstance(box, dict):
                        continue
                    try:
                        clean.append({k: float(box.get(k, 0.0)) for k in ("x", "y", "w", "h")})
                    except (TypeError, ValueError):
                        continue
                out[key] = clean
        return out

    def save_manual_vision_overrides(
        self,
        path: str | Path,
        faces: list[dict[str, float]],
        eyes: list[dict[str, float]],
    ) -> None:
        p = Path(path)
        resolved = str(p.resolve())
        qh = self.quick_hash(p)
        faces_json = json.dumps(faces, ensure_ascii=False, separators=(",", ":"))
        eyes_json = json.dumps(eyes, ensure_ascii=False, separators=(",", ":"))
        with self.conn:
            self.conn.execute(
                "DELETE FROM manual_vision_override WHERE path=? AND quick_hash<>?",
                (resolved, qh),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO manual_vision_override(path,quick_hash,faces_json,eyes_json,updated_at) "
                "VALUES(?,?,?,?,CURRENT_TIMESTAMP)",
                (resolved, qh, faces_json, eyes_json),
            )
            self._invalidate_cached_analysis_resolved(resolved)

    def clear_manual_vision_overrides(self, path: str | Path) -> None:
        p = Path(path)
        try:
            resolved = str(p.resolve())
        except OSError:
            resolved = str(p.absolute())
        with self.conn:
            self.conn.execute("DELETE FROM manual_vision_override WHERE path=?", (resolved,))
            self._invalidate_cached_analysis_resolved(resolved)

    def ai_consistency_stats(self, precision: str = "normal") -> list[dict[str, object]]:
        """Aggregate AI/classical consistency for current cached analyses.

        This is not model accuracy and not ground truth. It only summarizes how
        often a pinned local model agreed with, refined, or contradicted the
        classical Photo Doctor pipeline on the current archive/cache.
        """
        rows = self.conn.execute(
            "SELECT p.id AS photo_id, m.raw_json FROM metric m "
            "JOIN photo p ON p.id=m.photo_id "
            "WHERE m.name='ai_crosscheck' AND p.algorithm_version=? AND p.normalization_version=?",
            (_cache_algorithm_version(precision), NORMALIZATION_VERSION),
        ).fetchall()
        groups: dict[tuple[str, str], dict[str, object]] = {}
        for row in rows:
            try:
                raw = json.loads(row["raw_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            comparisons = raw.get("comparisons", []) if isinstance(raw, dict) else []
            if not isinstance(comparisons, list):
                continue
            for comparison in comparisons:
                if not isinstance(comparison, dict):
                    continue
                task = str(comparison.get("task", ""))
                model_id = str(comparison.get("model_id", ""))
                if not task or not model_id:
                    continue
                key = (task, model_id)
                group = groups.setdefault(key, {
                    "task": task, "model_id": model_id, "photos": set(), "comparisons": 0,
                    "agree": 0, "refine": 0, "contradict": 0, "low_confidence": 0, "not_comparable": 0,
                })
                group["photos"].add(int(row["photo_id"]))  # type: ignore[index,union-attr]
                group["comparisons"] = int(group["comparisons"]) + 1
                status = str(comparison.get("status", "not_comparable"))
                if status not in {"agree", "refine", "contradict", "low_confidence", "not_comparable"}:
                    status = "not_comparable"
                group[status] = int(group[status]) + 1

        result: list[dict[str, object]] = []
        for group in groups.values():
            evaluated = int(group["agree"]) + int(group["refine"]) + int(group["contradict"])
            compatibility = (
                (int(group["agree"]) + int(group["refine"])) / evaluated * 100.0
                if evaluated else None
            )
            result.append({
                "task": group["task"],
                "model_id": group["model_id"],
                "photo_count": len(group["photos"]),
                "comparison_count": int(group["comparisons"]),
                "agree": int(group["agree"]),
                "refine": int(group["refine"]),
                "contradict": int(group["contradict"]),
                "low_confidence": int(group["low_confidence"]),
                "not_comparable": int(group["not_comparable"]),
                "compatibility_pct": compatibility,
            })
        result.sort(key=lambda item: (-int(item["contradict"]), -int(item["comparison_count"]), str(item["model_id"])))
        return result

    def load_surface_feedback(self, path: str | Path) -> dict[str, dict[str, object]]:
        p = Path(path)
        try:
            resolved = str(p.resolve())
            qh = self.quick_hash(p)
        except OSError:
            return {}
        rows = self.conn.execute(
            "SELECT candidate_signature,user_label,ai_label,ai_model_id,ai_confidence,raw_ai_label,raw_ai_confidence,"
            "verification_label,verification_confidence,candidate_quality,candidate_kind,context_support,context_risk,"
            "x,y,w,h,polarity,strength,updated_at "
            "FROM surface_feedback WHERE path=? AND quick_hash=?",
            (resolved, qh),
        ).fetchall()
        return {
            str(row["candidate_signature"]): {
                "user_label": row["user_label"],
                "ai_label": row["ai_label"],
                "ai_model_id": row["ai_model_id"],
                "ai_confidence": row["ai_confidence"],
                "raw_ai_label": row["raw_ai_label"],
                "raw_ai_confidence": row["raw_ai_confidence"],
                "verification_label": row["verification_label"],
                "verification_confidence": row["verification_confidence"],
                "candidate_quality": row["candidate_quality"],
                "candidate_kind": row["candidate_kind"],
                "context_support": row["context_support"],
                "context_risk": row["context_risk"],
                "x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"],
                "polarity": row["polarity"], "strength": row["strength"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        }

    def all_surface_feedback(self) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT path,quick_hash,candidate_signature,algorithm_version,x,y,w,h,polarity,strength,"
            "ai_label,ai_model_id,ai_confidence,raw_ai_label,raw_ai_confidence,verification_label,verification_confidence,"
            "candidate_quality,candidate_kind,context_support,context_risk,user_label,updated_at "
            "FROM surface_feedback ORDER BY updated_at,path,id"
        ).fetchall()
        return [dict(row) for row in rows]

    def surface_feedback_stats(self) -> list[dict[str, object]]:
        rows = self.conn.execute(
            "SELECT ai_model_id,ai_label,ai_confidence,raw_ai_label,raw_ai_confidence,user_label FROM surface_feedback"
        ).fetchall()
        groups: dict[str, dict[str, object]] = {}
        for row in rows:
            model_id = str(row["ai_model_id"] or "unknown")
            group = groups.setdefault(model_id, {
                "model_id": model_id, "labeled": 0, "binary_evaluated": 0, "correct": 0, "wrong": 0,
                "user_uncertain": 0, "ai_uncertain_or_missing": 0,
                "tp_defect": 0, "fn_defect_as_natural": 0,
                "fp_natural_as_defect": 0, "tn_natural": 0,
                "high_confidence_wrong": 0, "correct_confidence_sum": 0.0, "wrong_confidence_sum": 0.0,
            })
            group["labeled"] = int(group["labeled"]) + 1
            user = str(row["user_label"] or "")
            ai = str(row["raw_ai_label"] or row["ai_label"] or "")
            try:
                confidence = float(row["raw_ai_confidence"] if row["raw_ai_confidence"] is not None else (row["ai_confidence"] or 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if user == "uncertain":
                group["user_uncertain"] = int(group["user_uncertain"]) + 1
                continue
            if user not in {"defect", "natural_detail"}:
                continue
            if ai not in {"defect", "natural_detail"}:
                group["ai_uncertain_or_missing"] = int(group["ai_uncertain_or_missing"]) + 1
                continue
            group["binary_evaluated"] = int(group["binary_evaluated"]) + 1
            correct = ai == user
            if correct:
                group["correct"] = int(group["correct"]) + 1
                group["correct_confidence_sum"] = float(group["correct_confidence_sum"]) + confidence
            else:
                group["wrong"] = int(group["wrong"]) + 1
                group["wrong_confidence_sum"] = float(group["wrong_confidence_sum"]) + confidence
                if confidence >= 0.80:
                    group["high_confidence_wrong"] = int(group["high_confidence_wrong"]) + 1
            if user == "defect" and ai == "defect":
                group["tp_defect"] = int(group["tp_defect"]) + 1
            elif user == "defect" and ai == "natural_detail":
                group["fn_defect_as_natural"] = int(group["fn_defect_as_natural"]) + 1
            elif user == "natural_detail" and ai == "defect":
                group["fp_natural_as_defect"] = int(group["fp_natural_as_defect"]) + 1
            elif user == "natural_detail" and ai == "natural_detail":
                group["tn_natural"] = int(group["tn_natural"]) + 1

        result: list[dict[str, object]] = []
        for group in groups.values():
            evaluated = int(group["binary_evaluated"])
            correct = int(group["correct"])
            wrong = int(group["wrong"])
            tp = int(group["tp_defect"]); fn = int(group["fn_defect_as_natural"])
            fp = int(group["fp_natural_as_defect"]); tn = int(group["tn_natural"])
            result.append({
                **{k: v for k, v in group.items() if not k.endswith("_sum")},
                "agreement_pct": (correct / evaluated * 100.0) if evaluated else None,
                "defect_recall_pct": (tp / (tp + fn) * 100.0) if (tp + fn) else None,
                "natural_recall_pct": (tn / (tn + fp) * 100.0) if (tn + fp) else None,
                "mean_confidence_correct": (float(group["correct_confidence_sum"]) / correct) if correct else None,
                "mean_confidence_wrong": (float(group["wrong_confidence_sum"]) / wrong) if wrong else None,
            })
        result.sort(key=lambda item: (-int(item["labeled"]), str(item["model_id"])))
        return result

    def record_correction_feedback(
        self,
        path: str | Path,
        validation_items: list[dict[str, object]] | tuple[dict[str, object], ...],
        selected_action_keys: set[str] | list[str] | tuple[str, ...],
        *,
        saved_format: str = "",
        context: dict[str, object] | None = None,
        user_strengths: Mapping[str, float] | None = None,
        parameter_suggestions: Mapping[str, Mapping[str, object]] | None = None,
    ) -> int:
        """Record the correction set that actually reached a successful save.

        In addition to keep/disable events, v0.4.6 stores the AI parameter
        recommendation and the exact strength the user saved.  This is the
        training signal for the local Parameter Recommender; it never changes
        the image-analysis model or Validator thresholds.
        """
        p = Path(path)
        resolved = str(p.resolve())
        qh = self.quick_hash(p)
        selected = {str(key) for key in selected_action_keys}
        strengths = {str(k): float(v) for k, v in (user_strengths or {}).items()}
        suggestions = {str(k): v for k, v in (parameter_suggestions or {}).items()}
        save_event_id = uuid.uuid4().hex
        context_json = json.dumps(context or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        rows: list[tuple[object, ...]] = []
        for item in validation_items:
            if not isinstance(item, dict) or not bool(item.get("tested", True)):
                continue
            action_key = str(item.get("action_key", "")).strip()
            if not action_key:
                continue
            accepted = bool(item.get("accepted", False))
            auto_eligible = bool(item.get("auto_eligible", True))
            user_selected = action_key in selected
            if accepted and user_selected:
                event = "auto_accepted_saved"
            elif accepted and not user_selected:
                event = "user_disabled_saved"
            elif (not accepted) and user_selected and not auto_eligible:
                event = "user_enabled_review_saved"
            elif (not accepted) and user_selected:
                event = "user_overrode_validator_saved"
            elif not auto_eligible:
                event = "manual_review_unselected"
            else:
                event = "validator_rejected_unselected"
            try:
                confidence = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0

            suggestion = suggestions.get(action_key, {})
            try:
                ai_strength = float(suggestion.get("suggested_strength")) if suggestion.get("suggested_strength") is not None else None
            except (TypeError, ValueError):
                ai_strength = None
            try:
                ai_parameter_confidence = float(suggestion.get("confidence")) if suggestion.get("confidence") is not None else None
            except (TypeError, ValueError):
                ai_parameter_confidence = None
            ai_model_id = str(suggestion.get("model_id", ""))
            feature_schema = str(suggestion.get("feature_schema", "") or "parameter_features_v1")
            features = suggestion.get("features", [])
            if not isinstance(features, (list, tuple)):
                features = []
            feature_json = json.dumps(list(features), ensure_ascii=True, separators=(",", ":"), default=float)

            user_strength = None
            if user_selected and bool(item.get("adjustable", False)):
                try:
                    user_strength = float(np.clip(strengths.get(action_key, float(item.get("default_strength", 1.0) or 1.0)), 0.0, 1.0))
                except (TypeError, ValueError):
                    user_strength = None

            validation_json = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            rows.append((
                save_event_id, resolved, qh, action_key, str(item.get("candidate", "")),
                int(accepted), int(user_selected), event, confidence, ai_strength, user_strength,
                ai_model_id, ai_parameter_confidence, feature_json, feature_schema,
                "preference", "user_feedback", validation_json, context_json,
                ALGORITHM_VERSION, str(saved_format or "").lower(),
            ))
        if not rows:
            return 0
        with self.conn:
            self.conn.executemany(
                """INSERT INTO correction_feedback(
                    save_event_id,path,quick_hash,action_key,candidate,validator_accepted,user_selected,
                    decision_event,validation_confidence,ai_suggested_strength,user_strength,
                    ai_parameter_model_id,ai_parameter_confidence,parameter_features_json,
                    parameter_feature_schema,feedback_kind,source_tier,
                    validation_json,context_json,algorithm_version,saved_format
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows,
            )
        return len(rows)

    def all_correction_feedback(self) -> list[dict[str, object]]:
        """Return correction feedback rows for privacy-safe training export."""
        rows = self.conn.execute(
            """SELECT save_event_id,path,quick_hash,action_key,candidate,validator_accepted,user_selected,
                      decision_event,validation_confidence,ai_suggested_strength,user_strength,
                      ai_parameter_model_id,ai_parameter_confidence,parameter_features_json,
                      parameter_feature_schema,feedback_kind,source_tier,
                      validation_json,context_json,algorithm_version,saved_format,created_at
               FROM correction_feedback ORDER BY created_at,id"""
        ).fetchall()
        return [dict(row) for row in rows]

    def correction_feedback_stats(self) -> list[dict[str, object]]:
        """Return compact per-action preference statistics for UI hints."""
        rows = self.conn.execute(
            """SELECT action_key,
                      COUNT(*) AS total_rows,
                      SUM(CASE WHEN decision_event='auto_accepted_saved' THEN 1 ELSE 0 END) AS auto_kept,
                      SUM(CASE WHEN decision_event='user_disabled_saved' THEN 1 ELSE 0 END) AS user_disabled,
                      SUM(CASE WHEN decision_event='user_overrode_validator_saved' THEN 1 ELSE 0 END) AS user_overrode,
                      SUM(CASE WHEN decision_event='user_enabled_review_saved' THEN 1 ELSE 0 END) AS user_enabled_review,
                      SUM(CASE WHEN user_selected=1 AND user_strength IS NOT NULL THEN 1 ELSE 0 END) AS strength_samples,
                      SUM(CASE WHEN user_selected=1 AND user_strength IS NOT NULL AND ai_suggested_strength IS NOT NULL
                                   AND ABS(user_strength-ai_suggested_strength)<=0.0051 THEN 1 ELSE 0 END) AS ai_strength_kept,
                      SUM(CASE WHEN user_selected=1 AND user_strength IS NOT NULL AND ai_suggested_strength IS NOT NULL
                                   AND ABS(user_strength-ai_suggested_strength)>0.0051 THEN 1 ELSE 0 END) AS ai_strength_changed,
                      AVG(CASE WHEN user_selected=1 AND user_strength IS NOT NULL AND ai_suggested_strength IS NOT NULL
                               THEN ABS(user_strength-ai_suggested_strength) END) AS mean_strength_delta
               FROM correction_feedback
               GROUP BY action_key
               ORDER BY action_key"""
        ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            auto_kept = int(row["auto_kept"] or 0)
            disabled = int(row["user_disabled"] or 0)
            overrode = int(row["user_overrode"] or 0)
            enabled_review = int(row["user_enabled_review"] or 0)
            strength_samples = int(row["strength_samples"] or 0)
            ai_strength_kept = int(row["ai_strength_kept"] or 0)
            ai_strength_changed = int(row["ai_strength_changed"] or 0)
            mean_strength_delta = None if row["mean_strength_delta"] is None else float(row["mean_strength_delta"])
            meaningful = auto_kept + disabled + overrode + enabled_review
            selected = auto_kept + overrode + enabled_review
            result.append({
                "action_key": str(row["action_key"]),
                "total_rows": int(row["total_rows"] or 0),
                "meaningful_decisions": meaningful,
                "selected_count": selected,
                "user_disabled": disabled,
                "user_overrode": overrode,
                "user_enabled_review": enabled_review,
                "strength_samples": strength_samples,
                "ai_strength_kept": ai_strength_kept,
                "ai_strength_changed": ai_strength_changed,
                "mean_strength_delta": mean_strength_delta,
                "selected_pct": (selected / meaningful * 100.0) if meaningful else None,
            })
        return result

    def correction_parameter_training_samples(
        self, action_key: str, *, model_id: str = "", feature_schema: str = ""
    ) -> list[dict[str, object]]:
        """Return saved strength examples without mixing incompatible feature schemas."""
        sql = """SELECT user_strength,parameter_features_json,ai_parameter_model_id,
                        parameter_feature_schema,quick_hash,feedback_kind,source_tier,created_at
                 FROM correction_feedback
                 WHERE action_key=? AND user_selected=1 AND user_strength IS NOT NULL
                   AND parameter_features_json IS NOT NULL AND parameter_features_json!='[]'"""
        params: list[object] = [str(action_key)]
        if model_id:
            from photodoctor.ai.parameter_recommender import LEGACY_MODEL_ID
            if str(model_id) == LEGACY_MODEL_ID:
                sql += " AND ai_parameter_model_id IN (?,?)"
                params.extend((str(model_id), str(model_id) + ":legacy"))
            else:
                sql += " AND ai_parameter_model_id=?"
                params.append(str(model_id))
        if feature_schema:
            sql += " AND parameter_feature_schema=?"
            params.append(str(feature_schema))
        sql += " ORDER BY id"
        rows = self.conn.execute(sql, tuple(params)).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            try:
                features = json.loads(row["parameter_features_json"] or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(features, list):
                continue
            result.append({
                "user_strength": float(row["user_strength"]),
                "features": features,
                "model_id": str(row["ai_parameter_model_id"] or ""),
                "feature_schema": str(row["parameter_feature_schema"] or ""),
                "quick_hash": str(row["quick_hash"] or ""),
                "source_group_id": str(row["quick_hash"] or ""),
                "feedback_kind": str(row["feedback_kind"] or "preference"),
                "source_tier": str(row["source_tier"] or "user_feedback"),
                "created_at": str(row["created_at"] or ""),
            })
        return result

    def save_surface_feedback(
        self, path: str | Path, candidate: dict[str, object], user_label: str
    ) -> str:
        if user_label not in SURFACE_FEEDBACK_LABELS:
            raise ValueError(f"Неподдерживаемая метка обратной связи по дефектам поверхности: {user_label}")
        p = Path(path)
        resolved = str(p.resolve())
        qh = self.quick_hash(p)
        signature = surface_candidate_signature(candidate)
        def optional_float(value: object) -> float | None:
            try:
                result = None if value is None else float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return result if result is None or np.isfinite(result) else None

        legacy_ai_label = str(candidate.get("ai_label", ""))
        legacy_ai_confidence = optional_float(candidate.get("ai_confidence"))
        raw_ai_label = str(candidate.get("ai_raw_label", legacy_ai_label))
        raw_ai_confidence = optional_float(candidate.get("ai_raw_confidence", legacy_ai_confidence))
        verification_label = str(candidate.get("verification_label", legacy_ai_label))
        verification_confidence = optional_float(candidate.get("verification_confidence", legacy_ai_confidence))
        values = (
            resolved, qh, signature, ALGORITHM_VERSION,
            float(candidate.get("x", 0.0) or 0.0), float(candidate.get("y", 0.0) or 0.0),
            float(candidate.get("w", 0.0) or 0.0), float(candidate.get("h", 0.0) or 0.0),
            str(candidate.get("polarity", "")), float(candidate.get("strength", 0.0) or 0.0),
            legacy_ai_label, str(candidate.get("ai_model_id", "")), legacy_ai_confidence,
            raw_ai_label, raw_ai_confidence, verification_label, verification_confidence,
            optional_float(candidate.get("candidate_quality")), str(candidate.get("candidate_kind", "")),
            optional_float(candidate.get("context_support")), optional_float(candidate.get("context_risk")),
            user_label,
        )
        with self.conn:
            self.conn.execute(
                """INSERT INTO surface_feedback(
                    path,quick_hash,candidate_signature,algorithm_version,x,y,w,h,polarity,strength,
                    ai_label,ai_model_id,ai_confidence,raw_ai_label,raw_ai_confidence,verification_label,verification_confidence,
                    candidate_quality,candidate_kind,context_support,context_risk,user_label
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path,quick_hash,candidate_signature) DO UPDATE SET
                    algorithm_version=excluded.algorithm_version, x=excluded.x, y=excluded.y,
                    w=excluded.w, h=excluded.h, polarity=excluded.polarity, strength=excluded.strength,
                    ai_label=excluded.ai_label, ai_model_id=excluded.ai_model_id, ai_confidence=excluded.ai_confidence,
                    raw_ai_label=excluded.raw_ai_label, raw_ai_confidence=excluded.raw_ai_confidence,
                    verification_label=excluded.verification_label, verification_confidence=excluded.verification_confidence,
                    candidate_quality=excluded.candidate_quality, candidate_kind=excluded.candidate_kind,
                    context_support=excluded.context_support, context_risk=excluded.context_risk,
                    user_label=excluded.user_label, updated_at=CURRENT_TIMESTAMP""",
                values,
            )
        return signature

    def delete_surface_feedback(self, path: str | Path, candidate: dict[str, object]) -> None:
        p = Path(path)
        try:
            resolved = str(p.resolve())
            qh = self.quick_hash(p)
        except OSError:
            return
        signature = surface_candidate_signature(candidate)
        with self.conn:
            self.conn.execute(
                "DELETE FROM surface_feedback WHERE path=? AND quick_hash=? AND candidate_signature=?",
                (resolved, qh, signature),
            )

    def save(self, result: AnalysisResult) -> None:
        p = result.image.path
        precision = "normal"
        profile_metric = result.metrics.get("analysis_profile")
        if profile_metric is not None and isinstance(profile_metric.raw_value, dict):
            precision = str(profile_metric.raw_value.get("key", "normal"))
        cache_algorithm_version = _cache_algorithm_version(precision)
        st = p.stat()
        qh = self.quick_hash(p)
        with self.conn:
            self.conn.execute(
                """INSERT INTO photo(path,size,modified_ns,quick_hash,width,height,format,algorithm_version,normalization_version)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(path) DO UPDATE SET
                  size=excluded.size, modified_ns=excluded.modified_ns, quick_hash=excluded.quick_hash,
                  width=excluded.width, height=excluded.height, format=excluded.format,
                  analyzed_at=CURRENT_TIMESTAMP, algorithm_version=excluded.algorithm_version,
                  normalization_version=excluded.normalization_version""",
                (str(p.resolve()), st.st_size, st.st_mtime_ns, qh, result.image.width, result.image.height,
                 result.image.format, cache_algorithm_version, NORMALIZATION_VERSION),
            )
            photo_id = self.conn.execute("SELECT id FROM photo WHERE path=?", (str(p.resolve()),)).fetchone()[0]
            self.conn.execute("DELETE FROM metric WHERE photo_id=?", (photo_id,))
            for metric in result.metrics.values():
                self.conn.execute(
                    "INSERT INTO metric(photo_id,name,raw_json,normalized_value,confidence,scale,region,diagnostic) VALUES(?,?,?,?,?,?,?,?)",
                    (photo_id, metric.name, json.dumps(metric.raw_value, ensure_ascii=False), metric.normalized_value,
                     metric.confidence, metric.scale, metric.region, metric.diagnostic),
                )
