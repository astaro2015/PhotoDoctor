from __future__ import annotations

from threading import Event

import pytest
import numpy as np
from PIL import Image

from photodoctor.core.batch import analyze_batch, collect_images
from photodoctor.core.database import AnalysisDatabase
from photodoctor.core.service import analyze_file


def _photo(path, value=128):
    Image.fromarray(np.full((64, 96, 3), value, np.uint8), "RGB").save(path)


def test_database_save_and_cache(tmp_path):
    photo = tmp_path / "a.png"
    _photo(photo)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        assert not db.is_current(photo)
        db.save(analyze_file(photo))
        assert db.is_current(photo)
        assert db.conn.execute("SELECT COUNT(*) FROM metric").fetchone()[0] >= 8


def test_batch_fault_isolation(tmp_path):
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.png"
    _photo(good)
    bad.write_bytes(b"not an image")
    with AnalysisDatabase(tmp_path / "a.db") as db:
        result = analyze_batch([good, bad], db)
    assert result.processed == 2
    assert result.succeeded == 1
    assert result.failed == 1


def test_batch_cache_hit(tmp_path):
    p = tmp_path / "one.png"
    _photo(p)
    with AnalysisDatabase(tmp_path / "a.db") as db:
        first = analyze_batch([p], db)
        second = analyze_batch([p], db)
    assert first.succeeded == 1
    assert second.cached == 1


def test_cancel_before_first_item(tmp_path):
    files = []
    for i in range(3):
        p = tmp_path / f"{i}.png"
        _photo(p, 50 + i)
        files.append(p)
    ev = Event(); ev.set()
    result = analyze_batch(files, cancel_event=ev)
    assert result.cancelled
    assert result.processed == 0


def test_collect_recursive(tmp_path):
    sub = tmp_path / "sub"; sub.mkdir()
    _photo(tmp_path / "a.png")
    _photo(sub / "b.jpg")
    (sub / "x.txt").write_text("x")
    assert len(collect_images(tmp_path, recursive=False)) == 1
    assert len(collect_images(tmp_path, recursive=True)) == 2




def test_collect_images_uses_all_supported_input_extensions(tmp_path):
    from photodoctor.core.image_formats import SUPPORTED_INPUT_EXTENSIONS

    for index, suffix in enumerate(sorted(SUPPORTED_INPUT_EXTENSIONS)):
        (tmp_path / f"photo_{index}{suffix}").write_bytes(b"placeholder")
    (tmp_path / "not_photo.gif").write_bytes(b"placeholder")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    found = collect_images(tmp_path, recursive=False)
    assert {p.suffix.lower() for p in found} == set(SUPPORTED_INPUT_EXTENSIONS)

def test_database_can_load_current_cached_metrics(tmp_path):
    photo = tmp_path / "cached.png"
    _photo(photo, 110)
    with AnalysisDatabase(tmp_path / "a.db") as db:
        result = analyze_file(photo)
        db.save(result)
        metrics = db.load_metrics(photo)
    assert metrics is not None
    assert "brightness" in metrics
    assert metrics["brightness"].normalized_value is not None
    assert isinstance(metrics["histogram"].raw_value, dict)


def test_batch_pause_can_be_cancelled_without_processing(tmp_path):
    files = []
    for i in range(2):
        p = tmp_path / f"pause_{i}.png"
        _photo(p, 70 + i)
        files.append(p)
    pause = Event(); pause.set()
    cancel = Event(); cancel.set()
    result = analyze_batch(files, cancel_event=cancel, pause_event=pause)
    assert result.cancelled
    assert result.processed == 0


def test_cache_distinguishes_analysis_precision(tmp_path):
    photo = tmp_path / "precision.png"
    from PIL import Image
    Image.new("RGB", (96, 96), (120, 130, 140)).save(photo)
    db_path = tmp_path / "precision.sqlite3"
    with AnalysisDatabase(db_path) as db:
        normal = analyze_file(photo, precision="normal")
        db.save(normal)
        assert db.is_current(photo, precision="normal")
        assert not db.is_current(photo, precision="precise")
        assert not db.is_current(photo, precision="maximum")
        assert db.load_metrics(photo, precision="normal") is not None
        assert db.load_metrics(photo, precision="precise") is None
        assert db.load_metrics(photo, precision="maximum") is None


def test_database_aggregates_ai_consistency_history(tmp_path):
    from photodoctor.core.models import MetricResult
    p1 = tmp_path / "ai1.png"; p2 = tmp_path / "ai2.png"
    _photo(p1, 90); _photo(p2, 100)
    db_path = tmp_path / "ai_history.sqlite3"
    with AnalysisDatabase(db_path) as db:
        for path, statuses in [(p1, ["agree", "refine"]), (p2, ["contradict", "low_confidence"])]:
            result = analyze_file(path)
            result.metrics["ai_crosscheck"] = MetricResult("ai_crosscheck", {
                "comparisons": [
                    {"task": "face_quality", "model_id": "face-v1", "region": f"face_{i+1}", "status": status}
                    for i, status in enumerate(statuses)
                ]
            }, None, 0.8, "ai", region="ai")
            db.save(result)
        stats = db.ai_consistency_stats()
    assert len(stats) == 1
    row = stats[0]
    assert row["photo_count"] == 2
    assert row["comparison_count"] == 4
    assert row["agree"] == 1 and row["refine"] == 1
    assert row["contradict"] == 1 and row["low_confidence"] == 1
    assert row["compatibility_pct"] == pytest.approx(66.666666, rel=1e-4)


def test_ai_consistency_history_ignores_other_precision(tmp_path):
    from photodoctor.core.models import MetricResult
    photo = tmp_path / "ai_precision.png"; _photo(photo, 120)
    with AnalysisDatabase(tmp_path / "ai_precision.sqlite3") as db:
        result = analyze_file(photo, precision="precise")
        result.metrics["ai_crosscheck"] = MetricResult("ai_crosscheck", {"comparisons": [
            {"task": "blur_refinement", "model_id": "blur-v1", "region": "global", "status": "agree"}
        ]}, None, 0.8, "ai", region="ai")
        db.save(result)
        assert db.ai_consistency_stats(precision="normal") == []
        assert db.ai_consistency_stats(precision="precise")[0]["agree"] == 1


def test_schema_v9_adds_manual_vision_overrides_without_breaking_existing_database(tmp_path):
    import sqlite3
    db_path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta(key,value) VALUES('schema_version','3');
        CREATE TABLE surface_feedback(
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            quick_hash TEXT NOT NULL,
            candidate_signature TEXT NOT NULL,
            algorithm_version TEXT NOT NULL,
            x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
            polarity TEXT NOT NULL DEFAULT '', strength REAL NOT NULL DEFAULT 0,
            ai_label TEXT NOT NULL DEFAULT '', ai_model_id TEXT NOT NULL DEFAULT '',
            ai_confidence REAL, user_label TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(path, quick_hash, candidate_signature)
        );
        """
    )
    conn.execute(
        "INSERT INTO surface_feedback(path,quick_hash,candidate_signature,algorithm_version,x,y,w,h,user_label) "
        "VALUES('legacy.jpg','abc','sig','0.3.4',0.1,0.1,0.2,0.2,'defect')"
    )
    conn.commit(); conn.close()

    with AnalysisDatabase(db_path) as db:
        version = db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        tables = {row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        count = db.conn.execute("SELECT COUNT(*) FROM surface_feedback").fetchone()[0]
    assert version == "9"
    assert "file_fingerprint" in tables
    assert "correction_feedback" in tables
    assert "manual_vision_override" in tables
    assert count == 1


def test_manual_vision_override_roundtrip_and_hash_binding(tmp_path):
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"first-image-payload")
    db_path = tmp_path / "manual.sqlite3"
    faces = [{"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]
    eyes = [{"x": 0.2, "y": 0.3, "w": 0.05, "h": 0.04}]
    with AnalysisDatabase(db_path) as db:
        db.save_manual_vision_overrides(image, faces, eyes)
        loaded = db.load_manual_vision_overrides(image)
    assert loaded["faces"] == faces
    assert loaded["eyes"] == eyes
    image.write_bytes(b"changed-payload")
    with AnalysisDatabase(db_path) as db:
        assert db.load_manual_vision_overrides(image) == {"faces": [], "eyes": []}


def test_manual_vision_override_invalidates_cache_and_batch_reuses_it(tmp_path):
    import numpy as np
    from PIL import Image
    from photodoctor.core.batch import analyze_batch

    image = tmp_path / "manual-batch.png"
    Image.fromarray(np.full((240, 320, 3), 128, np.uint8), "RGB").save(image)
    db_path = tmp_path / "analysis.sqlite3"
    with AnalysisDatabase(db_path) as db:
        first = analyze_batch([image], db, precision="fast")
        assert first.succeeded == 1
        assert db.is_current(image, precision="fast")
        db.save_manual_vision_overrides(
            image,
            [{"x": 0.20, "y": 0.15, "w": 0.45, "h": 0.55}],
            [{"x": 0.33, "y": 0.30, "w": 0.08, "h": 0.08}],
        )
        assert not db.is_current(image, precision="fast")
        second = analyze_batch([image], db, precision="fast")
        assert second.succeeded == 1
        metrics = db.load_metrics(image, precision="fast")
        assert metrics is not None
        assert metrics["faces"].raw_value["faces"][0]["source"] == "manual"
        assert metrics["eyes"].raw_value["eyes"][0]["source"] == "manual"


def test_database_load_current_result_requires_current_fingerprint(tmp_path):
    from photodoctor.core.loader import load_image

    photo = tmp_path / "cached-result.png"
    _photo(photo, 117)
    db_path = tmp_path / "cached-result.sqlite3"
    loaded = load_image(photo)
    with AnalysisDatabase(db_path) as db:
        result = analyze_file(photo, precision="fast", loaded_image=loaded)
        db.save(result)
        cached = db.load_current_result(loaded.info, precision="fast")
        assert cached is not None
        assert cached.image == loaded.info
        assert cached.metrics.keys() == result.metrics.keys()
        # SQLite persists raw metric payloads as JSON. Tuple-valued informational
        # model-spec fields therefore round-trip as lists, while the JSON meaning
        # and every executable/decision payload remain identical.
        import json
        fresh_json = json.loads(json.dumps({k: v.to_dict() for k, v in result.metrics.items()}, ensure_ascii=False))
        cached_json = json.loads(json.dumps({k: v.to_dict() for k, v in cached.metrics.items()}, ensure_ascii=False))
        assert cached_json == fresh_json

        # A source change must fail closed instead of serving stale metrics.
        _photo(photo, 118)
        changed = load_image(photo)
        assert db.load_current_result(changed.info, precision="fast") is None


def test_analysis_cache_invalidates_when_external_model_state_changes(tmp_path, monkeypatch):
    from photodoctor.core.loader import load_image

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setenv("PHOTODOCTOR_MODEL_DIR", str(model_dir))

    photo = tmp_path / "model-cache.png"
    _photo(photo, 119)
    loaded = load_image(photo)
    with AnalysisDatabase(tmp_path / "model-cache.sqlite3") as db:
        result = analyze_file(photo, precision="fast", loaded_image=loaded)
        db.save(result)
        assert db.load_current_result(loaded.info, precision="fast") is not None

        # Installing a model must invalidate any AI-dependent cached analysis.
        model = model_dir / "new-restoration.onnx"
        model.write_bytes(b"fake-model-state-v1")
        assert db.load_current_result(loaded.info, precision="fast") is None

        # Once the same analysis is saved under the new model-state key it is
        # current again, until that external state changes once more.
        db.save(result)
        assert db.load_current_result(loaded.info, precision="fast") is not None
        manifest = model_dir / "new-restoration.onnx.json"
        manifest.write_text('{"state":"ready"}', encoding="utf-8")
        assert db.load_current_result(loaded.info, precision="fast") is None


def test_database_load_current_result_respects_manual_override_invalidation(tmp_path):
    from photodoctor.core.loader import load_image

    photo = tmp_path / "cached-manual.png"
    _photo(photo, 122)
    db_path = tmp_path / "cached-manual.sqlite3"
    loaded = load_image(photo)
    with AnalysisDatabase(db_path) as db:
        db.save(analyze_file(photo, precision="fast", loaded_image=loaded))
        assert db.load_current_result(loaded.info, precision="fast") is not None
        db.save_manual_vision_overrides(photo, [{"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}], [])
        assert db.load_current_result(loaded.info, precision="fast") is None


def test_current_cache_can_reuse_precomputed_quick_hash_without_rereading(tmp_path, monkeypatch):
    from photodoctor.core.loader import load_image

    photo = tmp_path / "cached-hash.png"
    _photo(photo, 123)
    loaded = load_image(photo)
    with AnalysisDatabase(tmp_path / "cached-hash.sqlite3") as db:
        db.save(analyze_file(photo, precision="fast", loaded_image=loaded))
        quick_hash = db.quick_hash(photo)

        calls = 0
        original = AnalysisDatabase.quick_hash

        def counted(path, block_size=65536):
            nonlocal calls
            calls += 1
            return original(path, block_size)

        monkeypatch.setattr(AnalysisDatabase, "quick_hash", staticmethod(counted))
        manual = db.load_manual_vision_overrides(photo, quick_hash=quick_hash)
        cached = db.load_current_result(loaded.info, precision="fast", quick_hash=quick_hash)
        assert manual == {"faces": [], "eyes": []}
        assert cached is not None
        assert calls == 0


def test_is_current_still_checks_content_when_size_and_mtime_match(tmp_path):
    import os
    from photodoctor.core.loader import load_image

    photo = tmp_path / "same-stat.png"
    _photo(photo, 124)
    loaded = load_image(photo)
    db_path = tmp_path / "same-stat.sqlite3"
    with AnalysisDatabase(db_path) as db:
        db.save(analyze_file(photo, precision="fast", loaded_image=loaded))
        original_stat = photo.stat()
        original_size = original_stat.st_size
        original_mtime_ns = original_stat.st_mtime_ns
        assert db.is_current(photo, precision="fast")

        payload = bytearray(photo.read_bytes())
        # PNG ancillary/compressed payload: flip one byte near the middle while
        # preserving file length, then restore the old mtime exactly. The quick
        # hash must still reject it. Avoid the first/last block ambiguity by
        # using a larger opaque file for the actual fingerprint check below.
        opaque = tmp_path / "same-stat.bin"
        opaque.write_bytes(bytes(range(256)) * 1024)
        st = opaque.stat()
        resolved = str(opaque.resolve())
        qh = db.quick_hash(opaque)
        with db.conn:
            db.conn.execute(
                "INSERT OR REPLACE INTO photo(path,size,modified_ns,quick_hash,width,height,format,algorithm_version,normalization_version) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (resolved, st.st_size, st.st_mtime_ns, qh, 1, 1, "BIN", __import__('photodoctor.core.database', fromlist=['_cache_algorithm_version'])._cache_algorithm_version('fast'), __import__('photodoctor.core.database', fromlist=['NORMALIZATION_VERSION']).NORMALIZATION_VERSION),
            )
        changed = bytearray(opaque.read_bytes())
        changed[10] ^= 0x5A
        opaque.write_bytes(changed)
        os.utime(opaque, ns=(st.st_atime_ns, st.st_mtime_ns))
        assert opaque.stat().st_size == st.st_size
        assert opaque.stat().st_mtime_ns == st.st_mtime_ns
        assert not db.is_current(opaque, precision="fast")


def test_manual_override_lookup_skips_hash_when_path_has_no_override(tmp_path, monkeypatch):
    photo = tmp_path / "no-manual.png"
    _photo(photo, 125)
    with AnalysisDatabase(tmp_path / "no-manual.sqlite3") as db:
        calls = 0
        original = AnalysisDatabase.quick_hash

        def counted(path, block_size=65536):
            nonlocal calls
            calls += 1
            return original(path, block_size)

        monkeypatch.setattr(AnalysisDatabase, "quick_hash", staticmethod(counted))
        assert db.load_manual_vision_overrides(photo) == {"faces": [], "eyes": []}
        assert calls == 0


def test_analysis_cache_model_signature_detects_same_stat_content_replacement(tmp_path, monkeypatch):
    import os
    from photodoctor.core.database import _analysis_model_state_signature

    model_dir = tmp_path / "models-same-stat"
    model_dir.mkdir()
    monkeypatch.setenv("PHOTODOCTOR_MODEL_DIR", str(model_dir))
    model = model_dir / "manual.onnx"
    model.write_bytes(b"A" * (256 * 1024))
    st = model.stat()
    before = _analysis_model_state_signature()

    model.write_bytes(b"B" * (256 * 1024))
    os.utime(model, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert model.stat().st_size == st.st_size
    assert model.stat().st_mtime_ns == st.st_mtime_ns
    assert _analysis_model_state_signature() != before
