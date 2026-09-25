from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from photodoctor.core.database import AnalysisDatabase
from photodoctor.core.models import MetricResult
from photodoctor.core.series import (
    PHASH_METHOD,
    group_photo_series,
    perceptual_hash64,
    phash_similarity,
    technical_series_score,
)
from photodoctor.core.service import analyze_file


def _metric(name: str, score: float | None, raw=0.0, confidence: float = 0.9) -> MetricResult:
    return MetricResult(name, raw, score, confidence, "test")


def _score_metrics(sharpness: float, noise: float = 90.0, red_eye: float = 100.0):
    return {
        "laplacian": _metric("laplacian", sharpness),
        "tenengrad": _metric("tenengrad", sharpness),
        "brightness": _metric("brightness", 60.0, 0.48),
        "shadow_clipping": _metric("shadow_clipping", 96.0),
        "highlight_clipping": _metric("highlight_clipping", 96.0),
        "contrast": _metric("contrast", 75.0),
        "noise": _metric("noise", noise),
        "jpeg_artifacts": _metric("jpeg_artifacts", 95.0),
        "edge_artifacts": _metric("edge_artifacts", 95.0),
        "posterization": _metric("posterization", 95.0),
        "local_contrast": _metric("local_contrast", 80.0),
        "red_eye": _metric("red_eye", red_eye),
    }


def _photo(path: Path, offset: int = 0, variant: bool = False) -> None:
    y, x = np.mgrid[0:96, 0:128]
    base = (x * 1.3 + y * 0.7 + offset) % 256
    rgb = np.stack([base, np.roll(base, 3, axis=1), np.roll(base, 5, axis=0)], axis=-1).astype(np.uint8)
    if variant:
        rgb[25:65, 45:85] = 255 - rgb[25:65, 45:85]
    Image.fromarray(rgb, "RGB").save(path)


def _set_capture(result, stamp: str) -> None:
    metric = result.metrics["exif_context"]
    raw = dict(metric.raw_value)
    raw["available"] = True
    raw["datetime_original"] = stamp
    result.metrics["exif_context"] = MetricResult(
        "exif_context", raw, None, max(metric.confidence, 0.6), "metadata", region="metadata"
    )


def test_technical_series_score_rewards_sharpness_and_cleaner_image():
    weak, _ = technical_series_score(_score_metrics(35.0, noise=55.0, red_eye=65.0))
    strong, _ = technical_series_score(_score_metrics(88.0, noise=92.0, red_eye=100.0))
    assert strong > weak + 10.0


def test_phash_is_stable_for_small_tone_change(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _photo(a, 0)
    _photo(b, 4)
    similarity = phash_similarity(perceptual_hash64(a), perceptual_hash64(b))
    assert similarity >= 0.80


def test_series_groups_close_similar_exif_frames_and_selects_best(tmp_path):
    files = [tmp_path / f"burst_{i}.png" for i in range(3)]
    for i, path in enumerate(files):
        _photo(path, i * 2)

    with AnalysisDatabase(tmp_path / "series.sqlite3") as db:
        for i, path in enumerate(files):
            result = analyze_file(path)
            _set_capture(result, f"2026:09:18 12:00:0{i}")
            # Force an unambiguous technical ranking without changing the pixels
            # used by the series similarity detector.
            result.metrics.update(_score_metrics(40.0 + i * 25.0, noise=70.0 + i * 10.0))
            db.save(result)
        summary = group_photo_series(files, db)

    assert summary.series_count == 1
    assert summary.grouped_photos == 3
    group = summary.groups[0]
    assert group.best_path == files[2]
    assert [m.rank for m in group.members] == [1, 2, 3]
    assert group.members[0].relative_score == 100.0
    assert group.confidence > 0.4
    assert group.members[0].reasons


def test_series_does_not_group_visually_different_frames_even_when_time_is_close(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _photo(a, 0)
    # Deliberately unrelated random-looking frame.
    rng = np.random.default_rng(1234)
    Image.fromarray(rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8), "RGB").save(b)

    with AnalysisDatabase(tmp_path / "series.sqlite3") as db:
        for path, stamp in [(a, "2026:09:18 12:00:00"), (b, "2026:09:18 12:00:01")]:
            result = analyze_file(path)
            _set_capture(result, stamp)
            db.save(result)
        summary = group_photo_series([a, b], db)

    assert summary.series_count == 0
    assert summary.singleton_photos == 2


def test_series_fingerprint_is_cached_by_quick_hash(tmp_path):
    p = tmp_path / "cached.png"
    _photo(p)
    with AnalysisDatabase(tmp_path / "series.sqlite3") as db:
        result = analyze_file(p)
        _set_capture(result, "2026:09:18 12:00:00")
        db.save(result)
        group_photo_series([p], db)
        cached = db.load_file_fingerprint(p, PHASH_METHOD)
        assert cached is not None and len(cached) == 16
        count_before = db.conn.execute("SELECT COUNT(*) FROM file_fingerprint").fetchone()[0]
        group_photo_series([p], db)
        count_after = db.conn.execute("SELECT COUNT(*) FROM file_fingerprint").fetchone()[0]
    assert count_before == count_after == 1


def test_series_rejects_same_phash_when_tone_is_radically_different(tmp_path):
    dark = tmp_path / "dark.png"
    bright = tmp_path / "bright.png"
    Image.fromarray(np.full((96, 128, 3), 18, dtype=np.uint8), "RGB").save(dark)
    Image.fromarray(np.full((96, 128, 3), 235, dtype=np.uint8), "RGB").save(bright)
    # Flat frames can have identical pHash; the tone guard must prevent a false series.
    assert phash_similarity(perceptual_hash64(dark), perceptual_hash64(bright)) > 0.95
    with AnalysisDatabase(tmp_path / "tone_guard.sqlite3") as db:
        for path, stamp in [(dark, "2026:09:18 12:00:00"), (bright, "2026:09:18 12:00:01")]:
            result = analyze_file(path)
            _set_capture(result, stamp)
            db.save(result)
        summary = group_photo_series([dark, bright], db)
    assert summary.series_count == 0


def test_series_ignores_stale_metrics_when_file_changed_after_batch(tmp_path):
    p = tmp_path / "changed.png"
    _photo(p)
    with AnalysisDatabase(tmp_path / "stale.sqlite3") as db:
        result = analyze_file(p)
        _set_capture(result, "2026:09:18 12:00:00")
        db.save(result)
        Image.fromarray(np.zeros((96, 128, 3), dtype=np.uint8), "RGB").save(p)
        summary = group_photo_series([p], db)
    assert summary.total_photos == 0
    assert summary.series_count == 0
