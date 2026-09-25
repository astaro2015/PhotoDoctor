from photodoctor.core.histogram_diagnostics import build_histogram_diagnostic
from photodoctor.core.models import MetricResult


def m(name, raw, score=None, conf=0.9):
    return MetricResult(name, raw, score, conf, "test")


def test_normal_linear_histogram_explains_no_global_lift():
    metrics = {
        "brightness": m("brightness", 0.31, 100, 0.85),
        "histogram": m(
            "histogram",
            {"bins": [1, 2, 3], "p1": 0.005, "p5": 0.009, "p50": 0.379, "p95": 0.691, "p99": 0.747},
            None,
            0.98,
        ),
        "shadow_clipping": m("shadow_clipping", 0.12, 99),
        "highlight_clipping": m("highlight_clipping", 0.03, 99),
        "highlight_context": m("highlight_context", {"unexplained_clip_pct": 0.01}, None, 0.8),
        "faces": m("faces", {"face_count": 1, "faces": [{"brightness_linear": 0.30}]}, 90),
        "main_subject": m("main_subject", {"subject_kind": "person", "confidence": 0.9, "face_indices": [0]}, None),
    }
    d = build_histogram_diagnostic(metrics)
    assert d.status == "normal"
    assert d.headline == "Экспозиция: нормальная"
    assert "не требуют глобального осветления" in d.summary
    assert d.percentiles["p50"] == 0.379


def test_dark_histogram_matches_decision_engine_exposure_fix():
    metrics = {
        "brightness": m("brightness", 0.075, 50, 0.85),
        "histogram": m("histogram", {"p5": 0.003, "p50": 0.060, "p95": 0.31}, None, 0.98),
        "shadow_clipping": m("shadow_clipping", 1.4, 88),
        "highlight_clipping": m("highlight_clipping", 0.0, 100),
        "highlight_context": m("highlight_context", {"unexplained_clip_pct": 0.0}, None, 0.8),
    }
    d = build_histogram_diagnostic(metrics)
    assert d.status == "dark"
    assert d.exposure.decision == "fix"
    assert "мягкий подъём" in d.summary


def test_unexplained_highlight_clipping_warns_without_recommending_lift():
    metrics = {
        "brightness": m("brightness", 0.34, 100, 0.85),
        "histogram": m("histogram", {"p5": 0.04, "p50": 0.33, "p95": 0.88}, None, 0.98),
        "shadow_clipping": m("shadow_clipping", 0.05, 100),
        "highlight_clipping": m("highlight_clipping", 3.2, 75),
        "highlight_context": m("highlight_context", {"unexplained_clip_pct": 2.4}, None, 0.8),
    }
    d = build_histogram_diagnostic(metrics)
    assert d.status == "highlights"
    assert "света требуют внимания" in d.headline.lower()
    assert "осветление не требуется" in d.summary


def test_histogram_percentiles_are_clamped_for_drawing():
    metrics = {
        "brightness": m("brightness", 0.3, 100),
        "histogram": m("histogram", {"p5": -1, "p50": 0.4, "p95": 2}, None),
    }
    d = build_histogram_diagnostic(metrics)
    assert d.percentiles == {"p5": 0.0, "p50": 0.4, "p95": 1.0}
