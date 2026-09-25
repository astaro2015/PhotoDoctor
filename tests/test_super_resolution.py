from __future__ import annotations

from pathlib import Path

import numpy as np

from photodoctor.core.models import ImageInfo, MetricResult
from photodoctor.core.loader import LoadedImage
from photodoctor.core.super_resolution import analyze_super_resolution_need, apply_super_resolution_x2
from photodoctor.core.validator import validate_recommendations, preview_action_available, apply_selected_preview


def _metric(name: str, value: float | None, raw: object = 0.0, confidence: float = 0.8) -> MetricResult:
    return MetricResult(name, raw, value, confidence, "test")


def _make_metrics(*, sharpness: float = 56.0, noise: float = 66.0, jpeg: float = 62.0, face_count: int = 0) -> dict[str, MetricResult]:
    faces_raw = {
        "face_count": face_count,
        "faces": ([{"x": 16, "y": 16, "w": 24, "h": 24}] if face_count else []),
    }
    return {
        "local_sharpness": _metric("local_sharpness", sharpness, {"soft_area_pct": 18.0}),
        "noise": _metric("noise", noise, {"noise_score": noise}),
        "jpeg_artifacts": _metric("jpeg_artifacts", jpeg, {"artifact_score": jpeg}),
        "faces": _metric("faces", 70.0 if face_count else 100.0, faces_raw),
        "eyes": _metric("eyes", 70.0 if face_count else 100.0, {"eye_count": 0}),
        "semantic_context": _metric("semantic_context", None, {"classification": "unknown", "face_count": face_count}, confidence=0.7),
    }


def test_super_resolution_detector_recommends_small_soft_input():
    y, x = np.indices((720, 960))
    base = (((x // 24 + y // 24) % 2) * 60 + 88).astype(np.uint8)
    rgb = np.dstack([base, np.clip(base + 8, 0, 255), np.clip(base + 3, 0, 255)]).astype(np.uint8)
    metrics = _make_metrics(sharpness=52.0, noise=63.0, jpeg=58.0, face_count=1)
    advice = analyze_super_resolution_need(rgb, metrics)
    assert advice.recommend is True
    assert advice.suggested_scale == 2
    assert advice.need_score >= 56.0
    assert advice.face_protection >= 0.80



def test_super_resolution_detector_skips_flat_low_information_input():
    rgb = np.full((720, 960, 3), 118, dtype=np.uint8)
    metrics = _make_metrics(sharpness=40.0, noise=50.0, jpeg=50.0, face_count=0)
    advice = analyze_super_resolution_need(rgb, metrics)
    assert advice.recommend is False
    assert advice.need_score < 30.0
    assert "почти не содержит структуры" in advice.reasoning[0]

def test_super_resolution_detector_skips_large_modern_input():
    rgb = np.full((3000, 4000, 3), 118, dtype=np.uint8)
    metrics = _make_metrics(sharpness=86.0, noise=88.0, jpeg=92.0, face_count=0)
    advice = analyze_super_resolution_need(rgb, metrics)
    assert advice.recommend is False
    assert advice.need_score < 40.0


def test_apply_super_resolution_x2_doubles_dimensions():
    rgb = np.zeros((40, 50, 3), dtype=np.uint8)
    rgb[10:25, 12:32] = [170, 120, 90]
    out = apply_super_resolution_x2(
        rgb,
        strength=0.75,
        face_boxes=[(12, 10, 20, 15)],
        face_protection=0.9,
    )
    assert out.shape == (80, 100, 3)
    assert out.dtype == np.uint8


def test_validator_exposes_and_applies_super_resolution_preview():
    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[8:40, 12:52] = [140, 120, 105]
    info = ImageInfo(path=Path("dummy.jpg"), width=64, height=48, mode="RGB", format="JPEG")
    image = LoadedImage(info=info, srgb=rgb)
    metrics = _make_metrics(sharpness=54.0, noise=66.0, jpeg=60.0, face_count=1)
    advice = analyze_super_resolution_need(rgb, metrics)
    metrics["super_resolution"] = MetricResult(
        "super_resolution", advice.to_raw(), advice.need_score, advice.confidence, "source"
    )
    metrics["decision_plan"] = MetricResult(
        "decision_plan",
        {"items": [{"key": "super_resolution", "decision": "review"}]},
        None,
        advice.confidence,
        "source",
    )
    items = validate_recommendations(image, metrics, precision="normal")
    sr_item = next(item for item in items if item.action_key == "super_resolution")
    item_dict = sr_item.to_dict()
    assert preview_action_available(item_dict) is True
    preview = apply_selected_preview(rgb, [item_dict], {"super_resolution"})
    assert preview.shape == (96, 128, 3)


def test_validator_super_resolution_is_lazy(monkeypatch):
    rgb = np.zeros((96, 128, 3), dtype=np.uint8)
    y, x = np.indices((96, 128))
    rgb[..., 0] = ((x // 8 + y // 8) % 2 * 100 + 60).astype(np.uint8)
    rgb[..., 1] = rgb[..., 0]
    rgb[..., 2] = rgb[..., 0]
    info = ImageInfo(path=Path("lazy.jpg"), width=128, height=96, mode="RGB", format="JPEG")
    image = LoadedImage(info=info, srgb=rgb)
    metrics = _make_metrics(sharpness=54.0, noise=66.0, jpeg=60.0, face_count=0)
    advice = analyze_super_resolution_need(rgb, metrics)
    metrics["super_resolution"] = MetricResult("super_resolution", advice.to_raw(), advice.need_score, advice.confidence, "source")
    metrics["decision_plan"] = MetricResult(
        "decision_plan", {"items": [{"key": "super_resolution", "decision": "review"}]}, None, advice.confidence, "source"
    )

    def _must_not_run(*_args, **_kwargs):
        raise AssertionError("heavy x2 preview must be lazy during validate_recommendations")

    monkeypatch.setattr("photodoctor.core.validator.apply_super_resolution_x2", _must_not_run)
    items = validate_recommendations(image, metrics, precision="normal")
    sr_item = next(item for item in items if item.action_key == "super_resolution")
    assert sr_item.preview_available is True
    assert sr_item.auto_eligible is False


def test_normalized_face_box_really_protects_face_region():
    rng = np.random.default_rng(42)
    rgb = rng.integers(20, 235, size=(48, 64, 3), dtype=np.uint8)
    bicubic = __import__("cv2").resize(rgb, (128, 96), interpolation=__import__("cv2").INTER_CUBIC)
    unguarded = apply_super_resolution_x2(
        rgb, strength=1.0, face_boxes=[(0.25, 0.25, 0.50, 0.50)], face_protection=0.0,
        deblock_strength=0.0, denoise_strength=0.0, detail_strength=1.0,
    )
    guarded = apply_super_resolution_x2(
        rgb, strength=1.0, face_boxes=[(0.25, 0.25, 0.50, 0.50)], face_protection=1.0,
        deblock_strength=0.0, denoise_strength=0.0, detail_strength=1.0,
    )
    # Normalized source box maps to x=32..96, y=24..72 in the x2 output.
    box = np.s_[24:72, 32:96]
    unguarded_delta = float(np.mean(np.abs(unguarded[box].astype(np.float32) - bicubic[box].astype(np.float32))))
    guarded_delta = float(np.mean(np.abs(guarded[box].astype(np.float32) - bicubic[box].astype(np.float32))))
    assert guarded_delta < unguarded_delta * 0.65


def test_auto_backend_uses_installed_model_when_available(monkeypatch):
    class FakeBackend:
        def __init__(self):
            self.called = False
        def upscale_x2(self, rgb):
            self.called = True
            return np.full((rgb.shape[0] * 2, rgb.shape[1] * 2, 3), 210, dtype=np.uint8)

    fake = FakeBackend()
    monkeypatch.setattr("photodoctor.ai.sr_runtime.load_installed_sr_backend", lambda: (fake, None))
    rgb = np.full((24, 32, 3), 90, dtype=np.uint8)
    auto = apply_super_resolution_x2(
        rgb, strength=1.0, face_protection=0.0, deblock_strength=0.0, denoise_strength=0.0,
        detail_strength=0.0, backend_preference="auto",
    )
    classical = apply_super_resolution_x2(
        rgb, strength=1.0, face_protection=0.0, deblock_strength=0.0, denoise_strength=0.0,
        detail_strength=0.0, backend_preference="classical",
    )
    assert fake.called is True
    assert float(auto.mean()) > float(classical.mean()) + 20.0


def test_auto_backend_failure_falls_back_to_classical(monkeypatch):
    monkeypatch.setattr(
        "photodoctor.ai.sr_runtime.load_installed_sr_backend",
        lambda: (_ for _ in ()).throw(RuntimeError("runtime boom")),
    )
    rng = np.random.default_rng(7)
    rgb = rng.integers(20, 220, size=(30, 34, 3), dtype=np.uint8)
    auto = apply_super_resolution_x2(
        rgb, strength=0.7, face_protection=0.0, deblock_strength=0.0, denoise_strength=0.0,
        detail_strength=0.5, backend_preference="auto",
    )
    classical = apply_super_resolution_x2(
        rgb, strength=0.7, face_protection=0.0, deblock_strength=0.0, denoise_strength=0.0,
        detail_strength=0.5, backend_preference="classical",
    )
    assert np.array_equal(auto, classical)


def test_super_resolution_ignores_stale_local_region_and_still_returns_x2():
    rgb = np.zeros((36, 48, 3), dtype=np.uint8)
    rgb[8:28, 10:38] = [140, 120, 100]
    item = {
        "action_key": "super_resolution",
        "tested": True,
        "preview_available": True,
        "candidate": "almaz_x2_identity_guard_v1",
        "default_strength": 0.7,
        "parameters": {
            "backend_preference": "classical",
            "face_protection": 0.85,
            "deblock_strength": 0.0,
            "denoise_strength": 0.0,
            "detail_strength": 0.5,
            "face_boxes": [],
        },
    }
    preview = apply_selected_preview(
        rgb,
        [item],
        {"super_resolution"},
        action_regions={"super_resolution": {"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4}},
    )
    assert preview.shape == (72, 96, 3)


def test_detector_uses_real_source_size_not_bounded_analysis_copy():
    # Analyzer can pass a 2048px technical copy of a much larger original.
    y, x = np.indices((1024, 2048))
    base = (((x // 20 + y // 20) % 2) * 80 + 70).astype(np.uint8)
    rgb = np.dstack([base, base, base])
    metrics = _make_metrics(sharpness=45.0, noise=55.0, jpeg=55.0, face_count=0)
    advice = analyze_super_resolution_need(rgb, metrics, source_shape=(3000, 4000))
    assert advice.long_edge_px == 4000
    assert advice.megapixels == 12.0
    assert advice.recommend is False
    assert advice.need_score < 25.0
