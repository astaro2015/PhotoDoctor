from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping, Any, Callable

import cv2
import numpy as np

from .models import MetricResult


@dataclass(slots=True)
class SuperResolutionAdvice:
    recommend: bool
    suggested_scale: int
    confidence: float
    strength: float
    need_score: float
    face_protection: float
    long_edge_px: int
    short_edge_px: int
    megapixels: float
    detail_deficit: float
    jpeg_need: float
    noise_need: float
    reasoning: list[str]
    backend: str = "almaz_classical_x2_identity_guard_v1"

    def to_raw(self) -> dict[str, Any]:
        return asdict(self)


def _score(metrics: Mapping[str, MetricResult], key: str) -> float | None:
    metric = metrics.get(key)
    if metric is None or metric.normalized_value is None:
        return None
    try:
        value = float(metric.normalized_value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def _raw_dict(metrics: Mapping[str, MetricResult], key: str) -> dict[str, Any]:
    metric = metrics.get(key)
    return metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}


def analyze_super_resolution_need(
    rgb: np.ndarray,
    metrics: Mapping[str, MetricResult],
    *,
    source_shape: tuple[int, int] | None = None,
) -> SuperResolutionAdvice:
    # ``rgb`` may be Analyzer's bounded technical copy (typically <=2048 px).
    # Low-resolution decisions must use the *real source dimensions*, otherwise a
    # 12–50 MP original would look deceptively small after analysis normalization.
    probe_h, probe_w = rgb.shape[:2]
    if source_shape is None:
        source_h, source_w = probe_h, probe_w
    else:
        source_h, source_w = int(source_shape[0]), int(source_shape[1])
        if source_h <= 0 or source_w <= 0:
            source_h, source_w = probe_h, probe_w
    long_edge = int(max(source_h, source_w))
    short_edge = int(min(source_h, source_w))
    megapixels = float((source_h * source_w) / 1_000_000.0)

    # Reject nearly empty/flat technical frames.  Resolution alone is not a reason
    # to invent detail where the source contains essentially no visual structure.
    probe = rgb
    if long_edge > 640:
        scale = 640.0 / long_edge
        probe = cv2.resize(rgb, (max(1, round(probe_w * scale)), max(1, round(probe_h * scale))), interpolation=cv2.INTER_AREA)
    gray_probe = cv2.cvtColor(probe, cv2.COLOR_RGB2GRAY)
    information_std = float(np.std(gray_probe))
    edge_probe = cv2.Laplacian(gray_probe, cv2.CV_32F)
    edge_energy = float(np.mean(np.abs(edge_probe)))
    low_information = bool(information_std < 3.5 and edge_energy < 1.2)

    # Strong preference for genuinely low-resolution small photos.  Bigger modern
    # files should not be upscaled just because a user can.
    if long_edge <= 1280:
        size_need = 1.0
    elif long_edge <= 1600:
        size_need = 0.86
    elif long_edge <= 2000:
        size_need = 0.66
    elif long_edge <= 2400:
        size_need = 0.42
    else:
        size_need = 0.08

    if megapixels <= 1.2:
        mp_need = 1.0
    elif megapixels <= 2.2:
        mp_need = 0.90
    elif megapixels <= 3.2:
        mp_need = 0.60
    elif megapixels <= 4.8:
        mp_need = 0.30
    else:
        mp_need = 0.05

    sharp_values = [
        value for value in (
            _score(metrics, "local_sharpness"),
            _score(metrics, "faces"),
            _score(metrics, "eyes"),
        ) if value is not None
    ]
    worst_sharp = float(min(sharp_values)) if sharp_values else 72.0
    detail_deficit = float(np.clip((74.0 - worst_sharp) / 30.0, 0.0, 1.0))

    jpeg_score = _score(metrics, "jpeg_artifacts")
    jpeg_need = float(np.clip((72.0 - float(jpeg_score if jpeg_score is not None else 72.0)) / 30.0, 0.0, 1.0))

    noise_score = _score(metrics, "noise")
    noise_need = float(np.clip((70.0 - float(noise_score if noise_score is not None else 70.0)) / 34.0, 0.0, 1.0))

    semantic = _raw_dict(metrics, "semantic_context")
    classification = str(semantic.get("classification", "unknown"))
    archival_bonus = 0.05 if classification == "archival_portrait" else 0.0

    # A weighted blend rather than a hard threshold: small size dominates, but
    # weak detail / JPEG damage strengthen the case.
    benefit = (
        0.42 * size_need
        + 0.26 * mp_need
        + 0.18 * detail_deficit
        + 0.08 * jpeg_need
        + 0.03 * noise_need
        + archival_bonus
    )
    if long_edge > 2600 or megapixels > 5.5:
        benefit *= 0.55
    if long_edge > 3400 or megapixels > 8.0:
        benefit *= 0.25

    if low_information:
        benefit *= 0.18
    recommend = bool((not low_information) and benefit >= 0.56 and long_edge <= 2600 and megapixels <= 5.5)

    reasoning: list[str] = []
    if size_need >= 0.65 or mp_need >= 0.65:
        reasoning.append("Размер исходника небольшой; x2 может дать более современный рабочий размер.")
    if detail_deficit >= 0.35:
        reasoning.append("Есть признаки нехватки реальной детализации; нужен очень бережный x2 без дорисовки черт.")
    if jpeg_need >= 0.28:
        reasoning.append("JPEG-артефакты заметны; перед x2 нужна мягкая очистка блоков.")
    if noise_need >= 0.25:
        reasoning.append("Есть шум; до увеличения нужен осторожный предочиститель.")
    if classification == "archival_portrait":
        reasoning.append("Архивный портрет: лица требуют усиленной защиты идентичности.")
    if low_information:
        reasoning = ["Кадр почти не содержит структуры; увеличение x2 не должно выдумывать отсутствующие детали."]
    elif not reasoning:
        reasoning.append("Размер и детализация уже близки к норме; x2 не обязателен.")

    face_protection = 0.88 if _raw_dict(metrics, "faces").get("face_count", 0) else 0.55
    strength = float(np.clip(0.52 + 0.35 * benefit, 0.45, 0.84))
    confidence = float(np.clip(0.58 + abs(benefit - 0.50) * 0.55, 0.55, 0.92))
    return SuperResolutionAdvice(
        recommend=recommend,
        suggested_scale=2,
        confidence=confidence,
        strength=strength,
        need_score=float(np.clip(benefit * 100.0, 0.0, 100.0)),
        face_protection=face_protection,
        long_edge_px=long_edge,
        short_edge_px=short_edge,
        megapixels=megapixels,
        detail_deficit=detail_deficit,
        jpeg_need=jpeg_need,
        noise_need=noise_need,
        reasoning=reasoning,
    )


def _face_boxes_from_metrics(
    metrics: Mapping[str, MetricResult] | None, image_shape: tuple[int, int]
) -> list[tuple[float, float, float, float]]:
    if metrics is None:
        return []
    raw = _raw_dict(metrics, "faces")
    faces = raw.get("faces", []) if isinstance(raw, dict) else []
    boxes: list[tuple[float, float, float, float]] = []
    if not isinstance(faces, list):
        return boxes
    image_h, image_w = image_shape
    for face in faces:
        if not isinstance(face, dict):
            continue
        try:
            x = float(face.get("x", 0)); y = float(face.get("y", 0))
            w = float(face.get("w", 0)); h = float(face.get("h", 0))
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite([x, y, w, h]).all():
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
            x *= image_w; w *= image_w; y *= image_h; h *= image_h
        if w >= 4 and h >= 4:
            boxes.append((x, y, w, h))
    return boxes


def _pixel_face_boxes(
    face_boxes: list[tuple[float, float, float, float]], image_shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    image_h, image_w = image_shape
    out: list[tuple[int, int, int, int]] = []
    for x, y, w, h in face_boxes:
        x = float(x); y = float(y); w = float(w); h = float(h)
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
            x *= image_w; w *= image_w; y *= image_h; h *= image_h
        ix, iy, iw, ih = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        if iw >= 4 and ih >= 4:
            out.append((ix, iy, iw, ih))
    return out


def _mild_preclean(rgb: np.ndarray, deblock_strength: float, denoise_strength: float) -> np.ndarray:
    out = rgb
    if deblock_strength > 0.01:
        filtered = cv2.bilateralFilter(out, 5, 10.0 + 20.0 * deblock_strength, 3.0 + 2.0 * deblock_strength)
        alpha = 0.15 + 0.30 * deblock_strength
        out = np.clip(np.rint(out.astype(np.float32) * (1.0 - alpha) + filtered.astype(np.float32) * alpha), 0, 255).astype(np.uint8)
    if denoise_strength > 0.01:
        h = 1.0 + 2.7 * denoise_strength
        hc = 1.0 + 2.1 * denoise_strength
        out = cv2.fastNlMeansDenoisingColored(out, None, h, hc, 7, 21)
    return out


def _detail_boost(upscaled: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return upscaled.copy()
    lab = cv2.cvtColor(upscaled, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    lf = l.astype(np.float32)
    blur = cv2.GaussianBlur(lf, (0, 0), sigmaX=0.8)
    detail = lf - blur
    gate = np.clip((np.abs(detail) - 1.0) / 10.0, 0.0, 1.0)
    amount = 0.34 + 0.34 * strength
    l2 = np.clip(lf + amount * detail * gate, 0.0, 255.0).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
    return out


def _face_identity_guard(
    bicubic: np.ndarray,
    enhanced: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]],
    protection: float,
) -> np.ndarray:
    protection = float(np.clip(protection, 0.0, 1.0))
    if protection <= 0.01 or not face_boxes:
        return enhanced
    h, w = bicubic.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for x, y, bw, bh in face_boxes:
        sx = int(round(x * 2)); sy = int(round(y * 2))
        sw = max(2, int(round(bw * 2))); sh = max(2, int(round(bh * 2)))
        pad_x = max(2, int(round(sw * 0.18)))
        pad_y = max(2, int(round(sh * 0.18)))
        x0 = max(0, sx - pad_x); y0 = max(0, sy - pad_y)
        x1 = min(w, sx + sw + pad_x); y1 = min(h, sy + sh + pad_y)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        cv2.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 1.0, -1)
    if not np.any(mask):
        return enhanced
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=6.0, sigmaY=6.0)
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    enhanced_weight = 1.0 - 0.62 * protection
    mixed = enhanced.astype(np.float32) * (1.0 - mask + mask * enhanced_weight) + bicubic.astype(np.float32) * (mask * (1.0 - enhanced_weight))
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def apply_super_resolution_x2(
    rgb: np.ndarray,
    *,
    strength: float = 0.72,
    face_protection: float = 0.85,
    deblock_strength: float = 0.22,
    denoise_strength: float = 0.14,
    detail_strength: float = 0.55,
    metrics: Mapping[str, MetricResult] | None = None,
    face_boxes: list[tuple[float, float, float, float]] | None = None,
    backend_preference: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Conservative x2 restoration backend for ALMAZ Phase 1.

    This is intentionally *not* a hallucinating generative upscaler.  It provides
    the surrounding architecture and a safe bounded preview path while the future
    ONNX/CUDA/NPU backend is still to be integrated.
    """
    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    strength = float(np.clip(strength, 0.0, 1.0))
    emit("ALMAZ x2: подготавливаю исходное изображение")
    if strength <= 0.0:
        return cv2.resize(rgb, (rgb.shape[1] * 2, rgb.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    raw_boxes = list(face_boxes or _face_boxes_from_metrics(metrics, rgb.shape[:2]))
    face_boxes_px = _pixel_face_boxes(raw_boxes, rgb.shape[:2])
    emit("ALMAZ x2: мягко очищаю JPEG-артефакты и шум")
    preclean = _mild_preclean(
        rgb,
        deblock_strength=float(np.clip(deblock_strength * strength, 0.0, 1.0)),
        denoise_strength=float(np.clip(denoise_strength * strength, 0.0, 1.0)),
    )
    emit("ALMAZ x2: создаю безопасную базовую копию x2")
    bicubic = cv2.resize(preclean, (preclean.shape[1] * 2, preclean.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

    enhanced: np.ndarray | None = None
    if str(backend_preference).strip().lower() != "classical":
        try:
            from photodoctor.ai.sr_runtime import load_installed_sr_backend

            emit("ALMAZ x2: проверяю установленную AI-модель и ускоритель")
            backend, status = load_installed_sr_backend()
            if backend is not None:
                provider_label = getattr(status, "provider_label", None) or getattr(status, "provider", None) or "ONNX Runtime"
                emit(f"ALMAZ x2: запускаю модель через {provider_label}")
                last_pct = [-1]
                def on_tile(done: int, total: int) -> None:
                    pct = int(round(done * 100 / max(total, 1)))
                    if pct != last_pct[0] or done == total:
                        last_pct[0] = pct
                        emit(f"ALMAZ x2: восстанавливаю фрагменты {done}/{total} ({pct}%)")
                try:
                    model_out = backend.upscale_x2(preclean, progress=on_tile)
                except TypeError as exc:
                    if "progress" not in str(exc):
                        raise
                    model_out = backend.upscale_x2(preclean)
                    on_tile(1, 1)
                model_weight = float(np.clip(0.48 + 0.40 * strength, 0.45, 0.86))
                enhanced = np.clip(
                    np.rint(bicubic.astype(np.float32) * (1.0 - model_weight) + model_out.astype(np.float32) * model_weight),
                    0, 255,
                ).astype(np.uint8)
        except Exception:
            # Fail closed to the deterministic non-AI path.  A broken optional
            # runtime/model must never make preview/save unavailable.
            enhanced = None

    if enhanced is None:
        emit("ALMAZ x2: AI-модель недоступна, использую безопасный локальный fallback")
        enhanced = _detail_boost(bicubic, detail_strength * strength)
    emit("ALMAZ x2: смешиваю восстановленные детали с безопасной базой")
    emit("ALMAZ x2: защищаю лица и геометрию черт")
    guarded = _face_identity_guard(bicubic, enhanced, face_boxes_px, face_protection)
    emit("ALMAZ x2: восстановление завершено, передаю результат на проверку безопасности")
    return guarded


def face_identity_drift(
    original_rgb: np.ndarray,
    preview_rgb: np.ndarray,
    *,
    face_boxes: list[tuple[float, float, float, float]] | None = None,
    metrics: Mapping[str, MetricResult] | None = None,
) -> float:
    """Estimate bounded face drift after x2 preview.

    The x2 preview is downsampled back to source size before comparison.  This is
    a safety signal, not a biometric identity system.
    """
    raw_boxes = list(face_boxes or _face_boxes_from_metrics(metrics, original_rgb.shape[:2]))
    boxes = _pixel_face_boxes(raw_boxes, original_rgb.shape[:2])
    if not boxes:
        return 0.0
    down = cv2.resize(preview_rgb, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_AREA)
    deltas: list[float] = []
    for x, y, w, h in boxes:
        x0 = max(0, x); y0 = max(0, y)
        x1 = min(original_rgb.shape[1], x + w); y1 = min(original_rgb.shape[0], y + h)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        before = original_rgb[y0:y1, x0:x1].astype(np.float32)
        after = down[y0:y1, x0:x1].astype(np.float32)
        mae = float(np.mean(np.abs(before - after)))
        deltas.append(mae / 255.0 * 100.0)
    if not deltas:
        return 0.0
    return float(np.mean(deltas))
