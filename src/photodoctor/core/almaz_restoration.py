from __future__ import annotations

from typing import Mapping, Any, Callable

import cv2
import numpy as np

from photodoctor.ai.restoration_runtime import load_installed_restoration_backend
from .almaz_chroma_guard import guard_deblur_chroma


class AlmazRestorationUnavailable(RuntimeError):
    pass


def _pixel_face_boxes(
    face_boxes: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None,
    shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    result: list[tuple[int, int, int, int]] = []
    for face in face_boxes or []:
        if not isinstance(face, Mapping):
            continue
        try:
            x = float(face.get("x", 0.0)); y = float(face.get("y", 0.0))
            bw = float(face.get("w", 0.0)); bh = float(face.get("h", 0.0))
        except (TypeError, ValueError, OverflowError):
            continue
        vals = np.asarray([x, y, bw, bh], dtype=np.float64)
        if not np.isfinite(vals).all() or bw <= 0.0 or bh <= 0.0:
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0:
            x *= w; bw *= w; y *= h; bh *= h
        x0 = max(0, min(w - 1, int(round(x))))
        y0 = max(0, min(h - 1, int(round(y))))
        x1 = max(x0 + 1, min(w, int(round(x + bw))))
        y1 = max(y0 + 1, min(h, int(round(y + bh))))
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            result.append((x0, y0, x1 - x0, y1 - y0))
    return result


def _identity_guard_x1(
    original: np.ndarray,
    restored: np.ndarray,
    face_boxes: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None,
    protection: float,
) -> np.ndarray:
    protection = float(np.clip(protection, 0.0, 1.0))
    boxes = _pixel_face_boxes(face_boxes, original.shape[:2])
    if protection <= 0.0 or not boxes:
        return restored
    h, w = original.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for x, y, bw, bh in boxes:
        pad_x = max(2, int(round(bw * 0.16)))
        pad_y = max(2, int(round(bh * 0.16)))
        x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
        x1 = min(w, x + bw + pad_x); y1 = min(h, y + bh + pad_y)
        cv2.rectangle(mask, (x0, y0), (x1 - 1, y1 - 1), 1.0, -1)
    sigma = max(2.0, min(h, w) * 0.006)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    # At protection=1 retain only 35% of model delta inside faces.  We do not
    # freeze faces completely because real denoise/deblur may still be useful.
    model_weight = 1.0 - 0.65 * protection
    mixed = restored.astype(np.float32) * (1.0 - mask + mask * model_weight) + original.astype(np.float32) * (mask * (1.0 - model_weight))
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def apply_almaz_restoration(
    rgb: np.ndarray,
    *,
    task: str,
    strength: float,
    face_boxes: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...] | None = None,
    face_protection: float = 0.85,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Run a verified installed x1 ALMAZ model and blend it conservatively.

    There is deliberately no silent classical fallback here.  The correction
    candidate is offered only when a verified model was present during analysis;
    if it disappears or its runtime fails before preview, the UI should report
    that failure rather than pretend the user is looking at the AI result.
    """
    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    labels = {"denoise": "шумоподавление", "deblur": "устранение смаза", "jpeg_recovery": "восстановление JPEG"}
    task_label = labels.get(str(task), str(task))
    emit(f"ALMAZ: проверяю модель для задачи «{task_label}»")
    backend, status = load_installed_restoration_backend(task)
    if backend is None or not status.ready:
        raise AlmazRestorationUnavailable(status.detail)
    try:
        provider_label = getattr(status, "provider_label", None) or getattr(status, "provider", None) or "ONNX Runtime"
        emit(f"ALMAZ: запускаю «{task_label}» через {provider_label}")
        last_pct = [-1]
        def on_tile(done: int, total: int) -> None:
            pct = int(round(done * 100 / max(total, 1)))
            if pct != last_pct[0] or done == total:
                last_pct[0] = pct
                emit(f"ALMAZ: {task_label} — обрабатываю фрагменты {done}/{total} ({pct}%)")
        try:
            restored = backend.restore(rgb, progress=on_tile)
        except TypeError as exc:
            if "progress" not in str(exc):
                raise
            restored = backend.restore(rgb)
            on_tile(1, 1)
    except Exception as exc:
        raise AlmazRestorationUnavailable(f"ALMAZ {task} inference не выполнен: {exc}") from exc
    if restored.shape != rgb.shape:
        raise AlmazRestorationUnavailable(
            f"ALMAZ {task} нарушил x1-контракт: {restored.shape} вместо {rgb.shape}."
        )
    if str(task).strip().lower() == "deblur":
        emit("ALMAZ Deblur: проверяю паразитные цветовые пятна до смешивания")
        restored, chroma_guard = guard_deblur_chroma(rgb, restored)
        if chroma_guard.applied:
            emit(
                "ALMAZ Deblur: цветовая защита сработала "
                f"(chroma p99 {chroma_guard.raw_local_chroma_p99:.2f} -> "
                f"{chroma_guard.guarded_local_chroma_p99:.2f} Lab)"
            )
    emit(f"ALMAZ: {task_label} — смешиваю результат с исходником")
    strength = float(np.clip(strength, 0.0, 1.0))
    mixed = np.clip(
        np.rint(rgb.astype(np.float32) * (1.0 - strength) + restored.astype(np.float32) * strength),
        0, 255,
    ).astype(np.uint8)
    emit(f"ALMAZ: {task_label} — защищаю лица и важные детали")
    guarded = _identity_guard_x1(rgb, mixed, face_boxes, face_protection)
    emit(f"ALMAZ: {task_label} завершено, передаю результат на проверку безопасности")
    return guarded
