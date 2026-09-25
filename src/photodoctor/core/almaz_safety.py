from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import cv2
import numpy as np

from .almaz_chroma_guard import archival_chroma_profile


@dataclass(frozen=True, slots=True)
class AlmazSafetyResult:
    accepted: bool
    action_key: str
    candidate: str
    mean_abs_delta_pct: float
    face_delta_pct: float
    chroma_drift_lab: float
    local_chroma_p99_lab: float
    clipping_delta_pct: float
    edge_ratio: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _pixel_face_boxes(
    face_boxes: object, shape: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    result: list[tuple[int, int, int, int]] = []
    if not isinstance(face_boxes, (list, tuple)):
        return result
    for face in face_boxes:
        if isinstance(face, Mapping):
            values = [face.get(key, 0.0) for key in ("x", "y", "w", "h")]
        elif isinstance(face, (list, tuple)) and len(face) >= 4:
            values = face[:4]
        else:
            continue
        try:
            x, y, bw, bh = [float(value) for value in values]
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite([x, y, bw, bh]).all() or bw <= 0.0 or bh <= 0.0:
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


def _clipping_pct(rgb: np.ndarray) -> float:
    low = np.all(rgb <= 1, axis=2)
    high = np.all(rgb >= 254, axis=2)
    return float(np.mean(low | high) * 100.0)


def _edge_energy(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(lap)))


def _mean_chroma_lab(rgb: np.ndarray) -> tuple[float, float]:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(lab[..., 1].mean()), float(lab[..., 2].mean())


def _local_chroma_p99_lab(before: np.ndarray, after: np.ndarray) -> float:
    lab0 = cv2.cvtColor(before, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab1 = cv2.cvtColor(after, cv2.COLOR_RGB2LAB).astype(np.float32)
    da = lab1[..., 1] - lab0[..., 1]
    db = lab1[..., 2] - lab0[..., 2]
    return float(np.percentile(np.hypot(da, db), 99.0))


def validate_almaz_transition(
    before: np.ndarray,
    after: np.ndarray,
    *,
    action_key: str,
    candidate: str,
    parameters: Mapping[str, Any] | None = None,
) -> AlmazSafetyResult:
    """Fail-closed safety probe for one ALMAZ correction transition.

    It deliberately validates the ALMAZ step itself, not the whole correction
    recipe, so earlier WB/exposure/contrast edits are not misclassified as AI
    drift.  This is a conservative artifact/identity guard, not a biometric test.
    """
    if before.ndim != 3 or before.shape[2] != 3 or before.dtype != np.uint8:
        raise ValueError("ALMAZ safety expects uint8 RGB source.")
    if after.ndim != 3 or after.shape[2] != 3 or after.dtype != np.uint8:
        return AlmazSafetyResult(False, action_key, candidate, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0, "ALMAZ вернул изображение неверного формата.")

    is_sr = str(action_key) == "super_resolution" or "x2" in str(candidate)
    expected_shape = (before.shape[0] * 2, before.shape[1] * 2, 3) if is_sr else before.shape
    if tuple(after.shape) != tuple(expected_shape):
        return AlmazSafetyResult(
            False, action_key, candidate, 100.0, 100.0, 100.0, 100.0, 100.0, 99.0,
            f"ALMAZ нарушил размерный контракт: {after.shape} вместо {expected_shape}.",
        )

    comparable = cv2.resize(after, (before.shape[1], before.shape[0]), interpolation=cv2.INTER_AREA) if is_sr else after
    delta = np.abs(comparable.astype(np.float32) - before.astype(np.float32))
    mean_abs = float(delta.mean() / 255.0 * 100.0)

    params = parameters if isinstance(parameters, Mapping) else {}
    boxes = _pixel_face_boxes(params.get("face_boxes", []), before.shape[:2])
    face_values: list[float] = []
    for x, y, w, h in boxes:
        d = delta[y:y+h, x:x+w]
        if d.size:
            face_values.append(float(d.mean() / 255.0 * 100.0))
    face_delta = float(np.mean(face_values)) if face_values else 0.0

    a0, b0 = _mean_chroma_lab(before)
    a1, b1 = _mean_chroma_lab(comparable)
    chroma_drift = float(np.hypot(a1 - a0, b1 - b0))
    local_chroma_p99 = _local_chroma_p99_lab(before, comparable)
    archival_like, _archive_center, _archive_spread = archival_chroma_profile(before)
    clipping_delta = float(max(0.0, _clipping_pct(comparable) - _clipping_pct(before)))
    edge_before = _edge_energy(before)
    edge_after = _edge_energy(comparable)

    task = str(action_key)
    if is_sr:
        max_mean, max_face, max_chroma, max_local_chroma, max_clip, min_edge, max_edge = 7.0, 5.8, 5.5, 18.0, 2.0, 0.40, 2.8
    elif task in {"noise", "almaz_denoise"}:
        max_mean, max_face, max_chroma, max_local_chroma, max_clip, min_edge, max_edge = 8.5, 5.8, 5.0, 20.0, 1.5, 0.30, 1.8
    elif task in {"sharpness", "almaz_deblur"}:
        # Deblur must not invent local colour.  The stock GoPro checkpoint is
        # domain-specific and can emit coloured islands on unrelated blur.
        max_mean, max_face, max_chroma, max_local_chroma, max_clip, min_edge, max_edge = 10.0, 6.5, 4.0, 6.0, 1.8, 0.55, 3.5
    else:  # jpeg recovery and future conservative x1 restoration
        max_mean, max_face, max_chroma, max_local_chroma, max_clip, min_edge, max_edge = 10.0, 6.2, 5.0, 20.0, 1.8, 0.38, 2.8
    if archival_like:
        max_local_chroma = min(max_local_chroma, 4.0)

    # Edge preservation only has a meaningful ratio when the source actually
    # contains measurable edges.  A flat/near-flat patch has edge_before≈0, so
    # requiring (say) 30% preservation would reject a perfectly harmless uniform
    # denoise result.  In that case switch the guard around: allow a quiet output
    # but reject *invented* strong contours.
    if edge_before < 0.75:
        edge_ratio = 1.0 if edge_after < 0.75 else float(edge_after / 0.75)
        edge_ok = edge_after <= 2.0
    else:
        edge_ratio = float(edge_after / edge_before)
        edge_ok = min_edge <= edge_ratio <= max_edge

    checks = {
        "mean": mean_abs <= max_mean,
        "face": face_delta <= max_face,
        "chroma": chroma_drift <= max_chroma,
        "local_chroma": local_chroma_p99 <= max_local_chroma,
        "clip": clipping_delta <= max_clip,
        "edge": edge_ok,
    }
    accepted = all(checks.values())
    if accepted:
        message = (
            f"ALMAZ safety PASS: Δ={mean_abs:.2f}%, лицо={face_delta:.2f}%, "
            f"цвет={chroma_drift:.2f} Lab, local p99={local_chroma_p99:.2f} Lab, "
            f"clipping +{clipping_delta:.2f} п.п., edge×{edge_ratio:.2f}."
        )
    else:
        failed = ", ".join(key for key, ok in checks.items() if not ok)
        message = (
            f"ALMAZ safety BLOCK ({failed}): Δ={mean_abs:.2f}%, лицо={face_delta:.2f}%, "
            f"цвет={chroma_drift:.2f} Lab, local p99={local_chroma_p99:.2f} Lab, "
            f"clipping +{clipping_delta:.2f} п.п., edge×{edge_ratio:.2f}."
        )
    return AlmazSafetyResult(
        accepted, action_key, candidate, mean_abs, face_delta, chroma_drift, local_chroma_p99, clipping_delta, edge_ratio, message
    )
