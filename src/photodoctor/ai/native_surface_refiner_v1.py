from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

MODEL_ID = "native_surface_refiner_v1"
MODEL_VERSION = "1.0.0-experimental"
TRAINING_KIND = "procedural_precise_candidate_hograwmeta_v1"
TASK = "surface_defect_refinement"
INPUT_SIZE = 96
RAW_SIZE = 32
CONFIDENCE_THRESHOLD = 0.70
CONTRACT_ID = "native_gray96_hograwmeta_mlp_binary_v1"
LABELS = ("natural_detail", "defect")
PARAMETERS = 1_229_154

_HOG = cv2.HOGDescriptor((96, 96), (16, 16), (8, 8), (8, 8), 9)
_HOG_DIM = int(_HOG.getDescriptorSize())
_META_DIM = 9
_FEATURE_DIM = _HOG_DIM + RAW_SIZE * RAW_SIZE + _META_DIM
_MODEL_PATH = Path(__file__).resolve().parent / "models" / "native_surface_refiner_v1.npz"


class NativeSurfaceRefinerV1Error(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _weights() -> dict[str, np.ndarray]:
    try:
        with np.load(_MODEL_PATH) as data:
            return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
    except Exception as exc:  # pragma: no cover - guarded by packaging/integrity tests
        raise NativeSurfaceRefinerV1Error(f"Не удалось загрузить встроенные веса v1: {exc}") from exc


def _candidate_meta(box: dict[str, Any] | None) -> np.ndarray:
    box = box or {}
    w = max(float(box.get("w", 0.0) or 0.0), 1e-6)
    h = max(float(box.get("h", 0.0) or 0.0), 1e-6)
    aspect = max(w, h) / min(w, h)
    area = w * h
    strength = float(box.get("strength", 0.0) or 0.0)
    bright = 1.0 if str(box.get("polarity", "")) == "bright" else 0.0
    contour = box.get("contour", [])
    points: list[tuple[float, float]] = []
    if isinstance(contour, list):
        for item in contour:
            if isinstance(item, dict):
                try:
                    points.append((float(item.get("x", 0.0)), float(item.get("y", 0.0))))
                except (TypeError, ValueError):
                    continue
    perimeter = 0.0
    polygon_area = 0.0
    if len(points) >= 2:
        for a, b in zip(points, points[1:] + points[:1]):
            perimeter += float(np.hypot(b[0] - a[0], b[1] - a[1]))
    if len(points) >= 3:
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        polygon_area = abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))) * 0.5
    fill = float(np.clip(polygon_area / max(area, 1e-8), 0.0, 1.0))
    return np.asarray(
        [
            np.clip(np.log1p(aspect) / 3.0, 0.0, 1.5),
            np.clip(np.sqrt(area) * 8.0, 0.0, 1.5),
            np.clip(strength, 0.0, 1.0),
            bright,
            np.clip(len(points) / 32.0, 0.0, 1.0),
            np.clip(perimeter * 4.0, 0.0, 2.0),
            fill,
            np.clip(w * 12.0, 0.0, 2.0),
            np.clip(h * 12.0, 0.0, 2.0),
        ],
        dtype=np.float32,
    )


def feature_vector(rgb: np.ndarray, box: dict[str, Any] | None = None) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise NativeSurfaceRefinerV1Error("Ожидается непустой фрагмент RGB H×W×3.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    interp = cv2.INTER_AREA if max(gray.shape) > INPUT_SIZE else cv2.INTER_CUBIC
    gray = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(6, 6)).apply(gray)
    hog = _HOG.compute(clahe).reshape(-1).astype(np.float32)
    raw = cv2.resize(gray, (RAW_SIZE, RAW_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    mean = float(raw.mean())
    std = max(float(raw.std()), 0.06)
    raw = (np.clip((raw - mean) / std, -3.0, 3.0) / 3.0).reshape(-1).astype(np.float32)
    feature = np.concatenate([hog, raw, _candidate_meta(box)]).astype(np.float32, copy=False)
    if feature.size != _FEATURE_DIM:
        raise NativeSurfaceRefinerV1Error(f"Unexpected feature size {feature.size}, expected {_FEATURE_DIM}.")
    return feature


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def _forward(features: np.ndarray) -> np.ndarray:
    w = _weights()
    mean = w["feature_mean"]
    std = np.maximum(w["feature_std"], 1e-3)
    x = (features - mean) / std
    x = np.maximum(x @ w["fc1.weight"].T + w["fc1.bias"], 0.0)
    x = np.maximum(x @ w["fc2.weight"].T + w["fc2.bias"], 0.0)
    return x @ w["fc3.weight"].T + w["fc3.bias"]


def predict_patches(
    patches: Sequence[np.ndarray],
    boxes: Sequence[dict[str, Any] | None] | None = None,
    *,
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    if boxes is None:
        boxes = [None] * len(patches)
    if len(boxes) != len(patches):
        raise NativeSurfaceRefinerV1Error("Число фрагментов и рамок не совпадает")
    if not patches:
        return []
    batch_size = max(1, int(batch_size))
    outputs: list[dict[str, Any]] = []
    for start in range(0, len(patches), batch_size):
        stop = min(len(patches), start + batch_size)
        feats = np.stack([feature_vector(patches[i], boxes[i]) for i in range(start, stop)], axis=0)
        probs = _softmax(_forward(feats))
        for row in probs:
            raw_index = int(np.argmax(row))
            confidence = float(row[raw_index])
            raw_label = LABELS[raw_index]
            label = raw_label if confidence >= CONFIDENCE_THRESHOLD else "uncertain"
            outputs.append(
                {
                    "model_id": MODEL_ID,
                    "model_version": MODEL_VERSION,
                    "training_kind": TRAINING_KIND,
                    "contract_id": CONTRACT_ID,
                    "label": label,
                    "raw_label": raw_label,
                    "confidence": confidence,
                    "defect_probability": float(row[1]),
                    "probabilities": {LABELS[0]: float(row[0]), LABELS[1]: float(row[1])},
                    "uncertainty_threshold": CONFIDENCE_THRESHOLD,
                    "experimental": True,
                }
            )
    return outputs


def predict_patch(rgb: np.ndarray, box: dict[str, Any] | None = None) -> dict[str, Any]:
    return predict_patches([rgb], [box], batch_size=1)[0]


def model_metadata() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "version": MODEL_VERSION,
        "task": TASK,
        "contract_id": CONTRACT_ID,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "labels": ["natural_detail", "defect", "uncertain"],
        "provider": "native_numpy",
        "training_kind": TRAINING_KIND,
        "parameters": PARAMETERS,
        "weights_bytes": _MODEL_PATH.stat().st_size if _MODEL_PATH.is_file() else None,
        "synthetic_validation_accuracy": 0.925,
        "synthetic_selective_accuracy_0_70": 0.955752,
        "synthetic_selective_coverage_0_70": 0.904,
        "experimental": True,
        "automatic_edits": False,
    }
