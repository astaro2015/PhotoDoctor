from __future__ import annotations

import base64
import io
import zlib
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from .native_surface_weights import MODEL_ID, MODEL_VERSION, TRAINING_KIND, WEIGHTS_B85

INPUT_SIZE = 32
CONTRACT_ID = "native_gray32_cnn_binary_v1"
TASK = "surface_defect_refinement"
LABELS = ("not_defect", "defect")


class NativeSurfaceRefinerError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _weights() -> dict[str, np.ndarray]:
    try:
        payload = zlib.decompress(base64.b85decode(WEIGHTS_B85.encode("ascii")))
        with np.load(io.BytesIO(payload)) as data:
            return {key: np.asarray(data[key], dtype=np.float32) for key in data.files}
    except Exception as exc:  # pragma: no cover - embedded payload is covered by integrity test
        raise NativeSurfaceRefinerError(f"Не удалось загрузить встроенные веса: {exc}") from exc


def preprocess_patch(rgb: np.ndarray) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise NativeSurfaceRefinerError("Ожидается непустой фрагмент RGB H×W×3.")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    interp = cv2.INTER_AREA if max(gray.shape) > INPUT_SIZE else cv2.INTER_LINEAR
    gray = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=interp).astype(np.float32) / 255.0
    mean = float(gray.mean())
    std = max(float(gray.std()), 0.08)
    x = np.clip((gray - mean) / std, -3.0, 3.0) / 3.0
    return x.astype(np.float32, copy=False)


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # PyTorch Conv2d is cross-correlation. OpenCV filter2D uses the same kernel orientation.
    if x.ndim != 3:
        raise NativeSurfaceRefinerError("Внутренний тензор нейросети должен иметь порядок канал–высота–ширина.")
    out = np.empty((weight.shape[0], x.shape[1], x.shape[2]), dtype=np.float32)
    for oc in range(weight.shape[0]):
        acc = np.full((x.shape[1], x.shape[2]), float(bias[oc]), dtype=np.float32)
        for ic in range(weight.shape[1]):
            acc += cv2.filter2D(
                x[ic], cv2.CV_32F, weight[oc, ic], borderType=cv2.BORDER_CONSTANT
            )
        out[oc] = np.maximum(acc, 0.0)
    return out


def _maxpool2(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    x = x[:, : h2 * 2, : w2 * 2]
    return x.reshape(c, h2, 2, w2, 2).max(axis=(2, 4))


def _softmax2(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64) - float(np.max(logits))
    e = np.exp(z)
    return (e / max(float(e.sum()), 1e-12)).astype(np.float32)


def predict_patch(rgb: np.ndarray) -> dict[str, Any]:
    w = _weights()
    x = preprocess_patch(rgb)[None, :, :]
    x = _maxpool2(_conv2d(x, w["conv1.weight"], w["conv1.bias"]))
    x = _maxpool2(_conv2d(x, w["conv2.weight"], w["conv2.bias"]))
    x = _conv2d(x, w["conv3.weight"], w["conv3.bias"])
    features = x.mean(axis=(1, 2))
    logits = w["fc.weight"] @ features + w["fc.bias"]
    probs = _softmax2(logits)
    index = int(np.argmax(probs))
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "training_kind": TRAINING_KIND,
        "contract_id": CONTRACT_ID,
        "label": LABELS[index],
        "confidence": float(probs[index]),
        "defect_probability": float(probs[1]),
        "probabilities": {LABELS[0]: float(probs[0]), LABELS[1]: float(probs[1])},
        "experimental": True,
    }


def model_metadata() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "version": MODEL_VERSION,
        "task": TASK,
        "contract_id": CONTRACT_ID,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "labels": list(LABELS),
        "provider": "native_numpy",
        "training_kind": TRAINING_KIND,
        "experimental": True,
        "automatic_edits": False,
    }
