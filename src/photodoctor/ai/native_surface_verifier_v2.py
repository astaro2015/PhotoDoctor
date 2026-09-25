from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .native_surface_context_meta_v2 import predict_box as predict_context_meta_v2, predict_boxes as predict_context_meta_v2_batch

MODEL_ID = "native_surface_verifier_v2"
MODEL_VERSION = "2.3.0-candidate"
TRAINING_KIND = "candidate_pair_group_cv_segmentation_v2"
TASK = "surface_defect_refinement"
INPUT_SIZE = 96
CONTRACT_ID = "native_surface96_6ch_unet_contextmeta_candidate_v2"
PARAMETERS = 55_737
DEFAULT_SUGGESTION_THRESHOLD = 0.82
_MODEL_PATH = Path(__file__).resolve().parent / "models" / "native_surface_verifier_v2.npz"


class NativeSurfaceVerifierV2Error(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _bundle() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(_MODEL_PATH, allow_pickle=False) as data:
            weights = {
                key: np.asarray(data[key], dtype=np.float32)
                for key in data.files
                if key != "metadata_json"
            }
            raw_meta = str(np.asarray(data["metadata_json"]).item()) if "metadata_json" in data.files else "{}"
        metadata = json.loads(raw_meta)
        if not isinstance(metadata, dict):
            metadata = {}
        return weights, metadata
    except Exception as exc:  # pragma: no cover - packaging/integrity tests guard the file
        raise NativeSurfaceVerifierV2Error(f"Не удалось загрузить встроенные веса Surface AI v2: {exc}") from exc


def _conv2d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, padding: int = 1) -> np.ndarray:
    kh, kw = int(weight.shape[2]), int(weight.shape[3])
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    out = np.einsum("bchwij,ocij->bohw", windows, weight, optimize=True)
    out += bias[None, :, None, None]
    return out.astype(np.float32, copy=False)


def _group_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, *, groups: int = 4) -> np.ndarray:
    batch, channels, height, width = x.shape
    group_count = groups if channels % groups == 0 else 1
    z = x.reshape(batch, group_count, channels // group_count, height, width)
    mean = z.mean(axis=(2, 3, 4), keepdims=True, dtype=np.float32)
    var = z.var(axis=(2, 3, 4), keepdims=True, dtype=np.float32)
    z = (z - mean) / np.sqrt(var + np.float32(1e-5))
    z = z.reshape(batch, channels, height, width)
    return z * weight[None, :, None, None] + bias[None, :, None, None]


def _silu(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return x / (1.0 + np.exp(-clipped))


def _block(x: np.ndarray, weights: Mapping[str, np.ndarray], prefix: str) -> np.ndarray:
    x = _conv2d(x, weights[f"{prefix}.net.0.weight"], weights[f"{prefix}.net.0.bias"])
    x = _group_norm(x, weights[f"{prefix}.net.1.weight"], weights[f"{prefix}.net.1.bias"])
    x = _silu(x)
    x = _conv2d(x, weights[f"{prefix}.net.3.weight"], weights[f"{prefix}.net.3.bias"])
    x = _group_norm(x, weights[f"{prefix}.net.4.weight"], weights[f"{prefix}.net.4.bias"])
    return _silu(x).astype(np.float32, copy=False)


def _max_pool_2x2(x: np.ndarray) -> np.ndarray:
    b, c, h, w = x.shape
    if h % 2 or w % 2:
        raise NativeSurfaceVerifierV2Error("Внутренний размер v2 должен делиться на 2")
    return x.reshape(b, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def _conv_transpose_2x2(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    # PyTorch ConvTranspose2d(Cin,Cout,kernel=2,stride=2). With stride==kernel
    # the four output phases never overlap, so this exact vectorized form is cheap.
    b, _, h, w = x.shape
    cout = int(weight.shape[1])
    out = np.empty((b, cout, h * 2, w * 2), dtype=np.float32)
    for ky in range(2):
        for kx in range(2):
            out[:, :, ky::2, kx::2] = (
                np.einsum("bihw,io->bohw", x, weight[:, :, ky, kx], optimize=True)
                + bias[None, :, None, None]
            )
    return out


def _forward(features: np.ndarray) -> np.ndarray:
    weights, _ = _bundle()
    e1 = _block(features, weights, "e1")
    e2 = _block(_max_pool_2x2(e1), weights, "e2")
    bottleneck = _block(_max_pool_2x2(e2), weights, "b")
    u2 = _conv_transpose_2x2(bottleneck, weights["u2.weight"], weights["u2.bias"])
    d2 = _block(np.concatenate([u2, e2], axis=1), weights, "d2")
    u1 = _conv_transpose_2x2(d2, weights["u1.weight"], weights["u1.bias"])
    d1 = _block(np.concatenate([u1, e1], axis=1), weights, "d1")
    return _conv2d(d1, weights["out.weight"], weights["out.bias"], padding=0)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def _box_to_px(box: Mapping[str, Any], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    try:
        x = float(box.get("x", 0.0)); y = float(box.get("y", 0.0))
        bw = float(box.get("w", 0.0)); bh = float(box.get("h", 0.0))
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite([x, y, bw, bh]).all() or bw <= 0.0 or bh <= 0.0:
        return None
    x0 = int(round(x * w)); y0 = int(round(y * h))
    x1 = int(round((x + bw) * w)); y1 = int(round((y + bh) * h))
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _context_crop_bounds(box_px: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    x0, y0, x1, y1 = box_px
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(40.0, max(bw, bh) * 3.0, min(h, w) * 0.07)
    side = min(side, max(h, w) * 0.48)
    return (
        max(0, int(round(cx - side / 2.0))),
        max(0, int(round(cy - side / 2.0))),
        min(w, int(round(cx + side / 2.0))),
        min(h, int(round(cy + side / 2.0))),
    )


def _local_prior(
    box: Mapping[str, Any],
    full_shape: tuple[int, int],
    crop_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    h, w = full_shape
    cx0, cy0, cx1, cy1 = crop_bounds
    out = np.zeros((cy1 - cy0, cx1 - cx0), np.uint8)
    points: list[list[int]] = []
    contour = box.get("contour", [])
    if isinstance(contour, list):
        for item in contour:
            if not isinstance(item, Mapping):
                continue
            try:
                px = int(round(float(item.get("x", 0.0)) * w)) - cx0
                py = int(round(float(item.get("y", 0.0)) * h)) - cy0
            except (TypeError, ValueError, OverflowError):
                continue
            if -2 <= px <= out.shape[1] + 1 and -2 <= py <= out.shape[0] + 1:
                points.append([int(np.clip(px, 0, out.shape[1] - 1)), int(np.clip(py, 0, out.shape[0] - 1))])
    if len(points) >= 3:
        cv2.fillPoly(out, [np.asarray(points, np.int32)], 1)
    if int(out.sum()) < 2:
        bp = _box_to_px(box, full_shape)
        if bp is not None:
            x0, y0, x1, y1 = bp
            lx0 = max(0, x0 - cx0); ly0 = max(0, y0 - cy0)
            lx1 = min(out.shape[1], x1 - cx0); ly1 = min(out.shape[0], y1 - cy0)
            if lx1 > lx0 and ly1 > ly0:
                out[ly0:ly1, lx0:lx1] = 1
    return out


def _morphology(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bright: list[np.ndarray] = []
    dark: list[np.ndarray] = []
    for size in (3, 5, 9, 15):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        bright.append(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel))
        dark.append(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
    return np.maximum.reduce(bright).astype(np.float32), np.maximum.reduce(dark).astype(np.float32)


def _resize(channel: np.ndarray, *, binary: bool = False) -> np.ndarray:
    interp = cv2.INTER_NEAREST if binary else (cv2.INTER_AREA if max(channel.shape) > INPUT_SIZE else cv2.INTER_CUBIC)
    return cv2.resize(channel, (INPUT_SIZE, INPUT_SIZE), interpolation=interp)


def _prepare_feature(
    rgb: np.ndarray, box: Mapping[str, Any], *, gray_full: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise NativeSurfaceVerifierV2Error("Ожидается непустое изображение RGB H×W×3")
    h, w = rgb.shape[:2]
    bp = _box_to_px(box, (h, w))
    if bp is None:
        return None
    crop_bounds = _context_crop_bounds(bp, (h, w))
    x0, y0, x1, y1 = crop_bounds
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None
    crop = rgb[y0:y1, x0:x1]
    if gray_full is None:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    else:
        if gray_full.ndim != 2 or gray_full.shape[:2] != (h, w):
            raise NativeSurfaceVerifierV2Error("Размер кэшированного grayscale не совпадает с RGB")
        gray = gray_full[y0:y1, x0:x1]
    bright, dark = _morphology(gray)
    scale = max(10.0, float(np.percentile(np.maximum(bright, dark), 97)))
    bright_u8 = np.clip(bright / scale * 255.0, 0, 255).astype(np.uint8)
    dark_u8 = np.clip(dark / scale * 255.0, 0, 255).astype(np.uint8)

    sobx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sobx, soby)
    gscale = max(12.0, float(np.percentile(grad, 97)))
    edge_u8 = np.clip(grad / gscale * 255.0, 0, 255).astype(np.uint8)

    gf = gray.astype(np.float32)
    mean = cv2.GaussianBlur(gf, (0, 0), 2.0)
    mean2 = cv2.GaussianBlur(gf * gf, (0, 0), 2.0)
    local_std = np.sqrt(np.maximum(0.0, mean2 - mean * mean))
    tscale = max(5.0, float(np.percentile(local_std, 95)))
    texture_u8 = np.clip(local_std / tscale * 255.0, 0, 255).astype(np.uint8)

    prior = _local_prior(box, (h, w), crop_bounds)
    prior_small = (_resize(prior * 255, binary=True) > 0).astype(np.float32)
    if int(prior_small.sum()) < 2:
        return None
    features = np.stack(
        [
            _resize(gray), _resize(bright_u8), _resize(dark_u8),
            _resize(edge_u8), _resize(texture_u8), prior_small * 255.0,
        ],
        axis=0,
    ).astype(np.float32) / 255.0
    gray_mean = float(features[0].mean())
    gray_std = max(float(features[0].std()), 0.08)
    features[0] = np.clip((features[0] - gray_mean) / gray_std, -3.0, 3.0) / 3.0
    return features, prior_small


def suggestion_threshold() -> float:
    _, metadata = _bundle()
    gate = metadata.get("precision_gate", {}) if isinstance(metadata, dict) else {}
    try:
        threshold = float(gate.get("threshold", DEFAULT_SUGGESTION_THRESHOLD)) if isinstance(gate, dict) else DEFAULT_SUGGESTION_THRESHOLD
    except (TypeError, ValueError, OverflowError):
        threshold = DEFAULT_SUGGESTION_THRESHOLD
    if not np.isfinite(threshold):
        threshold = DEFAULT_SUGGESTION_THRESHOLD
    # Never silently loosen below the conservative built-in floor even if a stale
    # experimental metadata file is accidentally packaged.
    return float(np.clip(max(DEFAULT_SUGGESTION_THRESHOLD, threshold), 0.50, 0.995))


def predict_candidates(
    rgb: np.ndarray,
    boxes: Sequence[Mapping[str, Any]],
    *,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    if not boxes:
        return []
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise NativeSurfaceVerifierV2Error("Ожидается непустое изображение RGB H×W×3")
    # Converting the entire frame once wins when many overlapping context crops
    # are evaluated, but wastes work for the common one/few-candidate case.
    gray_full = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if len(boxes) >= 8 else None
    prepared: list[tuple[np.ndarray, np.ndarray] | None] = [
        _prepare_feature(rgb, box, gray_full=gray_full) for box in boxes
    ]
    valid_indices = [i for i, item in enumerate(prepared) if item is not None]
    outputs: list[dict[str, Any] | None] = [None] * len(boxes)
    threshold = suggestion_threshold()
    batch_size = max(1, min(16, int(batch_size)))
    try:
        context_meta_batch = predict_context_meta_v2_batch(list(boxes), rgb.shape[:2])
    except Exception:
        # Preserve the old per-candidate fault isolation if the batch helper ever
        # encounters malformed packaged data.
        context_meta_batch = []
    for start in range(0, len(valid_indices), batch_size):
        ids = valid_indices[start:start + batch_size]
        feats = np.stack([prepared[i][0] for i in ids], axis=0).astype(np.float32, copy=False)  # type: ignore[index]
        priors = np.stack([prepared[i][1] for i in ids], axis=0).astype(np.float32, copy=False)  # type: ignore[index]
        logits = _forward(feats)[:, 0]
        denom = np.maximum(priors.sum(axis=(1, 2)), 1.0)
        candidate_logits = (logits * priors).sum(axis=(1, 2)) / denom
        scores = _sigmoid(candidate_logits)
        pixel_probs = _sigmoid(logits)
        for j, idx in enumerate(ids):
            score = float(scores[j])
            label = "defect" if score >= threshold else "uncertain"
            confidence = score if label == "defect" else 0.50
            support = priors[j] > 0
            strong_fraction = float(np.mean(pixel_probs[j][support] >= 0.5)) if np.any(support) else 0.0
            if idx < len(context_meta_batch):
                context_meta = context_meta_batch[idx]
            else:
                try:
                    context_meta = predict_context_meta_v2(boxes[idx], rgb.shape[:2])
                except Exception as exc:
                    context_meta = {
                        "defect_probability": 0.5, "threshold": 1.0, "supports_defect": False,
                        "model_id": "native_surface_context_meta_v2", "reason": str(exc),
                    }
            outputs[idx] = {
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "training_kind": TRAINING_KIND,
                "contract_id": CONTRACT_ID,
                "label": label,
                "raw_label": "defect" if score >= 0.5 else "natural_detail",
                "confidence": confidence,
                "defect_probability": score,
                "candidate_mask_positive_fraction": strong_fraction,
                "suggestion_threshold": threshold,
                "context_meta_model_id": str(context_meta.get("model_id", "native_surface_context_meta_v2")),
                "context_meta_probability": float(context_meta.get("defect_probability", 0.5)),
                "context_meta_threshold": float(context_meta.get("threshold", 1.0)),
                "context_meta_supports_defect": bool(context_meta.get("supports_defect", False)),
                "experimental": True,
                "expert_verified": False,
                "automatic_edits": False,
            }
    for i, item in enumerate(outputs):
        if item is None:
            outputs[i] = {
                "model_id": MODEL_ID,
                "model_version": MODEL_VERSION,
                "training_kind": TRAINING_KIND,
                "contract_id": CONTRACT_ID,
                "label": "uncertain",
                "raw_label": "uncertain",
                "confidence": 0.0,
                "defect_probability": 0.5,
                "suggestion_threshold": threshold,
                "experimental": True,
                "expert_verified": False,
                "automatic_edits": False,
                "reason": "Некорректная или слишком малая геометрия кандидата.",
            }
    return [item for item in outputs if item is not None]


def model_metadata() -> dict[str, Any]:
    _, training = _bundle()
    gate = training.get("precision_gate", {}) if isinstance(training, dict) else {}
    return {
        "model_id": MODEL_ID,
        "version": MODEL_VERSION,
        "task": TASK,
        "contract_id": CONTRACT_ID,
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "input_channels": ["gray", "bright_morphology", "dark_morphology", "edge_magnitude", "local_texture", "candidate_prior"],
        "labels": ["defect", "uncertain"],
        "provider": "native_numpy",
        "training_kind": TRAINING_KIND,
        "parameters": int(training.get("parameters", PARAMETERS)) if isinstance(training, dict) else PARAMETERS,
        "weights_bytes": _MODEL_PATH.stat().st_size if _MODEL_PATH.is_file() else None,
        "suggestion_threshold": suggestion_threshold(),
        "weak_real_oof": training.get("oof", {}) if isinstance(training, dict) else {},
        "weak_real_precision_gate": gate,
        "expert_verified": False,
        "training_lane": "hard_mining_candidate",
        "experimental": True,
        "automatic_edits": False,
    }
