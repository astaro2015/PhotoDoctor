from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_ID = "native_surface_context_meta_v2"
MODEL_VERSION = "2.1.0-candidate"
CONTRACT_ID = "surface_context8_rf_candidate_v2"
FEATURE_COUNT = 8
DEFAULT_THRESHOLD = 0.35
_MODEL_PATH = Path(__file__).resolve().parent / "models" / "native_surface_context_meta_v2.npz"


class SurfaceContextMetaError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _bundle() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        with np.load(_MODEL_PATH, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files if key != "metadata_json"}
            raw = str(np.asarray(data["metadata_json"]).item()) if "metadata_json" in data.files else "{}"
        meta = json.loads(raw)
        if not isinstance(meta, dict):
            meta = {}
        return arrays, meta
    except Exception as exc:  # pragma: no cover - packaging tests cover the asset
        raise SurfaceContextMetaError(f"Не удалось загрузить Context Meta Verifier: {exc}") from exc


def suggestion_threshold() -> float:
    _, meta = _bundle()
    gate = meta.get("precision_gate", {}) if isinstance(meta, dict) else {}
    try:
        value = float(gate.get("threshold", DEFAULT_THRESHOLD)) if isinstance(gate, dict) else DEFAULT_THRESHOLD
    except (TypeError, ValueError, OverflowError):
        value = DEFAULT_THRESHOLD
    if not np.isfinite(value):
        value = DEFAULT_THRESHOLD
    # Do not silently loosen a packaged/stale experimental model below the built-in floor.
    return float(np.clip(max(0.35, value), 0.20, 0.995))


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _context_side_px(box: Mapping[str, Any], image_shape: tuple[int, int]) -> tuple[float, float, float, float]:
    h, w = image_shape
    x = _safe_float(box.get("x")); y = _safe_float(box.get("y")); bw_n = _safe_float(box.get("w")); bh_n = _safe_float(box.get("h"))
    if not np.isfinite([x, y, bw_n, bh_n]).all() or bw_n <= 0 or bh_n <= 0 or h <= 0 or w <= 0:
        return 1.0, 1.0, 1.0, 1.0
    # Match the dataset builder exactly: normalized box -> rounded pixel box ->
    # rounded/clipped context square. Tiny geometry drift is surprisingly good at
    # teaching a forest the wrong lesson, because apparently even trees enjoy loopholes.
    bx0 = max(0, min(w, int(round(x * w))))
    by0 = max(0, min(h, int(round(y * h))))
    bx1 = max(0, min(w, int(round((x + bw_n) * w))))
    by1 = max(0, min(h, int(round((y + bh_n) * h))))
    bw = float(max(1, bx1 - bx0)); bh = float(max(1, by1 - by0))
    side = max(40.0, max(bw, bh) * 3.0, min(h, w) * 0.07)
    side = min(side, max(h, w) * 0.48)
    cx = (bx0 + bx1) / 2.0; cy = (by0 + by1) / 2.0
    x0 = max(0, int(round(cx - side / 2.0))); x1 = min(w, int(round(cx + side / 2.0)))
    y0 = max(0, int(round(cy - side / 2.0))); y1 = min(h, int(round(cy + side / 2.0)))
    return bw, bh, float(max(1, x1 - x0)), float(max(1, y1 - y0))


def feature_vector(box: Mapping[str, Any], image_shape: tuple[int, int]) -> np.ndarray:
    arrays, _ = _bundle()
    med = np.asarray(arrays.get("feature_medians", np.zeros(FEATURE_COUNT, np.float32)), dtype=np.float32)
    if med.shape != (FEATURE_COUNT,):
        med = np.zeros(FEATURE_COUNT, np.float32)
    h, w = image_shape
    bw = _safe_float(box.get("w")) * float(w); bh = _safe_float(box.get("h")) * float(h)
    aspect = max(bw / bh, bh / bw) if np.isfinite([bw, bh]).all() and bw > 0 and bh > 0 else float("nan")
    kind = str(box.get("candidate_kind", ""))
    polarity = str(box.get("polarity", ""))
    # IMPORTANT: these features are deliberately independent of the weak-pair
    # label rule. candidate_quality and parallel_neighbor_risk are excluded
    # because they participated in candidate admission / weak-label creation.
    # Keeping them here would let the verifier learn the teacher's rubric instead
    # of the visual/context distinction we actually need at inference time.
    values = np.asarray([
        _safe_float(box.get("context_contrast")),
        _safe_float(box.get("texture_risk")),
        aspect,
        float(kind == "line"),
        float(kind == "spot"),
        float(kind == "irregular"),
        float(polarity == "bright"),
        float(polarity == "dark"),
    ], dtype=np.float32)
    bad = ~np.isfinite(values)
    values[bad] = med[bad]
    return values


def _tree_probability(features: np.ndarray, arrays: Mapping[str, np.ndarray], start: int, end: int) -> float:
    left = arrays["children_left"]; right = arrays["children_right"]
    feat = arrays["feature"]; threshold = arrays["threshold"]; prob = arrays["leaf_probability"]
    local = 0
    max_steps = max(8, end - start + 2)
    for _ in range(max_steps):
        idx = start + local
        if idx < start or idx >= end:
            return 0.5
        fi = int(feat[idx])
        if fi < 0 or int(left[idx]) < 0 or int(right[idx]) < 0:
            return float(np.clip(prob[idx], 0.0, 1.0))
        local = int(left[idx]) if float(features[fi]) <= float(threshold[idx]) else int(right[idx])
    return 0.5



def _tree_probabilities_batch(features: np.ndarray, arrays: Mapping[str, np.ndarray], start: int, end: int) -> np.ndarray:
    """Vectorized equivalent of ``_tree_probability`` for many candidate rows.

    Tree traversal is independent per candidate.  Keeping the tree loop in Python
    while evaluating all candidates in one NumPy operation removes tens of
    thousands of tiny Python calls without changing thresholds or tree decisions.
    """
    rows = int(features.shape[0])
    out = np.full(rows, 0.5, dtype=np.float64)
    if rows <= 0:
        return out
    left = arrays["children_left"]; right = arrays["children_right"]
    feat = arrays["feature"]; threshold = arrays["threshold"]; prob = arrays["leaf_probability"]
    local = np.zeros(rows, dtype=np.int32)
    active = np.ones(rows, dtype=bool)
    max_steps = max(8, end - start + 2)
    row_index = np.arange(rows, dtype=np.int32)
    for _ in range(max_steps):
        if not np.any(active):
            break
        idx = start + local
        valid = active & (idx >= start) & (idx < end)
        if not np.all(valid[active]):
            active &= valid
            if not np.any(active):
                break
        idx = start + local
        fi = feat[idx]
        leaf = active & ((fi < 0) | (left[idx] < 0) | (right[idx] < 0))
        if np.any(leaf):
            out[leaf] = np.clip(prob[idx[leaf]], 0.0, 1.0).astype(np.float64, copy=False)
            active[leaf] = False
        branch = row_index[active]
        if branch.size:
            node_idx = idx[branch]
            feature_idx = fi[branch].astype(np.intp, copy=False)
            go_left = features[branch, feature_idx] <= threshold[node_idx]
            local[branch] = np.where(go_left, left[node_idx], right[node_idx]).astype(np.int32, copy=False)
    return out


def predict_boxes(boxes: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...], image_shape: tuple[int, int]) -> list[dict[str, Any]]:
    """Batch form of :func:`predict_box` with identical model semantics."""
    if not boxes:
        return []
    arrays, meta = _bundle()
    offsets = np.asarray(arrays.get("tree_offsets", []), dtype=np.int32)
    if offsets.ndim != 1 or len(offsets) < 2:
        raise SurfaceContextMetaError("Повреждён встроенный лес Context Meta Verifier")
    features = np.stack([feature_vector(box, image_shape) for box in boxes], axis=0).astype(np.float32, copy=False)
    tree_count = len(offsets) - 1
    scores = np.empty((len(boxes), tree_count), dtype=np.float64)
    for i in range(tree_count):
        scores[:, i] = _tree_probabilities_batch(features, arrays, int(offsets[i]), int(offsets[i + 1]))
    probabilities = np.mean(scores, axis=1) if tree_count else np.full(len(boxes), 0.5, dtype=np.float64)
    threshold = suggestion_threshold()
    expert_verified = bool(meta.get("expert_verified", False)) if isinstance(meta, dict) else False
    return [{
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "contract_id": CONTRACT_ID,
        "defect_probability": float(probability),
        "threshold": threshold,
        "supports_defect": bool(float(probability) >= threshold),
        "expert_verified": expert_verified,
        "experimental": True,
        "automatic_edits": False,
    } for probability in probabilities]

def predict_box(box: Mapping[str, Any], image_shape: tuple[int, int]) -> dict[str, Any]:
    arrays, meta = _bundle()
    offsets = np.asarray(arrays.get("tree_offsets", []), dtype=np.int32)
    if offsets.ndim != 1 or len(offsets) < 2:
        raise SurfaceContextMetaError("Повреждён встроенный лес Context Meta Verifier")
    features = feature_vector(box, image_shape)
    scores = [_tree_probability(features, arrays, int(offsets[i]), int(offsets[i + 1])) for i in range(len(offsets) - 1)]
    probability = float(np.mean(scores)) if scores else 0.5
    threshold = suggestion_threshold()
    return {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "contract_id": CONTRACT_ID,
        "defect_probability": probability,
        "threshold": threshold,
        "supports_defect": bool(probability >= threshold),
        "expert_verified": bool(meta.get("expert_verified", False)) if isinstance(meta, dict) else False,
        "experimental": True,
        "automatic_edits": False,
    }
