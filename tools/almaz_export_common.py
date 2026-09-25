from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ('params_ema', 'params', 'state_dict'):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                return value
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    raise RuntimeError('Не удалось найти state_dict в checkpoint.')


def parity_onnx(
    onnx_path: str | Path,
    input_tensor,
    torch_output,
    *,
    max_abs_limit: float = 7e-4,
    mean_abs_limit: float = 7e-5,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return {
            'status': 'PENDING',
            'detail': f'onnxruntime недоступен для parity: {exc}',
            'max_abs': None,
            'mean_abs': None,
        }
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    inputs = session.get_inputs(); outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        return {'status': 'FAIL', 'detail': 'ONNX contract expects one input and one output.', 'max_abs': None, 'mean_abs': None}
    feed = input_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    got = np.asarray(session.run([outputs[0].name], {inputs[0].name: feed})[0], dtype=np.float32)
    expected = torch_output.detach().cpu().numpy().astype(np.float32, copy=False)
    if got.shape != expected.shape:
        return {
            'status': 'FAIL', 'detail': f'Parity shape mismatch: ONNX {got.shape}, PyTorch {expected.shape}.',
            'max_abs': None, 'mean_abs': None,
        }
    diff = np.abs(got - expected)
    max_abs = float(diff.max(initial=0.0))
    mean_abs = float(diff.mean()) if diff.size else 0.0
    passed = max_abs <= max_abs_limit and mean_abs <= mean_abs_limit
    return {
        'status': 'PASS' if passed else 'FAIL',
        'detail': f'max_abs={max_abs:.8g}, mean_abs={mean_abs:.8g}',
        'max_abs': max_abs,
        'mean_abs': mean_abs,
        'max_abs_limit': max_abs_limit,
        'mean_abs_limit': mean_abs_limit,
    }


def write_manifest(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return p
