from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from .sr_backend import choose_sr_provider


RESTORATION_CONTRACT_ID = "rgb01_nchw_static256_x1_v1"


class RestorationBackendError(RuntimeError):
    pass


def _raised_cosine_weights(length: int, overlap: int) -> np.ndarray:
    weights = np.ones((length,), dtype=np.float32)
    overlap = max(0, min(int(overlap), length // 2))
    if overlap <= 0:
        return weights
    phase = np.linspace(0.0, np.pi / 2.0, overlap, endpoint=False, dtype=np.float32)
    ramp = np.clip(np.sin(phase) ** 2, 1e-3, 1.0)
    weights[:overlap] = ramp
    weights[-overlap:] = ramp[::-1]
    return weights


class OnnxX1RestorationBackend:
    """Generic ALMAZ x1 restoration backend with overlap-tiled inference.

    Contract: one RGB float32 tensor in [0,1], NCHW, batch=1, static
    256x256 input and one same-size RGB output.  Static tiles make the same
    exported model easier to run on CUDA/DirectML/OpenVINO and to quantize later
    for QNN without relying on dynamic-shape support.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        compatible_providers: Iterable[str] = (
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ),
        preferred_provider: str | None = None,
        session_factory: Callable[..., Any] | None = None,
        input_size: int = 256,
    ) -> None:
        self.model_path = Path(model_path)
        self.input_size = max(64, int(input_size))
        if session_factory is None and not self.model_path.is_file():
            raise RestorationBackendError(f"Restoration model not found: {self.model_path}")
        choice = choose_sr_provider(compatible_providers, preferred=preferred_provider)
        if choice is None and session_factory is None:
            raise RestorationBackendError("Нет совместимого ONNX execution provider для ALMAZ restoration-модели.")
        self.provider = choice.provider if choice is not None else (preferred_provider or "TESTExecutionProvider")

        if session_factory is None:
            try:
                import onnxruntime as ort  # type: ignore
            except Exception as exc:
                raise RestorationBackendError(f"ONNX Runtime недоступен: {exc}") from exc
            session_factory = ort.InferenceSession

        try:
            self.session = session_factory(str(self.model_path), providers=[self.provider])
        except TypeError:
            self.session = session_factory(str(self.model_path), [self.provider])
        except Exception as exc:
            raise RestorationBackendError(f"Не удалось открыть ALMAZ restoration-модель: {exc}") from exc

        inputs = list(self.session.get_inputs())
        outputs = list(self.session.get_outputs())
        if len(inputs) != 1 or len(outputs) != 1:
            raise RestorationBackendError("Restoration contract expects exactly one input and one output tensor.")
        self.input_name = str(inputs[0].name)
        self.output_name = str(outputs[0].name)

    def _run_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        if tile_rgb.ndim != 3 or tile_rgb.shape[2] != 3 or tile_rgb.dtype != np.uint8:
            raise RestorationBackendError("Restoration input must be uint8 RGB HxWx3.")
        h, w = tile_rgb.shape[:2]
        if h > self.input_size or w > self.input_size:
            raise RestorationBackendError(
                f"Restoration tile {w}x{h} exceeds static model input {self.input_size}x{self.input_size}."
            )
        pad_h = self.input_size - h
        pad_w = self.input_size - w
        border = cv2.BORDER_REFLECT_101 if h > 1 and w > 1 else cv2.BORDER_REPLICATE
        padded = cv2.copyMakeBorder(tile_rgb, 0, pad_h, 0, pad_w, border)
        tensor = np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        try:
            result = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except Exception as exc:
            raise RestorationBackendError(f"ALMAZ restoration inference failed: {exc}") from exc
        arr = np.asarray(result)
        if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] != 3:
            raise RestorationBackendError(f"Unexpected restoration output shape: {arr.shape}")
        if arr.shape[2:] != (self.input_size, self.input_size):
            raise RestorationBackendError(
                f"Static x1 model output must be {self.input_size}x{self.input_size}, "
                f"got {arr.shape[3]}x{arr.shape[2]}."
            )
        hwc = np.transpose(arr[0], (1, 2, 0))[:h, :w]
        return np.clip(np.rint(hwc * 255.0), 0, 255).astype(np.uint8)

    def restore(
        self, rgb: np.ndarray, *, tile_size: int | None = None, overlap: int = 24,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise RestorationBackendError("Restoration input must be uint8 RGB HxWx3.")
        h, w = rgb.shape[:2]
        tile_size = self.input_size if tile_size is None else min(self.input_size, max(64, int(tile_size)))
        overlap = max(0, min(int(overlap), tile_size // 3))
        if h <= tile_size and w <= tile_size:
            out = self._run_tile(rgb)
            if progress is not None:
                progress(1, 1)
            return out

        step = max(16, tile_size - overlap * 2)
        y_starts = list(range(0, max(1, h - tile_size + 1), step))
        x_starts = list(range(0, max(1, w - tile_size + 1), step))
        if not y_starts or y_starts[-1] != max(0, h - tile_size):
            y_starts.append(max(0, h - tile_size))
        if not x_starts or x_starts[-1] != max(0, w - tile_size):
            x_starts.append(max(0, w - tile_size))
        y_starts = sorted(set(y_starts))
        x_starts = sorted(set(x_starts))

        accum = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = np.zeros((h, w, 1), dtype=np.float32)
        total_tiles = len(y_starts) * len(x_starts)
        done_tiles = 0
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(h, y0 + tile_size)
                x1 = min(w, x0 + tile_size)
                tile = rgb[y0:y1, x0:x1]
                out = self._run_tile(tile).astype(np.float32)
                oh, ow = out.shape[:2]
                wy = _raised_cosine_weights(oh, min(overlap, oh // 2))
                wx = _raised_cosine_weights(ow, min(overlap, ow // 2))
                weights = (wy[:, None] * wx[None, :])[..., None]
                accum[y0:y0 + oh, x0:x0 + ow] += out * weights
                weight_sum[y0:y0 + oh, x0:x0 + ow] += weights
                done_tiles += 1
                if progress is not None:
                    progress(done_tiles, total_tiles)
        if np.any(weight_sum <= 0):
            raise RestorationBackendError("Restoration tiling left uncovered pixels.")
        return np.clip(np.rint(accum / weight_sum), 0, 255).astype(np.uint8)
