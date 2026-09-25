from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any, Callable, Iterable

import cv2
import numpy as np


SR_CONTRACT_ID = "rgb01_nchw_static256_x2_v1"


class SuperResolutionBackendError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SRProviderStatus:
    provider: str
    label: str
    device_kind: str
    available: bool
    priority: int
    detail: str = ""


_PROVIDER_INFO: dict[str, tuple[str, str, int]] = {
    "QNNExecutionProvider": ("Qualcomm QNN / NPU", "npu", 100),
    "CUDAExecutionProvider": ("NVIDIA CUDA", "gpu", 90),
    "DmlExecutionProvider": ("Windows DirectML", "gpu", 70),
    "OpenVINOExecutionProvider": ("Intel OpenVINO", "accelerator", 65),
    "CPUExecutionProvider": ("ONNX Runtime CPU", "cpu", 30),
}


def _runtime_providers() -> tuple[set[str], str]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return set(), f"ONNX Runtime недоступен: {exc}"
    try:
        providers = {str(value) for value in ort.get_available_providers()}
    except Exception as exc:
        return set(), f"Не удалось получить список ONNX providers: {exc}"
    return providers, ""


def discover_sr_providers() -> list[SRProviderStatus]:
    available, runtime_error = _runtime_providers()
    machine = platform.machine().strip().lower()
    system = platform.system().strip().lower()
    statuses: list[SRProviderStatus] = []
    for provider, (label, kind, priority) in _PROVIDER_INFO.items():
        detail = runtime_error
        usable = provider in available
        if provider == "QNNExecutionProvider" and usable:
            # Local Snapdragon NPU inference is a Windows ARM64 deployment path.
            # x64 QNN installations may exist for model preparation/quantization,
            # so do not silently treat them as an NPU-capable target.
            if not (system == "windows" and machine in {"arm64", "aarch64"}):
                usable = False
                detail = "QNN найден, но локальный NPU-профиль включается только на Windows ARM64."
        statuses.append(SRProviderStatus(provider, label, kind, usable, priority, detail))
    statuses.sort(key=lambda item: item.priority, reverse=True)
    return statuses


def choose_sr_provider(
    compatible_providers: Iterable[str] | None = None,
    *,
    preferred: str | None = None,
) -> SRProviderStatus | None:
    compatible = {str(v) for v in compatible_providers} if compatible_providers is not None else None
    statuses = discover_sr_providers()
    if preferred:
        for status in statuses:
            if status.provider == preferred and status.available and (compatible is None or status.provider in compatible):
                return status
    for status in statuses:
        if status.available and (compatible is None or status.provider in compatible):
            return status
    return None


def _raised_cosine_weights(length: int, overlap: int) -> np.ndarray:
    weights = np.ones((length,), dtype=np.float32)
    overlap = max(0, min(int(overlap), length // 2))
    if overlap <= 0:
        return weights
    # Never reaches zero: adjacent tiles overlap and are normalized after sum.
    phase = np.linspace(0.0, np.pi / 2.0, overlap, endpoint=False, dtype=np.float32)
    ramp = np.sin(phase) ** 2
    ramp = np.clip(ramp, 1e-3, 1.0)
    weights[:overlap] = ramp
    weights[-overlap:] = ramp[::-1]
    return weights


class OnnxX2SuperResolutionBackend:
    """Generic x2 ONNX backend with overlap-tiled inference.

    Model contract: one RGB float32 input in [0,1], NCHW, batch=1, and one
    float32 RGB output at exactly 2x spatial resolution.  Provider-specific model
    compatibility is supplied explicitly by the caller; QNN is never assumed to
    accept an arbitrary float model.
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
            raise SuperResolutionBackendError(f"SR model not found: {self.model_path}")
        choice = choose_sr_provider(compatible_providers, preferred=preferred_provider)
        if choice is None and session_factory is None:
            raise SuperResolutionBackendError("Нет совместимого ONNX execution provider для SR-модели.")
        self.provider = choice.provider if choice is not None else (preferred_provider or "TESTExecutionProvider")

        if session_factory is None:
            try:
                import onnxruntime as ort  # type: ignore
            except Exception as exc:
                raise SuperResolutionBackendError(f"ONNX Runtime недоступен: {exc}") from exc
            session_factory = ort.InferenceSession

        try:
            self.session = session_factory(str(self.model_path), providers=[self.provider])
        except TypeError:
            # Simple test doubles may only accept the model path/provider list as
            # positional/partial arguments.
            self.session = session_factory(str(self.model_path), [self.provider])
        except Exception as exc:
            raise SuperResolutionBackendError(f"Не удалось открыть SR-модель: {exc}") from exc

        inputs = list(self.session.get_inputs())
        outputs = list(self.session.get_outputs())
        if len(inputs) != 1 or len(outputs) != 1:
            raise SuperResolutionBackendError("SR contract expects exactly one input and one output tensor.")
        self.input_name = str(inputs[0].name)
        self.output_name = str(outputs[0].name)

    def _run_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        if tile_rgb.ndim != 3 or tile_rgb.shape[2] != 3 or tile_rgb.dtype != np.uint8:
            raise SuperResolutionBackendError("SR input must be uint8 RGB HxWx3.")
        h, w = tile_rgb.shape[:2]
        if h > self.input_size or w > self.input_size:
            raise SuperResolutionBackendError(
                f"SR tile {w}x{h} exceeds static model input {self.input_size}x{self.input_size}."
            )
        pad_h = self.input_size - h
        pad_w = self.input_size - w
        border = cv2.BORDER_REFLECT_101 if h > 1 and w > 1 else cv2.BORDER_REPLICATE
        padded = cv2.copyMakeBorder(tile_rgb, 0, pad_h, 0, pad_w, border)
        tensor = np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
        try:
            result = self.session.run([self.output_name], {self.input_name: tensor})[0]
        except Exception as exc:
            raise SuperResolutionBackendError(f"SR inference failed: {exc}") from exc
        arr = np.asarray(result)
        if arr.ndim != 4 or arr.shape[0] != 1 or arr.shape[1] != 3:
            raise SuperResolutionBackendError(f"Unexpected SR output shape: {arr.shape}")
        expected = self.input_size * 2
        if arr.shape[2:] != (expected, expected):
            raise SuperResolutionBackendError(
                f"Static SR model output must be {expected}x{expected}, got {arr.shape[3]}x{arr.shape[2]}."
            )
        hwc = np.transpose(arr[0], (1, 2, 0))[: h * 2, : w * 2]
        return np.clip(np.rint(hwc * 255.0), 0, 255).astype(np.uint8)

    def upscale_x2(
        self, rgb: np.ndarray, *, tile_size: int | None = None, overlap: int = 24,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
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

        accum = np.zeros((h * 2, w * 2, 3), dtype=np.float32)
        weight_sum = np.zeros((h * 2, w * 2, 1), dtype=np.float32)
        total_tiles = len(y_starts) * len(x_starts)
        done_tiles = 0
        for y0 in y_starts:
            for x0 in x_starts:
                y1 = min(h, y0 + tile_size)
                x1 = min(w, x0 + tile_size)
                tile = rgb[y0:y1, x0:x1]
                out = self._run_tile(tile).astype(np.float32)
                oh, ow = out.shape[:2]
                wy = _raised_cosine_weights(oh, min(overlap * 2, oh // 2))
                wx = _raised_cosine_weights(ow, min(overlap * 2, ow // 2))
                weights = (wy[:, None] * wx[None, :])[..., None]
                oy0, ox0 = y0 * 2, x0 * 2
                accum[oy0:oy0 + oh, ox0:ox0 + ow] += out * weights
                weight_sum[oy0:oy0 + oh, ox0:ox0 + ow] += weights
                done_tiles += 1
                if progress is not None:
                    progress(done_tiles, total_tiles)
        if np.any(weight_sum <= 0):
            raise SuperResolutionBackendError("SR tiling left uncovered pixels.")
        return np.clip(np.rint(accum / weight_sum), 0, 255).astype(np.uint8)
