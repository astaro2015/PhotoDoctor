from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from .manager_types import ModelState


class AIExecutionError(RuntimeError):
    pass


@dataclass(slots=True)
class AIInferenceResult:
    model_id: str
    task: str
    contract_id: str
    confidence: float
    data: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        result = {
            "model_id": self.model_id,
            "task": self.task,
            "contract_id": self.contract_id,
            "confidence": float(self.confidence),
        }
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                result[key + "_shape"] = list(value.shape)
            elif isinstance(value, np.generic):
                result[key] = value.item()
            else:
                result[key] = value
        return result


def prepare_rgb01_nchw(rgb: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    if not isinstance(rgb, np.ndarray) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise AIExecutionError("Вход ИИ должен быть массивом RGB H×W×3.")
    if rgb.size == 0:
        raise AIExecutionError("Входное изображение ИИ пустое.")
    if not np.isfinite(rgb.astype(np.float32, copy=False)).all():
        raise AIExecutionError("Вход ИИ содержит недопустимые значения NaN/Inf.")

    target_w, target_h = int(input_size[0]), int(input_size[1])
    if target_w <= 0 or target_h <= 0:
        raise AIExecutionError("Некорректный размер входа модели.")

    arr = rgb.astype(np.float32)
    if np.issubdtype(rgb.dtype, np.integer):
        arr /= 255.0
    elif arr.max(initial=0.0) > 1.5:
        arr /= 255.0
    arr = np.clip(arr, 0.0, 1.0)

    h, w = arr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = max(1, min(target_w, int(round(w * scale))))
    new_h = max(1, min(target_h, int(round(h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
    canvas = np.full((target_h, target_w, 3), 0.5, dtype=np.float32)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    tensor = np.transpose(canvas, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exps = np.exp(shifted)
    denom = float(exps.sum())
    if denom <= 0.0 or not np.isfinite(denom):
        raise AIExecutionError("Выход классификации не удаётся нормализовать.")
    return exps / denom


class ONNXExecutor:
    """Strict executor for Photo Doctor's own micro-model contracts.

    It is not enabled by the application in 0.2.2 yet. The class is built and
    tested first so malformed tensors/models cannot silently influence metrics.
    """

    def __init__(self, session_factory: Callable[[str], Any] | None = None):
        self._session_factory = session_factory

    @staticmethod
    def _default_session_factory(path: str):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise AIExecutionError("Среда ONNX Runtime не установлена.") from exc
        try:
            return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        except Exception as exc:  # pragma: no cover - runtime-specific
            raise AIExecutionError(f"Не удалось создать сеанс ONNX: {exc}") from exc

    def _session(self, path: str):
        factory = self._session_factory or self._default_session_factory
        return factory(path)

    def run(self, state: ModelState, rgb: np.ndarray) -> AIInferenceResult:
        if not state.ready:
            raise AIExecutionError(f"Модель не готова: {state.status}")
        spec = state.spec
        if spec.contract_id not in {
            "rgb01_nchw_classification_v1",
            "rgb01_nchw_scalar_v1",
            "rgb01_nchw_segmentation_v1",
        }:
            raise AIExecutionError(f"Неподдерживаемый контракт модели: {spec.contract_id}")

        tensor = prepare_rgb01_nchw(rgb, spec.input_size)
        session = self._session(str(state.path))
        try:
            inputs = list(session.get_inputs())
        except Exception as exc:
            raise AIExecutionError(f"Не удалось проверить вход модели: {exc}") from exc
        if len(inputs) != 1 or not getattr(inputs[0], "name", None):
            raise AIExecutionError("Контракт модели требует ровно один именованный вход.")
        input_name = inputs[0].name
        try:
            outputs = session.run(None, {input_name: tensor})
        except Exception as exc:
            raise AIExecutionError(f"Запуск модели ONNX завершился ошибкой: {exc}") from exc
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise AIExecutionError("Контракт модели требует ровно один выходной тензор.")
        output = np.asarray(outputs[0], dtype=np.float32)
        if output.size == 0 or not np.isfinite(output).all():
            raise AIExecutionError("Выход модели пуст или содержит недопустимые NaN/Inf.")

        if spec.output_kind == "classification":
            values = output.reshape(-1)
            if len(spec.labels) == 0 or values.size != len(spec.labels):
                raise AIExecutionError(
                    f"Размер выхода классификации {values.size} не совпадает с {len(spec.labels)} метками."
                )
            looks_like_prob = bool(
                np.all(values >= 0.0)
                and np.all(values <= 1.0)
                and abs(float(values.sum()) - 1.0) <= 1e-3
            )
            probabilities = values if looks_like_prob else _softmax(values)
            index = int(np.argmax(probabilities))
            probs = {label: float(probabilities[i]) for i, label in enumerate(spec.labels)}
            return AIInferenceResult(
                spec.model_id,
                spec.task,
                spec.contract_id,
                float(probabilities[index]),
                {"label": spec.labels[index], "probabilities": probs},
            )

        if spec.output_kind == "scalar_0_1":
            values = output.reshape(-1)
            if values.size != 1:
                raise AIExecutionError("Скалярный контракт требует ровно одно выходное значение.")
            score = float(values[0])
            if not (0.0 <= score <= 1.0):
                raise AIExecutionError("Выход скалярного контракта должен находиться в диапазоне [0,1].")
            return AIInferenceResult(
                spec.model_id,
                spec.task,
                spec.contract_id,
                0.0,
                {
                    "score_0_1": score,
                    "score_0_100": score * 100.0,
                    "confidence_available": False,
                },
            )

        if spec.output_kind == "segmentation":
            squeezed = np.squeeze(output)
            if squeezed.ndim != 2:
                raise AIExecutionError("Контракт сегментации требует двумерную маску после удаления осей пакета и канала.")
            if float(squeezed.min()) < 0.0 or float(squeezed.max()) > 1.0:
                raise AIExecutionError("Значения маски сегментации должны находиться в диапазоне [0,1].")
            coverage = float(np.mean(squeezed >= 0.5) * 100.0)
            confidence = float(np.mean(np.abs(squeezed - 0.5) * 2.0))
            return AIInferenceResult(
                spec.model_id,
                spec.task,
                spec.contract_id,
                confidence,
                {"coverage_pct": coverage, "mask": squeezed.astype(np.float32)},
            )

        raise AIExecutionError(f"Неподдерживаемый тип выхода: {spec.output_kind}")


def _dim_is_dynamic(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _shape_matches_input(shape: Any, input_size: tuple[int, int]) -> bool:
    if not isinstance(shape, (list, tuple)) or len(shape) != 4:
        return False
    target_w, target_h = input_size
    expected = (1, 3, target_h, target_w)
    for actual, wanted in zip(shape, expected):
        if _dim_is_dynamic(actual):
            continue
        try:
            if int(actual) != wanted:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _static_product(shape: Any) -> int | None:
    if not isinstance(shape, (list, tuple)):
        return None
    product = 1
    for dim in shape:
        if _dim_is_dynamic(dim):
            return None
        try:
            product *= int(dim)
        except (TypeError, ValueError):
            return None
    return product


def preflight_onnx_contract(
    path: str,
    spec,
    session_factory: Callable[[str], Any] | None = None,
) -> tuple[bool, str]:
    """Validate ONNX graph IO against the declared Photo Doctor contract.

    This creates a session but performs no inference. Callers should cache the
    result by file SHA-256; session construction can be expensive.
    """
    factory = session_factory or ONNXExecutor._default_session_factory
    try:
        session = factory(path)
    except AIExecutionError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, f"Не удалось открыть ONNX для предварительной проверки: {exc}"

    try:
        inputs = list(session.get_inputs())
        outputs = list(session.get_outputs())
    except Exception as exc:
        return False, f"Не удалось прочитать входы/выходы ONNX: {exc}"
    if len(inputs) != 1:
        return False, f"Контракт требует 1 вход, модель имеет {len(inputs)}."
    if len(outputs) != 1:
        return False, f"Контракт требует 1 выход, модель имеет {len(outputs)}."

    inp = inputs[0]
    out = outputs[0]
    input_type = str(getattr(inp, "type", ""))
    output_type = str(getattr(out, "type", ""))
    if input_type and "float" not in input_type.lower():
        return False, f"Вход должен быть тензором float, получено {input_type}."
    if output_type and "float" not in output_type.lower():
        return False, f"Выход должен быть тензором float, получено {output_type}."
    if not _shape_matches_input(getattr(inp, "shape", None), spec.input_size):
        return False, (
            f"Входная форма {getattr(inp, 'shape', None)} не соответствует "
            f"NCHW [1,3,{spec.input_size[1]},{spec.input_size[0]}] (динамические размеры допустимы)."
        )

    output_shape = getattr(out, "shape", None)
    if spec.output_kind == "classification":
        if not spec.labels:
            return False, "Контракт классификации не содержит меток классов."
        if not isinstance(output_shape, (list, tuple)) or len(output_shape) not in {1, 2}:
            return False, f"Выход классификации должен быть [C] или [1,C], получено {output_shape}."
        last = output_shape[-1]
        if not _dim_is_dynamic(last):
            try:
                if int(last) != len(spec.labels):
                    return False, f"Число выходных классов {last} не равно числу меток={len(spec.labels)}."
            except (TypeError, ValueError):
                return False, f"Некорректная выходная форма: {output_shape}."
    elif spec.output_kind == "scalar_0_1":
        product = _static_product(output_shape)
        if product is not None and product != 1:
            return False, f"Скалярный выход должен содержать 1 значение, получена форма={output_shape}."
        if isinstance(output_shape, (list, tuple)) and len(output_shape) > 2:
            return False, f"Скалярный выход имеет неподходящую форму {output_shape}."
    elif spec.output_kind == "segmentation":
        if not isinstance(output_shape, (list, tuple)) or len(output_shape) != 4:
            return False, f"Выход сегментации должен быть NCHW [1,1,H,W], получено {output_shape}."
        batch, channels, _, _ = output_shape
        for value, expected, label in ((batch, 1, "batch"), (channels, 1, "channels")):
            if not _dim_is_dynamic(value):
                try:
                    if int(value) != expected:
                        return False, f"Сегментация: {label} должен быть {expected}, получено {value}."
                except (TypeError, ValueError):
                    return False, f"Некорректная форма сегментации {output_shape}."
    else:
        return False, f"Неизвестный тип выхода: {spec.output_kind}."

    return True, (
        f"Контракт совместим: вход={getattr(inp, 'shape', None)}, "
        f"output={output_shape}, тип={spec.output_kind}."
    )
