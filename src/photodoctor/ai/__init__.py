"""Optional local-AI support for Photo Doctor."""

from .manager import AIModelManager, ModelImportError, ModelSpec, ModelState
from .router import AIRoute, AIRouter
from .executor import AIExecutionError, AIInferenceResult, ONNXExecutor, prepare_rgb01_nchw

__all__ = [
    "AIModelManager", "ModelImportError", "ModelSpec", "ModelState",
    "AIRoute", "AIRouter",
    "AIExecutionError", "AIInferenceResult", "ONNXExecutor", "prepare_rgb01_nchw",
]
