from __future__ import annotations

from typing import Any

import numpy as np

from photodoctor.ai.executor import AIExecutionError, ONNXExecutor
from photodoctor.ai.manager import AIModelManager
from photodoctor.ai.router import AIRouter
from photodoctor.ai.native_surface_refiner_v1 import predict_patches as predict_native_surface_patches
from photodoctor.ai.native_surface_verifier_v2 import predict_candidates as predict_native_surface_candidates_v2
from photodoctor.ai.resources import AIResourcePlan, choose_ai_resource_plan, detect_system_resources

from .models import MetricResult


AI_RUNTIME_VERSION = "0.1.3"


def _face_crop(rgb: np.ndarray, face: dict[str, Any]) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    try:
        x = int(round(float(face.get("x", 0.0)) * w))
        y = int(round(float(face.get("y", 0.0)) * h))
        fw = int(round(float(face.get("w", 0.0)) * w))
        fh = int(round(float(face.get("h", 0.0)) * h))
    except (TypeError, ValueError, OverflowError):
        return None
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    fw = max(0, min(w - x, fw))
    fh = max(0, min(h - y, fh))
    if fw < 16 or fh < 16:
        return None
    return rgb[y:y + fh, x:x + fw]




def _surface_crop(rgb: np.ndarray, box: dict[str, Any], margin: float = 1.55) -> np.ndarray | None:
    h, w = rgb.shape[:2]
    try:
        x = float(box.get("x", 0.0)) * w
        y = float(box.get("y", 0.0)) * h
        bw = float(box.get("w", 0.0)) * w
        bh = float(box.get("h", 0.0)) * h
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite([x, y, bw, bh]).all() or bw <= 0 or bh <= 0:
        return None
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(20.0, max(bw, bh) * margin, min(w, h) * 0.035)
    try:
        x0 = max(0, int(round(cx - side / 2.0)))
        y0 = max(0, int(round(cy - side / 2.0)))
        x1 = min(w, int(round(cx + side / 2.0)))
        y1 = min(h, int(round(cy + side / 2.0)))
    except (TypeError, ValueError, OverflowError):
        return None
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return rgb[y0:y1, x0:x1]


def execute_ai_plan(
    rgb: np.ndarray,
    metrics: dict[str, MetricResult],
    *,
    manager: AIModelManager | None = None,
    executor: ONNXExecutor | None = None,
    max_model_runs: int | None = None,
    resource_plan: AIResourcePlan | None = None,
    precision: str = "normal",
) -> MetricResult:
    """Execute only requested+ready local models and isolate all failures.

    AI results are advisory in this version. They are deliberately stored as a
    separate metric and never modify classical scores or the Decision Engine.
    """
    manager = manager or AIModelManager()
    executor = executor or ONNXExecutor()
    resource_plan = resource_plan or choose_ai_resource_plan(detect_system_resources())
    effective_max_model_runs = resource_plan.max_model_runs if max_model_runs is None else max(0, int(max_model_runs))
    precision_key = str(precision or "normal").strip().lower()
    surface_candidate_limit = int(resource_plan.max_surface_candidates)
    if precision_key == "maximum":
        # Maximum checks roughly twice the previous fourth-mode Surface budget. It keeps
        # the same verifier thresholds; extra runtime buys recall, not weaker truth.
        surface_candidate_limit = min(320, max(surface_candidate_limit * 8, 320))
    elif precision_key == "precise":
        # Precise mode is intentionally allowed to spend time on ambiguity. The user
        # asked for useful depth, not a ceremonial 300 ms delay with a fancier label.
        surface_candidate_limit = min(96, max(surface_candidate_limit * 3, 80))
    elif precision_key == "fast":
        surface_candidate_limit = min(surface_candidate_limit, 14)
    else:
        # Surface v8: the native verifier is lightweight enough that normal mode must
        # not stop after 14 candidates merely because Windows/browser memory pressure
        # selected the conservative global AI profile. On damaged archival photos the
        # faint face/background crack often ranks behind stronger jacket scratches.
        surface_candidate_limit = min(64, max(surface_candidate_limit, 64))
    base_metrics = {
        key: value for key, value in metrics.items()
        if key not in {"ai_status", "ai_inference"}
    }
    router = AIRouter(manager)
    routes = router.plan(base_metrics)
    requested = [route for route in routes if route.requested]
    eligible = [route for route in requested if route.eligible_to_run]
    selected = eligible[:effective_max_model_runs]

    results: list[dict[str, Any]] = []
    attempted = 0
    successful = 0
    face_raw = base_metrics.get("faces")
    faces = []
    if face_raw is not None and isinstance(face_raw.raw_value, dict):
        maybe_faces = face_raw.raw_value.get("faces", [])
        if isinstance(maybe_faces, list):
            faces = [item for item in maybe_faces if isinstance(item, dict)]
    surface_raw = base_metrics.get("surface_defects")
    surface_boxes: list[dict[str, Any]] = []
    if surface_raw is not None and isinstance(surface_raw.raw_value, dict):
        maybe_boxes = surface_raw.raw_value.get("ai_boxes_norm", surface_raw.raw_value.get("boxes_norm", []))
        if isinstance(maybe_boxes, list):
            surface_boxes = [item for item in maybe_boxes if isinstance(item, dict)]

    for route in requested:
        state = router.model_state_for_task(route.task)
        if route not in selected:
            results.append({
                "task": route.task,
                "model_id": route.model_id,
                "status": "not_run",
                "reason": (
                    "Модель не готова." if not route.eligible_to_run
                    else "Превышен лимит AI-моделей на один снимок."
                ),
                "model_status": route.model_status,
            })
            continue
        if state is None:
            results.append({
                "task": route.task,
                "model_id": route.model_id,
                "status": "not_run",
                "reason": "Состояние модели недоступно.",
                "model_status": route.model_status,
            })
            continue

        inputs: list[tuple[str, np.ndarray, dict[str, Any] | None]] = []
        if route.task == "face_quality":
            for index, face in enumerate(faces[:3], start=1):
                crop = _face_crop(rgb, face)
                if crop is not None:
                    inputs.append((f"face_{index}", crop, None))
            if not inputs:
                results.append({
                    "task": route.task,
                    "model_id": route.model_id,
                    "status": "not_run",
                    "reason": "Нет пригодного фрагмента лица для модели ИИ.",
                    "model_status": route.model_status,
                })
                continue
        elif route.task == "surface_defect_refinement":
            for index, box in enumerate(surface_boxes[:surface_candidate_limit], start=1):
                if route.model_id == "native_surface_verifier_v2":
                    # v2 needs the full image so it can rebuild exactly the same 3x-context
                    # crop and candidate prior that were used during training.
                    inputs.append((f"surface_{index}", rgb, box))
                else:
                    crop = _surface_crop(rgb, box)
                    if crop is not None:
                        inputs.append((f"surface_{index}", crop, box))
            if not inputs:
                results.append({
                    "task": route.task,
                    "model_id": route.model_id,
                    "status": "not_run",
                    "reason": "Нет пригодных классических кандидатов дефектов для встроенного уточнителя.",
                    "model_status": route.model_status,
                })
                continue
        else:
            inputs.append(("global", rgb, None))

        task_results: list[dict[str, Any]] = []
        task_failed = False
        if state.spec.provider == "native_numpy" and route.task == "surface_defect_refinement":
            attempted += len(inputs)
            try:
                if state.spec.model_id == "native_surface_verifier_v2":
                    # Maximum can expose hundreds of manual-only Surface candidates.
                    # The native NumPy verifier becomes memory/allocator bound when all
                    # candidate feature tensors are prepared at once. Verify the same
                    # ordered list in bounded chunks; per-candidate math and thresholds
                    # are unchanged, only peak working-set size is capped.
                    surface_boxes_for_ai = [item[2] or {} for item in inputs]
                    payloads = []
                    verification_chunk = 64 if precision_key == "maximum" else len(surface_boxes_for_ai)
                    for chunk_start in range(0, len(surface_boxes_for_ai), max(1, verification_chunk)):
                        payloads.extend(predict_native_surface_candidates_v2(
                            rgb,
                            surface_boxes_for_ai[chunk_start:chunk_start + verification_chunk],
                            batch_size=resource_plan.patch_batch_size,
                        ))
                else:
                    payloads = predict_native_surface_patches(
                        [item[1] for item in inputs],
                        [item[2] for item in inputs],
                        batch_size=resource_plan.patch_batch_size,
                    )
            except Exception as exc:
                task_failed = True
                for region_name, _, _ in inputs:
                    task_results.append({"region": region_name, "status": "error", "error": f"Ошибка встроенного ИИ: {exc}"})
            else:
                if len(payloads) != len(inputs):
                    task_failed = True
                for index, (region_name, _, _) in enumerate(inputs):
                    if index >= len(payloads):
                        task_results.append({
                            "region": region_name,
                            "status": "error",
                            "error": "Встроенный ИИ вернул неполный пакет результатов.",
                        })
                        continue
                    payload = payloads[index]
                    if not isinstance(payload, dict):
                        task_failed = True
                        task_results.append({
                            "region": region_name,
                            "status": "error",
                            "error": "Встроенный ИИ вернул результат неподдерживаемого формата.",
                        })
                        continue
                    successful += 1
                    task_results.append({"region": region_name, "status": "ok", "result": payload})
                if len(payloads) > len(inputs):
                    task_failed = True
        else:
            for region_name, input_rgb, _ in inputs:
                attempted += 1
                try:
                    inference = executor.run(state, input_rgb)
                    payload = inference.to_serializable()
                except AIExecutionError as exc:
                    task_failed = True
                    task_results.append({"region": region_name, "status": "error", "error": str(exc)})
                except Exception as exc:  # last-resort isolation: optional AI must not crash analysis
                    task_failed = True
                    task_results.append({"region": region_name, "status": "error", "error": f"Непредвиденная ошибка ИИ: {exc}"})
                else:
                    successful += 1
                    task_results.append({"region": region_name, "status": "ok", "result": payload})
        results.append({
            "task": route.task,
            "model_id": route.model_id,
            "status": "partial_error" if task_failed else "ok",
            "model_status": route.model_status,
            "regions": task_results,
        })

    confidence = 1.0 if not selected else (successful / max(attempted, 1))
    return MetricResult(
        "ai_inference",
        {
            "runtime_version": AI_RUNTIME_VERSION,
            "advisory_only": True,
            "changes_classical_scores": False,
            "max_model_runs": int(effective_max_model_runs),
            "resource_profile": resource_plan.key,
            "resource_profile_label": resource_plan.label,
            "patch_batch_size": int(resource_plan.patch_batch_size),
            "max_surface_candidates": int(surface_candidate_limit),
            "base_max_surface_candidates": int(resource_plan.max_surface_candidates),
            "precision": precision_key,
            "working_memory_budget_gib": round(resource_plan.working_memory_budget_gib, 2),
            "requested_count": len(requested),
            "eligible_count": len(eligible),
            "selected_model_count": len(selected),
            "attempted_inferences": attempted,
            "successful_inferences": successful,
            "results": results,
        },
        None,
        float(confidence),
        "ai",
        region="ai",
        diagnostic="Результаты локальных микро-моделей ONNX; носят рекомендательный характер и не изменяют классические оценки/Модуль решений",
    )
