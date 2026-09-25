from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from .models import MetricResult

SURFACE_REFINEMENT_VERSION = "0.9.0"


def _clamp01(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(number):
        return default
    return float(np.clip(number, 0.0, 1.0))


def _fuse_verdict(box: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fuse detector context, Context Meta v2 and pixel Surface AI v2.

    Context Meta v2 is the primary verifier because its held-out weak-pair behaviour
    is materially more stable across source groups than the current pixel U-Net.
    The U-Net remains a pixel-level second opinion and localization aid: a strong
    negative can veto, but a merely uncertain pixel score must not erase a strong
    independent context verdict.  The pixel model remains advisory by itself; automatic repair is allowed only
    after the fused verdict also passes the separate face-safe repair guardrail.
    """
    raw_label = str(payload.get("label", "uncertain"))
    if raw_label not in {"defect", "natural_detail", "uncertain"}:
        raw_label = "uncertain"
    raw_conf = _clamp01(payload.get("confidence"), 0.0)
    p_defect = _clamp01(payload.get("defect_probability"), 0.5)
    training_kind = str(payload.get("training_kind", ""))
    expert_verified = bool(payload.get("expert_verified", False))
    synthetic_only = "procedural" in training_kind or not training_kind
    candidate_model = not expert_verified
    polarity = str(box.get("polarity", "bright"))
    candidate_kind = str(box.get("candidate_kind", ""))
    quality = _clamp01(box.get("candidate_quality"), 0.40)
    context_contrast = _clamp01(box.get("context_contrast"), 0.35)
    side_similarity = _clamp01(box.get("side_similarity"), 0.50)
    texture_risk = _clamp01(box.get("texture_risk"), 0.50)
    chroma_risk = _clamp01(box.get("chroma_risk"), 0.20)
    context_support = float(np.clip(0.70 * context_contrast + 0.30 * side_similarity, 0.0, 1.0))
    risk = float(np.clip(0.72 * texture_risk + 0.28 * chroma_risk, 0.0, 1.0))

    meta_probability = _clamp01(payload.get("context_meta_probability"), 0.5)
    meta_threshold = _clamp01(payload.get("context_meta_threshold"), 0.72)
    meta_available = "context_meta_probability" in payload
    meta_supports = bool(payload.get("context_meta_supports_defect", False)) if meta_available else False

    classical_evidence = float(np.clip(0.12 + 0.76 * quality + 0.10 * context_support - 0.16 * risk, 0.02, 0.94))
    if meta_available:
        # Context Meta is intentionally the dominant term. The current pixel U-Net
        # is useful, but its cross-photo stability is weaker and it must not be able
        # to veto a good context candidate merely by being indecisive.
        fused_score = float(np.clip(0.22 * p_defect + 0.56 * meta_probability + 0.22 * classical_evidence, 0.0, 1.0))
    else:
        fused_score = float(np.clip(0.56 * p_defect + 0.44 * classical_evidence, 0.0, 1.0))

    is_v2 = "candidate_pair_group_cv_segmentation_v2" in training_kind
    branch = str(box.get("detection_branch", "primary"))
    meta_margin = meta_probability - meta_threshold
    # The pixel U-Net is weakest on very pale archival cracks over light paper/skin.
    # For that *specific* class, a strong independent context verdict may override a
    # pixel hard-negative only when the geometry is exceptionally crack-like. This
    # does not relax the veto for ordinary edges, dark lines, spots or textured areas.
    low_contrast_context_rescue = bool(
        branch == "low_contrast_hough"
        and polarity == "bright"
        and candidate_kind == "line"
        and meta_available
        and meta_supports
        and meta_probability >= max(0.70, meta_threshold + 0.16)
        and quality >= 0.52
        and context_support >= 0.72
        and risk <= 0.26
    )
    pixel_hard_veto = bool(p_defect <= 0.16 and not low_contrast_context_rescue)
    # Compact bright spots are the nastiest false-positive class: text corners,
    # catchlights and skin highlights can look exactly like dust at tiny scale.
    # For spots, require explicit pixel support; in high-texture areas demand a
    # stronger U-Net vote. Lines/irregular cracks stay context-led.
    spot_pixel_ok = (
        candidate_kind != "spot"
        or (p_defect >= 0.72 and (texture_risk < 0.65 or p_defect >= 0.95))
    )
    v2_positive = (
        is_v2
        and meta_available
        and meta_supports
        and meta_probability >= meta_threshold
        and not pixel_hard_veto
        and spot_pixel_ok
        and quality >= 0.30
        and context_support >= 0.24
        and risk <= 0.90
        and fused_score >= 0.56
        # Dark candidates are especially easy to confuse with hair/edges. Require
        # either some pixel support or a clearly-above-threshold meta verdict.
        and (polarity != "dark" or (p_defect >= 0.88 and meta_probability >= meta_threshold) or meta_margin >= 0.22)
    )
    legacy_positive = (
        not is_v2
        and raw_label == "defect"
        and raw_conf >= 0.82
        and p_defect >= 0.84
        and quality >= 0.56
        and context_support >= 0.48
        and risk <= 0.74
        and fused_score >= 0.68
        and not (synthetic_only and polarity == "dark")
    )

    if v2_positive or legacy_positive:
        label = "defect"
        if meta_available:
            confidence = float(np.clip(0.62 * meta_probability + 0.20 * classical_evidence + 0.18 * p_defect, 0.0, 0.98))
            if raw_label == "defect":
                reason = "Context Meta Verifier, пиксельный Surface AI v2 и контекстный детектор поддерживают вероятный дефект."
            elif low_contrast_context_rescue:
                reason = "Слабоконтрастная светлая линия подтверждена геометрией и Context Meta Verifier; пиксельный Surface AI v2 имеет известную слабость на бледных трещинах светлого фона."
            else:
                reason = "Context Meta Verifier и контекстный детектор уверенно поддерживают вероятный дефект; Surface AI v2 не дал сильного отрицательного сигнала."
        else:
            confidence = min(raw_conf, float(np.clip(0.52 + 0.48 * quality, 0.0, 1.0)))
            reason = "Surface AI и контекстные признаки согласованно поддерживают вероятный дефект."
    elif (
        (meta_available and meta_probability <= min(0.28, max(0.05, meta_threshold - 0.18)) and p_defect <= 0.70)
        or (meta_available and pixel_hard_veto and meta_probability < meta_threshold)
        or (raw_label == "natural_detail" and raw_conf >= 0.78 and p_defect <= 0.24)
        or (quality <= 0.26 and p_defect <= 0.72)
        or (context_support <= 0.16 and p_defect <= 0.62)
    ):
        label = "natural_detail"
        confidence = float(np.clip(max(
            raw_conf if raw_label == "natural_detail" else 0.0,
            0.58 + (0.30 - quality),
            (1.0 - meta_probability) * 0.90 if meta_available else 0.0,
        ), 0.0, 0.97))
        reason = "Контекстный meta-verifier и локальные признаки больше похожи на естественную деталь, чем на повреждение."
    else:
        label = "uncertain"
        distance = abs(fused_score - 0.5)
        confidence = float(np.clip(0.54 + max(0.0, 0.20 - distance) * 1.0, 0.50, 0.76))
        reason = "Пиксельная модель и контекст не дают достаточного согласия для подтверждения дефекта."

    return {
        "label": label,
        "confidence": confidence,
        "fused_defect_score": fused_score,
        "raw_ai_label": raw_label,
        "raw_ai_confidence": raw_conf,
        "raw_ai_defect_probability": p_defect,
        "context_meta_probability": meta_probability,
        "context_meta_threshold": meta_threshold,
        "context_meta_supports_defect": meta_supports,
        "context_meta_model_id": str(payload.get("context_meta_model_id", "")),
        "candidate_quality": quality,
        "context_support": context_support,
        "context_risk": risk,
        "reason": reason,
        "model_id": str(payload.get("model_id", "")),
        "training_kind": training_kind,
        "expert_verified": expert_verified,
        "candidate_model": candidate_model,
    }


def build_surface_refinement_metric(metrics: Mapping[str, MetricResult]) -> MetricResult:
    surface = metrics.get("surface_defects")
    surface_raw = surface.raw_value if surface is not None and isinstance(surface.raw_value, dict) else {}
    boxes = surface_raw.get("boxes_norm", []) if isinstance(surface_raw, dict) else []
    boxes = [deepcopy(box) for box in boxes if isinstance(box, dict)] if isinstance(boxes, list) else []
    ai_boxes = surface_raw.get("ai_boxes_norm", boxes) if isinstance(surface_raw, dict) else boxes
    ai_boxes = [deepcopy(box) for box in ai_boxes if isinstance(box, dict)] if isinstance(ai_boxes, list) else list(boxes)
    classical_count = int(surface_raw.get("candidate_count", len(ai_boxes)) or 0) if isinstance(surface_raw, dict) else len(ai_boxes)

    inference = metrics.get("ai_inference")
    inf_raw = inference.raw_value if inference is not None and isinstance(inference.raw_value, dict) else {}
    results = inf_raw.get("results", []) if isinstance(inf_raw, dict) else []
    by_index: dict[int, dict[str, Any]] = {}
    if isinstance(results, list):
        for task in results:
            if not isinstance(task, dict) or task.get("task") != "surface_defect_refinement":
                continue
            regions = task.get("regions", [])
            if not isinstance(regions, list):
                continue
            for region in regions:
                if not isinstance(region, dict) or region.get("status") != "ok":
                    continue
                name = str(region.get("region", ""))
                if not name.startswith("surface_"):
                    continue
                try:
                    index = int(name.split("_", 1)[1]) - 1
                except (ValueError, IndexError):
                    continue
                payload = region.get("result")
                if isinstance(payload, dict):
                    by_index[index] = payload

    summary_counts = {"defect": 0, "natural_detail": 0, "uncertain": 0}
    raw_counts = {"defect": 0, "natural_detail": 0, "uncertain": 0}
    confidences: list[float] = []
    fused_by_index: dict[int, dict[str, Any]] = {}
    for index, payload in by_index.items():
        if index < 0 or (classical_count > 0 and index >= classical_count):
            continue
        candidate_box = ai_boxes[index] if index < len(ai_boxes) else {}
        fused = _fuse_verdict(candidate_box, payload)
        fused_by_index[index] = fused
        label = str(fused["label"])
        raw_label = str(fused["raw_ai_label"])
        summary_counts[label] += 1
        if raw_label in raw_counts:
            raw_counts[raw_label] += 1
        confidences.append(float(fused["confidence"]))

    def attach_fused(box: dict[str, Any], fused: Mapping[str, Any] | None) -> bool:
        if fused is None:
            box["ai_label"] = "unprocessed"
            box["verification_label"] = "unprocessed"
            return False
        box["ai_label"] = str(fused["label"])  # legacy overlay/downstream alias
        box["ai_confidence"] = float(fused["confidence"])
        box["verification_label"] = str(fused["label"])
        box["verification_confidence"] = float(fused["confidence"])
        box["verification_reason"] = str(fused["reason"])
        box["fused_defect_score"] = float(fused["fused_defect_score"])
        box["ai_raw_label"] = str(fused["raw_ai_label"])
        box["ai_raw_confidence"] = float(fused["raw_ai_confidence"])
        box["ai_defect_probability"] = float(fused["raw_ai_defect_probability"])
        box["context_meta_probability"] = float(fused.get("context_meta_probability", 0.5))
        box["context_meta_threshold"] = float(fused.get("context_meta_threshold", 0.72))
        box["context_meta_supports_defect"] = bool(fused.get("context_meta_supports_defect", False))
        box["context_meta_model_id"] = str(fused.get("context_meta_model_id", ""))
        box["ai_model_id"] = str(fused["model_id"])
        box["ai_expert_verified"] = bool(fused.get("expert_verified", False))
        box["ai_candidate_model"] = bool(fused.get("candidate_model", True))
        box["context_support"] = float(fused["context_support"])
        box["context_risk"] = float(fused["context_risk"])
        return True

    display_evaluated = 0
    for index, box in enumerate(boxes):
        if attach_fused(box, fused_by_index.get(index)):
            display_evaluated += 1

    # The classical display pool is intentionally small, but a faint crack on a
    # light background can rank below its first 20 entries. Surface v8 promotes
    # additional AI+context confirmed defects into the visible/repairable map while
    # leaving low-ranked uncertain/natural candidates hidden. ai_boxes starts with
    # the same objects/order as boxes, so later indices are safe to append directly.
    promoted_confirmed = 0
    for index in range(len(boxes), len(ai_boxes)):
        fused = fused_by_index.get(index)
        if fused is None or str(fused.get("label", "")) != "defect":
            continue
        box = deepcopy(ai_boxes[index])
        attach_fused(box, fused)
        box["promoted_from_ai_pool"] = True
        boxes.append(box)
        display_evaluated += 1
        promoted_confirmed += 1

    evaluated = sum(summary_counts.values())
    total_unprocessed = max(0, classical_count - evaluated)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    diagnostic = (
        f"Контекстная проверка уточнила {evaluated} из {classical_count} кандидатов: "
        f"подтверждено {summary_counts['defect']}, естественная деталь {summary_counts['natural_detail']}, "
        f"неоднозначно {summary_counts['uncertain']}; без проверки {total_unprocessed}. "
        f"Сырой Surface AI предлагал defect={raw_counts['defect']}, natural={raw_counts['natural_detail']}, uncertain={raw_counts['uncertain']}."
        if evaluated
        else "Контекстная/AI-проверка дефектов не выполнялась; карта содержит только классические кандидаты."
    )
    return MetricResult(
        "surface_refinement",
        {
            "version": SURFACE_REFINEMENT_VERSION,
            "advisory_only": True,
            "changes_classical_scores": False,
            "ai_cannot_confirm_alone": True,
            "synthetic_ai_cannot_confirm_alone": True,
            "classical_candidate_count": classical_count,
            "shown_candidate_count": len(boxes),
            "promoted_confirmed_count": promoted_confirmed,
            "evaluated_count": evaluated,
            "display_evaluated_count": display_evaluated,
            "confirmed_defect_count": summary_counts["defect"],
            "likely_natural_count": summary_counts["natural_detail"],
            "uncertain_count": summary_counts["uncertain"],
            "raw_ai_counts": raw_counts,
            "unprocessed_count": total_unprocessed,
            "refined_boxes_norm": boxes,
        },
        None,
        float(max(0.0, min(1.0, mean_confidence))),
        "context_ai_advisory",
        region="surface",
        diagnostic=diagnostic,
    )


def refine_surface_decision_plan(metrics: dict[str, MetricResult]) -> None:
    """Make the user-facing plan depend on verified surface candidates, not raw morphology.

    The sensitive detector is allowed to over-generate candidates internally.  A
    surface correction reaches the main Plan only when context + Surface AI agree.
    Ambiguous/unprocessed candidates remain visible on the Defects tab for review.
    """
    plan = metrics.get("decision_plan")
    refinement = metrics.get("surface_refinement")
    if plan is None or refinement is None:
        return
    if not isinstance(plan.raw_value, dict) or not isinstance(refinement.raw_value, dict):
        return
    raw = deepcopy(plan.raw_value)
    items = raw.get("items", [])
    if not isinstance(items, list):
        return
    evaluated = int(refinement.raw_value.get("evaluated_count", 0) or 0)
    confirmed = int(refinement.raw_value.get("confirmed_defect_count", 0) or 0)
    uncertain = int(refinement.raw_value.get("uncertain_count", 0) or 0)
    unprocessed = int(refinement.raw_value.get("unprocessed_count", 0) or 0)
    if evaluated <= 0:
        return

    changed = False
    new_items: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or str(item.get("key", "")) != "surface_defects":
            new_items.append(item)
            continue
        changed = True
        if confirmed <= 0:
            # Precision-first UI: internal ambiguous candidates are not themselves a
            # recommendation to repair the photo.
            continue
        updated = dict(item)
        updated["decision"] = "review"
        updated["confidence"] = float(np.clip(refinement.confidence, 0.0, 1.0))
        updated["reason"] = (
            f"Surface AI v2 + Context Meta Verifier + контекстный детектор подтвердили вероятные дефекты: {confirmed}. "
            f"Неоднозначных кандидатов: {uncertain}; без AI-проверки: {unprocessed}."
        )
        updated["guardrail"] = (
            "Автоприменение Surface отключено. Пользователь сам выбирает конкретные локальные области на вкладке «Дефекты». "
            "Не удалять волосы, швы, текст, ресницы и естественные границы; сомнительные кандидаты оставлять без изменения."
        )
        new_items.append(updated)
    if not changed:
        return
    raw["items"] = new_items
    raw["fix_count"] = sum(str(item.get("decision", "")) == "fix" for item in new_items if isinstance(item, dict))
    raw["review_count"] = sum(str(item.get("decision", "")) == "review" for item in new_items if isinstance(item, dict))
    raw["preserve_count"] = sum(str(item.get("decision", "")) == "preserve" for item in new_items if isinstance(item, dict))
    raw["skip_count"] = sum(str(item.get("decision", "")) == "skip" for item in new_items if isinstance(item, dict))
    raw["surface_refinement_applied"] = True
    confidence_values = [float(item.get("confidence", 0.0) or 0.0) for item in new_items if isinstance(item, dict)]
    metrics["decision_plan"] = MetricResult(
        plan.name,
        raw,
        plan.normalized_value,
        float(np.mean(confidence_values)) if confidence_values else 0.0,
        plan.scale,
        region=plan.region,
        diagnostic=plan.diagnostic + "; surface-рекомендация отфильтрована Surface AI v2 + Context Meta Verifier + контекстным детектором",
    )
