from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if result != result or result in (float("inf"), float("-inf")):
        return float(default)
    return result


def build_preview_recipe_signature(
    selected_action_keys: set[str] | list[str] | tuple[str, ...],
    action_strengths: Mapping[str, float] | None,
    action_regions: Mapping[str, Mapping[str, float]] | None,
    validation_items: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[object, ...]:
    """Return a hashable signature of the *effective* correction recipe.

    Cache correctness must not depend solely on UI callers remembering to bump a
    revision counter.  In particular, moving a strength slider (for example WB
    from 35% to 60%) must always create a different signature even if an
    invalidation call is accidentally missed.
    """
    selected = tuple(sorted({str(key) for key in selected_action_keys if str(key)}))
    strengths = {str(key): _finite_float(value, 1.0) for key, value in (action_strengths or {}).items()}
    regions = {
        str(key): value
        for key, value in (action_regions or {}).items()
        if isinstance(value, Mapping)
    }

    validation_by_key: dict[str, Mapping[str, Any]] = {}
    for item in validation_items or ():
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("action_key", ""))
        if key:
            validation_by_key[key] = item

    strength_signature: list[tuple[str, float]] = []
    candidate_signature: list[tuple[str, str]] = []
    region_signature: list[tuple[str, tuple[float, float, float, float]]] = []
    for key in selected:
        item = validation_by_key.get(key, {})
        default_strength = _finite_float(item.get("default_strength", 1.0), 1.0)
        effective_strength = strengths.get(key, default_strength)
        strength_signature.append((key, round(effective_strength, 6)))
        candidate_signature.append((key, str(item.get("candidate", ""))))

        region = regions.get(key)
        if region is not None:
            region_signature.append((
                key,
                tuple(round(_finite_float(region.get(axis, 0.0), 0.0), 6) for axis in ("x", "y", "w", "h")),
            ))

    return (
        selected,
        tuple(strength_signature),
        tuple(candidate_signature),
        tuple(region_signature),
    )
