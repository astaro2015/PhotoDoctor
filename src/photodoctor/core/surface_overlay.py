from __future__ import annotations

from collections.abc import Iterable, Mapping

from photodoctor.core.database import surface_candidate_signature


def visible_surface_boxes(
    boxes: Iterable[Mapping],
    selected_signatures: set[str] | frozenset[str],
    *,
    show_all: bool = False,
) -> list[dict]:
    """Return surface candidates that should be visible on the defect map.

    The default map is intentionally treatment-oriented: only candidates explicitly
    selected in the ``Лечить`` column are visible. Diagnostic mode can request all
    candidates without changing the treatment selection itself.
    """
    normalized = [dict(box) for box in boxes if isinstance(box, Mapping)]
    if show_all:
        return normalized
    selected = set(selected_signatures)
    return [
        box for box in normalized
        if surface_candidate_signature(box) in selected
    ]


def overlay_index_for_table_row(
    boxes: Iterable[Mapping],
    table_row: int,
    selected_signatures: set[str] | frozenset[str],
    *,
    show_all: bool = False,
) -> int | None:
    """Map a row in the full candidates table to the filtered overlay index."""
    normalized = [dict(box) for box in boxes if isinstance(box, Mapping)]
    if table_row < 0 or table_row >= len(normalized):
        return None
    if show_all:
        return table_row

    selected = set(selected_signatures)
    target_signature = surface_candidate_signature(normalized[table_row])
    if target_signature not in selected:
        return None

    visible_index = 0
    for box in normalized:
        signature = surface_candidate_signature(box)
        if signature not in selected:
            continue
        if signature == target_signature:
            return visible_index
        visible_index += 1
    return None
