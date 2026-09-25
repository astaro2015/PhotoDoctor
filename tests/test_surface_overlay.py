from photodoctor.core.database import surface_candidate_signature
from photodoctor.core.surface_overlay import overlay_index_for_table_row, visible_surface_boxes


def box(x: float) -> dict:
    return {"x": x, "y": 0.1, "w": 0.05, "h": 0.06, "polarity": "bright"}


def test_defect_map_defaults_to_selected_treatment_candidates_only():
    boxes = [box(0.1), box(0.2), box(0.3)]
    selected = {surface_candidate_signature(boxes[0]), surface_candidate_signature(boxes[2])}
    visible = visible_surface_boxes(boxes, selected)
    assert [item["x"] for item in visible] == [0.1, 0.3]


def test_show_all_mode_does_not_modify_treatment_selection():
    boxes = [box(0.1), box(0.2), box(0.3)]
    selected = {surface_candidate_signature(boxes[1])}
    visible = visible_surface_boxes(boxes, selected, show_all=True)
    assert [item["x"] for item in visible] == [0.1, 0.2, 0.3]
    assert selected == {surface_candidate_signature(boxes[1])}


def test_overlay_selection_index_is_remapped_after_filtering():
    boxes = [box(0.1), box(0.2), box(0.3), box(0.4)]
    selected = {surface_candidate_signature(boxes[1]), surface_candidate_signature(boxes[3])}
    assert overlay_index_for_table_row(boxes, 1, selected) == 0
    assert overlay_index_for_table_row(boxes, 3, selected) == 1
    assert overlay_index_for_table_row(boxes, 2, selected) is None


def test_overlay_selection_index_matches_table_in_show_all_mode():
    boxes = [box(0.1), box(0.2), box(0.3)]
    assert overlay_index_for_table_row(boxes, 2, set(), show_all=True) == 2
