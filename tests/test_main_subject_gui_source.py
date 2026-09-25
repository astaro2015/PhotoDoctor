from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")


def test_main_subject_overlay_controls_exist():
    assert 'QPushButton("Главный объект")' in SOURCE
    assert 'set_subject_box' in SOURCE
    assert 'set_subject_enabled' in SOURCE
    assert 'self.subject_label.setText(' in SOURCE


def test_plan_can_use_main_subject_as_local_correction_region():
    assert 'QPushButton("Главный", area_box)' in SOURCE
    assert '_use_main_subject_region' in SOURCE
    assert 'protection_box_norm' in SOURCE


def test_surface_defect_map_defaults_to_treatment_selection_and_has_show_all_mode():
    source = (ROOT / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert 'QCheckBox("Показывать все найденные на карте")' in source
    assert 'visible_surface_boxes(' in source
    assert 'overlay_index_for_table_row(' in source
    assert 'self._refresh_surface_overlay()' in source
    assert 'Колонка «Лечить» управляет и локальным восстановлением' in source
