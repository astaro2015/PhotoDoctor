from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/"src"/"photodoctor"/"gui"/"main_window.py").read_text(encoding="utf-8")

def test_aesthetic_quality_is_visible_separately():
    assert 'self.aesthetic_label = QLabel("<b>Эстетика:</b> —")' in SOURCE
    assert 'композиционный ориентир' in SOURCE
