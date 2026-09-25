from pathlib import Path


SOURCE = Path(__file__).parents[1] / "src" / "photodoctor" / "gui" / "main_window.py"


def _text():
    return SOURCE.read_text(encoding="utf-8")


def test_image_view_supports_left_drag_panning_without_breaking_selection_mode():
    text = _text()
    start = text.index("class ImageView(QScrollArea):")
    end = text.index("class FullscreenImageWindow", start)
    block = text[start:end]
    assert "self._pan_active" in block
    assert "event.globalPosition() - self._pan_start_global" in block
    assert "self.horizontalScrollBar().setValue" in block
    assert "self.verticalScrollBar().setValue" in block
    assert "and not selection_mode" in block
    assert "Qt.CursorShape.ClosedHandCursor" in block
    assert "Qt.CursorShape.OpenHandCursor" in block


def test_double_click_requests_fullscreen_from_image_view():
    text = _text()
    start = text.index("def eventFilter(self, watched, event):")
    end = text.index("def _stop_pan", start)
    block = text[start:end]
    assert "QEvent.Type.MouseButtonDblClick" in block
    assert "self.fullscreen_requested.emit()" in block


def test_viewer_window_has_direct_wheel_zoom_pan_resize_and_shortcuts():
    text = _text()
    start = text.index("class FullscreenImageWindow(QMainWindow):")
    end = text.index("class HistogramWidget", start)
    block = text[start:end]
    assert "ImageView(wheel_zoom_requires_ctrl=False)" in block
    assert "source_view.copy_visual_state_to(self.view, fit=True)" in block
    assert "self.setMinimumSize(640, 480)" in block
    assert "self.resize(1180, 760)" in block
    assert "self.view.fullscreen_requested.connect(self._toggle_maximized)" in block
    assert "def _toggle_maximized(self)" in block
    assert "self.showMaximized()" in block
    assert "self.showNormal()" in block
    assert "Qt.Key.Key_Escape" in block
    assert "Qt.Key.Key_Plus" in block
    assert "Qt.Key.Key_Minus" in block
    assert "Qt.Key.Key_0" in block
    assert "Qt.Key.Key_F" in block


def test_main_window_opens_resizable_viewer_and_keeps_it_alive():
    text = _text()
    assert "self.image_view.fullscreen_requested.connect(self._open_fullscreen_viewer)" in text
    assert "self._fullscreen_windows: list[FullscreenImageWindow] = []" in text
    assert "def _open_fullscreen_viewer(self) -> None:" in text
    assert "viewer.show()" in text
    assert "viewer.showFullScreen()" not in text
    assert "viewer.raise_()" in text
    assert "viewer.activateWindow()" in text
    assert "viewer.destroyed.connect(forget_window)" in text


def test_fullscreen_copies_active_visual_overlays_at_full_source_resolution():
    text = _text()
    start = text.index("def copy_visual_state_to")
    end = text.index("def _shown_zoom_percent", start)
    block = text[start:end]
    assert "target._pixmap = self._pixmap.copy()" in block
    assert "target._overlay_boxes" in block
    assert "target._sharpness_cells" in block
    assert "target._tone_cells" in block
    assert "target._contrast_cells" in block
    assert "target._face_boxes" in block
    assert "target._highlight_boxes" in block
    assert "target._subject_box" in block
