from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from threading import Event

from PySide6.QtCore import QEvent, QPointF, QRectF, QProcess, QProcessEnvironment, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QGroupBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from photodoctor import __author__, __version__
from photodoctor.resources import asset_path
from photodoctor.core.batch import analyze_batch, collect_images
from photodoctor.core.ai_support import build_ai_status_metric
from photodoctor.ai.manager import AIModelManager, ModelImportError
from photodoctor.ai.almaz_status import inspect_almaz_status
from photodoctor.ai.almaz_release_bundle import install_release_bundle
from photodoctor.ai.almaz_model_promotion import rollback_almaz_model
from photodoctor.ai.parameter_recommender import (
    FEATURE_SCHEMA as PARAMETER_FEATURE_SCHEMA,
    MODEL_ID as PARAMETER_MODEL_ID,
    recommend_strength,
)
from photodoctor.core.database import AnalysisDatabase, surface_candidate_signature
from photodoctor.core.surface_overlay import overlay_index_for_table_row, visible_surface_boxes
from photodoctor.core.image_formats import INPUT_FILE_DIALOG_FILTER
from photodoctor.core.loader import LoadedImage, load_image
from photodoctor.core.histogram_diagnostics import build_histogram_diagnostic
from photodoctor.core.service import analyze_file
from photodoctor.core.series import group_photo_series
from photodoctor.core.validator import apply_selected_preview, preview_action_available
from photodoctor.core.preview_cache import build_preview_recipe_signature
from photodoctor.core.exporter import ExportError, audit_metadata_preservation, save_rgb_copy
from photodoctor.core.surface_history import merge_surface_history, read_surface_history
from photodoctor.core.feedback_export import export_surface_feedback_dataset
from photodoctor.core.training_export import collect_training_data_stats, export_training_package
from photodoctor.core.precision import all_precisions, get_precision
from photodoctor.gui.ai_presentation import build_ai_inference_rows, build_simple_ai_view
from photodoctor.gui.localization import localize_metric_name, localize_model_id, localize_payload, localize_task, localize_value
from photodoctor.gui.presentation import (
    build_display_metrics,
    build_photo_profile,
    build_recommendation_groups,
    build_summary,
)


class BatchWorker(QThread):
    progress = Signal(int, int, str, str)
    finished_summary = Signal(object)
    failed = Signal(str)

    def __init__(self, files: list[Path], db_path: Path, precision: str = "normal"):
        super().__init__()
        self.files = files
        self.db_path = db_path
        self.precision = get_precision(precision).key
        self.cancel_event = Event()
        self.pause_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.clear()

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    def run(self) -> None:
        try:
            def on_progress(i: int, total: int, path: Path, status: str) -> None:
                self.progress.emit(i, total, str(path), status)

            with AnalysisDatabase(self.db_path) as db:
                result = analyze_batch(
                    self.files, db, self.cancel_event, on_progress, pause_event=self.pause_event, precision=self.precision
                )
            self.finished_summary.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SeriesWorker(QThread):
    progress = Signal(int, int, str)
    finished_summary = Signal(object, str)
    failed = Signal(str, str)

    def __init__(self, files: list[Path], db_path: Path, precision: str, root_key: str):
        super().__init__()
        self.files = files
        self.db_path = db_path
        self.precision = get_precision(precision).key
        self.root_key = root_key
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            def on_progress(i: int, total: int, path: Path) -> None:
                self.progress.emit(i, total, str(path))

            with AnalysisDatabase(self.db_path) as db:
                result = group_photo_series(
                    self.files, db, precision=self.precision, progress=on_progress, cancel_event=self.cancel_event
                )
            self.finished_summary.emit(result, self.root_key)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}", self.root_key)


class PhotoAnalysisWorker(QThread):
    stage = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, path: Path, db_path: Path, precision: str, reset_view: bool, completion_message: str | None = None):
        super().__init__()
        self.path = Path(path)
        self.db_path = Path(db_path)
        self.precision = get_precision(precision).key
        self.reset_view = bool(reset_view)
        self.completion_message = completion_message

    def run(self) -> None:
        try:
            warnings: list[str] = []
            self.stage.emit(f"Чтение изображения: {self.path.name}")
            loaded = load_image(self.path)
            self.stage.emit("Загрузка ручной разметки и кэша…")
            cached_result = None
            source_quick_hash = None
            try:
                source_quick_hash = AnalysisDatabase.quick_hash(self.path)
                with AnalysisDatabase(self.db_path) as db:
                    manual = db.load_manual_vision_overrides(self.path, quick_hash=source_quick_hash)
                    cached_result = db.load_current_result(
                        loaded.info, precision=self.precision, quick_hash=source_quick_hash
                    )
            except Exception as exc:
                manual = {"faces": [], "eyes": []}
                warnings.append(
                    "Не удалось прочитать ручную разметку из базы данных или кэш анализа: "
                    f"{type(exc).__name__}: {exc}"
                )
            manual_faces = list(manual.get("faces", []))
            manual_eyes = list(manual.get("eyes", []))
            profile = get_precision(self.precision)
            if cached_result is not None:
                self.stage.emit(f"Готовый анализ «{profile.label}» загружен из базы…")
                self.completed.emit({
                    "path": self.path,
                    "loaded": loaded,
                    "result": cached_result,
                    "manual_faces": manual_faces,
                    "manual_eyes": manual_eyes,
                    "precision": self.precision,
                    "reset_view": self.reset_view,
                    "completion_message": self.completion_message or f"Готово из кэша: анализ «{profile.label}»",
                    "warnings": warnings,
                    "source_quick_hash": source_quick_hash,
                })
                return
            if profile.key == "maximum":
                self.stage.emit("Анализ изображения — Максимальный: сверхплотный проход до 8192 px…")
            elif profile.key == "precise":
                self.stage.emit("Анализ изображения — Точно: глубокий проход до 4096 px…")
            else:
                self.stage.emit(f"Анализ изображения — {profile.label}…")
            result = analyze_file(
                self.path, precision=self.precision,
                manual_face_boxes=manual_faces, manual_eye_boxes=manual_eyes,
                loaded_image=loaded,
            )
            self.stage.emit("Сохранение результатов анализа…")
            try:
                with AnalysisDatabase(self.db_path) as db:
                    db.save(result)
            except Exception as exc:
                warnings.append(
                    "Анализ выполнен, но его не удалось сохранить в базе данных: "
                    f"{type(exc).__name__}: {exc}"
                )
            self.completed.emit({
                "path": self.path,
                "loaded": loaded,
                "result": result,
                "manual_faces": manual_faces,
                "manual_eyes": manual_eyes,
                "precision": self.precision,
                "reset_view": self.reset_view,
                "completion_message": self.completion_message,
                "warnings": warnings,
            })
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class CorrectionPreviewWorker(QThread):
    completed = Signal(object, object, int)
    failed = Signal(str, object)
    stage = Signal(str)

    def __init__(
        self,
        rgb: np.ndarray,
        validation_items: list[dict],
        selected: set[str],
        strengths: dict[str, float],
        regions: dict[str, dict[str, float]],
        cache_key: tuple[object, ...],
    ):
        super().__init__()
        self.rgb = rgb
        self.validation_items = validation_items
        self.selected = set(selected)
        self.strengths = dict(strengths)
        self.regions = {str(k): dict(v) for k, v in regions.items()}
        self.cache_key = cache_key
        self.safety_adjustment = ""

    def run(self) -> None:
        try:
            def on_progress(message: str) -> None:
                text = str(message)
                if text.startswith("ALMAZ safety: найден безопасный вариант"):
                    self.safety_adjustment = text.split("ALMAZ safety:", 1)[-1].strip()
                self.stage.emit(text)

            preview = apply_selected_preview(
                self.rgb, self.validation_items, self.selected, self.strengths, self.regions,
                progress=on_progress,
            )
            self.completed.emit(preview, self.cache_key, len(self.selected))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}", self.cache_key)



class AlmazReleaseWorker(QThread):
    stage = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, mode: str, *, bundle_path: Path | None = None, task: str | None = None):
        super().__init__()
        self.mode = str(mode)
        self.bundle_path = Path(bundle_path) if bundle_path is not None else None
        self.task = str(task) if task is not None else None

    def run(self) -> None:
        try:
            def on_progress(message: str) -> None:
                self.stage.emit(str(message))

            if self.mode == "install":
                if self.bundle_path is None:
                    raise ValueError("Не указан ALMAZ release-bundle.")
                result = install_release_bundle(self.bundle_path, progress=on_progress)
            elif self.mode == "rollback":
                if not self.task:
                    raise ValueError("Не указана ALMAZ-задача для отката.")
                result = rollback_almaz_model(self.task, progress=on_progress)
            else:
                raise ValueError(f"Неизвестная ALMAZ release-операция: {self.mode}")
            if result.accepted:
                self.completed.emit(result)
            else:
                self.failed.emit(result.detail)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")



class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem that sorts by numeric value instead of formatted text."""

    def __init__(self, text: str, value: float):
        super().__init__(text)
        self.sort_value = float(value)

    def __lt__(self, other):
        if isinstance(other, NumericTableWidgetItem):
            return self.sort_value < other.sort_value
        return super().__lt__(other)


class ImageView(QScrollArea):
    region_selected = Signal(str, object)
    zoom_changed = Signal(int, bool)
    fullscreen_requested = Signal()

    def __init__(self, *, wheel_zoom_requires_ctrl: bool = True):
        super().__init__()
        # Qt is allowed to dispatch events synchronously while widgets are being
        # wired up.  Keep the filter disabled until *all* ImageView state exists;
        # this prevents a newly-added field from becoming the next startup crash.
        self._event_filter_ready = False
        self._selection_mode: str | None = None
        self._selection_start: tuple[float, float] | None = None
        self._selection_preview: dict[str, float] | None = None
        self._pan_active = False
        self._pan_moved = False
        self._pan_start_global: QPointF | None = None
        self._pan_start_scroll = (0, 0)
        self._wheel_zoom_requires_ctrl = bool(wheel_zoom_requires_ctrl)
        self._pixmap: QPixmap | None = None
        self._fit = True
        self._zoom_percent = 100
        self._overlay_boxes: list[dict[str, float]] = []
        self._overlay_selected_index: int | None = None
        self._overlay_enabled = False
        self._sharpness_cells: list[dict[str, float | str]] = []
        self._sharpness_enabled = False
        self._tone_cells: list[dict[str, float | str]] = []
        self._tone_enabled = False
        self._contrast_cells: list[dict[str, float | str]] = []
        self._contrast_enabled = False
        self._face_boxes: list[dict[str, float]] = []
        self._eye_boxes: list[dict[str, float | int]] = []
        self._faces_enabled = False
        self._highlight_boxes: list[dict[str, float | str]] = []
        self._highlights_enabled = False
        self._subject_box: dict[str, float] | None = None
        self._subject_enabled = False
        self.label = QLabel(alignment=Qt.AlignCenter)
        self.label.setMouseTracking(True)
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        # Install the filter only after widget wiring and state initialization.
        self.label.installEventFilter(self)
        self._event_filter_ready = True
        self.setToolTip(
            "ЛКМ + перетаскивание — двигать увеличенное фото. "
            "Двойной щелчок — полноэкранный просмотр."
        )

    def _can_pan(self) -> bool:
        return bool(
            self._pixmap is not None
            and (self.horizontalScrollBar().maximum() > 0 or self.verticalScrollBar().maximum() > 0)
        )

    def _update_pan_cursor(self) -> None:
        if self._selection_mode:
            self.label.setCursor(Qt.CursorShape.CrossCursor)
        elif self._pan_active:
            self.label.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif self._can_pan():
            self.label.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.label.unsetCursor()

    def begin_region_selection(self, kind: str) -> None:
        self._selection_mode = str(kind)
        self._selection_start = None
        self._selection_preview = None
        self.label.setCursor(Qt.CursorShape.CrossCursor)
        self.refresh()

    def cancel_region_selection(self, *, refresh: bool = True) -> None:
        self._selection_mode = None
        self._selection_start = None
        self._selection_preview = None
        self._update_pan_cursor()
        if refresh:
            self.refresh()

    def _label_pos_to_norm(self, pos) -> tuple[float, float] | None:
        pixmap = self.label.pixmap()
        if pixmap is None or pixmap.width() <= 0 or pixmap.height() <= 0:
            return None
        left = (self.label.width() - pixmap.width()) / 2.0
        top = (self.label.height() - pixmap.height()) / 2.0
        px = float(pos.x()) - left
        py = float(pos.y()) - top
        if px < 0 or py < 0 or px > pixmap.width() or py > pixmap.height():
            return None
        return (
            float(np.clip(px / max(pixmap.width(), 1), 0.0, 1.0)),
            float(np.clip(py / max(pixmap.height(), 1), 0.0, 1.0)),
        )

    def eventFilter(self, watched, event):
        # installEventFilter() can synchronously trigger Qt events before __init__
        # has finished.  Do not touch ImageView state until construction is complete.
        if not getattr(self, "_event_filter_ready", False):
            return super().eventFilter(watched, event)

        selection_mode = self._selection_mode
        if watched is self.label and selection_mode:
            event_type = event.type()
            if event_type == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self.cancel_region_selection()
                    return True
                if event.button() == Qt.MouseButton.LeftButton:
                    point = self._label_pos_to_norm(event.position())
                    if point is not None:
                        self._selection_start = point
                        self._selection_preview = {"x": point[0], "y": point[1], "w": 0.0, "h": 0.0}
                        return True
            elif event_type == QEvent.Type.MouseMove and self._selection_start is not None:
                point = self._label_pos_to_norm(event.position())
                if point is not None:
                    sx, sy = self._selection_start
                    ex, ey = point
                    self._selection_preview = {
                        "x": min(sx, ex), "y": min(sy, ey),
                        "w": abs(ex - sx), "h": abs(ey - sy),
                    }
                    self.refresh()
                    return True
            elif event_type == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self._selection_start is not None:
                    point = self._label_pos_to_norm(event.position())
                    preview = None
                    if point is not None:
                        sx, sy = self._selection_start
                        ex, ey = point
                        preview = {
                            "x": min(sx, ex), "y": min(sy, ey),
                            "w": abs(ex - sx), "h": abs(ey - sy),
                        }
                    mode = selection_mode
                    self.cancel_region_selection(refresh=False)
                    self.refresh()
                    if preview is not None and preview["w"] >= 0.012 and preview["h"] >= 0.012:
                        self.region_selected.emit(str(mode), preview)
                    return True

        if watched is self.label and self._pixmap is not None and not selection_mode:
            event_type = event.type()
            if event_type == QEvent.Type.MouseButtonDblClick and event.button() == Qt.MouseButton.LeftButton:
                self._stop_pan()
                self.fullscreen_requested.emit()
                return True
            if event_type == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if self._can_pan():
                    self._pan_active = True
                    self._pan_moved = False
                    self._pan_start_global = event.globalPosition()
                    self._pan_start_scroll = (
                        self.horizontalScrollBar().value(),
                        self.verticalScrollBar().value(),
                    )
                    self.label.setCursor(Qt.CursorShape.ClosedHandCursor)
                    self.label.grabMouse()
                    return True
            elif event_type == QEvent.Type.MouseMove and self._pan_active and self._pan_start_global is not None:
                delta = event.globalPosition() - self._pan_start_global
                if abs(delta.x()) + abs(delta.y()) >= 2.0:
                    self._pan_moved = True
                self.horizontalScrollBar().setValue(int(round(self._pan_start_scroll[0] - delta.x())))
                self.verticalScrollBar().setValue(int(round(self._pan_start_scroll[1] - delta.y())))
                return True
            elif event_type == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                if self._pan_active:
                    self._stop_pan()
                    return True

        return super().eventFilter(watched, event)

    def _stop_pan(self) -> None:
        if getattr(self, "_pan_active", False):
            self._pan_active = False
            self._pan_start_global = None
            try:
                if self.label.mouseGrabber() is self.label:
                    self.label.releaseMouse()
            except RuntimeError:
                pass
        self._update_pan_cursor()

    def set_rgb(self, arr) -> None:
        h, w, _ = arr.shape
        qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self.refresh()

    def set_overlay_boxes(self, boxes: list[dict[str, float]] | None) -> None:
        self._overlay_boxes = list(boxes or [])
        self.refresh()

    def set_overlay_enabled(self, enabled: bool) -> None:
        self._overlay_enabled = enabled
        self.refresh()

    def set_overlay_selection(self, index: int | None) -> None:
        self._overlay_selected_index = None if index is None or index < 0 else int(index)
        self.refresh()

    def set_sharpness_cells(self, cells: list[dict[str, float | str]] | None) -> None:
        self._sharpness_cells = list(cells or [])
        self.refresh()

    def set_sharpness_enabled(self, enabled: bool) -> None:
        self._sharpness_enabled = enabled
        self.refresh()

    def set_tone_cells(self, cells: list[dict[str, float | str]] | None) -> None:
        self._tone_cells = list(cells or [])
        self.refresh()

    def set_tone_enabled(self, enabled: bool) -> None:
        self._tone_enabled = enabled
        self.refresh()

    def set_contrast_cells(self, cells: list[dict[str, float | str]] | None) -> None:
        self._contrast_cells = list(cells or [])
        self.refresh()

    def set_contrast_enabled(self, enabled: bool) -> None:
        self._contrast_enabled = enabled
        self.refresh()

    def set_face_boxes(self, boxes: list[dict[str, float]] | None) -> None:
        self._face_boxes = list(boxes or [])
        self.refresh()

    def set_eye_boxes(self, boxes: list[dict[str, float | int]] | None) -> None:
        self._eye_boxes = list(boxes or [])
        self.refresh()

    def set_faces_enabled(self, enabled: bool) -> None:
        self._faces_enabled = enabled
        self.refresh()

    def set_highlight_boxes(self, boxes: list[dict[str, float | str]] | None) -> None:
        self._highlight_boxes = list(boxes or [])
        self.refresh()

    def set_highlights_enabled(self, enabled: bool) -> None:
        self._highlights_enabled = enabled
        self.refresh()

    def set_subject_box(self, box: dict[str, float] | None) -> None:
        self._subject_box = dict(box) if isinstance(box, dict) else None
        self.refresh()

    def set_subject_enabled(self, enabled: bool) -> None:
        self._subject_enabled = enabled
        self.refresh()

    def copy_visual_state_to(self, target: "ImageView", *, fit: bool = True) -> None:
        target._pixmap = self._pixmap.copy() if self._pixmap is not None else None
        target._overlay_boxes = [dict(x) for x in self._overlay_boxes]
        target._overlay_selected_index = self._overlay_selected_index
        target._overlay_enabled = self._overlay_enabled
        target._sharpness_cells = [dict(x) for x in self._sharpness_cells]
        target._sharpness_enabled = self._sharpness_enabled
        target._tone_cells = [dict(x) for x in self._tone_cells]
        target._tone_enabled = self._tone_enabled
        target._contrast_cells = [dict(x) for x in self._contrast_cells]
        target._contrast_enabled = self._contrast_enabled
        target._face_boxes = [dict(x) for x in self._face_boxes]
        target._eye_boxes = [dict(x) for x in self._eye_boxes]
        target._faces_enabled = self._faces_enabled
        target._highlight_boxes = [dict(x) for x in self._highlight_boxes]
        target._highlights_enabled = self._highlights_enabled
        target._subject_box = dict(self._subject_box) if self._subject_box else None
        target._subject_enabled = self._subject_enabled
        target._selection_mode = None
        target._selection_start = None
        target._selection_preview = None
        target._zoom_percent = 100
        target.set_fit(bool(fit))

    def _shown_zoom_percent(self) -> int:
        if not self._pixmap:
            return int(self._zoom_percent)
        shown = self.label.pixmap()
        if shown is None or self._pixmap.width() <= 0:
            return int(self._zoom_percent)
        return max(1, int(round(shown.width() / self._pixmap.width() * 100.0)))

    def set_fit(self, fit: bool) -> None:
        self._fit = bool(fit)
        self.setWidgetResizable(self._fit)
        self.refresh()
        self._update_pan_cursor()
        self.zoom_changed.emit(self._shown_zoom_percent(), self._fit)

    def set_zoom_percent(self, percent: int, *, anchor=None) -> None:
        if not self._pixmap:
            self._zoom_percent = max(25, min(300, int(percent)))
            self._fit = False
            self.setWidgetResizable(False)
            self.zoom_changed.emit(self._zoom_percent, False)
            return
        percent = max(25, min(300, int(percent)))
        viewport = self.viewport()
        if anchor is None:
            anchor_x = viewport.width() / 2.0
            anchor_y = viewport.height() / 2.0
        else:
            anchor_x = float(anchor.x())
            anchor_y = float(anchor.y())

        shown = self.label.pixmap()
        if shown is not None and shown.width() > 0 and shown.height() > 0:
            content_x = self.horizontalScrollBar().value() + anchor_x
            content_y = self.verticalScrollBar().value() + anchor_y
            left = max(0.0, (self.label.width() - shown.width()) / 2.0)
            top = max(0.0, (self.label.height() - shown.height()) / 2.0)
            norm_x = float(np.clip((content_x - left) / shown.width(), 0.0, 1.0))
            norm_y = float(np.clip((content_y - top) / shown.height(), 0.0, 1.0))
        else:
            norm_x = norm_y = 0.5

        self._fit = False
        self._zoom_percent = percent
        self.setWidgetResizable(False)
        self.refresh()
        new_w = self.label.pixmap().width() if self.label.pixmap() is not None else self.label.width()
        new_h = self.label.pixmap().height() if self.label.pixmap() is not None else self.label.height()
        self.horizontalScrollBar().setValue(max(0, int(round(norm_x * new_w - anchor_x))))
        self.verticalScrollBar().setValue(max(0, int(round(norm_y * new_h - anchor_y))))
        self._update_pan_cursor()
        self.zoom_changed.emit(self._zoom_percent, False)

    def wheelEvent(self, event) -> None:
        zoom_modifier_ok = (
            not self._wheel_zoom_requires_ctrl
            or bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        )
        if zoom_modifier_ok and self._pixmap is not None and event.angleDelta().y() != 0:
            step = 10 if abs(event.angleDelta().y()) < 240 else 25
            direction = 1 if event.angleDelta().y() > 0 else -1
            base = self._shown_zoom_percent() if self._fit else self._zoom_percent
            self.set_zoom_percent(base + direction * step, anchor=event.position())
            event.accept()
            return
        super().wheelEvent(event)

    @staticmethod
    def _spatial_heatmap(shown: QPixmap, cells: list[dict[str, float | str]], kind: str) -> QPixmap:
        """Blend overlapping spatial samples into a smooth, low-opacity heatmap.

        Measurement windows stay explicit in the analyzer, but the GUI deliberately
        avoids drawing their borders.  A small working canvas keeps resize/toggle
        operations cheap even on large monitors.
        """
        sw, sh = shown.width(), shown.height()
        if sw <= 0 or sh <= 0 or not cells:
            return shown

        scale = min(1.0, 640.0 / max(sw, sh))
        mw = max(32, int(round(sw * scale)))
        mh = max(32, int(round(sh * scale)))
        color_acc = np.zeros((mh, mw, 3), dtype=np.float32)
        weight_acc = np.zeros((mh, mw), dtype=np.float32)

        for cell in cells:
            status = str(cell.get("status", ""))
            color: tuple[float, float, float] | None = None
            weight = 0.0
            if kind == "sharpness":
                if status == "low_texture":
                    continue
                try:
                    score = float(cell.get("score", 0.0))
                except (TypeError, ValueError):
                    continue
                if score < 45.0:
                    color, weight = (215.0, 70.0, 70.0), 1.0
                elif score < 65.0:
                    color, weight = (230.0, 170.0, 55.0), 0.72
                else:
                    color, weight = (60.0, 165.0, 95.0), 0.40
            elif kind == "tone":
                tone_map = {
                    "deep_shadow": ((55.0, 85.0, 185.0), 1.0),
                    "dark": ((80.0, 125.0, 210.0), 0.70),
                    "clipped_highlight": ((220.0, 85.0, 70.0), 1.0),
                    "bright": ((235.0, 190.0, 70.0), 0.62),
                }
                entry = tone_map.get(status)
                if entry is None:
                    continue
                color, weight = entry
            elif kind == "contrast":
                if status in {"low_texture", "good"}:
                    continue
                if status == "low":
                    color, weight = (190.0, 75.0, 155.0), 1.0
                else:
                    color, weight = (210.0, 155.0, 70.0), 0.65
            else:
                return shown

            try:
                x0 = int(round(float(cell.get("x", 0.0)) * mw))
                y0 = int(round(float(cell.get("y", 0.0)) * mh))
                x1 = int(round((float(cell.get("x", 0.0)) + float(cell.get("w", 0.0))) * mw))
                y1 = int(round((float(cell.get("y", 0.0)) + float(cell.get("h", 0.0))) * mh))
            except (TypeError, ValueError):
                continue
            x0, y0 = max(0, min(mw - 1, x0)), max(0, min(mh - 1, y0))
            x1, y1 = max(x0 + 1, min(mw, x1)), max(y0 + 1, min(mh, y1))
            color_acc[y0:y1, x0:x1] += np.asarray(color, dtype=np.float32) * weight
            weight_acc[y0:y1, x0:x1] += weight

        active = weight_acc > 1e-6
        if not np.any(active):
            return shown
        heat = np.zeros_like(color_acc)
        heat[active] = color_acc[active] / weight_acc[active, None]
        sigma = max(1.1, min(mw, mh) / 180.0)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
        support = cv2.GaussianBlur(weight_acc, (0, 0), sigmaX=sigma, sigmaY=sigma)
        # At normal 35-50% overlap an interior point is covered by several windows.
        # Saturate gently rather than making overlap count itself look like severity.
        alpha = np.clip(support / 2.8, 0.0, 1.0) * 0.24

        rgba = np.empty((mh, mw, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(heat, 0.0, 255.0).astype(np.uint8)
        rgba[..., 3] = np.clip(alpha * 255.0, 0.0, 255.0).astype(np.uint8)
        qimg = QImage(rgba.data, mw, mh, rgba.strides[0], QImage.Format_RGBA8888).copy()
        overlay = QPixmap.fromImage(qimg).scaled(shown.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        result = shown.copy()
        painter = QPainter(result)
        painter.drawPixmap(0, 0, overlay)
        painter.end()
        return result

    def refresh(self) -> None:
        if not self._pixmap:
            return
        if self._fit:
            size = self.viewport().size()
            shown = self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.resize(size)
        else:
            scale = max(0.25, min(3.0, self._zoom_percent / 100.0))
            width = max(1, int(round(self._pixmap.width() * scale)))
            height = max(1, int(round(self._pixmap.height() * scale)))
            transform = Qt.FastTransformation if self._zoom_percent >= 100 else Qt.SmoothTransformation
            shown = self._pixmap.scaled(width, height, Qt.IgnoreAspectRatio, transform)
            self.label.resize(shown.size())

        if self._overlay_enabled and self._overlay_boxes:
            shown = shown.copy()
            painter = QPainter(shown)
            for box_index, box in enumerate(self._overlay_boxes):
                try:
                    x = round(float(box.get("x", 0.0)) * shown.width())
                    y = round(float(box.get("y", 0.0)) * shown.height())
                    w = max(2, round(float(box.get("w", 0.0)) * shown.width()))
                    h = max(2, round(float(box.get("h", 0.0)) * shown.height()))
                except (TypeError, ValueError):
                    continue
                user_label = str(box.get("user_label", ""))
                ai_label = str(box.get("ai_label", "unprocessed"))
                effective_label = user_label or ai_label
                if effective_label == "defect":
                    color, pen_style, pen_width = QColor(225, 45, 70), Qt.SolidLine, 4 if user_label else 3
                elif effective_label == "natural_detail":
                    color, pen_style, pen_width = QColor(55, 185, 95), Qt.SolidLine if user_label else Qt.DotLine, 3 if user_label else 2
                elif effective_label == "uncertain":
                    color, pen_style, pen_width = QColor(235, 170, 40), Qt.SolidLine if user_label else Qt.DashLine, 3 if user_label else 2
                else:
                    polarity = str(box.get("polarity", "bright"))
                    color = QColor(210, 90, 90) if polarity == "bright" else QColor(70, 125, 210)
                    pen_style, pen_width = Qt.DashLine, 2
                painter.setPen(QPen(color, pen_width, pen_style))
                contour = box.get("contour")
                if isinstance(contour, list) and len(contour) >= 2:
                    points: list[QPointF] = []
                    for point in contour:
                        if not isinstance(point, dict):
                            continue
                        try:
                            points.append(QPointF(
                                float(point.get("x", 0.0)) * shown.width(),
                                float(point.get("y", 0.0)) * shown.height(),
                            ))
                        except (TypeError, ValueError):
                            continue
                    if len(points) >= 2:
                        path = QPainterPath(points[0])
                        for point in points[1:]:
                            path.lineTo(point)
                        if len(points) >= 3:
                            path.closeSubpath()
                        painter.drawPath(path)
                    else:
                        painter.drawRect(x, y, w, h)
                else:
                    painter.drawRect(x, y, w, h)
                if box_index == self._overlay_selected_index:
                    painter.setPen(QPen(QColor(255, 255, 255), 6, Qt.SolidLine))
                    painter.drawRect(x, y, w, h)
                    painter.setPen(QPen(QColor(35, 205, 245), 3, Qt.SolidLine))
                    painter.drawRect(x, y, w, h)
            painter.end()

        if self._sharpness_enabled and self._sharpness_cells:
            shown = self._spatial_heatmap(shown, self._sharpness_cells, "sharpness")

        if self._tone_enabled and self._tone_cells:
            shown = self._spatial_heatmap(shown, self._tone_cells, "tone")

        if self._contrast_enabled and self._contrast_cells:
            shown = self._spatial_heatmap(shown, self._contrast_cells, "contrast")

        if self._highlights_enabled and self._highlight_boxes:
            shown = shown.copy()
            painter = QPainter(shown)
            for box in self._highlight_boxes:
                try:
                    x = round(float(box.get("x", 0.0)) * shown.width())
                    y = round(float(box.get("y", 0.0)) * shown.height())
                    w = max(2, round(float(box.get("w", 0.0)) * shown.width()))
                    h = max(2, round(float(box.get("h", 0.0)) * shown.height()))
                except (TypeError, ValueError):
                    continue
                kind = str(box.get("kind", "specular_candidate"))
                color = QColor(45, 175, 205) if kind == "specular_candidate" else QColor(225, 165, 55)
                painter.setPen(QPen(color, 2, Qt.DashLine))
                painter.drawRect(x, y, w, h)
            painter.end()

        if self._faces_enabled and self._face_boxes:
            shown = shown.copy()
            painter = QPainter(shown)
            for index, box in enumerate(self._face_boxes, start=1):
                try:
                    x = round(float(box.get("x", 0.0)) * shown.width())
                    y = round(float(box.get("y", 0.0)) * shown.height())
                    w = max(2, round(float(box.get("w", 0.0)) * shown.width()))
                    h = max(2, round(float(box.get("h", 0.0)) * shown.height()))
                except (TypeError, ValueError):
                    continue
                manual = str(box.get("source", "auto")) == "manual"
                painter.setPen(QPen(QColor(245, 190, 45) if manual else QColor(45, 170, 95), 3 if manual else 2))
                painter.drawRect(x, y, w, h)
                painter.drawText(x + 4, y + 15, ("M" if manual else "") + str(index))
            if self._eye_boxes:
                for box in self._eye_boxes:
                    try:
                        x = round(float(box.get("x", 0.0)) * shown.width())
                        y = round(float(box.get("y", 0.0)) * shown.height())
                        w = max(2, round(float(box.get("w", 0.0)) * shown.width()))
                        h = max(2, round(float(box.get("h", 0.0)) * shown.height()))
                    except (TypeError, ValueError):
                        continue
                    manual = str(box.get("source", "auto")) == "manual"
                    painter.setPen(QPen(QColor(245, 190, 45) if manual else QColor(45, 175, 205), 3 if manual else 2))
                    painter.drawRect(x, y, w, h)
            painter.end()

        if self._subject_enabled and self._subject_box:
            shown = shown.copy()
            painter = QPainter(shown)
            box = self._subject_box
            try:
                x = round(float(box.get("x", 0.0)) * shown.width())
                y = round(float(box.get("y", 0.0)) * shown.height())
                w = max(2, round(float(box.get("w", 0.0)) * shown.width()))
                h = max(2, round(float(box.get("h", 0.0)) * shown.height()))
                painter.setPen(QPen(QColor(230, 95, 235), 3, Qt.DashLine))
                painter.drawRect(x, y, w, h)
                painter.drawText(x + 5, y + 18, "Главный объект")
            except (TypeError, ValueError):
                pass
            painter.end()

        if self._selection_preview is not None:
            shown = shown.copy()
            painter = QPainter(shown)
            box = self._selection_preview
            x = round(float(box.get("x", 0.0)) * shown.width())
            y = round(float(box.get("y", 0.0)) * shown.height())
            w = max(2, round(float(box.get("w", 0.0)) * shown.width()))
            h = max(2, round(float(box.get("h", 0.0)) * shown.height()))
            painter.setPen(QPen(QColor(245, 190, 45), 3, Qt.DashLine))
            painter.drawRect(x, y, w, h)
            painter.end()
        self.label.setPixmap(shown)
        self._update_pan_cursor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.refresh()


class FullscreenImageWindow(QMainWindow):
    def __init__(self, source_view: ImageView, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Photo Doctor — просмотр")
        self.setMinimumSize(640, 480)
        self.resize(1180, 760)
        self.setStyleSheet("background: #111111;")
        self.view = ImageView(wheel_zoom_requires_ctrl=False)
        self.view.setStyleSheet("background: #111111;")
        self.view.label.setStyleSheet("background: #111111;")
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setToolTip(
            "Колесо — масштаб. ЛКМ + перетаскивание — двигать фото. "
            "F — по окну, 0 — 100%, Esc — закрыть. "
            "Двойной щелчок — развернуть окно или вернуть обычный размер."
        )
        self.setCentralWidget(self.view)
        source_view.copy_visual_state_to(self.view, fit=True)
        self.view.fullscreen_requested.connect(self._toggle_maximized)
        self.view.zoom_changed.connect(self._update_title)

    def _toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def _update_title(self, percent: int, fit: bool) -> None:
        suffix = f"По окну · {percent}%" if fit else f"{percent}%"
        self.setWindowTitle(f"Photo Doctor — просмотр — {suffix}")

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self.close()
            return
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            base = self.view._shown_zoom_percent() if self.view._fit else self.view._zoom_percent
            self.view.set_zoom_percent(base + 25)
            return
        if key == Qt.Key.Key_Minus:
            base = self.view._shown_zoom_percent() if self.view._fit else self.view._zoom_percent
            self.view.set_zoom_percent(base - 25)
            return
        if key == Qt.Key.Key_0:
            self.view.set_zoom_percent(100)
            return
        if key == Qt.Key.Key_F:
            self.view.set_fit(True)
            return
        super().keyPressEvent(event)


class HistogramWidget(QWidget):
    SHADOW_END = 0.18
    MIDTONE_END = 0.45
    SHADOW_CLIP = 0.0031308
    HIGHLIGHT_CLIP = 0.99

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(190)
        self._bins: list[float] = []
        self._percentiles: dict[str, float] = {}

    def set_bins(self, bins: list[float] | None) -> None:
        self._bins = list(bins or [])
        self.update()

    def set_histogram(self, bins: list[float] | None, percentiles: dict[str, float] | None = None) -> None:
        self._bins = list(bins or [])
        self._percentiles = dict(percentiles or {})
        self.update()

    @staticmethod
    def _x_for_value(rect: QRectF, value: float) -> float:
        return rect.left() + min(max(float(value), 0.0), 1.0) * rect.width()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        full = QRectF(self.rect())
        plot = full.adjusted(18, 27, -18, -30)
        painter.fillRect(plot, self.palette().base())

        # Broad tonal regions are descriptive, not pass/fail ranges.
        zones = (
            (0.0, self.SHADOW_END, QColor(75, 115, 170, 28), "Тени"),
            (self.SHADOW_END, self.MIDTONE_END, QColor(95, 155, 115, 25), "Средние тона"),
            (self.MIDTONE_END, 1.0, QColor(205, 155, 80, 23), "Света"),
        )
        for lo, hi, color, label in zones:
            x1 = self._x_for_value(plot, lo)
            x2 = self._x_for_value(plot, hi)
            zone_rect = QRectF(x1, plot.top(), max(1.0, x2 - x1), plot.height())
            painter.fillRect(zone_rect, color)
            painter.setPen(self.palette().mid().color())
            painter.drawText(
                QRectF(x1, full.top() + 2, max(1.0, x2 - x1), 22),
                Qt.AlignHCenter | Qt.AlignVCenter,
                label,
            )

        # Reference grid for the actual linear-light 0..1 scale.
        grid_pen = QPen(self.palette().mid().color())
        grid_pen.setStyle(Qt.DotLine)
        painter.setPen(grid_pen)
        for value in (0.0, 0.25, 0.50, 0.75, 1.0):
            x = self._x_for_value(plot, value)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawText(
                QRectF(x - 24, plot.bottom() + 5, 48, 18),
                Qt.AlignHCenter | Qt.AlignTop,
                f"{value:.2f}" if value not in (0.0, 1.0) else f"{value:.0f}",
            )

        painter.setPen(self.palette().mid().color())
        painter.drawRect(plot)
        if not self._bins or plot.width() < 2 or plot.height() < 2:
            painter.setPen(self.palette().text().color())
            painter.drawText(plot, Qt.AlignCenter, "Гистограмма появится после анализа изображения")
            return

        peak = max(self._bins) or 1.0
        path = QPainterPath()
        for i, value in enumerate(self._bins):
            x = plot.left() + i * plot.width() / max(len(self._bins) - 1, 1)
            y = plot.bottom() - min(max(value / peak, 0.0), 1.0) * plot.height()
            point = QPointF(x, y)
            if i == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        curve_pen = QPen(self.palette().highlight().color(), 2)
        painter.setPen(curve_pen)
        painter.drawPath(path)

        # Exact analyzer clipping thresholds. They intentionally hug the edges.
        clip_pen = QPen(QColor(220, 90, 90, 190), 1)
        clip_pen.setStyle(Qt.DashLine)
        painter.setPen(clip_pen)
        for value in (self.SHADOW_CLIP, self.HIGHLIGHT_CLIP):
            x = self._x_for_value(plot, value)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))

        # P5/P50/P95 are the most useful anchors for quickly reading the distribution.
        marker_specs = (
            ("p5", "P5", QColor(75, 180, 220)),
            ("p50", "P50", self.palette().text().color()),
            ("p95", "P95", QColor(225, 165, 75)),
        )
        for key, label, color in marker_specs:
            value = self._percentiles.get(key)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            x = self._x_for_value(plot, value)
            pen = QPen(color, 1)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawText(
                QRectF(x - 35, plot.top() + 4, 70, 18),
                Qt.AlignHCenter | Qt.AlignTop,
                f"{label} {value:.3f}",
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Photo Doctor {__version__}")
        icon_path = asset_path("photodoctor.ico")
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.resize(1440, 900)
        self.db_path = Path.home() / ".photodoctor" / "analysis.sqlite3"
        self.worker: BatchWorker | None = None
        self.series_worker: SeriesWorker | None = None
        self.analysis_worker: PhotoAnalysisWorker | None = None
        self.preview_worker: CorrectionPreviewWorker | None = None
        self.analysis_started_at: float | None = None
        self.analysis_stage_text = ""
        self.analysis_timer = QTimer(self)
        self.analysis_timer.setInterval(250)
        self.analysis_timer.timeout.connect(self._update_analysis_elapsed)
        self.preview_started_at: float | None = None
        self.preview_stage_text = ""
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self._update_almaz_preview_elapsed)
        self.almaz_release_worker: AlmazReleaseWorker | None = None
        self.almaz_release_started_at: float | None = None
        self.almaz_release_stage_text = ""
        self.almaz_release_timer = QTimer(self)
        self.almaz_release_timer.setInterval(250)
        self.almaz_release_timer.timeout.connect(self._update_almaz_release_elapsed)
        self.current_series_summary = None
        self.batch_root: Path | None = None
        self.batch_started_at: float | None = None
        self.batch_pause_started: float | None = None
        self.batch_paused_total = 0.0
        self.batch_last_i = 0
        self.batch_total = 0
        self.batch_precision = "normal"
        self.current_rgb = None
        # Cached corrected preview for instant source/preview comparison.  It is
        # invalidated only when the correction recipe or analyzed photo changes.
        self._preview_cache_rgb = None
        self._preview_cache_key: tuple[object, ...] | None = None
        self._preview_recipe_revision = 0
        # Changing a correction recipe never starts an expensive preview on its
        # own.  Once a rendered preview becomes stale, the user explicitly
        # requests the next render via the preview button.
        self._preview_is_stale = False
        self.current_validation_items: list[dict] = []
        self.correction_strengths: dict[str, float] = {}
        self.correction_ai_suggestions: dict[str, dict[str, object]] = {}
        self.correction_regions: dict[str, dict[str, float]] = {}
        self.surface_repair_signatures: set[str] = set()
        self._surface_repair_seeded = False
        self._updating_surface_repair_choices = False
        self._updating_correction_choices = False
        self._updating_plan_choices = False
        self.current_path: Path | None = None
        self.current_image_info = None
        self.current_source_quick_hash: str | None = None
        self.current_metrics = None
        # Session-only cache for the current photo across precision switches.
        # Results are tiny compared with the source pixels, so keep only metrics/
        # metadata and reuse ``current_rgb`` rather than retaining another 24 MP
        # image per precision mode.
        self._analysis_mode_cache: dict[str, dict[str, object]] = {}
        self.current_surface_boxes: list[dict] = []
        self.manual_face_boxes: list[dict[str, float]] = []
        self.manual_eye_boxes: list[dict[str, float]] = []
        self.ai_install_process: QProcess | None = None
        self.almaz_prepare_process: QProcess | None = None
        self.almaz_restoration_process: QProcess | None = None
        self.almaz_restoration_task: str | None = None
        self.almaz_prepare_started_at: float | None = None
        self.almaz_prepare_stage_text = ""
        self.almaz_prepare_timer = QTimer(self)
        self.almaz_prepare_timer.setInterval(250)
        self.almaz_prepare_timer.timeout.connect(self._update_almaz_prepare_elapsed)
        self.settings = QSettings("PhotoDoctor", "PhotoDoctor")
        self._build_ui()
        self._restore_ui_state()

    def _build_ui(self) -> None:
        self._build_menu_bar()
        toolbar = QToolBar("Основное")
        self.addToolBar(toolbar)
        self.open_file_btn = QPushButton("Открыть")
        self.open_file_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        self.open_file_btn.setToolTip("Открыть фотографию")
        self.open_file_btn.clicked.connect(self.open_file)
        self.open_folder_action = QAction("Анализ папки", self)
        self.open_folder_action.triggered.connect(self.open_folder)
        toolbar.addWidget(self.open_file_btn)

        self.save_copy_btn = QPushButton("Сохранить")
        self.save_copy_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.save_copy_btn.setEnabled(False)
        self.save_copy_btn.setToolTip("Сохранить именно выбранный набор исправлений новым файлом. Исходник не перезаписывается; исходные метаданные сохраняются, кроме служебных полей, которые нельзя безопасно перенести.")
        self.save_copy_btn.clicked.connect(self._save_preview_copy)
        toolbar.addWidget(self.save_copy_btn)
        # Пакетное редактирование убрано из пользовательского интерфейса: внутренний batch-код
        # остаётся доступен библиотеке/тестам, но Photo Doctor больше не предлагает полный автомат.
        toolbar.addSeparator()

        self.fit_btn = QPushButton("По окну")
        self.fit_btn.setCheckable(True)
        self.fit_btn.setChecked(True)
        self.fit_btn.setToolTip("Вписать фотографию в окно. Масштаб просмотра не влияет на анализ.")
        self.fit_btn.clicked.connect(lambda: self.image_view.set_fit(True))
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedWidth(30)
        self.zoom_out_btn.setToolTip("Уменьшить масштаб просмотра")
        self.zoom_out_btn.clicked.connect(lambda: self._step_zoom(-25))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(25, 300)
        self.zoom_slider.setSingleStep(5)
        self.zoom_slider.setPageStep(25)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setMinimumWidth(120)
        self.zoom_slider.setMaximumWidth(180)
        self.zoom_slider.setToolTip("Масштаб просмотра 25–300%. Ctrl+колесо мыши меняет масштаб под курсором.")
        self.zoom_slider.valueChanged.connect(self._zoom_slider_changed)
        self.zoom_value_label = QLabel("100%")
        self.zoom_value_label.setMinimumWidth(42)
        self.zoom_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedWidth(30)
        self.zoom_in_btn.setToolTip("Увеличить масштаб просмотра")
        self.zoom_in_btn.clicked.connect(lambda: self._step_zoom(25))
        self.preview_btn = QPushButton("Предпросмотр")
        self.preview_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogContentsView))
        self.preview_btn.setCheckable(True)
        self.preview_btn.setEnabled(False)
        self.preview_btn.setToolTip("Показать выбранные исправления. Подтверждённые проверкой безопасности отмечены автоматически; неподтверждённые можно включить вручную. Исходный файл не изменяется.")
        self.preview_btn.toggled.connect(self._toggle_preview)
        self.confirmed_fixes_label = QLabel("Безопасные автоматические исправления не подтверждены")
        # Keep toolbar feedback readable in both light and dark themes.
        toolbar.addWidget(self.confirmed_fixes_label)
        toolbar.addSeparator()
        precision_label = QLabel("Точность:")
        self.precision_combo = QComboBox()
        self.precision_combo.setToolTip(
            "Быстро — облегчённые карты; Нормально — основной режим; Точно — глубокий анализ до 4096 px; "
            "Максимальный — самый глубокий проход до 8192 px: примерно в 2× больше локальных окон, чем в прежнем четвёртом режиме, и около 4× против «Точно»; "
            "лиц/глаз и кандидатов дефектов по сравнению с «Точно». Пороги качества не ослабляются; растёт объём проверки."
        )
        # New photos always start in the balanced profile. A user can still request
        # a full precise pass explicitly; manual face/eye markup also triggers it.
        selected_index = 0
        for index, profile in enumerate(all_precisions()):
            self.precision_combo.addItem(profile.label, profile.key)
            if profile.key == "normal":
                selected_index = index
        self.precision_combo.setCurrentIndex(selected_index)
        self.precision_combo.currentIndexChanged.connect(self._precision_changed)
        toolbar.addWidget(precision_label)
        toolbar.addWidget(self.precision_combo)
        toolbar_spacer = QWidget()
        toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(toolbar_spacer)
        toolbar.addWidget(self.preview_btn)
        # Keep overlays on their own toolbar so high-DPI Windows layouts do not
        # hide half of the controls behind Qt's overflow chevron.
        self.addToolBarBreak()
        view_toolbar = QToolBar("Просмотр, карты и области")
        self.addToolBar(view_toolbar)
        view_toolbar.addWidget(self.fit_btn)
        view_toolbar.addWidget(self.zoom_out_btn)
        view_toolbar.addWidget(self.zoom_slider)
        view_toolbar.addWidget(self.zoom_value_label)
        view_toolbar.addWidget(self.zoom_in_btn)
        view_toolbar.addSeparator()
        self.defects_btn = QPushButton("Карта дефектов")
        self.defects_btn.setCheckable(True)
        self.defects_btn.setEnabled(False)
        self.defects_btn.setToolTip("Показать карту дефектов. По умолчанию на ней видны только кандидаты, отмеченные в колонке «Лечить». На вкладке «Дефекты» можно временно показать все найденные кандидаты.")
        self.defects_btn.toggled.connect(self._toggle_defects)
        view_toolbar.addWidget(self.defects_btn)
        self.sharpness_btn = QPushButton("Карта резкости")
        self.sharpness_btn.setCheckable(True)
        self.sharpness_btn.setEnabled(False)
        self.sharpness_btn.setToolTip("Локальная карта детализации: красное — мягкие информативные области, зелёное — более резкие. Гладкий фон не окрашивается.")
        self.sharpness_btn.toggled.connect(self._toggle_sharpness)
        view_toolbar.addWidget(self.sharpness_btn)
        self.tone_btn = QPushButton("Карта тонов")
        self.tone_btn.setCheckable(True)
        self.tone_btn.setEnabled(False)
        self.tone_btn.setToolTip("Информационная карта локальных тонов: синие зоны — тёмные, жёлтые/красные — очень светлые. Это не автоматический список дефектов.")
        self.tone_btn.toggled.connect(self._toggle_tone)
        view_toolbar.addWidget(self.tone_btn)
        self.contrast_btn = QPushButton("Карта контраста")
        self.contrast_btn.setCheckable(True)
        self.contrast_btn.setEnabled(False)
        self.contrast_btn.setToolTip("Локальный тональный контраст: сиреневые зоны — плоские информативные участки, янтарные — умеренный контраст. Гладкие области не подсвечиваются.")
        self.contrast_btn.toggled.connect(self._toggle_contrast)
        view_toolbar.addWidget(self.contrast_btn)
        self.faces_btn = QPushButton("Лица")
        self.faces_btn.setCheckable(True)
        self.faces_btn.setEnabled(False)
        self.faces_btn.setToolTip("Показать области лиц, найденные локальным детектором.")
        self.faces_btn.toggled.connect(lambda checked: self.image_view.set_faces_enabled(checked))
        view_toolbar.addWidget(self.faces_btn)
        self.subject_btn = QPushButton("Главный объект")
        self.subject_btn.setCheckable(True)
        self.subject_btn.setEnabled(False)
        self.subject_btn.setToolTip("Показать область главного объекта/группы. Неопределённый объект не рисуется так, будто он известен.")
        self.subject_btn.toggled.connect(lambda checked: self.image_view.set_subject_enabled(checked))
        view_toolbar.addWidget(self.subject_btn)
        self.highlights_btn = QPushButton("Блики")
        self.highlights_btn.setCheckable(True)
        self.highlights_btn.setEnabled(False)
        self.highlights_btn.setToolTip("Показать компактные кандидаты на допустимые зеркальные блики / источники света. Это контекст для потерь в светах, не автоматическое подтверждение.")
        self.highlights_btn.toggled.connect(lambda checked: self.image_view.set_highlights_enabled(checked))
        view_toolbar.addWidget(self.highlights_btn)

        self._fullscreen_windows: list[FullscreenImageWindow] = []
        self.image_view = ImageView()
        self.image_view.region_selected.connect(self._on_manual_region_selected)
        self.image_view.zoom_changed.connect(self._on_image_zoom_changed)
        self.image_view.fullscreen_requested.connect(self._open_fullscreen_viewer)
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(190)

        self._build_result_tab()
        self._build_metrics_tab()
        self._build_histogram_tab()
        self._build_recommendations_tab()
        self._build_plan_tab()
        self._build_almaz_tab()
        self._build_faces_tab()
        self._build_defects_tab()
        self._build_batch_tab()
        self._build_series_tab()
        self._build_ai_tab()
        self._build_technical_tab()
        self._organize_primary_tabs()

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.addWidget(self.image_view)
        self.main_splitter.addWidget(self.tabs)
        self.main_splitter.setStretchFactor(0, 4)
        self.main_splitter.setStretchFactor(1, 2)
        self.main_splitter.setSizes([590, 270])

        self.status_label = QLabel("Готов")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Готово")
        self.progress.setMaximumWidth(260)
        self.pause_btn = QPushButton("Пауза")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_batch_pause)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_batch)

        bottom = QHBoxLayout()
        bottom.addWidget(self.status_label, 1)
        bottom.addWidget(self.progress)
        # Пауза/отмена относились к пакетному анализу и больше не занимают место в обычном UI.
        self.pause_btn.setVisible(False)
        self.cancel_btn.setVisible(False)

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self.main_splitter, 1)
        layout.addLayout(bottom)
        self.setCentralWidget(root)

    def _organize_primary_tabs(self) -> None:
        """Собрать рабочие разделы верхнего уровня и убрать пакетный UI.

        Внутренние batch/series-виджеты остаются созданными для совместимости,
        но пользовательская навигация больше их не показывает. Технические
        страницы собраны в один раздел «Диагностика».
        """
        pages: dict[str, QWidget] = {}
        for index in range(self.tabs.count()):
            pages[self.tabs.tabText(index)] = self.tabs.widget(index)

        while self.tabs.count():
            self.tabs.removeTab(0)

        overview = pages.get("Обзор")
        corrections = pages.get("Исправления")
        defects = pages.get("Дефекты")
        faces = pages.get("Лица")
        for widget, title in (
            (overview, "Обзор"),
            (corrections, "Исправления"),
            (defects, "Дефекты"),
            (faces, "Лица"),
        ):
            if widget is not None:
                self.tabs.addTab(widget, title)

        self.diagnostics_page = QWidget()
        diagnostics_layout = QVBoxLayout(self.diagnostics_page)
        diagnostics_layout.setContentsMargins(4, 4, 4, 4)
        self.diagnostics_tabs = QTabWidget(self.diagnostics_page)
        diagnostics_layout.addWidget(self.diagnostics_tabs)
        for title in ("Метрики", "Гистограмма", "Рекомендации", "ALMAZ", "ИИ", "Технические данные"):
            widget = pages.get(title)
            if widget is not None:
                self.diagnostics_tabs.addTab(widget, title)
        self.tabs.addTab(self.diagnostics_page, "Диагностика")

    def _build_result_tab(self) -> None:
        page = QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QFrame.Shape.NoFrame)
        page.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(7)
        page.setWidget(content)

        # One compact status line instead of four framed cards.  The abstract
        # "potential" score stays in diagnostics and no longer competes with
        # actionable information on the Overview page.
        self.overview_stats_label = QLabel(
            "<b>Качество</b> —/100 &nbsp;&nbsp;&nbsp; <b>Уверенность</b> —% "
            "&nbsp;&nbsp;&nbsp; <b>Проблем найдено</b> —"
        )
        self.overview_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.overview_stats_label.setWordWrap(True)
        self.overview_stats_label.setStyleSheet(
            "font-size: 11pt; padding: 5px 2px 7px 2px; border-bottom: 1px solid palette(mid);"
        )
        layout.addWidget(self.overview_stats_label)

        photo_title = QLabel("<b>О фотографии</b>")
        photo_title.setStyleSheet("font-size: 11pt; margin-top: 2px;")
        layout.addWidget(photo_title)

        self.profile_label = QLabel("—")
        self.profile_label.setWordWrap(True)
        self.profile_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.semantic_label = QLabel("—")
        self.semantic_label.setWordWrap(True)
        self.semantic_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.subject_label = QLabel("—")
        self.subject_label.setWordWrap(True)
        self.subject_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.photo_info_label = QLabel("Разрешение и формат: —")
        self.photo_info_label.setWordWrap(True)
        self.photo_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.camera_info_label = QLabel("Камера и параметры съёмки: —")
        self.camera_info_label.setWordWrap(True)
        self.camera_info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.camera_info_label.setStyleSheet("color: palette(text);")
        for widget in (
            self.profile_label, self.semantic_label, self.subject_label,
            self.photo_info_label, self.camera_info_label,
        ):
            layout.addWidget(widget)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        fixes_title = QLabel("<b>Исправления</b>")
        fixes_title.setStyleSheet("font-size: 11pt; margin-top: 2px;")
        layout.addWidget(fixes_title)

        self.fix_summary_label = QLabel("Безопасные автоматические исправления не подтверждены")
        self.fix_summary_label.setWordWrap(True)
        self.fix_summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.fix_summary_label)

        # Hidden canonical selection store shared with the Corrections tab and
        # preview engine.  Overview itself remains read-only.
        self.fix_actions_list = QListWidget(content)
        self.fix_actions_list.setVisible(False)
        self.fix_actions_list.itemChanged.connect(self._on_correction_item_changed)

        self.fix_summary_list = QListWidget(content)
        self.fix_summary_list.setWordWrap(True)
        self.fix_summary_list.setMaximumHeight(118)
        self.fix_summary_list.setFrameShape(QFrame.Shape.NoFrame)
        self.fix_summary_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.fix_summary_list.setFocusPolicy(Qt.NoFocus)
        self.fix_summary_list.setToolTip(
            "Только резюме найденных и предлагаемых исправлений. Выбор выполняется на вкладке «Исправления», "
            "а дефекты поверхности — только вручную на вкладке «Дефекты»."
        )
        layout.addWidget(self.fix_summary_list)

        self.active_corrections_label = QLabel("Будет применено: ничего")
        self.active_corrections_label.setWordWrap(True)
        self.active_corrections_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.active_corrections_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self.active_corrections_label)

        self.correction_feedback_checkbox = QCheckBox("Запоминать мои решения для локального обучения")
        feedback_value = str(self.settings.value("correction_feedback_enabled", "true")).strip().lower()
        self.correction_feedback_checkbox.setChecked(feedback_value not in {"0", "false", "no", "off"})
        self.correction_feedback_checkbox.setToolTip(
            "Сохраняет только локально, какие исправления вы оставили, отключили или включили вручную. "
            "Модель не переобучается автоматически после каждого сохранения."
        )
        self.correction_feedback_checkbox.stateChanged.connect(self._correction_feedback_setting_changed)
        self.correction_feedback_history_label = QLabel(
            "Локальная история решений: пока нет сохранённых решений. Автопереобучение выключено."
        )
        self.correction_feedback_history_label.setWordWrap(True)
        self.correction_feedback_history_label.setStyleSheet("color: #777;")

        self.aesthetic_label = QLabel("<b>Эстетика:</b> —")
        self.aesthetic_label.setWordWrap(True)
        self.aesthetic_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.reliability_label = QLabel("<b>Надёжность анализа:</b> —")
        self.reliability_label.setWordWrap(True)
        self.reliability_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # Detailed diagnostics are intentionally collapsed.  It is a plain
        # panel, not another titled rectangle inside a titled rectangle.
        self.result_text = QTextBrowser()
        self.result_text.setReadOnly(True)
        self.result_text.setFrameShape(QFrame.Shape.NoFrame)
        self.result_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.result_text.setMinimumHeight(90)
        self.result_text.setText("Откройте фотографию для анализа.")

        details_row = QHBoxLayout()
        details_row.addStretch(1)
        self.overview_details_btn = QPushButton("Показать подробности анализа")
        self.overview_details_btn.setCheckable(True)
        details_row.addWidget(self.overview_details_btn)
        layout.addLayout(details_row)

        self.overview_details_panel = QWidget(content)
        details_layout = QVBoxLayout(self.overview_details_panel)
        details_layout.setContentsMargins(0, 2, 0, 0)
        details_layout.setSpacing(5)
        details_layout.addWidget(self.aesthetic_label)
        details_layout.addWidget(self.reliability_label)
        details_layout.addWidget(self.result_text)
        self.overview_details_panel.setVisible(False)
        self.overview_details_btn.toggled.connect(self.overview_details_panel.setVisible)
        self.overview_details_btn.toggled.connect(
            lambda checked: self.overview_details_btn.setText(
                "Скрыть подробности анализа" if checked else "Показать подробности анализа"
            )
        )
        layout.addWidget(self.overview_details_panel)
        layout.addStretch(1)
        self.tabs.addTab(page, "Обзор")

    def _build_metrics_tab(self) -> None:
        self.metrics = QTableWidget(0, 5)
        self.metrics.setHorizontalHeaderLabels(["Статус", "Параметр", "Оценка", "Уверенность", "Диагноз"])
        header = self.metrics.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.metrics.setWordWrap(True)
        self.metrics.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics.verticalHeader().setVisible(False)
        self.metrics.verticalHeader().setMinimumSectionSize(28)
        self.metrics_resize_timer = QTimer(self)
        self.metrics_resize_timer.setSingleShot(True)
        self.metrics_resize_timer.setInterval(40)
        self.metrics_resize_timer.timeout.connect(self.metrics.resizeRowsToContents)
        header.sectionResized.connect(lambda *_args: self.metrics_resize_timer.start())
        self.tabs.addTab(self.metrics, "Метрики")


    def _build_menu_bar(self) -> None:
        help_menu = self.menuBar().addMenu("Справка")
        about_action = QAction("О программе", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "О программе Photo Doctor",
            (
                f"<b>Photo Doctor {__version__}</b><br><br>"
                "Локальная диагностика и безопасное улучшение фотографий.<br>"
                "Исходные фотографии не перезаписываются.<br><br>"
                f"<b>Автор проекта: {__author__}</b><br>"
                "AI-помощники разработки: ChatGPT / Codex<br>"
                "Локальный ИИ Photo Doctor: Astra<br><br>"
                "© 2026 Photo Doctor"
            ),
        )

    def _build_histogram_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.histogram_summary = QLabel("<b>Экспозиция:</b> —")
        self.histogram_summary.setWordWrap(True)
        self.histogram_summary.setTextFormat(Qt.RichText)
        self.histogram_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.histogram_summary.setFrameShape(QFrame.StyledPanel)
        self.histogram_summary.setMargin(8)
        self.histogram = HistogramWidget()
        self.histogram_info = QLabel("Процентили: —")
        self.histogram_info.setWordWrap(True)
        self.histogram_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.histogram_summary)
        layout.addWidget(self.histogram, 1)
        layout.addWidget(self.histogram_info)
        self.tabs.addTab(page, "Гистограмма")

    def _build_recommendations_tab(self) -> None:
        page = QWidget()
        layout = QHBoxLayout(page)

        # Keep explicit Python references to both the group boxes and their child
        # QListWidgets.  Returning only the child allowed the temporary QGroupBox
        # wrapper to be collected, which deleted the C++ child object as well.
        (self.safe_recommendations_box, self.safe_recommendations) = self._recommendation_group(
            "Можно безопасно"
        )
        (self.caution_recommendations_box, self.caution_recommendations) = self._recommendation_group(
            "С осторожностью"
        )
        (self.avoid_recommendations_box, self.avoid_recommendations) = self._recommendation_group(
            "Не делать автоматически"
        )
        for box in (
            self.safe_recommendations_box,
            self.caution_recommendations_box,
            self.avoid_recommendations_box,
        ):
            layout.addWidget(box, 1)
        self.tabs.addTab(page, "Рекомендации")

    @staticmethod
    def _recommendation_group(title: str) -> tuple[QGroupBox, QListWidget]:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        widget = QListWidget(box)
        widget.setWordWrap(True)
        widget.addItem("Рекомендации появятся после анализа изображения.")
        layout.addWidget(widget)
        return box, widget

    def _build_plan_tab(self) -> None:
        self.plan_page = QWidget()
        layout = QVBoxLayout(self.plan_page)
        layout.setContentsMargins(8, 6, 8, 6)

        hint = QLabel(
            "Выберите, что применить, настройте силу и при необходимости ограничьте область. "
            "Дефекты поверхности никогда не включаются автоматически: их выбор выполняется только на вкладке «Дефекты»."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        plan_choice_buttons = QHBoxLayout()
        self.plan_auto_select_btn = QPushButton("Выбрать рекомендуемые")
        self.plan_auto_select_btn.clicked.connect(self._reset_correction_selection)
        self.plan_select_all_btn = QPushButton("Выбрать всё доступное")
        self.plan_select_all_btn.setToolTip("Включить также коррекции, оставленные только для ручной проверки.")
        self.plan_select_all_btn.clicked.connect(self._select_all_corrections)
        self.plan_clear_btn = QPushButton("Снять всё")
        self.plan_clear_btn.clicked.connect(self._clear_correction_selection)
        plan_choice_buttons.addWidget(self.plan_auto_select_btn)
        plan_choice_buttons.addWidget(self.plan_select_all_btn)
        plan_choice_buttons.addWidget(self.plan_clear_btn)
        plan_choice_buttons.addStretch(1)
        layout.addLayout(plan_choice_buttons)

        feedback_row = QHBoxLayout()
        feedback_row.addWidget(self.correction_feedback_checkbox)
        feedback_row.addWidget(self.correction_feedback_history_label, 1)
        layout.addLayout(feedback_row)

        self.plan_table = QTableWidget(0, 11)
        self.plan_table.setHorizontalHeaderLabels([
            "Применить", "Решение", "Исправление", "Сила", "Персональная подстройка",
            "Область", "Приоритет", "Важность", "Исправимость", "Уверенность", "Статус"
        ])
        plan_header_tooltips = {
            3: "Сила коррекции, предложенная ИИ. Ползунок позволяет изменить её вручную; кнопка ↺ возвращает рекомендацию ИИ.",
            4: "Насколько рекомендация силы опирается на накопленные пользовательские решения и персональную подстройку.",
            5: "Где будет применена коррекция: всё фото, автоматическая локальная маска, главный объект или выбранная вами область.",
            6: "Сводная очередность исправления: учитывает важность проблемы, исправимость, уверенность анализа и ограничения безопасности.",
            7: "Насколько заметна или существенна найденная проблема. Чем больше число, тем сильнее проблема выражена.",
            8: "Насколько эту проблему можно исправить текущим методом без заметного риска испортить фотографию.",
            9: "Насколько анализ уверен, что описанная проблема действительно присутствует на фотографии.",
            10: "Результат проверки пробной коррекции: улучшила ли она целевую область и не ухудшила ли остальную фотографию.",
        }
        for column, tooltip in plan_header_tooltips.items():
            header_item = self.plan_table.horizontalHeaderItem(column)
            if header_item is not None:
                header_item.setToolTip(tooltip)
        self.plan_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.plan_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.plan_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.plan_table.setSortingEnabled(True)
        self.plan_table.setWordWrap(False)
        self.plan_table.verticalHeader().setVisible(False)
        self.plan_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.plan_table.verticalHeader().setDefaultSectionSize(32)
        header = self.plan_table.horizontalHeader()
        for col in (0, 1, 3, 4, 6, 7, 8, 9, 10):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(5, QHeaderView.Interactive)
        self.plan_table.setColumnWidth(5, 230)
        # В основном виде показываем только рабочие поля. Технические оценки
        # остаются в модели таблицы и выводятся в подробностях выбранной строки.
        for column in (4, 6, 7, 8):
            self.plan_table.setColumnHidden(column, True)
        self.plan_table.itemChanged.connect(self._on_plan_correction_item_changed)
        self.plan_table.currentCellChanged.connect(self._show_plan_details)
        layout.addWidget(self.plan_table, 1)

        details_box = QGroupBox("Подробности выбранной строки")
        details_layout = QVBoxLayout(details_box)
        self.plan_details = QLabel("Выберите строку плана.")
        self.plan_details.setWordWrap(True)
        self.plan_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.plan_details.setMinimumHeight(58)
        details_layout.addWidget(self.plan_details)
        layout.addWidget(details_box)

        self.tabs.addTab(self.plan_page, "Исправления")

    def _build_almaz_tab(self) -> None:
        # Диагностическая страница ALMAZ намеренно плоская: информационные
        # секции не заключаются в QGroupBox, чтобы рамки и внутренние поля не
        # съедали полезную высоту на небольших окнах и при DPI > 100%.
        self.almaz_page = QScrollArea()
        self.almaz_page.setWidgetResizable(True)
        self.almaz_page.setFrameShape(QFrame.Shape.NoFrame)
        self.almaz_page.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.almaz_page.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        self.almaz_page.setWidget(content)

        def add_section_title(title_text: str) -> None:
            label = QLabel(f"<b>{title_text}</b>")
            label.setStyleSheet("font-size: 11pt; margin-top: 3px;")
            layout.addWidget(label)

        def add_separator() -> None:
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(line)

        title = QLabel("<b>ALMAZ — AI-восстановление изображения</b>")
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(title)
        intro = QLabel(
            "Функции ALMAZ применяются только вручную через «Исправления» и предпросмотр. "
            "Если проверенной ONNX-модели нет, Photo Doctor использует безопасный fallback."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #666;")
        layout.addWidget(intro)

        add_section_title("Модули ALMAZ")

        def add_module_row(title_text: str, description: str, attr_prefix: str, callback) -> None:
            row = QWidget(content)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 1, 0, 1)
            row_layout.setSpacing(8)
            text_box = QVBoxLayout()
            text_box.setContentsMargins(0, 0, 0, 0)
            text_box.setSpacing(1)
            name = QLabel(f"<b>{title_text}</b>")
            detail = QLabel(description)
            detail.setWordWrap(True)
            detail.setStyleSheet("color: #666;")
            state = QLabel("Состояние: проверяется…")
            state.setTextInteractionFlags(Qt.TextSelectableByMouse)
            text_box.addWidget(name)
            text_box.addWidget(detail)
            text_box.addWidget(state)
            button = QPushButton("Подготовить модель")
            button.clicked.connect(callback)
            row_layout.addLayout(text_box, 1)
            row_layout.addWidget(button, 0, Qt.AlignTop)
            layout.addWidget(row)
            setattr(self, f"almaz_{attr_prefix}_state", state)
            setattr(self, f"almaz_{attr_prefix}_prepare_btn", button)

        add_module_row(
            "Super Resolution x2",
            "Бережное увеличение старых малых фото в 2 раза с Identity Guard и пост-проверкой.",
            "sr",
            self._prepare_almaz_x2,
        )
        add_module_row(
            "AI Denoise",
            "NAFNet-SIDD: включается только когда обычный анализ подтверждает шум.",
            "denoise",
            lambda: self._prepare_almaz_restoration("denoise"),
        )
        add_module_row(
            "AI Deblur",
            "NAFNet-GoPro: выполняется до x2 и только после проверки необходимости.",
            "deblur",
            lambda: self._prepare_almaz_restoration("deblur"),
        )
        add_module_row(
            "JPEG Recovery",
            "NAFNet-REDS для тяжёлых JPEG/blur-артефактов; мягкий deblock без причины не заменяет.",
            "jpeg",
            lambda: self._prepare_almaz_restoration("jpeg_recovery"),
        )

        add_separator()
        add_section_title("Ускоритель / runtime")
        self.almaz_provider_state = QLabel("Провайдер: проверяется…")
        self.almaz_provider_state.setWordWrap(True)
        self.almaz_provider_state.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.almaz_provider_state)

        # Кнопки намеренно разбиты на две строки: один длинный QHBoxLayout
        # обрезался на узких окнах и при системном масштабе 125–150%.
        runtime_row_1 = QHBoxLayout()
        self.almaz_runtime_main_btn = QPushButton("Установить ONNX Runtime")
        self.almaz_runtime_main_btn.clicked.connect(self._install_ai_runtime)
        self.almaz_prepare_all_btn = QPushButton("Подготовить Denoise + Deblur + JPEG Recovery")
        self.almaz_prepare_all_btn.clicked.connect(lambda: self._prepare_almaz_restoration("all"))
        self.almaz_refresh_btn = QPushButton("Обновить состояние")
        self.almaz_refresh_btn.clicked.connect(self._refresh_almaz_panel)
        runtime_row_1.addWidget(self.almaz_runtime_main_btn)
        runtime_row_1.addWidget(self.almaz_prepare_all_btn, 1)
        runtime_row_1.addWidget(self.almaz_refresh_btn)
        layout.addLayout(runtime_row_1)

        runtime_row_2 = QHBoxLayout()
        self.almaz_install_release_btn = QPushButton("Установить обученную модель…")
        self.almaz_install_release_btn.setToolTip("Установить один проверенный ALMAZ release-bundle ZIP после обучения и закрытого экзамена.")
        self.almaz_install_release_btn.clicked.connect(self._install_almaz_release_bundle)
        self.almaz_rollback_release_btn = QPushButton("Откатить модель…")
        self.almaz_rollback_release_btn.setToolTip("Вернуть последнюю проверенную резервную копию выбранной ALMAZ-модели.")
        self.almaz_rollback_release_btn.clicked.connect(self._rollback_almaz_release_model)
        self.almaz_open_plan_btn = QPushButton("Перейти к исправлениям")
        self.almaz_open_plan_btn.clicked.connect(lambda: self.tabs.setCurrentWidget(self.plan_page))
        runtime_row_2.addWidget(self.almaz_install_release_btn)
        runtime_row_2.addWidget(self.almaz_rollback_release_btn)
        runtime_row_2.addWidget(self.almaz_open_plan_btn)
        runtime_row_2.addStretch(1)
        layout.addLayout(runtime_row_2)

        add_separator()
        add_section_title("Текущее фото")
        self.almaz_current_photo = QLabel(
            "Фото ещё не анализировалось. После анализа здесь появится рекомендация x2 и состояние ALMAZ-коррекций."
        )
        self.almaz_current_photo.setWordWrap(True)
        self.almaz_current_photo.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.almaz_current_photo)

        add_separator()
        self.almaz_log_toggle_btn = QPushButton("Показать журнал подготовки")
        self.almaz_log_toggle_btn.setCheckable(True)
        self.almaz_log_toggle_btn.setChecked(False)
        layout.addWidget(self.almaz_log_toggle_btn, 0, Qt.AlignLeft)
        self.almaz_activity_log = QTextBrowser()
        self.almaz_activity_log.setReadOnly(True)
        self.almaz_activity_log.setMinimumHeight(90)
        self.almaz_activity_log.setMaximumHeight(160)
        self.almaz_activity_log.setPlaceholderText("Лог подготовки ALMAZ-моделей")
        self.almaz_activity_log.setVisible(False)
        self.almaz_log_toggle_btn.toggled.connect(self.almaz_activity_log.setVisible)
        self.almaz_log_toggle_btn.toggled.connect(
            lambda checked: self.almaz_log_toggle_btn.setText(
                "Скрыть журнал подготовки" if checked else "Показать журнал подготовки"
            )
        )
        layout.addWidget(self.almaz_activity_log)
        layout.addStretch(1)

        self.tabs.addTab(self.almaz_page, "ALMAZ")
        self._refresh_almaz_panel()

    def _build_faces_tab(self) -> None:
        self.faces_page = QWidget()
        layout = QVBoxLayout(self.faces_page)
        layout.setContentsMargins(6, 6, 6, 6)
        info = QLabel(
            "Автодетектор работает локально и может пропускать наклонённые/старые лица. "
            "Если это произошло, обведите лицо или глаз мышью: ручная отметка сохранится локально для этого файла."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        controls = QHBoxLayout()
        self.manual_face_btn = QPushButton("Добавить лицо вручную")
        self.manual_eye_btn = QPushButton("Добавить глаз вручную")
        self.manual_clear_btn = QPushButton("Очистить ручные отметки")
        self.manual_face_btn.setToolTip("Нажмите, затем обведите рамкой пропущенное лицо на фотографии.")
        self.manual_eye_btn.setToolTip("Нажмите, затем обведите пропущенный глаз внутри уже найденного/добавленного лица.")
        self.manual_clear_btn.setToolTip("Удалить ручные рамки лица/глаз для текущего файла и снова запустить автоматический анализ.")
        self.manual_face_btn.clicked.connect(lambda: self._start_manual_region_selection("face"))
        self.manual_eye_btn.clicked.connect(lambda: self._start_manual_region_selection("eye"))
        self.manual_clear_btn.clicked.connect(self._clear_manual_vision_overrides)
        controls.addWidget(self.manual_face_btn)
        controls.addWidget(self.manual_eye_btn)
        controls.addWidget(self.manual_clear_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.manual_vision_status = QLabel("Ручных отметок нет.")
        self.manual_vision_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.manual_vision_status)

        self.faces_table = QTableWidget(0, 9)
        self.faces_table.setHorizontalHeaderLabels([
            "№", "Источник", "Размер", "Резкость", "Яркость", "Глаз", "Худший глаз", "Уверенность", "Статус"
        ])
        header = self.faces_table.horizontalHeader()
        for col in range(9):
            header.setSectionResizeMode(col, QHeaderView.Stretch if col == 2 else QHeaderView.ResizeToContents)
        self.faces_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.faces_table.verticalHeader().setVisible(False)
        self.faces_table.setToolTip("Если лицо пропущено, используйте «Добавить лицо вручную». Отсутствие детекции не считается дефектом фотографии.")
        layout.addWidget(self.faces_table, 1)
        self.tabs.addTab(self.faces_page, "Лица")

    @staticmethod
    def _norm_iou(a: dict[str, float], b: dict[str, float]) -> float:
        ax, ay, aw, ah = (float(a.get(k, 0.0)) for k in ("x", "y", "w", "h"))
        bx, by, bw, bh = (float(b.get(k, 0.0)) for k in ("x", "y", "w", "h"))
        x1 = max(ax, bx); y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw); y2 = min(ay + ah, by + bh)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / max(union, 1e-9)

    def _start_manual_region_selection(self, kind: str) -> None:
        if self.current_path is None or self.current_rgb is None:
            QMessageBox.information(self, "Photo Doctor", "Сначала откройте фотографию.")
            return
        if kind == "eye":
            metric = self.current_metrics.get("faces") if isinstance(self.current_metrics, dict) else None
            raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
            if int(raw.get("face_count", 0) or 0) <= 0:
                QMessageBox.information(
                    self, "Photo Doctor", "Сначала добавьте/найдите лицо, затем отметьте глаз внутри него."
                )
                return
        self.faces_btn.setEnabled(True)
        self.faces_btn.setChecked(True)
        self.image_view.begin_region_selection(kind)
        what = "лицо" if kind == "face" else "глаз"
        self.status_label.setText(f"Ручная разметка: обведите {what} мышью. Правая кнопка отменяет режим.")

    def _validation_item_for_action(self, key: str) -> dict | None:
        """Вернуть ту же проверенную коррекцию, которую показывает таблица плана.

        Основной источник — `current_validation_items`; при перестроении интерфейса
        те же данные могут временно оставаться только в метрике текущего анализа.
        Единый поиск не даёт таблице показать кнопку, которая затем не сработает.
        """
        key = str(key or "").strip()
        if not key:
            return None
        for row in self.current_validation_items:
            if isinstance(row, dict) and str(row.get("action_key", "")) == key:
                return row
        metrics = self.current_metrics if isinstance(self.current_metrics, dict) else {}
        validation_metric = metrics.get("recommendation_validation")
        validation_raw = (
            validation_metric.raw_value
            if validation_metric is not None and isinstance(validation_metric.raw_value, dict)
            else {}
        )
        raw_items = validation_raw.get("items", []) if isinstance(validation_raw, dict) else []
        if isinstance(raw_items, list):
            for row in raw_items:
                if isinstance(row, dict) and str(row.get("action_key", "")) == key:
                    return row
        return None

    def _start_correction_region_selection(self, key: str) -> None:
        key = str(key or "")
        if not key or self.current_rgb is None:
            return
        item = self._validation_item_for_action(key)
        if item is None or not preview_action_available(item):
            QMessageBox.information(
                self,
                "Photo Doctor",
                "Эта строка сейчас является только диагностикой: исполняемой коррекции для неё нет.",
            )
            return
        if key in {"red_eye", "surface_defects"}:
            QMessageBox.information(self, "Photo Doctor", "Эта коррекция уже локальна по найденным объектам.")
            return
        candidate = str(item.get("candidate", ""))
        if key == "super_resolution":
            QMessageBox.information(self, "Photo Doctor", "ALMAZ x2 изменяет размер всего кадра и не применяется к отдельной области.")
            return
        if candidate.startswith("almaz_ai_"):
            QMessageBox.information(
                self, "Photo Doctor",
                "ALMAZ AI-восстановление сейчас работает на всём кадре. Локальные области для AI Denoise/Deblur/JPEG Recovery будут добавлены только после отдельной проверки масочного режима.",
            )
            return
        self.image_view.begin_region_selection(f"correction:{key}")
        self.status_label.setText("Обведите область применения коррекции мышью. Правая кнопка отменяет выбор.")

    def _clear_correction_region(self, key: str) -> None:
        key = str(key or "")
        if key in self.correction_regions:
            self.correction_regions.pop(key, None)
            if isinstance(self.current_metrics, dict):
                self._show_plan(self.current_metrics)
            self._update_correction_controls(refresh_preview=True)
            self.status_label.setText("Область коррекции сброшена: действие снова применяется ко всему фото.")

    def _show_red_eye_candidates_on_image(self) -> None:
        """Temporarily show the exact eye ROIs that triggered red-eye detection.

        The frame is diagnostic evidence only. It deliberately does not mean the
        candidate was proved to be a real red-eye defect.
        """
        metrics = self.current_metrics if isinstance(self.current_metrics, dict) else {}
        metric = metrics.get("red_eye")
        raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
        candidates = raw.get("candidates", []) if isinstance(raw, dict) else []
        boxes: list[dict[str, object]] = []
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict) or not bool(candidate.get("suspicious", False)):
                    continue
                try:
                    box = {
                        "x": float(candidate.get("x", 0.0)),
                        "y": float(candidate.get("y", 0.0)),
                        "w": float(candidate.get("w", 0.0)),
                        "h": float(candidate.get("h", 0.0)),
                        "user_label": "defect",
                        "overlay_kind": "red_eye",
                    }
                except (TypeError, ValueError, OverflowError):
                    continue
                if all(np.isfinite(float(box[k])) for k in ("x", "y", "w", "h")) and box["w"] > 0 and box["h"] > 0:
                    boxes.append(box)
        if not boxes:
            self.status_label.setText("Подозрительные красные глаза не найдены: показывать на фото нечего.")
            return
        if self.sharpness_btn.isChecked():
            self.sharpness_btn.setChecked(False)
        if self.tone_btn.isChecked():
            self.tone_btn.setChecked(False)
        if self.contrast_btn.isChecked():
            self.contrast_btn.setChecked(False)
        self.image_view.set_overlay_boxes(boxes)
        self.image_view.set_overlay_selection(None)
        self.image_view.set_overlay_enabled(True)
        self.status_label.setText(
            f"Показано подозрительных областей глаз: {len(boxes)}. Красная рамка показывает место срабатывания, а не подтверждённый дефект."
        )

    def _save_manual_vision_overrides(self) -> None:
        if self.current_path is None:
            return
        with AnalysisDatabase(self.db_path) as db:
            db.save_manual_vision_overrides(self.current_path, self.manual_face_boxes, self.manual_eye_boxes)

    def _on_manual_region_selected(self, kind: str, box_obj) -> None:
        if self.current_path is None or not isinstance(box_obj, dict):
            return
        try:
            box = {k: float(box_obj.get(k, 0.0)) for k in ("x", "y", "w", "h")}
        except (TypeError, ValueError):
            return
        if str(kind).startswith("correction:"):
            key = str(kind).split(":", 1)[1].strip()
            if key:
                self.correction_regions[key] = box
                if isinstance(self.current_metrics, dict):
                    self._show_plan(self.current_metrics)
                self._update_correction_controls(refresh_preview=True)
                self.status_label.setText(f"Область коррекции «{key}» выбрана; предпросмотр использует её локально.")
            return
        if kind == "face":
            if box["w"] < 0.03 or box["h"] < 0.03:
                QMessageBox.information(self, "Photo Doctor", "Рамка лица слишком мала. Обведите лицо целиком.")
                return
            self.manual_face_boxes = [old for old in self.manual_face_boxes if self._norm_iou(old, box) < 0.55]
            self.manual_face_boxes.append(box)
        elif kind == "eye":
            cx = box["x"] + box["w"] / 2.0
            cy = box["y"] + box["h"] / 2.0
            face_metric = self.current_metrics.get("faces") if isinstance(self.current_metrics, dict) else None
            face_raw = face_metric.raw_value if face_metric is not None and isinstance(face_metric.raw_value, dict) else {}
            faces = face_raw.get("faces", []) if isinstance(face_raw, dict) else []
            inside = any(
                isinstance(face, dict)
                and float(face.get("x", 0.0)) <= cx <= float(face.get("x", 0.0)) + float(face.get("w", 0.0))
                and float(face.get("y", 0.0)) <= cy <= float(face.get("y", 0.0)) + float(face.get("h", 0.0))
                for face in faces
            )
            if not inside:
                QMessageBox.information(
                    self, "Photo Doctor", "Глаз должен находиться внутри найденного или вручную добавленного лица."
                )
                return
            self.manual_eye_boxes = [old for old in self.manual_eye_boxes if self._norm_iou(old, box) < 0.45]
            self.manual_eye_boxes.append(box)
        else:
            return
        try:
            self._save_manual_vision_overrides()
        except Exception as exc:
            QMessageBox.warning(self, "Photo Doctor", f"Не удалось сохранить ручную разметку: {exc}")
            return
        path = self.current_path
        self._set_precision_combo("precise")
        self.status_label.setText("Ручная разметка сохранена; выполняется точный пересчёт текущего фото…")
        QApplication.processEvents()
        self._open_path(
            path, precision_override="precise", reset_view=False,
            completion_message="Готово: ручная разметка сохранена, точный пересчёт лиц, глаз и рекомендаций завершён",
        )

    def _clear_manual_vision_overrides(self) -> None:
        if self.current_path is None:
            return
        if not self.manual_face_boxes and not self.manual_eye_boxes:
            self.manual_vision_status.setText("Ручных отметок нет.")
            return
        try:
            with AnalysisDatabase(self.db_path) as db:
                db.clear_manual_vision_overrides(self.current_path)
        except Exception as exc:
            QMessageBox.warning(self, "Photo Doctor", f"Не удалось удалить ручную разметку: {exc}")
            return
        self.manual_face_boxes = []
        self.manual_eye_boxes = []
        path = self.current_path
        self._set_precision_combo("normal")
        self._open_path(
            path, precision_override="normal", reset_view=False,
            completion_message="Готово: ручные отметки удалены, автоматический анализ лиц/глаз завершён",
        )

    def _build_defects_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self.defects_feedback_info = QLabel(
            "Ваши метки сохраняются локально и используются для проверки/будущего обучения модели. "
            "Модель не переобучается автоматически во время работы."
        )
        self.defects_feedback_info.setWordWrap(True)
        layout.addWidget(self.defects_feedback_info)
        self.feedback_stats_label = QLabel("Ручная статистика модели появится после первых меток.")
        self.feedback_stats_label.setWordWrap(True)
        self.feedback_stats_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.feedback_stats_label.setStyleSheet("color: #666;")
        layout.addWidget(self.feedback_stats_label)

        self.defects_table = QTableWidget(0, 11)
        self.defects_table.setHorizontalHeaderLabels([
            "Лечить", "№", "Контекст", "Surface AI v2", "Meta-контекст", "Итог", "Моя метка",
            "Сигнал", "Тип", "Полярность", "Положение"
        ])
        self.defects_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.defects_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.defects_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.defects_table.verticalHeader().setVisible(False)
        header = self.defects_table.horizontalHeader()
        for col in range(0, 10):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(10, QHeaderView.Stretch)
        layout.addWidget(self.defects_table, 1)

        map_options = QHBoxLayout()
        self.surface_show_all_checkbox = QCheckBox("Показывать все найденные на карте")
        self.surface_show_all_checkbox.setChecked(False)
        self.surface_show_all_checkbox.setToolTip(
            "Диагностический режим: временно показать на карте все найденные кандидаты, "
            "не меняя галочки «Лечить». По умолчанию карта показывает только выбранные цели лечения."
        )
        self.surface_show_all_checkbox.toggled.connect(self._surface_overlay_mode_changed)
        map_options.addWidget(self.surface_show_all_checkbox)
        map_options.addStretch(1)
        layout.addLayout(map_options)

        buttons = QHBoxLayout()
        self.feedback_defect_btn = QPushButton("Это дефект [D]")
        self.feedback_natural_btn = QPushButton("Естественная деталь [N]")
        self.feedback_uncertain_btn = QPushButton("Не уверен [U]")
        self.feedback_clear_btn = QPushButton("Сбросить метку")
        self.repair_all_btn = QPushButton("Лечить все кандидаты")
        self.repair_clear_btn = QPushButton("Снять лечение")
        self.repair_all_btn.setToolTip("Явно включить локальное восстановление для всех показанных кандидатов. Перед сохранением обязательно проверьте предпросмотр.")
        self.repair_clear_btn.setToolTip("Снять все индивидуальные цели лечения дефектов поверхности.")
        self.repair_all_btn.clicked.connect(lambda: self._set_all_surface_repair(True))
        self.repair_clear_btn.clicked.connect(lambda: self._set_all_surface_repair(False))
        self.feedback_export_btn = QPushButton("Экспорт разметки…")
        self.feedback_export_btn.setToolTip("Экспортировать фрагменты 96×96 для совместимости, контекст 256×256 и manifest.jsonl для будущего обучения verifier v2. Полные пути к исходникам не экспортируются.")
        self.feedback_export_btn.clicked.connect(self._export_surface_feedback)
        self.feedback_defect_btn.setShortcut(QKeySequence("D"))
        self.feedback_natural_btn.setShortcut(QKeySequence("N"))
        self.feedback_uncertain_btn.setShortcut(QKeySequence("U"))
        self.feedback_clear_btn.setShortcut(QKeySequence(Qt.Key_Delete))
        self.feedback_defect_btn.clicked.connect(lambda: self._set_surface_feedback("defect"))
        self.feedback_natural_btn.clicked.connect(lambda: self._set_surface_feedback("natural_detail"))
        self.feedback_uncertain_btn.clicked.connect(lambda: self._set_surface_feedback("uncertain"))
        self.feedback_clear_btn.clicked.connect(lambda: self._set_surface_feedback(None))
        for button in (self.feedback_defect_btn, self.feedback_natural_btn, self.feedback_uncertain_btn, self.feedback_clear_btn):
            button.setEnabled(False)
            buttons.addWidget(button)
        buttons.addWidget(self.repair_all_btn)
        buttons.addWidget(self.repair_clear_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.feedback_export_btn)
        layout.addLayout(buttons)
        self.defects_table.itemSelectionChanged.connect(self._surface_feedback_selection_changed)
        self.defects_table.itemChanged.connect(self._on_surface_repair_item_changed)
        self.tabs.addTab(page, "Дефекты")

    def _surface_feedback_selection_changed(self) -> None:
        row = self.defects_table.currentRow()
        enabled = bool(self.current_path is not None and row >= 0 and self.current_surface_boxes)
        for button in (self.feedback_defect_btn, self.feedback_natural_btn, self.feedback_uncertain_btn, self.feedback_clear_btn):
            button.setEnabled(enabled)
        self._refresh_surface_overlay_selection(row if enabled else -1)
        if enabled and not self.defects_btn.isChecked():
            self.defects_btn.setChecked(True)

    def _surface_overlay_show_all(self) -> bool:
        checkbox = getattr(self, "surface_show_all_checkbox", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def _surface_overlay_mode_changed(self, _checked: bool) -> None:
        self._refresh_surface_overlay()

    def _refresh_surface_overlay_selection(self, table_row: int | None = None) -> None:
        if table_row is None:
            table = getattr(self, "defects_table", None)
            table_row = table.currentRow() if table is not None else -1
        overlay_index = overlay_index_for_table_row(
            self.current_surface_boxes,
            int(table_row),
            self.surface_repair_signatures,
            show_all=self._surface_overlay_show_all(),
        )
        self.image_view.set_overlay_selection(overlay_index)

    def _refresh_surface_overlay(self) -> None:
        visible = visible_surface_boxes(
            self.current_surface_boxes,
            self.surface_repair_signatures,
            show_all=self._surface_overlay_show_all(),
        )
        self.image_view.set_overlay_boxes(visible)
        self._refresh_surface_overlay_selection()
        info = getattr(self, "defects_feedback_info", None)
        if info is not None:
            total = len(self.current_surface_boxes)
            selected = sum(
                1 for box in self.current_surface_boxes
                if surface_candidate_signature(box) in self.surface_repair_signatures
            )
            mode_text = "все найденные" if self._surface_overlay_show_all() else "только выбранные для лечения"
            info.setText(
                f"Кандидатов найдено: {total} · выбрано для лечения: {selected} · на карте: {len(visible)} ({mode_text}). "
                "«Контекст» — качество кандидата по форме и окружению; «Surface AI v2» — локальный пиксельный голос; "
                "«Meta-контекст» — независимая проверка геометрии/окружения; «Итог» требует согласия этих сигналов. "
                "Проценты моделей — внутренние оценки, а не гарантированная вероятность. "
                "Ваши метки сохраняются локально для проверки/будущего обучения. D — дефект, N — естественная деталь, "
                "U — не уверен, Delete — убрать метку. Surface никогда не лечит автоматически. Колонка «Лечить» управляет и локальным восстановлением, и картой; галочку ставите только вы."
            )

    def _set_all_surface_repair(self, enabled: bool) -> None:
        self._surface_repair_seeded = True
        if enabled:
            self.surface_repair_signatures = {surface_candidate_signature(box) for box in self.current_surface_boxes}
        else:
            self.surface_repair_signatures.clear()
        self._updating_surface_repair_choices = True
        try:
            for row in range(self.defects_table.rowCount()):
                item = self.defects_table.item(row, 0)
                if item is not None:
                    item.setCheckState(Qt.Checked if enabled else Qt.Unchecked)
        finally:
            self._updating_surface_repair_choices = False
        self._sync_surface_repair_candidates()
        self._refresh_surface_overlay()
        if isinstance(self.current_metrics, dict):
            self._show_plan(self.current_metrics)
        self._update_correction_controls(refresh_preview=True)
        self.status_label.setText(
            f"Для лечения выбраны все кандидаты: {len(self.surface_repair_signatures)}" if enabled
            else "Все цели лечения дефектов поверхности сняты"
        )

    def _surface_validation_item(self) -> dict | None:
        for item in self.current_validation_items:
            if isinstance(item, dict) and str(item.get("action_key", "")) == "surface_defects":
                return item
        return None

    def _sync_surface_repair_candidates(self) -> None:
        item = self._surface_validation_item()
        if item is None:
            return
        selected = [
            dict(box) for box in self.current_surface_boxes
            if surface_candidate_signature(box) in self.surface_repair_signatures
        ]
        item["source_candidates"] = selected
        item["candidate"] = "bounded_surface_heal"
        item["preview_available"] = bool(selected)
        item["accepted"] = False
        item["auto_eligible"] = False
        item["message"] = (
            f"Ручная локальная коррекция применится только к выбранным вами кандидатам: {len(selected)}. "
            "Проверяйте волосы, швы и естественные контуры."
            if selected else
            "Автолечение отключено. Выберите кандидаты в колонке «Лечить» на вкладке «Дефекты»."
        )
        # The overall Surface action follows the user's explicit candidate choice.
        # AI/Validator never tick it on their own.
        action_item = self._fix_list_item_for_key("surface_defects")
        if action_item is not None:
            self._updating_correction_choices = True
            try:
                action_item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
            finally:
                self._updating_correction_choices = False

    def _on_surface_repair_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_surface_repair_choices or item.column() != 0:
            return
        signature = str(item.data(Qt.UserRole) or "")
        if not signature:
            return
        self._surface_repair_seeded = True
        if item.checkState() == Qt.Checked:
            self.surface_repair_signatures.add(signature)
        else:
            self.surface_repair_signatures.discard(signature)
        self._sync_surface_repair_candidates()
        self._refresh_surface_overlay()
        if isinstance(self.current_metrics, dict):
            self._show_plan(self.current_metrics)
        self._update_correction_controls(refresh_preview=True)

    @staticmethod
    def _feedback_label_text(label: str) -> str:
        return {
            "defect": "Дефект",
            "natural_detail": "Естественная деталь",
            "uncertain": "Не уверен",
            "": "—",
        }.get(label, label or "—")

    def _surface_boxes_with_feedback(self, metrics) -> list[dict]:
        refined = metrics.get("surface_refinement") if metrics else None
        surface = metrics.get("surface_defects") if metrics else None
        boxes: list[dict] = []
        if refined is not None and isinstance(refined.raw_value, dict):
            raw_boxes = refined.raw_value.get("refined_boxes_norm", [])
            if isinstance(raw_boxes, list):
                boxes = [dict(box) for box in raw_boxes if isinstance(box, dict)]
        if not boxes and surface is not None and isinstance(surface.raw_value, dict):
            raw_boxes = surface.raw_value.get("boxes_norm", [])
            if isinstance(raw_boxes, list):
                boxes = [dict(box) for box in raw_boxes if isinstance(box, dict)]
        if self.current_path is None or not boxes:
            return boxes
        try:
            with AnalysisDatabase(self.db_path) as db:
                feedback = db.load_surface_feedback(self.current_path)
        except Exception:
            feedback = {}
        for box in boxes:
            entry = feedback.get(surface_candidate_signature(box), {})
            if isinstance(entry, dict):
                box["user_label"] = str(entry.get("user_label", ""))
        return boxes

    def _refresh_surface_feedback_stats(self) -> None:
        try:
            with AnalysisDatabase(self.db_path) as db:
                stats = db.surface_feedback_stats()
        except Exception:
            stats = []
        if not stats:
            self.feedback_stats_label.setText(
                "Surface AI v2 по вашей разметке: данных пока недостаточно. Это локальная проверка, не внешний эталон точности."
            )
            self.feedback_stats_label.setToolTip("")
            return
        preferred = next((item for item in stats if item.get("model_id") == "native_surface_verifier_v2"), stats[0])
        labeled = int(preferred.get("labeled", 0) or 0)
        evaluated = int(preferred.get("binary_evaluated", 0) or 0)
        agreement = preferred.get("agreement_pct")
        high_wrong = int(preferred.get("high_confidence_wrong", 0) or 0)
        agreement_text = "—" if agreement is None else f"{float(agreement):.0f}%"
        self.feedback_stats_label.setText(
            f"Surface AI v2 по вашей разметке: совпадение {agreement_text} · уверенных ошибок {high_wrong} · меток {labeled}"
        )
        self.feedback_stats_label.setToolTip(
            f"Сравнимых D/N-меток: {evaluated}. Считается сырой ответ Surface AI v2 до Context Verifier; это не внешний экспертный эталон точности."
        )

    def _show_defects(self, metrics) -> None:
        self.current_surface_boxes = self._surface_boxes_with_feedback(metrics)
        self.defects_table.setRowCount(len(self.current_surface_boxes))
        if not self.current_surface_boxes:
            self.image_view.set_overlay_selection(None)
        # Surface v9 is manual-only: AI/Context may rank and label candidates,
        # but never seed the repair checkboxes. Persisted user D labels are the
        # only automatic restoration of an earlier *human* choice for this photo.
        if not self._surface_repair_seeded:
            self.surface_repair_signatures = {
                surface_candidate_signature(box) for box in self.current_surface_boxes
                if str(box.get("user_label", "")) == "defect"
            }
            self._surface_repair_seeded = True
        verdict_labels = {
            "defect": "Дефект", "natural_detail": "Естественная деталь",
            "uncertain": "Неоднозначно", "unprocessed": "Не проверено", "": "—",
        }
        kind_labels = {"line": "Линия", "spot": "Пыль/точка", "irregular": "Нерегулярный"}
        self._updating_surface_repair_choices = True
        try:
            for row, box in enumerate(self.current_surface_boxes):
                final_label = str(box.get("verification_label", box.get("ai_label", "unprocessed")))
                raw_ai_label = str(box.get("ai_raw_label", "unprocessed"))
                signature = surface_candidate_signature(box)
                repair_item = QTableWidgetItem("")
                repair_item.setFlags((repair_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                repair_item.setCheckState(Qt.Checked if signature in self.surface_repair_signatures else Qt.Unchecked)
                repair_item.setData(Qt.UserRole, signature)
                if str(box.get("detection_branch", "primary")) == "low_contrast_hough":
                    repair_item.setToolTip(
                        "Слабоконтрастная линия найдена чувствительным детектором Surface v9. AI + Context Meta помогают оценить "
                        "кандидата, но лечение никогда не включается автоматически. Поставьте галочку «Лечить» сами, если это действительно царапина/трещина."
                    )
                else:
                    repair_item.setToolTip("Отметьте, если этот конкретный кандидат нужно лечить в коррекции «Дефекты поверхности».")
                self.defects_table.setItem(row, 0, repair_item)
                try:
                    raw_ai_conf = float(box.get("ai_raw_confidence", 0.0) or 0.0)
                    final_conf = float(box.get("verification_confidence", box.get("ai_confidence", 0.0)) or 0.0)
                    quality = float(box.get("candidate_quality", 0.0) or 0.0)
                    strength = float(box.get("strength", 0.0) or 0.0)
                    meta_probability = float(box.get("context_meta_probability", 0.5) or 0.0)
                    meta_threshold = float(box.get("context_meta_threshold", 0.35) or 0.35)
                    x, y = float(box.get("x", 0.0)), float(box.get("y", 0.0))
                    w, h = float(box.get("w", 0.0)), float(box.get("h", 0.0))
                except (TypeError, ValueError):
                    raw_ai_conf, final_conf, quality, strength = 0.0, 0.0, 0.0, 0.0
                    meta_probability, meta_threshold = 0.5, 0.35
                    x, y, w, h = 0.0, 0.0, 0.0, 0.0
                raw_ai_text = verdict_labels.get(raw_ai_label, raw_ai_label)
                if raw_ai_label not in {"", "unprocessed"}:
                    raw_ai_text += f" {raw_ai_conf * 100:.0f}%"
                final_text = verdict_labels.get(final_label, final_label)
                if final_label not in {"", "unprocessed"}:
                    final_text += f" {final_conf * 100:.0f}%"
                meta_available = bool(box.get("context_meta_model_id"))
                meta_text = "—" if not meta_available else f"{meta_probability * 100:.0f}% / {meta_threshold * 100:.0f}%"
                values = (
                    str(row + 1),
                    f"{quality * 100:.0f}%",
                    raw_ai_text,
                    meta_text,
                    final_text,
                    self._feedback_label_text(str(box.get("user_label", ""))),
                    f"{strength * 100:.0f}%",
                    kind_labels.get(str(box.get("candidate_kind", "")), str(box.get("candidate_kind", "")) or "—"),
                    "Светлый" if str(box.get("polarity", "bright")) == "bright" else "Тёмный",
                    f"x={x:.3f}, y={y:.3f}, {w:.3f}×{h:.3f}",
                )
                for offset, value in enumerate(values, start=1):
                    cell = QTableWidgetItem(value)
                    if offset in (1, 2, 3, 4, 5, 7, 8, 9):
                        cell.setTextAlignment(Qt.AlignCenter)
                    if offset == 2:
                        cell.setToolTip(
                            "Качество кандидата по форме, локальному контрасту, сходству окружения и риску естественной текстуры. "
                            "Это не вероятность дефекта."
                        )
                    elif offset == 3:
                        cell.setToolTip(
                            "Локальный пиксельный Surface AI v2. Значение рядом с вердиктом — внутренняя уверенность модели, "
                            "не гарантированная вероятность дефекта и не самостоятельное разрешение на лечение."
                        )
                    elif offset == 4:
                        cell.setToolTip(
                            f"Context Meta Verifier: внутренний score {meta_probability * 100:.1f}% при пороге поддержки "
                            f"{meta_threshold * 100:.1f}%. Это не калиброванная вероятность: модель оценивает форму, "
                            "локальный контраст, риск текстуры и повторяемость структуры."
                        )
                    elif offset == 5:
                        detail = str(box.get("verification_reason", "Итог объединяет пиксельную модель и контекстные признаки."))
                        cell.setToolTip(
                            detail + " Итог — ансамблевый технический вердикт, а не математическая вероятность ошибки."
                        )
                        if final_label == "defect":
                            cell.setBackground(QBrush(QColor(255, 220, 220)))
                        elif final_label == "natural_detail":
                            cell.setBackground(QBrush(QColor(224, 244, 231)))
                        elif final_label == "uncertain":
                            cell.setBackground(QBrush(QColor(255, 244, 204)))
                    elif offset == 6:
                        user_label = str(box.get("user_label", ""))
                        if user_label == "defect":
                            cell.setBackground(QBrush(QColor(255, 220, 220)))
                        elif user_label == "natural_detail":
                            cell.setBackground(QBrush(QColor(224, 244, 231)))
                        elif user_label == "uncertain":
                            cell.setBackground(QBrush(QColor(255, 244, 204)))
                    self.defects_table.setItem(row, offset, cell)
        finally:
            self._updating_surface_repair_choices = False
        self._sync_surface_repair_candidates()
        self._refresh_surface_overlay()
        self._surface_feedback_selection_changed()
        self._refresh_surface_feedback_stats()

    def _export_surface_feedback(self) -> None:
        target = QFileDialog.getExistingDirectory(self, "Папка для набора данных разметки")
        if not target:
            return
        self._begin_blocking_activity("Экспорт ручной разметки дефектов…")
        try:
            summary = export_surface_feedback_dataset(self.db_path, target)
        except Exception as exc:
            self._end_blocking_activity("экспорт разметки завершился ошибкой", success=False)
            QMessageBox.critical(self, "Photo Doctor", f"Не удалось экспортировать разметку: {exc}")
            return
        QMessageBox.information(
            self,
            "Photo Doctor",
            f"Экспортировано фрагментов: {summary.exported}.\n"
            f"Пропущено: удалённых {summary.skipped_missing}, изменённых {summary.skipped_changed}, "
            f"некорректных {summary.skipped_invalid}.\n"
            f"Набор данных: {summary.output_dir}",
        )
        self._end_blocking_activity(f"экспортировано размеченных фрагментов: {summary.exported}")

    def _set_surface_feedback(self, user_label: str | None) -> None:
        row = self.defects_table.currentRow()
        if self.current_path is None or row < 0 or row >= len(self.current_surface_boxes):
            return
        candidate = self.current_surface_boxes[row]
        try:
            with AnalysisDatabase(self.db_path) as db:
                if user_label is None:
                    db.delete_surface_feedback(self.current_path, candidate)
                    candidate.pop("user_label", None)
                else:
                    db.save_surface_feedback(self.current_path, candidate, user_label)
                    candidate["user_label"] = user_label
                    self._surface_repair_seeded = True
                    signature = surface_candidate_signature(candidate)
                    if user_label == "defect":
                        self.surface_repair_signatures.add(signature)
                    else:
                        # Both N and U mean there is no explicit human approval to treat.
                        self.surface_repair_signatures.discard(signature)
        except Exception as exc:
            QMessageBox.warning(self, "Photo Doctor", f"Не удалось сохранить разметку: {exc}")
            return
        self._show_defects(self.current_metrics or {})
        if self.defects_table.rowCount() > 0:
            next_row = min(row + 1, self.defects_table.rowCount() - 1) if user_label is not None else min(row, self.defects_table.rowCount() - 1)
            self.defects_table.selectRow(next_row)
        self.status_label.setText("Ручная метка дефекта сохранена локально" if user_label else "Ручная метка дефекта удалена")
        self._refresh_ai_training_stats()

    def _build_batch_tab(self) -> None:
        self.batch_page = QWidget()
        page = self.batch_page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)

        self.batch_info = QLabel("Пакетный анализ ещё не запускался.")
        self.batch_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.batch_info)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Фильтр:"))
        self.batch_ai_filter = QComboBox()
        self.batch_ai_filter.addItem("Все", "all")
        self.batch_ai_filter.addItem("Требует ручной проверки", "manual")
        self.batch_ai_filter.addItem("ИИ использован", "used")
        self.batch_ai_filter.addItem("ИИ не использовался", "not_used")
        self.batch_ai_filter.addItem("Ошибка ИИ", "error")
        self.batch_ai_filter.currentIndexChanged.connect(self._apply_batch_ai_filter)
        filter_row.addWidget(self.batch_ai_filter)
        filter_row.addStretch(1)
        layout.addLayout(filter_row)

        self.batch_table = QTableWidget(0, 9)
        self.batch_table.setHorizontalHeaderLabels([
            "Файл", "Качество", "Потенциал", "Уверенность",
            "Проблемы", "Профиль", "Главное замечание", "ИИ", "Статус"
        ])
        self.batch_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_table.setSortingEnabled(True)
        self.batch_table.verticalHeader().setVisible(False)
        header = self.batch_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for col in (1, 2, 3, 4, 5, 7, 8):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.batch_table.cellDoubleClicked.connect(self._open_batch_row)
        layout.addWidget(self.batch_table, 1)
        self.tabs.addTab(page, "Папка")


    def _build_series_tab(self) -> None:
        self.series_page = QWidget()
        layout = QVBoxLayout(self.series_page)
        layout.setContentsMargins(6, 6, 6, 6)

        self.series_info = QLabel(
            "Серии появятся после анализа папки. Выбор лучшего кадра здесь только технический: "
            "выражение лица, момент и художественная ценность пока не оцениваются."
        )
        self.series_info.setWordWrap(True)
        self.series_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.series_info)

        self.series_table = QTableWidget(0, 6)
        self.series_table.setHorizontalHeaderLabels([
            "Серия", "Кадров", "Период", "Лучший технически", "Сходство", "Уверенность"
        ])
        self.series_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.series_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.series_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.series_table.verticalHeader().setVisible(False)
        series_header = self.series_table.horizontalHeader()
        for col in (0, 1, 2, 4, 5):
            series_header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        series_header.setSectionResizeMode(3, QHeaderView.Stretch)
        self.series_table.itemSelectionChanged.connect(self._show_selected_series)
        self.series_table.cellDoubleClicked.connect(self._open_series_best)
        layout.addWidget(self.series_table, 1)

        self.series_members_label = QLabel("Кадры выбранной серии")
        layout.addWidget(self.series_members_label)
        self.series_members_table = QTableWidget(0, 6)
        self.series_members_table.setHorizontalHeaderLabels([
            "#", "Файл", "Относительно", "Тех. оценка", "Почему выше", "Время"
        ])
        self.series_members_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.series_members_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.series_members_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.series_members_table.verticalHeader().setVisible(False)
        member_header = self.series_members_table.horizontalHeader()
        for col in (0, 2, 3, 5):
            member_header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        member_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        member_header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.series_members_table.cellDoubleClicked.connect(self._open_series_member)
        layout.addWidget(self.series_members_table, 1)
        self.tabs.addTab(self.series_page, "Серии")

    def _clear_series_results(self, message: str | None = None) -> None:
        self.current_series_summary = None
        self.series_table.setRowCount(0)
        self.series_members_table.setRowCount(0)
        self.series_members_label.setText("Кадры выбранной серии")
        if message:
            self.series_info.setText(message)

    def _start_series_analysis(self, summary) -> None:
        files = [Path(item.path) for item in summary.items if item.status in {"ok", "cached"}]
        self._clear_series_results(
            "Группировка серий: время съёмки + визуальное сходство + относительная техническая оценка. "
            "Художественный и эмоциональный выбор не выполняется."
        )
        if len(files) < 2 or self.batch_root is None:
            self.series_info.setText("Для поиска серий нужно минимум два успешно проанализированных кадра.")
            return
        if self.series_worker is not None:
            self.series_worker.cancel()
        root_key = str(self.batch_root.resolve())
        self.series_worker = SeriesWorker(files, self.db_path, self.batch_precision, root_key)
        self.series_worker.progress.connect(self.on_series_progress)
        self.series_worker.finished_summary.connect(self.on_series_finished)
        self.series_worker.failed.connect(self.on_series_failed)
        self.progress.setRange(0, 100)
        self.progress.setFormat("%p%")
        self.progress.setValue(0)
        self.status_label.setText(f"Поиск серий: 0 из {len(files)} (0%)")
        self.series_worker.start()

    def on_series_progress(self, i: int, total: int, path: str) -> None:
        if total <= 0:
            return
        percent = round(i * 100 / total)
        self.progress.setValue(percent)
        self.status_label.setText(f"Поиск серий: {i} из {total} ({percent}%): {Path(path).name}")

    def on_series_failed(self, message: str, root_key: str) -> None:
        failed_worker = self.sender()
        if failed_worker is self.series_worker:
            self.series_worker = None
        current_root = str(self.batch_root.resolve()) if self.batch_root is not None else ""
        if root_key != current_root:
            return
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Ошибка")
        self.status_label.setText("Ошибка поиска серий")
        QMessageBox.critical(self, "Ошибка поиска серий", message)

    def on_series_finished(self, summary, root_key: str) -> None:
        finished_worker = self.sender()
        if finished_worker is self.series_worker:
            self.series_worker = None
        current_root = str(self.batch_root.resolve()) if self.batch_root is not None else ""
        if root_key != current_root or getattr(summary, "cancelled", False):
            return
        self.current_series_summary = summary
        self._populate_series_results(summary)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.progress.setFormat("Готово")
        self.status_label.setText(
            f"Готово: серии найдены — {summary.series_count}; кадров в сериях {summary.grouped_photos}; "
            f"одиночных {summary.singleton_photos}"
        )

    @staticmethod
    def _format_series_span(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        if seconds < 60.0:
            return f"{seconds:.0f} с"
        minutes, sec = divmod(round(seconds), 60)
        return f"{minutes} мин {sec:02d} с"

    def _populate_series_results(self, summary) -> None:
        self.series_table.setSortingEnabled(False)
        self.series_table.setRowCount(len(summary.groups))
        for row, group in enumerate(summary.groups):
            series_item = QTableWidgetItem(group.series_id)
            series_item.setData(Qt.UserRole, row)
            self.series_table.setItem(row, 0, series_item)
            count_item = NumericTableWidgetItem(str(len(group.members)), len(group.members))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.series_table.setItem(row, 1, count_item)
            self.series_table.setItem(row, 2, QTableWidgetItem(self._format_series_span(group.time_span_seconds)))
            best_item = QTableWidgetItem(group.best_path.name)
            best_item.setData(Qt.UserRole, str(group.best_path))
            best_item.setToolTip(str(group.best_path) + "\n" + group.selection_note)
            self.series_table.setItem(row, 3, best_item)
            similarity = NumericTableWidgetItem(f"{group.similarity_mean * 100:.0f}%", group.similarity_mean)
            similarity.setTextAlignment(Qt.AlignCenter)
            self.series_table.setItem(row, 4, similarity)
            confidence = NumericTableWidgetItem(f"{group.confidence * 100:.0f}%", group.confidence)
            confidence.setTextAlignment(Qt.AlignCenter)
            confidence.setToolTip(group.selection_note)
            self.series_table.setItem(row, 5, confidence)
        self.series_table.setSortingEnabled(True)
        self.series_info.setText(
            f"Найдено серий: {summary.series_count}; кадров в сериях: {summary.grouped_photos}; "
            f"одиночных: {summary.singleton_photos}. Лучший кадр выбирается только по техническим признакам."
        )
        if summary.groups:
            self.series_table.selectRow(0)
        else:
            self.series_members_table.setRowCount(0)
            self.series_members_label.setText("Серии из похожих соседних кадров не найдены")

    def _selected_series(self):
        if self.current_series_summary is None:
            return None
        row = self.series_table.currentRow()
        if row < 0:
            return None
        item = self.series_table.item(row, 0)
        if item is None:
            return None
        try:
            index = int(item.data(Qt.UserRole))
        except (TypeError, ValueError):
            return None
        groups = self.current_series_summary.groups
        return groups[index] if 0 <= index < len(groups) else None

    def _show_selected_series(self) -> None:
        group = self._selected_series()
        if group is None:
            self.series_members_table.setRowCount(0)
            return
        self.series_members_label.setText(
            f"{group.series_id}: {len(group.members)} кадров. {group.selection_note}"
        )
        self.series_members_table.setSortingEnabled(False)
        self.series_members_table.setRowCount(len(group.members))
        for row, member in enumerate(group.members):
            rank_item = NumericTableWidgetItem(str(member.rank), member.rank)
            rank_item.setTextAlignment(Qt.AlignCenter)
            self.series_members_table.setItem(row, 0, rank_item)
            file_item = QTableWidgetItem(member.path.name)
            file_item.setData(Qt.UserRole, str(member.path))
            file_item.setToolTip(str(member.path))
            if member.rank == 1:
                file_item.setText("★ " + member.path.name)
                file_item.setBackground(QBrush(QColor(224, 244, 231)))
            self.series_members_table.setItem(row, 1, file_item)
            relative = NumericTableWidgetItem(f"{member.relative_score:.0f}", member.relative_score)
            relative.setTextAlignment(Qt.AlignCenter)
            self.series_members_table.setItem(row, 2, relative)
            technical = NumericTableWidgetItem(f"{member.technical_score:.1f}", member.technical_score)
            technical.setTextAlignment(Qt.AlignCenter)
            self.series_members_table.setItem(row, 3, technical)
            reason = "; ".join(member.reasons) if member.rank == 1 else "—"
            reason_item = QTableWidgetItem(reason)
            reason_item.setToolTip(reason)
            self.series_members_table.setItem(row, 4, reason_item)
            time_suffix = "EXIF" if member.time_source == "exif" else "файл"
            self.series_members_table.setItem(
                row, 5, QTableWidgetItem(member.capture_time.strftime("%H:%M:%S") + f" · {time_suffix}")
            )
        self.series_members_table.setSortingEnabled(True)

    def _open_series_best(self, row: int, column: int) -> None:
        item = self.series_table.item(row, 3)
        if item is None:
            return
        path_text = item.data(Qt.UserRole)
        if path_text:
            path = Path(str(path_text))
            if path.is_file():
                self._open_path(path)

    def _open_series_member(self, row: int, column: int) -> None:
        item = self.series_members_table.item(row, 1)
        if item is None:
            return
        path_text = item.data(Qt.UserRole)
        if path_text:
            path = Path(str(path_text))
            if path.is_file():
                self._open_path(path)


    def _build_ai_tab(self) -> None:
        # Простой ИИ-экран — плоский и прокручиваемый. Раньше пять QGroupBox
        # подряд тратили заметную часть высоты на рамки, заголовочные отступы
        # и внутренние поля, из-за чего текст визуально обрезался.
        self.ai_page = QScrollArea()
        self.ai_page.setWidgetResizable(True)
        self.ai_page.setFrameShape(QFrame.Shape.NoFrame)
        self.ai_page.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ai_page.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        self.ai_page.setWidget(content)

        def add_section_title(title_text: str) -> None:
            label = QLabel(f"<b>{title_text}</b>")
            label.setStyleSheet("font-size: 11pt; margin-top: 3px;")
            layout.addWidget(label)

        def add_separator() -> None:
            line = QFrame()
            line.setFrameShape(QFrame.Shape.HLine)
            line.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(line)

        add_section_title("Статус ИИ")
        self.ai_state_label = QLabel("<b>ИИ: анализ ещё не запускался</b>")
        self.ai_state_label.setWordWrap(True)
        self.ai_state_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.ai_state_label)
        self.ai_privacy_summary = QLabel("Локальный ИИ. Фото никуда не отправляется.")
        self.ai_privacy_summary.setWordWrap(True)
        self.ai_privacy_summary.setStyleSheet("color: #666;")
        layout.addWidget(self.ai_privacy_summary)

        add_separator()
        add_section_title("Что ИИ сделал на этом фото")
        self.ai_photo_summary = QLabel("Анализ ещё не запускался.")
        self.ai_photo_summary.setWordWrap(True)
        self.ai_photo_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.ai_photo_summary)

        add_separator()
        add_section_title("Доверие к ИИ")
        self.ai_trust_summary = QLabel("<b>Доверие к ИИ: НЕТ ДАННЫХ</b>")
        self.ai_trust_summary.setWordWrap(True)
        self.ai_trust_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.ai_trust_summary)

        add_separator()
        add_section_title("Модели и ресурсы")
        self.ai_models_summary = QLabel("<b>Встроенная модель:</b> Surface AI v2 + Context Meta v2 — проверка…")
        self.ai_resource_summary = QLabel("<b>Режим ресурсов:</b> —")
        self.ai_runtime_summary = QLabel("<b>ONNX Runtime:</b> —")
        for label in (self.ai_models_summary, self.ai_resource_summary, self.ai_runtime_summary):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(label)

        add_separator()
        add_section_title("Обучение")
        self.ai_training_summary = QLabel("Обучающие данные: проверка…")
        self.ai_training_summary.setWordWrap(True)
        self.ai_training_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.ai_training_summary)
        self.ai_training_detail = QLabel(
            "Решения хранятся локально; базовые модели не переобучаются автоматически после каждого клика."
        )
        self.ai_training_detail.setWordWrap(True)
        self.ai_training_detail.setStyleSheet("color: #666;")
        layout.addWidget(self.ai_training_detail)
        training_buttons = QHBoxLayout()
        self.ai_training_export_btn = QPushButton("Подготовить пакет для обучения ИИ…")
        self.ai_training_export_btn.setToolTip(
            "Создать ZIP без личных данных для Codex: сохранённые значения ползунков и фрагменты 96×96 с ручными метками. "
            "Полные пути, имена исходных фото и целые фотографии в пакет не включаются."
        )
        self.ai_training_export_btn.clicked.connect(self._export_ai_training_package)
        self.ai_training_refresh_btn = QPushButton("Обновить счётчики")
        self.ai_training_refresh_btn.clicked.connect(self._refresh_ai_training_stats)
        training_buttons.addWidget(self.ai_training_export_btn)
        training_buttons.addWidget(self.ai_training_refresh_btn)
        training_buttons.addStretch(1)
        layout.addLayout(training_buttons)

        add_separator()
        ai_controls = QHBoxLayout()
        self.ai_import_btn = QPushButton("Настроить внешние модели…")
        self.ai_import_btn.setToolTip("Импортировать локальную ONNX-модель. Встроенный Уточнитель дефектов поверхности уже входит в Photo Doctor.")
        self.ai_import_btn.clicked.connect(self._import_ai_model)
        self.ai_advanced_btn = QPushButton("Расширенно")
        self.ai_advanced_btn.setCheckable(True)
        self.ai_advanced_btn.toggled.connect(self._toggle_ai_advanced)
        ai_controls.addWidget(self.ai_import_btn)
        ai_controls.addStretch(1)
        ai_controls.addWidget(self.ai_advanced_btn)
        layout.addLayout(ai_controls)

        self.ai_advanced_tabs = QTabWidget()
        self.ai_advanced_tabs.setVisible(False)

        route_page = QWidget()
        route_layout = QVBoxLayout(route_page)
        self.ai_info = QLabel("Техническая информация маршрутизатора ИИ появится после анализа.")
        self.ai_info.setWordWrap(True)
        self.ai_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        route_layout.addWidget(self.ai_info)
        self.ai_routes_table = QTableWidget(0, 6)
        self.ai_routes_table.setHorizontalHeaderLabels([
            "Нужно", "Задача", "Приоритет", "Модель", "Состояние", "Почему"
        ])
        self.ai_routes_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ai_routes_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ai_routes_table.setWordWrap(True)
        self.ai_routes_table.verticalHeader().setVisible(False)
        route_header = self.ai_routes_table.horizontalHeader()
        for col in (0, 1, 2, 3, 4):
            route_header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        route_header.setSectionResizeMode(5, QHeaderView.Stretch)
        route_layout.addWidget(self.ai_routes_table, 1)
        self.ai_advanced_tabs.addTab(route_page, "Маршруты")

        inference_page = QWidget()
        inference_layout = QVBoxLayout(inference_page)
        self.ai_inference_table = QTableWidget(0, 9)
        self.ai_inference_table.setHorizontalHeaderLabels([
            "Задача", "Модель", "Область", "Результат", "Уверенность", "Статус", "Сверка", "Допуск", "Подробности"
        ])
        self.ai_inference_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ai_inference_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ai_inference_table.setWordWrap(True)
        self.ai_inference_table.verticalHeader().setVisible(False)
        inference_header = self.ai_inference_table.horizontalHeader()
        for col in (0, 1, 2, 4, 5, 6, 7):
            inference_header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        inference_header.setSectionResizeMode(3, QHeaderView.Stretch)
        inference_header.setSectionResizeMode(8, QHeaderView.Stretch)
        inference_layout.addWidget(self.ai_inference_table, 1)
        self.ai_advanced_tabs.addTab(inference_page, "Результаты")

        models_page = QWidget()
        models_layout = QVBoxLayout(models_page)
        self.ai_models_table = QTableWidget(0, 5)
        self.ai_models_table.setHorizontalHeaderLabels([
            "Задача", "Модель", "Версия", "Состояние", "Источник / файл"
        ])
        self.ai_models_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ai_models_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ai_models_table.verticalHeader().setVisible(False)
        model_header = self.ai_models_table.horizontalHeader()
        for col in (0, 1, 2, 3):
            model_header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        model_header.setSectionResizeMode(4, QHeaderView.Stretch)
        models_layout.addWidget(self.ai_models_table, 1)
        self.ai_advanced_tabs.addTab(models_page, "Модели")

        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        self.ai_history_label = QLabel("Согласованность с классическим анализом (не эталон и не точность модели)")
        self.ai_history_label.setWordWrap(True)
        history_layout.addWidget(self.ai_history_label)
        self.ai_history_table = QTableWidget(0, 9)
        self.ai_history_table.setHorizontalHeaderLabels([
            "Задача", "Модель", "Фото", "Сравнений", "Согласен", "Уточняет", "Противоречит", "Неуверенно", "Совместимость*"
        ])
        self.ai_history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ai_history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.ai_history_table.verticalHeader().setVisible(False)
        hist_header = self.ai_history_table.horizontalHeader()
        for col in range(9):
            hist_header.setSectionResizeMode(col, QHeaderView.ResizeToContents if col != 1 else QHeaderView.Stretch)
        history_layout.addWidget(self.ai_history_table, 1)
        self.ai_advanced_tabs.addTab(history_page, "История")

        diagnostics_page = QWidget()
        diagnostics_layout = QVBoxLayout(diagnostics_page)
        self.ai_diagnostics_info = QLabel("Диагностика среды выполнения и менеджера моделей появится после проверки состояния.")
        self.ai_diagnostics_info.setWordWrap(True)
        self.ai_diagnostics_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        diagnostics_layout.addWidget(self.ai_diagnostics_info)
        diagnostic_buttons = QHBoxLayout()
        self.ai_runtime_btn = QPushButton("Установить ONNX Runtime")
        self.ai_runtime_btn.setToolTip("Нужен только внешним ONNX-моделям. Встроенный Уточнитель дефектов поверхности работает без него.")
        self.ai_runtime_btn.clicked.connect(self._install_ai_runtime)
        self.almaz_prepare_btn = QPushButton("Подготовить ALMAZ x2")
        self.almaz_prepare_btn.setToolTip("Скачать официальный SwinIR x2, проверить SHA-256, экспортировать ONNX, проверить parity и установить модель. Доступно только в локальном .venv.")
        self.almaz_prepare_btn.clicked.connect(self._prepare_almaz_x2)
        self.ai_refresh_btn = QPushButton("Обновить диагностику")
        self.ai_refresh_btn.clicked.connect(self._refresh_ai_status)
        diagnostic_buttons.addWidget(self.ai_runtime_btn)
        diagnostic_buttons.addWidget(self.almaz_prepare_btn)
        diagnostic_buttons.addWidget(self.ai_refresh_btn)
        diagnostic_buttons.addStretch(1)
        diagnostics_layout.addLayout(diagnostic_buttons)
        self.ai_install_log = QTextBrowser()
        self.ai_install_log.setReadOnly(True)
        self.ai_install_log.setMaximumHeight(120)
        self.ai_install_log.setVisible(False)
        self.ai_install_log.setPlaceholderText("Лог установки ONNX Runtime")
        diagnostics_layout.addWidget(self.ai_install_log)
        diagnostics_layout.addStretch(1)
        self.ai_advanced_tabs.addTab(diagnostics_page, "Диагностика")

        layout.addWidget(self.ai_advanced_tabs, 1)
        layout.addStretch(1)
        self.tabs.addTab(self.ai_page, "ИИ")
        self._refresh_ai_training_stats()

    @staticmethod
    def _training_action_name(action_key: str) -> str:
        return {
            "exposure": "Средние тона",
            "white_balance": "Баланс белого",
            "auto_tone_color": "Автотон/контраст/цвет",
            "contrast": "Контраст",
            "sharpness": "Резкость",
            "noise": "Шум",
            "jpeg_artifacts": "JPEG",
            "edge_artifacts": "Ореолы",
            "posterization": "Banding",
            "red_eye": "Красные глаза",
            "surface_defects": "Дефекты поверхности",
        }.get(str(action_key), str(action_key) or "—")

    def _refresh_ai_training_stats(self) -> None:
        try:
            stats = collect_training_data_stats(self.db_path)
        except Exception as exc:
            if hasattr(self, "ai_training_summary"):
                self.ai_training_summary.setText(f"Не удалось прочитать обучающие данные: {exc}")
            return
        parameter_parts = [
            f"{self._training_action_name(key)} {count}/30 · фото {stats.parameter_source_groups_by_action.get(key, 0)}"
            for key, count in stats.parameter_by_action.items()
        ]
        if not parameter_parts:
            parameter_text = "ползунки: 0 примеров"
        else:
            parameter_text = "ползунки: " + " · ".join(parameter_parts)
        surface_text = (
            f"дефекты D {stats.surface_defect} · N {stats.surface_natural} · U {stats.surface_uncertain} "
            f"· фото {stats.surface_source_groups}"
        )
        ready = []
        if stats.parameter_ready_actions:
            ready.append("первый эксперимент по ползункам: " + ", ".join(
                self._training_action_name(key) for key in stats.parameter_ready_actions
            ))
        if stats.surface_ready:
            ready.append("Уточнитель дефектов поверхности: минимальный набор для первого эксперимента собран")
        readiness = "; ".join(ready) if ready else "данных пока недостаточно для первого реального обучения"
        self.ai_training_summary.setText(
            f"<b>Обучающие данные:</b> сохранённых решений {stats.meaningful_correction_decisions} · "
            f"примеров силы {stats.parameter_strength_samples}<br>{parameter_text}<br>{surface_text}<br>"
            f"<b>{readiness}</b>"
        )

    def _export_ai_training_package(self) -> None:
        self._refresh_ai_training_stats()
        stamp = time.strftime("%Y-%m-%d")
        suggested = str(Path.home() / f"PhotoDoctor_AI_Training_Data_{stamp}.zip")
        filename, _ = QFileDialog.getSaveFileName(
            self, "Подготовить пакет для обучения ИИ", suggested, "Архив ZIP (*.zip)"
        )
        if not filename:
            return
        self._begin_blocking_activity("Подготовка пакета обучающих данных ИИ…")
        try:
            summary = export_training_package(self.db_path, filename)
        except Exception as exc:
            self._end_blocking_activity("пакет обучающих данных не создан", success=False)
            QMessageBox.critical(self, "Photo Doctor", f"Не удалось подготовить пакет для обучения: {exc}")
            return
        stats = summary.stats
        self._refresh_ai_training_stats()
        QMessageBox.information(
            self,
            "Photo Doctor",
            f"Пакет для обучения создан:\n{summary.output_zip}\n\n"
            f"Решения по параметрам: {summary.parameter_exported} строк, "
            f"реальных значений ползунков: {stats.parameter_strength_samples}.\n"
            f"Фрагменты дефектов с ручными метками: {summary.surface_exported}; "
            f"пропущено изменённых/недоступных: "
            f"{summary.surface_skipped_changed + summary.surface_skipped_missing}.\n\n"
            "Полные пути, имена исходных фото и целые фотографии в ZIP не включены."
        )
        self._end_blocking_activity(f"пакет обучения ИИ подготовлен: {summary.output_zip.name}")

    def _toggle_ai_advanced(self, checked: bool) -> None:
        self.ai_advanced_tabs.setVisible(bool(checked))
        self.ai_advanced_btn.setText("Скрыть подробности" if checked else "Расширенно")

    def _build_technical_tab(self) -> None:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(6, 6, 6, 6)
        self.technical_info = QLabel("Параметры пространственного анализа появятся после открытия фотографии.")
        self.technical_info.setWordWrap(True)
        self.technical_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.technical_info)
        self.technical = QTableWidget(0, 6)
        self.technical.setHorizontalHeaderLabels(["Метрика", "Исходное значение", "Оценка", "Уверенность", "Масштаб", "Область"])
        header = self.technical.horizontalHeader()
        # Do not use ResizeToContents here: the technical table can contain dozens
        # of metrics and very long values. Measuring every cell on first paint is
        # exactly what made the tab slow to open and scroll.
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in (2, 3, 4, 5):
            header.setSectionResizeMode(col, QHeaderView.Fixed)
        self.technical.setColumnWidth(0, 210)
        self.technical.setColumnWidth(2, 86)
        self.technical.setColumnWidth(3, 104)
        self.technical.setColumnWidth(4, 118)
        self.technical.setColumnWidth(5, 132)
        self.technical.setWordWrap(False)
        self.technical.setTextElideMode(Qt.ElideRight)
        self.technical.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.technical.verticalHeader().setVisible(False)
        self.technical.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.technical.verticalHeader().setDefaultSectionSize(30)
        layout.addWidget(self.technical, 1)
        self.tabs.addTab(page, "Технические данные")


    def _toggle_defects(self, checked: bool) -> None:
        if checked and self.sharpness_btn.isChecked():
            self.sharpness_btn.setChecked(False)
        if checked and self.tone_btn.isChecked():
            self.tone_btn.setChecked(False)
        if checked and self.contrast_btn.isChecked():
            self.contrast_btn.setChecked(False)
        self.image_view.set_overlay_enabled(checked)

    def _toggle_sharpness(self, checked: bool) -> None:
        if checked and self.defects_btn.isChecked():
            self.defects_btn.setChecked(False)
        if checked and self.tone_btn.isChecked():
            self.tone_btn.setChecked(False)
        if checked and self.contrast_btn.isChecked():
            self.contrast_btn.setChecked(False)
        self.image_view.set_sharpness_enabled(checked)

    def _toggle_tone(self, checked: bool) -> None:
        if checked and self.defects_btn.isChecked():
            self.defects_btn.setChecked(False)
        if checked and self.sharpness_btn.isChecked():
            self.sharpness_btn.setChecked(False)
        if checked and self.contrast_btn.isChecked():
            self.contrast_btn.setChecked(False)
        self.image_view.set_tone_enabled(checked)

    def _toggle_contrast(self, checked: bool) -> None:
        if checked and self.defects_btn.isChecked():
            self.defects_btn.setChecked(False)
        if checked and self.sharpness_btn.isChecked():
            self.sharpness_btn.setChecked(False)
        if checked and self.tone_btn.isChecked():
            self.tone_btn.setChecked(False)
        self.image_view.set_contrast_enabled(checked)

    def _invalidate_preview_cache(self) -> None:
        self._preview_recipe_revision += 1
        self._preview_cache_rgb = None
        self._preview_cache_key = None

    def _mark_preview_stale_after_recipe_change(self) -> None:
        """Invalidate a rendered preview without silently recalculating it.

        A slider/checkbox/region edit is intentionally cheap: it returns the
        viewer to the source image and asks for an explicit preview refresh.
        Heavy ALMAZ inference must therefore only begin from the preview button.
        """
        had_rendered_preview = bool(
            getattr(self, "_preview_cache_rgb", None) is not None
            or (hasattr(self, "preview_btn") and self.preview_btn.isChecked())
        )
        self._invalidate_preview_cache()
        if not hasattr(self, "preview_btn"):
            return
        selected = self._selected_correction_keys()
        if not selected:
            self._preview_is_stale = False
            if self.preview_btn.isChecked():
                self.preview_btn.blockSignals(True)
                self.preview_btn.setChecked(False)
                self.preview_btn.blockSignals(False)
            if had_rendered_preview and self.current_rgb is not None:
                self.image_view.set_rgb(self.current_rgb)
                self.status_label.setText("Все исправления сняты. Показан исходник.")
            self.preview_btn.setText("Предпросмотр")
            return
        if had_rendered_preview:
            self._preview_is_stale = True
            if self.preview_btn.isChecked():
                self.preview_btn.blockSignals(True)
                self.preview_btn.setChecked(False)
                self.preview_btn.blockSignals(False)
            if self.current_rgb is not None:
                self.image_view.set_rgb(self.current_rgb)
            self.preview_btn.setText("Обновить предпросмотр")
            self.status_label.setText(
                "Параметры исправлений изменены. Нажмите «Обновить предпросмотр» для перерасчёта."
            )

    def _preview_key(self, selected: set[str] | list[str] | tuple[str, ...]) -> tuple[object, ...]:
        # Revision remains a cheap explicit invalidation mechanism, but cache
        # correctness no longer relies on every UI call site remembering to bump
        # it.  The effective strengths/regions/candidates are part of the key, so
        # e.g. WB 35% and WB 60% can never reuse the same preview.
        recipe_signature = build_preview_recipe_signature(
            selected,
            self.correction_strengths,
            self.correction_regions,
            self.current_validation_items,
        )
        return (
            int(self._preview_recipe_revision),
            str(self.current_source_quick_hash or ""),
            recipe_signature,
        )

    def _get_or_build_preview(self, selected: set[str] | list[str] | tuple[str, ...]):
        key = self._preview_key(selected)
        if self._preview_cache_rgb is not None and self._preview_cache_key == key:
            return self._preview_cache_rgb
        preview = apply_selected_preview(
            self.current_rgb, self.current_validation_items, selected, self.correction_strengths, self.correction_regions
        )
        self._preview_cache_rgb = preview
        self._preview_cache_key = key
        return preview

    def _selection_requires_almaz_worker(self, selected: set[str]) -> bool:
        selected_keys = {str(key) for key in selected}
        if "super_resolution" in selected_keys:
            return True
        for item in self.current_validation_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("action_key", "")) not in selected_keys:
                continue
            candidate = str(item.get("candidate", ""))
            if candidate.startswith("almaz_ai_") or candidate.startswith("almaz_x2_"):
                return True
        return False

    def _start_almaz_preview(self, selected: set[str]) -> None:
        if self.current_rgb is None:
            return
        key = self._preview_key(selected)
        if self._preview_cache_rgb is not None and self._preview_cache_key == key:
            self.image_view.set_rgb(self._preview_cache_rgb)
            return
        if self.preview_worker is not None:
            self.status_label.setText("⏳ ALMAZ уже строит предпросмотр…")
            return
        self.preview_started_at = time.monotonic()
        self.preview_stage_text = "ALMAZ: подготавливаю предпросмотр"
        self.preview_timer.start()
        self._begin_blocking_activity(self.preview_stage_text + "…")
        self.preview_btn.setEnabled(False)
        self.save_copy_btn.setEnabled(False)
        if hasattr(self, "open_file_btn"):
            self.open_file_btn.setEnabled(False)
        if hasattr(self, "open_folder_action"):
            self.open_folder_action.setEnabled(False)
        worker = CorrectionPreviewWorker(
            self.current_rgb.copy(),
            list(self.current_validation_items),
            set(selected),
            dict(self.correction_strengths),
            dict(self.correction_regions),
            key,
        )
        self.preview_worker = worker
        worker.stage.connect(self._almaz_preview_stage_changed)
        worker.completed.connect(self._almaz_preview_completed)
        worker.failed.connect(self._almaz_preview_failed)
        worker.finished.connect(self._almaz_preview_thread_finished)
        worker.start()

    def _almaz_preview_completed(self, preview: object, key: object, selected_count: int) -> None:
        self.preview_timer.stop()
        elapsed = 0.0 if self.preview_started_at is None else max(0.0, time.monotonic() - self.preview_started_at)
        self.preview_started_at = None
        current_selected = self._selected_correction_keys()
        current_key = self._preview_key(current_selected)
        if key != current_key or not isinstance(preview, np.ndarray):
            if self.preview_btn.isChecked():
                self.preview_btn.blockSignals(True)
                self.preview_btn.setChecked(False)
                self.preview_btn.blockSignals(False)
            if self.current_rgb is not None:
                self.image_view.set_rgb(self.current_rgb)
            if current_selected:
                self._preview_is_stale = True
                self.preview_btn.setText("Обновить предпросмотр")
                self._end_blocking_activity(
                    "ALMAZ закончил старый расчёт, но настройки уже изменились. Нажмите «Обновить предпросмотр»."
                )
            else:
                self._preview_is_stale = False
                self.preview_btn.setText("Предпросмотр")
                self._end_blocking_activity("Старый ALMAZ-расчёт отброшен: все исправления сняты.")
            return
        self._preview_cache_rgb = preview
        self._preview_cache_key = current_key
        self._preview_is_stale = False
        self.preview_btn.setText("Предпросмотр")
        if self.preview_btn.isChecked():
            self.image_view.set_rgb(preview)
        manual = self._selected_manual_override_count()
        suffix = f"; ручных включений: {manual}" if manual else ""
        sender = self.sender()
        safety_adjustment = str(getattr(sender, "safety_adjustment", "") or "").strip()
        safety_suffix = f"; {safety_adjustment}" if safety_adjustment else ""
        self._end_blocking_activity(
            f"ALMAZ-предпросмотр готов за {elapsed:.1f} с: исправлений {selected_count}{suffix}{safety_suffix}; исходный файл не изменён"
        )

    def _almaz_preview_failed(self, message: str, key: object) -> None:
        self.preview_timer.stop()
        elapsed = 0.0 if self.preview_started_at is None else max(0.0, time.monotonic() - self.preview_started_at)
        self.preview_started_at = None
        self.preview_btn.blockSignals(True)
        self.preview_btn.setChecked(False)
        self.preview_btn.blockSignals(False)
        current_selected = self._selected_correction_keys()
        current_key = self._preview_key(current_selected)
        if key != current_key:
            if self.current_rgb is not None:
                self.image_view.set_rgb(self.current_rgb)
            if current_selected:
                self._preview_is_stale = True
                self.preview_btn.setText("Обновить предпросмотр")
                self._end_blocking_activity(
                    "Старый ALMAZ-расчёт завершился с ошибкой после изменения настроек; ошибка отброшена. "
                    "Нажмите «Обновить предпросмотр»."
                )
            else:
                self._preview_is_stale = False
                self.preview_btn.setText("Предпросмотр")
                self._end_blocking_activity("Старый ALMAZ-расчёт с ошибкой отброшен: все исправления сняты.")
            return
        self._end_blocking_activity(f"ALMAZ-предпросмотр не построен после {elapsed:.1f} с", success=False)
        text = f"ALMAZ не применил небезопасный вариант.\n\n{message}"
        if "Проверка безопасности ALMAZ" in str(message) or "ALMAZ safety" in str(message):
            QMessageBox.warning(self, "Photo Doctor — ALMAZ защитил исходник", text)
        else:
            QMessageBox.critical(self, "Photo Doctor", f"Не удалось построить ALMAZ-предпросмотр: {message}")

    def _almaz_preview_stage_changed(self, message: str) -> None:
        self.preview_stage_text = str(message)
        self._update_almaz_preview_elapsed()

    def _update_almaz_preview_elapsed(self) -> None:
        if self.preview_started_at is None or not self.preview_stage_text:
            return
        elapsed = max(0.0, time.monotonic() - self.preview_started_at)
        stage = self.preview_stage_text.strip()
        if not stage.startswith("ALMAZ"):
            stage = "ALMAZ: " + stage
        self.status_label.setText(f"{stage}  ·  {elapsed:.1f} с")

    def _almaz_preview_thread_finished(self) -> None:
        finished = self.sender()
        if finished is self.preview_worker:
            self.preview_worker = None
        if self.preview_worker is None and self.preview_started_at is not None:
            self.preview_timer.stop()
        if isinstance(finished, QThread):
            finished.deleteLater()
        busy = self.analysis_worker is not None or self.worker is not None
        if hasattr(self, "open_file_btn"):
            self.open_file_btn.setEnabled(not busy)
        if hasattr(self, "open_folder_action"):
            self.open_folder_action.setEnabled(not busy)
        self._update_correction_controls(refresh_preview=False)

    def _toggle_preview(self, checked: bool) -> None:
        if self.current_rgb is None:
            self.preview_btn.blockSignals(True)
            self.preview_btn.setChecked(False)
            self.preview_btn.blockSignals(False)
            return
        selected = self._selected_correction_keys()
        if checked and selected:
            key = self._preview_key(selected)
            if self._preview_cache_rgb is not None and self._preview_cache_key == key:
                self.image_view.set_rgb(self._preview_cache_rgb)
                self._preview_is_stale = False
                self.preview_btn.setText("Предпросмотр")
                self.status_label.setText("✓ Предпросмотр взят из кэша текущих настроек; сила и область входят в ключ кэша")
            elif self._selection_requires_almaz_worker(set(selected)):
                self._start_almaz_preview(set(selected))
                return
            else:
                self._begin_blocking_activity(f"Подготовка предпросмотра: исправлений {len(selected)}…")
                try:
                    preview = self._get_or_build_preview(selected)
                except Exception as exc:
                    self.preview_btn.blockSignals(True)
                    self.preview_btn.setChecked(False)
                    self.preview_btn.blockSignals(False)
                    self._end_blocking_activity("предпросмотр не построен", success=False)
                    QMessageBox.critical(self, "Photo Doctor", f"Не удалось построить предпросмотр: {exc}")
                    return
                self.image_view.set_rgb(preview)
                self._preview_is_stale = False
                self.preview_btn.setText("Предпросмотр")
                manual = self._selected_manual_override_count()
                suffix = f"; ручных включений: {manual}" if manual else ""
                self._end_blocking_activity(
                    f"предпросмотр готов: исправлений {len(selected)}{suffix}; исходный файл не изменён"
                )
        else:
            self.image_view.set_rgb(self.current_rgb)
            if self.current_path is not None:
                self.status_label.setText(f"{self.current_path.name} — исходник")
        self._update_correction_controls(refresh_preview=False)

    def _save_preview_copy(self) -> None:
        if self.current_rgb is None or self.current_path is None:
            return
        selected = self._selected_correction_keys()
        if not selected:
            return
        if self._selection_requires_almaz_worker(set(selected)):
            key = self._preview_key(selected)
            if self._preview_cache_rgb is None or self._preview_cache_key != key:
                if self.preview_worker is None:
                    if not self.preview_btn.isChecked():
                        self.preview_btn.blockSignals(True)
                        self.preview_btn.setChecked(True)
                        self.preview_btn.blockSignals(False)
                    self._start_almaz_preview(set(selected))
                else:
                    self.status_label.setText("⏳ ALMAZ ещё считает предпросмотр; сохранение станет доступно после завершения")
                return
        default_name = self.current_path.with_name(self.current_path.stem + "_PhotoDoctor.png")
        name, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Сохранить исправленную копию",
            str(default_name),
            "PNG (*.png);;JPEG, качество 95 (*.jpg *.jpeg)",
        )
        if not name:
            return
        target = Path(name)
        if not target.suffix:
            target = target.with_suffix(".jpg" if "JPEG" in selected_filter else ".png")
        self._begin_blocking_activity(f"Сохранение исправленной копии: {target.name}…")
        try:
            preview = self._get_or_build_preview(selected)
            surface_history_payload = None
            if "surface_defects" in selected:
                surface_item = self._surface_validation_item()
                surface_candidates = surface_item.get("source_candidates", []) if isinstance(surface_item, dict) else []
                if isinstance(surface_candidates, list) and surface_candidates:
                    inherited_history = read_surface_history(self.current_path)
                    surface_history_payload = merge_surface_history(
                        inherited_history, [candidate for candidate in surface_candidates if isinstance(candidate, dict)]
                    )
            saved = save_rgb_copy(
                preview, target, source_path=self.current_path, jpeg_quality=95,
                photo_doctor_surface_history=surface_history_payload,
            )
            metadata_audit = audit_metadata_preservation(self.current_path, saved)
        except ExportError as exc:
            self._end_blocking_activity("копия не сохранена", success=False)
            QMessageBox.warning(self, "Photo Doctor", str(exc))
            return
        except Exception as exc:
            self._end_blocking_activity("копия не сохранена", success=False)
            QMessageBox.critical(self, "Photo Doctor", f"Не удалось сохранить копию: {exc}")
            return

        feedback_note = ""
        if self.correction_feedback_checkbox.isChecked():
            try:
                recorded = self._record_saved_correction_feedback(saved)
                if recorded:
                    feedback_note = f" · локально записано решений: {recorded}"
            except Exception as exc:
                feedback_note = f" · не удалось записать локальную историю решений: {exc}"

        if metadata_audit.omitted_classes:
            warning = metadata_audit.warning_text()
            QMessageBox.warning(
                self,
                "Метаданные сохранены частично",
                warning + "\n\nИзображение сохранено. Поле ориентации EXIF намеренно не переносится после физического поворота пикселей.",
            )
            self._end_blocking_activity(
                f"исправленная копия сохранена ({len(selected)} исправлений); часть метаданных не перенесена: {saved}{feedback_note}"
            )
        else:
            self._end_blocking_activity(
                f"исправленная копия сохранена ({len(selected)} исправлений): {saved}{feedback_note}"
            )

    def _open_fullscreen_viewer(self) -> None:
        if not hasattr(self, "image_view") or self.image_view._pixmap is None:
            return
        viewer = FullscreenImageWindow(self.image_view, self)
        self._fullscreen_windows.append(viewer)

        def forget_window(*_args, window=viewer):
            try:
                self._fullscreen_windows.remove(window)
            except ValueError:
                pass

        viewer.destroyed.connect(forget_window)
        viewer.show()
        viewer.raise_()
        viewer.activateWindow()

    def _step_zoom(self, delta: int) -> None:
        if not hasattr(self, "image_view"):
            return
        base = self.image_view._shown_zoom_percent() if self.image_view._fit else self.image_view._zoom_percent
        self.image_view.set_zoom_percent(base + int(delta))

    def _zoom_slider_changed(self, value: int) -> None:
        if not hasattr(self, "image_view"):
            return
        self.image_view.set_zoom_percent(int(value))

    def _on_image_zoom_changed(self, percent: int, fit: bool) -> None:
        if hasattr(self, "fit_btn"):
            self.fit_btn.blockSignals(True)
            self.fit_btn.setChecked(bool(fit))
            self.fit_btn.blockSignals(False)
        if hasattr(self, "zoom_slider"):
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(max(25, min(300, int(percent))))
            self.zoom_slider.blockSignals(False)
        if hasattr(self, "zoom_value_label"):
            self.zoom_value_label.setText(("По окну · " if fit else "") + f"{int(percent)}%")

    def _set_precision_combo(self, key: str) -> None:
        key = get_precision(key).key
        for index in range(self.precision_combo.count()):
            if str(self.precision_combo.itemData(index)) == key:
                self.precision_combo.blockSignals(True)
                self.precision_combo.setCurrentIndex(index)
                self.precision_combo.blockSignals(False)
                return

    def _precision_key(self) -> str:
        data = self.precision_combo.currentData() if hasattr(self, "precision_combo") else "normal"
        return get_precision(str(data or "normal")).key

    def _precision_changed(self, _index: int = -1) -> None:
        key = self._precision_key()
        self.settings.setValue("analysis_precision", key)
        profile = get_precision(key)
        if self.current_path is None:
            self.status_label.setText(f"Точность анализа: {profile.label}")
            return
        if self.worker is not None or self.analysis_worker is not None or self.series_worker is not None:
            self.status_label.setText(f"Точность «{profile.label}» выбрана; дождитесь завершения текущей операции")
            return
        self._open_path(
            self.current_path, precision_override=key, reset_view=False,
            completion_message=f"Готово: анализ «{profile.label}» применён",
        )

    def _restore_ui_state(self) -> None:
        state = self.settings.value("main_splitter")
        if state:
            self.main_splitter.restoreState(state)
        try:
            tab_index = int(self.settings.value("active_tab", 0))
        except (TypeError, ValueError):
            tab_index = 0
        if 0 <= tab_index < self.tabs.count():
            self.tabs.setCurrentIndex(tab_index)

    def closeEvent(self, event) -> None:
        self.settings.setValue("main_splitter", self.main_splitter.saveState())
        self.settings.setValue("active_tab", self.tabs.currentIndex())
        if self.worker is not None:
            self.worker.cancel()
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            self.status_label.setText("Завершение текущего анализа перед закрытием…")
            QApplication.processEvents()
            if not self.analysis_worker.wait(5000):
                self.status_label.setText("Анализ ещё выполняется; дождитесь завершения перед закрытием")
                event.ignore()
                return
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self.status_label.setText("ALMAZ ещё строит предпросмотр; дождитесь завершения перед закрытием…")
            QApplication.processEvents()
            if not self.preview_worker.wait(5000):
                self.status_label.setText("ALMAZ всё ещё работает; закрытие отложено до завершения расчёта")
                event.ignore()
                return
        if self.almaz_release_worker is not None and self.almaz_release_worker.isRunning():
            self.status_label.setText("ALMAZ ещё проверяет/устанавливает модель; дождитесь завершения перед закрытием…")
            QApplication.processEvents()
            if not self.almaz_release_worker.wait(5000):
                self.status_label.setText("Операция с ALMAZ-моделью ещё выполняется; закрытие отложено")
                event.ignore()
                return
        if self.series_worker is not None:
            self.series_worker.cancel()
            self.series_worker.wait(2000)
        super().closeEvent(event)

    def open_file(self) -> None:
        name, _ = QFileDialog.getOpenFileName(self, "Открыть фото", "", INPUT_FILE_DIALOG_FILTER)
        if not name:
            return
        self._set_precision_combo("normal")
        self._open_path(Path(name), precision_override="normal", reset_view=True)

    def _set_single_analysis_busy(self, busy: bool, message: str = "") -> None:
        if busy:
            self.progress.setRange(0, 0)
            self.progress.setFormat("Выполняется…")
            if message:
                self.status_label.setText(message)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
            self.progress.setFormat("Готово")
        self.precision_combo.setEnabled(not busy and self.worker is None)
        if hasattr(self, "open_file_btn"):
            self.open_file_btn.setEnabled(not busy)
        if hasattr(self, "open_folder_action"):
            self.open_folder_action.setEnabled(not busy and self.worker is None)
        if busy:
            self.preview_btn.setEnabled(False)
            self.save_copy_btn.setEnabled(False)

    def _analysis_stage_changed(self, message: str) -> None:
        self.analysis_stage_text = str(message)
        self._update_analysis_elapsed()

    def _begin_blocking_activity(self, message: str) -> None:
        self.progress.setRange(0, 0)
        self.progress.setFormat("Выполняется…")
        self.status_label.setText(f"⏳ {message}")
        QApplication.processEvents()

    def _end_blocking_activity(self, message: str, *, success: bool = True) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)
        self.progress.setFormat("Готово" if success else "Ошибка")
        self.status_label.setText(("✓ " if success else "Ошибка: ") + message)
        QApplication.processEvents()

    def _update_analysis_elapsed(self) -> None:
        if self.analysis_started_at is None or not self.analysis_stage_text:
            return
        elapsed = max(0.0, time.monotonic() - self.analysis_started_at)
        self.status_label.setText(f"⏳ {self.analysis_stage_text}  ·  {elapsed:.1f} с")

    def _analysis_thread_finished(self) -> None:
        finished = self.sender()
        if finished is self.analysis_worker:
            self.analysis_worker = None
        if isinstance(finished, QThread):
            finished.deleteLater()

    def _open_path(
        self, path: Path, *, precision_override: str | None = None, reset_view: bool = True,
        completion_message: str | None = None,
    ) -> None:
        if self.analysis_worker is not None:
            self.status_label.setText("Анализ уже выполняется; дождитесь завершения текущего расчёта")
            return
        if precision_override is None:
            precision_override = "normal"
            self._set_precision_combo("normal")
        precision_key = get_precision(precision_override or "normal").key
        profile = get_precision(precision_key)
        resolved_path = Path(path)

        # Switching Normal -> Precise -> Normal on the same unchanged photo must
        # not repeat an expensive analysis already completed in this session.
        # Fail closed when the source or manual vision markup changed.
        cached = self._analysis_mode_cache.get(precision_key) if self.current_path == resolved_path else None
        if cached is not None and self.current_rgb is not None:
            try:
                current_hash = AnalysisDatabase.quick_hash(resolved_path)
            except OSError:
                current_hash = None
            if (
                current_hash
                and current_hash == cached.get("quick_hash")
                and list(cached.get("manual_faces", [])) == self.manual_face_boxes
                and list(cached.get("manual_eyes", [])) == self.manual_eye_boxes
            ):
                result = cached.get("result")
                if result is not None:
                    self.analysis_started_at = time.monotonic()
                    self.analysis_stage_text = f"Готовый анализ «{profile.label}» из кэша…"
                    self.status_label.setText(f"⚡ {self.analysis_stage_text}")
                    loaded = LoadedImage(info=result.image, srgb=self.current_rgb, linear_rgb=None)
                    self._analysis_completed({
                        "path": resolved_path,
                        "loaded": loaded,
                        "result": result,
                        "manual_faces": list(self.manual_face_boxes),
                        "manual_eyes": list(self.manual_eye_boxes),
                        "precision": precision_key,
                        "reset_view": reset_view,
                        "completion_message": f"Готово из кэша: анализ «{profile.label}»",
                        "warnings": [],
                    })
                    return

        self.analysis_started_at = time.monotonic()
        self.analysis_stage_text = f"Запуск анализа «{profile.label}»: {resolved_path.name}…"
        self._set_single_analysis_busy(True, self.analysis_stage_text)
        self.analysis_timer.start()
        worker = PhotoAnalysisWorker(
            Path(path), self.db_path, precision_key, reset_view, completion_message
        )
        self.analysis_worker = worker
        worker.stage.connect(self._analysis_stage_changed)
        worker.completed.connect(self._analysis_completed)
        worker.failed.connect(self._analysis_failed)
        worker.finished.connect(self._analysis_thread_finished)
        worker.start()

    def _analysis_failed(self, message: str) -> None:
        elapsed = 0.0 if self.analysis_started_at is None else max(0.0, time.monotonic() - self.analysis_started_at)
        self.analysis_timer.stop()
        self.analysis_stage_text = ""
        self.analysis_started_at = None
        self._set_single_analysis_busy(False)
        self.status_label.setText(f"Ошибка анализа после {elapsed:.1f} с")
        QMessageBox.critical(self, "Ошибка", message)

    def _analysis_completed(self, payload: object) -> None:
        elapsed = 0.0 if self.analysis_started_at is None else max(0.0, time.monotonic() - self.analysis_started_at)
        self.analysis_timer.stop()
        self.analysis_stage_text = ""
        self.analysis_started_at = None
        if not isinstance(payload, dict):
            self._set_single_analysis_busy(False)
            self.status_label.setText("Ошибка: анализ вернул некорректный результат")
            return
        path = Path(payload["path"])
        loaded = payload["loaded"]
        result = payload["result"]
        incoming_hash = payload.get("source_quick_hash")
        if not incoming_hash:
            try:
                incoming_hash = AnalysisDatabase.quick_hash(path)
            except OSError:
                incoming_hash = None
        if self.current_path != path or (
            self.current_source_quick_hash is not None
            and incoming_hash is not None
            and self.current_source_quick_hash != incoming_hash
        ):
            self._analysis_mode_cache.clear()
        self.manual_face_boxes = list(payload.get("manual_faces", []))
        self.manual_eye_boxes = list(payload.get("manual_eyes", []))
        reset_view = bool(payload.get("reset_view", True))
        self.current_path = path
        self.current_source_quick_hash = str(incoming_hash) if incoming_hash else None
        self.current_rgb = loaded.srgb.copy()
        self._invalidate_preview_cache()
        self._preview_is_stale = False
        if hasattr(self, "preview_btn"):
            self.preview_btn.blockSignals(True)
            self.preview_btn.setChecked(False)
            self.preview_btn.blockSignals(False)
            self.preview_btn.setText("Предпросмотр")
        self.current_metrics = result.metrics
        self.current_image_info = result.image
        precision_key = get_precision(str(payload.get("precision") or "normal")).key
        if incoming_hash:
            self._analysis_mode_cache[precision_key] = {
                "quick_hash": incoming_hash,
                "result": result,
                "manual_faces": list(self.manual_face_boxes),
                "manual_eyes": list(self.manual_eye_boxes),
            }
        validation_metric = result.metrics.get("recommendation_validation")
        validation_raw = validation_metric.raw_value if validation_metric is not None else None
        validation_items = validation_raw.get("items", []) if isinstance(validation_raw, dict) else []
        self.current_validation_items = [
            item for item in validation_items if isinstance(item, dict)
        ] if isinstance(validation_items, list) else []
        self.correction_regions = {}
        self.surface_repair_signatures = set()
        self._surface_repair_seeded = False
        self.preview_btn.blockSignals(True)
        self.preview_btn.setChecked(False)
        self.preview_btn.blockSignals(False)
        self.preview_btn.setEnabled(False)
        self.save_copy_btn.setEnabled(False)
        self.image_view.set_rgb(self.current_rgb)
        if reset_view:
            self.image_view.set_fit(True)
        self._show_analysis(result.metrics)
        surface = result.metrics.get("surface_defects")
        refined_surface = result.metrics.get("surface_refinement")
        refinement_raw = refined_surface.raw_value if refined_surface is not None and isinstance(refined_surface.raw_value, dict) else {}
        boxes = self._surface_boxes_with_feedback(result.metrics)
        self.current_surface_boxes = boxes
        self._refresh_surface_overlay()
        self.defects_btn.setEnabled(bool(boxes))
        if isinstance(refinement_raw, dict) and int(refinement_raw.get("evaluated_count", 0) or 0) > 0:
            self.defects_btn.setToolTip(
                "Карта дефектов: по умолчанию показаны только отмеченные «Лечить». В режиме «все найденные»: красный — ИИ подтверждает дефект; зелёный — вероятная естественная деталь; "
                "янтарный — неоднозначно; пунктир без AI-цвета — только классический кандидат. "
                f"Проверено ИИ: {int(refinement_raw.get('evaluated_count', 0) or 0)}, "
                f"дефектов: {int(refinement_raw.get('confirmed_defect_count', 0) or 0)}, "
                f"естественных: {int(refinement_raw.get('likely_natural_count', 0) or 0)}, "
                f"неоднозначных: {int(refinement_raw.get('uncertain_count', 0) or 0)}."
            )
        else:
            self.defects_btn.setToolTip("Показать карту дефектов. По умолчанию видны только отмеченные «Лечить»; все классические кандидаты можно временно показать на вкладке «Дефекты».")
        if not boxes:
            self.defects_btn.setChecked(False)

        sharpness_metric = result.metrics.get("local_sharpness")
        sharpness_cells = []
        if sharpness_metric is not None and isinstance(sharpness_metric.raw_value, dict):
            raw_cells = sharpness_metric.raw_value.get("cells_norm", [])
            if isinstance(raw_cells, list):
                sharpness_cells = [cell for cell in raw_cells if isinstance(cell, dict)]
        self.image_view.set_sharpness_cells(sharpness_cells)
        self.sharpness_btn.setEnabled(bool(sharpness_cells))
        if not sharpness_cells:
            self.sharpness_btn.setChecked(False)

        tone_metric = result.metrics.get("local_tone")
        tone_cells = []
        if tone_metric is not None and isinstance(tone_metric.raw_value, dict):
            raw_tone_cells = tone_metric.raw_value.get("cells_norm", [])
            if isinstance(raw_tone_cells, list):
                tone_cells = [cell for cell in raw_tone_cells if isinstance(cell, dict)]
        self.image_view.set_tone_cells(tone_cells)
        self.tone_btn.setEnabled(bool(tone_cells))
        if not tone_cells:
            self.tone_btn.setChecked(False)

        contrast_metric = result.metrics.get("local_contrast")
        contrast_cells = []
        if contrast_metric is not None and isinstance(contrast_metric.raw_value, dict):
            raw_contrast_cells = contrast_metric.raw_value.get("cells_norm", [])
            if isinstance(raw_contrast_cells, list):
                contrast_cells = [cell for cell in raw_contrast_cells if isinstance(cell, dict)]
        self.image_view.set_contrast_cells(contrast_cells)
        self.contrast_btn.setEnabled(bool(contrast_cells))
        if not contrast_cells:
            self.contrast_btn.setChecked(False)

        face_metric = result.metrics.get("faces")
        face_boxes = []
        if face_metric is not None and isinstance(face_metric.raw_value, dict):
            raw_faces = face_metric.raw_value.get("faces", [])
            if isinstance(raw_faces, list):
                face_boxes = [face for face in raw_faces if isinstance(face, dict)]
        self.image_view.set_face_boxes(face_boxes)
        self.faces_btn.setEnabled(bool(face_boxes))
        if not face_boxes:
            self.faces_btn.setChecked(False)

        eye_metric = result.metrics.get("eyes")
        eye_boxes = []
        if eye_metric is not None and isinstance(eye_metric.raw_value, dict):
            raw_eyes = eye_metric.raw_value.get("eyes", [])
            if isinstance(raw_eyes, list):
                eye_boxes = [eye for eye in raw_eyes if isinstance(eye, dict)]
        self.image_view.set_eye_boxes(eye_boxes)

        subject_metric = result.metrics.get("main_subject")
        subject_box = None
        if subject_metric is not None and isinstance(subject_metric.raw_value, dict):
            raw_subject_box = subject_metric.raw_value.get("box_norm")
            confidence = float(subject_metric.raw_value.get("confidence", subject_metric.confidence) or 0.0)
            if isinstance(raw_subject_box, dict) and confidence >= 0.42:
                subject_box = {k: float(raw_subject_box.get(k, 0.0) or 0.0) for k in ("x", "y", "w", "h")}
        self.image_view.set_subject_box(subject_box)
        self.subject_btn.setEnabled(subject_box is not None)
        if subject_box is None:
            self.subject_btn.setChecked(False)

        highlight_metric = result.metrics.get("highlight_context")
        highlight_boxes = []
        if highlight_metric is not None and isinstance(highlight_metric.raw_value, dict):
            raw_highlights = highlight_metric.raw_value.get("boxes_norm", [])
            if isinstance(raw_highlights, list):
                highlight_boxes = [box for box in raw_highlights if isinstance(box, dict)]
        self.image_view.set_highlight_boxes(highlight_boxes)
        self.highlights_btn.setEnabled(bool(highlight_boxes))
        if not highlight_boxes:
            self.highlights_btn.setChecked(False)

        self._set_single_analysis_busy(False)
        completion = str(payload.get("completion_message") or "").strip()
        profile = get_precision(str(payload.get("precision") or "normal"))
        warnings = [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()]
        if completion:
            status_text = f"{completion}  ·  {elapsed:.1f} с"
        else:
            status_text = f"Готово: {path.name} — анализ «{profile.label}» завершён за {elapsed:.1f} с"
        if warnings:
            self.status_label.setText(status_text + " · ⚠ есть предупреждение")
            QMessageBox.warning(self, "Анализ завершён с предупреждением", "\n\n".join(warnings))
        else:
            self.status_label.setText(status_text)

    def _show_analysis(self, metrics) -> None:
        self._show_result(metrics)
        self._show_metrics(metrics)
        self._show_histogram(metrics)
        self._show_recommendations(metrics)
        self._show_plan(metrics)
        self._show_faces(metrics)
        self._show_defects(metrics)
        self._show_ai(metrics)
        self._refresh_almaz_panel()
        self._show_technical(metrics)

    @staticmethod
    def _correction_action_label(item: dict) -> str:
        key = str(item.get("action_key", ""))
        if key == "red_eye":
            candidates = item.get("source_candidates")
            count = len(candidates) if isinstance(candidates, list) else int(float(item.get("target_before", 0) or 0))
            return f"Красные глаза — {max(1, count)}"
        if key == "exposure":
            return "Средние тона — локально по карте" if str(item.get("candidate", "")) == "spatial_exposure_v1" else "Средние тона — мягкая коррекция"
        if key == "white_balance":
            if str(item.get("candidate", "")) == "spatial_white_balance_v1":
                return "Баланс белого — локально по карте освещения"
            params = item.get("parameters") if isinstance(item, dict) else None
            if isinstance(params, dict):
                try:
                    temp = float(params.get("recommended_temperature_shift_k", 0.0) or 0.0)
                    tint = float(params.get("recommended_tint_shift", 0.0) or 0.0)
                    return f"Баланс белого — {temp:+.0f} K · tint {tint:+.1f}"
                except (TypeError, ValueError):
                    pass
            return "Баланс белого — температура / tint"
        if key == "auto_tone_color":
            return "Автотон + автоконтраст + автоцвет — Photoshop-подобный профиль"
        if key == "contrast":
            return "Локальный контраст — по карте" if str(item.get("candidate", "")) == "spatial_contrast_v1" else "Локальный контраст — мягкая коррекция"
        if key == "sharpness":
            candidate = str(item.get("candidate", ""))
            if candidate == "almaz_ai_deblur_v1":
                return "Резкость / смаз — ALMAZ AI Deblur"
            return "Резкость — локально по карте" if candidate == "spatial_sharpness_v1" else "Резкость — мягкое усиление существующих деталей"
        if key == "super_resolution":
            return "ALMAZ — x2 восстановление с защитой лица"
        if key == "almaz_denoise":
            return "ALMAZ — AI Denoise"
        if key == "almaz_deblur":
            return "ALMAZ — AI Deblur"
        if key == "almaz_jpeg_recovery":
            return "ALMAZ — JPEG Recovery"
        if key == "surface_defects":
            candidates = item.get("source_candidates")
            count = len(candidates) if isinstance(candidates, list) else 0
            return f"Дефекты поверхности — локальное лечение {count} участков" if count else "Дефекты поверхности — локальное лечение"
        if key == "noise":
            candidate = str(item.get("candidate", ""))
            if candidate == "almaz_ai_denoise_v1":
                return "Шум — ALMAZ AI Denoise"
            return "Шум — локально по карте" if candidate == "spatial_denoise_v1" else "Шум — мягкое шумоподавление"
        if key == "jpeg_artifacts":
            if str(item.get("candidate", "")) == "almaz_ai_jpeg_recovery_v1":
                return "JPEG / смаз — ALMAZ AI Recovery"
            return "JPEG — мягкое сглаживание блоков"
        if key == "edge_artifacts":
            return "Ореолы — мягкое смягчение границ"
        if key == "posterization":
            return "Полосатость градаций — мягкое сглаживание градаций"
        return key or "Исправление"

    def _correction_preference_stats(self) -> dict[str, dict[str, object]]:
        try:
            with AnalysisDatabase(self.db_path) as db:
                rows = db.correction_feedback_stats()
        except Exception:
            return {}
        return {str(row.get("action_key", "")): row for row in rows if row.get("action_key")}

    def _parameter_training_histories(self, action_keys: set[str]) -> dict[str, list[dict[str, object]]]:
        histories: dict[str, list[dict[str, object]]] = {key: [] for key in action_keys}
        if not action_keys:
            return histories
        try:
            with AnalysisDatabase(self.db_path) as db:
                for key in action_keys:
                    histories[key] = db.correction_parameter_training_samples(
                        key, model_id=PARAMETER_MODEL_ID, feature_schema=PARAMETER_FEATURE_SCHEMA
                    )
        except Exception:
            return histories
        return histories

    def _populate_correction_choices(self, validation_items: list[dict]) -> None:
        self._updating_correction_choices = True
        self.fix_actions_list.clear()
        self.fix_summary_list.clear()
        self.correction_strengths = {}
        self.correction_ai_suggestions = {}
        preference_stats = self._correction_preference_stats()
        adjustable_keys = {
            str(item.get("action_key", ""))
            for item in validation_items
            if isinstance(item, dict) and preview_action_available(item) and bool(item.get("adjustable", False))
        }
        histories = self._parameter_training_histories(adjustable_keys)
        recommender_context = self._correction_feedback_context()
        available_count = 0
        try:
            for item in validation_items:
                if not isinstance(item, dict) or not preview_action_available(item):
                    continue
                available_count += 1
                key = str(item.get("action_key", ""))
                accepted = bool(item.get("accepted", False))
                suggestion: dict[str, object] | None = None
                if bool(item.get("adjustable", False)):
                    try:
                        suggestion = recommend_strength(
                            key, item, recommender_context, histories.get(key, [])
                        ).to_dict()
                        self.correction_ai_suggestions[key] = suggestion
                        self.correction_strengths[key] = float(np.clip(float(suggestion["suggested_strength"]), 0.0, 1.0))
                    except Exception:
                        try:
                            self.correction_strengths[key] = float(np.clip(float(item.get("default_strength", 1.0) or 1.0), 0.0, 1.0))
                        except (TypeError, ValueError):
                            self.correction_strengths[key] = 1.0
                auto_eligible = bool(item.get("auto_eligible", True))
                technical_passed = item.get("technical_passed")
                if accepted:
                    status = "подтверждено проверкой безопасности"
                elif not auto_eligible and technical_passed is False:
                    status = "только вручную · проба не прошла техпроверку"
                elif not auto_eligible:
                    status = "только вручную · рекомендация требует проверки"
                else:
                    status = "вручную · проверка безопасности не подтвердила"
                text = f"{self._correction_action_label(item)} · {status}"
                if suggestion is not None:
                    ai_pct = int(round(float(suggestion.get("suggested_strength", 0.0)) * 100.0))
                    personal = bool(suggestion.get("personalized", False))
                    samples = int(suggestion.get("sample_count", 0) or 0)
                    text += f" · ИИ {ai_pct}%"
                    if personal:
                        text += f" по вашим решениям ({samples})"
                history = preference_stats.get(key, {})
                meaningful = int(history.get("meaningful_decisions", 0) or 0)
                selected_pct = history.get("selected_pct")
                if meaningful >= 5 and selected_pct is not None:
                    text += f" · вы оставляете {float(selected_pct):.0f}% ({meaningful})"
                list_item = QListWidgetItem(text)
                list_item.setFlags(list_item.flags() | Qt.ItemIsUserCheckable)
                list_item.setCheckState(Qt.Checked if accepted else Qt.Unchecked)
                list_item.setData(Qt.UserRole, key)
                list_item.setData(Qt.UserRole + 1, accepted)
                message = str(item.get("message", ""))
                if accepted:
                    tooltip = "Проверка безопасности разрешила автоматическое применение."
                elif not auto_eligible and technical_passed is False:
                    tooltip = "Модуль решений оставил задачу для ручной проверки, а пробная коррекция не прошла технические ограничения проверки безопасности. Галочка включает только явный ручной эксперимент."
                elif not auto_eligible:
                    tooltip = "Модуль решений оставил задачу для ручной проверки. Галочка включает только ручной эксперимент; автоматически это исправление не применяется."
                else:
                    tooltip = "Проверка безопасности не разрешила автоматическое применение. Галочка означает ваш ручной эксперимент для предпросмотра."
                if suggestion is not None:
                    ai_pct = int(round(float(suggestion.get("suggested_strength", 0.0)) * 100.0))
                    confidence = int(round(float(suggestion.get("confidence", 0.0)) * 100.0))
                    tooltip += (
                        f"\n\nРекомендатель параметров ИИ: {ai_pct}% · уверенность {confidence}%. "
                        + str(suggestion.get("explanation", ""))
                    )
                if message:
                    tooltip += "\n\n" + message
                list_item.setToolTip(tooltip)
                self.fix_actions_list.addItem(list_item)
                summary_item = QListWidgetItem(text)
                summary_item.setToolTip(tooltip)
                summary_item.setFlags(summary_item.flags() & ~Qt.ItemIsUserCheckable)
                self.fix_summary_list.addItem(summary_item)
            if available_count == 0:
                placeholder = QListWidgetItem("Нет доступных автоматических или ручных коррекций для этого кадра.")
                placeholder.setFlags(placeholder.flags() & ~Qt.ItemIsEnabled)
                self.fix_actions_list.addItem(placeholder)
                summary_placeholder = QListWidgetItem("Нет доступных коррекций для этого кадра.")
                summary_placeholder.setFlags(summary_placeholder.flags() & ~Qt.ItemIsEnabled)
                self.fix_summary_list.addItem(summary_placeholder)
        finally:
            self._updating_correction_choices = False
        self._update_correction_controls(refresh_preview=False)
        self._refresh_correction_feedback_history()

    def _selected_correction_keys(self) -> set[str]:
        selected: set[str] = set()
        for row in range(self.fix_actions_list.count()):
            item = self.fix_actions_list.item(row)
            key = item.data(Qt.UserRole)
            if key and item.checkState() == Qt.Checked:
                selected.add(str(key))
        return selected

    def _selected_manual_override_count(self) -> int:
        count = 0
        for row in range(self.fix_actions_list.count()):
            item = self.fix_actions_list.item(row)
            if item.checkState() != Qt.Checked or not item.data(Qt.UserRole):
                continue
            if not bool(item.data(Qt.UserRole + 1)):
                count += 1
        return count

    def _selected_correction_labels(self) -> list[str]:
        labels: list[str] = []
        for row in range(self.fix_actions_list.count()):
            item = self.fix_actions_list.item(row)
            key = item.data(Qt.UserRole)
            if not key or item.checkState() != Qt.Checked:
                continue
            source = next(
                (entry for entry in self.current_validation_items if isinstance(entry, dict) and str(entry.get("action_key", "")) == str(key)),
                None,
            )
            label = self._correction_action_label(source) if isinstance(source, dict) else str(key)
            if isinstance(source, dict) and bool(source.get("adjustable", False)):
                strength = int(round(self.correction_strengths.get(str(key), float(source.get("default_strength", 1.0) or 1.0)) * 100.0))
                label += f" · {strength}%"
                suggestion = self.correction_ai_suggestions.get(str(key), {})
                if suggestion:
                    try:
                        ai_strength = int(round(float(suggestion.get("suggested_strength", 0.0)) * 100.0))
                        if ai_strength != strength:
                            label += f" (ИИ {ai_strength}%)"
                    except (TypeError, ValueError):
                        pass
            if str(key) in self.correction_regions:
                label += " · выбранная область"
            labels.append(label)
        return labels

    def _update_correction_controls(self, *, refresh_preview: bool = True) -> None:
        if refresh_preview:
            self._mark_preview_stale_after_recipe_change()
        selected = self._selected_correction_keys()
        accepted_selected = 0
        available_count = 0
        for row in range(self.fix_actions_list.count()):
            item = self.fix_actions_list.item(row)
            if not item.data(Qt.UserRole):
                continue
            available_count += 1
            if item.checkState() == Qt.Checked and bool(item.data(Qt.UserRole + 1)):
                accepted_selected += 1
        manual = self._selected_manual_override_count()
        has_selected = bool(selected)
        self.preview_btn.setEnabled(has_selected)
        self.save_copy_btn.setEnabled(has_selected)
        if selected:
            text = f"Выбрано: {len(selected)} · безопасных: {accepted_selected}"
            if manual:
                text += f" · вручную: {manual}"
            labels = self._selected_correction_labels()
            applied_prefix = "Сейчас применено в предпросмотре" if self.preview_btn.isChecked() else "Будет применено"
            self.active_corrections_label.setText(applied_prefix + ": " + "; ".join(labels))
            if self.preview_btn.isChecked():
                text = "Сейчас применено в предпросмотре: " + str(len(selected)) + text[text.find(" ·"):]
            self.confirmed_fixes_label.setText(text)
        elif available_count:
            self.confirmed_fixes_label.setText("Есть предложения · выберите исправления")
            self.active_corrections_label.setText("Будет применено: ничего")
        else:
            self.confirmed_fixes_label.setText("Предложений для применения нет")
            self.active_corrections_label.setText("Будет применено: ничего")
        if not has_selected and self.preview_btn.isChecked():
            self.preview_btn.blockSignals(True)
            self.preview_btn.setChecked(False)
            self.preview_btn.blockSignals(False)
            if self.current_rgb is not None:
                self.image_view.set_rgb(self.current_rgb)
        self._sync_plan_corrections_from_result()

    def _enforce_auto_trio_exclusivity(self, changed_key: str) -> None:
        """Keep compound auto tone/color mutually exclusive with atomic tone actions."""
        changed = self._fix_list_item_for_key(changed_key)
        if changed is None or changed.checkState() != Qt.Checked:
            return
        if changed_key == "auto_tone_color":
            conflicts = {"exposure", "contrast", "white_balance"}
        elif changed_key == "white_balance":
            conflicts = {"auto_tone_color"}
        elif changed_key in {"exposure", "contrast"}:
            conflicts = {"auto_tone_color"}
        else:
            conflicts = set()
        if not conflicts:
            return
        self._updating_correction_choices = True
        try:
            for key in conflicts:
                other = self._fix_list_item_for_key(key)
                if other is not None:
                    other.setCheckState(Qt.Unchecked)
        finally:
            self._updating_correction_choices = False

    def _on_correction_item_changed(self, _item: QListWidgetItem) -> None:
        if self._updating_correction_choices:
            return
        key = str(_item.data(Qt.UserRole) or "")
        if key:
            self._enforce_auto_trio_exclusivity(key)
        self._update_correction_controls(refresh_preview=True)

    def _reset_correction_selection(self) -> None:
        self._updating_correction_choices = True
        try:
            for row in range(self.fix_actions_list.count()):
                item = self.fix_actions_list.item(row)
                if item.data(Qt.UserRole):
                    item.setCheckState(Qt.Checked if bool(item.data(Qt.UserRole + 1)) else Qt.Unchecked)
        finally:
            self._updating_correction_choices = False
        self._update_correction_controls(refresh_preview=True)

    def _select_all_corrections(self) -> None:
        self._updating_correction_choices = True
        try:
            for row in range(self.fix_actions_list.count()):
                item = self.fix_actions_list.item(row)
                if item.data(Qt.UserRole):
                    key = str(item.data(Qt.UserRole) or "")
                    if key == "surface_defects":
                        item.setCheckState(Qt.Checked if bool(self.surface_repair_signatures) else Qt.Unchecked)
                    else:
                        item.setCheckState(Qt.Checked)
        finally:
            self._updating_correction_choices = False
        self._update_correction_controls(refresh_preview=True)

    def _clear_correction_selection(self) -> None:
        self._updating_correction_choices = True
        try:
            for row in range(self.fix_actions_list.count()):
                item = self.fix_actions_list.item(row)
                if item.data(Qt.UserRole):
                    item.setCheckState(Qt.Unchecked)
        finally:
            self._updating_correction_choices = False
        self._update_correction_controls(refresh_preview=True)

    def _correction_feedback_setting_changed(self, state: int) -> None:
        enabled = bool(state)
        self.settings.setValue("correction_feedback_enabled", enabled)
        self._refresh_correction_feedback_history()

    def _refresh_correction_feedback_history(self) -> None:
        if not self.correction_feedback_checkbox.isChecked():
            self.correction_feedback_history_label.setText(
                "Запоминание решений отключено. Уже сохранённая история не удаляется, но Рекомендатель параметров ИИ не получает новые примеры."
            )
            return
        stats = self._correction_preference_stats()
        meaningful = sum(int(row.get("meaningful_decisions", 0) or 0) for row in stats.values())
        overrides = sum(
            int(row.get("user_overrode", 0) or 0) + int(row.get("user_enabled_review", 0) or 0)
            for row in stats.values()
        )
        disabled = sum(int(row.get("user_disabled", 0) or 0) for row in stats.values())
        strength_samples = sum(int(row.get("strength_samples", 0) or 0) for row in stats.values())
        ai_strength_kept = sum(int(row.get("ai_strength_kept", 0) or 0) for row in stats.values())
        ai_strength_changed = sum(int(row.get("ai_strength_changed", 0) or 0) for row in stats.values())
        if meaningful <= 0:
            self.correction_feedback_history_label.setText(
                "Локальная история решений: пока нет сохранённых решений. После сохранения Рекомендатель параметров ИИ начнёт учиться на ваших значениях ползунков."
            )
            return
        strength_compare = ""
        if ai_strength_kept or ai_strength_changed:
            strength_compare = f" · как предложил ИИ {ai_strength_kept} · изменено {ai_strength_changed}"
        self.correction_feedback_history_label.setText(
            f"Локальная история: {meaningful} решений · примеров силы {strength_samples}{strength_compare} · "
            f"вручную добавлено {overrides} · отключено {disabled}. "
            "Рекомендатель параметров ИИ использует сохранённые значения при следующих фото; базовые модели анализа и проверка безопасности не переобучаются."
        )

    def _correction_feedback_context(self) -> dict[str, object]:
        context: dict[str, object] = {
            "app_version": __version__,
            "metrics": {},
            "correction_strengths": {key: round(float(value), 4) for key, value in self.correction_strengths.items()},
            "ai_parameter_suggestions": {
                key: {
                    "suggested_strength": round(float(value.get("suggested_strength", 0.0) or 0.0), 4),
                    "baseline_strength": round(float(value.get("baseline_strength", 0.0) or 0.0), 4),
                    "confidence": round(float(value.get("confidence", 0.0) or 0.0), 4),
                    "personalized": bool(value.get("personalized", False)),
                    "sample_count": int(value.get("sample_count", 0) or 0),
                    "model_id": str(value.get("model_id", "")),
                    "feature_schema": str(value.get("feature_schema", "")),
                }
                for key, value in self.correction_ai_suggestions.items()
            },
            "correction_regions": {key: {k: round(float(v), 5) for k, v in region.items()} for key, region in self.correction_regions.items()},
        }
        if not isinstance(self.current_metrics, dict):
            return context
        metric_payload: dict[str, object] = {}
        for key in (
            "brightness", "contrast", "local_contrast", "neutral_balance", "white_balance_advisor", "local_white_balance",
            "red_eye", "noise", "sharpness",
            "laplacian", "tenengrad", "jpeg_artifacts", "edge_artifacts", "posterization",
            "surface_defects", "faces", "eyes", "semantic_context", "main_subject", "aesthetic_quality", "image_tone",
            "surface_refinement", "ai_trust_gate",
        ):
            metric = self.current_metrics.get(key)
            if metric is None:
                continue
            payload: dict[str, object] = {
                "normalized_value": metric.normalized_value,
                "confidence": metric.confidence,
                "diagnostic": metric.diagnostic,
            }
            if key == "semantic_context" and isinstance(metric.raw_value, dict):
                value = metric.raw_value.get("archival_likelihood")
                if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
                    payload["archival_likelihood"] = float(np.clip(float(value), 0.0, 1.0))
            metric_payload[key] = payload
        context["metrics"] = metric_payload
        decision_metric = self.current_metrics.get("decision_plan")
        if decision_metric is not None and isinstance(decision_metric.raw_value, dict):
            items = decision_metric.raw_value.get("items", [])
            if isinstance(items, list):
                context["decision_plan"] = [
                    {
                        "key": item.get("key"),
                        "decision": item.get("decision"),
                        "severity": item.get("severity"),
                        "repairability": item.get("repairability"),
                        "confidence": item.get("confidence"),
                        "priority": item.get("priority"),
                    }
                    for item in items if isinstance(item, dict)
                ]
        return context

    def _record_saved_correction_feedback(self, saved: Path) -> int:
        if not self.correction_feedback_checkbox.isChecked() or self.current_path is None:
            return 0
        if self.current_source_quick_hash is not None:
            try:
                current_hash = AnalysisDatabase.quick_hash(self.current_path)
            except OSError as exc:
                raise RuntimeError("исходник недоступен после анализа; обратная связь не записана") from exc
            if current_hash != self.current_source_quick_hash:
                raise RuntimeError("исходник изменился после анализа; обратная связь не записана")
        selected = self._selected_correction_keys()
        with AnalysisDatabase(self.db_path) as db:
            count = db.record_correction_feedback(
                self.current_path,
                self.current_validation_items,
                selected,
                saved_format=saved.suffix.lower().lstrip("."),
                context=self._correction_feedback_context(),
                user_strengths=self.correction_strengths,
                parameter_suggestions=self.correction_ai_suggestions,
            )
        self._refresh_correction_feedback_history()
        self._refresh_ai_training_stats()
        return count

    def _update_fix_summary(self, metrics) -> None:
        validation_items = [item for item in self.current_validation_items if isinstance(item, dict)]
        if not validation_items:
            validation_metric = metrics.get("recommendation_validation")
            validation_raw = validation_metric.raw_value if validation_metric is not None and isinstance(validation_metric.raw_value, dict) else {}
            raw_items = validation_raw.get("items", []) if isinstance(validation_raw, dict) else []
            validation_items = [item for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []
        validation_by_key = {str(item.get("action_key", "")): item for item in validation_items}

        decision_metric = metrics.get("decision_plan")
        decision_raw = decision_metric.raw_value if decision_metric is not None and isinstance(decision_metric.raw_value, dict) else {}
        decision_items = decision_raw.get("items", []) if isinstance(decision_raw, dict) else []
        review_lines: list[str] = []
        preserve_lines: list[str] = []
        if isinstance(decision_items, list):
            for item in decision_items:
                if not isinstance(item, dict):
                    continue
                decision = str(item.get("decision", ""))
                key = str(item.get("key", ""))
                title = str(item.get("title", key or "Проверка"))
                if decision == "review":
                    review_lines.append(f"! {title}")
                elif decision == "preserve":
                    preserve_lines.append(f"• {title}")

        red_eye_metric = metrics.get("red_eye")
        red_eye_raw = red_eye_metric.raw_value if red_eye_metric is not None and isinstance(red_eye_metric.raw_value, dict) else {}
        red_count = int(red_eye_raw.get("suspicious_eye_count", 0) or 0)
        red_validation = validation_by_key.get("red_eye", {})
        if red_count <= 0:
            red_eye_line = "Красные глаза: не обнаружено"
        elif bool(red_validation.get("accepted", False)):
            red_eye_line = f"Красные глаза: обнаружено {red_count} · Можно исправить автоматически"
        elif preview_action_available(red_validation):
            red_eye_line = f"Красные глаза: обнаружено {red_count} · Можно проверить вручную"
        else:
            red_eye_line = f"Красные глаза: обнаружено {red_count} · Требуется ручная проверка"

        sections: list[str] = [f"<b>{red_eye_line}</b>"]
        if review_lines:
            sections.append(f"<b>Нужно проверить вручную: {len(review_lines)}</b><br>" + "<br>".join(review_lines[:5]))
        if preserve_lines:
            sections.append(f"<b>Лучше не трогать: {len(preserve_lines)}</b><br>" + "<br>".join(preserve_lines[:3]))
        if not review_lines and not preserve_lines:
            sections.append("Критических спорных решений не обнаружено.")
        self.fix_summary_label.setText("<br><br>".join(sections))
        self._populate_correction_choices(validation_items)

    @staticmethod
    def _format_file_size(size_bytes: int | None) -> str:
        if size_bytes is None or size_bytes < 0:
            return ""
        value = float(size_bytes)
        for unit in ("Б", "КБ", "МБ", "ГБ"):
            if value < 1024.0 or unit == "ГБ":
                return f"{value:.0f} {unit}" if unit in {"Б", "КБ"} else f"{value:.2f} {unit}"
            value /= 1024.0
        return ""

    @staticmethod
    def _format_shutter(exposure_s: object) -> str | None:
        try:
            value = float(exposure_s)
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.isfinite(value) or value <= 0:
            return None
        if value < 1.0:
            denominator = max(1, int(round(1.0 / value)))
            return f"1/{denominator} с"
        return f"{value:.2f} с".replace(".00", "")

    def _update_overview_photo_metadata(self, metrics) -> None:
        info = self.current_image_info
        if info is None and self.current_rgb is not None:
            height, width = self.current_rgb.shape[:2]
            fmt = self.current_path.suffix.lstrip(".").upper() if self.current_path is not None else ""
            megapixels = width * height / 1_000_000.0
            self.photo_info_label.setText(f"{width}×{height} · {megapixels:.2f} Мп" + (f" · {fmt}" if fmt else ""))
        elif info is not None:
            width, height = int(info.width), int(info.height)
            megapixels = width * height / 1_000_000.0
            parts = [f"{width}×{height}", f"{megapixels:.2f} Мп", str(info.format or "").upper()]
            if bool(info.icc_present):
                parts.append("ICC → sRGB")
            else:
                parts.append("sRGB")
            if self.current_path is not None:
                try:
                    file_size = self._format_file_size(self.current_path.stat().st_size)
                except OSError:
                    file_size = ""
                if file_size:
                    parts.append(file_size)
            self.photo_info_label.setText(" · ".join(part for part in parts if part))

        exif_metric = metrics.get("exif_context")
        raw = exif_metric.raw_value if exif_metric is not None else None
        if not isinstance(raw, dict) or not bool(raw.get("available", False)):
            self.camera_info_label.setText("Камера и параметры съёмки: нет данных EXIF")
            return

        camera_parts: list[str] = []
        make = str(raw.get("make") or "").strip()
        model = str(raw.get("model") or "").strip()
        camera = " ".join(part for part in (make, model) if part)
        if camera:
            camera_parts.append(camera)
        lens = str(raw.get("lens_model") or "").strip()
        if lens and lens.lower() not in camera.lower():
            camera_parts.append(lens)
        try:
            aperture = float(raw.get("aperture_f"))
            if np.isfinite(aperture) and aperture > 0:
                camera_parts.append(f"f/{aperture:.1f}")
        except (TypeError, ValueError, OverflowError):
            pass
        shutter = self._format_shutter(raw.get("exposure_s"))
        if shutter:
            camera_parts.append(shutter)
        try:
            iso = float(raw.get("iso"))
            if np.isfinite(iso) and iso > 0:
                camera_parts.append(f"ISO {iso:.0f}")
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            focal = float(raw.get("focal_length_mm"))
            if np.isfinite(focal) and focal > 0:
                camera_parts.append(f"{focal:g} мм")
        except (TypeError, ValueError, OverflowError):
            pass
        date = str(raw.get("datetime_original") or "").strip()
        if date:
            camera_parts.append(date.replace(":", "-", 2))
        self.camera_info_label.setText(" · ".join(camera_parts) if camera_parts else "Параметры съёмки EXIF присутствуют частично")

    def _show_result(self, metrics) -> None:
        summary = build_summary(metrics)
        profile = build_photo_profile(metrics)
        self._update_fix_summary(metrics)
        self.overview_stats_label.setText(
            f"<b>Качество</b> {summary['quality']:.1f}/100 &nbsp;&nbsp;&nbsp; "
            f"<b>Уверенность анализа</b> {summary['confidence']:.0f}% &nbsp;&nbsp;&nbsp; "
            f"<b>Проблем найдено</b> {summary['issues']}"
        )
        self.profile_label.setText(
            f"{profile.label} · уверенность {profile.confidence * 100:.0f}%"
        )
        self.profile_label.setToolTip(profile.explanation)
        semantic_metric = metrics.get("semantic_context")
        semantic_raw = semantic_metric.raw_value if semantic_metric is not None else None
        if isinstance(semantic_raw, dict):
            kind = str(semantic_raw.get("classification", "unknown"))
            people = str(semantic_raw.get("people_context", "none"))
            labels = {
                "archival_portrait": "Архивный портрет",
                "group_portrait": "Групповой портрет",
                "portrait": "Портрет",
                "general_photo": "Обычный кадр / не определён как портрет",
            }
            context_text = labels.get(kind, "Не определён")
            if kind == "archival_portrait" and people == "group":
                context_text += " · несколько лиц"
            confidence = float(semantic_metric.confidence or 0.0)
            self.semantic_label.setText(
                f"{context_text} &nbsp; "
                f"<span style='color:gray'>(уверенность {confidence * 100:.0f}%)</span>"
            )
        else:
            self.semantic_label.setText("Контекст: —")

        subject_metric = metrics.get("main_subject")
        subject_raw = subject_metric.raw_value if subject_metric is not None else None
        if isinstance(subject_raw, dict):
            kind = str(subject_raw.get("subject_kind", "unknown"))
            scene = str(subject_raw.get("scene_kind", "unknown_scene"))
            confidence = float(subject_raw.get("confidence", subject_metric.confidence) or 0.0)
            kind_labels = {"person": "человек", "people_group": "группа людей", "visual_region": "визуально выделяющаяся область", "unknown": "не определён"}
            scene_labels = {"archival_people": "архивный кадр с людьми", "people_group": "групповой кадр", "people_portrait": "портретный кадр", "general_scene": "общая сцена", "unknown_scene": "сцена не определена"}
            if kind == "unknown":
                self.subject_label.setText(f"Главный объект не определён &nbsp; <span style='color:gray'>(уверенность {confidence * 100:.0f}%)</span>")
            else:
                area = float(subject_raw.get("subject_area_pct", 0.0) or 0.0)
                self.subject_label.setText(
                    f"{kind_labels.get(kind, kind)} · {scene_labels.get(scene, scene)} · "
                    f"область ~{area:.0f}% кадра &nbsp; <span style='color:gray'>(уверенность {confidence * 100:.0f}%)</span>"
                )
        else:
            self.subject_label.setText("Главный объект: —")

        self._update_overview_photo_metadata(metrics)

        aesthetic_metric = metrics.get("aesthetic_quality")
        aesthetic_raw = aesthetic_metric.raw_value if aesthetic_metric is not None else None
        if isinstance(aesthetic_raw, dict) and aesthetic_raw.get("score") is not None:
            score = float(aesthetic_raw.get("score", 0.0) or 0.0)
            confidence = float(aesthetic_raw.get("confidence", aesthetic_metric.confidence) or 0.0)
            self.aesthetic_label.setText(
                f"<b>Эстетика:</b> {score:.0f}/100 &nbsp; <span style='color:gray'>(композиционный ориентир, уверенность {confidence * 100:.0f}%)</span>"
            )
        else:
            self.aesthetic_label.setText("<b>Эстетика:</b> недостаточно данных для надёжной оценки")

        reliability_metric = metrics.get("analysis_reliability")
        reliability_raw = reliability_metric.raw_value if reliability_metric is not None and isinstance(reliability_metric.raw_value, dict) else None
        if isinstance(reliability_raw, dict):
            status = str(reliability_raw.get("status", "unknown"))
            calibrated = float(reliability_raw.get("calibrated_confidence", reliability_metric.confidence) or 0.0)
            labels = {
                "reliable": "Надёжно",
                "caution": "С осторожностью",
                "unknown": "Недостаточно данных",
                "ood_candidate": "Нетипичный вход",
            }
            self.reliability_label.setText(
                f"<b>Надёжность анализа:</b> {labels.get(status, 'Недостаточно данных')} &nbsp; "
                f"<span style='color:gray'>(калиброванная поддержка {calibrated * 100:.0f}%; не вероятность истины)</span>"
            )
        else:
            self.reliability_label.setText("<b>Надёжность анализа:</b> —")

        rows = build_display_metrics(metrics)
        row_map = {row.key: row for row in rows}
        noteworthy = [r for r in rows if r.status in {"Проблема", "Внимание", "Проверить"}]
        features = [r for r in rows if r.status == "Особенность"]
        lines: list[str] = []
        if noteworthy:
            lines.append("Что требует внимания:")
            sharpness_keys = {"sharpness", "detail_loss_type", "local_sharpness", "faces", "eyes"}
            sharpness_rows = [r for r in noteworthy if r.key in sharpness_keys]
            for row in noteworthy:
                if row.key in sharpness_keys:
                    continue
                lines.append(f"• {row.label} [{row.status}]: {row.diagnosis}")
            if sharpness_rows:
                base = row_map.get("sharpness") or sharpness_rows[0]
                details: list[str] = []
                detail_metric = metrics.get("detail_loss_type")
                if detail_metric is not None and isinstance(detail_metric.raw_value, dict):
                    kind = str(detail_metric.raw_value.get("classification", "unknown"))
                    conf = float(detail_metric.raw_value.get("confidence", detail_metric.confidence) or 0.0)
                    kind_labels = {
                        "camera_shake_like": "похоже на дрожание камеры",
                        "subject_motion_like": "похоже на движение главного объекта",
                        "motion_like": "есть направленный смаз, источник неясен",
                        "local_subject_softness": "главный объект мягче фона",
                        "defocus_like": "похоже на дефокус/оптическую мягкость",
                        "degradation_like": "скорее деградация старого оригинала",
                        "mixed_or_degraded": "тип потери деталей смешанный",
                        "mixed": "тип потери деталей смешанный",
                        "unknown": "тип потери деталей не определён",
                    }
                    details.append(f"{kind_labels.get(kind, 'тип потери деталей не определён')} ({conf * 100:.0f}%)")
                local_metric = metrics.get("local_sharpness")
                if local_metric is not None and isinstance(local_metric.raw_value, dict):
                    soft_pct = float(local_metric.raw_value.get("soft_area_pct", 0.0))
                    if soft_pct >= 8.0:
                        details.append(f"локальная карта: около {soft_pct:.0f}% информативной площади мягкие")
                face_metric = metrics.get("faces")
                if face_metric is not None and isinstance(face_metric.raw_value, dict):
                    face_count = int(face_metric.raw_value.get("face_count", 0) or 0)
                    if face_count > 0 and face_metric.normalized_value is not None and face_metric.normalized_value < 65:
                        details.append(f"лиц для локальной проверки: {face_count}")
                eye_metric = metrics.get("eyes")
                if eye_metric is not None and isinstance(eye_metric.raw_value, dict):
                    eye_count = int(eye_metric.raw_value.get("eye_count", 0) or 0)
                    if eye_count > 0 and eye_metric.normalized_value is not None and eye_metric.normalized_value < 65:
                        details.append(f"глаз для проверки в 100%: {eye_count}")
                suffix = ("; " + "; ".join(details)) if details else ""
                base_text = base.diagnosis.rstrip(" .;")
                lines.append(f"• Резкость [{base.status}]: {base_text}{suffix}.")
        else:
            lines.append("По текущим базовым метрикам критических технических проблем не найдено.")
        if features:
            lines.append("")
            lines.append("Особенности изображения, которые не считаем дефектом автоматически:")
            lines.extend(f"• {r.label}: {r.diagnosis}" for r in features)
        self.result_text.setPlainText("\n".join(lines))
        self.result_text.verticalScrollBar().setValue(0)

    def _show_metrics(self, metrics) -> None:
        rows = build_display_metrics(metrics)
        self.metrics.setRowCount(len(rows))
        status_colors = {
            "Хорошо": QColor(224, 244, 231),
            "Норма": QColor(236, 244, 250),
            "Внимание": QColor(255, 244, 204),
            "Проблема": QColor(255, 224, 224),
            "Проверить": QColor(244, 235, 255),
            "Особенность": QColor(235, 239, 244),
        }
        for row_index, row in enumerate(rows):
            score = "—" if row.score is None else f"{row.score:.1f}/100"
            conf = f"{row.confidence * 100:.0f}%"
            for col, text in enumerate((row.status, row.label, score, conf, row.diagnosis)):
                item = QTableWidgetItem(text)
                if col in (0, 2, 3):
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 0 and row.status in status_colors:
                    item.setBackground(QBrush(status_colors[row.status]))
                    item.setForeground(QBrush(QColor(32, 32, 32)))
                if col == 3:
                    item.setToolTip("Уверенность текущего алгоритма именно в этой оценке, а не качество фотографии.")
                self.metrics.setItem(row_index, col, item)
        # Row heights must be calculated only after all rows have been inserted.
        # ResizeToContents changes column widths while the table is being filled;
        # calculating each row immediately leaves earlier rows with stale heights.
        self.metrics.resizeRowsToContents()
        self.metrics_resize_timer.start(0)

    def _show_histogram(self, metrics) -> None:
        metric = metrics.get("histogram")
        raw = metric.raw_value if metric is not None else None
        if not isinstance(raw, dict):
            self.histogram.set_histogram([], {})
            self.histogram_summary.setText("<b>Экспозиция:</b> —")
            self.histogram_info.setText("Процентили: —")
            return

        diagnostic = build_histogram_diagnostic(metrics)
        self.histogram.set_histogram(raw.get("bins", []), diagnostic.percentiles)
        values = []
        for key in ("p1", "p5", "p50", "p95", "p99"):
            if key in diagnostic.percentiles:
                values.append(f"{key.upper()}={diagnostic.percentiles[key]:.3f}")

        self.histogram_summary.setText(
            f"<b>{diagnostic.headline}</b><br>{diagnostic.summary}"
        )
        self.histogram_info.setText(
            "Процентили линейной яркости: " + "   ".join(values)
            + "<br><span style='color:gray'>Шкала 0…1 относится к линейному свету, а не к обычным значениям RGB 0…255. "
              "Пунктирные P5/P50/P95 показывают положение основной массы тонов; красный пунктир у краёв — пороги клиппинга анализатора.</span>"
        )
        self.histogram_info.setTextFormat(Qt.RichText)

    def _show_recommendations(self, metrics) -> None:
        groups = build_recommendation_groups(metrics)
        for widget, values in (
            (self.safe_recommendations, groups.safe),
            (self.caution_recommendations, groups.caution),
            (self.avoid_recommendations, groups.avoid),
        ):
            widget.clear()
            for text in values:
                widget.addItem(text)

    def _main_subject_region(self) -> dict[str, float] | None:
        if not isinstance(self.current_metrics, dict):
            return None
        metric = self.current_metrics.get("main_subject")
        raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
        confidence = float(raw.get("confidence", getattr(metric, "confidence", 0.0)) or 0.0) if isinstance(raw, dict) else 0.0
        kind = str(raw.get("subject_kind", "unknown")) if isinstance(raw, dict) else "unknown"
        if confidence < 0.50 or kind == "unknown":
            return None
        box = raw.get("protection_box_norm") or raw.get("box_norm")
        if not isinstance(box, dict):
            return None
        try:
            region = {key: float(box.get(key, 0.0) or 0.0) for key in ("x", "y", "w", "h")}
        except (TypeError, ValueError):
            return None
        if region["w"] <= 0.0 or region["h"] <= 0.0:
            return None
        return region

    def _use_main_subject_region(self, key: str) -> None:
        region = self._main_subject_region()
        if not key or region is None:
            self.status_label.setText("Надёжная область главного объекта для этого кадра не определена")
            return
        self.correction_regions[str(key)] = dict(region)
        if isinstance(self.current_metrics, dict):
            self._show_plan(self.current_metrics)
        self._update_correction_controls(refresh_preview=True)
        self.status_label.setText("Коррекция ограничена областью главного объекта")

    def _show_plan(self, metrics) -> None:
        metric = metrics.get("decision_plan")
        raw = metric.raw_value if metric is not None else None
        items = raw.get("items", []) if isinstance(raw, dict) else []
        if not isinstance(items, list):
            items = []
        validation_items = [item for item in self.current_validation_items if isinstance(item, dict)]
        if not validation_items:
            validation_metric = metrics.get("recommendation_validation")
            validation_raw = validation_metric.raw_value if validation_metric is not None else None
            raw_items = validation_raw.get("items", []) if isinstance(validation_raw, dict) else []
            validation_items = [item for item in raw_items if isinstance(item, dict)] if isinstance(raw_items, list) else []
        validation_by_key: dict[str, dict] = {
            str(validation.get("action_key", "")): validation for validation in validation_items
        }

        # Standalone ALMAZ modules are manual diagnostic/restoration actions and
        # therefore do not belong to the classical Decision Engine.  Surface them
        # as explicit Plan rows so Denoise/Deblur/JPEG Recovery can be toggled one
        # by one instead of hiding behind Noise/Sharpness/JPEG.
        existing_keys = {str(item.get("key", "")) for item in items if isinstance(item, dict)}
        almaz_titles = {
            "almaz_denoise": "ALMAZ — AI Denoise",
            "almaz_deblur": "ALMAZ — AI Deblur",
            "almaz_jpeg_recovery": "ALMAZ — JPEG Recovery",
        }
        items = list(items)
        for action_key, title in almaz_titles.items():
            validation = validation_by_key.get(action_key)
            if validation is None or action_key in existing_keys or not preview_action_available(validation):
                continue
            params = validation.get("parameters") if isinstance(validation.get("parameters"), dict) else {}
            degradation = float(params.get("degradation_score", 0.0) or 0.0)
            confidence = float(validation.get("confidence", 0.0) or 0.0)
            items.append({
                "key": action_key,
                "title": title,
                "decision": "review",
                "severity": float(np.clip(degradation * 100.0, 0.0, 100.0)),
                "repairability": 60.0,
                "confidence": confidence,
                "priority": float(np.clip(25.0 + degradation * 35.0, 0.0, 100.0)),
                "reason": "Отдельный ручной модуль ALMAZ. Можно включить самостоятельно для сравнения и диагностики артефактов.",
                "guardrail": "Не применять автоматически; сравнивать результат в 100%, особенно лица, цвет и повреждённые области.",
            })

        selected_keys = self._selected_correction_keys()
        self._updating_plan_choices = True
        self.plan_table.setSortingEnabled(False)
        self.plan_table.setRowCount(len(items))
        decision_labels = {
            "fix": "Исправить",
            "review": "Проверить",
            "preserve": "Сохранить",
            "skip": "Не трогать",
        }
        decision_colors = {
            "fix": QColor(224, 244, 231),
            "review": QColor(255, 244, 204),
            "preserve": QColor(235, 239, 255),
            "skip": QColor(238, 238, 238),
        }
        try:
            for row, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                decision = str(item.get("decision", "review"))
                key = str(item.get("key", ""))
                title = str(item.get("title", key or "—"))
                priority = float(item.get("priority", 0.0) or 0.0)
                severity = float(item.get("severity", 0.0) or 0.0)
                repairability = float(item.get("repairability", 0.0) or 0.0)
                confidence = float(item.get("confidence", 0.0) or 0.0)
                reason = str(item.get("reason", ""))
                guardrail = str(item.get("guardrail", ""))
                validation = validation_by_key.get(key)

                apply_item = QTableWidgetItem("")
                if validation is not None and preview_action_available(validation):
                    apply_item.setFlags((apply_item.flags() | Qt.ItemIsUserCheckable) & ~Qt.ItemIsEditable)
                    apply_item.setCheckState(Qt.Checked if key in selected_keys else Qt.Unchecked)
                    apply_item.setData(Qt.UserRole, key)
                    if key == "surface_defects":
                        apply_item.setToolTip("Дефекты поверхности применяются только по вашему явному выбору на вкладке «Дефекты». AI не включает лечение автоматически.")
                    elif bool(validation.get("accepted", False)):
                        apply_item.setToolTip("Проверка безопасности подтвердила эту коррекцию. Галочку можно снять для сравнения.")
                    else:
                        apply_item.setToolTip("Ручной эксперимент: проверка безопасности не подтвердила автоматическое применение.")
                else:
                    if decision in {"fix", "review"}:
                        apply_item.setText("Нет")
                        apply_item.setToolTip(
                            "Для этой найденной проблемы пока нет исполняемой безопасной коррекции. "
                            "Это диагностический пункт, а не скрытая отключённая галочка."
                        )
                        apply_item.setForeground(QBrush(QColor(196, 120, 0)))
                    else:
                        apply_item.setText("—")
                    apply_item.setFlags(apply_item.flags() & ~Qt.ItemIsEnabled)
                self.plan_table.setItem(row, 0, apply_item)

                first = QTableWidgetItem(decision_labels.get(decision, decision))
                if decision in decision_colors:
                    first.setBackground(QBrush(decision_colors[decision]))
                    first.setForeground(QBrush(QColor(32, 32, 32)))
                self.plan_table.setItem(row, 1, first)

                task_item = QTableWidgetItem(title)
                task_item.setData(Qt.UserRole, key)
                task_item.setData(Qt.UserRole + 10, reason)
                task_item.setData(Qt.UserRole + 11, guardrail)
                self.plan_table.setItem(row, 2, task_item)

                suggestion = self.correction_ai_suggestions.get(key, {})
                if validation is not None and preview_action_available(validation) and bool(validation.get("adjustable", False)):
                    strength_box = QWidget(self.plan_table)
                    strength_layout = QHBoxLayout(strength_box)
                    strength_layout.setContentsMargins(3, 0, 3, 0)
                    slider = QSlider(Qt.Horizontal, strength_box)
                    slider.setRange(0, 100)
                    slider.setSingleStep(5)
                    slider.setPageStep(10)
                    slider.setMinimumWidth(105)
                    current_strength = float(self.correction_strengths.get(key, float(validation.get("default_strength", 1.0) or 1.0)))
                    slider.setValue(int(round(np.clip(current_strength, 0.0, 1.0) * 100.0)))
                    value_label = QLabel(f"{slider.value()}%", strength_box)
                    value_label.setMinimumWidth(38)
                    value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    ai_reset_btn = QPushButton("↺", strength_box)
                    ai_reset_btn.setFixedWidth(28)
                    ai_reset_btn.setToolTip("Вернуть силу, предложенную Рекомендателем параметров ИИ")
                    slider.setToolTip("Сила, которая реально будет применена к фотографии. Это интенсивность эффекта, а не уверенность ИИ.")
                    slider.valueChanged.connect(
                        lambda value, action_key=key, label=value_label: self._on_plan_strength_changed(action_key, value, label)
                    )
                    ai_reset_btn.clicked.connect(
                        lambda _checked=False, action_key=key, slider_ref=slider, label=value_label: self._reset_plan_strength_to_ai(action_key, slider_ref, label)
                    )
                    strength_layout.addWidget(slider, 1)
                    strength_layout.addWidget(value_label)
                    strength_layout.addWidget(ai_reset_btn)
                    self.plan_table.setCellWidget(row, 3, strength_box)

                    if suggestion:
                        ai_conf = int(round(np.clip(float(suggestion.get("confidence", 0.0) or 0.0), 0.0, 1.0) * 100.0))
                        ai_conf_item = NumericTableWidgetItem(f"{ai_conf}%", ai_conf)
                        ai_conf_item.setTextAlignment(Qt.AlignCenter)
                        ai_conf_item.setToolTip(
                            "Персональная подстройка предложенной силы по накопленным пользовательским решениям. "
                            "Не путать с уверенностью анализа в наличии проблемы."
                        )
                        self.plan_table.setItem(row, 4, ai_conf_item)
                    else:
                        ai_item = QTableWidgetItem("—")
                        ai_item.setTextAlignment(Qt.AlignCenter)
                        ai_item.setFlags(ai_item.flags() & ~Qt.ItemIsEnabled)
                        self.plan_table.setItem(row, 4, ai_item)
                else:
                    for col in (3, 4):
                        strength_item = QTableWidgetItem("—")
                        strength_item.setTextAlignment(Qt.AlignCenter)
                        strength_item.setFlags(strength_item.flags() & ~Qt.ItemIsEnabled)
                        self.plan_table.setItem(row, col, strength_item)

                # A QTableWidget cell may keep both a previous QTableWidgetItem and
                # a newly installed cell widget. Clear both explicitly so stale text
                # cannot bleed through underneath the buttons.
                self.plan_table.removeCellWidget(row, 5)
                self.plan_table.takeItem(row, 5)

                if validation is not None and preview_action_available(validation):
                    if key == "red_eye":
                        area_box = QWidget(self.plan_table)
                        area_layout = QHBoxLayout(area_box)
                        area_layout.setContentsMargins(2, 0, 2, 0)
                        area_layout.setSpacing(4)
                        area_label = QLabel("Локально", area_box)
                        red_raw = {}
                        red_metric = metrics.get("red_eye") if isinstance(metrics, dict) else None
                        if red_metric is not None and isinstance(red_metric.raw_value, dict):
                            red_raw = red_metric.raw_value
                        suspicious_count = int(red_raw.get("suspicious_eye_count", 0) or 0)
                        show_btn = QPushButton(f"Показать ({suspicious_count})", area_box)
                        show_btn.setEnabled(suspicious_count > 0)
                        show_btn.setToolTip(
                            "Показать на фотографии рамки глаз, которые детектор счёл подозрительными. "
                            "Рамка показывает место срабатывания, а не доказывает наличие красного глаза."
                        )
                        show_btn.clicked.connect(lambda _checked=False: self._show_red_eye_candidates_on_image())
                        area_layout.addWidget(area_label)
                        area_layout.addWidget(show_btn)
                        self.plan_table.setCellWidget(row, 5, area_box)
                    elif key == "surface_defects":
                        area_item = QTableWidgetItem("Локально")
                        area_item.setTextAlignment(Qt.AlignCenter)
                        area_item.setToolTip("Эта коррекция ограничена найденными кандидатами на дефекты поверхности.")
                        area_item.setFlags(area_item.flags() & ~Qt.ItemIsEnabled)
                        self.plan_table.setItem(row, 5, area_item)
                    elif key == "super_resolution":
                        area_item = QTableWidgetItem("Всё фото ×2")
                        area_item.setTextAlignment(Qt.AlignCenter)
                        area_item.setToolTip("ALMAZ x2 меняет размер всего изображения; локальная область для увеличения не поддерживается.")
                        area_item.setFlags(area_item.flags() & ~Qt.ItemIsEnabled)
                        self.plan_table.setItem(row, 5, area_item)
                    elif str(validation.get("candidate", "")).startswith("almaz_ai_"):
                        area_item = QTableWidgetItem("Всё фото · ALMAZ")
                        area_item.setTextAlignment(Qt.AlignCenter)
                        area_item.setToolTip("ALMAZ AI Denoise/Deblur/JPEG Recovery в текущей версии применяется ко всему кадру; локальный AI-масочный режим пока намеренно отключён.")
                        area_item.setFlags(area_item.flags() & ~Qt.ItemIsEnabled)
                        self.plan_table.setItem(row, 5, area_item)
                    elif key == "auto_tone_color" or (key == "white_balance" and str(validation.get("candidate", "")) != "spatial_white_balance_v1"):
                        area_item = QTableWidgetItem("Всё фото")
                        area_item.setTextAlignment(Qt.AlignCenter)
                        area_item.setToolTip(
                            "Баланс белого рассчитан как глобальная коррекция источника света и не применяется локально."
                            if key == "white_balance" else
                            "Комбинированный автотон/контраст/цвет рассчитан как глобальная поканальная коррекция и не применяется локально."
                        )
                        area_item.setFlags(area_item.flags() & ~Qt.ItemIsEnabled)
                        self.plan_table.setItem(row, 5, area_item)
                    else:
                        area_box = QWidget(self.plan_table)
                        area_layout = QHBoxLayout(area_box)
                        area_layout.setContentsMargins(2, 0, 2, 0)
                        area_layout.setSpacing(4)
                        area_box.setMinimumWidth(235)
                        spatial_candidate = str(validation.get("candidate", "")) in {
                            "spatial_exposure_v1", "spatial_contrast_v1", "spatial_sharpness_v1", "spatial_denoise_v1", "spatial_white_balance_v1"
                        }
                        if key in self.correction_regions:
                            area_text = "Область ✓"
                        elif spatial_candidate:
                            area_text = "Авто-маска"
                        else:
                            area_text = "Всё фото"
                        area_btn = QPushButton(area_text, area_box)
                        area_btn.setMinimumWidth(92)
                        if key in self.correction_regions:
                            area_tip = "Автоматическая коррекция дополнительно ограничена выбранной вами областью. Нажмите, чтобы выбрать область заново."
                        elif spatial_candidate:
                            area_tip = (
                                "Коррекция уже применяется только к областям, которые локальная карта признала проблемными. "
                                "Нажмите, если хотите дополнительно ограничить авто-маску своей областью."
                            )
                        else:
                            area_tip = "Сейчас коррекция применяется ко всему фото. Нажмите, чтобы мышью обвести свою область."
                        area_btn.setToolTip(area_tip)
                        area_btn.clicked.connect(lambda _checked=False, action_key=key: self._start_correction_region_selection(action_key))
                        subject_region = self._main_subject_region()
                        subject_area_btn = QPushButton("Главный", area_box)
                        subject_area_btn.setMinimumWidth(82)
                        subject_area_btn.setEnabled(subject_region is not None)
                        subject_area_btn.setToolTip(
                            "Ограничить эту коррекцию автоматически определённым главным объектом."
                            if subject_region is not None else "Главный объект определён недостаточно уверенно; выберите область вручную."
                        )
                        subject_area_btn.clicked.connect(lambda _checked=False, action_key=key: self._use_main_subject_region(action_key))
                        clear_btn = QPushButton("×", area_box)
                        clear_btn.setFixedWidth(24)
                        clear_btn.setEnabled(key in self.correction_regions)
                        clear_btn.setToolTip(
                            "Сбросить ручное ограничение и вернуться к автоматической маске"
                            if spatial_candidate else "Сбросить область и снова применять коррекцию ко всему фото"
                        )
                        clear_btn.clicked.connect(lambda _checked=False, action_key=key: self._clear_correction_region(action_key))
                        area_layout.addWidget(area_btn)
                        area_layout.addWidget(subject_area_btn)
                        area_layout.addWidget(clear_btn)
                        self.plan_table.setCellWidget(row, 5, area_box)
                else:
                    area_item = QTableWidgetItem("—")
                    area_item.setTextAlignment(Qt.AlignCenter)
                    area_item.setFlags(area_item.flags() & ~Qt.ItemIsEnabled)
                    self.plan_table.setItem(row, 5, area_item)

                self.plan_table.setItem(row, 6, NumericTableWidgetItem(f"{priority:.1f}", priority))
                self.plan_table.setItem(row, 7, NumericTableWidgetItem(f"{severity:.1f}", severity))
                self.plan_table.setItem(row, 8, NumericTableWidgetItem(f"{repairability:.1f}", repairability))
                analysis_conf_item = NumericTableWidgetItem(f"{confidence * 100:.0f}%", confidence * 100.0)
                analysis_conf_item.setToolTip(
                    "Уверенность анализа в том, что описанная проблема действительно присутствует. "
                    "Это не сила коррекции и не уверенность Рекомендателя параметров ИИ."
                )
                self.plan_table.setItem(row, 9, analysis_conf_item)

                if validation is not None:
                    accepted = bool(validation.get("accepted", False))
                    target_before = validation.get("target_before")
                    target_after = validation.get("target_after")
                    message = str(validation.get("message", ""))
                    if accepted and target_before is not None and target_after is not None:
                        validation_text = f"Подтверждено: {float(target_before):.3f} → {float(target_after):.3f}. {message}"
                        validation_short = "Подтверждено"
                    elif accepted:
                        validation_text = "Подтверждено. " + message
                        validation_short = "Подтверждено"
                    else:
                        validation_text = "Не подтверждено. " + message
                        validation_short = "Вручную" if preview_action_available(validation) else "Не подтверждено"
                elif decision == "fix":
                    validation_text = "Проверка безопасности не запускалась для этого типа исправления."
                    validation_short = "Нет автопроверки"
                elif decision == "review":
                    validation_text = "Требуется ручная проверка."
                    validation_short = "Проверить"
                elif decision == "preserve":
                    validation_text = "Ограничение безопасности: сохранять."
                    validation_short = "Сохранить"
                else:
                    validation_text = "Проверка не требуется."
                    validation_short = "—"
                status_item = QTableWidgetItem(validation_short)
                status_item.setToolTip(validation_text)
                task_item.setData(Qt.UserRole + 12, validation_text)
                self.plan_table.setItem(row, 10, status_item)
        finally:
            self._updating_plan_choices = False
            self.plan_table.setSortingEnabled(True)

        self.plan_table.sortItems(6, Qt.DescendingOrder)
        if self.plan_table.rowCount():
            self.plan_table.selectRow(0)
            self._show_plan_details(0, 0, -1, -1)
        else:
            self.plan_details.setText("План действий для этого кадра пуст.")

    def _show_plan_details(self, current_row: int, _current_col: int, _previous_row: int, _previous_col: int) -> None:
        if current_row < 0 or current_row >= self.plan_table.rowCount():
            self.plan_details.setText("Выберите строку плана.")
            return
        task_item = self.plan_table.item(current_row, 2)
        if task_item is None:
            self.plan_details.setText("Выберите строку плана.")
            return
        reason = str(task_item.data(Qt.UserRole + 10) or "—")
        guardrail = str(task_item.data(Qt.UserRole + 11) or "—")
        validation = str(task_item.data(Qt.UserRole + 12) or "—")
        priority = self.plan_table.item(current_row, 6)
        severity = self.plan_table.item(current_row, 7)
        repairability = self.plan_table.item(current_row, 8)
        confidence = self.plan_table.item(current_row, 9)
        technical = " · ".join(
            part for part in (
                f"приоритет {priority.text()}" if priority is not None else "",
                f"важность {severity.text()}" if severity is not None else "",
                f"исправимость {repairability.text()}" if repairability is not None else "",
                f"уверенность {confidence.text()}" if confidence is not None else "",
            ) if part
        )
        self.plan_details.setText(
            f"<b>Почему:</b> {reason}<br><b>Ограничение:</b> {guardrail}<br>"
            f"<b>Проверка:</b> {validation}<br><span style='color:#777'><b>Технические оценки:</b> {technical or '—'}</span>"
        )

    def _fix_list_item_for_key(self, key: str) -> QListWidgetItem | None:
        for row in range(self.fix_actions_list.count()):
            item = self.fix_actions_list.item(row)
            if str(item.data(Qt.UserRole) or "") == key:
                return item
        return None

    def _on_plan_strength_changed(self, key: str, value: int, label: QLabel | None = None) -> None:
        if not key:
            return
        value = max(0, min(100, int(value)))
        self.correction_strengths[str(key)] = value / 100.0
        if label is not None:
            label.setText(f"{value}%")
        self._update_correction_controls(refresh_preview=True)

    def _reset_plan_strength_to_ai(self, key: str, slider: QSlider | None = None, label: QLabel | None = None) -> None:
        suggestion = self.correction_ai_suggestions.get(str(key), {})
        if not suggestion:
            return
        try:
            value = int(round(float(suggestion.get("suggested_strength", 0.0)) * 100.0))
        except (TypeError, ValueError):
            return
        value = max(0, min(100, value))
        if slider is not None:
            slider.setValue(value)
        else:
            self.correction_strengths[str(key)] = value / 100.0
            if label is not None:
                label.setText(f"{value}%")
            self._update_correction_controls(refresh_preview=True)

    def _on_plan_correction_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_plan_choices or item.column() != 0:
            return
        key = str(item.data(Qt.UserRole) or "")
        if not key:
            return
        source_item = self._fix_list_item_for_key(key)
        if source_item is None:
            return
        self._updating_correction_choices = True
        try:
            source_item.setCheckState(item.checkState())
        finally:
            self._updating_correction_choices = False
        self._enforce_auto_trio_exclusivity(key)
        self._update_correction_controls(refresh_preview=True)

    def _sync_plan_corrections_from_result(self) -> None:
        if not hasattr(self, "plan_table"):
            return
        selected = self._selected_correction_keys()
        self._updating_plan_choices = True
        try:
            for row in range(self.plan_table.rowCount()):
                item = self.plan_table.item(row, 0)
                if item is None:
                    continue
                key = str(item.data(Qt.UserRole) or "")
                if key:
                    item.setCheckState(Qt.Checked if key in selected else Qt.Unchecked)
        finally:
            self._updating_plan_choices = False

    def _show_faces(self, metrics) -> None:
        metric = metrics.get("faces")
        raw = metric.raw_value if metric is not None else None
        faces = raw.get("faces", []) if isinstance(raw, dict) else []
        if not isinstance(faces, list):
            faces = []

        eye_metric = metrics.get("eyes")
        eye_raw = eye_metric.raw_value if eye_metric is not None else None
        eyes = eye_raw.get("eyes", []) if isinstance(eye_raw, dict) else []
        if not isinstance(eyes, list):
            eyes = []
        eyes_by_face: dict[int, list[dict]] = {}
        for eye in eyes:
            if not isinstance(eye, dict):
                continue
            try:
                face_index = int(eye.get("face_index", -1))
            except (TypeError, ValueError):
                continue
            eyes_by_face.setdefault(face_index, []).append(eye)

        self.faces_table.setRowCount(len(faces))
        for row_index, face in enumerate(faces):
            if not isinstance(face, dict):
                continue
            w_pct = float(face.get("w", 0.0)) * 100.0
            h_pct = float(face.get("h", 0.0)) * 100.0
            sharp = float(face.get("sharpness_score", 0.0))
            bright = float(face.get("brightness_linear", 0.0))
            conf = float(face.get("measurement_confidence", 0.0))
            face_eyes = eyes_by_face.get(row_index, [])
            eye_count = len(face_eyes)
            worst_eye = min((float(eye.get("sharpness_score", 0.0)) for eye in face_eyes), default=None)
            if sharp < 45:
                status = "Внимание"
            elif sharp < 65:
                status = "Проверить"
            else:
                status = "Норма"
            if worst_eye is not None and worst_eye < 45:
                status = "Внимание"
            source = "Вручную" if str(face.get("source", "auto")) == "manual" else "Авто"
            values = (
                str(row_index + 1),
                source,
                f"{w_pct:.1f}% × {h_pct:.1f}%",
                f"{sharp:.1f}/100",
                f"{bright:.3f}",
                str(eye_count) if eye_count else "—",
                f"{worst_eye:.1f}/100" if worst_eye is not None else "—",
                f"{conf * 100:.0f}%",
                status,
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.faces_table.setItem(row_index, col, item)

        if hasattr(self, "manual_vision_status"):
            manual_face_count = sum(1 for face in faces if isinstance(face, dict) and str(face.get("source", "auto")) == "manual")
            manual_eye_count = sum(1 for eye in eyes if isinstance(eye, dict) and str(eye.get("source", "auto")) == "manual")
            if manual_face_count or manual_eye_count:
                self.manual_vision_status.setText(
                    f"Ручная разметка: лиц {manual_face_count}, глаз {manual_eye_count}. "
                    "Она сохранена локально и участвует в текущем анализе и проверке красных глаз."
                )
            else:
                self.manual_vision_status.setText("Ручных отметок нет. Используется автоматическое распознавание.")


    def _set_almaz_release_busy(self, busy: bool) -> None:
        for name in ("almaz_install_release_btn", "almaz_rollback_release_btn", "almaz_refresh_btn"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(not busy)

    def _update_almaz_release_elapsed(self) -> None:
        if self.almaz_release_started_at is None:
            return
        elapsed = max(0.0, time.monotonic() - self.almaz_release_started_at)
        stage = self.almaz_release_stage_text or "работаю с моделью"
        self.status_label.setText(f"ALMAZ модели: {stage} · {elapsed:.1f} с")

    def _almaz_release_stage(self, text: str) -> None:
        self.almaz_release_stage_text = str(text).strip() or "работаю с моделью"
        if hasattr(self, "almaz_activity_log"):
            self.almaz_activity_log.append(self.almaz_release_stage_text)
        self._update_almaz_release_elapsed()

    def _start_almaz_release_worker(self, worker: AlmazReleaseWorker) -> None:
        if self.almaz_release_worker is not None and self.almaz_release_worker.isRunning():
            QMessageBox.information(self, "Photo Doctor", "Операция с ALMAZ-моделью уже выполняется.")
            return
        self.almaz_release_worker = worker
        self.almaz_release_started_at = time.monotonic()
        self.almaz_release_stage_text = "начинаю проверку"
        self._set_almaz_release_busy(True)
        self.almaz_release_timer.start()
        worker.stage.connect(self._almaz_release_stage)
        worker.completed.connect(self._almaz_release_completed)
        worker.failed.connect(self._almaz_release_failed)
        worker.finished.connect(self._almaz_release_finished)
        worker.start()

    def _install_almaz_release_bundle(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Установить обученную модель ALMAZ", "",
            "ALMAZ release-bundle (*.zip);;ZIP-архивы (*.zip)",
        )
        if not name:
            return
        if hasattr(self, "almaz_activity_log"):
            self.almaz_activity_log.append(f"Установка ALMAZ release-bundle: {Path(name).name}")
        self._start_almaz_release_worker(AlmazReleaseWorker("install", bundle_path=Path(name)))

    def _rollback_almaz_release_model(self) -> None:
        labels = [
            ("Super Resolution x2", "sr_x2"),
            ("AI Denoise", "denoise"),
            ("AI Deblur", "deblur"),
            ("JPEG Recovery", "jpeg_recovery"),
        ]
        names = [label for label, _task in labels]
        selected, ok = QInputDialog.getItem(self, "Откат ALMAZ", "Какую модель откатить?", names, 0, False)
        if not ok or not selected:
            return
        task = dict(labels)[str(selected)]
        answer = QMessageBox.question(
            self, "Photo Doctor — откат ALMAZ",
            f"Вернуть последнюю проверенную резервную копию «{selected}»?\n\nТекущая модель тоже будет сохранена перед откатом.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        self._start_almaz_release_worker(AlmazReleaseWorker("rollback", task=task))

    def _almaz_release_completed(self, result) -> None:
        elapsed = max(0.0, time.monotonic() - self.almaz_release_started_at) if self.almaz_release_started_at else 0.0
        self.status_label.setText(f"{result.detail} · {elapsed:.1f} с")
        if hasattr(self, "almaz_activity_log"):
            self.almaz_activity_log.append(result.detail)
            if result.backup_dir:
                self.almaz_activity_log.append(f"Backup: {result.backup_dir}")
        self._refresh_almaz_panel()
        QMessageBox.information(self, "Photo Doctor — ALMAZ", result.detail)

    def _almaz_release_failed(self, message: str) -> None:
        elapsed = max(0.0, time.monotonic() - self.almaz_release_started_at) if self.almaz_release_started_at else 0.0
        self.status_label.setText(f"ALMAZ: модель не установлена/не откатана · {elapsed:.1f} с")
        if hasattr(self, "almaz_activity_log"):
            self.almaz_activity_log.append(f"ОШИБКА release: {message}")
        QMessageBox.warning(self, "Photo Doctor — ALMAZ release", str(message))

    def _almaz_release_finished(self) -> None:
        self.almaz_release_timer.stop()
        self.almaz_release_started_at = None
        self.almaz_release_stage_text = ""
        self._set_almaz_release_busy(False)
        worker = self.almaz_release_worker
        self.almaz_release_worker = None
        if worker is not None:
            worker.deleteLater()

    def _refresh_almaz_panel(self) -> None:
        if not hasattr(self, "almaz_sr_state"):
            return
        try:
            status = inspect_almaz_status()
            sr = status.sr if isinstance(status.sr, dict) else {}
            restoration = status.restoration if isinstance(status.restoration, dict) else {}

            def render_state(raw: dict, fallback: str) -> str:
                if bool(raw.get("ready", False)):
                    provider = raw.get("provider_label") or raw.get("provider") or "provider —"
                    model = raw.get("model_id") or "модель"
                    return f"✓ Готово: {model} · {provider}"
                detail = str(raw.get("detail") or fallback)
                return f"○ Не готово: {detail}"

            self.almaz_sr_state.setText(render_state(sr, "x2-модель не установлена"))
            denoise = restoration.get("denoise", {}) if isinstance(restoration.get("denoise", {}), dict) else {}
            deblur = restoration.get("deblur", {}) if isinstance(restoration.get("deblur", {}), dict) else {}
            jpeg = restoration.get("jpeg_recovery", {}) if isinstance(restoration.get("jpeg_recovery", {}), dict) else {}
            self.almaz_denoise_state.setText(render_state(denoise, "Denoise-модель не установлена"))
            self.almaz_deblur_state.setText(render_state(deblur, "Deblur-модель не установлена"))
            self.almaz_jpeg_state.setText(render_state(jpeg, "JPEG Recovery-модель не установлена"))
            self.almaz_provider_state.setText(
                f"Ускоритель: {status.best_provider_label or 'ONNX Runtime / аппаратный provider не обнаружен'} · "
                f"готовых ALMAZ-моделей {status.ready_models}/{status.total_models}"
            )

            prepare_busy = self.almaz_prepare_process is not None and self.almaz_prepare_process.state() != QProcess.ProcessState.NotRunning
            restoration_busy = self.almaz_restoration_process is not None and self.almaz_restoration_process.state() != QProcess.ProcessState.NotRunning
            can_sr = self._can_prepare_almaz_here()
            can_rest = self._can_prepare_almaz_restoration_here()
            self.almaz_sr_prepare_btn.setText("x2 готов" if bool(sr.get("ready", False)) else "Подготовить x2")
            self.almaz_sr_prepare_btn.setEnabled(can_sr and not bool(sr.get("ready", False)) and not prepare_busy and not restoration_busy)
            task_states = {
                "denoise": (self.almaz_denoise_prepare_btn, denoise),
                "deblur": (self.almaz_deblur_prepare_btn, deblur),
                "jpeg_recovery": (self.almaz_jpeg_prepare_btn, jpeg),
            }
            for task, (button, raw) in task_states.items():
                button.setText("Готово" if bool(raw.get("ready", False)) else "Подготовить модель")
                button.setEnabled(can_rest and not bool(raw.get("ready", False)) and not prepare_busy and not restoration_busy)
            self.almaz_prepare_all_btn.setEnabled(can_rest and not prepare_busy and not restoration_busy and status.ready_models < status.total_models)
            self.almaz_runtime_main_btn.setEnabled(self._can_install_ai_runtime_here() and status.best_provider is None and not prepare_busy and not restoration_busy)
            self.almaz_runtime_main_btn.setText("Runtime готов" if status.best_provider is not None else "Установить ONNX Runtime")
        except Exception as exc:
            message = f"Состояние ALMAZ не удалось прочитать: {exc}"
            for label in (self.almaz_sr_state, self.almaz_denoise_state, self.almaz_deblur_state, self.almaz_jpeg_state):
                label.setText(message)
            self.almaz_provider_state.setText(message)

        metrics = self.current_metrics or {}
        sr_metric = metrics.get("super_resolution") if isinstance(metrics, dict) else None
        sr_raw = sr_metric.raw_value if sr_metric is not None and isinstance(sr_metric.raw_value, dict) else {}
        if sr_raw:
            need = float(sr_raw.get("need_score", 0.0) or 0.0)
            recommendation = "рекомендуется x2" if bool(sr_raw.get("recommend", False)) else "x2 не требуется"
            photo_parts = [f"Super Resolution: {recommendation} · потребность {need:.0f}/100."]
            almaz_actions = []
            for item in self.current_validation_items:
                if not isinstance(item, dict):
                    continue
                candidate = str(item.get("candidate", ""))
                if candidate.startswith("almaz_ai_") or str(item.get("action_key", "")) == "super_resolution":
                    almaz_actions.append(self._correction_action_label(item))
            if almaz_actions:
                photo_parts.append("Доступно в «Исправлениях»: " + ", ".join(almaz_actions) + ".")
            else:
                photo_parts.append("ALMAZ-коррекции для этого фото сейчас не предложены; это нормально, они не должны включаться на каждом кадре.")
            self.almaz_current_photo.setText(" ".join(photo_parts))
        else:
            self.almaz_current_photo.setText(
                "Фото ещё не анализировалось. После анализа здесь появится рекомендация x2 и состояние ALMAZ-коррекций."
            )

    def _can_prepare_almaz_restoration_here(self) -> bool:
        if not self._can_install_ai_runtime_here():
            return False
        script = Path(__file__).resolve().parents[3] / "tools" / "prepare_almaz_nafnet.py"
        return script.is_file()

    def _can_install_ai_runtime_here(self) -> bool:
        if getattr(sys, "frozen", False):
            return False
        base_prefix = getattr(sys, "base_prefix", sys.prefix)
        return bool(sys.prefix != base_prefix)

    def _install_ai_runtime(self) -> None:
        if not self._can_install_ai_runtime_here():
            QMessageBox.information(
                self,
                "Photo Doctor",
                "Установка из интерфейса доступна только при запуске через START_PhotoDoctor.bat из локального .venv. "
                "В автономной EXE-сборке ONNX Runtime должен быть включён при сборке.",
            )
            return
        if self.ai_install_process is not None and self.ai_install_process.state() != QProcess.ProcessState.NotRunning:
            return

        variants = [
            ("CPU — универсально и без дополнительных драйверов", "onnxruntime>=1.20,<2"),
        ]
        if sys.platform.startswith("win"):
            variants.extend([
                ("NVIDIA CUDA 12 — для GeForce/RTX", "onnxruntime-gpu>=1.20,<2"),
                ("DirectML — универсальный DirectX 12 GPU", "onnxruntime-directml>=1.20,<2"),
                ("Intel OpenVINO — CPU/iGPU/dGPU/NPU Intel", "onnxruntime-openvino>=1.20,<2"),
            ])
        labels = [label for label, _package in variants]
        selected, ok = QInputDialog.getItem(
            self, "ONNX Runtime для ALMAZ", "Выберите исполнитель. В одном окружении должен быть установлен один вариант ONNX Runtime:",
            labels, 0, False,
        )
        if not ok or not selected:
            return
        package = dict(variants)[selected]
        self.ai_install_log.clear()
        self.ai_install_log.setVisible(True)
        self.ai_install_log.append(f"Установка: {selected}…")
        self.ai_runtime_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Выполняется…")
        self.status_label.setText("⏳ Установка ONNX Runtime…")
        process = QProcess(self)
        self.ai_install_process = process
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_ai_runtime_install_output)
        process.finished.connect(self._ai_runtime_install_finished)
        process.setProgram(sys.executable)
        process.setArguments([
            "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", package,
        ])
        process.start()

    def _read_ai_runtime_install_output(self) -> None:
        process = self.ai_install_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.ai_install_log.moveCursor(QTextCursor.MoveOperation.End)
            self.ai_install_log.insertPlainText(data)
            self.ai_install_log.moveCursor(QTextCursor.MoveOperation.End)
            if hasattr(self, "almaz_activity_log"):
                self.almaz_activity_log.moveCursor(QTextCursor.MoveOperation.End)
                self.almaz_activity_log.insertPlainText(data)
                self.almaz_activity_log.moveCursor(QTextCursor.MoveOperation.End)

    def _ai_runtime_install_finished(self, exit_code: int, exit_status) -> None:
        self._read_ai_runtime_install_output()
        success = int(exit_code) == 0
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)
        self.progress.setFormat("Готово" if success else "Ошибка")
        if success:
            importlib.invalidate_caches()
            self.ai_install_log.append("\nONNX Runtime установлен. Обновляю состояние…")
            self.status_label.setText("✓ ONNX Runtime установлен")
        else:
            self.ai_install_log.append(f"\nУстановка завершилась с ошибкой, код {exit_code}.")
            self.status_label.setText("Ошибка: не удалось установить ONNX Runtime")
        self.ai_install_process = None
        self._refresh_ai_status()
        self._refresh_almaz_panel()

    @staticmethod
    def _set_almaz_process_utf8(process: QProcess) -> None:
        """Force UTF-8 for model-preparation Python and every child it spawns."""
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUTF8", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)

    def _begin_almaz_prepare_activity(self, message: str) -> None:
        self.almaz_prepare_started_at = time.monotonic()
        self.almaz_prepare_stage_text = str(message)
        self.almaz_prepare_timer.start()
        self._update_almaz_prepare_elapsed()

    def _set_almaz_prepare_stage(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        if len(text) > 180:
            text = text[:177] + "..."
        self.almaz_prepare_stage_text = text
        self._update_almaz_prepare_elapsed()

    def _update_almaz_prepare_elapsed(self) -> None:
        if self.almaz_prepare_started_at is None or not self.almaz_prepare_stage_text:
            return
        elapsed = max(0.0, time.monotonic() - self.almaz_prepare_started_at)
        self.status_label.setText(f"ALMAZ: {self.almaz_prepare_stage_text}  ·  {elapsed:.1f} с")

    def _end_almaz_prepare_activity(self) -> float:
        elapsed = 0.0 if self.almaz_prepare_started_at is None else max(0.0, time.monotonic() - self.almaz_prepare_started_at)
        self.almaz_prepare_timer.stop()
        self.almaz_prepare_started_at = None
        self.almaz_prepare_stage_text = ""
        return elapsed

    @staticmethod
    def _last_process_stage(data: str) -> str:
        lines = [line.strip() for line in str(data).splitlines() if line.strip()]
        return lines[-1] if lines else ""

    def _can_prepare_almaz_here(self) -> bool:
        if not self._can_install_ai_runtime_here():
            return False
        script = Path(__file__).resolve().parents[3] / "tools" / "prepare_almaz_swinir.py"
        return script.is_file()

    def _prepare_almaz_x2(self) -> None:
        if not self._can_prepare_almaz_here():
            QMessageBox.information(
                self,
                "Photo Doctor",
                "Подготовка ALMAZ x2 из интерфейса доступна только в исходной версии, запущенной из локального .venv. "
                "В автономный EXE verified ONNX-модель должна быть включена заранее.",
            )
            return
        if self.almaz_prepare_process is not None and self.almaz_prepare_process.state() != QProcess.ProcessState.NotRunning:
            return
        answer = QMessageBox.question(
            self,
            "Подготовить ALMAZ x2",
            "Будет скачан официальный SwinIR-S x2, затем Photo Doctor проверит его размер и SHA-256, "
            "экспортирует ONNX и примет модель только после PyTorch↔ONNX parity=PASS. Продолжить?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        script = Path(__file__).resolve().parents[3] / "tools" / "prepare_almaz_swinir.py"
        self.ai_install_log.clear()
        self.ai_install_log.setVisible(True)
        self.ai_install_log.append("Подготовка ALMAZ x2…\n")
        if hasattr(self, "almaz_activity_log"):
            self.almaz_activity_log.clear()
            self.almaz_activity_log.append("Подготовка ALMAZ x2…\n")
        self.almaz_prepare_btn.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("Выполняется…")
        self._begin_almaz_prepare_activity("x2: подготавливаю окружение и проверяю зависимости")
        process = QProcess(self)
        self.almaz_prepare_process = process
        self._set_almaz_process_utf8(process)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_almaz_prepare_output)
        process.finished.connect(self._almaz_prepare_finished)
        process.setProgram(sys.executable)
        process.setArguments([str(script)])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        process.start()

    def _read_almaz_prepare_output(self) -> None:
        process = self.almaz_prepare_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            stage = self._last_process_stage(data)
            if stage:
                self._set_almaz_prepare_stage(f"x2: {stage}")
            self.ai_install_log.moveCursor(QTextCursor.MoveOperation.End)
            self.ai_install_log.insertPlainText(data)
            self.ai_install_log.moveCursor(QTextCursor.MoveOperation.End)
            if hasattr(self, "almaz_activity_log"):
                self.almaz_activity_log.moveCursor(QTextCursor.MoveOperation.End)
                self.almaz_activity_log.insertPlainText(data)
                self.almaz_activity_log.moveCursor(QTextCursor.MoveOperation.End)

    def _almaz_prepare_error_detail(self) -> str:
        parts: list[str] = []
        for widget_name in ("almaz_activity_log", "ai_install_log"):
            widget = getattr(self, widget_name, None)
            if widget is None or not hasattr(widget, "toPlainText"):
                continue
            text = str(widget.toPlainText() or "").strip()
            if text:
                parts.extend(line.rstrip() for line in text.splitlines() if line.strip())
                break
        tail = parts[-12:]
        return "\n".join(tail)

    def _almaz_prepare_finished(self, exit_code: int, exit_status) -> None:
        self._read_almaz_prepare_output()
        elapsed = self._end_almaz_prepare_activity()
        success = int(exit_code) == 0
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)
        self.progress.setFormat("Готово" if success else "Ошибка")
        if success:
            self.ai_install_log.append("\nALMAZ x2 подготовлен и установлен. Обновляю диагностику…")
            self.status_label.setText(f"ALMAZ x2 готов · {elapsed:.1f} с")
        else:
            self.ai_install_log.append(f"\nПодготовка ALMAZ x2 завершилась с ошибкой, код {exit_code}.")
            self.status_label.setText(f"Ошибка: ALMAZ x2 не подготовлен · {elapsed:.1f} с")
            detail = self._almaz_prepare_error_detail()
            QMessageBox.critical(
                self,
                "ALMAZ — ошибка подготовки модели",
                f"Подготовка ALMAZ x2 завершилась с кодом {exit_code}.\n\n"
                + (detail or "Подробный лог отсутствует. Проверьте доступ к интернету и состояние .venv."),
            )
        self.almaz_prepare_process = None
        self._refresh_ai_status()
        self._refresh_almaz_panel()

    def _prepare_almaz_restoration(self, task: str) -> None:
        task = str(task).strip().lower()
        labels = {
            "denoise": "AI Denoise",
            "deblur": "AI Deblur",
            "jpeg_recovery": "JPEG Recovery",
            "all": "Denoise + Deblur + JPEG Recovery",
        }
        if task not in labels:
            QMessageBox.warning(self, "Photo Doctor", f"Неизвестная ALMAZ-задача: {task}")
            return
        if not self._can_prepare_almaz_restoration_here():
            QMessageBox.information(
                self,
                "Photo Doctor",
                "Подготовка ALMAZ Restoration из интерфейса доступна только в исходной версии, запущенной из локального .venv. "
                "В автономный EXE verified ONNX-модели должны быть включены заранее.",
            )
            return
        if self.almaz_restoration_process is not None and self.almaz_restoration_process.state() != QProcess.ProcessState.NotRunning:
            return
        if self.almaz_prepare_process is not None and self.almaz_prepare_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Photo Doctor", "Сначала дождитесь завершения подготовки ALMAZ x2.")
            return
        answer = QMessageBox.question(
            self,
            f"Подготовить {labels[task]}",
            "Photo Doctor скачает закреплённый NAFNet checkpoint, проверит размер и SHA-256, "
            "экспортирует ONNX и установит модель только после parity=PASS. Продолжить?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        script = Path(__file__).resolve().parents[3] / "tools" / "prepare_almaz_nafnet.py"
        self.almaz_restoration_task = task
        self.almaz_activity_log.clear()
        self.almaz_activity_log.append(f"Подготовка ALMAZ {labels[task]}…\n")
        if hasattr(self, "ai_install_log"):
            self.ai_install_log.clear()
            self.ai_install_log.setVisible(True)
            self.ai_install_log.append(f"Подготовка ALMAZ {labels[task]}…\n")
        self.progress.setRange(0, 0)
        self.progress.setFormat("Выполняется…")
        self._begin_almaz_prepare_activity(f"{labels[task]}: подготавливаю окружение и проверяю зависимости")
        process = QProcess(self)
        self.almaz_restoration_process = process
        self._set_almaz_process_utf8(process)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_almaz_restoration_output)
        process.finished.connect(self._almaz_restoration_finished)
        process.setProgram(sys.executable)
        process.setArguments([str(script), "--task", task])
        process.setWorkingDirectory(str(Path(__file__).resolve().parents[3]))
        self._refresh_almaz_panel()
        process.start()

    def _read_almaz_restoration_output(self) -> None:
        process = self.almaz_restoration_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        stage = self._last_process_stage(data)
        if stage:
            task_label = {
                "denoise": "AI Denoise", "deblur": "AI Deblur",
                "jpeg_recovery": "JPEG Recovery", "all": "Restoration",
            }.get(self.almaz_restoration_task or "", "Restoration")
            self._set_almaz_prepare_stage(f"{task_label}: {stage}")
        if hasattr(self, "almaz_activity_log"):
            self.almaz_activity_log.moveCursor(QTextCursor.MoveOperation.End)
            self.almaz_activity_log.insertPlainText(data)
            self.almaz_activity_log.moveCursor(QTextCursor.MoveOperation.End)
        if hasattr(self, "ai_install_log"):
            self.ai_install_log.moveCursor(QTextCursor.MoveOperation.End)
            self.ai_install_log.insertPlainText(data)
            self.ai_install_log.moveCursor(QTextCursor.MoveOperation.End)

    def _almaz_restoration_finished(self, exit_code: int, exit_status) -> None:
        self._read_almaz_restoration_output()
        elapsed = self._end_almaz_prepare_activity()
        task = self.almaz_restoration_task or "restoration"
        labels = {
            "denoise": "AI Denoise",
            "deblur": "AI Deblur",
            "jpeg_recovery": "JPEG Recovery",
            "all": "Denoise + Deblur + JPEG Recovery",
        }
        label = labels.get(task, task)
        success = int(exit_code) == 0
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if success else 0)
        self.progress.setFormat("Готово" if success else "Ошибка")
        message = (
            f"ALMAZ {label} подготовлен и установлен за {elapsed:.1f} с."
            if success else f"Подготовка ALMAZ {label} завершилась с ошибкой, код {exit_code}, через {elapsed:.1f} с."
        )
        if hasattr(self, "almaz_activity_log"):
            self.almaz_activity_log.append("\n" + message)
        if hasattr(self, "ai_install_log"):
            self.ai_install_log.append("\n" + message)
        self.status_label.setText(("✓ " if success else "Ошибка: ") + message)
        if not success:
            detail = self._almaz_prepare_error_detail()
            QMessageBox.critical(
                self,
                "ALMAZ — ошибка подготовки модели",
                f"{message}\n\n"
                + (detail or "Подробный лог отсутствует. Проверьте доступ к интернету и состояние .venv."),
            )
        self.almaz_restoration_process = None
        self.almaz_restoration_task = None
        self._refresh_ai_status()
        self._refresh_almaz_panel()

    def _refresh_ai_status(self) -> None:
        metrics = self.current_metrics or {}
        metric = build_ai_status_metric(metrics)
        if self.current_metrics is not None:
            self.current_metrics["ai_status"] = metric
        self._show_ai({**metrics, "ai_status": metric})
        self._refresh_almaz_panel()
        self.status_label.setText("Состояние локальных AI-моделей обновлено")

    def _import_ai_model(self) -> None:
        manager = AIModelManager()
        candidates = [spec for spec in manager.specs if getattr(spec, "provider", "onnx") == "onnx"]
        if not candidates:
            QMessageBox.information(self, "Photo Doctor", "В каталоге нет внешних ONNX-моделей для импорта.")
            return
        task_labels = {
            "blur_refinement": "Тип смаза / потери деталей",
            "surface_defect_refinement": "Дефекты поверхности",
            "face_quality": "Качество лица",
            "iqa_refinement": "Дополнительная оценка качества",
            "semantic_context_refinement": "Контекст кадра",
        }
        labels = [f"{task_labels.get(spec.task, spec.task)} — {spec.model_id}" for spec in candidates]
        selected, ok = QInputDialog.getItem(
            self, "Импорт внешней модели ИИ", "Выберите назначение модели:", labels, 0, False
        )
        if not ok or not selected:
            return
        index = labels.index(selected)
        model_id = candidates[index].model_id
        filename, _ = QFileDialog.getOpenFileName(
            self, "Импорт локальной ONNX-модели", "", "Модель ONNX (*.onnx)"
        )
        if not filename:
            return
        try:
            state = manager.import_model(model_id, filename)
        except ModelImportError as exc:
            QMessageBox.warning(self, "Photo Doctor", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Photo Doctor", f"Не удалось импортировать модель: {exc}")
            return
        self._refresh_ai_status()
        QMessageBox.information(
            self,
            "Photo Doctor",
            f"Модель {localize_model_id(model_id)} импортирована.\nСостояние: {localize_value(state.status)}.\n"
            "SHA-256 закреплён для контроля целостности, но не подтверждает происхождение модели.",
        )

    def _show_ai(self, metrics) -> None:
        metric = metrics.get("ai_status")
        raw = metric.raw_value if metric is not None else None
        if not isinstance(raw, dict):
            self.ai_state_label.setText("<b>ИИ: анализ ещё не запускался</b>")
            self.ai_state_label.setStyleSheet("")
            self.ai_photo_summary.setText("Анализ ещё не запускался.")
            self.ai_trust_summary.setText("<b>Доверие к ИИ: НЕТ ДАННЫХ</b>")
            self.ai_models_summary.setText(
                "<b>ИИ-модели:</b><br>"
                "• <b>Surface AI v2 + Context Meta v2:</b> candidate · состояние не проверено<br>"
                "• <b>Рекомендатель параметров ИИ v2:</b> состояние не проверено<br>"
                "<b>Модули коррекции:</b><br>"
                "• <b>WB Advisor v1:</b> гибридный советник, не нейросеть<br>"
                "• Автотон / автоконтраст / автоцвет — детерминированный Photoshop-подобный модуль<br>"
                "• Экспозиция / средние тона — пространственная авто-маска v1<br>"
                "• Локальный контраст — пространственная авто-маска v1"
            )
            self.ai_resource_summary.setText("<b>Режим ресурсов:</b> —")
            self.ai_runtime_summary.setText("<b>ONNX Runtime:</b> —")
            self.ai_info.setText("Маршрутизатор ИИ не запускался. Классический анализ при этом остаётся полностью доступен.")
            self.ai_diagnostics_info.setText("Диагностика появится после анализа или ручного обновления состояния моделей.")
            self.ai_routes_table.setRowCount(0)
            self.ai_inference_table.setRowCount(0)
            self.ai_models_table.setRowCount(0)
            self._refresh_ai_history()
            return

        simple = build_simple_ai_view(metrics)
        display_state = simple.state
        display_title = simple.title
        display_actions = list(simple.actions)
        parameter_count = len(self.correction_ai_suggestions)
        parameter_personalized = sum(1 for row in self.correction_ai_suggestions.values() if bool(row.get("personalized", False)))
        if parameter_count:
            display_state = "USED" if simple.state in {"READY_UNUSED", "CLASSIC_ONLY"} else simple.state
            if display_state == "USED":
                display_title = "ИИ ИСПОЛЬЗОВАН НА ЭТОМ ФОТО"
            display_actions.append(f"Рекомендатель параметров ИИ предложил силу для {parameter_count} коррекций")
            if parameter_personalized:
                display_actions.append(f"Из них {parameter_personalized} — с учётом ваших сохранённых решений")
        state_colors = {
            "USED": "#16803a",
            "READY_UNUSED": "#a46a00",
            "CLASSIC_ONLY": "#777777",
            "PARTIAL_ERROR": "#a46a00",
            "ERROR": "#b42318",
        }
        color = state_colors.get(display_state, "#777777")
        self.ai_state_label.setText(f"<span style='color:{color}; font-size:16pt'>●</span> <b>{display_title}</b>")
        self.ai_photo_summary.setText("<br>".join(f"• {line}" for line in display_actions))
        self.ai_trust_summary.setText(
            f"<b>Доверие к ИИ: {simple.trust}</b><br><span style='color:#666'>{simple.trust_detail}</span>"
        )

        runtime = bool(raw.get("runtime_available", False))
        ready_count = int(raw.get("ready_count", 0) or 0)
        requested_count = int(raw.get("requested_count", 0) or 0)
        blocked_count = int(raw.get("blocked_count", 0) or 0)
        model_dir = str(raw.get("model_dir", ""))
        native_count = int(raw.get("native_count", 0) or 0)
        native_ready = int(raw.get("native_ready_count", 0) or 0)
        external_ready = int(raw.get("external_ready_count", 0) or 0)
        external_count = int(raw.get("external_count", 0) or 0)
        external_installed = int(raw.get("external_installed_count", 0) or 0)
        catalog_count = int(raw.get("catalog_count", 0) or 0)

        inference_metric = metrics.get("ai_inference")
        inference_raw = inference_metric.raw_value if inference_metric is not None and isinstance(inference_metric.raw_value, dict) else {}
        attempted = int(inference_raw.get("attempted_inferences", 0) or 0) if isinstance(inference_raw, dict) else 0
        successful = int(inference_raw.get("successful_inferences", 0) or 0) if isinstance(inference_raw, dict) else 0
        cross_metric = metrics.get("ai_crosscheck")
        cross_raw = cross_metric.raw_value if cross_metric is not None and isinstance(cross_metric.raw_value, dict) else {}
        cross_counts = cross_raw.get("counts", {}) if isinstance(cross_raw, dict) else {}
        if not isinstance(cross_counts, dict):
            cross_counts = {}
        agree_count = int(cross_counts.get("agree", 0) or 0)
        refine_count = int(cross_counts.get("refine", 0) or 0)
        contradict_count = int(cross_counts.get("contradict", 0) or 0)
        low_conf_count = int(cross_counts.get("low_confidence", 0) or 0)
        trust_metric = metrics.get("ai_trust_gate")
        trust_raw = trust_metric.raw_value if trust_metric is not None and isinstance(trust_metric.raw_value, dict) else {}
        trust_label = str(trust_raw.get("overall_label", "Нет данных")) if isinstance(trust_raw, dict) else "Нет данных"

        if native_count > 0 and native_ready == 0 and ready_count == 0:
            self.ai_state_label.setText("<span style='color:#b42318; font-size:16pt'>●</span> <b>ВСТРОЕННАЯ МОДЕЛЬ НЕДОСТУПНА</b>")

        builtin_text = "готова" if native_ready > 0 else "недоступна"
        parameter_stats = self._correction_preference_stats()
        parameter_samples = sum(int(row.get("strength_samples", 0) or 0) for row in parameter_stats.values())
        personalized_now = sum(1 for row in self.correction_ai_suggestions.values() if bool(row.get("personalized", False)))
        parameter_state = f"готов · примеров силы {parameter_samples}"
        if personalized_now:
            parameter_state += f" · персональных рекомендаций сейчас {personalized_now}"
        wb_metric = metrics.get("white_balance_advisor")
        wb_raw = wb_metric.raw_value if wb_metric is not None and isinstance(wb_metric.raw_value, dict) else {}
        wb_state = "нет данных"
        if wb_raw:
            if bool(wb_raw.get("correction_needed", False)):
                wb_state = f"совет готов · уверенность {float(wb_raw.get('confidence', 0.0) or 0.0) * 100:.0f}%"
            else:
                wb_state = "коррекция не требуется"
        try:
            almaz = inspect_almaz_status()
            sr_raw = almaz.sr if isinstance(almaz.sr, dict) else {}
            restoration_raw = almaz.restoration if isinstance(almaz.restoration, dict) else {}
            if bool(sr_raw.get("ready", False)):
                almaz_sr_state = f"AI готов · {sr_raw.get('provider_label') or sr_raw.get('provider') or 'provider —'}"
            else:
                almaz_sr_state = "AI-модель не установлена · классический безопасный fallback"
            almaz_tasks = {
                "denoise": "Denoise",
                "deblur": "Deblur",
                "jpeg_recovery": "JPEG Recovery",
            }
            ready_task_labels = [
                label for key, label in almaz_tasks.items()
                if isinstance(restoration_raw.get(key), dict) and bool(restoration_raw[key].get("ready", False))
            ]
            almaz_restoration_state = (
                ", ".join(ready_task_labels) + f" · готово {len(ready_task_labels)}/3"
                if ready_task_labels else "модели не установлены · используются штатные безопасные коррекции"
            )
            almaz_provider_state = almaz.best_provider_label or "ONNX accelerator не обнаружен"
        except Exception as exc:
            almaz_sr_state = "статус недоступен"
            almaz_restoration_state = "статус недоступен"
            almaz_provider_state = f"диагностика недоступна: {exc}"
        self.ai_models_summary.setText(
            f"<b>ИИ-модели:</b><br>"
            f"• <b>Surface AI v2 + Context Meta v2:</b> {builtin_text} · Context Meta — основной проверяющий, U-Net — второе мнение<br>"
            f"• <b>Рекомендатель параметров ИИ v2:</b> {parameter_state}<br>"
            f"• <b>ALMAZ x2:</b> {almaz_sr_state}<br>"
            f"• <b>ALMAZ Restoration:</b> {almaz_restoration_state}<br>"
            f"• <b>ALMAZ ускоритель:</b> {almaz_provider_state}<br>"
            f"<b>Модули коррекции:</b><br>"
            f"• <b>WB Advisor v1:</b> {wb_state} · гибридный базовый советник, не нейросеть<br>"
            f"• Автотон / автоконтраст / автоцвет — готов · детерминированный Photoshop-подобный модуль<br>"
            f"• Экспозиция / средние тона — готов · пространственная авто-маска v1<br>"
            f"• Локальный контраст — готов · пространственная авто-маска v1<br>"
            f"<b>Внешние модели:</b> {external_installed} установлено · готово {external_ready} из {external_count}"
        )
        resources = raw.get("resources", {})
        if isinstance(resources, dict):
            plan = resources.get("plan", {})
            if not isinstance(plan, dict):
                plan = {}
            self.ai_resource_summary.setText(
                f"<b>Режим ресурсов:</b> {plan.get('label', '—')} · размер пакета {plan.get('patch_batch_size', '—')}"
            )
            resource_detail = (
                f"Процессор: {resources.get('logical_cpus', '—')} логических потоков; ОЗУ {resources.get('total_ram_gib', '—')} ГБ "
                f"(свободно {resources.get('available_ram_gib', '—')} ГБ); бюджет ИИ "
                f"{plan.get('working_memory_budget_gib', '—')} ГБ; максимум кандидатов дефектов "
                f"{plan.get('max_surface_candidates', '—')}"
            )
        else:
            self.ai_resource_summary.setText("<b>Режим ресурсов:</b> —")
            resource_detail = "Ресурсный профиль недоступен"
        self.ai_runtime_summary.setText(
            "<b>ONNX Runtime:</b> установлен" if runtime else
            "<b>ONNX Runtime:</b> не установлен — нужен только для внешних моделей"
        )

        can_install = self._can_install_ai_runtime_here()
        self.ai_runtime_btn.setEnabled((not runtime) and can_install and self.ai_install_process is None)
        self.ai_runtime_btn.setText("ONNX Runtime установлен" if runtime else "Установить ONNX Runtime")

        runtime_text = "доступен" if runtime else "не установлен"
        self.ai_info.setText(
            f"<b>Локальный маршрутизатор ИИ:</b> запросов {requested_count}; готовых моделей {ready_count}/{catalog_count}; "
            f"заблокированных/непроверенных {blocked_count}; инференсов {successful}/{attempted}; "
            f"сверка: согласий {agree_count}, уточнений {refine_count}, противоречий {contradict_count}, "
            f"низкая уверенность {low_conf_count}; проверка доверия: {trust_label}.<br>"
            f"<span style='color:gray'>Папка моделей: {model_dir}. Среда ONNX: {runtime_text}. "
            "Фото никуда не отправляются; результат ИИ носит рекомендательный характер и не меняет классические оценки.</span>"
        )

        model_states = raw.get("models", [])
        model_diag: list[str] = []
        if isinstance(model_states, list):
            for model in model_states:
                if not isinstance(model, dict):
                    continue
                spec = model.get("spec", {})
                spec = spec if isinstance(spec, dict) else {}
                hash_state = "совпадает" if model.get("hash_ok") is True else ("не совпадает" if model.get("hash_ok") is False else "нет данных")
                model_diag.append(
                    f"{localize_model_id(spec.get('model_id', '—'))}: исполнитель={localize_value(spec.get('provider', '—'))}, "
                    f"состояние={localize_value(model.get('status', '—'))}, контрольная сумма={hash_state}, "
                    f"проверка контракта={localize_value(model.get('status', '—'))}, путь={model.get('path', '—')}; {model.get('detail', '')}"
                )
        cross_version = cross_raw.get("version", "—") if isinstance(cross_raw, dict) else "—"
        trust_version = trust_raw.get("version", "—") if isinstance(trust_raw, dict) else "—"
        self.ai_diagnostics_info.setText(
            f"<b>Среда выполнения:</b> {inference_raw.get('runtime_version', '—')} · сведения об исполнителе ниже<br>"
            f"<b>Маршрутизатор ИИ:</b> {raw.get('router_version', '—')} · <b>Сверка с классикой:</b> {cross_version} · "
            f"<b>проверка доверия:</b> {trust_version}<br>"
            f"<b>Каталог моделей:</b> {raw.get('catalog_version', '—')} · <b>Путь:</b> {model_dir}<br>"
            f"<b>Менеджер ресурсов:</b> {resource_detail}<br>"
            + ("<br>".join(model_diag) if model_diag else "Состояния моделей отсутствуют")
        )

        task_labels = {
            "blur_refinement": "Тип смаза / потери деталей",
            "surface_defect_refinement": "Дефекты поверхности",
            "face_quality": "Качество лица",
            "iqa_refinement": "Дополнительная оценка качества",
            "semantic_context_refinement": "Контекст кадра",
        }
        status_labels = {
            "missing": "Нет модели",
            "runtime_missing": "Нет ONNX Runtime",
            "unverified": "Нет закреплённой SHA-256",
            "hash_mismatch": "SHA-256 не совпал",
            "contract_mismatch": "Несовместимый ONNX",
            "unreadable": "Файл не читается",
            "ready": "Готова",
            "not_in_catalog": "Нет в каталоге",
        }

        routes = raw.get("routes", [])
        if not isinstance(routes, list):
            routes = []
        self.ai_routes_table.setRowCount(len(routes))
        for row, route in enumerate(routes):
            if not isinstance(route, dict):
                continue
            requested = bool(route.get("requested", False))
            eligible = bool(route.get("eligible_to_run", False))
            status = str(route.get("model_status", "not_in_catalog"))
            need_text = "Да" if requested else "Нет"
            if requested and eligible:
                need_text = "Да · готово"
            values = (
                need_text,
                task_labels.get(str(route.get("task", "")), str(route.get("task", "—"))),
                f"{float(route.get('priority', 0.0) or 0.0):.1f}",
                localize_model_id(route.get("model_id") or "—"),
                status_labels.get(status, status),
                str(route.get("reason", "")),
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col in (0, 2, 4):
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 0 and requested:
                    item.setBackground(QBrush(QColor(255, 244, 204)))
                if col == 4 and status == "ready":
                    item.setBackground(QBrush(QColor(224, 244, 231)))
                self.ai_routes_table.setItem(row, col, item)
            self.ai_routes_table.resizeRowToContents(row)

        inference_rows = build_ai_inference_rows(metrics)
        self.ai_inference_table.setRowCount(len(inference_rows))
        for row, inference in enumerate(inference_rows):
            confidence_text = "—" if inference.confidence is None else f"{inference.confidence * 100:.0f}%"
            detail_text = inference.detail
            if inference.comparison_detail:
                detail_text = (detail_text + " | " if detail_text else "") + inference.comparison_detail
            if inference.trust_detail:
                detail_text = (detail_text + " | " if detail_text else "") + inference.trust_detail
            values = (
                inference.task, localize_model_id(inference.model_id), str(localize_value(inference.region)), inference.result,
                confidence_text, inference.status, inference.comparison, inference.trust, detail_text,
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in (2, 4, 5, 6, 7):
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 5 and inference.status == "Готово":
                    item.setBackground(QBrush(QColor(224, 244, 231)))
                elif col == 5 and inference.status == "Ошибка":
                    item.setBackground(QBrush(QColor(255, 220, 220)))
                elif col == 5 and inference.status == "Не запущено":
                    item.setBackground(QBrush(QColor(245, 245, 245)))
                if col == 6 and inference.comparison == "Согласен":
                    item.setBackground(QBrush(QColor(224, 244, 231)))
                elif col == 6 and inference.comparison == "Уточняет":
                    item.setBackground(QBrush(QColor(255, 244, 204)))
                elif col == 6 and inference.comparison == "Противоречит":
                    item.setBackground(QBrush(QColor(255, 220, 220)))
                if col == 7 and inference.trust == "Можно как подсказку":
                    item.setBackground(QBrush(QColor(224, 244, 231)))
                elif col == 7 and inference.trust == "С осторожностью":
                    item.setBackground(QBrush(QColor(255, 244, 204)))
                elif col == 7 and inference.trust == "Только вручную":
                    item.setBackground(QBrush(QColor(255, 220, 220)))
                elif col == 7 and inference.trust == "Игнорировать":
                    item.setBackground(QBrush(QColor(245, 245, 245)))
                self.ai_inference_table.setItem(row, col, item)
            self.ai_inference_table.resizeRowToContents(row)

        models = raw.get("models", [])
        if not isinstance(models, list):
            models = []
        self.ai_models_table.setRowCount(len(models))
        for row, model in enumerate(models):
            if not isinstance(model, dict):
                continue
            spec = model.get("spec", {})
            if not isinstance(spec, dict):
                spec = {}
            status = str(model.get("status", "unknown"))
            values = (
                task_labels.get(str(spec.get("task", "")), str(spec.get("task", "—"))),
                localize_model_id(spec.get("model_id", "—")),
                str(spec.get("version", "—")),
                status_labels.get(status, status),
                str(model.get("path", "")),
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 3 and status == "ready":
                    item.setBackground(QBrush(QColor(224, 244, 231)))
                elif col == 3 and status in {"hash_mismatch", "unreadable", "contract_mismatch"}:
                    item.setBackground(QBrush(QColor(255, 220, 220)))
                self.ai_models_table.setItem(row, col, item)

        self._refresh_ai_history()

    def _refresh_ai_history(self) -> None:
        if not hasattr(self, "ai_history_table"):
            return
        try:
            with AnalysisDatabase(self.db_path) as db:
                stats = db.ai_consistency_stats(precision=self._precision_key())
        except Exception:
            stats = []
        task_labels = {
            "blur_refinement": "Тип смаза / потери деталей",
            "surface_defect_refinement": "Дефекты поверхности",
            "face_quality": "Качество лица",
            "iqa_refinement": "Дополнительная оценка качества",
            "semantic_context_refinement": "Контекст кадра",
        }
        self.ai_history_table.setRowCount(len(stats))
        for row, stat in enumerate(stats):
            compatibility = stat.get("compatibility_pct")
            compatibility_text = "—" if compatibility is None else f"{float(compatibility):.0f}%"
            values = (
                task_labels.get(str(stat.get("task", "")), str(stat.get("task", "—"))),
                localize_model_id(stat.get("model_id", "—")),
                str(int(stat.get("photo_count", 0) or 0)),
                str(int(stat.get("comparison_count", 0) or 0)),
                str(int(stat.get("agree", 0) or 0)),
                str(int(stat.get("refine", 0) or 0)),
                str(int(stat.get("contradict", 0) or 0)),
                str(int(stat.get("low_confidence", 0) or 0)),
                compatibility_text,
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col >= 2:
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 6 and int(stat.get("contradict", 0) or 0) > 0:
                    item.setBackground(QBrush(QColor(255, 220, 220)))
                self.ai_history_table.setItem(row, col, item)

    @staticmethod
    def _technical_compact_value(raw: object, *, max_chars: int = 720) -> str:
        """Return a cheap one-line preview without serializing giant spatial arrays."""
        def key_label(key: object) -> str:
            key_text = str(key)
            localized = localize_payload({key_text: ""})
            if isinstance(localized, dict) and localized:
                return str(next(iter(localized.keys())))
            return key_text

        def scalar_text(value: object) -> str:
            if isinstance(value, float):
                return f"{value:.6g}"
            text = str(localize_value(value)).replace("\n", " ").replace("\r", " ")
            return text if len(text) <= 120 else text[:117] + "…"

        if isinstance(raw, np.ndarray):
            shape = "×".join(str(x) for x in raw.shape) or str(raw.size)
            return f"массив {shape} · {raw.size} значений"
        if isinstance(raw, (list, tuple, set)):
            return f"{len(raw)} элементов"
        if not isinstance(raw, dict):
            return scalar_text(raw)

        parts: list[str] = []
        for key, value in raw.items():
            label = key_label(key)
            if isinstance(value, np.ndarray):
                shape = "×".join(str(x) for x in value.shape) or str(value.size)
                rendered = f"массив {shape} · {value.size} значений"
            elif isinstance(value, (list, tuple, set)):
                rendered = f"{len(value)} элементов"
            elif isinstance(value, dict):
                rendered = f"{len(value)} полей"
            else:
                rendered = scalar_text(value)
            parts.append(f"{label}: {rendered}")
        text = "; ".join(parts)
        return text if len(text) <= max_chars else text[: max_chars - 1] + "…"

    def _show_technical(self, metrics) -> None:
        profile_metric = metrics.get("analysis_profile")
        profile_raw = profile_metric.raw_value if profile_metric is not None and isinstance(profile_metric.raw_value, dict) else {}
        parts: list[str] = []
        if isinstance(profile_raw, dict) and profile_raw:
            parts.append(
                f"Точность: {profile_raw.get('label', profile_raw.get('key', '—'))} | "
                f"исходник {profile_raw.get('source_width', '—')}×{profile_raw.get('source_height', '—')} | "
                f"техническая копия {profile_raw.get('technical_width', '—')}×{profile_raw.get('technical_height', '—')} | "
                f"карты {profile_raw.get('spatial_width', '—')}×{profile_raw.get('spatial_height', '—')}"
            )
        map_names = (("local_sharpness", "резкость"), ("local_tone", "тоны"), ("local_contrast", "контраст"), ("noise", "шум"), ("local_white_balance", "цветовой свет"))
        map_parts: list[str] = []
        for key, label in map_names:
            metric = metrics.get(key)
            raw = metric.raw_value if metric is not None and isinstance(metric.raw_value, dict) else {}
            if not isinstance(raw, dict) or not raw:
                continue
            cells = raw.get("map_cells", raw.get("total_cells", 0))
            map_parts.append(
                f"{label}: {cells} зон, окно {raw.get('map_window_px', '—')} px, шаг {raw.get('map_step_px', '—')} px"
            )
        if map_parts:
            parts.append("Карты — " + "; ".join(map_parts))
        self.technical_info.setText("<br>".join(parts) if parts else "Параметры пространственного анализа недоступны.")
        self.technical.setRowCount(len(metrics))
        for row, metric in enumerate(metrics.values()):
            raw_text = self._technical_compact_value(metric.raw_value)
            score = "" if metric.normalized_value is None else f"{metric.normalized_value:.1f}"
            conf = f"{metric.confidence * 100:.0f}%"
            values = (localize_metric_name(metric.name), raw_text, score, conf, str(localize_value(metric.scale)), str(localize_value(metric.region)))
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.technical.setItem(row, col, item)

    def open_folder(self) -> None:
        if self.analysis_worker is not None:
            self.status_label.setText("Анализ текущего фото ещё выполняется; дождитесь завершения")
            return
        folder = QFileDialog.getExistingDirectory(self, "Выбрать папку")
        if not folder:
            return
        files = collect_images(folder, recursive=True)
        if not files:
            QMessageBox.information(self, "Photo Doctor", "Поддерживаемые изображения не найдены.")
            return
        if self.series_worker is not None:
            self.series_worker.cancel()
            self.series_worker = None
        self.batch_root = Path(folder)
        self._clear_series_results("Папка выбрана. Серии будут построены после пакетного анализа.")
        self.batch_table.setSortingEnabled(False)
        self.batch_table.setRowCount(0)
        self.batch_table.setSortingEnabled(True)
        self.batch_precision = self._precision_key()
        precision_label = get_precision(self.batch_precision).label
        self.batch_info.setText(f"Папка: {folder} — найдено файлов: {len(files)} — точность: {precision_label}")
        self.worker = BatchWorker(files, self.db_path, precision=self.batch_precision)
        self.precision_combo.setEnabled(False)
        if hasattr(self, "open_file_btn"):
            self.open_file_btn.setEnabled(False)
        if hasattr(self, "open_folder_action"):
            self.open_folder_action.setEnabled(False)
        self.worker.progress.connect(self.on_batch_progress)
        self.worker.finished_summary.connect(self.on_batch_finished)
        self.worker.failed.connect(self.on_batch_failed)
        self.batch_started_at = time.monotonic()
        self.batch_pause_started = None
        self.batch_paused_total = 0.0
        self.batch_last_i = 0
        self.batch_total = len(files)
        self.pause_btn.setText("Пауза")
        self.pause_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)
        self.progress.setRange(0, 100)
        self.progress.setFormat("%p%")
        self.progress.setValue(0)
        self.status_label.setText(f"Пакетный анализ: 0 из {len(files)} (0%)")
        self.worker.start()

    @staticmethod
    def _format_eta(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    def on_batch_progress(self, i: int, total: int, path: str, status: str) -> None:
        self.batch_last_i = i
        self.batch_total = total
        percent = round(i * 100 / max(total, 1))
        self.progress.setValue(percent)
        now = time.monotonic()
        active_elapsed = 0.0
        if self.batch_started_at is not None:
            active_elapsed = max(0.0, now - self.batch_started_at - self.batch_paused_total)
            if self.batch_pause_started is not None:
                active_elapsed = max(0.0, active_elapsed - (now - self.batch_pause_started))
        rate = i / active_elapsed if active_elapsed >= 0.25 and i > 0 else 0.0
        eta = (total - i) / rate if rate > 0 else 0.0
        rate_text = f"{rate:.2f} файл/с" if rate > 0 else "скорость: …"
        eta_text = f"осталось ~{self._format_eta(eta)}" if rate > 0 and i < total else ""
        tail = f" • {rate_text}" + (f" • {eta_text}" if eta_text else "")
        self.status_label.setText(f"{i} из {total} ({percent}%): {Path(path).name} [{status}]{tail}")

    def on_batch_failed(self, message: str) -> None:
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Пауза")
        self.cancel_btn.setEnabled(False)
        self.precision_combo.setEnabled(True)
        if hasattr(self, "open_file_btn"):
            self.open_file_btn.setEnabled(True)
        if hasattr(self, "open_folder_action"):
            self.open_folder_action.setEnabled(True)
        self.worker = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Ошибка")
        self.status_label.setText("Ошибка пакетного анализа")
        QMessageBox.critical(self, "Ошибка пакетного анализа", message)

    def on_batch_finished(self, summary) -> None:
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Пауза")
        self.cancel_btn.setEnabled(False)
        self.precision_combo.setEnabled(True)
        if hasattr(self, "open_file_btn"):
            self.open_file_btn.setEnabled(True)
        if hasattr(self, "open_folder_action"):
            self.open_folder_action.setEnabled(True)
        suffix = " — отменено" if summary.cancelled else ""
        self.status_label.setText(
            f"Готово: {summary.succeeded} успешно, {summary.cached} из кэша, "
            f"{summary.failed} ошибок{suffix}"
        )
        self._show_batch_results(summary)
        self.worker = None
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if not summary.cancelled else 0)
        self.progress.setFormat("Готово" if not summary.cancelled else "Отменено")
        if not summary.cancelled:
            self._start_series_analysis(summary)

    @staticmethod
    def _primary_batch_diagnosis(metrics) -> str:
        rows = build_display_metrics(metrics)
        priority = {"Проблема": 0, "Внимание": 1, "Проверить": 2, "Особенность": 3, "Норма": 4, "Хорошо": 5}
        rows = sorted(rows, key=lambda row: (priority.get(row.status, 9), row.confidence * -1.0))
        for row in rows:
            if row.status in {"Проблема", "Внимание", "Проверить"}:
                return f"{row.label}: {row.diagnosis}"
        return "Критических технических замечаний по текущим метрикам нет."

    def _show_batch_results(self, summary) -> None:
        self.batch_table.setSortingEnabled(False)
        self.batch_table.setRowCount(len(summary.items))
        with AnalysisDatabase(self.db_path) as db:
            for row_index, batch_item in enumerate(summary.items):
                path = Path(batch_item.path)
                try:
                    display_path = str(path.relative_to(self.batch_root)) if self.batch_root else path.name
                except ValueError:
                    display_path = path.name

                path_item = QTableWidgetItem(display_path)
                path_item.setData(Qt.UserRole, str(path))
                path_item.setToolTip(str(path))
                self.batch_table.setItem(row_index, 0, path_item)

                if batch_item.status == "error":
                    for col in range(1, 8):
                        self.batch_table.setItem(row_index, col, QTableWidgetItem("—"))
                    error_item = QTableWidgetItem("Ошибка")
                    error_item.setBackground(QBrush(QColor(255, 224, 224)))
                    error_item.setToolTip(batch_item.error or "Неизвестная ошибка")
                    self.batch_table.setItem(row_index, 8, error_item)
                    if batch_item.error:
                        self.batch_table.item(row_index, 6).setText(batch_item.error)
                    continue

                metrics = db.load_metrics(path, precision=self.batch_precision)
                if not metrics:
                    for col in range(1, 8):
                        self.batch_table.setItem(row_index, col, QTableWidgetItem("—"))
                    self.batch_table.setItem(row_index, 8, QTableWidgetItem("Нет данных"))
                    continue

                summary_row = build_summary(metrics)
                profile = build_photo_profile(metrics)
                values = (
                    (1, summary_row["quality"]),
                    (2, summary_row["potential"]),
                    (3, summary_row["confidence"]),
                    (4, summary_row["issues"]),
                )
                for col, value in values:
                    if col in (1, 2):
                        text = f"{float(value):.1f}"
                    elif col == 3:
                        text = f"{float(value):.0f}%"
                    else:
                        text = str(int(value))
                    item = NumericTableWidgetItem(text, float(value))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.batch_table.setItem(row_index, col, item)
                self.batch_table.setItem(row_index, 5, QTableWidgetItem(profile.label))
                diagnosis = self._primary_batch_diagnosis(metrics)
                diagnosis_item = QTableWidgetItem(diagnosis)
                diagnosis_item.setToolTip(diagnosis)
                self.batch_table.setItem(row_index, 6, diagnosis_item)
                ai_view = build_simple_ai_view(metrics)
                ai_item = QTableWidgetItem(ai_view.batch_label)
                ai_item.setData(Qt.UserRole, ai_view.batch_key)
                ai_item.setData(Qt.UserRole + 1, ai_view.state)
                ai_item.setData(Qt.UserRole + 2, ai_view.used)
                ai_item.setTextAlignment(Qt.AlignCenter)
                if ai_view.batch_key == "hint":
                    ai_item.setBackground(QBrush(QColor(224, 244, 231)))
                elif ai_view.batch_key == "caution":
                    ai_item.setBackground(QBrush(QColor(255, 244, 204)))
                elif ai_view.batch_key in {"manual", "error"}:
                    ai_item.setBackground(QBrush(QColor(255, 220, 220)))
                else:
                    ai_item.setBackground(QBrush(QColor(245, 245, 245)))
                self.batch_table.setItem(row_index, 7, ai_item)

                status_text = "Из кэша" if batch_item.status == "cached" else "Готово"
                self.batch_table.setItem(row_index, 8, QTableWidgetItem(status_text))

        self.batch_table.setSortingEnabled(True)
        self.batch_table.sortItems(1, Qt.AscendingOrder)
        self.batch_info.setText(
            f"Файлов: {summary.total}; обработано: {summary.processed}; "
            f"новых: {summary.succeeded}; из кэша: {summary.cached}; ошибок: {summary.failed}. "
            "По умолчанию сверху показаны снимки с худшей технической оценкой. Двойной клик открывает фото."
        )
        self._apply_batch_ai_filter()
        batch_tab_index = self.tabs.indexOf(self.batch_page)
        if batch_tab_index >= 0:
            self.tabs.setCurrentIndex(batch_tab_index)

    def _apply_batch_ai_filter(self) -> None:
        if not hasattr(self, "batch_ai_filter") or not hasattr(self, "batch_table"):
            return
        selected = str(self.batch_ai_filter.currentData() or "all")
        for row in range(self.batch_table.rowCount()):
            item = self.batch_table.item(row, 7)
            batch_key = str(item.data(Qt.UserRole)) if item is not None and item.data(Qt.UserRole) is not None else "not_used"
            state = str(item.data(Qt.UserRole + 1)) if item is not None and item.data(Qt.UserRole + 1) is not None else "CLASSIC_ONLY"
            used = bool(item.data(Qt.UserRole + 2)) if item is not None else False
            if selected == "all":
                visible = True
            elif selected == "manual":
                visible = batch_key == "manual"
            elif selected == "used":
                visible = used
            elif selected == "not_used":
                visible = batch_key == "not_used"
            elif selected == "error":
                visible = batch_key == "error"
            else:
                visible = True
            self.batch_table.setRowHidden(row, not visible)

    def _open_batch_row(self, row: int, column: int) -> None:
        item = self.batch_table.item(row, 0)
        if item is None:
            return
        path_text = item.data(Qt.UserRole)
        if not path_text:
            return
        path = Path(str(path_text))
        if not path.is_file():
            QMessageBox.warning(self, "Photo Doctor", f"Файл больше не найден:\n{path}")
            return
        self._open_path(path)

    def toggle_batch_pause(self) -> None:
        if not self.worker:
            return
        if self.worker.is_paused():
            self.worker.resume()
            if self.batch_pause_started is not None:
                self.batch_paused_total += max(0.0, time.monotonic() - self.batch_pause_started)
            self.batch_pause_started = None
            self.pause_btn.setText("Пауза")
            self.status_label.setText(
                f"Пакетный анализ продолжен: {self.batch_last_i} из {self.batch_total}"
            )
        else:
            self.worker.pause()
            self.batch_pause_started = time.monotonic()
            self.pause_btn.setText("Продолжить")
            self.status_label.setText(
                f"Пауза после текущего файла: {self.batch_last_i} из {self.batch_total}"
            )

    def cancel_batch(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("Пауза")
            self.status_label.setText("Отмена запрошена…")


def run() -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.show()
    return app.exec()
