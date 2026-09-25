from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "photodoctor" / "gui" / "main_window.py"


def test_results_are_below_image_in_vertical_splitter():
    text = SOURCE.read_text(encoding="utf-8")
    assert "QSplitter(Qt.Vertical)" in text
    assert "self.main_splitter.addWidget(self.image_view)" in text
    assert "self.main_splitter.addWidget(self.tabs)" in text


def test_expected_diagnostic_tabs_exist():
    text = SOURCE.read_text(encoding="utf-8")
    for name in ["Результат", "Метрики", "Гистограмма", "Рекомендации", "Технические данные"]:
        assert f'"{name}"' in text


def test_primary_metrics_table_has_no_raw_column():
    text = SOURCE.read_text(encoding="utf-8")
    assert '["Статус", "Параметр", "Оценка", "Уверенность", "Диагноз"]' in text
    assert '["Метрика", "Исходное значение", "Оценка", "Уверенность", "Масштаб", "Область"]' in text


def test_recommendations_are_split_by_risk():
    text = SOURCE.read_text(encoding="utf-8")
    for name in ["Можно безопасно", "С осторожностью", "Не делать автоматически"]:
        assert f'"{name}"' in text


def test_result_shows_photo_profile():
    text = SOURCE.read_text(encoding="utf-8")
    assert "profile = build_photo_profile(metrics)" in text
    assert "self.profile_label.setText(" in text
    assert "profile.label" in text
    assert "build_photo_profile" in text


def test_splitter_and_active_tab_are_persisted():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'QSettings("PhotoDoctor", "PhotoDoctor")' in text
    assert 'self.main_splitter.saveState()' in text
    assert 'self.main_splitter.restoreState(state)' in text
    assert 'self.tabs.currentIndex()' in text


def test_surface_defect_overlay_toggle_exists():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Карта дефектов"' in text
    assert "set_overlay_boxes" in text
    assert "set_overlay_enabled" in text
    assert 'boxes_norm' in text


def test_surface_overlay_distinguishes_candidate_polarity():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'box.get("polarity", "bright")' in text
    assert 'Qt.DashLine' in text


def test_faces_tab_and_overlay_exist():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.tabs.addTab(self.faces_page, "Лица")' in text
    assert 'Добавить лицо вручную' in text
    assert 'Добавить глаз вручную' in text
    assert 'Очистить ручные отметки' in text
    assert 'region_selected.connect(self._on_manual_region_selected)' in text
    assert 'self.faces_btn = QPushButton("Лица")' in text
    assert "set_face_boxes" in text
    assert "set_faces_enabled" in text
    assert 'QColor(45, 170, 95)' in text


def test_recommendation_widgets_keep_explicit_qt_owners_alive():
    text = SOURCE.read_text(encoding="utf-8")
    assert "self.safe_recommendations_box, self.safe_recommendations" in text
    assert "self.caution_recommendations_box, self.caution_recommendations" in text
    assert "self.avoid_recommendations_box, self.avoid_recommendations" in text
    assert "return box, widget" in text
    assert "layout.addWidget(box, 1)" in text
    assert "box.parentWidget()" not in text




def test_primary_navigation_is_compact_and_batch_ui_is_not_exposed():
    text = SOURCE.read_text(encoding="utf-8")
    assert "def _organize_primary_tabs" in text
    assert '(overview, "Обзор")' in text
    assert '(corrections, "Исправления")' in text
    assert '(defects, "Дефекты")' in text
    assert '(faces, "Лица")' in text
    assert 'self.tabs.addTab(self.diagnostics_page, "Диагностика")' in text
    organizer = text[text.index("def _organize_primary_tabs"):text.index("def _build_result_tab")]
    assert '"Папка"' not in organizer
    assert '"Серии"' not in organizer
    toolbar = text[text.index("toolbar = QToolBar"):text.index("self.fit_btn = QPushButton")]
    assert "toolbar.addAction(self.open_folder_action)" not in toolbar
    assert "self.pause_btn.setVisible(False)" in text
    assert "self.cancel_btn.setVisible(False)" in text


def test_overview_hides_long_diagnostics_until_requested():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'photo_title = QLabel("<b>О фотографии</b>")' in text
    assert 'self.overview_details_btn = QPushButton("Показать подробности анализа")' in text
    assert 'self.overview_details_panel = QWidget(content)' in text
    assert "self.overview_details_panel.setVisible(False)" in text
    assert 'overview_open_corrections_btn' not in text
    assert 'overview_preview_btn' not in text
    assert 'self.fix_summary_box = QGroupBox' not in text


def test_overview_is_flat_compact_and_shows_photo_camera_metadata():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _build_result_tab")
    end = text.index("def _build_metrics_tab", start)
    block = text[start:end]
    assert 'self.overview_stats_label = QLabel(' in block
    assert 'self.photo_info_label = QLabel("Разрешение и формат: —")' in block
    assert 'self.camera_info_label = QLabel("Камера и параметры съёмки: —")' in block
    assert 'Потенциал улучшения' not in block
    assert 'QGroupBox("О фотографии")' not in block
    assert 'QGroupBox("Резюме исправлений")' not in block
    assert 'def _update_overview_photo_metadata' in text
    assert 'f"{width}×{height}"' in text
    assert 'raw.get("make")' in text
    assert 'raw.get("model")' in text
    assert 'raw.get("iso")' in text
    assert 'raw.get("focal_length_mm")' in text


def test_corrections_keep_technical_scores_but_hide_them_from_main_table():
    text = SOURCE.read_text(encoding="utf-8")
    assert "for column in (4, 6, 7, 8):" in text
    assert "self.plan_table.setColumnHidden(column, True)" in text
    assert '<b>Технические оценки:</b>' in text

def test_recommendation_list_has_explicit_groupbox_parent():
    text = SOURCE.read_text(encoding="utf-8")
    assert "widget = QListWidget(box)" in text


def test_result_diagnostics_have_vertical_scroll_without_moving_summary_cards():
    text = SOURCE.read_text(encoding="utf-8")
    assert "page = QScrollArea()" in text
    assert "page.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)" in text
    assert "self.result_text = QTextBrowser()" in text
    assert "self.result_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)" in text
    assert "self.result_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)" in text
    assert "self.result_text.setMinimumHeight(90)" in text
    assert 'self.tabs.addTab(page, "Обзор")' in text


def test_result_scroll_returns_to_top_after_new_analysis():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.result_text.setPlainText("\\n".join(lines))' in text or 'self.result_text.setPlainText("\n".join(lines))' in text
    assert "self.result_text.verticalScrollBar().setValue(0)" in text


def test_local_sharpness_overlay_toggle_exists_and_excludes_low_texture_cells():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.sharpness_btn = QPushButton("Карта резкости")' in text
    assert "set_sharpness_cells" in text
    assert "set_sharpness_enabled" in text
    assert 'status == "low_texture"' in text
    assert 'sharpness_metric = result.metrics.get("local_sharpness")' in text


def test_defect_and_sharpness_maps_do_not_stack_accidentally():
    text = SOURCE.read_text(encoding="utf-8")
    assert "def _toggle_defects" in text
    assert "def _toggle_sharpness" in text
    assert "self.sharpness_btn.setChecked(False)" in text
    assert "self.defects_btn.setChecked(False)" in text


def test_result_summary_collapses_global_local_and_face_sharpness_into_one_item():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'sharpness_keys = {"sharpness", "detail_loss_type", "local_sharpness", "faces", "eyes"}' in text
    assert 'lines.append(f"• Резкость [{base.status}]' in text
    assert 'локальная карта: около {soft_pct:.0f}% информативной площади мягкие' in text
    assert 'лиц для локальной проверки: {face_count}' in text


def test_tone_map_button_and_cells_are_wired():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.tone_btn = QPushButton("Карта тонов")' in text
    assert 'tone_metric = result.metrics.get("local_tone")' in text
    assert 'self.image_view.set_tone_cells(tone_cells)' in text
    assert 'def set_tone_cells' in text


def test_three_analysis_maps_are_mutually_exclusive():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'def _toggle_tone' in text
    assert 'self.tone_btn.setChecked(False)' in text
    assert 'self.sharpness_btn.setChecked(False)' in text
    assert 'self.defects_btn.setChecked(False)' in text


def test_batch_analysis_has_sortable_results_tab_and_open_on_double_click():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.tabs.addTab(page, "Папка")' in text
    assert '"Файл", "Качество", "Потенциал", "Уверенность"' in text
    assert "self.batch_table.setSortingEnabled(True)" in text
    assert "self.batch_table.cellDoubleClicked.connect(self._open_batch_row)" in text
    assert "db.load_metrics(path, precision=self.batch_precision)" in text
    assert "self._open_path(path)" in text


def test_batch_results_default_to_worst_quality_first():
    text = SOURCE.read_text(encoding="utf-8")
    assert "self.batch_table.sortItems(1, Qt.AscendingOrder)" in text
    assert "с худшей технической оценкой" in text


def test_batch_tab_keeps_explicit_qt_owner_reference():
    text = SOURCE.read_text(encoding="utf-8")
    assert "self.batch_page = QWidget()" in text
    assert "self.tabs.indexOf(self.batch_page)" in text
    assert "self.batch_table.parentWidget()" not in text


def test_local_contrast_map_is_wired_and_exclusive():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.contrast_btn = QPushButton("Карта контраста")' in text
    assert "set_contrast_cells" in text
    assert "set_contrast_enabled" in text
    assert 'contrast_metric = result.metrics.get("local_contrast")' in text
    assert "def _toggle_contrast" in text
    assert "self.contrast_btn.setChecked(False)" in text


def test_batch_has_pause_speed_percent_and_eta_controls():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.pause_btn = QPushButton("Пауза")' in text
    assert "def toggle_batch_pause" in text
    assert 'self.pause_btn.setText("Продолжить")' in text
    assert "time.monotonic()" in text
    assert 'файл/с' in text
    assert 'осталось ~' in text
    assert '({percent}%)' in text


def test_specular_highlight_overlay_is_wired():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.highlights_btn = QPushButton("Блики")' in text
    assert "set_highlight_boxes" in text
    assert "set_highlights_enabled" in text
    assert 'highlight_metric = result.metrics.get("highlight_context")' in text
    assert 'QColor(45, 175, 205)' in text


def test_face_overlay_also_draws_detected_eye_regions():
    text = SOURCE.read_text(encoding="utf-8")
    assert "def set_eye_boxes" in text
    assert "self.image_view.set_eye_boxes(eye_boxes)" in text
    assert "if self._eye_boxes:" in text
    assert 'eye.get("face_index"' in text


def test_result_tab_shows_semantic_context_separately_from_tone_profile():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.semantic_label = QLabel("—")' in text
    assert 'semantic_metric = metrics.get("semantic_context")' in text
    assert '"archival_portrait": "Архивный портрет"' in text
    assert '"group_portrait": "Групповой портрет"' in text
    assert "self.semantic_label," in text


def test_decision_plan_tab_is_wired_compact_and_numeric_priority_is_sortable():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.tabs.addTab(self.plan_page, "Исправления")' in text
    assert '"Применить", "Решение", "Исправление", "Сила", "Персональная подстройка"' in text
    assert '"Область", "Приоритет", "Важность", "Исправимость", "Уверенность", "Статус"' in text
    assert '"Совет ИИ"' not in text
    assert "def _show_plan" in text
    assert 'metrics.get("decision_plan")' in text
    assert 'NumericTableWidgetItem(f"{priority:.1f}", priority)' in text
    assert "self.plan_table.sortItems(6, Qt.DescendingOrder)" in text
    assert "setDefaultSectionSize(32)" in text
    assert "self.plan_table.setWordWrap(False)" in text


def test_plan_tab_surfaces_validator_result_in_compact_status_and_details():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'details_box = QGroupBox("Подробности выбранной строки")' in text
    assert 'metrics.get("recommendation_validation")' in text
    assert 'validation_text = f"Подтверждено:' in text
    assert 'validation_text = "Не подтверждено.' in text
    assert 'f"<b>Почему:</b> {reason}<br><b>Ограничение:</b> {guardrail}<br>"' in text
    assert '<b>Технические оценки:</b>' in text


def test_main_toolbar_uses_short_labels_and_keeps_open_save_left_preview_right():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.open_file_btn = QPushButton("Открыть")' in text
    assert 'self.save_copy_btn = QPushButton("Сохранить")' in text
    assert 'self.preview_btn = QPushButton("Предпросмотр")' in text
    open_pos = text.index('toolbar.addWidget(self.open_file_btn)')
    save_pos = text.index('toolbar.addWidget(self.save_copy_btn)')
    spacer_pos = text.index('toolbar.addWidget(toolbar_spacer)')
    preview_pos = text.index('toolbar.addWidget(self.preview_btn)')
    assert open_pos < save_pos < spacer_pos < preview_pos
    assert 'SP_DialogOpenButton' in text
    assert 'SP_DialogSaveButton' in text
    assert 'SP_FileDialogContentsView' in text


def test_non_destructive_validated_preview_button_is_wired():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.preview_btn = QPushButton("Предпросмотр")' in text
    assert "self.preview_btn.setEnabled(False)" in text
    assert "self.preview_btn.toggled.connect(self._toggle_preview)" in text
    assert "self.correction_regions" in text
    assert "apply_selected_preview(" in text
    assert "self.current_rgb = loaded.srgb.copy()" in text
    assert "исходный файл не изменён" in text


def test_save_copy_is_non_destructive_and_uses_validated_preview():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.save_copy_btn = QPushButton("Сохранить")' in text
    assert "self.save_copy_btn.setEnabled(False)" in text
    assert "self.save_copy_btn.clicked.connect(self._save_preview_copy)" in text
    assert 'self.current_path.stem + "_PhotoDoctor.png"' in text
    assert "self.correction_regions" in text
    assert "apply_selected_preview(" in text
    assert "photo_doctor_surface_history=surface_history_payload" in text


def test_ai_tab_exists_and_is_informational_only():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.tabs.addTab(self.ai_page, "ИИ")' in text
    assert 'self.ai_routes_table' in text
    assert 'self.ai_models_table' in text
    assert 'носит рекомендательный характер' in text
    assert 'Фото никуда не отправляются' in text
    assert 'self.ai_state_label = QLabel("<b>ИИ: анализ ещё не запускался</b>")' in text
    assert 'self.ai_models_summary' in text
    assert 'self.ai_photo_summary' in text
    assert 'self.ai_resource_summary' in text
    assert 'self.ai_advanced_btn = QPushButton("Расширенно")' in text


def test_ai_tab_supports_manual_model_import_and_refresh():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'QPushButton("Настроить внешние модели…")' in text
    assert 'self._import_ai_model' in text
    assert 'manager.import_model(model_id, filename)' in text
    assert 'self.ai_refresh_btn = QPushButton("Обновить диагностику")' in text
    assert 'build_ai_status_metric' in text
    assert 'SHA-256 закреплён' in text


def test_ai_runtime_can_be_installed_from_gui_without_shell():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'QPushButton("Установить ONNX Runtime")' in text
    assert 'QProcess(self)' in text
    assert 'process.setProgram(sys.executable)' in text
    assert '"-m", "pip", "install"' in text
    assert '"onnxruntime>=1.20,<2"' in text
    assert 'QProcess.ProcessChannelMode.MergedChannels' in text
    assert 'getattr(sys, "frozen", False)' in text
    assert 'START_PhotoDoctor.bat' in text


def test_spatial_maps_use_smoothed_heatmap_not_visible_cell_grid():
    text = (ROOT / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert "def _spatial_heatmap" in text
    assert 'self._spatial_heatmap(shown, self._sharpness_cells, "sharpness")' in text
    assert 'self._spatial_heatmap(shown, self._tone_cells, "tone")' in text
    assert 'self._spatial_heatmap(shown, self._contrast_cells, "contrast")' in text
    assert "cv2.GaussianBlur" in text


def test_surface_overlay_prefers_component_contour_over_large_bbox():
    text = (ROOT / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert 'contour = box.get("contour")' in text
    assert "painter.drawPath(path)" in text


def test_precision_selector_defaults_new_photos_to_normal_and_manual_vision_rechecks_precisely():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'precision_label = QLabel("Точность:")' in text
    assert 'self.precision_combo = QComboBox()' in text
    assert 'if profile.key == "normal"' in text
    assert 'self._set_precision_combo("normal")' in text
    assert 'precision_override="normal"' in text
    assert 'self._set_precision_combo("precise")' in text
    assert 'precision_override="precise"' in text
    assert 'manual_face_boxes=manual_faces' in text
    assert 'manual_eye_boxes=manual_eyes' in text
    assert 'BatchWorker(files, self.db_path, precision=self.batch_precision)' in text
    assert 'self.precision_combo.setEnabled(False)' in text
    assert 'self.precision_combo.setEnabled(True)' in text


def test_technical_tab_exposes_real_spatial_map_parameters():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.technical_info = QLabel(' in text
    assert 'map_window_px' in text
    assert 'map_step_px' in text
    assert 'map_cells' in text


def test_ai_tab_has_actual_inference_results_table():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.ai_inference_table = QTableWidget(0, 9)' in text
    assert 'self.ai_advanced_tabs.addTab(inference_page, "Результаты")' in text
    assert 'build_ai_inference_rows(metrics)' in text
    assert 'inference.status == "Готово"' in text
    assert 'inference.status == "Ошибка"' in text


def test_ai_tab_shows_crosscheck_summary_and_column():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Сверка"' in text
    assert 'agree_count' in text
    assert 'contradict_count' in text
    assert 'inference.comparison == "Противоречит"' in text


def test_batch_tab_has_ai_trust_column_and_filters():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.batch_table = QTableWidget(0, 9)' in text
    assert '"Главное замечание", "ИИ", "Статус"' in text
    assert 'Требует ручной проверки' in text
    assert 'ИИ использован' in text
    assert 'ИИ не использовался' in text
    assert 'Ошибка ИИ' in text
    assert 'def _apply_batch_ai_filter' in text
    assert 'visible = used' in text


def test_ai_tab_has_model_consistency_history_not_accuracy_claim():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.ai_history_table = QTableWidget(0, 9)' in text
    assert 'Согласованность с классическим анализом (не эталон и не точность модели)' in text
    assert 'не эталон и не точность модели' in text
    assert 'db.ai_consistency_stats(precision=self._precision_key())' in text
    assert '"Совместимость*"' in text


def test_defects_tab_supports_persistent_human_feedback():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.tabs.addTab(page, "Дефекты")' in text
    assert 'self.feedback_defect_btn = QPushButton("Это дефект [D]")' in text
    assert 'self.feedback_natural_btn = QPushButton("Естественная деталь [N]")' in text
    assert 'self.feedback_uncertain_btn = QPushButton("Не уверен [U]")' in text
    assert 'db.save_surface_feedback(self.current_path, candidate, user_label)' in text
    assert 'db.delete_surface_feedback(self.current_path, candidate)' in text
    assert 'user_label = str(box.get("user_label", ""))' in text
    assert 'self.feedback_export_btn = QPushButton("Экспорт разметки…")' in text
    assert 'export_surface_feedback_dataset(self.db_path, target)' in text


def test_surface_feedback_selection_highlights_candidate_and_has_shortcuts():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self._overlay_selected_index: int | None = None' in text
    assert 'def set_overlay_selection(self, index: int | None)' in text
    assert 'box_index == self._overlay_selected_index' in text
    assert 'self.feedback_defect_btn.setShortcut(QKeySequence("D"))' in text
    assert 'self.feedback_natural_btn.setShortcut(QKeySequence("N"))' in text
    assert 'self.feedback_uncertain_btn.setShortcut(QKeySequence("U"))' in text
    assert 'QKeySequence(Qt.Key_Delete)' in text
    assert 'self.defects_btn.setChecked(True)' in text
    assert 'next_row = min(row + 1' in text


def test_defects_tab_shows_human_feedback_model_stats_without_claiming_benchmark_accuracy():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.feedback_stats_label' in text
    assert 'db.surface_feedback_stats()' in text
    assert 'Surface AI v2 по вашей разметке: совпадение' in text
    assert 'уверенных ошибок' in text
    assert 'не внешний эталон точности' in text
    assert 'Модель не переобучается автоматически во время работы.' in text


def test_result_tab_has_compact_fix_summary_and_red_eye_status():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'fixes_title = QLabel("<b>Исправления</b>")' in text
    assert 'self.fix_summary_list.setFrameShape(QFrame.Shape.NoFrame)' in text
    assert 'Красные глаза: обнаружено' in text
    assert 'Можно исправить автоматически' in text
    assert 'Требуется ручная проверка' in text
    assert 'Безопасные автоматические исправления не подтверждены' in text


def test_ai_simple_mode_is_flat_scrollable_and_hides_internal_terms():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _build_ai_tab")
    advanced_start = text.index('self.ai_advanced_tabs = QTabWidget()', start)
    simple_text = text[start:advanced_start]
    assert 'self.ai_page = QScrollArea()' in simple_text
    assert 'self.ai_page.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)' in simple_text
    assert 'add_section_title("Что ИИ сделал на этом фото")' in simple_text
    assert 'add_section_title("Доверие к ИИ")' in simple_text
    assert 'add_section_title("Модели и ресурсы")' in simple_text
    assert 'QGroupBox(' not in simple_text
    assert 'self.ai_advanced_tabs.addTab(diagnostics_page, "Диагностика")' in text
    assert 'ONNX Runtime:</b> не установлен — нужен только для внешних моделей' in text
    for forbidden in ("Router", "Manifest", "SHA-256", "tensor", "Crosscheck", "Trust Gate"):
        assert forbidden not in simple_text


def test_almaz_tab_is_flat_scrollable_and_collapses_idle_log():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _build_almaz_tab")
    end = text.index("def _build_faces_tab", start)
    block = text[start:end]
    assert 'self.almaz_page = QScrollArea()' in block
    assert 'self.almaz_page.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)' in block
    assert 'QGroupBox(' not in block
    assert 'self.almaz_log_toggle_btn = QPushButton("Показать журнал подготовки")' in block
    assert 'self.almaz_activity_log.setVisible(False)' in block
    assert 'runtime_row_1 = QHBoxLayout()' in block
    assert 'runtime_row_2 = QHBoxLayout()' in block


def test_save_copy_audits_metadata_and_warns_on_omitted_classes():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'audit_metadata_preservation(self.current_path, saved)' in text
    assert 'metadata_audit.omitted_classes' in text
    assert 'Метаданные сохранены частично' in text


def test_series_tab_is_wired_to_batch_without_reanalysis():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.tabs.addTab(self.series_page, "Серии")' in text
    assert 'group_photo_series' in text
    assert 'class SeriesWorker(QThread)' in text
    assert 'self._start_series_analysis(summary)' in text
    assert 'item.status in {"ok", "cached"}' in text
    assert 'self.series_members_table.cellDoubleClicked.connect(self._open_series_member)' in text


def test_series_ui_labels_best_frame_as_technical_only():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Лучший технически"' in text
    assert 'выражение лица, момент и художественная ценность пока не оцениваются' in text
    assert 'Лучший кадр выбирается только по техническим признакам' in text


def test_result_tab_is_read_only_summary_and_selection_lives_elsewhere():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _build_result_tab")
    end = text.index("def _build_", start + len("def _build_result_tab"))
    block = text[start:end]
    assert 'self.fix_actions_list = QListWidget(content)' in block
    assert 'self.fix_actions_list.setVisible(False)' in block
    assert 'self.fix_summary_list = QListWidget(content)' in block
    assert 'self.fix_summary_list.setSelectionMode(QAbstractItemView.NoSelection)' in block
    assert 'self.fix_summary_list.setFocusPolicy(Qt.NoFocus)' in block
    assert 'layout.addWidget(self.fix_summary_list)' in block
    assert 'auto_select_fixes_btn' not in block
    assert 'select_all_fixes_btn' not in block
    assert 'clear_fixes_btn' not in block
    assert 'self.active_corrections_label = QLabel("Будет применено: ничего")' in block

    populate_start = text.index("def _populate_correction_choices")
    populate_end = text.index("def _selected_correction_keys", populate_start)
    populate = text[populate_start:populate_end]
    assert 'summary_item.setFlags(summary_item.flags() & ~Qt.ItemIsUserCheckable)' in populate
    assert 'self.fix_summary_list.addItem(summary_item)' in populate

def test_user_correction_feedback_is_local_optional_and_written_only_after_save():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.correction_feedback_checkbox = QCheckBox("Запоминать мои решения для локального обучения")' in text
    assert 'self.settings.setValue("correction_feedback_enabled", enabled)' in text
    assert 'db.record_correction_feedback(' in text
    assert 'recorded = self._record_saved_correction_feedback(saved)' in text
    assert 'self.current_source_quick_hash = str(incoming_hash) if incoming_hash else None' in text
    assert 'исходник изменился после анализа; обратная связь не записана' in text
    assert 'Модель не переобучается автоматически' in text or 'текущая модель сама себя не меняет' in text


def test_plan_tab_has_visible_apply_checkboxes_synchronised_with_hidden_selection_store():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'Выберите, что применить, настройте силу и при необходимости ограничьте область.' in text
    assert 'apply_item.setCheckState(Qt.Checked if key in selected_keys else Qt.Unchecked)' in text
    assert 'self.plan_table.itemChanged.connect(self._on_plan_correction_item_changed)' in text
    assert 'source_item.setCheckState(item.checkState())' in text
    assert 'def _sync_plan_corrections_from_result' in text
    assert 'self._sync_plan_corrections_from_result()' in text

def test_overlay_controls_use_second_toolbar_to_avoid_high_dpi_overflow():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.addToolBarBreak()' in text
    assert 'view_toolbar = QToolBar("Просмотр, карты и области")' in text
    assert 'view_toolbar.addWidget(self.faces_btn)' in text
    assert 'view_toolbar.addWidget(self.highlights_btn)' in text


def test_plan_decision_pastel_cells_force_dark_foreground_for_dark_theme_readability():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'first.setForeground(QBrush(QColor(32, 32, 32)))' in text


def test_plan_tab_exposes_same_quick_selection_controls_as_result_tab():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.plan_auto_select_btn = QPushButton("Выбрать рекомендуемые")' in text
    assert 'self.plan_select_all_btn = QPushButton("Выбрать всё доступное")' in text
    assert 'self.plan_clear_btn = QPushButton("Снять всё")' in text
    assert 'self.plan_auto_select_btn.clicked.connect(self._reset_correction_selection)' in text
    assert 'self.plan_select_all_btn.clicked.connect(self._select_all_corrections)' in text
    assert 'self.plan_clear_btn.clicked.connect(self._clear_correction_selection)' in text


def test_plan_has_live_strength_slider_for_adjustable_corrections():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Применить", "Решение", "Исправление", "Сила", "Персональная подстройка"' in text
    assert "slider = QSlider(Qt.Horizontal" in text
    assert "def _on_plan_strength_changed" in text
    assert "self.correction_strengths" in text
    assert "value / 100.0" in text


def test_plan_does_not_silently_show_dash_for_actionable_problem_without_preview():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'if decision in {"fix", "review"}:' in text
    assert 'apply_item.setText("Нет")' in text
    assert 'Для этой найденной проблемы пока нет исполняемой безопасной коррекции.' in text


def test_image_view_supports_fit_25_to_300_zoom_and_ctrl_wheel():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.zoom_slider.setRange(25, 300)' in text
    assert 'def set_zoom_percent' in text
    assert 'def wheelEvent' in text
    assert 'Qt.KeyboardModifier.ControlModifier' in text
    assert 'self.image_view.zoom_changed.connect(self._on_image_zoom_changed)' in text
    assert 'Qt.FastTransformation if self._zoom_percent >= 100' in text


def test_plan_supports_local_user_selected_correction_regions():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Область"' in text
    assert 'def _start_correction_region_selection' in text
    assert 'self.image_view.begin_region_selection(f"correction:{key}")' in text
    assert 'self.correction_regions[key] = box' in text
    assert 'def _clear_correction_region' in text


def test_surface_defects_have_per_candidate_repair_selection():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Лечить", "№", "Контекст", "Surface AI v2", "Meta-контекст", "Итог"' in text
    assert '"Сигнал", "Тип", "Полярность", "Положение"' in text
    assert 'self.repair_all_btn = QPushButton("Лечить все кандидаты")' in text
    assert 'self.repair_clear_btn = QPushButton("Снять лечение")' in text
    assert 'def _on_surface_repair_item_changed' in text
    assert 'item["source_candidates"] = selected' in text


def test_plan_strength_slider_is_the_single_ai_strength_surface_and_has_reset():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Сила", "Персональная подстройка"' in text
    assert '"Совет ИИ"' not in text
    assert 'ai_strength_item = NumericTableWidgetItem' not in text
    assert 'ai_conf_item = NumericTableWidgetItem(f"{ai_conf}%"' in text
    assert 'ai_reset_btn = QPushButton("↺"' in text
    assert 'Вернуть силу, предложенную Рекомендателем параметров ИИ' in text
    assert 'Сила, которая реально будет применена к фотографии.' in text
    assert 'def _reset_plan_strength_to_ai' in text
    assert 'self.correction_ai_suggestions' in text


def test_saved_feedback_records_ai_and_user_strengths():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'user_strengths=self.correction_strengths' in text
    assert 'parameter_suggestions=self.correction_ai_suggestions' in text
    assert 'Рекомендатель параметров ИИ использует сохранённые значения' in text


def test_ai_tab_mentions_parameter_recommender_usage():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'Рекомендатель параметров ИИ v2:' in text
    assert 'Рекомендатель параметров ИИ предложил силу' in text
    assert 'персональных рекомендаций сейчас' in text


def test_ai_tab_has_training_counters_and_privacy_safe_codex_export():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'add_section_title("Обучение")' in text
    assert 'self.ai_training_export_btn = QPushButton("Подготовить пакет для обучения ИИ…")' in text
    assert 'self.ai_training_refresh_btn = QPushButton("Обновить счётчики")' in text
    assert 'collect_training_data_stats(self.db_path)' in text
    assert 'export_training_package(self.db_path, filename)' in text
    assert 'Полные пути, имена исходных фото и целые фотографии в ZIP не включены.' in text
    assert 'дефекты D {stats.surface_defect} · N {stats.surface_natural} · U {stats.surface_uncertain}' in text


def test_result_tab_exposes_analysis_reliability_guard():
    text = (ROOT / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert 'self.reliability_label = QLabel("<b>Надёжность анализа:</b> —")' in text
    assert '"ood_candidate": "Нетипичный вход"' in text
    assert 'калиброванная поддержка' in text


def test_histogram_tab_explains_linear_light_and_draws_tonal_guides():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'self.histogram_summary = QLabel("<b>Экспозиция:</b> —")' in text
    assert 'build_histogram_diagnostic(metrics)' in text
    assert '"Тени"' in text
    assert '"Средние тона"' in text
    assert '"Света"' in text
    assert '("p5", "P5"' in text
    assert '("p50", "P50"' in text
    assert '("p95", "P95"' in text
    assert 'Шкала 0…1 относится к линейному свету' in text
    assert 'пороги клиппинга анализатора' in text


def test_plan_area_cell_clears_stale_item_and_has_stable_button_widths():
    text = _text() if '_text' in globals() else (Path(__file__).parents[1] / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert "self.plan_table.removeCellWidget(row, 5)" in text
    assert "self.plan_table.takeItem(row, 5)" in text
    assert "header.setSectionResizeMode(5, QHeaderView.Interactive)" in text
    assert "self.plan_table.setColumnWidth(5, 230)" in text
    assert "area_box.setMinimumWidth(235)" in text
    assert "area_btn.setMinimumWidth(92)" in text
    assert "subject_area_btn.setMinimumWidth(82)" in text


def test_plan_region_click_uses_same_validator_lookup_as_rendering():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'def _validation_item_for_action' in text
    assert 'item = self._validation_item_for_action(key)' in text
    assert 'metrics.get("recommendation_validation")' in text
    assert 'Сейчас коррекция применяется ко всему фото.' in text
    assert 'Ограничить эту коррекцию автоматически определённым главным объектом.' in text


def test_single_photo_analysis_runs_off_gui_thread_and_reports_busy_state():
    text = SOURCE.read_text(encoding="utf-8")
    assert "class PhotoAnalysisWorker(QThread):" in text
    assert 'stage = Signal(str)' in text
    assert 'completed = Signal(object)' in text
    assert 'self.progress.setRange(0, 0)' in text
    assert 'self.progress.setFormat("Выполняется…")' in text
    assert 'self.analysis_timer.setInterval(250)' in text
    assert 'self.status_label.setText(f"⏳ {self.analysis_stage_text}' in text
    assert 'completion_message=f"Готово: анализ «{profile.label}» применён"' in text
    assert 'self.precision_combo.setEnabled(not busy and self.worker is None)' in text


def test_manual_vision_does_not_claim_reanalysis_finished_before_worker_completion():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'completion_message="Готово: ручная разметка сохранена, точный пересчёт лиц, глаз и рекомендаций завершён"' in text
    assert 'completion_message="Готово: ручные отметки удалены, автоматический анализ лиц/глаз завершён"' in text


def test_other_long_actions_use_same_busy_done_status_pattern():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'def _begin_blocking_activity(self, message: str) -> None:' in text
    assert 'def _end_blocking_activity(self, message: str, *, success: bool = True) -> None:' in text
    assert 'Подготовка предпросмотра: исправлений' in text
    assert 'Сохранение исправленной копии:' in text
    assert 'Экспорт ручной разметки дефектов…' in text
    assert 'Подготовка пакета обучающих данных ИИ…' in text
    assert '⏳ Установка ONNX Runtime…' in text


def test_wb_advisor_supports_global_fallback_and_spatial_auto_mask():
    source = SOURCE.read_text(encoding="utf-8")
    assert 'return "Баланс белого — температура / tint"' in source
    assert 'return "Баланс белого — локально по карте освещения"' in source
    assert '"spatial_white_balance_v1"' in source
    assert 'key == "white_balance" and str(validation.get("candidate", "")) != "spatial_white_balance_v1"' in source
    assert 'WB Advisor v1:' in source
    assert 'гибридный базовый советник, не нейросеть' in source


def test_surface_ai_remains_advisory_for_repair_checkboxes_and_uncertain_clears_treatment():
    text = SOURCE.read_text(encoding="utf-8")
    show_start = text.index("def _show_defects")
    show_end = text.index("def _export_surface_feedback", show_start)
    show_block = text[show_start:show_end]
    assert 'if str(box.get("user_label", "")) == "defect"' in show_block
    assert 'str(box.get("ai_label", "")) == "defect"' not in show_block

    feedback_start = text.index("def _set_surface_feedback")
    feedback_end = text.index("def _build_batch_tab", feedback_start)
    feedback_block = text[feedback_start:feedback_end]
    assert "else:" in feedback_block
    assert "self.surface_repair_signatures.discard(signature)" in feedback_block


def test_parameter_recommender_context_contains_wb_features():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _correction_feedback_context")
    end = text.index("def _record_saved_correction_feedback", start)
    block = text[start:end]
    assert '"neutral_balance"' in block
    assert '"white_balance_advisor"' in block


def test_background_workers_surface_failures_instead_of_leaving_busy_ui_stuck():
    text = SOURCE.read_text(encoding="utf-8")
    batch_start = text.index("class BatchWorker(QThread):")
    series_start = text.index("class SeriesWorker(QThread):")
    photo_start = text.index("class PhotoAnalysisWorker(QThread):")
    batch_block = text[batch_start:series_start]
    series_block = text[series_start:photo_start]
    assert "failed = Signal(str)" in batch_block
    assert 'self.failed.emit(f"{type(exc).__name__}: {exc}")' in batch_block
    assert "failed = Signal(str, str)" in series_block
    assert 'self.failed.emit(f"{type(exc).__name__}: {exc}", self.root_key)' in series_block
    assert "self.worker.failed.connect(self.on_batch_failed)" in text
    assert "self.series_worker.failed.connect(self.on_series_failed)" in text
    assert 'self.progress.setFormat("Ошибка")' in text


def test_single_analysis_reports_database_read_and_cache_write_warnings():
    text = SOURCE.read_text(encoding="utf-8")
    worker_start = text.index("class PhotoAnalysisWorker(QThread):")
    worker_end = text.index("class NumericTableWidgetItem", worker_start)
    worker_block = text[worker_start:worker_end]
    assert "warnings: list[str] = []" in worker_block
    assert "Не удалось прочитать ручную разметку из базы данных" in worker_block
    assert "Анализ выполнен, но его не удалось сохранить в базе данных" in worker_block
    assert '"warnings": warnings' in worker_block
    assert "Анализ завершён с предупреждением" in text
    assert "⚠ есть предупреждение" in text


def test_surface_repair_seed_restores_only_explicit_human_defect_labels():
    text = SOURCE.read_text(encoding="utf-8")
    assert "self._surface_repair_seeded = False" in text
    show_start = text.index("def _show_defects")
    show_end = text.index("def _export_surface_feedback", show_start)
    show = text[show_start:show_end]
    assert "if not self._surface_repair_seeded:" in show
    assert 'if str(box.get("user_label", "")) == "defect"' in show
    assert 'surface_candidate_auto_repair_eligible(box)' not in show
    assert "self._surface_repair_seeded = True" in show
    all_start = text.index("def _set_all_surface_repair")
    all_end = text.index("def _surface_validation_item", all_start)
    assert "self._surface_repair_seeded = True" in text[all_start:all_end]
    choice_start = text.index("def _on_surface_repair_item_changed")
    choice_end = text.index("def _feedback_label_text", choice_start)
    assert "self._surface_repair_seeded = True" in text[choice_start:choice_end]

def test_plan_red_eye_can_show_exact_trigger_regions():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'show_btn = QPushButton(f"Показать ({suspicious_count})"' in text
    assert 'def _show_red_eye_candidates_on_image' in text
    assert 'candidate.get("suspicious", False)' in text
    assert 'Красная рамка показывает место срабатывания, а не подтверждённый дефект.' in text


def test_plan_separates_ai_strength_confidence_and_analysis_confidence_without_duplicate_advice_column():
    text = SOURCE.read_text(encoding="utf-8")
    assert '"Сила", "Персональная подстройка"' in text
    assert '"Совет ИИ"' not in text
    assert '"Уверенность"' in text
    assert 'Сила, которая реально будет применена к фотографии.' in text
    assert 'Персональная подстройка предложенной силы по накопленным пользовательским решениям.' in text
    assert 'Уверенность анализа в том, что описанная проблема действительно присутствует.' in text


def test_plan_headers_explain_ambiguous_columns_with_tooltips():
    text = SOURCE.read_text(encoding="utf-8")
    assert 'plan_header_tooltips = {' in text
    assert 'Сила коррекции, предложенная ИИ.' in text
    assert 'Насколько рекомендация силы опирается на накопленные пользовательские решения и персональную подстройку.' in text
    assert 'Где будет применена коррекция:' in text
    assert 'Сводная очередность исправления:' in text
    assert 'Насколько заметна или существенна найденная проблема.' in text
    assert 'Насколько эту проблему можно исправить текущим методом' in text
    assert 'Насколько анализ уверен, что описанная проблема действительно присутствует' in text
    assert 'Результат проверки пробной коррекции:' in text
    assert 'header_item.setToolTip(tooltip)' in text


def test_ai_summary_names_active_surface_v2_not_legacy_v1():
    text = (ROOT / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")
    assert "Surface AI v2 + Context Meta v2" in text
    active_summary_lines = [line for line in text.splitlines() if "ai_models_summary" in line or "Surface AI v2 + Context Meta v2" in line]
    assert active_summary_lines


def test_surface_candidates_are_manual_only_even_after_ai_verification():
    text = SOURCE.read_text(encoding="utf-8")
    show_start = text.index("def _show_defects")
    show_end = text.index("def _export_surface_feedback", show_start)
    show_block = text[show_start:show_end]
    assert 'surface_candidate_auto_repair_eligible(box)' not in show_block
    assert 'if str(box.get("user_label", "")) == "defect"' in show_block
    assert 'лечение никогда не включается автоматически' in show_block
    assert 'Поставьте галочку «Лечить» сами' in show_block

    sync_start = text.index("def _sync_surface_repair_candidates")
    sync_end = text.index("def _on_surface_repair_item_changed", sync_start)
    sync_block = text[sync_start:sync_end]
    assert 'item["accepted"] = False' in sync_block
    assert 'item["auto_eligible"] = False' in sync_block
    assert 'Автолечение отключено' in sync_block

def test_plan_select_all_cannot_arm_surface_without_human_candidate_choice():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _select_all_corrections")
    end = text.index("def _clear_correction_selection", start)
    block = text[start:end]
    assert 'if key == "surface_defects"' in block
    assert 'Qt.Checked if bool(self.surface_repair_signatures) else Qt.Unchecked' in block



def test_main_toolbar_groups_open_and_save_with_icons_and_pushes_preview_right():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _build_ui")
    end = text.index("def _build_overview_tab", start) if "def _build_overview_tab" in text[start:] else text.index("def _build_menu_bar", start)
    block = text[start:end]
    assert "QStyle.StandardPixmap.SP_DialogOpenButton" in block
    assert "QStyle.StandardPixmap.SP_DialogSaveButton" in block
    assert "QStyle.StandardPixmap.SP_FileDialogContentsView" in block
    assert "toolbar_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)" in block
    assert block.index("toolbar.addWidget(self.open_file_btn)") < block.index("toolbar.addWidget(self.save_copy_btn)")
    assert block.index("toolbar.addWidget(self.precision_combo)") < block.index("toolbar.addWidget(toolbar_spacer)")
    assert block.index("toolbar.addWidget(toolbar_spacer)") < block.index("toolbar.addWidget(self.preview_btn)")


def test_metrics_rows_are_resized_after_fill_and_again_after_column_width_changes():
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("def _show_metrics")
    end = text.index("def _show_histogram", start)
    block = text[start:end]
    assert "resizeRowToContents(row_index)" not in block
    assert "self.metrics.resizeRowsToContents()" in block
    assert "self.metrics_resize_timer.start(0)" in block
    assert "header.sectionResized.connect" in text
    assert "self.metrics_resize_timer.timeout.connect(self.metrics.resizeRowsToContents)" in text


def test_technical_tab_uses_compact_raw_values_and_avoids_content_width_measurement():
    text = SOURCE.read_text(encoding="utf-8")
    build_start = text.index("def _build_technical_tab")
    build_end = text.index("def _toggle_defects", build_start)
    build = text[build_start:build_end]
    show_start = text.index("def _show_technical")
    show_end = text.index("def open_folder", show_start)
    show = text[show_start:show_end]
    assert "def _technical_compact_value" in text
    assert 'return f"{len(raw)} элементов"' in text
    assert "header.setSectionResizeMode(0, QHeaderView.ResizeToContents)" not in build
    assert "header.setSectionResizeMode(col, QHeaderView.ResizeToContents)" not in build
    assert "self.technical.setWordWrap(False)" in build
    assert "self.technical.setTextElideMode(Qt.ElideRight)" in build
    assert "self._technical_compact_value(metric.raw_value)" in show
    assert "json.dumps(localize_payload(raw)" not in show
