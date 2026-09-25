from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "src" / "photodoctor" / "gui" / "main_window.py").read_text(encoding="utf-8")


def test_almaz_ai_restoration_is_presented_as_full_frame_only():
    source = _source()
    assert 'candidate.startswith("almaz_ai_")' in source
    assert 'QTableWidgetItem("Всё фото · ALMAZ")' in source
    assert "локальный AI-масочный режим пока намеренно отключён" in source


def test_almaz_is_available_inside_diagnostics_with_all_four_functions():
    source = _source()
    assert 'for title in ("Метрики", "Гистограмма", "Рекомендации", "ALMAZ", "ИИ", "Технические данные"):' in source
    assert '"Super Resolution x2"' in source
    assert '"AI Denoise"' in source
    assert '"AI Deblur"' in source
    assert '"JPEG Recovery"' in source
    assert 'self._build_almaz_tab()' in source


def test_almaz_tab_exposes_model_setup_and_runtime_controls():
    source = _source()
    assert 'self.almaz_runtime_main_btn = QPushButton("Установить ONNX Runtime")' in source
    assert 'self.almaz_prepare_all_btn = QPushButton("Подготовить Denoise + Deblur + JPEG Recovery")' in source
    assert 'self._prepare_almaz_restoration("denoise")' in source
    assert 'self._prepare_almaz_restoration("deblur")' in source
    assert 'self._prepare_almaz_restoration("jpeg_recovery")' in source
    assert 'def _refresh_almaz_panel' in source
    assert 'def _prepare_almaz_restoration' in source


def test_almaz_tab_links_back_to_plan_and_shows_current_photo_state():
    source = _source()
    assert 'self.almaz_open_plan_btn = QPushButton("Перейти к исправлениям")' in source
    assert 'self.tabs.setCurrentWidget(self.plan_page)' in source
    assert 'self.almaz_current_photo' in source
    assert 'ALMAZ-коррекции для этого фото сейчас не предложены' in source


def test_almaz_prepare_failure_surfaces_real_log_instead_of_only_exit_code():
    source = _source()
    assert 'def _almaz_prepare_error_detail' in source
    assert '"ALMAZ — ошибка подготовки модели"' in source
    assert 'detail = self._almaz_prepare_error_detail()' in source


def test_almaz_prepare_process_forces_utf8_for_windows_qprocess_children():
    source = _source()
    assert "QProcessEnvironment" in source
    assert 'env.insert("PYTHONUTF8", "1")' in source
    assert 'env.insert("PYTHONIOENCODING", "utf-8")' in source
    assert source.count("self._set_almaz_process_utf8(process)") >= 2


def test_almaz_long_operations_show_real_stage_and_elapsed_time():
    source = _source()
    assert 'self.preview_timer.timeout.connect(self._update_almaz_preview_elapsed)' in source
    assert 'def _almaz_preview_stage_changed' in source
    assert 'def _update_almaz_preview_elapsed' in source
    assert 'self.almaz_prepare_timer.timeout.connect(self._update_almaz_prepare_elapsed)' in source
    assert 'def _set_almaz_prepare_stage' in source
    assert 'def _update_almaz_prepare_elapsed' in source
    assert 'self._last_process_stage(data)' in source
    assert 'elapsed:.1f' in source


def test_plan_exposes_three_almaz_x1_modules_as_independent_rows():
    source = _source()
    assert '"almaz_denoise": "ALMAZ — AI Denoise"' in source
    assert '"almaz_deblur": "ALMAZ — AI Deblur"' in source
    assert '"almaz_jpeg_recovery": "ALMAZ — JPEG Recovery"' in source
    assert "Standalone ALMAZ modules are manual diagnostic/restoration actions" in source


def test_almaz_tab_exposes_trained_release_install_and_rollback_controls():
    source = _source()
    assert 'self.almaz_install_release_btn = QPushButton("Установить обученную модель…")' in source
    assert 'self.almaz_rollback_release_btn = QPushButton("Откатить модель…")' in source
    assert 'def _install_almaz_release_bundle' in source
    assert 'def _rollback_almaz_release_model' in source
    assert 'AlmazReleaseWorker("install"' in source
    assert 'AlmazReleaseWorker("rollback"' in source


def test_almaz_release_worker_reports_real_stage_and_elapsed_time():
    source = _source()
    assert 'class AlmazReleaseWorker(QThread)' in source
    assert 'self.almaz_release_timer.timeout.connect(self._update_almaz_release_elapsed)' in source
    assert 'def _almaz_release_stage' in source
    assert 'def _update_almaz_release_elapsed' in source
    assert 'ALMAZ модели: {stage} · {elapsed:.1f} с' in source
    assert 'progress=on_progress' in source
