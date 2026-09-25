from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "photodoctor" / "gui" / "main_window.py"


def _text() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_preview_has_persistent_cache_and_recipe_revision():
    text = _text()
    assert "self._preview_cache_rgb = None" in text
    assert "self._preview_cache_key" in text
    assert "self._preview_recipe_revision" in text
    assert "def _invalidate_preview_cache" in text
    assert "def _get_or_build_preview" in text


def test_toggle_and_save_reuse_cached_preview():
    text = _text()
    toggle = text[text.index("    def _toggle_preview"): text.index("    def _save_preview_copy")]
    save = text[text.index("    def _save_preview_copy"): text.index("    def _update_analysis_elapsed")]
    assert "self._get_or_build_preview(selected)" in toggle
    assert "apply_selected_preview(" not in toggle
    assert "self._get_or_build_preview(selected)" in save


def test_recipe_changes_mark_preview_stale_without_silent_recalculation():
    text = _text()
    block = text[text.index("    def _update_correction_controls"): text.index("    def _enforce_auto_trio_exclusivity")]
    assert "if refresh_preview:" in block
    assert "self._mark_preview_stale_after_recipe_change()" in block
    assert "self._get_or_build_preview(selected)" not in block


def test_stale_preview_requires_explicit_button_refresh():
    text = _text()
    block = text[text.index("    def _mark_preview_stale_after_recipe_change"): text.index("    def _preview_key")]
    assert "self._invalidate_preview_cache()" in block
    assert 'self.preview_btn.setChecked(False)' in block
    assert 'self.image_view.set_rgb(self.current_rgb)' in block
    assert 'self.preview_btn.setText("Обновить предпросмотр")' in block
    assert 'Нажмите «Обновить предпросмотр» для перерасчёта' in block
    assert "_get_or_build_preview" not in block


def test_clearing_last_correction_restores_source_and_resets_refresh_state():
    text = _text()
    block = text[text.index("    def _mark_preview_stale_after_recipe_change"): text.index("    def _preview_key")]
    empty = block[block.index("        if not selected:"): block.index("        if had_rendered_preview:")]
    assert 'self.preview_btn.setChecked(False)' in empty
    assert 'self.image_view.set_rgb(self.current_rgb)' in empty
    assert 'self._preview_is_stale = False' in empty
    assert 'self.preview_btn.setText("Предпросмотр")' in empty
    assert 'Все исправления сняты. Показан исходник.' in empty


def test_successful_preview_clears_stale_refresh_label():
    text = _text()
    toggle = text[text.index("    def _toggle_preview"): text.index("    def _save_preview_copy")]
    completed = text[text.index("    def _almaz_preview_completed"): text.index("    def _almaz_preview_failed")]
    assert 'self._preview_is_stale = False' in toggle
    assert 'self.preview_btn.setText("Предпросмотр")' in toggle
    assert 'self._preview_is_stale = False' in completed
    assert 'self.preview_btn.setText("Предпросмотр")' in completed


def test_finished_almaz_result_from_old_recipe_is_discarded_cleanly_when_selection_is_empty():
    text = _text()
    completed = text[text.index("    def _almaz_preview_completed"): text.index("    def _almaz_preview_failed")]
    assert "if current_selected:" in completed
    assert 'self.image_view.set_rgb(self.current_rgb)' in completed
    assert 'self._preview_is_stale = False' in completed
    assert 'self.preview_btn.setText("Предпросмотр")' in completed
    assert 'Старый ALMAZ-расчёт отброшен: все исправления сняты.' in completed


def test_failed_almaz_result_from_old_recipe_does_not_show_obsolete_error_dialog():
    text = _text()
    failed = text[text.index("    def _almaz_preview_failed"): text.index("    def _almaz_preview_stage_changed")]
    assert "current_key = self._preview_key(current_selected)" in failed
    assert "if key != current_key:" in failed
    stale_branch = failed[failed.index("        if key != current_key:"): failed.index('        self._end_blocking_activity(f"ALMAZ-предпросмотр не построен')]
    assert "QMessageBox.warning" not in stale_branch
    assert "QMessageBox.critical" not in stale_branch
    assert 'self.preview_btn.setText("Обновить предпросмотр")' in stale_branch
    assert 'self.preview_btn.setText("Предпросмотр")' in stale_branch


def test_new_analysis_invalidates_preview_cache():
    text = _text()
    marker = "self.current_rgb = loaded.srgb.copy()"
    pos = text.index(marker)
    nearby = text[pos: pos + 300]
    assert "self._invalidate_preview_cache()" in nearby


def test_almaz_preview_runs_in_background_worker():
    text = _text()
    assert "class CorrectionPreviewWorker(QThread)" in text
    assert "def _start_almaz_preview" in text
    toggle = text[text.index("    def _toggle_preview"): text.index("    def _save_preview_copy")]
    assert "def _selection_requires_almaz_worker" in text
    assert "elif self._selection_requires_almaz_worker(set(selected)):" in toggle
    assert "candidate.startswith(\"almaz_ai_\")" in text
    assert "self._start_almaz_preview(set(selected))" in toggle
    worker = text[text.index("class CorrectionPreviewWorker"): text.index("class NumericTableWidgetItem")]
    assert "apply_selected_preview(" in worker


def test_almaz_save_waits_for_cached_preview_instead_of_blocking_ui():
    text = _text()
    save = text[text.index("    def _save_preview_copy"): text.index("    def _update_analysis_elapsed")]
    assert "if self._selection_requires_almaz_worker(set(selected)):" in save
    assert "self._preview_cache_rgb is None" in save
    assert "self._start_almaz_preview(set(selected))" in save


def test_close_event_does_not_kill_running_almaz_worker():
    text = _text()
    block = text[text.index("    def closeEvent"): text.index("    def open_file")]
    assert "self.preview_worker" in block
    assert "self.preview_worker.wait(5000)" in block
    assert "event.ignore()" in block


def test_preview_key_contains_effective_recipe_not_only_revision():
    text = _text()
    block = text[text.index("    def _preview_key"): text.index("    def _get_or_build_preview")]
    assert "build_preview_recipe_signature(" in block
    assert "self.correction_strengths" in block
    assert "self.correction_regions" in block
    assert "self.current_validation_items" in block


def test_almaz_worker_reports_real_stage_and_elapsed_time():
    text = _text()
    worker = text[text.index("class CorrectionPreviewWorker"): text.index("class NumericTableWidgetItem")]
    assert "stage = Signal(str)" in worker
    assert "progress=on_progress" in worker
    assert "self.stage.emit(text)" in worker
    assert "self.preview_timer" in text
    assert "def _almaz_preview_stage_changed" in text
    assert "def _update_almaz_preview_elapsed" in text
    assert "elapsed:.1f" in text
