from __future__ import annotations

import logging
import os
import sys
import traceback

from photodoctor.utils.logging_setup import configure_logging


def _native_error_box(title: str, message: str) -> None:
    """Best-effort error dialog even if Qt itself failed to import."""
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
            return
        except Exception:
            pass
    print(f"{title}: {message}", file=sys.stderr)


def main() -> int:
    log_file = configure_logging()
    splash = None
    try:
        # Show a responsive splash before importing the heavy analysis/GUI module.
        from PySide6.QtCore import QSettings
        from PySide6.QtWidgets import QApplication
        from photodoctor.gui.startup import StartupSplash, prepare_performance

        app = QApplication.instance() or QApplication([])
        splash = StartupSplash()
        splash.show_centered()
        splash.set_stage(6, "Загрузка компонентов Photo Doctor…")

        settings = QSettings("PhotoDoctor", "PhotoDoctor")
        state = prepare_performance(splash, settings)
        if state.tuned_now:
            splash.set_stage(91, "Профиль производительности сохранён.")
        else:
            splash.set_stage(82, "Профиль производительности загружен.")

        splash.set_stage(94, "Подготовка интерфейса…")
        from photodoctor.gui.main_window import MainWindow

        window = MainWindow()
        splash.set_stage(100, "Готово")
        window.show()
        splash.close()
        splash = None
        return int(app.exec())
    except Exception as exc:
        if splash is not None:
            try:
                splash.close()
            except Exception:
                pass
        logging.getLogger(__name__).exception("Не удалось запустить Photo Doctor")
        details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        _native_error_box(
            "Photo Doctor — ошибка запуска",
            f"Не удалось запустить Photo Doctor.\n\n{details}\n\nЛог: {log_file}",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
