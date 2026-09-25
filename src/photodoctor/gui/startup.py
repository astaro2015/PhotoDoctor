from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from photodoctor import __version__
from photodoctor.resources import asset_path


@dataclass(frozen=True)
class StartupPerformanceState:
    cv_threads: int
    parallel_cv_threads: int
    pipeline_workers: int
    validator_cv_threads: int
    validator_workers: int
    tuned_now: bool
    benchmark_ms: float


class StartupSplash(QWidget):
    """Small responsive startup window shown before heavy imports/CPU tuning."""

    def __init__(self) -> None:
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setObjectName("startupSplash")
        self.setFixedSize(600, 330)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        card = QFrame(self)
        card.setObjectName("startupCard")
        outer.addWidget(card)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(18)
        logo = QLabel(card)
        logo.setFixedSize(92, 92)
        logo_path = asset_path("photodoctor.png")
        if logo_path.is_file():
            pix = QPixmap(str(logo_path))
            if not pix.isNull():
                logo.setPixmap(
                    pix.scaled(92, 92, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
        top.addWidget(logo, 0, Qt.AlignTop)

        title_box = QVBoxLayout()
        title_box.setSpacing(3)
        title = QLabel("Photo Doctor", card)
        font = QFont()
        font.setPointSize(23)
        font.setBold(True)
        title.setFont(font)
        subtitle = QLabel(f"Версия {__version__}", card)
        subtitle.setObjectName("startupMuted")
        self.headline = QLabel("Подготовка программы", card)
        headline_font = QFont()
        headline_font.setPointSize(11)
        headline_font.setBold(True)
        self.headline.setFont(headline_font)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        title_box.addSpacing(7)
        title_box.addWidget(self.headline)
        top.addLayout(title_box, 1)
        layout.addLayout(top)
        layout.addStretch(1)

        self.status = QLabel("Запуск…", card)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.progress = QProgressBar(card)
        self.progress.setRange(0, 100)
        self.progress.setValue(3)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        layout.addWidget(self.progress)

        self.detail = QLabel(
            "При первом запуске Photo Doctor подбирает быстрый профиль для этого компьютера.",
            card,
        )
        self.detail.setObjectName("startupMuted")
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)

        self.setStyleSheet(
            """
            #startupCard {
                background: #202225;
                border: 1px solid #45484d;
                border-radius: 14px;
            }
            QLabel { color: #f2f2f2; background: transparent; }
            QLabel#startupMuted { color: #aeb2b8; }
            QProgressBar {
                background: #35383d;
                border: none;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: #64a6ff;
                border-radius: 3px;
            }
            """
        )
        icon_path = asset_path("photodoctor.ico")
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))

    def show_centered(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            self.move(
                area.x() + (area.width() - self.width()) // 2,
                area.y() + (area.height() - self.height()) // 2,
            )
        self.show()
        self.raise_()
        QApplication.processEvents()

    def set_stage(self, value: int, status: str, headline: str | None = None) -> None:
        if headline:
            self.headline.setText(headline)
        self.status.setText(status)
        self.progress.setValue(max(0, min(100, int(value))))
        QApplication.processEvents()


def _settings_profile(settings: QSettings):
    from photodoctor.core.performance import PERFORMANCE_PROFILE_VERSION, PerformanceProfile, hardware_signature

    stored_version = int(settings.value("performance/profile_version", 0) or 0)
    stored_signature = str(settings.value("performance/hardware_signature", "") or "")
    stored_threads = int(settings.value("performance/cv_threads", 0) or 0)
    stored_parallel_threads = int(settings.value("performance/parallel_cv_threads", 0) or 0)
    stored_workers = int(settings.value("performance/pipeline_workers", 0) or 0)
    stored_validator_threads = int(settings.value("performance/validator_cv_threads", 0) or 0)
    stored_validator_workers = int(settings.value("performance/validator_workers", 0) or 0)
    if (
        stored_version != PERFORMANCE_PROFILE_VERSION
        or stored_signature != hardware_signature()
        or stored_threads < 1
        or stored_parallel_threads < 1
        or stored_workers < 1
        or stored_validator_threads < 1
        or stored_validator_workers < 1
    ):
        return None
    return PerformanceProfile(
        profile_version=stored_version,
        hardware_signature=stored_signature,
        cv_threads=stored_threads,
        parallel_cv_threads=stored_parallel_threads,
        pipeline_workers=stored_workers,
        benchmark_ms=float(settings.value("performance/benchmark_ms", 0.0) or 0.0),
        validator_cv_threads=stored_validator_threads,
        validator_workers=stored_validator_workers,
    )


def prepare_performance(splash: StartupSplash, settings: QSettings) -> StartupPerformanceState:
    from photodoctor.core.performance import apply_performance_profile, benchmark_performance

    profile = _settings_profile(settings)
    if profile is not None:
        splash.set_stage(
            64,
            f"Профиль: анализ OpenCV {profile.cv_threads} пот.; "
            f"Validator {profile.validator_workers} задач × {profile.validator_cv_threads} пот.",
        )
        apply_performance_profile(profile)
        return StartupPerformanceState(
            profile.cv_threads, profile.parallel_cv_threads, profile.pipeline_workers,
            profile.validator_cv_threads, profile.validator_workers, False, profile.benchmark_ms
        )

    splash.set_stage(12, "Первый запуск: подбираю настройки процессора…", "Оптимизация для этого компьютера")

    def update(value: int, text: str) -> None:
        splash.set_stage(value, text, "Оптимизация для этого компьютера")

    try:
        profile = benchmark_performance(progress=update)
    except Exception:
        # Startup tuning is an optimisation, never a launch requirement.
        import cv2
        import os
        from photodoctor.core.performance import PERFORMANCE_PROFILE_VERSION, PerformanceProfile, hardware_signature

        safe_threads = max(1, min(4, int(os.cpu_count() or 1)))
        cv2.setNumThreads(safe_threads)
        profile = PerformanceProfile(
            profile_version=PERFORMANCE_PROFILE_VERSION,
            hardware_signature=hardware_signature(),
            cv_threads=safe_threads,
            parallel_cv_threads=1,
            pipeline_workers=1,
            benchmark_ms=0.0,
            validator_cv_threads=1,
            validator_workers=1,
        )
        splash.set_stage(84, f"Используется безопасный профиль: OpenCV {safe_threads} пот.")

    settings.setValue("performance/profile_version", profile.profile_version)
    settings.setValue("performance/hardware_signature", profile.hardware_signature)
    settings.setValue("performance/cv_threads", profile.cv_threads)
    settings.setValue("performance/parallel_cv_threads", profile.parallel_cv_threads)
    settings.setValue("performance/pipeline_workers", profile.pipeline_workers)
    settings.setValue("performance/validator_cv_threads", profile.validator_cv_threads)
    settings.setValue("performance/validator_workers", profile.validator_workers)
    settings.setValue("performance/benchmark_ms", round(profile.benchmark_ms, 3))
    settings.sync()
    splash.set_stage(
        88,
        f"Настройка завершена: анализ OpenCV {profile.cv_threads} пот.; "
        f"Validator {profile.validator_workers} задач × {profile.validator_cv_threads} пот.",
    )
    return StartupPerformanceState(
        profile.cv_threads, profile.parallel_cv_threads, profile.pipeline_workers,
        profile.validator_cv_threads, profile.validator_workers, True, profile.benchmark_ms
    )
