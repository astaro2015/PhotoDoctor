from pathlib import Path

from photodoctor.core.performance import (
    _candidate_pipeline_configs,
    candidate_cv_threads,
    choose_best_thread_count,
    choose_pipeline_config,
)


ROOT = Path(__file__).parents[1]


def test_thread_candidates_are_bounded_unique_and_include_one():
    values = candidate_cv_threads(16)
    assert values == sorted(set(values))
    assert values[0] == 1
    assert values[-1] <= 16
    assert all(value >= 1 for value in values)


def test_thread_candidates_respect_small_cpu_count():
    assert candidate_cv_threads(1) == [1]
    assert candidate_cv_threads(3) == [1, 2, 3]


def test_choose_best_prefers_fewer_threads_when_effectively_tied():
    threads, elapsed = choose_best_thread_count([(1, 12.0), (2, 10.1), (4, 10.0), (8, 10.5)])
    assert threads == 2
    assert elapsed == 10.1


def test_choose_best_keeps_clear_faster_winner():
    threads, elapsed = choose_best_thread_count([(1, 20.0), (2, 14.0), (4, 10.0), (8, 13.0)])
    assert threads == 4
    assert elapsed == 10.0



def test_pipeline_candidates_include_parallel_mode_on_four_cpus():
    configs = _candidate_pipeline_configs(4, 4)
    assert (4, 1) in configs
    assert any(workers >= 2 for _threads, workers in configs)


def test_choose_pipeline_config_keeps_clear_parallel_winner():
    threads, workers, elapsed = choose_pipeline_config([
        (4, 1, 1000.0),
        (2, 2, 710.0),
        (1, 4, 760.0),
    ])
    assert (threads, workers, elapsed) == (2, 2, 710.0)


def test_choose_pipeline_config_prefers_lighter_profile_when_effectively_tied():
    threads, workers, elapsed = choose_pipeline_config([
        (4, 1, 100.0),
        (2, 2, 101.0),
        (1, 4, 101.5),
    ])
    assert (threads, workers, elapsed) == (4, 1, 100.0)

def test_startup_splash_is_shown_before_heavy_main_window_import():
    source = (ROOT / "src" / "photodoctor" / "__main__.py").read_text(encoding="utf-8")
    assert "StartupSplash" in source
    assert "splash.show_centered()" in source
    assert "prepare_performance" in source
    assert source.index("splash.show_centered()") < source.index("from photodoctor.gui.main_window import MainWindow")


def test_startup_text_explains_cpu_tuning_in_russian():
    source = (ROOT / "src" / "photodoctor" / "gui" / "startup.py").read_text(encoding="utf-8")
    assert "Оптимизация для этого компьютера" in source
    assert "Проверка процессора" in (ROOT / "src" / "photodoctor" / "core" / "performance.py").read_text(encoding="utf-8")
    assert "При первом запуске" in source
    assert 'performance/hardware_signature' in source
    assert 'performance/cv_threads' in source
    assert 'performance/parallel_cv_threads' in source
    assert 'performance/pipeline_workers' in source


def test_validator_config_prefers_material_parallel_win_and_more_workers_when_close():
    from photodoctor.core.performance import choose_validator_config
    threads, workers, elapsed = choose_validator_config([
        (4, 1, 1000.0),
        (2, 2, 620.0),
        (1, 4, 650.0),
    ])
    assert (threads, workers, elapsed) == (1, 4, 650.0)


def test_validator_config_keeps_serial_when_parallel_gain_is_small():
    from photodoctor.core.performance import choose_validator_config
    threads, workers, elapsed = choose_validator_config([
        (4, 1, 1000.0),
        (2, 2, 910.0),
        (1, 4, 900.0),
    ])
    assert (threads, workers, elapsed) == (4, 1, 1000.0)
