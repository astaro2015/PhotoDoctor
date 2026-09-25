from __future__ import annotations

import hashlib
import os
import platform
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np


PERFORMANCE_PROFILE_VERSION = 4
ProgressCallback = Callable[[int, str], None]
_ACTIVE_SERIAL_CV_THREADS = max(1, int(cv2.getNumThreads()))
_ACTIVE_PARALLEL_CV_THREADS = 1
_ACTIVE_PIPELINE_WORKERS = 1
_ACTIVE_VALIDATOR_CV_THREADS = 1
_ACTIVE_VALIDATOR_WORKERS = 1


@dataclass(frozen=True)
class PerformanceProfile:
    profile_version: int
    hardware_signature: str
    cv_threads: int
    parallel_cv_threads: int
    pipeline_workers: int
    benchmark_ms: float
    validator_cv_threads: int = 1
    validator_workers: int = 1

    def as_dict(self) -> dict[str, object]:
        return {
            "profile_version": int(self.profile_version),
            "hardware_signature": str(self.hardware_signature),
            "cv_threads": int(self.cv_threads),
            "parallel_cv_threads": int(self.parallel_cv_threads),
            "pipeline_workers": int(self.pipeline_workers),
            "benchmark_ms": float(self.benchmark_ms),
            "validator_cv_threads": int(self.validator_cv_threads),
            "validator_workers": int(self.validator_workers),
        }


def hardware_signature() -> str:
    """Return a stable, privacy-light signature for performance-relevant CPU state."""
    parts = [
        platform.system(),
        platform.machine(),
        platform.processor(),
        os.environ.get("PROCESSOR_IDENTIFIER", ""),
        str(os.cpu_count() or 1),
    ]
    payload = "\n".join(parts).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:24]


def candidate_cv_threads(logical_cpus: int | None = None) -> list[int]:
    """Conservative candidates; avoid absurd oversubscription on large CPUs."""
    logical = max(1, int(logical_cpus or os.cpu_count() or 1))
    ceiling = min(logical, 16)
    values = [1]
    value = 2
    while value <= ceiling:
        values.append(value)
        value *= 2
    if ceiling not in values:
        values.append(ceiling)
    return sorted(set(values))


def choose_best_thread_count(results: Iterable[tuple[int, float]]) -> tuple[int, float]:
    """Choose fastest result; prefer fewer threads when timings are effectively tied."""
    measured = [(max(1, int(t)), max(0.0, float(ms))) for t, ms in results]
    if not measured:
        return 1, 0.0
    fastest_ms = min(ms for _, ms in measured)
    # Within 2% of the fastest result, fewer threads are preferable: they leave
    # CPU headroom for the GUI and for the independent-analysis block.
    near_best = [(t, ms) for t, ms in measured if ms <= fastest_ms * 1.08 + 0.01]
    return min(near_best, key=lambda item: (item[0], item[1]))


def _benchmark_frame() -> np.ndarray:
    """Deterministic photo-like input without disk/network access."""
    h, w = 1024, 1536
    yy, xx = np.mgrid[0:h, 0:w]
    r = (0.36 * xx + 0.17 * yy + 31.0 * np.sin(xx / 23.0)) % 256.0
    g = (0.14 * xx + 0.41 * yy + 27.0 * np.cos(yy / 19.0)) % 256.0
    b = (0.23 * xx + 0.29 * yy + 19.0 * np.sin((xx + yy) / 17.0)) % 256.0
    return np.dstack((r, g, b)).astype(np.uint8)


def _one_benchmark_round(rgb: np.ndarray) -> float:
    start = time.perf_counter()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    _ = cv2.magnitude(gx, gy)
    _ = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    _ = cv2.resize(rgb, (1024, 683), interpolation=cv2.INTER_AREA)
    return (time.perf_counter() - start) * 1000.0


def _measure_threads(rgb: np.ndarray, threads: int, target_seconds: float = 0.35) -> float:
    cv2.setNumThreads(max(1, int(threads)))
    samples: list[float] = []
    started = time.perf_counter()
    _one_benchmark_round(rgb)  # warm-up
    while (time.perf_counter() - started) < max(0.10, float(target_seconds)) or len(samples) < 2:
        samples.append(_one_benchmark_round(rgb))
    samples.sort()
    return float(samples[len(samples) // 2])


def _measure_face_scan_threads(rgb: np.ndarray, threads: int) -> float:
    """Benchmark the dominant serial kernel using the real frontal-face cascade.

    The previous generic Sobel/Gaussian microbenchmark could prefer every logical
    CPU even when Haar face detection was measurably faster with one core left
    free.  Face/eye detection dominates the serial part of Precise mode, so use
    the actual OpenCV cascade pyramid to choose this budget.
    """
    cascade_path = Path(getattr(cv2.data, "haarcascades", "")) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        return _measure_threads(rgb, threads, target_seconds=0.20)
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return _measure_threads(rgb, threads, target_seconds=0.20)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (768, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    cv2.setNumThreads(max(1, int(threads)))

    def one() -> float:
        started = time.perf_counter()
        cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return (time.perf_counter() - started) * 1000.0

    one()  # warm-up caches / thread pool
    samples = sorted((one(), one(), one()))
    return float(samples[1])


def _pipeline_benchmark_jobs(rgb: np.ndarray):
    """Real Photo Doctor kernels used to choose the concurrent-stage policy."""
    # Lazy imports keep normal module import light and avoid pulling the analysis
    # stack into startup unless a new hardware benchmark is actually required.
    from .surface import detect_surface_defects
    from .local_sharpness import analyze_local_sharpness
    from .local_contrast import analyze_local_contrast
    from .jpeg_artifacts import analyze_jpeg_artifacts

    return (
        lambda: detect_surface_defects(rgb, max_boxes=20, max_ai_boxes=64, protection_boxes=[]),
        lambda: analyze_local_sharpness(rgb, map_rgb=rgb, precision="normal"),
        lambda: analyze_local_contrast(rgb, map_rgb=rgb, precision="normal"),
        lambda: analyze_jpeg_artifacts(rgb, "JPEG"),
    )


def _measure_pipeline_config(rgb: np.ndarray, cv_threads: int, workers: int) -> float:
    """Measure actual independent Photo Doctor stages under one scheduling policy."""
    cv2.setNumThreads(max(1, int(cv_threads)))
    workers = max(1, int(workers))
    jobs = _pipeline_benchmark_jobs(rgb)
    start = time.perf_counter()
    if workers <= 1:
        for job in jobs:
            job()
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pd-bench") as pool:
            futures = [pool.submit(job) for job in jobs]
            for future in futures:
                future.result()
    return (time.perf_counter() - start) * 1000.0

def _candidate_pipeline_configs(best_serial_threads: int, logical_cpus: int) -> list[tuple[int, int]]:
    """Candidate (OpenCV threads per task, concurrent task count) profiles."""
    logical = max(1, int(logical_cpus))
    configs: list[tuple[int, int]] = [(max(1, int(best_serial_threads)), 1)]
    if logical >= 2 and logical < 4:
        configs.append((1, 2))
    if logical >= 4:
        configs.extend([(2, 2), (1, min(4, logical))])
    if logical >= 6:
        configs.append((2, min(3, logical // 2)))
    if logical >= 8:
        configs.append((2, 4))
    return list(dict.fromkeys((max(1, cvt), max(1, workers)) for cvt, workers in configs))


def _validator_benchmark_plan() -> dict[str, object]:
    coords = [(x, y, 0.50, 0.50) for y in (0.0, 0.25, 0.50) for x in (0.0, 0.25, 0.50)]
    return {
        "exposure_cells": [
            {"x": x, "y": y, "w": w, "h": h, "delta_gamma": -0.10} for x, y, w, h in coords
        ],
        "contrast_cells": [
            {"x": x, "y": y, "w": w, "h": h, "strength": 0.42} for x, y, w, h in coords
        ],
        "sharpness_cells": [
            {"x": x, "y": y, "w": w, "h": h, "strength": 0.42} for x, y, w, h in coords
        ],
        "face_regions": [],
        "eye_regions": [],
    }


def _validator_benchmark_jobs(rgb: np.ndarray):
    """Representative independent Validator previews, excluding rare heavy NLM.

    The common Validator workload is exposure/contrast/sharpness/WB-like bounded
    previews that can run independently.  A noise action is handled more
    conservatively at runtime because NLM is unusually memory hungry.
    """
    from .local_correction_planner import apply_local_exposure, apply_local_contrast, apply_local_sharpness
    from .validator import _basic_probe, _PROBE_EXPOSURE, _PROBE_CONTRAST, _PROBE_DETAIL

    plan = _validator_benchmark_plan()
    return (
        lambda: _basic_probe(apply_local_exposure(rgb, plan, 0.80), _PROBE_EXPOSURE),
        lambda: _basic_probe(apply_local_contrast(rgb, plan, 0.80), _PROBE_CONTRAST),
        lambda: _basic_probe(apply_local_sharpness(rgb, plan, 0.82), _PROBE_DETAIL),
    )


def _measure_validator_config(rgb: np.ndarray, cv_threads: int, workers: int) -> float:
    cv2.setNumThreads(max(1, int(cv_threads)))
    workers = max(1, int(workers))
    jobs = _validator_benchmark_jobs(rgb)
    start = time.perf_counter()
    if workers <= 1:
        for job in jobs:
            job()
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pd-validator-bench") as pool:
            futures = [pool.submit(job) for job in jobs]
            for future in futures:
                future.result()
    return (time.perf_counter() - start) * 1000.0


def _candidate_validator_configs(best_serial_threads: int, logical_cpus: int) -> list[tuple[int, int]]:
    logical = max(1, int(logical_cpus))
    configs: list[tuple[int, int]] = [(max(1, int(best_serial_threads)), 1)]
    if logical >= 2 and logical < 4:
        configs.append((1, 2))
    if logical >= 4:
        configs.extend([(2, 2), (1, min(4, logical))])
    if logical >= 6:
        configs.append((2, min(3, logical // 2)))
    if logical >= 8:
        configs.append((2, 4))
    return list(dict.fromkeys((max(1, t), max(1, w)) for t, w in configs))


def choose_validator_config(results: Iterable[tuple[int, int, float]]) -> tuple[int, int, float]:
    measured = [(max(1, int(cvt)), max(1, int(w)), max(0.0, float(ms))) for cvt, w, ms in results]
    if not measured:
        return 1, 1, 0.0
    serial_rows = [row for row in measured if row[1] == 1]
    serial = min(serial_rows, key=lambda row: row[2]) if serial_rows else min(measured, key=lambda row: row[2])
    parallel = [row for row in measured if row[1] > 1]
    if not parallel:
        return serial
    fastest = min(parallel, key=lambda row: row[2])
    # Parallel validation increases temporary memory, so demand a material win.
    if fastest[2] > serial[2] * 0.88:
        return serial
    # On large real photos, more independent workers tend to scale better than
    # giving extra native OpenCV threads to each preview.  If profiles are within
    # 10% of the fastest, prefer more workers and fewer threads per worker.
    near = [row for row in parallel if row[2] <= fastest[2] * 1.10 + 0.02]
    return min(near, key=lambda row: (-row[1], row[0], row[2]))


def choose_pipeline_config(results: Iterable[tuple[int, int, float]]) -> tuple[int, int, float]:
    measured = [(max(1, int(cvt)), max(1, int(w)), max(0.0, float(ms))) for cvt, w, ms in results]
    if not measured:
        return 1, 1, 0.0

    # External task parallelism increases peak memory and can oversubscribe OpenCV
    # on real 20--30 MP photographs even when a tiny synthetic benchmark shows a
    # marginal win.  Keep the serial/native-threaded profile unless concurrency
    # demonstrates a *material* advantage.  This is a scheduling-only gate; no
    # analysis algorithm, resolution or threshold changes.
    serial_rows = [row for row in measured if row[1] == 1]
    serial = min(serial_rows, key=lambda row: row[2]) if serial_rows else min(measured, key=lambda row: row[2])
    parallel_rows = [row for row in measured if row[1] > 1]
    if not parallel_rows:
        return serial
    parallel_fastest = min(parallel_rows, key=lambda row: row[2])
    required_gain = 0.12
    if parallel_fastest[2] > serial[2] * (1.0 - required_gain):
        return serial

    fastest = parallel_fastest[2]
    near = [row for row in parallel_rows if row[2] <= fastest * 1.03 + 0.02]
    return min(near, key=lambda row: (row[0] * row[1], row[1], row[2]))


def get_pipeline_profile() -> tuple[int, int, int]:
    """Return (serial_cv_threads, parallel_cv_threads, pipeline_workers)."""
    return (
        max(1, int(_ACTIVE_SERIAL_CV_THREADS)),
        max(1, int(_ACTIVE_PARALLEL_CV_THREADS)),
        max(1, int(_ACTIVE_PIPELINE_WORKERS)),
    )


def get_pipeline_workers() -> int:
    """Backward-compatible helper used by older code/tests."""
    return get_pipeline_profile()[2]


def get_validator_profile() -> tuple[int, int]:
    """Return (OpenCV threads per Validator task, concurrent Validator tasks)."""
    return (max(1, int(_ACTIVE_VALIDATOR_CV_THREADS)), max(1, int(_ACTIVE_VALIDATOR_WORKERS)))


def benchmark_performance(progress: ProgressCallback | None = None) -> PerformanceProfile:
    """Tune serial OpenCV threads and the independent-stage scheduling policy.

    Only execution scheduling changes. Analysis thresholds, source resolution and
    algorithms remain identical.
    """
    candidates = candidate_cv_threads()
    rgb = _benchmark_frame()
    original_threads = max(1, int(cv2.getNumThreads()))
    serial_results: list[tuple[int, float]] = []
    overall_start = time.perf_counter()
    try:
        for index, threads in enumerate(candidates, start=1):
            if progress is not None:
                pct = 16 + int(50 * (index - 1) / max(1, len(candidates)))
                progress(pct, f"Проверка процессора: последовательный режим, OpenCV {threads} пот.")
            serial_results.append((threads, _measure_face_scan_threads(rgb, threads)))
        serial_threads, _serial_ms = choose_best_thread_count(serial_results)

        logical = max(1, int(os.cpu_count() or 1))
        pipeline_results: list[tuple[int, int, float]] = []
        configs = _candidate_pipeline_configs(serial_threads, logical)
        pipeline_rgb = cv2.resize(rgb, (704, 469), interpolation=cv2.INTER_AREA)
        # One untimed warm-up removes module import/cache startup from the first
        # scheduling candidate so the comparison reflects execution policy.
        cv2.setNumThreads(serial_threads)
        for job in _pipeline_benchmark_jobs(pipeline_rgb):
            job()
        for index, (parallel_threads, workers) in enumerate(configs, start=1):
            if progress is not None:
                pct = 68 + int(16 * (index - 1) / max(1, len(configs)))
                progress(pct, f"Проверка параллельного режима: {workers} задач × OpenCV {parallel_threads} пот.")
            elapsed = _measure_pipeline_config(pipeline_rgb, parallel_threads, workers)
            pipeline_results.append((parallel_threads, workers, elapsed))
        parallel_threads, workers, _parallel_ms = choose_pipeline_config(pipeline_results)

        validator_rgb = cv2.resize(rgb, (800, 533), interpolation=cv2.INTER_AREA)
        validator_results: list[tuple[int, int, float]] = []
        validator_configs = _candidate_validator_configs(serial_threads, logical)
        # The validator benchmark is intentionally focused on the common
        # exposure/contrast/sharpness preview mix. It adds only a few seconds to
        # first launch and prevents one scheduling policy from being forced on two
        # very different workloads.
        for index, (validator_threads, validator_workers) in enumerate(validator_configs, start=1):
            if progress is not None:
                pct = 84 + int(10 * (index - 1) / max(1, len(validator_configs)))
                progress(pct, f"Проверка Validator: {validator_workers} задач × OpenCV {validator_threads} пот.")
            elapsed = _measure_validator_config(validator_rgb, validator_threads, validator_workers)
            validator_results.append((validator_threads, validator_workers, elapsed))
        validator_threads, validator_workers, _validator_ms = choose_validator_config(validator_results)
        cv2.setNumThreads(serial_threads)
    except Exception:
        cv2.setNumThreads(original_threads)
        raise

    total_ms = (time.perf_counter() - overall_start) * 1000.0
    if progress is not None:
        progress(
            86,
            f"Выбран профиль: последовательно OpenCV {serial_threads} пот.; "
            f"анализ {workers} задач × {parallel_threads} пот.; "
            f"Validator {validator_workers} задач × {validator_threads} пот.",
        )
    profile = PerformanceProfile(
        profile_version=PERFORMANCE_PROFILE_VERSION,
        hardware_signature=hardware_signature(),
        cv_threads=serial_threads,
        parallel_cv_threads=parallel_threads,
        pipeline_workers=workers,
        benchmark_ms=total_ms,
        validator_cv_threads=validator_threads,
        validator_workers=validator_workers,
    )
    apply_performance_profile(profile)
    return profile


def apply_performance_profile(profile: PerformanceProfile) -> None:
    global _ACTIVE_SERIAL_CV_THREADS, _ACTIVE_PARALLEL_CV_THREADS, _ACTIVE_PIPELINE_WORKERS
    global _ACTIVE_VALIDATOR_CV_THREADS, _ACTIVE_VALIDATOR_WORKERS
    _ACTIVE_SERIAL_CV_THREADS = max(1, int(profile.cv_threads))
    _ACTIVE_PARALLEL_CV_THREADS = max(1, int(profile.parallel_cv_threads))
    _ACTIVE_PIPELINE_WORKERS = max(1, int(profile.pipeline_workers))
    _ACTIVE_VALIDATOR_CV_THREADS = max(1, int(profile.validator_cv_threads))
    _ACTIVE_VALIDATOR_WORKERS = max(1, int(profile.validator_workers))
    cv2.setNumThreads(_ACTIVE_SERIAL_CV_THREADS)
