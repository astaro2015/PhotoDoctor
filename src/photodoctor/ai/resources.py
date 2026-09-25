from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from functools import lru_cache
from dataclasses import asdict, dataclass
from typing import Any

_GIB = 1024 ** 3


@dataclass(frozen=True, slots=True)
class SystemResources:
    logical_cpus: int
    total_ram_bytes: int
    available_ram_bytes: int
    cpu_name: str = ""
    gpu_name: str | None = None
    gpu_vram_bytes: int | None = None

    @property
    def total_ram_gib(self) -> float:
        return self.total_ram_bytes / _GIB

    @property
    def available_ram_gib(self) -> float:
        return self.available_ram_bytes / _GIB


@dataclass(frozen=True, slots=True)
class AIResourcePlan:
    key: str
    label: str
    reserve_for_os_bytes: int
    working_memory_budget_bytes: int
    patch_batch_size: int
    max_surface_candidates: int
    max_model_runs: int
    spatial_ai_long_side: int
    cpu_threads: int
    reason: str

    @property
    def reserve_for_os_gib(self) -> float:
        return self.reserve_for_os_bytes / _GIB

    @property
    def working_memory_budget_gib(self) -> float:
        return self.working_memory_budget_bytes / _GIB

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["reserve_for_os_gib"] = round(self.reserve_for_os_gib, 2)
        data["working_memory_budget_gib"] = round(self.working_memory_budget_gib, 2)
        return data


def _memory_windows() -> tuple[int, int] | None:
    if os.name != "nt":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except Exception:
        return None
    if not ok:
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _memory_posix() -> tuple[int, int] | None:
    try:
        page = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        avail_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    if min(page, total_pages, avail_pages) <= 0:
        return None
    return page * total_pages, page * avail_pages


def _cpu_name() -> str:
    name = (platform.processor() or "").strip()
    if name:
        return name
    return (platform.machine() or "Unknown CPU").strip()


@lru_cache(maxsize=1)
def _nvidia_info() -> tuple[str | None, int | None]:
    # Best effort only. No GPU dependency is required for Photo Doctor.
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if result.returncode != 0:
        return None, None
    line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not line or "," not in line:
        return None, None
    name, memory = [part.strip() for part in line.rsplit(",", 1)]
    try:
        mib = float(memory)
    except ValueError:
        return name or None, None
    return name or None, int(mib * 1024 * 1024)


def detect_system_resources() -> SystemResources:
    memory = _memory_windows() or _memory_posix() or (8 * _GIB, 4 * _GIB)
    gpu_name, gpu_vram = _nvidia_info()
    return SystemResources(
        logical_cpus=max(1, int(os.cpu_count() or 1)),
        total_ram_bytes=max(1, int(memory[0])),
        available_ram_bytes=max(1, int(memory[1])),
        cpu_name=_cpu_name(),
        gpu_name=gpu_name,
        gpu_vram_bytes=gpu_vram,
    )


def _os_reserve_bytes(total: int) -> int:
    gib = total / _GIB
    if gib < 20:
        reserve_gib = max(4.5, gib * 0.34)
    elif gib < 48:
        reserve_gib = max(6.0, gib * 0.25)
    else:
        reserve_gib = max(8.0, gib * 0.20)
    # Never reserve so much that less than 2 GiB remains for Photo Doctor.
    return int(min(reserve_gib * _GIB, max(0, total - 2 * _GIB)))


def choose_ai_resource_plan(resources: SystemResources) -> AIResourcePlan:
    total = resources.total_ram_bytes
    available = resources.available_ram_bytes
    reserve = _os_reserve_bytes(total)
    # Respect both physical size and current free memory. Leave another 1.5 GiB
    # of live headroom so a temporary Windows/browser spike does not cause swap.
    usable_by_total = max(512 * 1024 ** 2, total - reserve)
    usable_by_free = max(512 * 1024 ** 2, available - int(1.5 * _GIB))
    usable = min(usable_by_total, usable_by_free)
    working = int(max(512 * 1024 ** 2, usable * 0.72))

    threads = resources.logical_cpus
    total_gib = resources.total_ram_gib
    usable_gib = usable / _GIB

    if total_gib >= 48 and usable_gib >= 24 and threads >= 8:
        key, label = "maximum", "Максимально"
        batch, candidates, runs, spatial = 64, 64, 5, 4096
        reason = "48+ ГБ ОЗУ и многопоточный процессор: разрешён максимальный профиль процессора."
    elif total_gib >= 28 and usable_gib >= 14 and threads >= 8:
        key, label = "extended", "Расширенно"
        batch, candidates, runs, spatial = 32, 40, 4, 3584
        reason = "32+ ГБ ОЗУ: увеличен размер пакета и число одновременно уточняемых кандидатов."
    elif total_gib >= 15 and usable_gib >= 7 and threads >= 8:
        key, label = "standard", "Нормально"
        batch, candidates, runs, spatial = 20, 28, 3, 3072
        reason = "Базовый профиль для 8-ядерного процессора и 16 ГБ ОЗУ; SMT не требуется."
    else:
        key, label = "conservative", "Бережно"
        batch, candidates, runs, spatial = 8, 14, 2, 2048
        reason = "Ресурсов меньше базовой цели; уменьшен размер пакета и рабочее разрешение ИИ."

    if resources.gpu_vram_bytes and resources.gpu_vram_bytes >= 8 * _GIB:
        reason += " Обнаружен видеокарта с 8+ ГБ видеопамяти; тяжёлые модели для видеокарты можно подключать отдельно."

    return AIResourcePlan(
        key=key,
        label=label,
        reserve_for_os_bytes=reserve,
        working_memory_budget_bytes=working,
        patch_batch_size=batch,
        max_surface_candidates=candidates,
        max_model_runs=runs,
        spatial_ai_long_side=spatial,
        cpu_threads=max(1, min(threads, 16 if key == "standard" else threads)),
        reason=reason,
    )


def resource_snapshot(resources: SystemResources | None = None) -> dict[str, Any]:
    resources = resources or detect_system_resources()
    plan = choose_ai_resource_plan(resources)
    return {
        "logical_cpus": resources.logical_cpus,
        "cpu_name": resources.cpu_name,
        "total_ram_gib": round(resources.total_ram_gib, 2),
        "available_ram_gib": round(resources.available_ram_gib, 2),
        "gpu_name": resources.gpu_name,
        "gpu_vram_gib": None if resources.gpu_vram_bytes is None else round(resources.gpu_vram_bytes / _GIB, 2),
        "plan": plan.to_dict(),
    }
