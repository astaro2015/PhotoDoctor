from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from almaz_training_contract import load_contract, load_dataset_stats, validate_dataset_stats

ROOT = Path(__file__).resolve().parents[1]


def _cuda_status() -> dict[str, object]:
    try:
        import torch  # type: ignore
        available = bool(torch.cuda.is_available())
        return {
            "torch": str(torch.__version__),
            "cuda_available": available,
            "cuda_device_count": int(torch.cuda.device_count()) if available else 0,
            "cuda_device": torch.cuda.get_device_name(0) if available else "",
        }
    except Exception as exc:
        return {"torch": "unavailable", "cuda_available": False, "error": str(exc)}


def _print_stage(text: str, started: float) -> None:
    print(f"[ALMAZ TRAIN] {text} | {time.monotonic() - started:.1f} s", flush=True)


def build_plan(dataset_root: Path, run_root: Path) -> dict[str, object]:
    return {
        "dataset_root": str(dataset_root.resolve()),
        "run_root": str(run_root.resolve()),
        "tasks": ["denoise", "deblur", "jpeg_recovery", "sr_x2", "archive_restore"],
        "policy": "fine-tune pretrained task-specific weights; never train release model from smoke data",
        "checkpoint_selection": "closed exam + ALMAZ safety gate; PSNR alone is insufficient",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Photo Doctor ALMAZ Training Coach")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--run-root", default="training_runs/almaz")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--execute", action="store_true", help="Execute trainer commands from run_plan.json. Default is validation/planning only.")
    p.add_argument("--trainer", action="append", default=[], help="Explicit trainer command. Repeat for multiple tasks.")
    args = p.parse_args(argv)

    started = time.monotonic()
    dataset_root = Path(args.dataset_root)
    run_root = Path(args.run_root)
    stats_path = dataset_root / "dataset_stats.json"
    if not stats_path.is_file():
        print(f"ERROR: missing {stats_path}")
        return 2
    _print_stage("проверяю объём и разделение train/val/exam", started)
    stats = load_dataset_stats(stats_path)
    failures = validate_dataset_stats(stats, load_contract())
    if failures:
        print("DATASET GATE: FAIL")
        for item in failures:
            print(f" - {item}")
        print("Smoke/малый набор не допускается к релизному обучению.")
        return 3

    hw = _cuda_status()
    _print_stage("проверяю ускоритель", started)
    print(json.dumps(hw, ensure_ascii=False, indent=2))
    if not hw.get("cuda_available") and not args.allow_cpu:
        print("ERROR: CUDA GPU not available. Release fine-tuning is intentionally not started on CPU.")
        return 4

    run_root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(dataset_root, run_root)
    plan.update({"hardware": hw, "host": platform.platform(), "python": sys.version})
    plan_path = run_root / "run_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_stage(f"план сохранён: {plan_path}", started)

    if not args.execute:
        print("PLAN ONLY: dataset passed. Use --execute with explicit --trainer commands after trainer review.")
        return 0
    if not args.trainer:
        print("ERROR: --execute requires at least one explicit --trainer command")
        return 5

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    for index, command in enumerate(args.trainer, 1):
        _print_stage(f"запускаю trainer {index}/{len(args.trainer)}: {command}", started)
        subprocess.run(command, shell=True, check=True, cwd=str(ROOT), env=env)
    _print_stage("обучающие процессы завершены; checkpoint ещё должен пройти закрытый экзамен", started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
