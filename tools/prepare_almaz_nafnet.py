from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TOOLS = Path(__file__).resolve().parent
for entry in (SRC, TOOLS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from photodoctor.ai.restoration_catalog import preferred_restoration_candidate
from photodoctor.ai.restoration_runtime import default_model_dir
from almaz_download import download_verified
from almaz_prepare_env import ensure_export_stack


TASKS = ("denoise", "deblur", "jpeg_recovery")



def _download_text(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "PhotoDoctor-ALMAZ/0.5.4"})
    with urlopen(request, timeout=45) as response:
        data = response.read()
    if len(data) < 500:
        raise RuntimeError(f"Слишком короткий upstream source: {url}")
    target.write_bytes(data)


def _prepare_repo(root: Path, commit: str) -> Path:
    repo = root / f"NAFNet-{commit[:8]}"
    files = {
        "basicsr/models/archs/NAFNet_arch.py": "basicsr/models/archs/NAFNet_arch.py",
        "basicsr/models/archs/local_arch.py": "basicsr/models/archs/local_arch.py",
        "basicsr/models/archs/arch_util.py": "basicsr/models/archs/arch_util.py",
    }
    for relative, remote in files.items():
        target = repo / relative
        url = f"https://raw.githubusercontent.com/megvii-research/NAFNet/{commit}/{remote}"
        _download_text(url, target)
    for package in (repo / "basicsr", repo / "basicsr/models", repo / "basicsr/models/archs"):
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").touch()
    utils = repo / "basicsr/utils"
    utils.mkdir(parents=True, exist_ok=True)
    (utils / "__init__.py").write_text(
        "import logging\n\ndef get_root_logger(*args, **kwargs):\n    return logging.getLogger('basicsr')\n",
        encoding="utf-8",
    )
    return repo


def _prepare_one(task: str, model_dir: Path, cache: Path) -> None:
    spec = preferred_restoration_candidate(task)
    task_cache = cache / task
    task_cache.mkdir(parents=True, exist_ok=True)
    checkpoint = task_cache / spec.upstream_filename
    print(f"[{task}] Скачивание + SHA-проверка {spec.upstream_filename}...", flush=True)
    download_verified(
        spec.download_urls, checkpoint,
        expected_sha256=spec.source_checkpoint_sha256,
        expected_size=spec.source_checkpoint_size,
    )
    repo = _prepare_repo(cache / "source", spec.source_commit)
    out = task_cache / spec.planned_onnx_filename
    print(f"[{task}] Экспорт ONNX + parity...", flush=True)
    subprocess.run([
        sys.executable, str(TOOLS / "export_almaz_nafnet.py"),
        "--task", task, "--nafnet-repo", str(repo), "--checkpoint", str(checkpoint), "--output", str(out),
    ], check=True, cwd=str(ROOT))
    subprocess.run([
        sys.executable, str(TOOLS / "install_almaz_onnx.py"), str(out), "--model-dir", str(model_dir)
    ], check=True, cwd=str(ROOT))
    print(f"[{task}] Готово: {model_dir / spec.planned_onnx_filename}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download, verify, export and install pinned NAFNet models for Photo Doctor ALMAZ.")
    p.add_argument("--task", choices=(*TASKS, "all"), default="all")
    p.add_argument("--model-dir", default="")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--no-auto-install", action="store_true", help="Не устанавливать недостающие зависимости автоматически")
    args = p.parse_args(argv)
    ensure_export_stack(need_timm=False, auto_install=not args.no_auto_install)
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else default_model_dir().resolve()
    cache = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else (model_dir / ".almaz_prepare" / "nafnet")
    selected = TASKS if args.task == "all" else (args.task,)
    for task in selected:
        _prepare_one(task, model_dir, cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
