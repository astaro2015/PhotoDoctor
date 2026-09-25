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

from photodoctor.ai.sr_catalog import preferred_sr_candidate
from photodoctor.ai.restoration_runtime import default_model_dir
from almaz_download import download_verified
from almaz_prepare_env import ensure_export_stack



def _download_text(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "PhotoDoctor-ALMAZ/0.5.4"})
    with urlopen(request, timeout=45) as response:
        data = response.read()
    if len(data) < 1000:
        raise RuntimeError(f"Слишком короткий upstream source: {url}")
    target.write_bytes(data)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download, verify, export and install official SwinIR-S x2 for Photo Doctor ALMAZ.")
    p.add_argument("--model-dir", default="")
    p.add_argument("--cache-dir", default="")
    p.add_argument("--keep-cache", action="store_true")
    p.add_argument("--no-auto-install", action="store_true", help="Не устанавливать недостающие зависимости автоматически")
    args = p.parse_args(argv)

    ensure_export_stack(need_timm=True, auto_install=not args.no_auto_install)

    spec = preferred_sr_candidate()
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else default_model_dir().resolve()
    cache = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else (model_dir / ".almaz_prepare" / "swinir")
    cache.mkdir(parents=True, exist_ok=True)
    checkpoint = cache / spec.upstream_filename

    print("[1/4] Скачивание и SHA-проверка официального SwinIR x2...", flush=True)
    result = download_verified(
        (spec.upstream_url, *spec.upstream_mirrors), checkpoint,
        expected_sha256=spec.source_checkpoint_sha256,
        expected_size=spec.source_checkpoint_size,
    )
    print(f"      OK: {result.path.name} | {result.sha256}", flush=True)

    print("[2/4] Подготовка pinned SwinIR architecture...", flush=True)
    repo = cache / "SwinIR-pinned"
    source = repo / "models" / "network_swinir.py"
    source_url = f"https://raw.githubusercontent.com/JingyunLiang/SwinIR/{spec.source_commit}/models/network_swinir.py"
    _download_text(source_url, source)
    (source.parent / "__init__.py").write_text("", encoding="utf-8")

    print("[3/4] Экспорт ONNX + PyTorch<->ONNX parity...", flush=True)
    out = cache / spec.planned_onnx_filename
    cmd = [
        sys.executable, str(TOOLS / "export_almaz_swinir.py"),
        "--swinir-repo", str(repo), "--checkpoint", str(checkpoint), "--output", str(out),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    print("[4/4] Установка только verified ONNX...", flush=True)
    subprocess.run([
        sys.executable, str(TOOLS / "install_almaz_onnx.py"), str(out), "--model-dir", str(model_dir)
    ], check=True, cwd=str(ROOT))
    print(f"ALMAZ x2 готов: {model_dir / spec.planned_onnx_filename}", flush=True)

    if not args.keep_cache:
        # Keep the verified checkpoint for cheap re-export; delete only source clone fragments.
        for child in sorted(repo.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink(): child.unlink(missing_ok=True)
            elif child.is_dir(): child.rmdir()
        repo.rmdir() if repo.exists() else None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
