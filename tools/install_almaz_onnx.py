from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from photodoctor.ai.restoration_catalog import restoration_candidates
from photodoctor.ai.sr_catalog import preferred_sr_candidate
from photodoctor.ai.restoration_runtime import default_model_dir
from almaz_export_common import sha256_file


def _known_specs():
    sr = preferred_sr_candidate()
    specs = {sr.model_id: (sr.planned_onnx_filename, sr.contract_id, None, sr.source_checkpoint_sha256, sr.source_checkpoint_size, sr.source_commit)}
    for spec in restoration_candidates():
        specs[spec.model_id] = (
            spec.planned_onnx_filename, spec.contract_id, spec.task,
            spec.source_checkpoint_sha256, spec.source_checkpoint_size, spec.source_commit,
        )
    return specs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Install a parity-verified Photo Doctor ALMAZ ONNX + manifest pair.')
    p.add_argument('onnx')
    p.add_argument('--model-dir', default='')
    args = p.parse_args(argv)
    src = Path(args.onnx).expanduser().resolve()
    manifest = src.with_suffix(src.suffix + '.json')
    if not src.is_file() or not manifest.is_file():
        raise SystemExit('Both .onnx and sibling .onnx.json manifest are required.')
    try:
        meta = json.loads(manifest.read_text(encoding='utf-8'))
    except Exception as exc:
        raise SystemExit(f'Invalid manifest: {exc}') from exc
    model_id = str(meta.get('model_id', ''))
    known = _known_specs()
    if model_id not in known:
        raise SystemExit(f'Unknown ALMAZ model_id: {model_id}')
    filename, contract_id, task, source_sha, source_size, source_commit = known[model_id]
    if str(meta.get('contract_id', '')) != contract_id:
        raise SystemExit('Manifest contract_id does not match Photo Doctor catalog.')
    if task is not None and str(meta.get('task', '')) != task:
        raise SystemExit('Manifest task does not match Photo Doctor catalog.')
    if str(meta.get('parity_status', '')).upper() != 'PASS':
        raise SystemExit('Refusing to install ALMAZ model without parity_status=PASS.')
    if str(meta.get('source_checkpoint_sha256', '')).lower() != source_sha:
        raise SystemExit('Manifest source checkpoint SHA-256 does not match Photo Doctor catalog.')
    if int(meta.get('source_checkpoint_size', -1) or -1) != int(source_size):
        raise SystemExit('Manifest source checkpoint size does not match Photo Doctor catalog.')
    if str(meta.get('source_commit', '')) != source_commit:
        raise SystemExit('Manifest source architecture commit does not match Photo Doctor catalog.')
    expected = str(meta.get('onnx_sha256', '')).lower()
    actual = sha256_file(src)
    if expected != actual:
        raise SystemExit('ONNX SHA-256 does not match its manifest.')
    root = Path(args.model_dir).expanduser().resolve() if args.model_dir else default_model_dir().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dst = root / filename
    dst_manifest = dst.with_suffix(dst.suffix + '.json')
    shutil.copy2(src, dst)
    shutil.copy2(manifest, dst_manifest)
    print(f'Installed: {dst}')
    print(f'Manifest: {dst_manifest}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
