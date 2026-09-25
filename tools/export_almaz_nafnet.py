from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from photodoctor.ai.restoration_catalog import preferred_restoration_candidate
from almaz_export_common import extract_state_dict, parity_onnx, sha256_file, write_manifest
from almaz_download import verify_file


_CONFIGS = {
    'denoise': dict(local=False, width=32, enc=[2, 2, 4, 8], middle=12, dec=[2, 2, 2, 2]),
    'deblur': dict(local=True, width=32, enc=[1, 1, 1, 28], middle=1, dec=[1, 1, 1, 1]),
    'jpeg_recovery': dict(local=True, width=64, enc=[1, 1, 1, 28], middle=1, dec=[1, 1, 1, 1]),
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Export official NAFNet checkpoint to Photo Doctor ALMAZ x1 ONNX.')
    p.add_argument('--task', choices=tuple(_CONFIGS), required=True)
    p.add_argument('--nafnet-repo', required=True, help='Path to local official megvii-research/NAFNet repository.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', default='')
    p.add_argument('--input-size', type=int, default=256)
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--allow-pending-parity', action='store_true')
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(args.nafnet_repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f'Checkpoint not found: {checkpoint}')
    if not (repo / 'basicsr' / 'models' / 'archs' / 'NAFNet_arch.py').is_file():
        raise SystemExit(f'Official NAFNet architecture not found under: {repo}')
    spec = preferred_restoration_candidate(args.task)
    verify_file(checkpoint, expected_sha256=spec.source_checkpoint_sha256, expected_size=spec.source_checkpoint_size)
    input_size = int(args.input_size)
    if input_size < 64 or input_size % 16:
        raise SystemExit('--input-size must be >=64 and divisible by 16.')

    try:
        import torch
        import onnx  # noqa: F401
    except Exception as exc:
        raise SystemExit(f'PyTorch + onnx are required for export: {exc}') from exc

    sys.path.insert(0, str(repo))
    try:
        mod = importlib.import_module('basicsr.models.archs.NAFNet_arch')
        NAFNet = getattr(mod, 'NAFNet')
        NAFNetLocal = getattr(mod, 'NAFNetLocal')
    except Exception as exc:
        raise SystemExit(
            'Failed to import official NAFNet architecture. Install its repository dependencies first: ' + str(exc)
        ) from exc

    cfg = _CONFIGS[args.task]
    cls = NAFNetLocal if cfg['local'] else NAFNet
    kwargs = dict(
        img_channel=3, width=cfg['width'], middle_blk_num=cfg['middle'],
        enc_blk_nums=cfg['enc'], dec_blk_nums=cfg['dec'],
    )
    if cfg['local']:
        kwargs['train_size'] = (1, 3, input_size, input_size)
        kwargs['fast_imp'] = False
    model = cls(**kwargs)
    payload = torch.load(str(checkpoint), map_location='cpu')
    state = extract_state_dict(payload)
    model.load_state_dict(state, strict=True)
    model.eval()

    class ExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x):
            return torch.clamp(self.inner(x), 0.0, 1.0)

    wrapped = ExportWrapper(model).eval()
    out = Path(args.output).expanduser() if args.output else (Path.home() / '.photodoctor' / 'models' / spec.planned_onnx_filename)
    out = out.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.linspace(0.0, 1.0, steps=input_size * input_size * 3, dtype=torch.float32).reshape(1, 3, input_size, input_size)
    with torch.no_grad():
        torch_ref = wrapped(dummy)
    torch.onnx.export(
        wrapped, dummy, str(out), input_names=['input'], output_names=['output'],
        opset_version=int(args.opset), do_constant_folding=True, dynamic_axes=None, dynamo=False,
    )
    parity = parity_onnx(out, dummy, torch_ref)
    if parity['status'] != 'PASS' and not (args.allow_pending_parity and parity['status'] == 'PENDING'):
        out.unlink(missing_ok=True)
        raise SystemExit(f"ONNX parity did not pass: {parity['detail']}")

    manifest = {
        'schema_version': 1,
        'task': spec.task,
        'model_id': spec.model_id,
        'display_name': spec.display_name,
        'contract_id': spec.contract_id,
        'input_size': input_size,
        'opset': int(args.opset),
        'onnx_sha256': sha256_file(out),
        'source_checkpoint': checkpoint.name,
        'source_checkpoint_sha256': spec.source_checkpoint_sha256,
        'source_checkpoint_size': spec.source_checkpoint_size,
        'source_commit': spec.source_commit,
        'upstream_repo': spec.upstream_repo,
        'license_id': spec.license_id,
        'generative': False,
        'parity_status': parity['status'],
        'parity': parity,
        'exported_utc': datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out.with_suffix(out.suffix + '.json')
    write_manifest(manifest_path, manifest)
    print(f'ONNX: {out}')
    print(f'Manifest: {manifest_path}')
    print(f"Parity: {parity['status']} ({parity['detail']})")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
