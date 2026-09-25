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

from photodoctor.ai.sr_catalog import preferred_sr_candidate
from almaz_export_common import extract_state_dict, parity_onnx, sha256_file, write_manifest
from almaz_download import verify_file


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Export official lightweight SwinIR x2 checkpoint to Photo Doctor ALMAZ ONNX.')
    p.add_argument('--swinir-repo', required=True, help='Path to local official JingyunLiang/SwinIR repository.')
    p.add_argument('--checkpoint', required=True, help='Path to 002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth')
    p.add_argument('--output', default='', help='Output ONNX path. Default: ~/.photodoctor/models/<catalog filename>.')
    p.add_argument('--input-size', type=int, default=256, help='Static square input size. Must be divisible by 8.')
    p.add_argument('--opset', type=int, default=17)
    p.add_argument('--allow-pending-parity', action='store_true', help='Write PENDING manifest when onnxruntime is unavailable. Runtime will still refuse to use it.')
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(args.swinir_repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f'Checkpoint not found: {checkpoint}')
    if not (repo / 'models' / 'network_swinir.py').is_file():
        raise SystemExit(f'Official SwinIR network file not found under: {repo}')
    spec = preferred_sr_candidate()
    verify_file(checkpoint, expected_sha256=spec.source_checkpoint_sha256, expected_size=spec.source_checkpoint_size)
    input_size = int(args.input_size)
    if input_size < 64 or input_size % 8:
        raise SystemExit('--input-size must be >=64 and divisible by SwinIR window size 8.')

    try:
        import torch
    except Exception as exc:
        raise SystemExit(f'PyTorch is required for export: {exc}') from exc
    try:
        import onnx  # noqa: F401
    except Exception as exc:
        raise SystemExit(f'Python package onnx is required for export: {exc}') from exc

    sys.path.insert(0, str(repo))
    try:
        mod = importlib.import_module('models.network_swinir')
        SwinIR = getattr(mod, 'SwinIR')
    except Exception as exc:
        raise SystemExit(f'Failed to import official SwinIR architecture: {exc}') from exc

    model = SwinIR(
        upscale=2, in_chans=3, img_size=64, window_size=8, img_range=1.0,
        depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
        mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv',
    )
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
        'task': 'super_resolution_x2',
        'model_id': spec.model_id,
        'display_name': spec.display_name,
        'contract_id': spec.contract_id,
        'scale': 2,
        'input_size': input_size,
        'opset': int(args.opset),
        'onnx_sha256': sha256_file(out),
        'source_checkpoint': checkpoint.name,
        'source_checkpoint_sha256': spec.source_checkpoint_sha256,
        'source_checkpoint_size': spec.source_checkpoint_size,
        'source_commit': spec.source_commit,
        'upstream_url': spec.upstream_url,
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
