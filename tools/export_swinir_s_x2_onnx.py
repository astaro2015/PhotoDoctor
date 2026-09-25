from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _verify_onnx_parity(model, onnx_path: Path, torch_module, tile_size: int) -> tuple[str, dict[str, float | str]]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        return "PENDING", {"reason": f"onnxruntime unavailable: {exc}"}
    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        rng = np.random.default_rng(20260922)
        samples = [
            rng.random((1, 3, tile_size, tile_size), dtype=np.float32),
            np.broadcast_to(
                np.linspace(0.0, 1.0, tile_size, dtype=np.float32)[None, None, None, :],
                (1, 3, tile_size, tile_size),
            ).copy(),
        ]
        maxima: list[float] = []
        means: list[float] = []
        with torch_module.no_grad():
            for sample in samples:
                torch_out = model(torch_module.from_numpy(sample)).cpu().numpy()
                ort_out = np.asarray(session.run([output_name], {input_name: sample})[0], dtype=np.float32)
                delta = np.abs(torch_out.astype(np.float32) - ort_out)
                maxima.append(float(delta.max(initial=0.0)))
                means.append(float(delta.mean()))
        max_abs = max(maxima, default=0.0)
        mean_abs = max(means, default=0.0)
        passed = bool(max_abs <= 1.0e-3 and mean_abs <= 5.0e-5)
        return ("PASS" if passed else "FAIL"), {
            "provider": "CPUExecutionProvider",
            "samples": float(len(samples)),
            "max_abs": max_abs,
            "max_mean_abs": mean_abs,
            "max_abs_limit": 1.0e-3,
            "mean_abs_limit": 5.0e-5,
        }
    except Exception as exc:
        return "FAIL", {"reason": f"parity execution failed: {exc}"}

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert official SwinIR-S lightweight x2 weights into Photo Doctor's "
            "static 256x256 -> 512x512 ONNX contract."
        )
    )
    parser.add_argument("--weights", required=True, type=Path, help="Official SwinIR-S x2 .pth checkpoint")
    parser.add_argument("--swinir-repo", required=True, type=Path, help="Checkout of official JingyunLiang/SwinIR repo")
    parser.add_argument("--output", required=True, type=Path, help="Destination .onnx file")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--tile-size", type=int, default=256)
    args = parser.parse_args()

    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")
    network_file = args.swinir_repo / "models" / "network_swinir.py"
    if not network_file.is_file():
        raise SystemExit(f"Official SwinIR network file not found: {network_file}")
    if args.tile_size <= 0 or args.tile_size % 8:
        raise SystemExit("--tile-size must be a positive multiple of SwinIR window size 8")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required only for model export.") from exc
    try:
        import onnx
    except ImportError as exc:
        raise SystemExit("onnx package is required for export verification (pip install onnx).") from exc

    sys.path.insert(0, str(args.swinir_repo.resolve()))
    try:
        from models.network_swinir import SwinIR
    except Exception as exc:
        raise SystemExit(f"Cannot import official SwinIR implementation: {exc}") from exc

    model = SwinIR(
        upscale=2,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
    )
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    state = checkpoint.get("params", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros((1, 3, args.tile_size, args.tile_size), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)
    parity_status, parity_metrics = _verify_onnx_parity(model, args.output, torch, args.tile_size)

    manifest = {
        "model_id": "swinir_s_classical_x2",
        "contract_id": "rgb01_nchw_static256_x2_v1",
        "input_shape": [1, 3, args.tile_size, args.tile_size],
        "output_shape": [1, 3, args.tile_size * 2, args.tile_size * 2],
        "scale": 2,
        "opset": args.opset,
        "upstream_weights": args.weights.name,
        "upstream_weights_sha256": _sha256(args.weights),
        "onnx_filename": args.output.name,
        "onnx_sha256": _sha256(args.output),
        "parity_status": parity_status,
        "parity_metrics": parity_metrics,
        "release_blocker": (None if parity_status == "PASS" else "PyTorch SwinIR ↔ ONNX parity must PASS before runtime use."),
        "license": "Apache-2.0",
        "qnn_note": "QNN/NPU requires a separately validated quantized/QDQ artifact.",
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
