#!/usr/bin/env python3
"""Export official NAFNet restoration checkpoints to Photo Doctor static x1 ONNX.

Developer-only tool.  It intentionally does not vendor NAFNet or PyTorch into the
Photo Doctor runtime.  Supply an official megvii-research/NAFNet checkout and an
official checkpoint, then run parity tests before the resulting ONNX is allowed
into a release model bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CONTRACT_ID = "rgb01_nchw_static256_x1_v1"
TASKS = {
    "denoise": {
        "model_id": "nafnet_sidd_width32",
        "width": 32,
        "enc": [2, 2, 4, 8],
        "middle": 12,
        "dec": [2, 2, 2, 2],
        "upstream_filename": "NAFNet-SIDD-width32.pth",
    },
    "deblur": {
        "model_id": "nafnet_gopro_width32",
        "width": 32,
        "enc": [1, 1, 1, 28],
        "middle": 1,
        "dec": [1, 1, 1, 1],
        "upstream_filename": "NAFNet-GoPro-width32.pth",
    },
    "jpeg_recovery": {
        "model_id": "nafnet_reds_width64",
        "width": 64,
        "enc": [1, 1, 1, 28],
        "middle": 1,
        "dec": [1, 1, 1, 1],
        "upstream_filename": "NAFNet-REDS-width64.pth",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _compare_arrays(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    delta = np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))
    return float(delta.max(initial=0.0)), float(delta.mean())


def verify_parity(model, official_local_model, onnx_path: Path, torch_module) -> tuple[str, dict[str, object]]:
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
            rng.random((1, 3, 256, 256), dtype=np.float32),
            np.broadcast_to(
                np.linspace(0.0, 1.0, 256, dtype=np.float32)[None, None, None, :],
                (1, 3, 256, 256),
            ).copy(),
        ]
        onnx_max: list[float] = []
        onnx_mean: list[float] = []
        local_max: list[float] = []
        local_mean: list[float] = []
        with torch_module.no_grad():
            for sample in samples:
                tensor = torch_module.from_numpy(sample)
                base_out = model(tensor).cpu().numpy()
                ort_out = np.asarray(session.run([output_name], {input_name: sample})[0], dtype=np.float32)
                mx, mean = _compare_arrays(base_out, ort_out)
                onnx_max.append(mx); onnx_mean.append(mean)
                if official_local_model is not None:
                    local_out = official_local_model(tensor).cpu().numpy()
                    mx, mean = _compare_arrays(base_out, local_out)
                    local_max.append(mx); local_mean.append(mean)
        max_abs = max(onnx_max, default=0.0)
        mean_abs = max(onnx_mean, default=0.0)
        local_max_abs = max(local_max, default=0.0)
        local_mean_abs = max(local_mean, default=0.0)
        passed = bool(
            max_abs <= 1.0e-3 and mean_abs <= 5.0e-5
            and local_max_abs <= 1.0e-3 and local_mean_abs <= 5.0e-5
        )
        return ("PASS" if passed else "FAIL"), {
            "provider": "CPUExecutionProvider",
            "samples": len(samples),
            "onnx_max_abs": max_abs,
            "onnx_max_mean_abs": mean_abs,
            "official_local_max_abs": local_max_abs,
            "official_local_max_mean_abs": local_mean_abs,
            "max_abs_limit": 1.0e-3,
            "mean_abs_limit": 5.0e-5,
        }
    except Exception as exc:
        return "FAIL", {"reason": f"parity execution failed: {exc}"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--nafnet-root", type=Path, required=True, help="Official megvii-research/NAFNet checkout")
    parser.add_argument("--weights", type=Path, required=True, help="Official NAFNet .pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TASKS[args.task]
    root = args.nafnet_root.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    if not (root / "basicsr" / "models" / "archs" / "NAFNet_arch.py").is_file():
        raise SystemExit("--nafnet-root does not look like the official NAFNet checkout")
    if not weights.is_file():
        raise SystemExit(f"Checkpoint not found: {weights}")

    sys.path.insert(0, str(root))
    try:
        import torch
        import onnx
        from basicsr.models.archs.NAFNet_arch import NAFNet, NAFNetLocal
    except Exception as exc:
        raise SystemExit(f"Export dependencies/import failed: {exc}") from exc

    model = NAFNet(
        img_channel=3,
        width=int(cfg["width"]),
        middle_blk_num=int(cfg["middle"]),
        enc_blk_nums=list(cfg["enc"]),
        dec_blk_nums=list(cfg["dec"]),
    ).eval()
    checkpoint = torch.load(str(weights), map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise SystemExit("Unsupported NAFNet checkpoint structure")
    model.load_state_dict(state, strict=True)

    official_local_model = None
    if args.task in {"deblur", "jpeg_recovery"}:
        official_local_model = NAFNetLocal(
            img_channel=3,
            width=int(cfg["width"]),
            middle_blk_num=int(cfg["middle"]),
            enc_blk_nums=list(cfg["enc"]),
            dec_blk_nums=list(cfg["dec"]),
            train_size=(1, 3, 256, 256),
            fast_imp=False,
        ).eval()
        official_local_model.load_state_dict(state, strict=True)

    dummy = torch.rand(1, 3, 256, 256, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["input"],
            output_names=["output"],
            opset_version=int(args.opset),
            do_constant_folding=True,
            dynamo=False,
        )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    parity_status, parity_metrics = verify_parity(model, official_local_model, output, torch)

    manifest = {
        "task": args.task,
        "model_id": cfg["model_id"],
        "contract_id": CONTRACT_ID,
        "input_shape": [1, 3, 256, 256],
        "output_shape": [1, 3, 256, 256],
        "opset": int(args.opset),
        "license": "MIT",
        "upstream_repo": "https://github.com/megvii-research/NAFNet",
        "upstream_filename": cfg["upstream_filename"],
        "upstream_weights_sha256": sha256(weights),
        "onnx_sha256": sha256(output),
        "parity_status": parity_status,
        "parity_metrics": parity_metrics,
        "release_blocker": (None if parity_status == "PASS" else (
            "Official PyTorch NAFNet/NAFNetLocal ↔ ONNX parity on fixed 256x256 tiles must PASS before runtime use."
        )),
        "qnn_note": "QNN/NPU requires a separately quantized QDQ artifact and independent quality parity.",
    }
    manifest_path = output.with_suffix(output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
