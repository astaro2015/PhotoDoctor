from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photodoctor.ai.almaz_model_promotion import promote_finetuned_model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Safely promote a trained ALMAZ ONNX model")
    p.add_argument("--task", required=True, choices=["sr_x2", "denoise", "deblur", "jpeg_recovery"])
    p.add_argument("--onnx", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--exam", required=True)
    p.add_argument("--parity", required=True)
    p.add_argument("--model-dir", default="")
    args = p.parse_args(argv)
    result = promote_finetuned_model(
        args.task, args.onnx, args.manifest, args.exam, args.parity,
        model_dir=(args.model_dir or None),
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
