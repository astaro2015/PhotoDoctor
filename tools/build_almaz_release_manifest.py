from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from photodoctor.ai.almaz_release_gate import (
    TRAINING_CONTRACT_SHA256,
    evaluate_exam_report,
    load_exam_report,
    sha256_file,
)
from photodoctor.ai.restoration_catalog import preferred_restoration_candidate
from photodoctor.ai.sr_catalog import preferred_sr_candidate


def _descriptor(task: str):
    key = task.strip().lower()
    if key == "sr_x2":
        spec = preferred_sr_candidate()
        return "sr_x2", spec
    spec = preferred_restoration_candidate(key)
    return spec.task, spec


def _load_parity(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("parity report must be a JSON object")
    status = str(raw.get("status", raw.get("parity_status", ""))).upper()
    if status != "PASS":
        raise ValueError("parity report is not PASS")
    return raw


def build_manifest(
    *, task: str, onnx: Path, training_checkpoint: Path, exam: Path, parity: Path,
    variant_id: str,
) -> dict:
    canonical_task, spec = _descriptor(task)
    if not onnx.is_file():
        raise FileNotFoundError(onnx)
    if not training_checkpoint.is_file():
        raise FileNotFoundError(training_checkpoint)
    if not exam.is_file():
        raise FileNotFoundError(exam)
    if not parity.is_file():
        raise FileNotFoundError(parity)

    exam_report = load_exam_report(exam)
    gate = evaluate_exam_report(exam_report)
    if not gate.accepted:
        raise ValueError("exam gate failed: " + "; ".join(gate.failures))
    _load_parity(parity)

    return {
        "schema": "photodoctor_almaz_finetuned_v1",
        "variant_kind": "finetuned",
        "variant_id": variant_id,
        "task": canonical_task,
        "model_id": spec.model_id,
        "contract_id": spec.contract_id,
        "base_source_checkpoint_sha256": spec.source_checkpoint_sha256,
        "base_source_checkpoint_size": spec.source_checkpoint_size,
        "base_source_commit": spec.source_commit,
        "training_checkpoint_sha256": sha256_file(training_checkpoint),
        "training_checkpoint_size": training_checkpoint.stat().st_size,
        "training_contract_sha256": TRAINING_CONTRACT_SHA256,
        "parity_status": "PASS",
        "parity_report_sha256": sha256_file(parity),
        "exam_report_sha256": sha256_file(exam),
        "onnx_sha256": sha256_file(onnx),
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build verified ALMAZ fine-tune release manifest")
    p.add_argument("--task", required=True, choices=["sr_x2", "denoise", "deblur", "jpeg_recovery"])
    p.add_argument("--onnx", required=True)
    p.add_argument("--training-checkpoint", required=True)
    p.add_argument("--exam", required=True)
    p.add_argument("--parity", required=True)
    p.add_argument("--variant-id", default="")
    p.add_argument("--output", default="")
    args = p.parse_args(argv)

    onnx = Path(args.onnx)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    variant_id = args.variant_id.strip() or f"almaz_{args.task}_ft_{stamp.lower()}"
    try:
        manifest = build_manifest(
            task=args.task,
            onnx=onnx,
            training_checkpoint=Path(args.training_checkpoint),
            exam=Path(args.exam),
            parity=Path(args.parity),
            variant_id=variant_id,
        )
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 2
    output = Path(args.output) if args.output else onnx.with_suffix(onnx.suffix + ".release.json")
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
