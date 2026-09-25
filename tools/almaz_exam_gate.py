from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from almaz_training_contract import load_contract


@dataclass(frozen=True, slots=True)
class GateResult:
    accepted: bool
    failures: tuple[str, ...]


def evaluate_report(report: dict[str, Any], contract: dict[str, Any] | None = None) -> GateResult:
    gate = (contract or load_contract())["acceptance_gate"]
    failures: list[str] = []
    comparisons = {
        "psnr_delta_db": (">=", float(gate["min_psnr_delta_db"])),
        "ssim_delta": (">=", float(gate["min_ssim_delta"])),
        "face_identity_drift_delta": ("<=", float(gate["max_face_identity_drift_delta"])),
        "near_monochrome_chroma_artifact_delta": ("<=", float(gate["max_near_monochrome_chroma_artifact_delta"])),
        "clipping_delta_pp": ("<=", float(gate["max_clipping_delta_pp"])),
        "false_edge_ratio_delta": ("<=", float(gate["max_false_edge_ratio_delta"])),
    }
    metrics = dict(report.get("metrics", {}))
    for key, (op, limit) in comparisons.items():
        if key not in metrics:
            failures.append(f"missing metric: {key}")
            continue
        got = float(metrics[key])
        ok = got >= limit if op == ">=" else got <= limit
        if not ok:
            failures.append(f"{key}: {got:g} {op} {limit:g} required")
    if bool(gate.get("must_not_regress_any_task", True)):
        for task, delta in dict(report.get("task_quality_delta", {})).items():
            if float(delta) < -0.000001:
                failures.append(f"task regression: {task} delta={float(delta):g}")
    if bool(gate.get("manual_exam_required", True)) and not bool(report.get("manual_exam_passed", False)):
        failures.append("manual exam not passed")
    return GateResult(not failures, tuple(failures))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ALMAZ release checkpoint safety gate")
    p.add_argument("report")
    p.add_argument("--contract", default="")
    args = p.parse_args(argv)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    contract = load_contract(args.contract) if args.contract else load_contract()
    result = evaluate_report(report, contract)
    print("ALMAZ EXAM: PASS" if result.accepted else "ALMAZ EXAM: FAIL")
    for item in result.failures:
        print(f" - {item}")
    return 0 if result.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
