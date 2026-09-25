import json
from pathlib import Path
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from almaz_exam_gate import evaluate_report
from almaz_training_contract import DatasetStats, load_contract, validate_dataset_stats


def _good_stats():
    return DatasetStats(
        train_pairs_total=100_000,
        train_source_groups_total=2_500,
        val_source_groups_total=250,
        exam_source_groups_total=500,
        train_pairs_by_task={
            "sr_x2": 25_000,
            "denoise": 25_000,
            "deblur": 20_000,
            "jpeg_recovery": 20_000,
            "archive_restore": 10_000,
        },
        exam_source_groups_by_domain={"modern": 150, "archive": 200, "faces": 100, "near_monochrome": 100},
    )


def test_contract_rejects_tiny_smoke_dataset():
    s = _good_stats()
    tiny = DatasetStats(100, 10, 5, 5, s.train_pairs_by_task, s.exam_source_groups_by_domain)
    failures = validate_dataset_stats(tiny)
    assert failures
    assert any("train_pairs_total" in item for item in failures)


def test_contract_accepts_minimum_real_dataset():
    assert validate_dataset_stats(_good_stats()) == []


def test_exam_gate_rejects_green_chroma_regression_even_when_psnr_improves():
    report = {
        "metrics": {
            "psnr_delta_db": 0.5,
            "ssim_delta": 0.003,
            "face_identity_drift_delta": -0.001,
            "near_monochrome_chroma_artifact_delta": 0.01,
            "clipping_delta_pp": -0.1,
            "false_edge_ratio_delta": -0.01,
        },
        "task_quality_delta": {"denoise": 0.01},
        "manual_exam_passed": True,
    }
    result = evaluate_report(report)
    assert not result.accepted
    assert any("chroma" in item for item in result.failures)


def test_exam_gate_requires_manual_exam():
    report = {
        "metrics": {
            "psnr_delta_db": 0.1,
            "ssim_delta": 0.001,
            "face_identity_drift_delta": -0.001,
            "near_monochrome_chroma_artifact_delta": -0.01,
            "clipping_delta_pp": -0.1,
            "false_edge_ratio_delta": -0.01,
        },
        "task_quality_delta": {"denoise": 0.01, "deblur": 0.02},
        "manual_exam_passed": False,
    }
    result = evaluate_report(report)
    assert not result.accepted
    assert "manual exam not passed" in result.failures


def test_sources_file_has_no_blocked_source_enabled_by_default():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "training/almaz/SOURCES.json").read_text(encoding="utf-8"))
    blocked = set(payload["policy"]["blocked_by_default"])
    for source in payload["sources"]:
        if source["commercial_training_default"]:
            license_text = str(source["license"])
            assert not any(token.lower() in license_text.lower() for token in blocked)
