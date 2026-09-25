from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "export_nafnet_restoration_onnx.py"


def test_nafnet_export_tool_pins_static_contract_and_official_configs():
    text = TOOL.read_text(encoding="utf-8")
    assert 'CONTRACT_ID = "rgb01_nchw_static256_x1_v1"' in text
    assert '"nafnet_sidd_width32"' in text
    assert '"width": 32' in text
    assert '"enc": [2, 2, 4, 8]' in text
    assert '"nafnet_gopro_width32"' in text
    assert '"enc": [1, 1, 1, 28]' in text
    assert '"nafnet_reds_width64"' in text
    assert '"width": 64' in text
    assert '"parity_status": parity_status' in text
    assert "verify_parity(" in text
    assert "CPUExecutionProvider" in text
    assert 'dynamo=False' in text


def test_nafnet_export_tool_does_not_bundle_or_download_weights():
    text = TOOL.read_text(encoding="utf-8")
    assert "urllib" not in text
    assert "requests" not in text
    assert "--weights" in text
    assert "--nafnet-root" in text
