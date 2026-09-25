from photodoctor.ai.restoration_backend import RESTORATION_CONTRACT_ID
from photodoctor.ai.restoration_catalog import preferred_restoration_candidate, restoration_candidates


def test_restoration_catalog_has_three_non_generative_tasks():
    specs = restoration_candidates()
    assert {spec.task for spec in specs} == {"denoise", "deblur", "jpeg_recovery"}
    assert all(spec.generative is False for spec in specs)
    assert all(spec.contract_id == RESTORATION_CONTRACT_ID for spec in specs)
    assert all(spec.qnn_requires_separate_quantized_artifact for spec in specs)


def test_denoise_and_deblur_use_width32_candidates():
    assert "width32" in preferred_restoration_candidate("denoise").model_id
    assert "width32" in preferred_restoration_candidate("deblur").model_id
    assert preferred_restoration_candidate("jpeg_recovery").model_id == "nafnet_reds_width64"
