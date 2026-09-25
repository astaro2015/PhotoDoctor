from photodoctor.ai.sr_backend import SR_CONTRACT_ID
from photodoctor.ai.sr_catalog import preferred_sr_candidate


def test_preferred_almaz_candidate_is_non_generative_x2():
    spec = preferred_sr_candidate()
    assert spec.model_id == "swinir_s_classical_x2"
    assert spec.scale == 2
    assert spec.generative is False
    assert spec.license_id == "Apache-2.0"
    assert spec.contract_id == SR_CONTRACT_ID


def test_qnn_is_not_silently_claimed_compatible_with_float_candidate():
    spec = preferred_sr_candidate()
    assert "QNNExecutionProvider" not in spec.compatible_providers
    assert spec.qnn_requires_separate_quantized_artifact is True
