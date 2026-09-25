from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .sr_backend import SR_CONTRACT_ID


@dataclass(frozen=True, slots=True)
class SRModelCandidate:
    model_id: str
    display_name: str
    architecture: str
    scale: int
    upstream_filename: str
    upstream_url: str
    upstream_mirrors: tuple[str, ...]
    source_checkpoint_sha256: str
    source_checkpoint_size: int
    source_commit: str
    planned_onnx_filename: str
    license_id: str
    contract_id: str
    compatible_providers: tuple[str, ...]
    qnn_requires_separate_quantized_artifact: bool
    generative: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SWINIR_S_X2_CLASSICAL = SRModelCandidate(
    model_id="swinir_s_classical_x2",
    display_name="SwinIR-S Classical x2",
    architecture="SwinIR lightweight / pixelshuffledirect",
    scale=2,
    upstream_filename="002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
    upstream_url=(
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
        "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    ),
    upstream_mirrors=(
        "https://huggingface.co/deepinv/swinir/resolve/main/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
    ),
    source_checkpoint_sha256="193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7",
    source_checkpoint_size=17147989,
    source_commit="33f616625268d08ba600f8db89388eec0328edb1",
    planned_onnx_filename="swinir_s_classical_x2_256.onnx",
    license_id="Apache-2.0",
    contract_id=SR_CONTRACT_ID,
    compatible_providers=(
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    ),
    qnn_requires_separate_quantized_artifact=True,
    generative=False,
    note=(
        "Первый кандидат ALMAZ: classical/lightweight x2 без GAN. "
        "QNN/NPU не включать для этого файла автоматически; подготовить отдельный QDQ/quantized артефакт."
    ),
)


def preferred_sr_candidate() -> SRModelCandidate:
    return SWINIR_S_X2_CLASSICAL
