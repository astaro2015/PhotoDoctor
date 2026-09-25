from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .restoration_backend import RESTORATION_CONTRACT_ID


@dataclass(frozen=True, slots=True)
class RestorationModelCandidate:
    task: str
    model_id: str
    display_name: str
    architecture: str
    upstream_filename: str
    upstream_repo: str
    download_urls: tuple[str, ...]
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


_COMMON_PROVIDERS = (
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)

NAFNET_SIDD_W32 = RestorationModelCandidate(
    task="denoise",
    model_id="nafnet_sidd_width32",
    display_name="NAFNet-SIDD width32",
    architecture="NAFNet width32 / SIDD real-image denoising",
    upstream_filename="NAFNet-SIDD-width32.pth",
    upstream_repo="https://github.com/megvii-research/NAFNet",
    download_urls=("https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-SIDD-width32.pth",),
    source_checkpoint_sha256="89c70e808d1783b6c07911306e106aaf0d4f7f3da8c61078b99ff7f8929a26f4",
    source_checkpoint_size=116861841,
    source_commit="2b4af71ebe098a92a75910c233a3965a3e93ede4",
    planned_onnx_filename="nafnet_sidd_w32_256.onnx",
    license_id="MIT",
    contract_id=RESTORATION_CONTRACT_ID,
    compatible_providers=_COMMON_PROVIDERS,
    qnn_requires_separate_quantized_artifact=True,
    generative=False,
    note="Первый кандидат ALMAZ Denoise: реальный шум SIDD, без GAN; применять консервативно и с защитой лица.",
)

NAFNET_GOPRO_W32 = RestorationModelCandidate(
    task="deblur",
    model_id="nafnet_gopro_width32",
    display_name="NAFNet-GoPro width32",
    architecture="NAFNet/NAFNetLocal width32 / GoPro deblurring",
    upstream_filename="NAFNet-GoPro-width32.pth",
    upstream_repo="https://github.com/megvii-research/NAFNet",
    download_urls=("https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-GoPro-width32.pth",),
    source_checkpoint_sha256="19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c",
    source_checkpoint_size=68671121,
    source_commit="2b4af71ebe098a92a75910c233a3965a3e93ede4",
    planned_onnx_filename="nafnet_gopro_w32_256.onnx",
    license_id="MIT",
    contract_id=RESTORATION_CONTRACT_ID,
    compatible_providers=_COMMON_PROVIDERS,
    qnn_requires_separate_quantized_artifact=True,
    generative=False,
    note="Первый кандидат ALMAZ Deblur: motion blur GoPro; только ручной preview до калибровки на старых фото.",
)

NAFNET_REDS_W64 = RestorationModelCandidate(
    task="jpeg_recovery",
    model_id="nafnet_reds_width64",
    display_name="NAFNet-REDS width64",
    architecture="NAFNetLocal width64 / REDS blur+JPEG restoration",
    upstream_filename="NAFNet-REDS-width64.pth",
    upstream_repo="https://github.com/megvii-research/NAFNet",
    download_urls=("https://huggingface.co/nyanko7/nafnet-models/resolve/main/NAFNet-REDS-width64.pth",),
    source_checkpoint_sha256="175fe8b3cdf3abedfbc87769779c3d9f491e05bb2e73ea9d627883f90a4b2df3",
    source_checkpoint_size=271756689,
    source_commit="2b4af71ebe098a92a75910c233a3965a3e93ede4",
    planned_onnx_filename="nafnet_reds_w64_256.onnx",
    license_id="MIT",
    contract_id=RESTORATION_CONTRACT_ID,
    compatible_providers=_COMMON_PROVIDERS,
    qnn_requires_separate_quantized_artifact=True,
    generative=False,
    note="Тяжёлый кандидат для blur+JPEG recovery; не использовать вместо обычного deblock без явной рекомендации.",
)

_CANDIDATES = {
    "denoise": NAFNET_SIDD_W32,
    "deblur": NAFNET_GOPRO_W32,
    "jpeg_recovery": NAFNET_REDS_W64,
}


def preferred_restoration_candidate(task: str) -> RestorationModelCandidate:
    key = str(task).strip().lower()
    if key not in _CANDIDATES:
        raise KeyError(f"Unknown ALMAZ restoration task: {task}")
    return _CANDIDATES[key]


def restoration_candidates() -> tuple[RestorationModelCandidate, ...]:
    return tuple(_CANDIDATES[key] for key in ("denoise", "deblur", "jpeg_recovery"))
