from __future__ import annotations

from .manager_types import ModelSpec

# All current micro-model contracts use one deliberately boring image input:
# RGB float32 in [0,1], NCHW, one image per batch. This makes model replacement
# predictable and testable. There are intentionally no download URLs.
BUILTIN_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="blur_refiner_v1",
        task="blur_refinement",
        filename="blur_refiner_v1.onnx",
        version="1.0.0",
        description="Уточнение типа потери деталей: дефокус / движение / деградация / смешанный случай.",
        input_size=(256, 256),
        contract_id="rgb01_nchw_classification_v1",
        output_kind="classification",
        labels=("clean", "defocus", "motion", "degradation", "mixed"),
    ),
    ModelSpec(
        model_id="native_surface_verifier_v2",
        task="surface_defect_refinement",
        filename="<builtin>",
        version="2.3.0-candidate",
        description=(
            "Встроенный Surface AI v2.3: Context Meta Verifier v2.1 является основным контекстным проверяющим, "
            "а компактный 6-канальный U-Net (~56 тыс. параметров) — пиксельным вторым мнением и локализатором. "
            "Context Meta обучается без признаков, участвовавших в weak-label rule; сильный отрицательный U-Net может "
            "ветировать кандидата, но его неопределённость не отменяет сильный независимый контекст. Weak real-pairs "
            "используются только как candidate/hard-mining; "
            "автоматические правки запрещены."
        ),
        input_size=(96, 96),
        contract_id="native_surface96_6ch_unet_contextmeta_candidate_v2",
        output_kind="segmentation_candidate_score",
        labels=("defect", "uncertain"),
        provider="native_numpy",
        bundled=True,
        provenance="photodoctor_surface_v2_candidate_pair_20260921",
    ),
    ModelSpec(
        model_id="face_quality_v1",
        task="face_quality",
        filename="face_quality_v1.onnx",
        version="1.0.0",
        description="Локальная оценка технического качества лица без идентификации личности.",
        input_size=(224, 224),
        contract_id="rgb01_nchw_scalar_v1",
        output_kind="scalar_0_1",
    ),
    ModelSpec(
        model_id="iqa_refiner_v1",
        task="iqa_refinement",
        filename="iqa_refiner_v1.onnx",
        version="1.0.0",
        description="Дополнительная оценка качества без эталона для неоднозначных случаев.",
        input_size=(224, 224),
        contract_id="rgb01_nchw_scalar_v1",
        output_kind="scalar_0_1",
    ),
    ModelSpec(
        model_id="semantic_context_v1",
        task="semantic_context_refinement",
        filename="semantic_context_v1.onnx",
        version="1.0.0",
        description="Осторожное уточнение контекста кадра без идентификации людей.",
        input_size=(224, 224),
        contract_id="rgb01_nchw_classification_v1",
        output_kind="classification",
        labels=("general_photo", "portrait", "group_portrait", "archival_portrait"),
    ),
)
