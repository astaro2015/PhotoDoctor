from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .loader import LoadedImage, srgb_to_linear
from .local_sharpness import analyze_local_sharpness
from .local_tone import analyze_local_tone
from .local_contrast import analyze_local_contrast
from .local_correction_planner import build_local_correction_plan
from .precision import get_precision
from .performance import get_pipeline_profile
from .jpeg_artifacts import analyze_jpeg_artifacts
from .rendering_artifacts import analyze_edge_artifacts, analyze_posterization
from .noise import analyze_noise
from .highlights import analyze_highlight_context
from .exif_context import analyze_exif_context
from .nss_baseline import analyze_nss_baseline
from .semantic_context import analyze_semantic_context
from .main_subject import analyze_main_subject
from .aesthetic_quality import analyze_aesthetic_quality
from .analysis_reliability import analyze_analysis_reliability
from .decisions import build_decision_plan
from .versioning import DECISION_ENGINE_VERSION
from .eyes import EyeAnalysis, EyeRegion, analyze_eyes
from .red_eye import analyze_red_eye, supplement_red_eye_eye_regions
from .blur_type import analyze_blur_type
from .detail_loss_context import refine_detail_loss_type
from .faces import FaceAnalysis, FaceRegion, analyze_faces
from .models import AnalysisResult, MetricResult
from .surface import detect_surface_defects, merge_surface_detections
from .auto_tone_color import analyze_auto_tone_color
from .white_balance import analyze_white_balance, analyze_spatial_white_balance
from .super_resolution import analyze_super_resolution_need


NORMALIZED_LONG_EDGE = 2048


def _resize_for_technical(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= NORMALIZED_LONG_EDGE:
        return rgb, f"native_{w}x{h}"
    scale = NORMALIZED_LONG_EDGE / long_edge
    out = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return out, f"normalized_{NORMALIZED_LONG_EDGE}px"


def _resize_for_spatial(rgb: np.ndarray, long_edge_limit: int) -> tuple[np.ndarray, str]:
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= long_edge_limit:
        return rgb, f"native_{w}x{h}"
    scale = long_edge_limit / long_edge
    out = cv2.resize(rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return out, f"spatial_{long_edge_limit}px"


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 100.0))


def _linear_working_copies(
    image: LoadedImage, rgb: np.ndarray, spatial_rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build technical/spatial linear-light copies with bounded peak memory.

    When the native image is larger than the spatial target, process one source
    colour channel at a time. OpenCV INTER_AREA is channel-separable, and this
    produces bit-identical float32 results to resizing a full three-channel linear
    source while avoiding a huge native float32 RGB allocation.
    """
    source = image.srgb
    supplied = image.linear_rgb
    if supplied is not None:
        linear = supplied if rgb.shape[:2] == supplied.shape[:2] else cv2.resize(
            supplied, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA
        )
        spatial_linear = supplied if spatial_rgb.shape[:2] == supplied.shape[:2] else cv2.resize(
            supplied, (spatial_rgb.shape[1], spatial_rgb.shape[0]), interpolation=cv2.INTER_AREA
        )
        return linear, spatial_linear

    source_pixels = int(source.shape[0]) * int(source.shape[1])
    # For ordinary camera files up to roughly 32 MP, one three-channel conversion
    # is faster and still comfortably bounded on the 16-GB minimum target. Keep
    # the channel-wise memory-safe path for genuinely large originals.
    if spatial_rgb.shape[:2] == source.shape[:2] or source_pixels <= 32_000_000:
        source_linear = srgb_to_linear(source)
        linear = source_linear if rgb.shape[:2] == source.shape[:2] else cv2.resize(
            source_linear, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA
        )
        spatial_linear = source_linear if spatial_rgb.shape[:2] == source.shape[:2] else cv2.resize(
            source_linear, (spatial_rgb.shape[1], spatial_rgb.shape[0]), interpolation=cv2.INTER_AREA
        )
        return linear, spatial_linear

    rh, rw = rgb.shape[:2]
    sh, sw = spatial_rgb.shape[:2]
    same_target = (rh, rw) == (sh, sw)
    linear = np.empty((rh, rw, 3), dtype=np.float32)
    spatial_linear = linear if same_target else np.empty((sh, sw, 3), dtype=np.float32)
    for c in range(3):
        source_channel = srgb_to_linear(source[..., c])
        linear[..., c] = cv2.resize(source_channel, (rw, rh), interpolation=cv2.INTER_AREA)
        if not same_target:
            spatial_linear[..., c] = cv2.resize(source_channel, (sw, sh), interpolation=cv2.INTER_AREA)
        del source_channel
    return linear, spatial_linear


def _native_center_crop(rgb: np.ndarray, long_edge_limit: int = 6000) -> tuple[np.ndarray, str]:
    """Return a native-pixel crop without resampling.

    JPEG grid diagnostics become less trustworthy after arbitrary resize because
    the original 8x8 lattice is destroyed.  Precise mode therefore uses native
    pixels; exceptionally large images are bounded by a central crop rather than
    resampled.
    """
    h, w = rgb.shape[:2]
    long_edge = max(h, w)
    if long_edge <= long_edge_limit:
        return rgb, f"native_{w}x{h}"
    scale = long_edge_limit / max(long_edge, 1)
    crop_w = max(1, min(w, int(round(w * scale))))
    crop_h = max(1, min(h, int(round(h * scale))))
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    return rgb[y0:y0 + crop_h, x0:x0 + crop_w], f"native_crop_{crop_w}x{crop_h}"


def _scale_box(x: int, y: int, w: int, h: int, src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)
    nx = int(round(x * sx)); ny = int(round(y * sy))
    nw = max(1, int(round(w * sx))); nh = max(1, int(round(h * sy)))
    nx = max(0, min(nx, max(dst_w - 1, 0)))
    ny = max(0, min(ny, max(dst_h - 1, 0)))
    nw = max(1, min(nw, dst_w - nx))
    nh = max(1, min(nh, dst_h - ny))
    return nx, ny, nw, nh


def _rescale_face_analysis(analysis: FaceAnalysis, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> FaceAnalysis:
    src_h, src_w = src_shape; dst_h, dst_w = dst_shape
    faces: list[FaceRegion] = []
    for face in analysis.faces:
        x, y, w, h = _scale_box(face.x, face.y, face.w, face.h, src_w, src_h, dst_w, dst_h)
        faces.append(FaceRegion(
            x=x, y=y, w=w, h=h, brightness_linear=face.brightness_linear,
            laplacian=face.laplacian, tenengrad=face.tenengrad,
            sharpness_score=face.sharpness_score, measurement_confidence=face.measurement_confidence,
            source=face.source,
        ))
    return FaceAnalysis(analysis.detector_available, faces, analysis.detector_confidence, analysis.detector_name)


def _rescale_eye_analysis(analysis: EyeAnalysis, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> EyeAnalysis:
    src_h, src_w = src_shape; dst_h, dst_w = dst_shape
    eyes: list[EyeRegion] = []
    for eye in analysis.eyes:
        x, y, w, h = _scale_box(eye.x, eye.y, eye.w, eye.h, src_w, src_h, dst_w, dst_h)
        eyes.append(EyeRegion(
            face_index=eye.face_index, x=x, y=y, w=w, h=h,
            laplacian=eye.laplacian, tenengrad=eye.tenengrad, sharpness_score=eye.sharpness_score,
            measurement_confidence=eye.measurement_confidence, source=eye.source,
        ))
    return EyeAnalysis(
        analysis.detector_available, eyes, analysis.detector_confidence, analysis.faces_with_eyes, analysis.detector_name
    )




def _brightness_quality_score(mean_luma: float) -> float:
    """Return technical brightness quality from *linear-light* mean luma.

    The old implementation treated linear luma almost like display-space
    brightness and therefore marked many normal photographs as too dark.
    A broad plateau is intentional: average scene brightness is a weak signal
    and must not punish low-key/high-key style by itself.
    """
    value = float(np.clip(mean_luma, 0.0, 1.0))
    if 0.18 <= value <= 0.45:
        return 100.0
    if value < 0.18:
        return _clip01(100.0 - (0.18 - value) * 450.0)
    return _clip01(100.0 - (value - 0.45) * 180.0)

def analyze_classical(
    image: LoadedImage,
    precision: str = "normal",
    manual_face_boxes: list[dict[str, float]] | None = None,
    manual_eye_boxes: list[dict[str, float]] | None = None,
) -> AnalysisResult:
    precision_profile = get_precision(precision)
    rgb, scale_name = _resize_for_technical(image.srgb)
    spatial_rgb, spatial_scale_name = _resize_for_spatial(image.srgb, precision_profile.spatial_long_edge)
    if precision_profile.deep_analysis:
        if precision_profile.detail_long_edge == precision_profile.spatial_long_edge:
            # Precise currently uses the same 4096px source for both purposes.
            # Reuse the exact same array instead of keeping a duplicate ~32 MiB copy.
            detail_rgb, detail_scale_name = spatial_rgb, spatial_scale_name
        else:
            detail_rgb, detail_scale_name = _resize_for_spatial(image.srgb, precision_profile.detail_long_edge)
    else:
        detail_rgb, detail_scale_name = rgb, scale_name

    linear, spatial_linear = _linear_working_copies(image, rgb, spatial_rgb)

    gray8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_lin = (0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]).astype(np.float32)

    p1, p5, p50, p95, p99 = np.percentile(gray_lin, [1, 5, 50, 95, 99])
    mean_luma = float(gray_lin.mean())
    shadow_clip = float(np.mean(gray_lin <= 0.0031308) * 100.0)
    highlight_clip = float(np.mean(gray_lin >= 0.99) * 100.0)
    contrast = float(p95 - p5)

    lap_var = float(cv2.Laplacian(gray8, cv2.CV_64F).var())
    gx = cv2.Sobel(gray8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray8, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))

    channel_mean = rgb.reshape(-1, 3).mean(axis=0)
    mean_rgb = float(channel_mean.mean())
    channel_spread = float((channel_mean.max() - channel_mean.min()) / max(mean_rgb, 1.0))

    rgbf = rgb.astype(np.float32) / 255.0
    pix_max = rgbf.max(axis=2)
    pix_min = rgbf.min(axis=2)
    pix_mean = rgbf.mean(axis=2)
    pix_chroma = (pix_max - pix_min) / np.maximum(pix_mean, 1.0 / 255.0)
    mean_chroma = float(np.mean(pix_chroma))
    neutral_fraction = float(np.mean((pix_max - pix_min) <= (8.0 / 255.0)))
    rch, gch, bch = rgbf[..., 0], rgbf[..., 1], rgbf[..., 2]
    valid_tone = pix_mean > (12.0 / 255.0)
    if int(valid_tone.sum()) >= 64:
        warm_order_fraction = float(np.mean(((rch >= gch) & (gch >= bch))[valid_tone]))
    else:
        warm_order_fraction = float(np.mean((rch >= gch) & (gch >= bch)))

    if mean_chroma < 0.045 or neutral_fraction > 0.90:
        tone_class = "monochrome"
        tone_confidence = float(np.clip(0.62 + max(0.0, 0.90 - mean_chroma) * 0.25, 0.0, 0.96))
    elif channel_mean[0] > channel_mean[1] > channel_mean[2] and warm_order_fraction >= 0.68 and 0.05 <= channel_spread <= 0.42:
        tone_class = "sepia"
        tone_confidence = float(np.clip(0.55 + (warm_order_fraction - 0.68) * 0.9 + min(channel_spread, 0.30) * 0.35, 0.0, 0.94))
    else:
        tone_class = "color"
        tone_confidence = float(np.clip(0.58 + min(mean_chroma, 0.45) * 0.7, 0.0, 0.93))

    hist = cv2.calcHist([gray8], [0], None, [256], [0, 256]).reshape(-1)
    hist = (hist / max(hist.sum(), 1.0)).astype(np.float32)

    # The spatial linear-light copy can exceed 150 MiB in Precise mode.  Local
    # tone is the only downstream consumer, so run it now and release the array
    # before face/surface analysis instead of carrying it through the pipeline.
    local_tone = analyze_local_tone(spatial_linear, precision=precision_profile.key)
    del spatial_linear
    # These full-frame intermediates have already been reduced to scalar/histogram
    # measurements.  Keeping them alive until the end of analyze_classical caused
    # severe memory pressure and swap-like stalls on otherwise fast later stages.
    del linear, gray8, gx, gy, rgbf, pix_max, pix_min, pix_mean, pix_chroma, valid_tone, rch, gch, bch

    detail_source = detail_rgb if precision_profile.deep_analysis else rgb

    jpeg_rgb, jpeg_scale_name = rgb, scale_name
    if precision_profile.deep_analysis and str(image.info.format or "").upper() in {"JPEG", "JPG"}:
        jpeg_rgb, jpeg_scale_name = _native_center_crop(image.srgb, 6000)

    serial_cv_threads, parallel_cv_threads, pipeline_workers = get_pipeline_profile()

    # Run the large independent full-frame maps before Haar face/eye cascades.
    # They do not depend on semantic regions.  On large Precise photos, doing
    # distance transforms / Sobel / NLM-like diagnostics after dozens of cascade
    # pyramid allocations caused severe allocator stalls despite low steady RSS.
    # Reordering changes scheduling only; pixels, thresholds and formulas are the
    # same. Surface stays after faces because it needs semantic protection boxes.
    if pipeline_workers >= 2:
        cv2.setNumThreads(parallel_cv_threads)
        try:
            with ThreadPoolExecutor(max_workers=pipeline_workers, thread_name_prefix="pd-analysis") as pool:
                futures = {
                    "sharpness": pool.submit(analyze_local_sharpness, rgb, map_rgb=spatial_rgb, precision=precision_profile.key),
                    "contrast": pool.submit(analyze_local_contrast, rgb, map_rgb=spatial_rgb, precision=precision_profile.key),
                    "jpeg": pool.submit(analyze_jpeg_artifacts, jpeg_rgb, image.info.format),
                    "edge": pool.submit(analyze_edge_artifacts, detail_source),
                    "posterization": pool.submit(analyze_posterization, detail_source),
                    "noise": pool.submit(analyze_noise, detail_source),
                    "blur": pool.submit(analyze_blur_type, rgb),
                    "nss": pool.submit(analyze_nss_baseline, rgb),
                    "auto_tone": pool.submit(analyze_auto_tone_color, rgb),
                }
                local_sharpness = futures["sharpness"].result()
                local_contrast = futures["contrast"].result()
                jpeg_artifacts = futures["jpeg"].result()
                edge_artifacts = futures["edge"].result()
                posterization = futures["posterization"].result()
                noise = futures["noise"].result()
                blur_type = futures["blur"].result()
                nss_baseline = futures["nss"].result()
                auto_tone_color = futures["auto_tone"].result()
        finally:
            cv2.setNumThreads(serial_cv_threads)
    else:
        local_sharpness = analyze_local_sharpness(rgb, map_rgb=spatial_rgb, precision=precision_profile.key)
        local_contrast = analyze_local_contrast(rgb, map_rgb=spatial_rgb, precision=precision_profile.key)
        jpeg_artifacts = analyze_jpeg_artifacts(jpeg_rgb, image.info.format)
        edge_artifacts = analyze_edge_artifacts(detail_source)
        posterization = analyze_posterization(detail_source)
        noise = analyze_noise(detail_source)
        blur_type = analyze_blur_type(rgb)
        nss_baseline = analyze_nss_baseline(rgb)
        auto_tone_color = analyze_auto_tone_color(rgb)

    # Face/eye detection remains on the CPU profile's best serial OpenCV setting.
    # Its many cascades already use OpenCV's native threading.
    cv2.setNumThreads(serial_cv_threads)
    vision_rgb = detail_source
    vision_face_analysis = analyze_faces(
        vision_rgb, precision=precision_profile.key, manual_boxes=manual_face_boxes
    )
    vision_eye_analysis = analyze_eyes(
        vision_rgb, vision_face_analysis.faces, manual_boxes=manual_eye_boxes, precision=precision_profile.key
    )
    red_eye_eyes = supplement_red_eye_eye_regions(vision_rgb, vision_face_analysis.faces, vision_eye_analysis.eyes)
    red_eye_analysis = analyze_red_eye(vision_rgb, red_eye_eyes)

    vh, vw = vision_rgb.shape[:2]
    protection_boxes: list[tuple[int, int, int, int, float]] = []
    for face in vision_face_analysis.faces:
        pad_x = int(round(face.w * 0.12)); pad_y = int(round(face.h * 0.16))
        x0 = max(0, face.x - pad_x); y0 = max(0, face.y - pad_y)
        x1 = min(vw, face.x + face.w + pad_x); y1 = min(vh, face.y + face.h + pad_y)
        protection_boxes.append((x0, y0, max(0, x1 - x0), max(0, y1 - y0), 0.58))
    for eye in vision_eye_analysis.eyes:
        pad_x = int(round(eye.w * 0.45)); pad_y = int(round(eye.h * 0.55))
        x0 = max(0, eye.x - pad_x); y0 = max(0, eye.y - pad_y)
        x1 = min(vw, eye.x + eye.w + pad_x); y1 = min(vh, eye.y + eye.h + pad_y)
        protection_boxes.append((x0, y0, max(0, x1 - x0), max(0, y1 - y0), 0.95))

    if precision_profile.key == "maximum":
        surface_max_boxes, surface_max_ai_boxes = 256, 320
    elif precision_profile.key == "precise":
        surface_max_boxes, surface_max_ai_boxes = 64, 144
    elif precision_profile.key == "fast":
        surface_max_boxes, surface_max_ai_boxes = 20, 48
    else:
        # Surface v9 favours recall in the visible candidate map. Repair is
        # manual-only, so hiding plausible scratches is worse than showing a few
        # extra candidates for the user to reject.
        surface_max_boxes, surface_max_ai_boxes = 36, 96
    if precision_profile.key == "maximum":
        # Maximum is additive, not an alternative threshold set. Preserve the entire
        # Precise Surface search, then add the denser Maximum pass and merge them.
        # This prevents a changed connected-component/Hough layout from making the
        # deepest mode forget a crack that Precise already saw.
        surface_precise = detect_surface_defects(
            detail_source,
            max_boxes=surface_max_boxes,
            max_ai_boxes=surface_max_ai_boxes,
            protection_boxes=protection_boxes,
            precision="precise",
        )
        surface_maximum = detect_surface_defects(
            detail_source,
            max_boxes=surface_max_boxes,
            max_ai_boxes=surface_max_ai_boxes,
            protection_boxes=protection_boxes,
            precision="maximum",
        )
        # Second independent proposal pass: local-contrast normalisation can reveal
        # faint scratches/cracks that are geometrically present but weak in the
        # original luminance. This is proposal-only; AI/Context and the user still
        # judge candidates against the untouched source image.
        probe_lab = cv2.cvtColor(detail_source, cv2.COLOR_RGB2LAB)
        probe_l, probe_a, probe_b = cv2.split(probe_lab)
        probe_l = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8)).apply(probe_l)
        surface_probe_rgb = cv2.cvtColor(cv2.merge((probe_l, probe_a, probe_b)), cv2.COLOR_LAB2RGB)
        surface_maximum_probe = detect_surface_defects(
            surface_probe_rgb,
            max_boxes=surface_max_boxes,
            max_ai_boxes=surface_max_ai_boxes,
            protection_boxes=protection_boxes,
            precision="maximum",
        )
        for collection in (surface_maximum_probe.boxes_norm, surface_maximum_probe.ai_boxes_norm):
            for box in collection:
                if isinstance(box, dict):
                    box["surface_precision_pass"] = "maximum_clahe_probe"

        # Maximum gets a second, independent local-normalisation proposal pass.
        # A smaller CLAHE tile grid can suppress very broad illumination gradients,
        # while the finer grid below preserves short scratches and broken crack
        # fragments.  Both are proposal-only: original-image AI/Context verification
        # and the user's explicit Defects-tab choice still control repair.
        probe_l_fine = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(12, 12)).apply(cv2.cvtColor(detail_source, cv2.COLOR_RGB2LAB)[:, :, 0])
        surface_probe_fine_rgb = cv2.cvtColor(
            cv2.merge((probe_l_fine, probe_a, probe_b)),
            cv2.COLOR_LAB2RGB,
        )
        surface_maximum_fine_probe = detect_surface_defects(
            surface_probe_fine_rgb,
            max_boxes=surface_max_boxes,
            max_ai_boxes=surface_max_ai_boxes,
            protection_boxes=protection_boxes,
            precision="maximum",
        )
        for collection in (surface_maximum_fine_probe.boxes_norm, surface_maximum_fine_probe.ai_boxes_norm):
            for box in collection:
                if isinstance(box, dict):
                    box["surface_precision_pass"] = "maximum_clahe_fine_probe"
        del surface_probe_rgb, surface_probe_fine_rgb, probe_l_fine, probe_lab, probe_l, probe_a, probe_b
        surface = merge_surface_detections(
            surface_maximum, surface_precise, surface_maximum_probe, surface_maximum_fine_probe,
            max_boxes=surface_max_boxes, max_ai_boxes=surface_max_ai_boxes,
        )
    else:
        surface = detect_surface_defects(
            detail_source,
            max_boxes=surface_max_boxes,
            max_ai_boxes=surface_max_ai_boxes,
            protection_boxes=protection_boxes,
            precision=precision_profile.key,
        )

    exif_context = analyze_exif_context(image.info.exif)

    if vision_rgb.shape[:2] != rgb.shape[:2]:
        face_analysis = _rescale_face_analysis(vision_face_analysis, vision_rgb.shape[:2], rgb.shape[:2])
        eye_analysis = _rescale_eye_analysis(vision_eye_analysis, vision_rgb.shape[:2], rgb.shape[:2])
    else:
        face_analysis = vision_face_analysis
        eye_analysis = vision_eye_analysis
    semantic_context = analyze_semantic_context(
        face_count=len(face_analysis.faces),
        face_detector_confidence=face_analysis.detector_confidence,
        tone_class=tone_class,
        tone_confidence=tone_confidence,
        surface_candidate_count=surface.candidate_count,
        surface_confidence=surface.confidence,
    )
    main_subject = analyze_main_subject(
        rgb,
        face_analysis.faces,
        eye_analysis.eyes,
        semantic_context,
        face_detector_confidence=face_analysis.detector_confidence,
    )
    white_balance_advice = analyze_white_balance(
        rgb,
        face_boxes=[(face.x, face.y, face.w, face.h) for face in face_analysis.faces],
        tone_class=tone_class,
        archival_likelihood=semantic_context.archival_likelihood,
        auto_tone_profile=auto_tone_color,
    )
    spatial_white_balance = analyze_spatial_white_balance(
        spatial_rgb,
        tone_class=tone_class,
        archival_likelihood=semantic_context.archival_likelihood,
    )
    detail_loss = refine_detail_loss_type(
        rgb,
        blur_type,
        local_sharpness,
        main_subject,
        exif_context,
        archival_likelihood=semantic_context.archival_likelihood,
        surface_candidate_count=surface.candidate_count,
    )
    blur_class = detail_loss.classification
    blur_confidence = detail_loss.confidence
    aesthetic_quality = analyze_aesthetic_quality(rgb, main_subject)
    highlight_context = analyze_highlight_context(
        rgb,
        gray_lin,
        [(face.x, face.y, face.w, face.h) for face in face_analysis.faces],
    )
    face_raw = [face.to_raw(rgb.shape[1], rgb.shape[0]) for face in face_analysis.faces]
    eye_raw = [eye.to_raw(rgb.shape[1], rgb.shape[0]) for eye in eye_analysis.eyes]
    if eye_analysis.eyes:
        eye_scores = [eye.sharpness_score for eye in eye_analysis.eyes]
        eye_measure_conf = [eye.measurement_confidence for eye in eye_analysis.eyes]
        eye_score = float(min(eye_scores))
        eye_confidence = float(min(eye_analysis.detector_confidence, sum(eye_measure_conf) / len(eye_measure_conf)))
    else:
        eye_score = None
        eye_confidence = eye_analysis.detector_confidence
    if face_analysis.faces:
        face_scores = [face.sharpness_score for face in face_analysis.faces]
        face_confidences = [face.measurement_confidence for face in face_analysis.faces]
        face_score = float(min(face_scores))
        face_confidence = float(min(face_analysis.detector_confidence, sum(face_confidences) / len(face_confidences)))
    else:
        face_score = None
        face_confidence = face_analysis.detector_confidence

    metrics = {
        "analysis_profile": MetricResult(
            "analysis_profile",
            {
                "key": precision_profile.key,
                "label": precision_profile.label,
                "source_width": image.info.width,
                "source_height": image.info.height,
                "technical_width": rgb.shape[1],
                "technical_height": rgb.shape[0],
                "technical_scale": scale_name,
                "spatial_width": spatial_rgb.shape[1],
                "spatial_height": spatial_rgb.shape[0],
                "spatial_scale": spatial_scale_name,
                "spatial_long_edge_limit": precision_profile.spatial_long_edge,
                "detail_width": detail_rgb.shape[1],
                "detail_height": detail_rgb.shape[0],
                "detail_scale": detail_scale_name,
                "detail_long_edge_limit": precision_profile.detail_long_edge,
                "deep_analysis": precision_profile.deep_analysis,
                "deep_stages": (
                    ["surface", "faces", "eyes", "red_eye", "noise", "edge_artifacts", "posterization", "jpeg_native_grid"]
                    if precision_profile.deep_analysis else []
                ),
            },
            None,
            1.0,
            "configuration",
            region="config",
            diagnostic="Профиль точности и фактические разрешения технического/пространственного анализа",
        ),
        "brightness": MetricResult("brightness", mean_luma, _brightness_quality_score(mean_luma), 0.85, scale_name,
                                   diagnostic="Средняя линейная яркость кадра; нормированная оценка учитывает широкий безопасный диапазон линейной яркости"),
        "histogram": MetricResult("histogram", {"bins": hist.tolist(), "p1": float(p1), "p5": float(p5), "p50": float(p50), "p95": float(p95), "p99": float(p99)}, None, 0.98, scale_name,
                                  diagnostic="Гистограмма яркости и основные процентили"),
        "shadow_clipping": MetricResult("shadow_clipping", shadow_clip, _clip01(100.0 - shadow_clip * 8.0), 0.90, scale_name,
                                        diagnostic="Доля почти полностью проваленных теней"),
        "highlight_clipping": MetricResult(
            "highlight_clipping",
            highlight_clip,
            _clip01(100.0 - highlight_context.unexplained_clip_pct * 8.0),
            0.90,
            scale_name,
            diagnostic="Общая доля почти полностью выбитых светов; нормированная оценка отдельно учитывает найденные кандидаты зеркальных бликов/источников света",
        ),
        "highlight_context": MetricResult(
            "highlight_context",
            {
                "total_clip_pct": highlight_context.total_clip_pct,
                "protected_clip_pct": highlight_context.protected_clip_pct,
                "unexplained_clip_pct": highlight_context.unexplained_clip_pct,
                "candidate_count": highlight_context.candidate_count,
                "face_rejected_count": highlight_context.face_rejected_count,
                "border_rejected_count": highlight_context.border_rejected_count,
                "boxes_norm": highlight_context.boxes_norm,
                "method": "compact_highlight_context_v1",
            },
            None,
            highlight_context.confidence,
            scale_name,
            region="highlights",
            diagnostic="Контекст потерь в светах: компактные кандидаты зеркальных бликов/источников света отделяются от необъяснённых выбитых светов",
        ),
        "contrast": MetricResult("contrast", contrast, _clip01(contrast * 140.0), 0.80, scale_name,
                                 diagnostic="Робастный глобальный контраст P95-P5"),
        "laplacian": MetricResult("laplacian", lap_var, _clip01(20.0 * np.log10(1.0 + lap_var)), 0.72, scale_name,
                                  diagnostic="Базовая оценка высокочастотной резкости; сама по себе не определяет физическую причину потери деталей"),
        "tenengrad": MetricResult("tenengrad", tenengrad, _clip01(14.0 * np.log10(1.0 + tenengrad)), 0.78, scale_name,
                                  diagnostic="Градиентная оценка резкости"),
        "detail_loss_type": MetricResult(
            "detail_loss_type",
            {
                "classification": blur_class,
                "raw_classification": blur_type.classification,
                "confidence": blur_confidence,
                "direction_deg": blur_type.direction_deg,
                "anisotropy": blur_type.anisotropy,
                "fine_to_coarse_ratio": blur_type.fine_to_coarse_ratio,
                "informative_patches": blur_type.informative_patches,
                "subject_sharpness": detail_loss.subject_sharpness,
                "background_sharpness": detail_loss.background_sharpness,
                "subject_background_delta": detail_loss.subject_background_delta,
                "subject_blur_classification": detail_loss.subject_blur_classification,
                "subject_blur_confidence": detail_loss.subject_blur_confidence,
                "shutter_risk": detail_loss.shutter_risk,
                "explanation_code": detail_loss.explanation_code,
                "reasons": list(detail_loss.reasons),
                "method": "contextual_detail_loss_v2",
            },
            None,
            blur_confidence,
            scale_name,
            region="global",
            diagnostic="Контекстная классификация потери деталей: глобальный/локальный смаз, дефокус, мягкость объекта или деградация оригинала",
        ),
        "jpeg_artifacts": MetricResult(
            "jpeg_artifacts",
            {
                "classification": jpeg_artifacts.classification,
                "block_ratio": jpeg_artifacts.block_ratio,
                "boundary_excess": jpeg_artifacts.boundary_excess,
                "best_offset": jpeg_artifacts.best_offset,
                "grid_dominance": jpeg_artifacts.grid_dominance,
                "source_is_jpeg": jpeg_artifacts.source_is_jpeg,
                "method": "jpeg_8x8_blocking_v1",
            },
            jpeg_artifacts.score,
            jpeg_artifacts.confidence,
            jpeg_scale_name,
            diagnostic="Оценка 8x8 блочности JPEG; регулярная геометрия сцены может давать ложные срабатывания",
        ),
        "edge_artifacts": MetricResult(
            "edge_artifacts",
            {
                "classification": edge_artifacts.classification,
                "halo_likelihood": edge_artifacts.halo_likelihood,
                "ringing_likelihood": edge_artifacts.ringing_likelihood,
                "near_edge_p90": edge_artifacts.near_edge_p90,
                "outer_ring_excess": edge_artifacts.outer_ring_excess,
                "edge_coverage": edge_artifacts.edge_coverage,
                "polarity_balance": edge_artifacts.polarity_balance,
                "method": "edge_halo_ringing_v1",
            },
            edge_artifacts.score,
            edge_artifacts.confidence,
            detail_scale_name if precision_profile.deep_analysis else scale_name,
            diagnostic="Кандидаты на ореолы, звон и перешарп вокруг сильных границ; контрастная геометрия может давать ложные срабатывания",
        ),
        "posterization": MetricResult(
            "posterization",
            {
                "classification": posterization.classification,
                "severity": posterization.severity,
                "occupied_luma_levels": posterization.occupied_luma_levels,
                "occupied_channel_levels_median": posterization.occupied_channel_levels_median,
                "tonal_span": posterization.tonal_span,
                "plateau_ratio": posterization.plateau_ratio,
                "jump_ratio": posterization.jump_ratio,
                "smooth_transition_samples": posterization.smooth_transition_samples,
                "method": "posterization_banding_v1",
            },
            posterization.score,
            posterization.confidence,
            detail_scale_name if precision_profile.deep_analysis else scale_name,
            diagnostic="Осторожная оценка постеризации и ступенчатых тональных переходов",
        ),
        "noise": MetricResult(
            "noise",
            {
                "sigma_luma": noise.sigma_luma,
                "sigma_p75": noise.sigma_p75,
                "classification": noise.classification,
                "flat_area_pct": noise.flat_area_pct,
                "usable_tiles": noise.usable_tiles,
                "total_tiles": noise.total_tiles,
                "cells_norm": noise.cells_norm,
                "map_cells": len(noise.cells_norm),
                "map_window_px": 96,
                "map_step_px": 96,
                "method": "weak_gradient_laplacian_tiles_v3_spatial",
            },
            noise.score,
            noise.confidence,
            detail_scale_name if precision_profile.deep_analysis else scale_name,
            diagnostic="Робастная оценка мелкого яркостного шума по слаботекстурным плиткам; дефекты поверхности анализируются отдельно",
        ),
        "color_cast": MetricResult("color_cast", {"mean_r": float(channel_mean[0]), "mean_g": float(channel_mean[1]), "mean_b": float(channel_mean[2]), "relative_spread": channel_spread},
                                   _clip01(100.0 - channel_spread * 180.0), 0.45, scale_name,
                                   diagnostic="Базовый признак цветового сдвига; художественный стиль и освещение учитываются на уровне профиля фото"),
        "neutral_balance": MetricResult(
            "neutral_balance",
            auto_tone_color.to_raw(),
            _clip01(100.0 - auto_tone_color.cast_strength * 520.0),
            auto_tone_color.confidence,
            scale_name,
            diagnostic="Баланс нейтральных полутонов: ищет цветовой сдвиг там, где средние RGB всего кадра могут взаимно компенсироваться",
        ),
        "white_balance_advisor": MetricResult(
            "white_balance_advisor",
            white_balance_advice.to_raw(),
            _clip01(100.0 * (1.0 - white_balance_advice.cast_strength)),
            white_balance_advice.confidence,
            scale_name,
            diagnostic="Советник баланса белого: объединяет несколько независимых оценок источника света и выдаёт температуру, tint и рекомендуемую степень нейтрализации",
        ),
        "local_white_balance": MetricResult(
            "local_white_balance",
            spatial_white_balance.to_raw(),
            _clip01(100.0 * (1.0 - spatial_white_balance.mixed_light_score)),
            spatial_white_balance.confidence,
            spatial_scale_name,
            diagnostic="Карта локального цветового освещения: использует только окна с надёжными нейтральными опорами, выявляет смешанный свет и не балансирует клетки независимо друг от друга",
        ),
        "image_tone": MetricResult(
            "image_tone",
            {
                "classification": tone_class,
                "confidence": tone_confidence,
                "mean_chroma": mean_chroma,
                "neutral_fraction": neutral_fraction,
                "warm_order_fraction": warm_order_fraction,
            },
            None,
            tone_confidence,
            scale_name,
            diagnostic="Профиль цветопередачи: цветное, почти монохромное или сепия-подобное изображение",
        ),
        "exif_context": MetricResult(
            "exif_context",
            exif_context.to_raw(),
            None,
            exif_context.confidence,
            "metadata",
            region="metadata",
            diagnostic="Контекст параметров съёмки из EXIF; используется только как дополнительное подтверждение, а не самостоятельная оценка качества",
        ),
        "nss_baseline": MetricResult(
            "nss_baseline",
            {
                "backend": nss_baseline.backend,
                "feature_count": nss_baseline.feature_count,
                "features": nss_baseline.features,
                "informative": nss_baseline.informative,
                "image_std": nss_baseline.image_std,
                "score_available": False,
                "reason": "BRISQUE/NIQE regression model is not bundled; this is an uncalibrated NSS feature baseline",
            },
            None,
            nss_baseline.confidence,
            scale_name,
            region="global",
            diagnostic="BRISQUE-style 36D natural-scene-statistics baseline without a pretrained quality regressor; informational only",
        ),
        "semantic_context": MetricResult(
            "semantic_context",
            semantic_context.to_raw(),
            None,
            semantic_context.confidence,
            scale_name,
            region="semantic",
            diagnostic="Осторожный сюжетный контекст на базе уже подтверждённых лиц, профиля тона и признаков архивного отпечатка; без идентификации людей",
        ),
        "main_subject": MetricResult(
            "main_subject",
            main_subject.to_raw(),
            None,
            main_subject.confidence,
            scale_name,
            region="semantic",
            diagnostic="Главный объект кадра: сначала подтверждённые лица/группа, иначе осторожный визуальный кандидат; слабый сигнал остаётся неопределённым",
        ),
        "aesthetic_quality": MetricResult(
            "aesthetic_quality",
            aesthetic_quality.to_raw(),
            aesthetic_quality.score,
            aesthetic_quality.confidence,
            scale_name,
            region="aesthetic",
            diagnostic="Отдельная композиционная оценка относительно главного объекта; не влияет на техническое качество и проверку безопасности",
        ),
        "faces": MetricResult(
            "faces",
            {
                "detector_available": face_analysis.detector_available,
                "detector": face_analysis.detector_name,
                "face_count": len(face_raw),
                "faces": face_raw,
            },
            face_score,
            face_confidence,
            scale_name,
            region="faces",
            diagnostic="Локализация лиц и базовая техническая оценка резкости/яркости внутри найденных областей",
        ),
        "eyes": MetricResult(
            "eyes",
            {
                "detector_available": eye_analysis.detector_available,
                "detector": eye_analysis.detector_name,
                "eye_count": len(eye_raw),
                "faces_with_eyes": eye_analysis.faces_with_eyes,
                "face_count": len(face_raw),
                "eyes": eye_raw,
            },
            eye_score,
            eye_confidence,
            scale_name,
            region="eyes",
            diagnostic="Осторожная локализация глаз внутри найденных лиц и локальная оценка их резкости; отсутствие детекции не трактуется как закрытые глаза",
        ),
        "local_tone": MetricResult(
            "local_tone",
            {
                "dark_area_pct": local_tone.dark_area_pct,
                "bright_area_pct": local_tone.bright_area_pct,
                "deep_shadow_area_pct": local_tone.deep_shadow_area_pct,
                "highlight_clip_area_pct": local_tone.highlight_clip_area_pct,
                "total_cells": local_tone.total_cells,
                "cells_norm": local_tone.cells_norm,
                "map_window_px": local_tone.map_window_px,
                "map_step_px": local_tone.map_step_px,
                "map_target_positions": local_tone.map_target_positions,
                "map_width": local_tone.map_width,
                "map_height": local_tone.map_height,
                "precision": precision_profile.key,
                "method": "resolution_aware_overlapping_linear_luma_v3",
            },
            None,
            local_tone.confidence,
            scale_name,
            region="grid",
            diagnostic="Информационная локальная карта тонов; сама по себе не считает тёмные/светлые объекты дефектом",
        ),
        "local_contrast": MetricResult(
            "local_contrast",
            {
                "low_contrast_area_pct": local_contrast.low_contrast_area_pct,
                "median_local_range": local_contrast.median_local_range,
                "fading_likelihood": local_contrast.fading_likelihood,
                "classification": local_contrast.classification,
                "informative_cells": local_contrast.informative_cells,
                "total_cells": local_contrast.total_cells,
                "map_cells": local_contrast.map_cells,
                "cells_norm": local_contrast.cells_norm,
                "map_window_px": local_contrast.map_window_px,
                "map_step_px": local_contrast.map_step_px,
                "map_target_positions": local_contrast.map_target_positions,
                "map_width": local_contrast.map_width,
                "map_height": local_contrast.map_height,
                "precision": precision_profile.key,
                "method": "resolution_aware_overlapping_local_contrast_v3",
            },
            local_contrast.score,
            local_contrast.confidence,
            scale_name,
            region="grid",
            diagnostic="Локальная оценка тонального разделения и осторожный признак выцветания/сжатия полутонов",
        ),
        "red_eye": MetricResult(
            "red_eye",
            {
                "eye_count": len(eye_analysis.eyes),
                "checked_eye_count": len(red_eye_eyes),
                "geometric_fallback_eye_count": sum(1 for eye in red_eye_eyes if eye.source == "red_eye_geometry"),
                "suspicious_eye_count": red_eye_analysis.suspicious_eye_count,
                "candidates": [c.to_raw(vision_rgb.shape[1], vision_rgb.shape[0]) for c in red_eye_analysis.candidates],
                "method": red_eye_analysis.detector_name,
            },
            _clip01(100.0 - min(100.0, sum(c.severity for c in red_eye_analysis.candidates if c.severity >= 14.0))),
            red_eye_analysis.confidence,
            detail_scale_name if precision_profile.deep_analysis else scale_name,
            region="eyes",
            diagnostic="Многоступенчатый поиск красного рефлекса: подтверждённый глаз + локальная краснота/форма/положение; геометрический fallback допускается только при согласованном binocular-подтверждении",
        ),
        "local_sharpness": MetricResult(
            "local_sharpness",
            {
                "soft_area_pct": local_sharpness.soft_area_pct,
                "informative_cells": local_sharpness.informative_cells,
                "total_cells": local_sharpness.total_cells,
                "cells_norm": local_sharpness.cells_norm,
                "map_window_px": local_sharpness.map_window_px,
                "map_step_px": local_sharpness.map_step_px,
                "map_target_positions": local_sharpness.map_target_positions,
                "map_width": local_sharpness.map_width,
                "map_height": local_sharpness.map_height,
                "map_cells": len(local_sharpness.cells_norm),
                "precision": precision_profile.key,
                "method": "resolution_aware_overlapping_laplacian_tenengrad_v3",
            },
            local_sharpness.score,
            local_sharpness.confidence,
            scale_name,
            region="grid",
            diagnostic="Локальная карта высокочастотной детализации с исключением малоинформативных гладких областей",
        ),
        "surface_defects": MetricResult(
            "surface_defects",
            {
                "candidate_density_pct": surface.candidate_density,
                "candidate_count": surface.candidate_count,
                "shown_count": len(surface.boxes_norm),
                "truncated": surface.candidate_count > len(surface.boxes_norm),
                "threshold": surface.threshold,
                "boxes_norm": surface.boxes_norm,
                "ai_boxes_norm": surface.ai_boxes_norm,
                "ai_candidate_count": len(surface.ai_boxes_norm),
                "method": "context_aware_morphology_v9_deep_recall_manual_only",
            },
            surface.score,
            surface.confidence,
            detail_scale_name if precision_profile.deep_analysis else scale_name,
            diagnostic="Контекстно проверенные кандидаты на локальные дефекты поверхности; итоговая семантическая проверка остаётся отдельным этапом",
        ),
    }
    super_resolution = analyze_super_resolution_need(rgb, metrics, source_shape=image.srgb.shape[:2])
    metrics["super_resolution"] = MetricResult(
        "super_resolution",
        super_resolution.to_raw(),
        super_resolution.need_score,
        super_resolution.confidence,
        "source",
        region="restoration",
        diagnostic="Осторожный детектор малых исходников и потенциала бережного x2-восстановления с защитой лица; не запускает увеличение сам по себе",
    )
    local_correction_plan = build_local_correction_plan(metrics)
    metrics["local_correction_plan"] = MetricResult(
        "local_correction_plan",
        local_correction_plan.to_raw(),
        None,
        max(
            local_correction_plan.exposure_confidence,
            local_correction_plan.contrast_confidence,
            local_correction_plan.sharpness_confidence,
            local_correction_plan.noise_confidence,
        ),
        scale_name,
        region="grid",
        diagnostic="План пространственной коррекции по согласованным картам тона/контраста/резкости/шума; нормальные области остаются нетронутыми",
    )
    reliability = analyze_analysis_reliability(metrics)
    metrics["analysis_reliability"] = MetricResult(
        "analysis_reliability",
        reliability.to_raw(),
        None,
        reliability.calibrated_confidence,
        scale_name,
        region="confidence",
        diagnostic="Единая защита надёжности: поддержка измерений, неопределённые результаты и эвристический признак нетипичного входа; не является вероятностью истины",
    )
    decision_plan = build_decision_plan(metrics, local_plan=local_correction_plan)
    if decision_plan:
        plan_confidence = float(sum(item.confidence for item in decision_plan) / len(decision_plan))
    else:
        plan_confidence = 0.0
    metrics["decision_plan"] = MetricResult(
        "decision_plan",
        {
            "engine_version": DECISION_ENGINE_VERSION,
            "items": [item.to_dict() for item in decision_plan],
            "fix_count": sum(item.decision == "fix" for item in decision_plan),
            "review_count": sum(item.decision == "review" for item in decision_plan),
            "preserve_count": sum(item.decision == "preserve" for item in decision_plan),
            "skip_count": sum(item.decision == "skip" for item in decision_plan),
        },
        None,
        plan_confidence,
        scale_name,
        region="decision",
        diagnostic="Приоритетный план действий на основе важности, исправимости и уверенности; сам по себе не меняет метрики качества",
    )
    return AnalysisResult(image=image.info, metrics=metrics)
