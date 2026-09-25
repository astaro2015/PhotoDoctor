from __future__ import annotations

import cv2
import numpy as np

from photodoctor.core.almaz_chroma_guard import archival_chroma_profile, guard_archival_chroma, guard_deblur_chroma
from photodoctor.core.almaz_safety import validate_almaz_transition


def _sepia_textured(h: int = 120, w: int = 160) -> np.ndarray:
    y, x = np.indices((h, w))
    gray = (72 + ((x // 9 + y // 11) % 2) * 74 + ((x * 2 + y * 3) % 23)).astype(np.uint8)
    # Warm, coherent sepia cast with real luminance texture.
    rgb = np.dstack([
        np.clip(gray.astype(np.int16) + 22, 0, 255),
        np.clip(gray.astype(np.int16) + 10, 0, 255),
        np.clip(gray.astype(np.int16) - 8, 0, 255),
    ]).astype(np.uint8)
    return rgb


def _add_green_islands(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    # >1% of frame so the p99 local-chroma statistic must see it, but small
    # enough that whole-frame mean chroma can remain deceptively modest.
    out[h // 6:h // 6 + h // 8, w // 7:w // 7 + w // 7] = np.array([65, 142, 74], np.uint8)
    out[h * 3 // 5:h * 3 // 5 + h // 10, w * 2 // 3:w * 2 // 3 + w // 9] = np.array([58, 132, 68], np.uint8)
    return out


def test_archive_profile_recognizes_coherent_sepia_but_not_colourful_photo():
    sepia = _sepia_textured()
    archival, center, spread = archival_chroma_profile(sepia)
    assert archival is True
    assert center < 42.0
    assert spread <= 10.0

    colourful = sepia.copy()
    colourful[:, : colourful.shape[1] // 3] = [220, 40, 50]
    colourful[:, colourful.shape[1] // 3: 2 * colourful.shape[1] // 3] = [35, 190, 60]
    colourful[:, 2 * colourful.shape[1] // 3:] = [40, 70, 220]
    archival2, _center2, _spread2 = archival_chroma_profile(colourful)
    assert archival2 is False


def test_guard_suppresses_green_islands_while_preserving_almaz_luminance():
    before = _sepia_textured()
    # Simulate useful ALMAZ luminance/detail work first.
    lab = cv2.cvtColor(before, cv2.COLOR_RGB2LAB)
    l = lab[..., 0]
    detail = cv2.addWeighted(l, 1.12, cv2.GaussianBlur(l, (0, 0), 0.8), -0.12, 0)
    lab2 = lab.copy()
    lab2[..., 0] = detail
    useful = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    corrupted = _add_green_islands(useful)

    raw_safety = validate_almaz_transition(
        before, corrupted, action_key="almaz_denoise", candidate="almaz_ai_denoise_v1",
        parameters={"face_boxes": []},
    )
    assert raw_safety.accepted is False
    assert raw_safety.local_chroma_p99_lab > 4.0
    assert "local_chroma" in raw_safety.message

    guarded, info = guard_archival_chroma(before, corrupted)
    assert info.applied is True
    assert info.archival_like is True
    assert info.guarded_local_chroma_p99 < info.raw_local_chroma_p99
    assert info.guarded_local_chroma_p99 <= 1.5

    # The AI luminance/detail survives; only colour is pulled back toward source.
    useful_l = cv2.cvtColor(useful, cv2.COLOR_RGB2LAB)[..., 0].astype(np.int16)
    guarded_l = cv2.cvtColor(guarded, cv2.COLOR_RGB2LAB)[..., 0].astype(np.int16)
    assert float(np.mean(np.abs(useful_l - guarded_l))) <= 1.5

    guarded_safety = validate_almaz_transition(
        before, guarded, action_key="almaz_denoise", candidate="almaz_ai_denoise_v1",
        parameters={"face_boxes": []},
    )
    assert guarded_safety.accepted is True
    assert guarded_safety.local_chroma_p99_lab <= 4.0


def test_guard_leaves_normal_colour_photo_untouched():
    h, w = 90, 120
    y, x = np.indices((h, w))
    normal = np.dstack([
        (40 + x * 170 // max(1, w - 1)).astype(np.uint8),
        (35 + y * 180 // max(1, h - 1)).astype(np.uint8),
        (210 - x * 120 // max(1, w - 1)).astype(np.uint8),
    ])
    after = cv2.GaussianBlur(normal, (0, 0), 0.5)
    guarded, info = guard_archival_chroma(normal, after)
    assert info.applied is False
    assert info.archival_like is False
    assert np.array_equal(guarded, after)


def _colourful_textured(h: int = 120, w: int = 160) -> np.ndarray:
    y, x = np.indices((h, w))
    r = 45 + (x * 150 // max(1, w - 1)) + ((x + y) % 13)
    g = 55 + (y * 135 // max(1, h - 1)) + ((x * 2 + y) % 11)
    b = 205 - (x * 105 // max(1, w - 1)) + ((x + y * 3) % 9)
    return np.clip(np.dstack([r, g, b]), 0, 255).astype(np.uint8)


def test_deblur_guard_suppresses_green_islands_on_normal_colour_photo():
    before = _colourful_textured()
    lab = cv2.cvtColor(before, cv2.COLOR_RGB2LAB)
    l = lab[..., 0]
    sharpened = cv2.addWeighted(l, 1.18, cv2.GaussianBlur(l, (0, 0), 0.9), -0.18, 0)
    lab2 = lab.copy(); lab2[..., 0] = sharpened
    useful = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    corrupted = _add_green_islands(useful)

    raw = validate_almaz_transition(
        before, corrupted, action_key="almaz_deblur", candidate="almaz_ai_deblur_v1",
        parameters={"face_boxes": []},
    )
    assert raw.accepted is False
    assert "local_chroma" in raw.message
    assert raw.local_chroma_p99_lab > 6.0

    guarded, info = guard_deblur_chroma(before, corrupted)
    assert info.applied is True
    assert info.guarded_local_chroma_p99 < info.raw_local_chroma_p99
    assert info.guarded_local_chroma_p99 <= 2.0

    # Deblur detail lives mainly in luminance and must survive the colour clamp.
    useful_l = cv2.cvtColor(useful, cv2.COLOR_RGB2LAB)[..., 0].astype(np.int16)
    guarded_l = cv2.cvtColor(guarded, cv2.COLOR_RGB2LAB)[..., 0].astype(np.int16)
    assert float(np.mean(np.abs(useful_l - guarded_l))) <= 1.5

    safe = validate_almaz_transition(
        before, guarded, action_key="almaz_deblur", candidate="almaz_ai_deblur_v1",
        parameters={"face_boxes": []},
    )
    assert safe.local_chroma_p99_lab <= 6.0


def test_deblur_guard_skips_small_legitimate_chroma_change():
    before = _colourful_textured(80, 100)
    after = before.copy().astype(np.int16)
    after[..., 0] += 1
    after[..., 1] -= 1
    after = np.clip(after, 0, 255).astype(np.uint8)
    guarded, info = guard_deblur_chroma(before, after)
    assert info.applied is False
    assert np.array_equal(guarded, after)
