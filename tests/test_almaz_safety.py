from __future__ import annotations

import cv2
import numpy as np

from photodoctor.core.almaz_safety import validate_almaz_transition


def _textured(h=64, w=80):
    y, x = np.indices((h, w))
    base = (70 + ((x // 7 + y // 9) % 2) * 80 + ((x * 3 + y * 2) % 17)).astype(np.uint8)
    return np.dstack([base, np.clip(base + 5, 0, 255), np.clip(base - 4, 0, 255)]).astype(np.uint8)


def test_almaz_safety_accepts_conservative_x2():
    before = _textured()
    after = cv2.resize(before, (before.shape[1] * 2, before.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    result = validate_almaz_transition(
        before, after, action_key="super_resolution", candidate="almaz_x2_identity_guard_v1",
        parameters={"face_boxes": [{"x": 0.25, "y": 0.25, "w": 0.3, "h": 0.3}]},
    )
    assert result.accepted is True


def test_almaz_safety_blocks_wrong_x2_shape():
    before = _textured()
    result = validate_almaz_transition(
        before, before.copy(), action_key="super_resolution", candidate="almaz_x2_identity_guard_v1",
    )
    assert result.accepted is False
    assert "размерный контракт" in result.message


def test_almaz_safety_blocks_extreme_hallucinated_output():
    before = _textured()
    after = np.full((before.shape[0] * 2, before.shape[1] * 2, 3), 255, dtype=np.uint8)
    result = validate_almaz_transition(
        before, after, action_key="super_resolution", candidate="almaz_x2_identity_guard_v1",
        parameters={"face_boxes": [{"x": 0.2, "y": 0.2, "w": 0.5, "h": 0.5}]},
    )
    assert result.accepted is False
    assert result.mean_abs_delta_pct > 20.0


def test_almaz_safety_flat_source_does_not_fail_edge_ratio():
    before = np.full((48, 64, 3), 100, dtype=np.uint8)
    after = np.full_like(before, 107)
    result = validate_almaz_transition(
        before, after, action_key="noise", candidate="almaz_ai_denoise_v1",
        parameters={"face_boxes": []},
    )
    assert result.accepted is True
    assert result.edge_ratio == 1.0


def test_almaz_safety_flat_source_blocks_invented_strong_edges():
    before = np.full((48, 64, 3), 100, dtype=np.uint8)
    after = before.copy()
    after[:, ::2] = 135
    result = validate_almaz_transition(
        before, after, action_key="noise", candidate="almaz_ai_denoise_v1",
        parameters={"face_boxes": []},
    )
    assert result.accepted is False
    assert "edge" in result.message
