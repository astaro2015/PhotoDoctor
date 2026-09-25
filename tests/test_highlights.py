import numpy as np

from photodoctor.core.highlights import analyze_highlight_context


def to_linear(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32) / 255.0
    linear = np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4)
    return 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]


def test_small_isolated_glint_is_protected_candidate():
    img = np.full((320, 420, 3), 80, np.uint8)
    yy, xx = np.indices(img.shape[:2])
    mask = (xx - 210) ** 2 + (yy - 150) ** 2 <= 7 ** 2
    img[mask] = 255
    result = analyze_highlight_context(img, to_linear(img))
    assert result.candidate_count >= 1
    assert result.protected_clip_pct > 0.0
    assert result.unexplained_clip_pct < result.total_clip_pct


def test_large_white_region_is_not_auto_protected():
    img = np.full((320, 420, 3), 70, np.uint8)
    img[70:250, 180:380] = 255
    result = analyze_highlight_context(img, to_linear(img))
    assert result.total_clip_pct > 10
    assert result.protected_clip_pct == 0.0
    assert result.unexplained_clip_pct == result.total_clip_pct


def test_border_connected_white_area_is_not_auto_protected():
    img = np.full((320, 420, 3), 70, np.uint8)
    img[0:90, 0:60] = 255
    result = analyze_highlight_context(img, to_linear(img))
    assert result.border_rejected_count >= 1
    assert result.protected_clip_pct == 0.0


def test_bright_spot_inside_face_box_is_not_auto_protected():
    img = np.full((320, 420, 3), 75, np.uint8)
    img[145:155, 205:215] = 255
    result = analyze_highlight_context(img, to_linear(img), [(160, 95, 120, 140)])
    assert result.face_rejected_count >= 1
    assert result.protected_clip_pct == 0.0


def test_full_analysis_uses_unexplained_clipping_for_normalized_score(tmp_path):
    from PIL import Image
    from photodoctor.core.service import analyze_file

    img = np.full((320, 420, 3), 70, np.uint8)
    yy, xx = np.indices(img.shape[:2])
    mask = (xx - 210) ** 2 + (yy - 150) ** 2 <= 12 ** 2
    img[mask] = 255
    path = tmp_path / "glint.png"
    Image.fromarray(img).save(path)
    result = analyze_file(path)
    clip = result.metrics["highlight_clipping"]
    context = result.metrics["highlight_context"]
    assert clip.raw_value > 0.2
    assert clip.normalized_value > 99.0
    assert context.raw_value["protected_clip_pct"] > 0.0
    assert context.raw_value["candidate_count"] >= 1
