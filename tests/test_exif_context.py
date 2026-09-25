from photodoctor.core.exif_context import analyze_exif_context


def test_missing_exif_is_unknown_and_non_penalizing_context():
    result = analyze_exif_context({})
    assert not result.available
    assert result.confidence == 0.0
    assert result.shutter_risk == "unknown"
    assert result.iso_noise_risk == "unknown"


def test_fractional_shutter_and_focal_length_produce_handheld_hint():
    result = analyze_exif_context({"exposure_time": "1/30", "focal_length": "50", "iso": "400"})
    assert result.available
    assert abs(result.exposure_s - 1 / 30) < 1e-6
    assert result.focal_length_mm == 50.0
    assert result.shutter_ratio_to_reciprocal > 1.5
    assert result.shutter_risk == "high"
    assert result.iso_noise_risk == "moderate"


def test_fast_shutter_and_low_iso_are_low_risk():
    result = analyze_exif_context({"exposure_time": "1/500", "focal_length": "50/1", "iso": "100"})
    assert result.shutter_risk == "low"
    assert result.iso_noise_risk == "low"


def test_high_iso_is_only_context_not_quality_score():
    result = analyze_exif_context({"iso": "3200"})
    assert result.iso_noise_risk == "high"
    assert result.shutter_risk == "unknown"
    assert 0.28 <= result.confidence <= 0.72


def test_invalid_exif_values_do_not_crash_or_invent_numbers():
    result = analyze_exif_context({"exposure_time": "n/a", "focal_length": "0/0", "iso": "???", "model": "Scanner"})
    assert result.available
    assert result.exposure_s is None
    assert result.focal_length_mm is None
    assert result.iso is None
    assert result.shutter_risk == "unknown"
