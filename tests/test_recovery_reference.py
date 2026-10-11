"""Seated heuristic arithmetic and recording metadata persistence."""
import pytest

from pcg.recovery_reference import estimate_seated_recovery
from pcg import recording_metadata


def test_metadata_roundtrip_identity_and_no_cross_recording_leak(tmp_path):
    data = {"version": 1, "person": {"age": 40, "gender": "self-described"},
            "recoveries": {"10.0": {"protocol": "adult_cycle"}}}
    recording_metadata.save(tmp_path, "fp-one", data)
    assert recording_metadata.load(tmp_path, "fp-one") == data
    assert recording_metadata.load(tmp_path, "fp-two")["person"] == {}
    assert not recording_metadata.metadata_path(tmp_path, "fp-one").with_suffix(".tmp").exists()
    with pytest.raises(ValueError):
        recording_metadata.load(tmp_path, "../outside")


def test_corrupt_metadata_is_not_silently_replaced(tmp_path):
    path = recording_metadata.metadata_path(tmp_path, "fp")
    path.write_text('{"version":2}', encoding="utf-8")
    with pytest.raises(ValueError):
        recording_metadata.load(tmp_path, "fp")
    assert path.read_text() == '{"version":2}'


def test_seated_heuristic_matches_research_examples_and_five_minute_limit():
    assert estimate_seated_recovery(150, 74, 0) == 150
    for seconds, expected in zip((30, 60, 120, 180, 300), (124, 110, 98, 95, 93)):
        assert round(estimate_seated_recovery(150, 74, seconds)) == expected
    assert estimate_seated_recovery(80, 80, 300) == 80
    values = [estimate_seated_recovery(150, 74, seconds) for seconds in (0, 30, 60, 120, 180, 300)]
    assert all(a > b for a, b in zip(values, values[1:]))
    assert all(74 < value <= 150 for value in values)


@pytest.mark.parametrize("start,rest,seconds", [
    (150, None, 60), (None, 74, 60), (150, float("nan"), 60),
    (150, 74, float("inf")), (150, 0, 60), (70, 74, 60),
    (150, 74, -1), (150, 74, 301), (150, 74, 600), (True, 74, 60),
])
def test_seated_heuristic_rejects_missing_invalid_inputs_and_extrapolation(start, rest, seconds):
    with pytest.raises(ValueError):
        estimate_seated_recovery(start, rest, seconds)


@pytest.mark.parametrize("seconds,age_coefficient,sex_coefficient", [
    (30, -0.18, 2.15), (60, -0.18, 3.98),
    (120, -0.13, 6.04), (180, -0.12, 6.74),
])
def test_age_sex_checkpoint_effects(seconds, age_coefficient, sex_coefficient):
    baseline = estimate_seated_recovery(150, 74, seconds)
    female20 = estimate_seated_recovery(150, 74, seconds, age=20, reported_sex="female")
    male40 = estimate_seated_recovery(150, 74, seconds, age=40, reported_sex="male")
    assert female20 == pytest.approx(baseline)
    assert male40 == pytest.approx(baseline - 20 * age_coefficient + sex_coefficient)


@pytest.mark.parametrize("age,male,female", [(20, 114, 110), (40, 118, 114), (60, 121, 117)])
def test_age_sex_matches_document_one_minute_examples(age, male, female):
    assert round(estimate_seated_recovery(150, 74, 60, age=age, reported_sex="male")) == male
    assert round(estimate_seated_recovery(150, 74, 60, age=age, reported_sex="female")) == female


@pytest.mark.parametrize("age,sex", [
    (None, "male"), (16, "male"), (70, "female"),
    (float("nan"), "male"), (True, "female"), (20, None), (20, "other"),
])
def test_age_sex_requires_supported_demographics(age, sex):
    with pytest.raises(ValueError):
        estimate_seated_recovery(150, 74, 60, age=age, reported_sex=sex)


def test_age_sex_omits_unsupported_or_incompatible_estimates():
    assert estimate_seated_recovery(150, 74, 0, age=40, reported_sex="male") == 150
    for seconds in (45, 300):
        assert estimate_seated_recovery(150, 74, seconds, age=40, reported_sex="male") is None
    # Neither invent recovery below rest nor clamp a negative cumulative drop.
    assert estimate_seated_recovery(81, 80, 60, age=60, reported_sex="male") is None
    assert estimate_seated_recovery(81, 80, 60, age=17, reported_sex="female") is None
