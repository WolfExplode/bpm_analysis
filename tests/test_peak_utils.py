"""Peak typing + prominence (scalar vs vectorized parity)."""
import numpy as np

import peak_utils as pu
from peak_utils import PeakType


def test_peaktype_is_s1_covers_paired_and_lone():
    assert PeakType.is_s1(PeakType.S1_PAIRED.value)
    assert PeakType.is_s1(PeakType.LONE_S1_VALIDATED.value)
    assert PeakType.is_s1(PeakType.LONE_S1_LAST.value)
    assert PeakType.is_s1("  S1 (Paired)  ")  # tolerates whitespace
    assert not PeakType.is_s1(PeakType.S2_PAIRED.value)
    assert not PeakType.is_s1(PeakType.NOISE.value)


def test_peaktype_is_s2():
    assert PeakType.is_s2(PeakType.S2_PAIRED.value)
    assert PeakType.is_s2(PeakType.S2_CORRECTED_GAP.value)
    assert not PeakType.is_s2(PeakType.S1_PAIRED.value)


def test_get_peak_type_from_debug_dict_str_and_legacy():
    assert pu._get_peak_type_from_debug({"peak_type": "S1 (Paired)"}) == "S1 (Paired)"
    assert pu._get_peak_type_from_debug("S2 (Paired)§TAG§v") == "S2 (Paired)"
    assert pu._get_peak_type_from_debug("") == ""
    assert pu._get_peak_type_from_debug(None) == ""


def test_simple_label_from_debug():
    assert pu._simple_label_from_debug({"peak_type": PeakType.S1_PAIRED.value}) == "S1"
    assert pu._simple_label_from_debug({"peak_type": PeakType.LONE_S1_VALIDATED.value}) == "S1"
    assert pu._simple_label_from_debug({"peak_type": PeakType.S2_PAIRED.value}) == "S2"
    assert pu._simple_label_from_debug({"peak_type": PeakType.NOISE.value}) == "Noise"
    assert pu._simple_label_from_debug({}) == "Unknown"
    assert pu._simple_label_from_debug(None) == "Unknown"


def _example_envelope():
    # troughs at indices 1,3,5,7 ; peaks at 2,4,6
    env = np.array([0.0, 0.2, 1.0, 0.1, 0.8, 0.3, 0.9, 0.0])
    troughs = np.array([1, 3, 5, 7])
    peaks = np.array([2, 4, 6])
    return env, troughs, peaks


def test_prominence_key_col_is_higher_trough():
    env, troughs, _ = _example_envelope()
    # peak 4 (amp 0.8): left trough idx3 amp0.1, right trough idx5 amp0.3 -> key col 0.3
    d = pu.get_peak_prominence_details(4, env, troughs)
    assert d["key_col_amp"] == 0.3
    assert abs(d["prominence"] - 0.5) < 1e-12


def test_prominence_clamped_non_negative():
    # peak lower than its surrounding troughs -> prominence floored at 0
    env = np.array([0.0, 0.9, 0.2, 0.9, 0.0])
    troughs = np.array([1, 3])
    d = pu.get_peak_prominence_details(2, env, troughs)
    assert d["prominence"] == 0.0


def test_prominence_no_troughs_returns_peak_amp():
    env, _, _ = _example_envelope()
    d = pu.get_peak_prominence_details(2, env, np.array([], dtype=int))
    assert d["prominence"] == env[2]
    assert d["key_col_amp"] == 0.0


def test_vectorized_cache_matches_scalar():
    env, troughs, peaks = _example_envelope()
    cache = pu.build_peak_prominence_detail_cache(peaks, env, troughs, sample_rate=600)
    for p in peaks:
        scalar = pu.get_peak_prominence_details(int(p), env, troughs, sample_rate=600)
        vec = cache[int(p)]
        assert abs(scalar["prominence"] - vec["prominence"]) < 1e-12
        assert abs(scalar["key_col_amp"] - vec["key_col_amp"]) < 1e-12
        assert scalar["left_trough_idx"] == vec["left_trough_idx"]
        assert scalar["right_trough_idx"] == vec["right_trough_idx"]


def test_vectorized_handles_peak_after_last_trough():
    # Peak index beyond the last trough must not index out of bounds.
    env = np.array([0.0, 0.2, 1.0, 0.1, 0.8])
    troughs = np.array([1, 3])
    peaks = np.array([2, 4])  # peak 4 has no right trough
    cache = pu.build_peak_prominence_detail_cache(peaks, env, troughs)
    assert cache[4]["right_trough_idx"] is None
    assert cache[4]["left_trough_idx"] == 3
    # key col = left trough amp (only neighbor)
    assert cache[4]["key_col_amp"] == env[3]


def test_calculate_peak_prominence_uses_cache():
    env, troughs, peaks = _example_envelope()
    cache = pu.build_peak_prominence_detail_cache(peaks, env, troughs)
    val = pu.calculate_peak_prominence(4, env, troughs, detail_cache=cache)
    assert abs(val - 0.5) < 1e-12
