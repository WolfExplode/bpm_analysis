"""fft_profiles: peak selection/aggregation helpers used to build S1/S2 frequency profiles."""
import numpy as np

from fft_profiles import (
    _align_s2_to_s1_in_band,
    _collect_s1_s2_indices,
    _get_pairing_confidence,
    _peak_indices_to_full_rate,
    _select_top_peaks_by_confidence,
    aggregate_fft_profiles,
    compute_frequency_separation,
)


# --- _get_pairing_confidence --------------------------------------------------

def test_pairing_confidence_prefers_label_scores_for_s1():
    entry = {"peak_type": "S1 (Paired)", "label_scores": {"S1": 0.8, "S2": 0.1}}
    assert _get_pairing_confidence(entry) == 0.8


def test_pairing_confidence_prefers_label_scores_for_s2():
    entry = {"peak_type": "S2 (Paired)", "label_scores": {"S1": 0.1, "S2": 0.9}}
    assert _get_pairing_confidence(entry) == 0.9


def test_pairing_confidence_falls_back_to_confidence_trace():
    entry = {
        "peak_type": "S1 (Paired)",
        "sections": [
            {"type": "other", "steps": []},
            {
                "type": "confidence_trace",
                "steps": [{"result": 0.2}, {"result": 0.55}],
            },
        ],
    }
    assert _get_pairing_confidence(entry) == 0.55


def test_pairing_confidence_non_dict_entry_returns_none():
    assert _get_pairing_confidence("not a dict") is None


def test_pairing_confidence_missing_sections_returns_none():
    assert _get_pairing_confidence({"peak_type": "S1 (Paired)"}) is None


def test_pairing_confidence_non_numeric_label_score_falls_through():
    entry = {
        "peak_type": "S1 (Paired)",
        "label_scores": {"S1": "not a number"},
        "sections": [
            {"type": "confidence_trace", "steps": [{"result": 0.4}]},
        ],
    }
    assert _get_pairing_confidence(entry) == 0.4


# --- _collect_s1_s2_indices ---------------------------------------------------

def _classifications():
    return {
        1: {"peak_type": "S1 (Paired)"},
        2: {"peak_type": "S2 (Paired)"},
        3: {"peak_type": "Lone S1"},
        4: {"peak_type": "Noise/Rejected"},
    }


def test_collect_indices_splits_by_type():
    s1, s2 = _collect_s1_s2_indices(_classifications())
    assert s1 == [1, 3]
    assert s2 == [2]


def test_collect_indices_paired_s1_only_excludes_lone_s1():
    s1, s2 = _collect_s1_s2_indices(_classifications(), paired_s1_only=True)
    assert s1 == [1]
    assert s2 == [2]


# --- _select_top_peaks_by_confidence ------------------------------------------

def test_select_top_peaks_sorts_descending_and_caps():
    classifications = {
        1: {"peak_type": "S1 (Paired)", "label_scores": {"S1": 0.2}},
        2: {"peak_type": "S1 (Paired)", "label_scores": {"S1": 0.9}},
        3: {"peak_type": "S1 (Paired)", "label_scores": {"S1": 0.5}},
    }
    s1, _ = _select_top_peaks_by_confidence(classifications, [1, 2, 3], [], max_per_type=2)
    assert s1 == [2, 3]


def test_select_top_peaks_missing_confidence_defaults_to_zero():
    classifications = {1: {"peak_type": "S1 (Paired)"}, 2: {"peak_type": "S1 (Paired)", "label_scores": {"S1": 0.1}}}
    s1, _ = _select_top_peaks_by_confidence(classifications, [1, 2], [], max_per_type=10)
    assert s1 == [2, 1]


def test_select_top_peaks_empty_indices_returns_empty():
    s1, s2 = _select_top_peaks_by_confidence({}, [], [], max_per_type=5)
    assert s1 == []
    assert s2 == []


# --- _peak_indices_to_full_rate -----------------------------------------------

def test_peak_indices_to_full_rate_scales_correctly():
    result = _peak_indices_to_full_rate([600], envelope_sample_rate=600, full_sample_rate=32000)
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, [32000])


def test_peak_indices_to_full_rate_empty_input_returns_empty_int64_array():
    result = _peak_indices_to_full_rate([], envelope_sample_rate=600, full_sample_rate=32000)
    assert result.dtype == np.int64
    assert result.size == 0


# --- _align_s2_to_s1_in_band ---------------------------------------------------

def test_align_s2_to_s1_shifts_s2_mean_to_match_s1():
    freqs = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    s1_db = np.array([1.0, 2.0, 10.0, 10.0, 3.0])
    s2_db = np.array([1.0, 2.0, 5.0, 7.0, 3.0])
    aligned_s1, aligned_s2 = _align_s2_to_s1_in_band(freqs, s1_db, s2_db, low_hz=10.0, high_hz=15.0)
    np.testing.assert_allclose(aligned_s1, s1_db)  # S1 untouched
    # Mean of aligned S2 in-band must now equal mean of S1 in-band.
    mask = (freqs >= 10.0) & (freqs <= 15.0)
    assert np.mean(aligned_s2[mask]) == np.mean(aligned_s1[mask])


def test_align_s2_to_s1_empty_mask_returns_inputs_unchanged():
    freqs = np.array([0.0, 5.0, 10.0])
    s1_db = np.array([1.0, 2.0, 3.0])
    s2_db = np.array([4.0, 5.0, 6.0])
    aligned_s1, aligned_s2 = _align_s2_to_s1_in_band(freqs, s1_db, s2_db, low_hz=100.0, high_hz=200.0)
    np.testing.assert_array_equal(aligned_s1, s1_db)
    np.testing.assert_array_equal(aligned_s2, s2_db)


# --- compute_frequency_separation ---------------------------------------------

def test_compute_frequency_separation_basic():
    freqs = np.linspace(0, 100, 11)  # 0..100 step 10
    s1_db = np.full(11, 5.0)
    s2_db = np.full(11, 3.0)
    result = compute_frequency_separation(freqs, s1_db, s2_db, params={"fft_separation_low_hz": 10.0, "fft_separation_high_hz": 50.0})
    assert result is not None
    np.testing.assert_allclose(result["separation_db"], 2.0)
    assert result["freqs"].min() >= 10.0
    assert result["freqs"].max() <= 50.0


def test_compute_frequency_separation_none_freqs_returns_none():
    assert compute_frequency_separation(None, np.array([1.0]), np.array([1.0])) is None


def test_compute_frequency_separation_empty_freqs_returns_none():
    assert compute_frequency_separation(np.array([]), np.array([]), np.array([])) is None


def test_compute_frequency_separation_mismatched_lengths_returns_none():
    freqs = np.array([1.0, 2.0, 3.0])
    s1_db = np.array([1.0, 2.0])
    s2_db = np.array([1.0, 2.0, 3.0])
    assert compute_frequency_separation(freqs, s1_db, s2_db) is None


def test_compute_frequency_separation_band_outside_range_returns_none():
    freqs = np.array([1.0, 2.0, 3.0])
    s1_db = np.array([1.0, 2.0, 3.0])
    s2_db = np.array([1.0, 2.0, 3.0])
    result = compute_frequency_separation(
        freqs, s1_db, s2_db, params={"fft_separation_low_hz": 1000.0, "fft_separation_high_hz": 2000.0}
    )
    assert result is None


# --- aggregate_fft_profiles ---------------------------------------------------

def _file_result(freqs, s1_val, s2_val, n1, n2):
    n = len(freqs)
    return (freqs, np.full(n, s1_val), np.full(n, s2_val), np.full(n, s1_val), np.full(n, s2_val), n1, n2)


def test_aggregate_fft_profiles_empty_input_returns_empty_arrays():
    freqs, r1, r2, b1, b2 = aggregate_fft_profiles([])
    assert freqs.size == 0 and r1.size == 0 and r2.size == 0 and b1.size == 0 and b2.size == 0


def test_aggregate_fft_profiles_weighted_by_peak_count():
    freqs = np.array([0.0, 100.0])
    results = [
        _file_result(freqs, s1_val=10.0, s2_val=10.0, n1=1, n2=1),
        _file_result(freqs, s1_val=20.0, s2_val=20.0, n1=3, n2=3),
    ]
    agg_freqs, agg_r1, agg_r2, agg_b1, agg_b2 = aggregate_fft_profiles(
        results, params={"fft_neutral_band_low_hz": 0.0, "fft_neutral_band_high_hz": 100.0}
    )
    np.testing.assert_array_equal(agg_freqs, freqs)
    # weighted mean of [10*1, 20*3] / 4 = 17.5, and s1==s2 so neutral-band alignment keeps it unchanged.
    np.testing.assert_allclose(agg_r1, 17.5)
    np.testing.assert_allclose(agg_r2, 17.5)


def test_aggregate_fft_profiles_skips_bin_count_mismatch():
    freqs = np.array([0.0, 100.0])
    mismatched_freqs = np.array([0.0, 50.0, 100.0])
    results = [
        _file_result(freqs, s1_val=10.0, s2_val=10.0, n1=1, n2=1),
        _file_result(mismatched_freqs, s1_val=999.0, s2_val=999.0, n1=5, n2=5),
    ]
    agg_freqs, agg_r1, agg_r2, agg_b1, agg_b2 = aggregate_fft_profiles(results)
    np.testing.assert_array_equal(agg_freqs, freqs)
    np.testing.assert_allclose(agg_r1, 10.0)


def test_aggregate_fft_profiles_all_zero_counts_returns_zero_sums():
    freqs = np.array([0.0, 100.0])
    results = [_file_result(freqs, s1_val=10.0, s2_val=10.0, n1=0, n2=0)]
    agg_freqs, agg_r1, agg_r2, agg_b1, agg_b2 = aggregate_fft_profiles(results)
    np.testing.assert_array_equal(agg_r1, [0.0, 0.0])
    np.testing.assert_array_equal(agg_r2, [0.0, 0.0])
