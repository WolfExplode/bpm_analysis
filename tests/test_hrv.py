"""HRV outlier masks, duration clamps, and windowed metrics."""
import numpy as np

import hrv


def test_global_mad_mask_flags_single_outlier():
    # Needs non-zero spread among the inliers, else MAD==0 and the rule keeps all.
    v = np.array([1.0, 2.0, 1.5, 2.5, 100.0])
    keep = hrv._median_mad_keep_mask_global(v, mad_k=3.0)
    assert keep[:4].all()
    assert not keep[4]


def test_global_mad_mask_keeps_all_when_no_spread():
    v = np.array([5.0, 5.0, 5.0])
    assert hrv._median_mad_keep_mask_global(v, mad_k=3.0).all()


def test_global_mad_mask_empty():
    assert hrv._median_mad_keep_mask_global(np.array([]), 3.0).size == 0


def test_time_window_mad_mask_local_outlier():
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    v = np.array([1.0, 2.0, 9.0, 2.0, 1.0])  # spread keeps local MAD > 0
    keep = hrv._median_mad_keep_mask_time_window(t, v, half_window_sec=2.0, mad_k=2.0)
    assert not keep[2]
    assert keep[0] and keep[4]


def test_filter_interval_durations_drops_out_of_bounds():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([0.001, 0.2, 5.0, 0.3])  # too short, ok, too long, ok
    params = {
        "systole_duration_clamp_min_sec": 0.02,
        "systole_duration_clamp_max_sec": 3.0,
    }
    tt, vv = hrv.filter_interval_durations_by_limits(t, v, kind="systole", params=params)
    np.testing.assert_allclose(vv, [0.2, 0.3])
    np.testing.assert_allclose(tt, [1.0, 3.0])


def test_filter_interval_durations_unknown_kind_passthrough():
    t = np.array([0.0, 1.0])
    v = np.array([0.1, 0.2])
    tt, vv = hrv.filter_interval_durations_by_limits(t, v, kind="bogus")
    np.testing.assert_array_equal(tt, t)
    np.testing.assert_array_equal(vv, v)


def test_windowed_hrv_regular_rhythm():
    sr = 100
    # 10 beats exactly 1s apart -> RR = 1.0s, bpm = 60, rmssd ~ 0
    peaks = np.arange(10) * sr
    params = {
        "hrv_window_size_beats": 4,
        "hrv_step_size_beats": 2,
        "enable_hrv_frequency_domain": False,
    }
    df = hrv.calculate_windowed_hrv(peaks, sr, params)
    assert not df.empty
    assert np.allclose(df["bpm"], 60.0)
    assert np.allclose(df["rmssdc"], 0.0, atol=1e-9)
    assert np.allclose(df["sdnn"], 0.0, atol=1e-9)


def test_windowed_hrv_too_few_beats_returns_empty_frame():
    params = {"hrv_window_size_beats": 40, "hrv_step_size_beats": 5}
    df = hrv.calculate_windowed_hrv(np.arange(5) * 100, 100, params)
    assert df.empty
    assert list(df.columns) == ["time", "rmssdc", "sdnn", "bpm"]
