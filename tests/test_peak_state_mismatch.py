"""
Tests for the peak/state agreement detector.

An S1 peak should sit under an S1 state band, an S2 peak under an S2 band. When an
S1 peak sits under an S2 band (or vice-versa) the cycle's labels are swapped — the
"marker says S1 but the strip shows S2" bug behind the missing-S1 sequence break.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_helpers.peak_state_mismatch_detector import (  # noqa: E402
    find_peak_state_mismatches,
    summarize,
)


def test_agreement_no_mismatch():
    pc = {100: {"peak_type": "S1 (Paired)"}, 300: {"peak_type": "S2 (Paired)"}}
    bounds = [(80, 120, "S1", {}), (120, 280, "systole", {}),
              (280, 320, "S2", {}), (320, 500, "diastole", {})]
    assert find_peak_state_mismatches(pc, bounds) == []


def test_swapped_cycle_flagged():
    # S1 peak under an S2 band and S2 peak under an S1 band -> swapped labels.
    pc = {100: {"peak_type": "S1 (Paired)"}, 300: {"peak_type": "S2 (Paired)"}}
    bounds = [(80, 120, "S2", {}), (120, 280, "diastole", {}),
              (280, 320, "S1", {}), (320, 500, "systole", {})]
    recs = find_peak_state_mismatches(pc, bounds, sample_rate=600)
    assert len(recs) == 2
    s = summarize(recs)
    assert s["s1_peak_under_s2_band"] == 1
    assert s["s2_peak_under_s1_band"] == 1
    assert recs[0]["peak"] == 100 and recs[0]["state"] == "S2"
    assert recs[0]["peak_sec"] == 100 / 600


def test_noise_peak_not_flagged():
    # A Noise peak may fall in any state; never a mismatch.
    pc = {100: {"peak_type": "Noise/Rejected"}}
    bounds = [(80, 120, "S2", {})]
    assert find_peak_state_mismatches(pc, bounds) == []


def test_peak_in_systole_not_flagged():
    # Only S1/S2 bands count; a peak over systole/diastole is not a mismatch.
    pc = {200: {"peak_type": "S1 (Paired)"}}
    bounds = [(120, 280, "systole", {})]
    assert find_peak_state_mismatches(pc, bounds) == []
