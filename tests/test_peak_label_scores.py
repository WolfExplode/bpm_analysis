"""Label-score helpers (path-dependent heuristic masses)."""
import peak_label_scores as pls


def test_clip01_bounds():
    assert pls._clip01(-1.0) == 0.0
    assert pls._clip01(2.0) == 1.0
    assert pls._clip01(0.3) == 0.3


def test_paired_s1_masses():
    d = pls.label_scores_paired_s1(0.8)
    assert d[pls.LABEL_SCORE_S1] == 0.8
    assert d[pls.LABEL_SCORE_S2] == 0.0
    assert abs(d[pls.LABEL_SCORE_NOISE] - 0.2) < 1e-12


def test_paired_s1_clips_out_of_range_confidence():
    d = pls.label_scores_paired_s1(1.5)
    assert d[pls.LABEL_SCORE_S1] == 1.0
    assert d[pls.LABEL_SCORE_NOISE] == 0.0


def test_paired_s2_masses():
    d = pls.label_scores_paired_s2(0.6)
    assert d[pls.LABEL_SCORE_S2] == 0.6
    assert d[pls.LABEL_SCORE_S1] == 0.0


def test_noise_rejected_lone_adds_s2_hint_when_pairing_weak():
    # Weak pairing (pc=0) -> s2 = 0.25 * (1-0) = 0.25
    d = pls.label_scores_noise_rejected_lone(0.1, pairing_failure_confidence=0.0)
    assert abs(d[pls.LABEL_SCORE_S2] - 0.25) < 1e-12
    # No pairing info -> no s2 mass
    d2 = pls.label_scores_noise_rejected_lone(0.1)
    assert d2[pls.LABEL_SCORE_S2] == 0.0


def test_structural_and_last_peak_constants_sum_to_one():
    for d in (pls.label_scores_noise_middle_structural(), pls.label_scores_lone_s1_last_peak()):
        total = d[pls.LABEL_SCORE_S1] + d[pls.LABEL_SCORE_S2] + d[pls.LABEL_SCORE_NOISE]
        assert abs(total - 1.0) < 1e-12


def test_extract_pairing_final_confidence_prefers_final_step():
    steps = [
        {"step": "Base", "result": 0.4},
        {"step": "Final", "result": 0.73},
        {"step": "Extra", "result": 0.1},
    ]
    assert pls.extract_pairing_final_confidence_from_steps(steps) == 0.73


def test_extract_pairing_final_confidence_falls_back_to_last():
    steps = [{"step": "Base", "result": 0.4}, {"step": "Adjust", "result": 0.55}]
    assert pls.extract_pairing_final_confidence_from_steps(steps) == 0.55


def test_extract_pairing_final_confidence_handles_empty_and_bad():
    assert pls.extract_pairing_final_confidence_from_steps([]) == 0.0
    assert pls.extract_pairing_final_confidence_from_steps([{"step": "x", "result": None}]) == 0.0
