"""Range reruns: audio context, clock alignment, channel isolation and safe edits."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pcg import annotation as an, recording, reanalysis
from pcg.cli import main


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / "stereo.wav"
    t = np.arange(5000) / 100.0
    sf.write(path, np.column_stack((t / 50, -t / 50)), 100, subtype="FLOAT")
    return path


@pytest.mark.parametrize("algorithm", ["springer", "native"])
def test_range_context_channel_offset_and_hint(audio_path, monkeypatch, algorithm):
    captured = {}

    def engine(wav, params, **kwargs):
        samples, sr = sf.read(wav)
        captured.update(wav=wav, params=params, kwargs=kwargs)
        assert sr == 100
        # Selected range is 20-22; processed range is 5-37, on the right channel.
        assert len(samples) == 3200
        assert samples[0] == pytest.approx(-0.1)
        return SimpleNamespace(sample_rate=100, algorithm_used=algorithm,
                               analysis_data={"pass3_state_boundaries": [
                                   (1400, 1520, "S1", {}),  # crosses selection start
                                   (1520, 1600, "systole", {}),
                                   (1600, 1650, "S2", {}),
                                   (1680, 1750, "S1", {}),  # crosses selection end
                                   (1800, 1900, "S2", {}),  # context only
                               ]}, bpm_failure_report={"reasons": ["example warning"]})

    monkeypatch.setattr(reanalysis, "run_analysis", engine)
    result = reanalysis.run_range(audio_path, 20, 22, algorithm=algorithm, channel="right", bpm_hint=90)
    assert (result.context_start, result.context_end) == (5, 37)
    assert [(s.kind, s.start, s.end) for s in result.spans] == [
        ("S1", 20, 20.2), ("S2", 21, 21.5), ("S1", 21.8, 22)]
    assert result.spans[0].clipped and result.spans[-1].clipped
    assert all(s.origin == an.ORIGIN_ALGORITHM for s in result.spans)
    assert captured["params"]["analysis_start_sec"] == 0
    assert captured["params"]["auto_switch_algorithm"] is False
    assert captured["params"]["use_springer_algorithm"] == (algorithm == "springer")
    assert captured["kwargs"]["start_bpm_hint"] == 90
    curve = captured["kwargs"]["heart_rate_curve"]
    if algorithm == "springer":
        np.testing.assert_array_equal(curve[0], [0, 32])
        np.testing.assert_array_equal(curve[1], [90, 90])
    else:
        assert curve is None
    assert not Path(captured["wav"]).exists()
    assert reanalysis.RangeResult.from_dict(result.to_dict()) == result


def test_recording_edges_and_sample_alignment(audio_path):
    audio, offset = recording.load_range(audio_path, 0.017, 0.045)
    assert offset == 0.01
    assert len(audio.samples) == 4  # floor start, ceil stop
    tail, offset = recording.load_range(audio_path, 49, 65)
    assert offset == 49 and tail.duration_sec == 1


def test_range_rejects_empty_detection_and_out_of_recording(audio_path, monkeypatch):
    monkeypatch.setattr(reanalysis, "run_analysis", lambda *args, **kwargs: SimpleNamespace(
        sample_rate=100, analysis_data={}, algorithm_used="native", bpm_failure_report={}))
    with pytest.raises(ValueError, match="No S1/S2"):
        reanalysis.run_range(audio_path, 20, 22, algorithm="native")
    with pytest.raises(ValueError, match="beyond"):
        reanalysis.run_range(audio_path, 49, 51)
    with pytest.raises(ValueError, match="outside"):
        reanalysis.run_range(audio_path, 70, 71)


@pytest.mark.parametrize("start,end,kwargs", [
    (-1, 2, {}), (1, 1, {}), (2, 1, {}), (float("nan"), 2, {}),
    (1, float("inf"), {}), (1, 2, {"context_sec": -1}),
    (1, 2, {"channel": "all"}), (1, 2, {"algorithm": "unknown"}),
    (1, 2, {"bpm_hint": 0}), (1, 2, {"bpm_hint": float("nan")}),
])
def test_invalid_requests_fail_before_decoding(start, end, kwargs):
    with pytest.raises(ValueError):
        reanalysis.run_range("nonexistent.wav", start, end, **kwargs)


def test_replace_selection_preserves_crossing_sounds_and_noisy_tails():
    original = (an.Span("S1", 0.9, 1.1), an.Span("noisy", 1.2, 2.1), an.Span("S2", 3, 3.1))
    incoming = (an.Span("S2", 0.95, 1.15, an.ORIGIN_ALGORITHM),
                an.Span("S1", 1.8, 2.05, an.ORIGIN_ALGORITHM))
    result = an.replace_selection(original, 1, 2, incoming)
    assert [(s.kind, s.start, s.end) for s in result] == [
        ("S1", .9, 1), ("S2", 1, 1.15), ("S1", 1.8, 2), ("noisy", 2, 2.1), ("S2", 3, 3.1)]
    an.validate(result)
    assert result[-1] is original[-1]


def test_cli_returns_result_without_writing_annotation(audio_path, monkeypatch, capsys):
    result = reanalysis.RangeResult(20, 22, 5, 37, "native", (an.Span("S1", 20, 20.1),))
    calls = []
    monkeypatch.setattr(reanalysis, "run_range", lambda *args, **kwargs: calls.append((args, kwargs)) or result)
    assert main(["reanalyze-range", str(audio_path), "--from", "20", "--to", "22",
                 "--algorithm", "native", "--progress-json"]) == 0
    import json
    payload = json.loads(capsys.readouterr().out)
    assert reanalysis.RangeResult.from_dict(payload["range_result"]) == result
    assert calls[0][1]["algorithm"] == "native"
    assert list(audio_path.parent.iterdir()) == [audio_path]


def test_selection_preserves_tiny_boundary_tail():
    original = (an.Span("S1", .9995, 1.1),)
    result = an.replace_selection(original, 1, 2, (an.Span("S2", 1, 1.1),))
    assert result[0].start == .9995 and result[0].end == 1


def test_cli_reports_failure(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("no sounds")

    monkeypatch.setattr(reanalysis, "run_range", fail)
    assert main(["reanalyze-range", "missing.wav", "--from", "1", "--to", "2", "--progress-json"]) == 1
    import json
    assert json.loads(capsys.readouterr().out) == {"range_result": None, "error": "no sounds"}


@pytest.mark.parametrize("algorithm", ["springer", "native"])
def test_real_pipeline_on_synthetic_pcg(tmp_path, algorithm):
    # Alternating lower-frequency S1 and higher-frequency S2 at 80 BPM.
    sr = 2000
    times = np.arange(20 * sr) / sr
    signal = np.zeros_like(times)
    for s1 in np.arange(.5, 19.5, .75):
        for center, frequency, amplitude in ((s1, 65, 1), (s1 + .25, 110, .65)):
            relative = times - center
            signal += amplitude * np.exp(-(relative / .025) ** 2) * np.sin(2 * np.pi * frequency * relative)
    path = tmp_path / "pcg.wav"
    sf.write(path, signal, sr, subtype="FLOAT")
    result = reanalysis.run_range(path, 6, 8, algorithm=algorithm, context_sec=5, bpm_hint=80)
    assert result.algorithm == algorithm
    assert (result.start, result.end, result.context_start, result.context_end) == (6, 8, 1, 13)
    assert len(result.spans) >= 2
    assert any(s.kind == "S1" for s in result.spans)
    assert all(6 <= s.start < s.end <= 8 for s in result.spans)
    an.validate(result.spans)
    assert list(tmp_path.iterdir()) == [path]
