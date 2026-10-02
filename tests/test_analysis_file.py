"""Analysis file format, staleness, library, traces and Recording fingerprints (no engine run)."""
import numpy as np
import pytest
import soundfile as sf

from pcg import analysis as A
from pcg import recording
from pcg.engine.defects import Defect
from pcg.engine.traces import KIND_LINE, KIND_POINTS, KIND_SPANS, Trace, emit_trace, describe


def _analysis(**over):
    traces = [
        Trace("Env", "Preprocessing", "signal", KIND_LINE, x=[], y=np.arange(10.0), dt=0.5, visible=True),
        Trace("Pts", "Pass 2", "bpm", KIND_POINTS, x=[1.0, 2.0], y=[60.0, 61.0], text=["a", "b"]),
        Trace("Win", "Pass 3", "signal", KIND_SPANS, x=[1.0], x_end=[2.0]),
    ]
    a = A.Analysis(
        fingerprint="f" * 32, channel="mixed", recording_name="r.wav", created="2026-10-01T00:00:00",
        params=A._jsonable(A.effective_params({})), run_settings={}, start_bpm_hint=None,
        code_fingerprint=A.code_fingerprint(), algorithm_used="native", algorithm_switch_reason=None,
        sample_rate=600, duration_sec=5.0, time_offset_sec=0.0, summary={"bpm": {"min_bpm": 60.0}},
        peaks=A.Peaks(time=np.array([1.0, 1.3]), amp=np.array([2.0, 1.0]), label=["S1 (Paired)", "S2 (Paired)"],
                      state=["S1", "S2"], final_s1=np.array([True, False]), s1_score=np.array([0.9, np.nan]),
                      s2_score=np.array([0.1, np.nan]), noise_score=np.array([0.0, np.nan]),
                      reasoning=[["why → S1"], []]),
        states=A.States(start=np.array([0.95]), end=np.array([1.05]), state=["S1"], source=[""], reasoning=[["r"]]),
        traces=traces, defects=[Defect("overlap", 1.0, 1.1, "x")],
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_round_trip(tmp_path):
    a = _analysis()
    p = tmp_path / "x.analysis.zip"
    A.save(a, p)
    b = A.load(p)
    assert b.summary == a.summary and b.params == a.params and b.defects == a.defects
    assert b.peaks.label == a.peaks.label and b.peaks.reasoning == a.peaks.reasoning
    np.testing.assert_array_equal(b.peaks.final_s1, a.peaks.final_s1)
    np.testing.assert_array_equal(b.states.start, a.states.start)
    env = b.trace("Env")
    assert env.uniform and env.dt == 0.5 and env.visible
    np.testing.assert_allclose(env.times()[-1], 4.5)
    assert b.trace("Pts").text == ["a", "b"]
    np.testing.assert_array_equal(b.trace("Win").x_end, [2.0])


def test_staleness():
    a = _analysis()
    assert A.stale_reasons(a) == []
    cfg = A.current_config_params()
    cfg["min_s1_s2_interval_sec"] = 123.0
    assert A.stale_reasons(a, config_params=cfg) == ["parameters changed: min_s1_s2_interval_sec"]
    assert A.stale_reasons(a, code_fp="different") == ["engine code changed"]
    # Run settings are part of the Analysis, not a reason for staleness.
    b = _analysis(run_settings={"analysis_start_sec": 3.0}, params=A._jsonable(A.effective_params({"analysis_start_sec": 3.0})))
    assert A.stale_reasons(b) == []


def test_library_paths_and_latest(tmp_path):
    lib = A.Library(tmp_path)
    a = _analysis()
    p = lib.save(a)
    assert p.name == "f" * 32 + ".analysis.zip"
    assert lib.latest("f" * 32) == p
    assert lib.path_for("f" * 32, "left").name.endswith("-left.analysis.zip")
    assert not lib.open(p).is_stale


def test_trace_shift_window_value():
    t = Trace("u", "g", "signal", KIND_LINE, x=[], y=[0.0, 1.0, 2.0], dt=1.0)
    s = t.shifted(10.0)
    assert s.t0 == 10.0 and s.value_at(11.5) == pytest.approx(1.5)
    assert list(s.window(10.6, 11.2)) == [0, 1, 2]
    p = Trace("p", "g", "bpm", KIND_POINTS, x=[1.0, 2.0, 3.0], y=[1, 2, 3]).shifted(1.0)
    assert list(p.window(2.5, 3.5)) == [1]


def test_trace_validation():
    with pytest.raises(ValueError):
        Trace("bad", "g", "bpm", KIND_POINTS, x=[1.0, 2.0], y=[1.0])
    with pytest.raises(ValueError):
        Trace("bad", "g", "bpm", "weird", x=[], y=[])


def test_emit_trace_and_describe():
    ad = {}
    emit_trace(ad, "Mine", group="Pass 3", lane="bpm", x=[1.0], y=[2.0])
    emit_trace(ad, "Windows", group="Pass 3", lane="signal", x=[1.0], x_end=[2.0])
    assert [t.kind for t in ad["debug_traces"]] == [KIND_LINE, KIND_SPANS]
    assert describe({"a": 1.23456, "b": {"c": [1, 2]}}) == ["a: 1.235", "b:", "  c:", "    1, 2"]


def _write(path, data, sr=4000, **kw):
    sf.write(str(path), data, sr, **kw)


def test_fingerprint_survives_rename_and_container_not_edits(tmp_path):
    rng = np.random.default_rng(0)
    data = (rng.standard_normal((8000, 2)) * 3000).astype(np.int16)
    _write(tmp_path / "a.wav", data, subtype="PCM_16")
    _write(tmp_path / "a.flac", data, subtype="PCM_16")
    fp = recording.fingerprint(tmp_path / "a.wav")
    (tmp_path / "a.wav").rename(tmp_path / "a [80,70-90bpm].wav")
    assert recording.fingerprint(tmp_path / "a [80,70-90bpm].wav") == fp
    assert recording.fingerprint(tmp_path / "a.flac") == fp
    _write(tmp_path / "trim.wav", data[10:], subtype="PCM_16")
    assert recording.fingerprint(tmp_path / "trim.wav") != fp


def test_fingerprint_cache(tmp_path):
    _write(tmp_path / "a.wav", np.zeros(100, dtype=np.int16), subtype="PCM_16")
    cache = recording.FingerprintCache(tmp_path / "fp.json")
    fp = cache.get(tmp_path / "a.wav")
    assert recording.FingerprintCache(tmp_path / "fp.json").get(tmp_path / "a.wav") == fp


def test_engine_inputs_channels(tmp_path):
    data = np.stack([np.ones(100), -np.ones(100)], axis=1).astype(np.float32)
    _write(tmp_path / "s.wav", data, subtype="FLOAT")
    assert recording.engine_inputs(tmp_path / "s.wav", "mixed", tmp_path) == [("mixed", str(tmp_path / "s.wav"))]
    [(ch, p)] = recording.engine_inputs(tmp_path / "s.wav", "right", tmp_path)
    assert ch == "right" and np.allclose(sf.read(p)[0], -1)
    assert [c for c, _ in recording.engine_inputs(tmp_path / "s.wav", "all", tmp_path)] == ["left", "right"]


def test_is_inside(tmp_path):
    (tmp_path / "inputs" / "sub").mkdir(parents=True)
    assert recording.is_inside(tmp_path / "inputs" / "sub" / "x.wav", [tmp_path / "inputs"])
    assert not recording.is_inside(tmp_path / "inputs2" / "x.wav", [tmp_path / "inputs"])
