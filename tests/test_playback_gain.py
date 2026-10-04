"""Audition gain changes samples only on output, at absolute recording times."""
import numpy as np
import pytest

from pcg.app.audio import Player, SOURCE_FILTERED
from pcg.app.gain import GainEdit, gain_envelope


def render(player, frames):
    output = np.empty((frames, 1), np.float32)
    player._callback(output, frames, None, None)
    return output[:, 0]


def test_region_is_muted_with_smooth_edges_and_unaffected_neighbors():
    times = np.array([0.399, 0.4, 0.405, 0.41, 0.5, 0.59, 0.595, 0.6, 0.601])
    values = gain_envelope(times, (GainEdit(0.5, 0.2, -60),))
    np.testing.assert_allclose(values, [1, 1, 0.5, 0, 0, 0, 0.5, 1, 1], atol=1e-6)


def test_overlapping_gain_edits_multiply():
    values = gain_envelope(np.array([0.5]), (GainEdit(0.5, 0.2, -6), GainEdit(0.5, 0.1, 3)))
    np.testing.assert_allclose(values, [10 ** (-3 / 20)], rtol=1e-6)


def test_bell_center_and_half_width_follow_db_and_q():
    times = np.array([0, 0.45, 0.5, 0.55, 1])
    edit = GainEdit(0.5, 0.2, -12, q=2, shape="bell")
    values = gain_envelope(times, (edit,))
    np.testing.assert_allclose(20 * np.log10(values), [0, -6, -12, -6, 0], atol=1e-5)
    broad = gain_envelope(times, (GainEdit(0.5, 0.2, -12, shape="bell"),))
    assert broad[1] < values[1]
    assert values[2] == pytest.approx(broad[2])


def test_bell_boost_and_mute_are_symmetric_and_local():
    times = np.linspace(0, 1, 1001)
    for db in (-60, -12, 12):
        values = gain_envelope(times, (GainEdit(0.5, 0.1, db, shape="bell"),))
        np.testing.assert_allclose(values, values[::-1], atol=1e-6)
        assert values[0] == values[-1] == 1
        assert values[500] == pytest.approx(0 if db == -60 else 10 ** (db / 20))


def test_disabled_point_bypasses_only_its_own_effect():
    times = np.array([0.5])
    edits = (GainEdit(0.5, 0.2, -60, shape="bell", enabled=False),
             GainEdit(0.5, 0.2, -6, shape="bell"))
    np.testing.assert_allclose(gain_envelope(times, edits), [10 ** (-6 / 20)], rtol=1e-6)


@pytest.mark.parametrize("source", ["original", SOURCE_FILTERED])
def test_playback_applies_gain_without_mutating_sources(source):
    player = Player()
    original = np.ones(1000, np.float32)
    filtered = np.full(1000, 0.5, np.float32)
    player.set_audio(original, 1000)
    player.set_filtered(filtered)
    player.source = source
    before = player.samples(source).copy()
    player.set_gain_edits((GainEdit(0.5, 0.2, -6),))
    player.seek(0.45)
    output = render(player, 50)
    np.testing.assert_allclose(output, before[450:500] * 10 ** (-6 / 20), rtol=1e-6)
    np.testing.assert_array_equal(player.samples(source), before)
    np.testing.assert_array_equal(original, np.ones(1000))
    np.testing.assert_array_equal(filtered, np.full(1000, 0.5))


def test_loop_wrap_uses_recording_time_and_boost_is_bounded():
    player = Player()
    player.set_audio(np.ones(1000, np.float32), 1000)
    player.set_gain_edits((GainEdit(0.5, 0.2, -60),))
    player.set_loop((0.45, 0.55))
    np.testing.assert_array_equal(render(player, 250), np.zeros(250))
    assert player.position == pytest.approx(0.5)
    player.set_gain_edits((GainEdit(0.5, 0.2, 12),))
    boosted = render(player, 50)
    assert np.max(boosted) <= 1
    assert boosted[-1] == 1


def test_live_bypass_crossfades_and_clear_resets_recording():
    player = Player()
    player.set_audio(np.ones(1000, np.float32), 1000)
    edits = (GainEdit(0.5, 0.4, -60),)
    player.set_gain_edits(edits)
    player.seek(0.4)
    np.testing.assert_array_equal(render(player, 20), np.zeros(20))
    player.set_gain_edits(edits, enabled=False)
    values = render(player, 20)
    assert values[0] == 0
    assert np.all(np.diff(values) >= 0)
    np.testing.assert_array_equal(values[5:], np.ones(15))
    player.set_loop((0.4, 0.6))
    player.clear()
    player.set_audio(np.ones(1000, np.float32), 1000)
    np.testing.assert_array_equal(render(player, 20), np.ones(20))
    assert player._loop is None


@pytest.mark.parametrize("edit", [GainEdit(1, 0, 0), GainEdit(1, 0.2, 13), GainEdit(float("nan"), 0.2, 0),
                                  GainEdit(1, 0.2, 0, q=0), GainEdit(1, 0.2, 0, shape="unknown")])
def test_invalid_gain_edit_rejected(edit):
    with pytest.raises(ValueError):
        Player().set_gain_edits((edit,))
