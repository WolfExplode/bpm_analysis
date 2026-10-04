"""Unfiltered display envelopes retain amplitude, alignment and chunk continuity."""
import numpy as np
import pytest
from scipy.ndimage import uniform_filter1d

from pcg.app.audio import original_audio_envelope


def test_original_envelope_preserves_unfiltered_amplitude_and_input():
    sr = 16000
    t = np.arange(sr) / sr
    signal = (0.2 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    before = signal.copy()
    dt, envelope = original_audio_envelope(signal, sr)
    assert dt == 0.005
    assert len(envelope) == 200
    np.testing.assert_allclose(envelope, 0.2, atol=1e-5)
    np.testing.assert_array_equal(signal, before)


def test_original_envelope_stays_aligned_across_chunk_boundaries():
    sr = 2000
    t = np.arange(sr * 61) / sr
    amplitude = 0.4 + 0.1 * np.sin(2 * np.pi * 2 * t)
    signal = (amplitude * np.sin(2 * np.pi * 240 * t)).astype(np.float32)
    dt, envelope = original_audio_envelope(signal, sr)
    expected = uniform_filter1d(amplitude, size=100, mode="nearest")[::10]
    assert len(envelope) == len(expected)
    assert dt == 0.005
    np.testing.assert_allclose(envelope, expected, atol=1e-4)


def test_empty_and_tiny_original_audio_envelopes():
    dt, envelope = original_audio_envelope(np.zeros(0, np.float32), 16000)
    assert dt == 0.005 and len(envelope) == 0
    _, envelope = original_audio_envelope(np.array([0.3], np.float32), 16000)
    np.testing.assert_allclose(envelope, [0.3])


@pytest.mark.parametrize("sr,smoothing", [(0, 50), (16000, -1), (16000, float("nan"))])
def test_invalid_original_envelope_parameters(sr, smoothing):
    with pytest.raises(ValueError):
        original_audio_envelope(np.zeros(10, np.float32), sr, smoothing)
