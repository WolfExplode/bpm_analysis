# audio_io.py
# Audio file preparation around the engine: format conversion (pydub/FFmpeg),
# stereo channel selection/splitting, and writing debug WAVs for HTML playback.
# Not part of the analysis engine: the engine (engine.py) takes a WAV path and
# never writes files. Consumed by batch_runner, gui, batch_cli and pipeline.
import logging
import os
from typing import List

import librosa
import numpy as np
from scipy.io import wavfile

from file_io import output_stem_from_path

try:
    from pydub import AudioSegment
except ImportError:
    logging.warning("Pydub library not found. Install with 'pip install pydub'.")
    AudioSegment = None

CHANNEL_MODE_MIXED = "mixed"
CHANNEL_MODE_LEFT = "left"
CHANNEL_MODE_RIGHT = "right"
CHANNEL_MODE_ALL = "all"
CHANNEL_MODES = (CHANNEL_MODE_MIXED, CHANNEL_MODE_LEFT, CHANNEL_MODE_RIGHT, CHANNEL_MODE_ALL)
MAX_STEREO_CHANNEL_COUNT = 2


class TooManyAudioChannelsError(ValueError):
    """Raised when channel-specific processing is requested on >2-channel audio."""


def normalize_channel_mode(mode: str) -> str:
    m = (mode or CHANNEL_MODE_MIXED).strip().lower()
    if m not in CHANNEL_MODES:
        raise ValueError(f"Invalid channel_mode {mode!r}; expected one of {CHANNEL_MODES}.")
    return m


def _export_mono_channel(
    segment,
    file_path: str,
    output_directory: str,
    channel_index: int,
) -> str:
    """Export one 0-based mono channel; filename uses 1-based _chN suffix."""
    ch_num = channel_index + 1
    base_name = output_stem_from_path(file_path)
    out_path = os.path.join(output_directory, f"{base_name}_ch{ch_num}.wav")
    segment.export(out_path, format="wav")
    return out_path


def write_peak_normalized_debug_wav(
    out_path: str,
    signal: np.ndarray,
    orig_sr: int,
    target_sr: int = 10000,
) -> None:
    """Peak-normalize, resample for HTML playback, write int16 mono WAV."""
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    norm = signal / peak if peak > 0 else signal
    debug_audio = librosa.resample(norm, orig_sr=orig_sr, target_sr=target_sr)
    normalized_audio = np.int16(np.clip(debug_audio, -1.0, 1.0) * 32767)
    wavfile.write(out_path, target_sr, normalized_audio)


def write_peak_normalized_wav_native_rate(out_path: str, signal: np.ndarray, sr: int) -> None:
    """Peak-normalize and write int16 mono WAV at the given sample rate (no resampling)."""
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    norm = signal / peak if peak > 0 else signal
    normalized_audio = np.int16(np.clip(norm, -1.0, 1.0) * 32767)
    wavfile.write(out_path, int(sr), normalized_audio)


def convert_to_wav(file_path: str, target_path: str) -> bool:
    """Converts a given audio file to WAV format."""
    if not AudioSegment:
        raise ImportError("Pydub/FFmpeg is required for audio conversion.")

    logging.info("Converting %s to WAV format...", os.path.basename(file_path))
    try:
        sound = AudioSegment.from_file(file_path)
        # Preserve original channel layout; downstream logic may choose to split channels.
        sound.export(target_path, format="wav")
        return True
    except Exception as e:
        logging.error("Could not convert file %s. Error: %s", file_path, e)
        return False


def resolve_wav_for_channel_mode(
    file_path: str,
    output_directory: str,
    channel_mode: str = CHANNEL_MODE_MIXED,
) -> List[str]:
    """
    Select working WAV path(s) for analysis based on channel_mode.

    mixed — use file as-is (librosa downmixes at load time).
    left / right — export one stereo channel (ch1 = left, ch2 = right).
    all — export each stereo channel as separate mono WAVs.

    Mono inputs are returned unchanged for left/right/all. More than two channels
    raises TooManyAudioChannelsError.
    """
    mode = normalize_channel_mode(channel_mode)
    if mode == CHANNEL_MODE_MIXED:
        return [file_path]

    if not AudioSegment:
        logging.warning("Pydub not available; cannot select channels. Using original file only.")
        return [file_path]

    try:
        sound = AudioSegment.from_file(file_path)
    except Exception as e:
        logging.warning("Failed to open WAV for channel selection (%s): %s", file_path, e)
        return [file_path]

    n_channels = sound.channels
    if n_channels <= 1:
        return [file_path]

    if n_channels > MAX_STEREO_CHANNEL_COUNT:
        raise TooManyAudioChannelsError(
            f"'{os.path.basename(file_path)}' has {n_channels} audio channels; "
            "left/right/separate-channel analysis supports stereo (2 channels) only."
        )

    mono_segments = sound.split_to_mono()
    if mode == CHANNEL_MODE_ALL:
        channel_paths: List[str] = []
        for idx, seg in enumerate(mono_segments):
            try:
                channel_paths.append(_export_mono_channel(seg, file_path, output_directory, idx))
            except Exception as e:
                logging.warning("Failed to export channel %d for %s: %s", idx + 1, file_path, e)
        if not channel_paths:
            return [file_path]
        logging.info(
            "Split %s into %d mono channel file(s): %s",
            os.path.basename(file_path),
            len(channel_paths),
            ", ".join(os.path.basename(p) for p in channel_paths),
        )
        return channel_paths

    channel_index = 0 if mode == CHANNEL_MODE_LEFT else 1
    side_label = "left" if mode == CHANNEL_MODE_LEFT else "right"
    try:
        out_path = _export_mono_channel(
            mono_segments[channel_index], file_path, output_directory, channel_index
        )
    except Exception as e:
        logging.warning("Failed to export %s channel for %s: %s", side_label, file_path, e)
        return [file_path]

    logging.info(
        "Using %s channel only from %s: %s",
        side_label,
        os.path.basename(file_path),
        os.path.basename(out_path),
    )
    return [out_path]
