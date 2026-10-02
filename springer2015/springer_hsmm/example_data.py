"""
Load example_data.mat (PCG recordings and R-peak / end-T annotations).
"""

from typing import Any

import numpy as np
from scipy.io import loadmat


def _extract_audio_list(raw_audio: Any) -> list[np.ndarray]:
    audio_list: list[np.ndarray] = []
    if isinstance(raw_audio, np.ndarray):
        raw_flat = raw_audio.flatten()
        for i in range(raw_flat.size):
            elem = raw_flat.flat[i]
            arr = np.asarray(elem).flatten()
            audio_list.append(arr)
        if not audio_list and raw_audio.size > 0:
            audio_list.append(np.asarray(raw_audio).flatten())
    else:
        audio_list.append(np.asarray(raw_audio).flatten())
    return audio_list if audio_list else [np.array([], dtype=np.float64)]


def _extract_annotations_list(raw_annot: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    annotations_list: list[tuple[np.ndarray, np.ndarray]] = []
    if isinstance(raw_annot, np.ndarray):
        if raw_annot.ndim == 2 and raw_annot.shape[1] >= 2:
            for i in range(raw_annot.shape[0]):
                s1 = np.asarray(raw_annot[i, 0]).flatten()
                s2 = np.asarray(raw_annot[i, 1]).flatten()
                annotations_list.append((s1, s2))
        else:
            raw_flat = raw_annot.flatten()
            for i in range(raw_flat.size):
                row = raw_flat.flat[i]
                try:
                    r = np.asarray(row)
                    if r.ndim >= 1 and r.shape[0] >= 2:
                        s1 = np.asarray(r.flat[0]).flatten()
                        s2 = np.asarray(r.flat[1]).flatten()
                    else:
                        s1 = np.asarray(row).flatten()
                        s2 = np.array([])
                except Exception:
                    s1 = np.asarray(row).flatten()
                    s2 = np.array([])
                annotations_list.append((s1, s2))
        if not annotations_list:
            annotations_list.append((np.array([]), np.array([])))
    else:
        annotations_list.append((np.array([]), np.array([])))
    return annotations_list


def load_example_data(mat_path: str) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]], float]:
    """
    Load example_data.mat.

    Returns
    -------
    audio_list : list of 1D arrays (one per recording)
    annotations_list : list of (s1_positions, s2_positions) in samples at audio Fs
    fs : float, sampling frequency (e.g. 1000)
    """
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)

    if "example_data" in data:
        ed = data["example_data"]
        st = ed
        if hasattr(ed, "flat") and ed.size >= 1:
            st = ed.flat[0]
        if hasattr(st, "example_audio_data"):
            raw_audio = st.example_audio_data
            raw_annot = getattr(st, "example_annotations", np.array([]))
        else:
            raw_audio = data.get("example_audio_data", data.get("audio_data"))
            raw_annot = data.get("example_annotations", data.get("annotations", np.array([])))
    elif "example_audio_data" in data:
        raw_audio = data["example_audio_data"]
        raw_annot = data.get("example_annotations", data.get("annotations", np.array([])))
    elif "audio_data" in data:
        raw_audio = data["audio_data"]
        raw_annot = data.get("annotations", np.array([]))
    else:
        raise KeyError(f"No example_data / example_audio_data / audio_data in {mat_path}")

    if raw_annot is None or (isinstance(raw_annot, np.ndarray) and raw_annot.size == 0):
        raw_annot = np.array([])

    audio_list = _extract_audio_list(raw_audio)
    annotations_list = _extract_annotations_list(raw_annot)
    while len(annotations_list) < len(audio_list):
        annotations_list.append((np.array([]), np.array([])))
    annotations_list = annotations_list[: len(audio_list)]

    fs = 1000.0
    if "Fs" in data:
        f = data["Fs"]
        if isinstance(f, np.ndarray) and f.size > 0:
            fs = float(f.flat[0])
        else:
            fs = float(f)
    elif "fs" in data:
        f = data["fs"]
        if isinstance(f, np.ndarray) and f.size > 0:
            fs = float(f.flat[0])
        else:
            fs = float(f)
    return audio_list, annotations_list, fs
