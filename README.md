<p align="center">
  <a href="README.md">English</a> |
  <a href="README-JP.md">日本語</a>
</p>

# Heartbeat BPM Analyzer

This tool is a heuristic based algorithm for phonocardiogram (PCG) Analysis.
It analyzes audio recordings of heart sounds to detect heartbeats and graphs the Beats Per Minute (BPM) over time.

### **GUI Interface:**
_You only need to generate the heart rate graph but there are other options in case you need more information_

<img width="480" height="380" alt="image" src="https://github.com/user-attachments/assets/d1325e51-4c0c-4eab-bb1a-b2fcc6c17227" />

### [🔗 Outputs Heart Rate Graph:](https://youtu.be/uzc9XESJmb8)
[![Watch the video|857x482](https://github.com/user-attachments/assets/b35ccc4a-dd20-49f6-a21d-64da8c746a92)](https://youtu.be/uzc9XESJmb8)

### **Spectrogram View:**
_This script includes a spectrogram view for debugging but it is very slow to generate_
![brave_ykQQ36DQv](https://github.com/user-attachments/assets/7a10acc5-0208-455a-9a3a-0300e5a4d722)

## Configuration
All tunable parameters for the `pipeline.py` engine are located in `config.py`
The parameters are organized into logical categories for easier navigation and tuning.
- Multi-Format Audio Support: Accepts most common media files such as WAV, MP3, M4A, MOV, by converting them to .wav format for analysis.

## Dependencies
To run this script, you will need Python and the following libraries:
- **`numpy`**, **`pandas`**, **`scipy`**, **`plotly`**, **`ttkbootstrap`**, **`pydub`**
- **`librosa`** (handles audio loading and resampling)
- **`soxr`** (improves resampling quality when used with librosa)
- **`matplotlib`** (used for spectrogram and plotting)
- **`kaleido`** (required for exporting Plotly graphs to PNG)
- **`PyWavelets`** (provides the `pywt` module used for wavelet denoising)

You will also need **FFmpeg** installed and accessible in your system's PATH for `pydub` to function correctly. Follow the installation instructions for your operating system from the official [FFmpeg website](https://ffmpeg.org/download.html).

On Windows, ensure you have [Microsoft Visual C++ Redistributable Latest supported v14](https://aka.ms/vc14/vc_redist.x64.exe) (for Visual Studio 2017–2026).

## Installation

**1. Clone or download this repository, then open a terminal in the project directory.**

**2. (Recommended) Install all dependencies from the requirements file:**
```bash
pip install -r requirements.txt
```

Alternatively, install only the core dependencies manually:
```bash
pip install numpy pandas scipy plotly ttkbootstrap pydub librosa soxr matplotlib PyWavelets kaleido
```

## How to Run

From the project directory in a terminal:
```bash
python main.py
```

## Command-Line (Headless Batch)

For scripting, automation, or processing many files at once, use `batch_cli.py`. It runs the same analysis pipeline as the GUI, with no window.

```bash
# Analyze one or more files
python batch_cli.py path/to/a.wav path/to/b.mp3

# Run 4 files in parallel, write PNGs instead of HTML
python batch_cli.py --jobs 4 --png --no-html *.wav

# Custom output directory, parse starting BPM from each file name
python batch_cli.py --output-dir out --bpm-from-filename inputs/*.wav
```

Defaults come from `ui_settings.json` (the same settings the GUI writes), so the CLI behaves like your last GUI configuration unless you override a setting with a flag. At least one output type must be enabled.

See the full list with:
```bash
python batch_cli.py --help
```

**Common options**

| Flag | Description |
|------|-------------|
| `PATH ...` | One or more audio files to analyze (required). |
| `--jobs N` | Number of parallel worker processes (default 1). |
| `--output-dir DIR` | Output base directory (default `processed_files`). |
| `--output-next-to-input` / `--no-output-next-to-input` | Write outputs beside each input vs. under `--output-dir`. |
| `--bpm FLOAT` | Global starting-BPM hint for all files. |
| `--bpm-from-filename` / `--no-bpm-from-filename` | Parse starting BPM from each file name. |
| `--channel {mixed,left,right,all}` | Channel selection (`all` analyzes each stereo channel separately). |
| `--rename-input-with-bpm` / `--no-...` | After success, rename each input file with a detected-BPM tag. |
| `--quiet` | Reduce console logging to warnings/errors. |
| `--algorithm-verbose` / `--no-algorithm-verbose` | Toggle verbose per-pass algorithm logs. |

**Output toggles** (each has a `--no-` counterpart): `--html`, `--png`, `--csv`, `--summary`, `--debug`, `--filtered-wav`, `--working-wav-in-output`, `--spectrogram`, `--fft-profiles`.

**HTML extras**: `--output-all-passes`, `--html-s1-s2-hover-on`, `--html-inline-script` (each with a `--no-` counterpart).

## Build

From the project directory, with the same Python environment and dependencies you use to run the app, install PyInstaller if needed, then run:

```bash
pip install pyinstaller
pyinstaller BPM_Analyzer.spec
```

PyInstaller writes the standalone app under `dist/` (for example `dist/BPM_Analyzer.exe` on Windows).

## Testing

Two layers of tests guard the pipeline:

**Unit tests** — fast, deterministic checks on the pure helper functions (math/logic), no audio fixtures or GUI:
```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```
See [tests/README.md](tests/README.md) for the per-module coverage table.

**Benchmark runner** — validates the full end-to-end pipeline against manually labelled recordings. For each WAV with a `_manual_state_sequence.csv`, it runs analysis and compares predicted S1 segments against ground truth, reporting per-file error counts (`phase_flip`, `miss`, `extra`):
```bash
python run_benchmark.py [input_dir]   # default input_dir: inputs/Difficulty 3
```
A JSON summary is written to `benchmark_result.json`.

## Extra Features:
Import the generated heart rate graph into Blender to easily calculate the change in bpm over time.
Blender file and scripts are located in Blender BPM tool folder

<img src="https://github.com/user-attachments/assets/20130a36-d990-43ba-9cb2-c4d4d248d069" alt="Import BlenderAsj3vbrst4v" width="360" />

Select the Geometry Nodes object and enter edit mode. This will allow you to calculate:
- Heart Rate Recovery (HRR)
- maximal rate of heart rate increase

<img src="https://github.com/user-attachments/assets/f41d8e27-f525-4736-b67a-18de4e4b98e5" alt="Place BlenderAsj3zdst4v" width="360" />
<img src="https://github.com/user-attachments/assets/5d033948-f5b8-485f-9ebe-e9b87a6ee94c" alt="Adjust BlenderAsj3zny4v" width="360" />

You can also make any BPM/Time graph and export it out of blender using the `Export graph data.py` script

Import any CSV file with format: Time(Seconds), Beats Per Minute