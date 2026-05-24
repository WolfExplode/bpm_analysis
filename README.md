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
All tunable parameters for the `bpm_analysis.py` engine are located in `config.py`
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
Or use the windowless launcher:
```bash
python main.pyw
```
Tip: You can double-click `main.pyw` to launch the app without opening a command prompt.

## Build

From the project directory, with the same Python environment and dependencies you use to run the app, install PyInstaller if needed, then run:

```bash
pip install pyinstaller
pyinstaller BPM_Analyzer.spec
```

PyInstaller writes the standalone app under `dist/` (for example `dist/BPM_Analyzer.exe` on Windows).

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