import os
import ttkbootstrap as ttkb
from gui import BPMApp
import logging
import sys

def _cli_initial_files(argv_paths):
    """Return existing file paths from command-line args (normalized)."""
    out = []
    for raw in argv_paths:
        p = os.path.normpath(raw)
        if os.path.isfile(p):
            out.append(p)
    return out

def main():
    """
    Initializes and runs the BPM Analysis GUI.
    This is the main entry point for the application.
    Optional args: paths to audio files (e.g. from Directory Opus) pre-fill the file list.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        stream=sys.stdout
    )

    # Project root as cwd so ui_settings.json, processed_files, and relative paths behave consistently.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    initial_files = _cli_initial_files(sys.argv[1:])

    root = ttkb.Window(themename="minty")
    app = BPMApp(root, initial_files=initial_files)
    root.mainloop()

if __name__ == "__main__":
    main()
