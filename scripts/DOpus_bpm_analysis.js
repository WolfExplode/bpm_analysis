// Directory Opus JScript button: analyze the selected recordings headless (python -m pcg analyze),
// or open the workspace app (python -m pcg) when nothing is selected.
// Paste into a Script Function (JScript) per DOPUS_SCRIPTING.md. ES3 — no let/const/=>.

function quoteWinArg(s) {
    return '"' + String(s).replace(/"/g, '""') + '"';
}

// Must match pcg.recording.AUDIO_EXTENSIONS (audio/video inputs the analyzer accepts).
function getExtensionLower(pathStr) {
    var s = String(pathStr);
    var dot = s.lastIndexOf(".");
    if (dot < 0) {
        return "";
    }
    return s.substring(dot).toLowerCase();
}

function isSupportedMediaPath(pathStr) {
    var ext = getExtensionLower(pathStr);
    var allowed = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".mp4", ".mkv", ".mov"];
    var i;
    for (i = 0; i < allowed.length; i++) {
        if (ext === allowed[i]) {
            return true;
        }
    }
    return false;
}

function OnClick(clickData) {
    // --- Edit REPO_ROOT if your clone lives elsewhere ---
    var REPO_ROOT = "C:\\Users\\WXP\\Documents\\GitHub\\bpm_analysis";
    // The Python that has PySide6 / pyqtgraph / sounddevice installed (see requirements.txt).
    var PYTHON_LAUNCHER = "C:\\Users\\WXP\\.pyenv\\pyenv-win\\versions\\3.13.13\\python.exe";

    // Extra args for `python -m pcg analyze` (leading space if non-empty). --rename writes the
    // BPM tag into each filename; -j runs recordings in parallel (native engine only — Springer
    // and auto-switch always run one at a time because the HSMM needs several GB per minute).
    var EXTRA_ANALYZE_ARGS = " --rename -j 8";

    var paths = [];
    var tab = clickData.func.sourcetab;
    var hadFileSelection = tab.selstats.selfiles > 0;
    if (hadFileSelection) {
        var en = new Enumerator(tab.selected_files);
        for (; !en.atEnd(); en.moveNext()) {
            var item = en.item();
            var pathObj = item.realpath;
            pathObj.Resolve();
            var p = String(pathObj);
            if (isSupportedMediaPath(p)) {
                paths.push(p);
            }
        }
    }

    if (hadFileSelection && paths.length === 0) {
        var dlg = clickData.func.Dlg;
        dlg.title = "PCG Workspace";
        dlg.message = "No supported media file selected.\n\nUse: .wav .flac .mp3 .m4a .ogg .aiff .mp4 .mkv .mov";
        dlg.buttons = "OK";
        dlg.icon = "warn";
        dlg.Show();
        return;
    }

    var shell = new ActiveXObject("WScript.Shell");
    shell.CurrentDirectory = REPO_ROOT;

    var cmd;
    var windowStyle;

    if (paths.length > 0) {
        // Headless: Analyses go to the library (library/), results print to the console.
        cmd = quoteWinArg(PYTHON_LAUNCHER) + " -m pcg analyze" + EXTRA_ANALYZE_ARGS;
        var i;
        for (i = 0; i < paths.length; i++) {
            cmd += " " + quoteWinArg(paths[i]);
        }
        // 1 = show console so progress and errors are visible.
        windowStyle = 1;
    } else {
        // No selection: open the workspace app.
        cmd = quoteWinArg(PYTHON_LAUNCHER) + " -m pcg";
        // 0 = hidden window (no console flash) for the app.
        windowStyle = 0;
    }

    shell.Run(cmd, windowStyle, false);
}
