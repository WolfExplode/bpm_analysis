// Directory Opus JScript button: run headless batch on selected files (batch_cli.py),
// or open the GUI (main.py) when nothing is selected so you can pick files and options.
// Paste into a Script Function (JScript) per DOPUS_SCRIPTING.md. ES3 — no let/const/=>.

function quoteWinArg(s) {
    return '"' + String(s).replace(/"/g, '""') + '"';
}

function OnClick(clickData) {
    // --- Edit REPO_ROOT if your clone lives elsewhere ---
    var REPO_ROOT = "C:\\Users\\WXP\\Documents\\GitHub\\bpm_analysis";
    // Use python.exe (same folder has pythonw.exe). Hidden Run window keeps Kaleido happier than pythonw for some setups.
    var PYTHON_LAUNCHER = "C:\\Users\\WXP\\AppData\\Local\\Programs\\Python\\Python310\\python.exe";

    // Optional extra args for batch_cli.py only (e.g. " --jobs 4" or " --png --no-html"). Leading space if non-empty.
    var EXTRA_BATCH_CLI_ARGS = "";

    var paths = [];
    var tab = clickData.func.sourcetab;
    if (tab.selstats.selfiles > 0) {
        var en = new Enumerator(tab.selected_files);
        for (; !en.atEnd(); en.moveNext()) {
            var item = en.item();
            var pathObj = item.realpath;
            pathObj.Resolve();
            paths.push(String(pathObj));
        }
    }

    var shell = new ActiveXObject("WScript.Shell");
    shell.CurrentDirectory = REPO_ROOT;

    var cmd;
    var windowStyle;

    if (paths.length > 0) {
        // Headless batch via CLI (outputs under processed_files by default).
        var batchCli = REPO_ROOT + "\\batch_cli.py";
        cmd = quoteWinArg(PYTHON_LAUNCHER) + " " + quoteWinArg(batchCli) + EXTRA_BATCH_CLI_ARGS;
        var i;
        for (i = 0; i < paths.length; i++) {
            cmd += " " + quoteWinArg(paths[i]);
        }
        // 1 = show console so batch progress and errors are visible.
        windowStyle = 1;
    } else {
        // No selection: open GUI to choose files and output options.
        var mainPy = REPO_ROOT + "\\main.py";
        cmd = quoteWinArg(PYTHON_LAUNCHER) + " " + quoteWinArg(mainPy);
        // 0 = hidden window (no console flash) for GUI launch.
        windowStyle = 0;
    }

    shell.Run(cmd, windowStyle, false);
}
