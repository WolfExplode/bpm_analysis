// Directory Opus JScript button: open BPM Analysis GUI with selected files.
// Paste into a Script Function (JScript) per DOPUS_SCRIPTING.md. ES3 — no let/const/=>.

function quoteWinArg(s) {
    return '"' + String(s).replace(/"/g, '""') + '"';
}

function OnClick(clickData) {
    // --- Edit REPO_ROOT if your clone lives elsewhere ---
    var REPO_ROOT = "C:\\Users\\WXP\\Documents\\GitHub\\bpm_analysis";
    // Use python.exe (same folder has pythonw.exe). Hidden Run window keeps Kaleido happier than pythonw for some setups.
    var PYTHON_LAUNCHER = "C:\\Users\\WXP\\AppData\\Local\\Programs\\Python\\Python310\\python.exe";

    var paths = [];
    var tab = clickData.func.sourcetab;
    // No selection: open GUI with no pre-filled files (same as running main.py alone).
    if (tab.selstats.selfiles > 0) {
        var en = new Enumerator(tab.selected_files);
        for (; !en.atEnd(); en.moveNext()) {
            var item = en.item();
            var pathObj = item.realpath;
            pathObj.Resolve();
            paths.push(String(pathObj));
        }
    }

    var mainPy = REPO_ROOT + "\\main.py";
    var shell = new ActiveXObject("WScript.Shell");
    shell.CurrentDirectory = REPO_ROOT;

    var cmd = quoteWinArg(PYTHON_LAUNCHER) + " " + quoteWinArg(mainPy);
    var i;
    for (i = 0; i < paths.length; i++) {
        cmd += " " + quoteWinArg(paths[i]);
    }
    // 0 = hidden window (no console flash); 1 = normal. If PNG export hangs again, try 1 or switch to pythonw.exe.
    shell.Run(cmd, 0, false);
}
