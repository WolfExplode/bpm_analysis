// PCG Workspace — launches the bpm_analysis app (python -m pcg app) from this repo.
//
// Click: open the app with the selection and start analysing right away. One recording opens
// in the Workspace tab; several files (or folders) go to the Run queue and start.
// No selection: just open the app.
//
// Paste into a Script Function (JScript) per DOpus_SCRIPTING.md. ES3 — no let/const/=>.

/** Repo root (python -m pcg must run from here). */
var REPO_ROOT = "C:\\Users\\WXP\\Documents\\GitHub\\bpm_analysis";
/**
 * pythonw.exe of the Python that has requirements.txt installed. pythonw (no console) with a
 * normal window: a GUI started through a hidden console (python.exe + Run(..., 0)) inherits
 * the hidden show state and never appears.
 */
var PYTHONW = "C:\\Users\\WXP\\.pyenv\\pyenv-win\\versions\\3.13.13\\pythonw.exe";

// Must match pcg.recording.AUDIO_EXTENSIONS (audio/video inputs the analyzer accepts).
var MEDIA_EXTS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aiff", ".aif", ".mp4", ".mkv", ".mov"];

function quoteArg(s) {
    return '"' + String(s).replace(/"/g, '""') + '"';
}

/** shell.Popup avoids DOpus.dlg.message 0x8000ffff in some contexts. flags: 16=critical, 48=warn, 64=info */
function popup(shell, text, title, flags) {
    shell.Popup(String(text), 0, String(title), flags == null ? 48 : flags);
}

function isMediaFile(path) {
    var s = String(path).toLowerCase();
    var i;
    for (i = 0; i < MEDIA_EXTS.length; i++) {
        var ext = MEDIA_EXTS[i];
        if (s.length > ext.length && s.substring(s.length - ext.length) === ext) {
            return true;
        }
    }
    return false;
}

/** Selected media files and folders (folders are searched for recordings by the app). */
function collectSelectedPaths(tab, fso) {
    var paths = [];
    if (!tab || tab.selstats.selitems === 0) {
        return paths;
    }
    var en = new Enumerator(tab.selected);
    for (; !en.atEnd(); en.moveNext()) {
        var item = en.item();
        var pathObj = item.realpath;
        pathObj.Resolve();
        var p = String(pathObj);
        if (fso.FolderExists(p) || (fso.FileExists(p) && isMediaFile(p))) {
            paths.push(p);
        }
    }
    return paths;
}

function OnClick(clickData) {
    var tab = clickData.func.sourcetab;
    var shell = new ActiveXObject("WScript.Shell");
    var fso = new ActiveXObject("Scripting.FileSystemObject");

    if (!fso.FileExists(PYTHONW)) {
        popup(shell, "pythonw.exe not found:\n" + PYTHONW + "\n\nSet PYTHONW in DOpus_bpm_analysis.js.", "PCG Workspace", 16);
        return;
    }
    if (!fso.FolderExists(REPO_ROOT)) {
        popup(shell, "Repo not found:\n" + REPO_ROOT + "\n\nSet REPO_ROOT in DOpus_bpm_analysis.js.", "PCG Workspace", 16);
        return;
    }

    var paths = collectSelectedPaths(tab, fso);
    if (tab && tab.selstats.selitems > 0 && paths.length === 0) {
        popup(shell, "No recording or folder selected.\n\nUse: " + MEDIA_EXTS.join(" "), "PCG Workspace", 48);
        return;
    }

    var cmd = quoteArg(PYTHONW) + " -m pcg app";
    if (paths.length > 0) {
        cmd += " --analyze";
        var i;
        for (i = 0; i < paths.length; i++) {
            cmd += " " + quoteArg(paths[i]);
        }
    }
    shell.CurrentDirectory = REPO_ROOT;
    DOpus.Output("PCG Workspace: " + cmd);
    shell.Run(cmd, 1, false);
}
