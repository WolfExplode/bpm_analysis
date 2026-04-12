# Directory Opus (DOpus) Scripting Reference

Practical reference for writing JScript button scripts in Directory Opus.
Based on scripts in this repo + the official docs at https://docs.dopus.com

---

## Table of Contents

1. [What is a DOpus Script Button?](#1-what-is-a-dopus-script-button)
2. [Installing a Script into DOpus](#2-installing-a-script-into-dopus)
3. [Script Structure & Entry Point](#3-script-structure--entry-point)
4. [Key DOpus Objects](#4-key-dopus-objects)
5. [Working with Selected Files](#5-working-with-selected-files)
6. [Running External Programs](#6-running-external-programs)
7. [File System via ActiveX (FSO)](#7-file-system-via-activex-fso)
8. [Clipboard](#8-clipboard)
9. [Script Dialogs](#9-script-dialogs)
10. [XML Dialog Resource Format](#10-xml-dialog-resource-format)
11. [Dialog Control Types & Values](#11-dialog-control-types--values)
12. [Message Loop & Events](#12-message-loop--events)
13. [DOpus.Output (Logging)](#13-dopusoutput-logging)
14. [JScript Gotchas (ES3)](#14-jscript-gotchas-es3)
15. [Persistent Settings Pattern](#15-persistent-settings-pattern)
16. [Quick Recipe Cheatsheet](#16-quick-recipe-cheatsheet)

---

## 1. What is a DOpus Script Button?

A **Script Function** is a button or hotkey in DOpus whose action is written in JScript (or VBScript).
The script lives inside the button definition itself — it is **not** a standalone `.js` file that DOpus loads.

There are two separate things that make up a complete script button:

| Part | What it is |
|------|-----------|
| **Script code** (`.js`) | The JScript logic — stored in the button's "Script Code" tab |
| **Resources** (`.xml`) | XML defining dialog layouts — stored in the button's "Resources" tab |

When developing outside DOpus (in an editor like Cursor), keep the `.js` and `.xml` as separate files for source control, then paste them into the button editor when ready.

---

## 2. Installing a Script into DOpus

1. Enter **Customize** mode: `Settings → Customize Toolbars` (or right-click a toolbar → Customize).
2. In the **Commands** panel, drag a new button onto any toolbar (or find an existing one to edit).
3. Right-click the button → **Edit**.
4. Change the function type from *Standard Function* to **Script Function**.
5. Choose **JScript** from the language dropdown.
6. Paste your `.js` code into the **Script Code** tab.
7. Switch to the **Resources** tab:
   - Click `Dialogs` dropdown → **New Dialog**, give it the exact name referenced by `dlg.template` in your script.
   - Paste the XML from your `.xml` file into the raw XML editor (accessible via the XML button in the dialog editor).
8. Click **OK** to save.

> The dialog name in the XML `<resource name="...">` must match `dlg.template = "..."` exactly.

---

## 3. Script Structure & Entry Point

Every button script must define `OnClick`. DOpus calls it when the button is clicked.

```js
function OnClick(clickData) {
    // clickData is the ClickData object
    var func = clickData.func;          // Func object
    var tab  = func.sourcetab;          // Tab object (current source folder tab)
    var cmd  = func.command;            // Command object (to run DOpus commands)

    // Common ActiveX objects needed for almost everything
    var shell = new ActiveXObject("WScript.Shell");
    var fso   = new ActiveXObject("Scripting.FileSystemObject");
}
```

### clickData properties

| Property | Type | Description |
|----------|------|-------------|
| `clickData.func` | `Func` | The Func object for this invocation |
| `clickData.func.sourcetab` | `Tab` | The active source folder tab |
| `clickData.func.desttab` | `Tab` | The active destination folder tab |
| `clickData.func.command` | `Command` | For running DOpus internal commands |
| `clickData.func.qualifiers` | `string` | Modifier keys held: `"shift"`, `"ctrl"`, `"alt"`, `"none"`, etc. |

---

## 4. Key DOpus Objects

### DOpus (global singleton)

```js
DOpus.Output("message");         // Print to the script log / output pane
DOpus.dlg;                       // Create a new Dialog object
DOpus.GetClip("text");           // Read clipboard as text (NOT GetClipText)
DOpus.version;                   // DOpus version string
```

### Tab object (`clickData.func.sourcetab`)

```js
var tab = clickData.func.sourcetab;

tab.path;                        // Current folder path (string-like Path object)
tab.selected;                    // Collection of selected Item objects (files + folders)
tab.selected_files;              // Collection of selected files only
tab.selected_dirs;               // Collection of selected folders only
tab.all;                         // All items in the tab
tab.selstats.selfiles;           // Count of selected files (int)
tab.selstats.seldirs;            // Count of selected dirs (int)
tab.selstats.selitems;           // Count of all selected items (int)
```

### Item object (from `tab.selected`, etc.)

```js
var item = ...;                  // obtained from an Enumerator over tab.selected_files

item.name;                       // Full filename with extension, e.g. "photo.jpg"
item.name_stem;                  // Filename without extension, e.g. "photo"
item.ext;                        // Extension with dot, e.g. ".jpg"
item.path;                       // Parent folder path
item.realpath;                   // Full path including filename (resolves junctions/links)
item.size;                       // File size (FileSize object; use +0 or String() to convert)
item.is_dir;                     // bool: true if it's a folder
item.is_junction;                // bool: true if junction/symlink
item.modify;                     // Last modified date (Date object)
```

> **Important:** Use `item.realpath` (not `item.path + "\\" + item.name`) for the full file path.
> Call `.Resolve()` on `realpath` when you need to dereference symlinks/junctions, then stringify with `+""`.

```js
var en = new Enumerator(tab.selected_files);
for (; !en.atEnd(); en.moveNext()) {
    var item = en.item();
    var fullPath = item.realpath + "";   // force to string
}
```

### Command object (`clickData.func.command`)

```js
var cmd = clickData.func.command;
cmd.RunCommand("Go REFRESH");           // Run any DOpus internal command
cmd.RunCommand("Select ALL");
```

---

## 5. Working with Selected Files

```js
function OnClick(clickData) {
    var tab      = clickData.func.sourcetab;
    var selected = tab.selected_files;      // files only, not folders

    if (tab.selstats.selfiles == 0) {
        DOpus.dlg.message("No files selected.", "Error");
        return;
    }

    var en = new Enumerator(selected);
    for (; !en.atEnd(); en.moveNext()) {
        var item = en.item();
        DOpus.Output(item.realpath + "");
    }

    // Refresh the file display after modifying files
    clickData.func.command.RunCommand("Go REFRESH");
}
```

---

## 6. Running External Programs

Use `WScript.Shell` — **not** DOpus's Command object — for launching external processes.

```js
var shell = new ActiveXObject("WScript.Shell");

// shell.Run(command, windowStyle, bWait)
// windowStyle: 0=hidden, 1=normal, 2=minimized, 3=maximized
// bWait: true = block until process exits, false = fire and forget

shell.Run('notepad.exe "C:\\path\\file.txt"', 1, false);   // non-blocking
shell.Run('ffmpeg.exe -i "input.mp4" "output.mp3"', 0, true); // hidden, wait for exit

var exitCode = shell.Run('some.exe', 0, true);  // returns exit code when bWait=true
```

### Avoid cmd.exe for complex commands — use a .ps1 temp file

`cmd.exe` expands `%` characters in yt-dlp output templates and breaks them.
The pattern used in this repo is to write a temporary PowerShell script and run that:

```js
var tempPs1 = shell.ExpandEnvironmentStrings("%TEMP%") + "\\my-script.ps1";
var ps1 = fso.CreateTextFile(tempPs1, true, false);
ps1.WriteLine("Set-Location 'C:\\some\\path'");
ps1.WriteLine("yt-dlp -o \"%(title)s.%(ext)s\" \"https://...\"");
ps1.Close();

// -NoExit keeps the window open; omit it to close when done
shell.Run('powershell -ExecutionPolicy Bypass -File "' + tempPs1 + '"', 1, false);
```

### Environment variable expansion

```js
var appdata = shell.ExpandEnvironmentStrings("%APPDATA%");
var temp    = shell.ExpandEnvironmentStrings("%TEMP%");
```

---

## 7. File System via ActiveX (FSO)

```js
var fso = new ActiveXObject("Scripting.FileSystemObject");

// Existence checks
fso.FileExists("C:\\path\\file.txt");          // bool
fso.FolderExists("C:\\path\\dir");             // bool

// Create folder
fso.CreateFolder("C:\\path\\newdir");

// Read a text file
var f = fso.OpenTextFile("C:\\path\\file.txt", 1, false);  // 1=ForReading
var content = f.ReadAll();
f.Close();

// Write/overwrite a text file
var f = fso.CreateTextFile("C:\\path\\file.txt", true, false);  // overwrite=true, unicode=false
f.WriteLine("line 1");
f.WriteLine("line 2");
f.Close();

// Append to a file
var f = fso.OpenTextFile("C:\\path\\file.txt", 8, true);  // 8=ForAppending, create=true
f.WriteLine("appended line");
f.Close();

// OpenTextFile modes: 1=ForReading, 2=ForWriting (overwrites), 8=ForAppending
```

---

## 8. Clipboard

```js
// Read text from clipboard
var clip = DOpus.GetClip("text");   // returns string or null
if (clip) {
    var url = clip.replace(/^\s+|\s+$/g, "");  // trim (no String.trim in ES3)
}
```

> **Do not use** `DOpus.GetClipText()` — it does not exist. The correct method is `DOpus.GetClip("text")`.

---

## 9. Script Dialogs

DOpus supports custom GUI dialogs defined in XML. There are two modes: **simple** and **detached**.

### Simple dialog (no event handling)

```js
var dlg = DOpus.dlg;
dlg.window   = clickData.func.sourcetab;  // parent window
dlg.template = "MyDialogName";            // must match <resource name="...">
dlg.Create();

// Pre-set control values here if needed
dlg.control("my_edit").value = "default text";

dlg.Show();   // blocks until user closes the dialog

// After Show() returns, read values
var result = dlg.result;  // "1"=OK button, "2"=Cancel, "0"=window X closed
var text   = dlg.control("my_edit").value;
```

### Detached dialog (with event handling / message loop)

Used when you need to react to control changes while the dialog is open (e.g. refreshing a dropdown).

```js
var dlg = DOpus.dlg;
dlg.window   = clickData.func.sourcetab;
dlg.template = "MyDialogName";
dlg.detach   = true;    // enables message loop; calling Create() also sets this automatically
dlg.Create();           // creates the window hidden

// Set control values before showing
dlg.control("my_check").value = true;

dlg.Show();  // shows the dialog; returns immediately because detach=true

var dialogResult = 0;
while (true) {
    var msg = dlg.GetMsg();         // blocks until next event
    if (!msg.result) {              // dialog was closed
        dialogResult = dlg.result;
        break;
    }

    // Handle events
    if (msg.event === "click" && msg.control === "my_button") {
        // user clicked my_button
    }
    if (msg.event === "selchange" && msg.control === "my_combo") {
        // combo box selection changed
    }
    if (msg.event === "editchange" && msg.control === "my_edit") {
        // text was typed in my_edit
    }
}

// After loop: dialogResult is the close button result
// "1" = OK/default button, "2" = Cancel, "0" = window X
```

### dlg.result values

| Value | Meaning |
|-------|---------|
| `"1"` | User clicked the button with `close="1"` (OK / default) |
| `"2"` | User clicked the button with `close="2"` (Cancel) |
| `"0"` | User closed the dialog via the title bar X |

> Check with `==` not `===` since `dlg.result` is a string and you may be comparing to an int.

### Simple popup dialogs (no XML needed)

```js
// Message box (info)
DOpus.dlg.message("Something went wrong.", "Title");

// OK/Cancel confirmation
var r = DOpus.dlg.request("Are you sure?", "OK|Cancel", "Confirm");
// r == 1 for OK, r == 0 for Cancel/X

// WScript.Shell popup (also works, has icon flags)
var shell = new ActiveXObject("WScript.Shell");
shell.Popup("Error text", 0, "Title", 16);   // 16 = error icon
shell.Popup("Info text",  0, "Title", 64);   // 64 = info icon
```

---

## 10. XML Dialog Resource Format

XML goes in the button's **Resources** tab. The `<resource name>` must match `dlg.template`.

```xml
<resources>
  <resource name="MyDialogName" type="dialog">
    <dialog fontsize="9" height="200" lang="english" title="My Dialog Title" width="350">

      <!-- Static label -->
      <control halign="left" height="10" name="lbl1" title="Enter a value:" type="static" width="200" x="12" y="10"/>

      <!-- Single-line text input -->
      <control height="14" name="my_edit" type="edit" width="300" x="12" y="24"/>

      <!-- Checkbox -->
      <control height="12" name="my_check" title="Enable option" type="check" width="300" x="12" y="44"/>

      <!-- Radio button group: first in group gets group="yes" -->
      <control group="yes" height="12" name="radio1" title="Option A" type="radio" width="140" x="12" y="62"/>
      <control           height="12" name="radio2" title="Option B" type="radio" width="140" x="12" y="78"/>

      <!-- Combo box (dropdown) — static items in XML -->
      <control height="14" name="mode_combo" type="combo" width="150" x="12" y="96">
        <contents>
          <item text="Video" />
          <item text="Audio" />
        </contents>
      </control>

      <!-- Combo box populated dynamically (no contents here) -->
      <control height="120" name="dyn_combo" type="combo" width="300" x="12" y="116"/>

      <!-- Buttons -->
      <!-- close="1" = OK (sends result "1"), default="yes" = default button (Enter key) -->
      <!-- close="2" = Cancel (sends result "2") -->
      <control close="1" default="yes" height="18" name="ok_btn"     title="OK"     type="button" width="70" x="190" y="170"/>
      <control close="2"               height="18" name="cancel_btn" title="Cancel" type="button" width="70" x="270" y="170"/>

      <!-- Button without close — triggers a click event in the message loop -->
      <control height="18" name="refresh_btn" title="Refresh" type="button" width="80" x="12" y="170"/>

    </dialog>
  </resource>
</resources>
```

### Layout coordinates

- `x`, `y` — position from top-left of dialog interior, in dialog units (not pixels)
- `width`, `height` — size in dialog units
- Dialog units are roughly 1 DU ≈ font_size/4 pixels horizontally, font_size/8 pixels vertically at `fontsize="9"`
- Dialog `width`/`height` includes the frame; control positions are relative to the client area

---

## 11. Dialog Control Types & Values

### How `.value` works per control type

| Control type | `.value` read | `.value` write |
|---|---|---|
| `edit` | `string` — current text | `string` — sets text |
| `check` | `bool` — true if checked | `bool` — check or uncheck |
| `radio` | `bool` — true if this radio is selected | `bool` — set to true to select |
| `combo` | `DialogListItem` object (has `.index`, `.name`) | `int` — 0-based index, or `DialogListItem` |
| `list` (listbox) | `DialogListItem` object | `int` — 0-based index |

> **Combo `.value` quirk:** `.value` on a combo returns a `DialogListItem`, not a plain index or string.
> Use `.value.index` for the 0-based index, `.value.name` for the label text.

### Combo/list box methods

```js
var ctrl = dlg.control("my_combo");

// Clear all items
ctrl.RemoveItem(-1);

// Add items — AddItem(label, data_value)
ctrl.AddItem("Display Label", "optional_data");   // data can be any value

// Select by index
ctrl.SelectItem(0);                // first item
ctrl.SelectItem(someItem);         // by DialogListItem

// Read selection
var sel = ctrl.value;              // DialogListItem
var idx = sel.index;               // 0-based int
var lbl = sel.name;                // display label string

// Get item by name
var item = ctrl.GetItemByName("Display Label");   // returns DialogListItem or null
```

### Accessing controls

```js
dlg.control("control_name")           // returns Control object by name
dlg.control("control_name").value     // read value
dlg.control("control_name").value = x // write value
dlg.control("control_name").enabled = false  // disable
dlg.control("control_name").label = "New text"  // change label (static, button, checkbox title)
```

---

## 12. Message Loop & Events

The `Msg` object (returned by `dlg.GetMsg()`) has these key properties:

| Property | Type | Description |
|----------|------|-------------|
| `msg.result` | `bool` | `false` when the dialog has been closed — exit the loop |
| `msg.event` | `string` | Event type (see below) |
| `msg.control` | `string` | Name of the control that triggered the event |
| `msg.data` | `variant` | Event-specific data (e.g. `true`/`false` for checkbox state) |
| `msg.value` | `variant` | Current value of the control |
| `msg.focus` | `bool` | Whether the control had focus (useful for radio/edit to distinguish user vs script changes) |

### Common event types

| `msg.event` | When it fires |
|---|---|
| `"click"` | A button was clicked, or a checkbox/radio was toggled |
| `"selchange"` | A combo box or list box selection changed |
| `"editchange"` | Text was typed in an edit control |
| `"dblclk"` | An item in a list was double-clicked |
| `"focus"` | A control gained focus |
| `"close"` | Dialog is being closed (can be used to prevent close) |

### Handling button clicks with `close` attribute

Buttons with `close="1"` or `close="2"` automatically close the dialog **and** stop `GetMsg()` from returning further events (the loop condition `!msg.result` triggers). You do **not** receive a separate `click` event for close buttons — the loop exits, and you check `dlg.result`.

Buttons **without** a `close` attribute send a `click` event to the message loop but do not close the dialog.

---

## 13. DOpus.Output (Logging)

```js
DOpus.Output("Some message");   // appears in Script Log (View → Output Window or Ctrl+F6 ?)
```

The output pane is visible via `Help → Logs → Script Log` in DOpus.
Use it liberally for debugging — it does not affect script execution.

---

## 14. JScript Gotchas (ES3)

DOpus uses the Windows Script Host JScript engine which is **ES3** (circa 1999). Many modern JS features are unavailable.

| Feature | ES3 situation | Workaround |
|---------|--------------|------------|
| `String.prototype.trim()` | Does not exist | `s.replace(/^\s+|\s+$/g, "")` |
| `Array.forEach`, `.map`, `.filter` | Do not exist | Use `for` loops |
| `Array.isArray` | Does not exist | `Object.prototype.toString.call(x) === "[object Array]"` |
| `let` / `const` | Do not exist | Use `var` |
| Arrow functions `=>` | Do not exist | Use `function` |
| Template literals `` `${x}` `` | Do not exist | String concatenation |
| `parseInt` radix | Always provide: `parseInt(val, 10)` | — |
| `for...of` | Does not exist | Use `for` loop or `Enumerator` |
| Iterating DOpus collections | Cannot use `for...in` or `for...of` | Use `new Enumerator(collection)` |

### Enumerator pattern for DOpus collections

```js
var en = new Enumerator(tab.selected_files);
for (; !en.atEnd(); en.moveNext()) {
    var item = en.item();
    // use item
}
```

### Comparing dialog result

`dlg.result` is a string (`"1"`, `"2"`, `"0"`). Use `==` (loose equality) to compare:

```js
if (dialogResult == "0" || dialogResult == "2") {
    // cancelled
    return;
}
```

---

## 15. Persistent Settings Pattern

DOpus scripts live inside buttons and have no persistent state between runs.
Store settings in a plain INI-style text file under `%APPDATA%`:

```js
// Path helper — lazily resolved on first call
var SETTINGS_FILE = null;
function getSettingsPath(shell) {
    if (!SETTINGS_FILE) {
        SETTINGS_FILE = shell.ExpandEnvironmentStrings("%APPDATA%") + "\\DOpus_myscript_settings.ini";
    }
    return SETTINGS_FILE;
}

function loadSettings(shell, fso) {
    var out = { mode: 0, someText: "", enabled: 1 };
    try {
        var path = getSettingsPath(shell);
        if (!fso.FileExists(path)) return out;
        var f = fso.OpenTextFile(path, 1, false);
        var lines = f.ReadAll().split("\n");
        f.Close();
        for (var i = 0; i < lines.length; i++) {
            var line = lines[i].replace(/\r$/, "");
            var eq = line.indexOf("=");
            if (eq < 1) continue;
            var k = line.substring(0, eq);
            var v = line.substring(eq + 1);
            if      (k === "mode")     out.mode     = parseInt(v, 10) || 0;
            else if (k === "someText") out.someText = v;
            else if (k === "enabled")  out.enabled  = (parseInt(v, 10) === 0) ? 0 : 1;
        }
    } catch (e) { /* use defaults on any error */ }
    return out;
}

function saveSettings(shell, fso, mode, someText, enabled) {
    try {
        var f = fso.OpenTextFile(getSettingsPath(shell), 2, true); // ForWriting, create
        f.WriteLine("mode=" + mode);
        f.WriteLine("someText=" + String(someText).replace(/[\r\n]/g, " "));
        f.WriteLine("enabled=" + enabled);
        f.Close();
    } catch (e) { /* ignore */ }
}
```

---

## 16. Quick Recipe Cheatsheet

### Minimal button script (no dialog)

```js
function OnClick(clickData) {
    var shell = new ActiveXObject("WScript.Shell");
    var tab   = clickData.func.sourcetab;
    DOpus.Output("Current folder: " + tab.path);
    shell.Run('notepad.exe', 1, false);
}
```

### Get current folder path

```js
var destPath = String(clickData.func.sourcetab.path);
```

### Check if any files are selected

```js
if (clickData.func.sourcetab.selstats.selfiles == 0) {
    DOpus.dlg.message("Select at least one file.", "Error");
    return;
}
```

### Iterate selected files and get full paths

```js
var paths = [];
var en = new Enumerator(clickData.func.sourcetab.selected_files);
for (; !en.atEnd(); en.moveNext()) {
    var item = en.item();
    var pathObj = item.realpath;
    pathObj.Resolve();
    paths.push(pathObj + "");
}
```

### Show a simple OK/Cancel dialog (no XML needed)

```js
var r = DOpus.dlg.request("Proceed?", "Yes|No", "Confirm");
if (r != 1) return;   // user clicked No or closed
```

### Open a detached dialog and wait for OK

```js
var dlg = DOpus.dlg;
dlg.window   = clickData.func.sourcetab;
dlg.template = "MyDialog";
dlg.detach   = true;
dlg.Create();
dlg.control("my_edit").value = "default";
dlg.Show();

var result = 0;
while (true) {
    var msg = dlg.GetMsg();
    if (!msg.result) { result = dlg.result; break; }
}
if (result == "0" || result == "2") return;  // cancelled

var text = dlg.control("my_edit").value;
```

### Dynamically populate a combo box

```js
var combo = dlg.control("my_combo");
combo.RemoveItem(-1);                          // clear all
combo.AddItem("First option", "val1");
combo.AddItem("Second option", "val2");
combo.SelectItem(0);                           // select first
```

### Read selected combo item

```js
var sel   = dlg.control("my_combo").value;    // DialogListItem
var idx   = sel.index;                         // 0-based
var label = sel.name;                          // display text
```

### Run PowerShell script hidden, wait for exit

```js
var exitCode = shell.Run('powershell -NoProfile -ExecutionPolicy Bypass -File "' + tempPs1 + '"', 0, true);
```

### Run PowerShell with visible window, keep open (for user to see output)

```js
shell.Run('powershell -NoExit -ExecutionPolicy Bypass -File "' + tempPs1 + '"', 1, false);
```

### Refresh the file display after changes

```js
clickData.func.command.RunCommand("Go REFRESH");
```

### Error / info popup via WScript.Shell

```js
var shell = new ActiveXObject("WScript.Shell");
shell.Popup("Something failed: " + msg, 0, "Error", 16);   // 16 = error icon (red X)
shell.Popup("Done!",                      0, "Info",  64);  // 64 = info icon (blue i)
```

---

## Official Documentation

- Full scripting reference: https://docs.dopus.com/doku.php?id=reference:scripting_reference
- Script functions overview: https://docs.dopus.com/doku.php?id=scripting:script_functions
- Script dialogs: https://docs.dopus.com/doku.php?id=scripting:script_dialogs
- Dialog object: https://docs.dopus.com/doku.php?id=reference:scripting_reference:scripting_objects:dialog
- Control object: https://docs.dopus.com/doku.php?id=reference:scripting_reference:scripting_objects:control
- Tab object: https://docs.dopus.com/doku.php?id=reference:scripting_reference:scripting_objects:tab
- Item object: https://docs.dopus.com/doku.php?id=reference:scripting_reference:scripting_objects:item
- Func object: https://docs.dopus.com/doku.php?id=reference:scripting_reference:scripting_objects:func
- Community forum (Resource Centre): https://resource.dopus.com
