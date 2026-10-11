"""Recording metadata and simplified seated-recovery estimates."""
from html import escape

from PySide6 import QtCore, QtWidgets

from pcg import recording_metadata
from pcg.clock import clock_text
from pcg.recovery_reference import estimate_seated_recovery
from .recovery import INTERVALS


class RecordingPanel(QtWidgets.QScrollArea):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.data = {"version": 1, "person": {}, "recoveries": {}}
        self._loading = False
        self._load_error = ""
        self.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)
        self.save_status = QtWidgets.QLabel()
        self.save_status.setWordWrap(True)
        layout.addWidget(self.save_status)
        self.save_status.hide()
        self.form = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(self.form)
        form.setContentsMargins(0, 0, 0, 0)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.age = self._number(120, 1, " years")
        self.sex = self._combo({"male": "Male", "female": "Female"})
        self.fitness = self._combo({"unknown": "Unknown", "low": "Low", "medium": "Medium", "high": "High"})
        self.fitness.setToolTip(
            "Approximate adult setting (ages 18–79). Low: sedentary; High: regularly active; "
            "Medium: assumed midpoint. Replaces the age/sex adjustment. Children are unaffected.")
        self.seated_rest = self._number(250, 1, " bpm")
        self.seated_rest.setToolTip("Pre-exercise resting HR measured while sitting still. Used by the stop-and-sit estimate.")
        for label, widget in (("Age at recording", self.age), ("Reported sex", self.sex),
                              ("Resting heart rate", self.seated_rest), ("Fitness level (approximate)", self.fitness)):
            form.addRow(label, widget)
        layout.addWidget(self.form)

        self.context_form = QtWidgets.QWidget()
        context = QtWidgets.QFormLayout(self.context_form)
        context.setContentsMargins(0, 8, 0, 0)
        context.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.marker = QtWidgets.QComboBox()
        context.addRow("Recovery marker", self.marker)
        layout.addWidget(self.context_form)
        self.preview_form = QtWidgets.QWidget()
        preview = QtWidgets.QFormLayout(self.preview_form)
        preview.setContentsMargins(0, 8, 0, 0)
        preview.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapAllRows)
        self.preview_start = self._number(300, 1, " bpm")
        preview.addRow("Starting heart rate", self.preview_start)
        layout.addWidget(self.preview_form)
        self.results = QtWidgets.QTextBrowser()
        self.results.setOpenExternalLinks(True)
        self.results.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.results.document().documentLayout().documentSizeChanged.connect(
            lambda size: self.results.setFixedHeight(max(200, int(size.height()) + 20)))
        layout.addWidget(self.results)
        layout.addStretch()
        for spin in (self.age, self.seated_rest, self.preview_start):
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(self._save_person)
        self.sex.currentIndexChanged.connect(self._save_person)
        self.fitness.currentIndexChanged.connect(self._save_person)
        self.marker.currentIndexChanged.connect(self._choose_marker)
        self.form.setEnabled(False)
        self.preview_form.setEnabled(False)
        self.context_form.setEnabled(False)
        self.refresh()

    @staticmethod
    def _number(maximum, decimals, suffix):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0, maximum)
        spin.setDecimals(decimals)
        spin.setSpecialValueText("Not supplied")
        spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _combo(options):
        combo = QtWidgets.QComboBox()
        for key, label in options.items():
            combo.addItem(label, key)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Fixed)
        return combo

    @staticmethod
    def _set_combo(combo, value):
        combo.setCurrentIndex(max(0, combo.findData(value)))

    def load_recording(self):
        self._loading = True
        self._load_error = ""
        try:
            self.data = recording_metadata.load(self.ws.library.root, self.ws.fingerprint)
        except (OSError, ValueError) as exc:
            self.data = {"version": 1, "person": {}, "recoveries": {}}
            self._load_error = f"Cannot read recording metadata: {exc}. Existing metadata will not be overwritten."
        person = self.data["person"]
        for spin, key in ((self.age, "age"), (self.seated_rest, "seated_resting_hr"),
                          (self.preview_start, "preview_start_bpm")):
            value = person.get(key)
            spin.setValue(value if isinstance(value, (int, float)) else 0)
        self._set_combo(self.sex, person.get("reported_sex", ""))
        self._set_combo(self.fitness, person.get("fitness_level", "unknown"))
        self.form.setEnabled(not self._load_error)
        self.preview_form.setEnabled(not self._load_error)
        self.save_status.setText(self._load_error)
        self.save_status.setVisible(bool(self._load_error))
        self._loading = False
        self.refresh()

    def _save(self):
        try:
            recording_metadata.save(self.ws.library.root, self.ws.fingerprint, self.data)
            self.save_status.clear()
            self.save_status.hide()
        except (OSError, ValueError) as exc:
            self.save_status.setText(f"Metadata not saved: {exc}")
            self.save_status.show()

    def _save_person(self, *_):
        if self._loading or self._load_error or not self.ws.fingerprint:
            return
        self.data["person"].update(age=self.age.value() or None,
                                   reported_sex=self.sex.currentData(),
                                   fitness_level=self.fitness.currentData(),
                                   preview_start_bpm=self.preview_start.value() or None,
                                   seated_resting_hr=self.seated_rest.value() or None)
        self._save()
        self._render()

    def _choose_marker(self, *_):
        if self._loading:
            return
        index = self.marker.currentIndex()
        if index >= 0:
            self.ws.recovery_editor.select(index)
            self.marker.setFocus()
        else:
            self.refresh()

    def marker_moved(self, old, new):
        """Move any legacy marker metadata alongside its position."""
        if self._load_error or not self.ws.fingerprint:
            return
        recovery = self.data["recoveries"].pop(str(old), None)
        if recovery is not None:
            self.data["recoveries"][str(new)] = recovery
            if not self._load_error:
                self._save()

    def marker_removed(self, start):
        if self._load_error or not self.ws.fingerprint:
            return
        if self.data["recoveries"].pop(str(start), None) is not None and not self._load_error:
            self._save()

    def refresh(self):
        editor = self.ws.recovery_editor
        previous = self.marker.currentData()
        self._loading = True
        self.marker.clear()
        for index, start in enumerate(editor.starts):
            self.marker.addItem(f"R{index + 1} · {clock_text(start)}", str(start))
        selected = editor.selected
        if selected is None:
            selected = max(0, self.marker.findData(previous))
        self.marker.setCurrentIndex(selected if editor.starts else -1)
        self.context_form.setEnabled(bool(editor.starts) and bool(self.ws.fingerprint) and not self._load_error)
        self._loading = False
        self._render()

    def _render(self):
        index = self.marker.currentIndex()
        self.context_form.setVisible(index >= 0)
        self.preview_form.setVisible(index < 0)
        if index < 0:
            self._render_estimate(self.preview_start.value() or None, "Predicted recovery", None)
            return
        self._render_seated_estimate(index)

    def _render_seated_estimate(self, index):
        editor = self.ws.recovery_editor
        start_bpm = next((y for i, _, y in editor.positions if i == index), None)
        observed = {int(line.end - line.start): line for i, line in editor.lines if i == index}
        self._render_estimate(start_bpm, f"R{index + 1} · {editor.source}", observed)

    def _render_estimate(self, start_bpm, title, observed):
        person = self.data["person"]
        rest = person.get("seated_resting_hr")
        age = person.get("age")
        # Use the displayed selection, including Male for legacy missing categories.
        demographics = {"age": age, "reported_sex": self.sex.currentData()} if age is not None else {}
        demographics["fitness_level"] = self.fitness.currentData()
        reason = ""
        estimates = {}
        try:
            if start_bpm is None:
                raise ValueError("Enter starting heart rate to calculate the prediction." if observed is None
                                 else "Starting BPM is unavailable at this marker.")
            if rest is None:
                raise ValueError("Enter resting heart rate to calculate the estimate.")
            for seconds in INTERVALS:
                if seconds <= 300:
                    value = estimate_seated_recovery(start_bpm, rest, seconds, **demographics)
                    if value is not None:
                        estimates[seconds] = value
        except ValueError as exc:
            reason = str(exc)
        start_label = "start BPM unavailable" if start_bpm is None else f"start {start_bpm:.1f} bpm"
        html = [f"<b>{escape(title)} · {start_label}</b>",
                f"<p>{escape(reason)}</p>" if reason else ""]
        if observed is None:
            if estimates:
                html.append("<table cellpadding='4'><tr><th>Time</th><th>Estimated BPM</th><th>Estimated drop</th></tr>")
                for seconds, value in estimates.items():
                    html.append(f"<tr><td>{seconds} s</td><td>{value:.1f} bpm</td>"
                                f"<td>{start_bpm - value:.1f} bpm</td></tr>")
                html.append("</table>")
            self.results.setHtml("".join(html))
            return
        html.append("<table cellpadding='4'><tr><th>Time</th><th>Observed drop</th><th>Estimated drop</th></tr>")
        for seconds in INTERVALS:
            line = observed.get(seconds)
            measured = f"{-line.change:.1f} bpm" if line else ""
            value = estimates.get(seconds)
            if line is None and value is None:
                continue
            estimated = f"{start_bpm - value:.1f} bpm" if value is not None else ""
            html.append(f"<tr><td>{seconds} s</td><td>{measured}</td><td>{estimated}</td></tr>")
        html.append("</table>")
        if estimates:
            html.append("<p>Estimated endpoint BPM:<br>")
            html.extend(f"{seconds} s: {value:.1f} bpm<br>" for seconds, value in estimates.items())
            html.append("</p>")
        self.results.setHtml("".join(html))
