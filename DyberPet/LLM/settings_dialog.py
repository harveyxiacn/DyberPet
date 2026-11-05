# -*- coding: utf-8 -*-
"""Settings dialog for the optional local assistant model."""

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from DyberPet.LLM.local_client import (
    DEFAULT_SERVERS,
    LocalLlmError,
    normalize_base_url,
    probe,
)
from DyberPet.LLM.runtime import LlmConfig


class LLMSettingsDialog(QDialog):
    """Enable a loopback OpenAI-compatible server and pick one of its models."""

    def __init__(self, parent: Optional[QWidget] = None, config: Optional[LlmConfig] = None):
        super().__init__(parent)
        self.config = config or LlmConfig(False, DEFAULT_SERVERS[0][2], "")

        self.setWindowTitle(self.tr("Assistant settings"))
        self.setModal(True)
        self.resize(460, 0)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        intro = QLabel(
            self.tr(
                "The assistant answers from the pet's own data without any model. "
                "Connecting a local model lets it phrase those answers more naturally "
                "and handle open questions."
            )
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        self.enabled = QCheckBox(self.tr("Use a local model when one is available"))
        self.enabled.setChecked(bool(self.config.enabled))
        self.enabled.toggled.connect(self._sync_enabled)
        root.addWidget(self.enabled)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.server = QComboBox()
        for key, label, _url in DEFAULT_SERVERS:
            self.server.addItem(label, key)
        self.server.addItem(self.tr("Custom"), "custom")
        self.server.currentIndexChanged.connect(self._server_changed)
        form.addRow(self.tr("Server"), self.server)

        endpoint_row = QHBoxLayout()
        self.endpoint = QLineEdit(self.config.base_url)
        self.endpoint.setPlaceholderText("http://127.0.0.1:11434/v1")
        self.check_button = QPushButton(self.tr("Check"))
        self.check_button.clicked.connect(self.check_server)
        endpoint_row.addWidget(self.endpoint, 1)
        endpoint_row.addWidget(self.check_button)
        form.addRow(self.tr("Endpoint"), endpoint_row)

        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setInsertPolicy(QComboBox.NoInsert)
        if self.config.model:
            self.model.addItem(self.config.model)
            self.model.setCurrentText(self.config.model)
        form.addRow(self.tr("Model"), self.model)

        root.addLayout(form)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #7a7a7a; font-size: 11px;")
        root.addWidget(self.status)

        note = QLabel(
            self.tr(
                "Only localhost addresses are accepted. The model server must expose an "
                "OpenAI-compatible /v1/chat/completions endpoint. No API key is sent."
            )
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a8a8a; font-size: 10px;")
        root.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._select_server_for(self.config.base_url)
        self._sync_enabled(self.enabled.isChecked())

    # -- helpers ------------------------------------------------------------

    def _select_server_for(self, url: str):
        clean = (url or "").rstrip("/")
        for index in range(self.server.count()):
            key = self.server.itemData(index)
            for server_key, _label, server_url in DEFAULT_SERVERS:
                if key == server_key and clean in (
                    server_url,
                    server_url[: -len("/v1")],
                ):
                    self.server.setCurrentIndex(index)
                    return
        self.server.setCurrentIndex(self.server.count() - 1)

    def _server_changed(self):
        key = self.server.currentData()
        for server_key, _label, url in DEFAULT_SERVERS:
            if key == server_key:
                self.endpoint.setText(url)
                self.status.setText("")
                return

    def _sync_enabled(self, on: bool):
        for widget in (self.server, self.endpoint, self.check_button, self.model):
            widget.setEnabled(on)
        if not on:
            self.status.setText(self.tr("Using built-in answers only."))

    # -- actions ------------------------------------------------------------

    def check_server(self):
        raw = self.endpoint.text().strip()
        if not raw:
            self.status.setText(self.tr("Enter an endpoint first."))
            return
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            result = probe(raw, timeout=2.0)
        finally:
            QApplication.restoreOverrideCursor()

        if not result.reachable:
            issue = {
                "not_running": self.tr("Nothing is listening there. Start the server first."),
                "rejected": self.tr("The server refused the request."),
                "incompatible": self.tr("That address is not a valid loopback endpoint."),
            }.get(result.issue or "", self.tr("Could not reach the server."))
            self.status.setText(issue)
            return

        self.endpoint.setText(result.base_url)
        current = self.model.currentText().strip()
        self.model.clear()
        self.model.addItems(result.models)
        if current in result.models:
            self.model.setCurrentText(current)
        elif result.models:
            self.model.setCurrentIndex(0)
        if result.issue == "no_models":
            self.status.setText(
                self.tr("Connected, but no models are installed yet. Pull one in the server.")
            )
        else:
            self.status.setText(
                self.tr("Connected. Found %d model(s).") % len(result.models)
            )

    def save(self):
        enabled = self.enabled.isChecked()
        raw = self.endpoint.text().strip()
        model = self.model.currentText().strip()
        if enabled:
            try:
                base = normalize_base_url(raw)
            except LocalLlmError:
                QMessageBox.warning(
                    self,
                    self.tr("Assistant settings"),
                    self.tr(
                        "Use a localhost address such as http://127.0.0.1:11434/v1."
                    ),
                )
                return
            if not model:
                QMessageBox.warning(
                    self,
                    self.tr("Assistant settings"),
                    self.tr("Choose a model, or use Check to list the installed ones."),
                )
                return
        else:
            base = raw[:300]

        self.config = LlmConfig(enabled, base, model[:120])
        self.accept()
