# -*- coding: utf-8 -*-
"""Chat window for the optional DyberPet assistant."""

import html
import os
import threading
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

import DyberPet.settings as settings
from DyberPet.LLM.assistant import (
    BUILTIN_ONLY_INTENTS,
    answer_question,
    build_context,
    task_added_text,
    task_completed_text,
    task_deleted_text,
    task_not_found_text,
)
from DyberPet.LLM.local_client import LocalLlmError, chat
from DyberPet.LLM.runtime import (
    LlmConfig,
    collect_snapshot,
    execute_action,
    language_code,
    load_llm_config,
    save_llm_config,
)
from DyberPet.LLM.settings_dialog import LLMSettingsDialog

basedir = settings.BASEDIR


class _ChatRunner(QObject):
    """Runs one request on a daemon thread and hands the result to the UI.

    A plain daemon thread is used instead of QThread so quitting the app can
    never leave a QThread alive that Qt would destroy mid-request. A late
    signal is ignored through the window's generation token.
    """

    done = Signal(dict)

    def start(
        self,
        config: LlmConfig,
        question: str,
        context: str,
        reference: str,
        token: int,
    ):
        thread = threading.Thread(
            target=self._run,
            args=(config, question, context, reference, token),
            daemon=True,
        )
        thread.start()

    def _run(
        self,
        config: LlmConfig,
        question: str,
        context: str,
        reference: str,
        token: int,
    ):
        try:
            text = chat(config.base_url, config.model, question, context, reference)
            self.done.emit(
                {"ok": True, "text": text, "model": config.model, "token": token}
            )
        except LocalLlmError as error:
            self.done.emit({"ok": False, "code": error.code, "token": token})
        except Exception:
            self.done.emit(
                {"ok": False, "code": "LLM_UNREACHABLE", "token": token}
            )


class AssistantWindow(QDialog):
    """A small chat window opened from the pet's right-click menu."""

    def __init__(self, task_panel: Any = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.task_panel = task_panel
        self.config = load_llm_config()
        self.messages: List[Dict[str, str]] = []
        self.runner = _ChatRunner()
        self.runner.done.connect(self._on_model_done)
        self.busy = False
        self.token = 0
        self.started = False

        self.setWindowTitle(self.tr("AI Assistant"))
        self.setWindowIcon(QIcon(os.path.join(basedir, "res/icons/Dialogue_icon.png")))
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.resize(430, 580)
        self.setMinimumSize(340, 420)

        self._build_ui()
        self._apply_style()

    # -- UI -----------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(self.tr("Pet Assistant"))
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.status = QLabel(self.tr("Built-in answers"))
        self.status.setStyleSheet("color: #7a7a7a; font-size: 11px;")
        clear = QPushButton(self.tr("Clear"))
        clear.setFlat(True)
        clear.clicked.connect(self.clear_chat)
        config = QPushButton(self.tr("Settings"))
        config.setFlat(True)
        config.clicked.connect(self.open_settings)
        header.addWidget(title)
        header.addWidget(self.status)
        header.addStretch(1)
        header.addWidget(clear)
        header.addWidget(config)
        root.addLayout(header)

        self.log = QTextBrowser()
        self.log.setOpenExternalLinks(False)
        self.log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.log, 1)

        suggestions = QHBoxLayout()
        suggestions.setSpacing(6)
        for label, command in (
            (self.tr("Tasks"), "show my tasks"),
            (self.tr("Progress"), "how is my progress?"),
            (self.tr("Pet"), "how are you?"),
        ):
            button = QPushButton(label)
            button.setFlat(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, text=command: self.send_text(text))
            suggestions.addWidget(button)
        suggestions.addStretch(1)
        root.addLayout(suggestions)

        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText(self.tr("Ask about tasks, focus, coins, or the pet…"))
        self.input.returnPressed.connect(self.send)
        self.send_button = QPushButton(self.tr("Send"))
        self.send_button.clicked.connect(self.send)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_button)
        root.addLayout(row)

        note = QLabel(
            self.tr(
                "Built-in answers stay on this device. A local model is optional and only "
                "reached through a loopback address."
            )
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #8a8a8a; font-size: 10px;")
        root.addWidget(note)

    def _apply_style(self):
        self.setStyleSheet(
            """
            QDialog { background: #faf8f3; }
            QTextBrowser {
                background: #ffffff;
                border: 1px solid #e2ddd2;
                border-radius: 10px;
                padding: 6px;
            }
            QLineEdit {
                border: 1px solid #d8d2c6;
                border-radius: 8px;
                padding: 6px 8px;
                background: #ffffff;
            }
            QPushButton {
                border-radius: 8px;
                padding: 5px 10px;
            }
            """
        )

    # -- Chat ----------------------------------------------------------------

    def show_window(self):
        if not self.started:
            self.started = True
            self._greet()
        self._update_status()
        self.show()
        self.raise_()
        self.activateWindow()
        self.input.setFocus()

    def _greet(self):
        snapshot = self._snapshot()
        greeting = answer_question("hello", snapshot, language_code())
        self._append("assistant", greeting.text, self.tr("Built-in answers"))

    def clear_chat(self):
        self.token += 1
        self.busy = False
        self.messages = []
        self.log.clear()
        self._set_busy(False)
        self._greet()

    def send_text(self, text: str):
        self.input.setText(text)
        self.send()

    def send(self):
        if self.busy:
            return
        question = self.input.text().strip()
        if not question:
            return
        self.input.clear()
        self._append("user", question, "")
        snapshot = self._snapshot()
        reply = answer_question(question, snapshot, language_code())

        text = reply.text
        source = self.tr("Built-in answers")
        if reply.action:
            outcome = execute_action(reply.action, self.task_panel)
            if outcome.get("ok"):
                if reply.intent == "add_task":
                    text = task_added_text(outcome.get("task", ""), language_code())
                elif reply.intent == "complete_task":
                    text = task_completed_text(
                        outcome.get("task", ""),
                        int(outcome.get("reward", 0)),
                        language_code(),
                    )
                elif reply.intent == "delete_task":
                    text = task_deleted_text(outcome.get("task", ""), language_code())
            else:
                text = task_not_found_text(
                    str(reply.action.get("query", "")), language_code()
                )
            self._append("assistant", text, source)
            return

        if (
            self.config.enabled
            and self.config.model
            and reply.intent not in BUILTIN_ONLY_INTENTS
        ):
            token = self.token + 1
            self.token = token
            context = build_context(snapshot, self.messages, snapshot.get("now"))
            # An "unknown" built-in reply is a fallback, not a fact: leave the
            # reference empty so the model can answer open questions freely.
            reference = "" if reply.intent == "unknown" else text
            # Show the grounded answer immediately; swap in the model's phrasing
            # only if it arrives successfully.
            self._append("assistant", text, source)
            self._set_busy(True, self.tr("Asking %s…") % self.config.model)
            self.runner.start(self.config, question, context, reference, token)
            return

        self._append("assistant", text, source)

    def _on_model_done(self, result: Dict[str, Any]):
        if result.get("token") != self.token:
            return
        self._set_busy(False)
        if not result.get("ok") or not str(result.get("text", "")).strip():
            issue = self._issue_text(str(result.get("code", "")))
            self.status.setText(issue)
            return
        # The deterministic answer is still the last assistant bubble unless the
        # chat was cleared while the request ran.
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        self._append(
            "assistant",
            str(result["text"]),
            self.tr("Local · %s") % str(result.get("model", "")),
        )
        self._update_status()

    def _set_busy(self, busy: bool, note: str = ""):
        self.busy = busy
        self.input.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        if note:
            self.status.setText(note)
        elif not busy:
            self._update_status()

    def _update_status(self, extra: str = ""):
        if self.config.enabled and self.config.model:
            text = self.tr("Local · %s") % self.config.model
        else:
            text = self.tr("Built-in answers")
        if extra:
            text = text + " · " + extra
        self.status.setText(text)

    # -- Rendering -----------------------------------------------------------

    def _append(self, role: str, text: str, source: str):
        self.messages.append({"role": role, "text": text, "source": source})
        self.messages = self.messages[-16:]
        self._render(role, text, source)
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _render(self, role: str, text: str, source: str):
        safe = html.escape(text).replace("\n", "<br>")
        if role == "user":
            block = (
                '<p style="margin:8px 0; text-align:right;">'
                '<span style="color:#7a6a55; font-size:10pt;">%s</span><br>'
                '<span style="background:#f3e6d0; padding:5px 8px; border-radius:8px;">%s</span>'
                "</p>" % (html.escape(self.tr("You")), safe)
            )
        else:
            source_html = ""
            if source:
                source_html = (
                    '<br><span style="color:#8a8a8a; font-size:9pt;">%s</span>'
                    % html.escape(source)
                )
            block = (
                '<p style="margin:8px 0;">'
                '<span style="color:#5f7a6a; font-size:10pt;">%s</span><br>'
                '<span style="background:#eef4ee; padding:5px 8px; border-radius:8px;">%s</span>'
                "%s</p>" % (html.escape(self.tr("Pet")), safe, source_html)
            )
        self.log.append(block)

    # -- Settings and data ---------------------------------------------------

    def open_settings(self):
        dialog = LLMSettingsDialog(self, self.config)
        if dialog.exec():
            try:
                save_llm_config(dialog.config)
            except Exception:
                QMessageBox.warning(
                    self,
                    self.tr("Assistant settings"),
                    self.tr("Settings could not be saved."),
                )
                return
            self.config = dialog.config
            self._update_status()

    def _snapshot(self) -> Dict[str, Any]:
        try:
            return collect_snapshot()
        except Exception:
            return {"pet": {}, "tasks": {"todo": []}, "progress": {}}

    def _issue_text(self, code: str) -> str:
        mapping = {
            "LLM_UNREACHABLE": self.tr("server not running"),
            "LLM_REJECTED": self.tr("server refused the request"),
            "LLM_EMPTY": self.tr("the model returned nothing"),
            "LLM_BAD_RESPONSE": self.tr("not an OpenAI-compatible response"),
            "LLM_BAD_ENDPOINT": self.tr("invalid endpoint"),
            "LLM_NOT_LOOPBACK": self.tr("only a loopback address is allowed"),
            "LLM_BAD_INPUT": self.tr("invalid input"),
            "LLM_TOO_LARGE": self.tr("the response was too large"),
        }
        return mapping.get(code, self.tr("local model failed"))

    def closeEvent(self, event):
        event.ignore()
        self.hide()
