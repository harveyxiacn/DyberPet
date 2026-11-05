# -*- coding: utf-8 -*-
"""Glue between the pure assistant core and the running DyberPet app.

The functions here read and write the game's existing settings and task data.
They never touch the network; the optional model call is made by the chat
window through ``local_client``.
"""

from dataclasses import dataclass
from datetime import datetime
import uuid
from typing import Any, Dict, Optional

from PySide6.QtCore import QObject, Signal

import DyberPet.settings as settings
from .assistant import find_task, task_items


class _Bridge(QObject):
    """Lets the assistant ask an open task panel to refresh itself."""

    tasks_changed = Signal()


bridge = _Bridge()


@dataclass
class LlmConfig:
    enabled: bool
    base_url: str
    model: str


def language_code() -> str:
    code = str(getattr(settings, "language_code", "") or "").lower()
    return "zh" if code.startswith("zh") else "en"


def load_llm_config() -> LlmConfig:
    return LlmConfig(
        enabled=bool(getattr(settings, "llm_enabled", False)),
        base_url=str(getattr(settings, "llm_base_url", "http://127.0.0.1:11434/v1") or ""),
        model=str(getattr(settings, "llm_model", "") or ""),
    )


def save_llm_config(config: LlmConfig) -> None:
    settings.llm_enabled = bool(config.enabled)
    settings.llm_base_url = str(config.base_url or "")
    settings.llm_model = str(config.model or "")
    settings.save_settings()


def collect_snapshot() -> Dict[str, Any]:
    """Read the current game state; no credentials, paths or item names leave."""

    pet = settings.pet_data
    task_data = settings.task_data.taskData
    todo = task_data.get("tasks_todo", {}) or {}
    done = task_data.get("tasks_done", {}) or {}
    history = task_data.get("history") or []
    today_minutes = 0
    if history and isinstance(history[-1], (list, tuple)) and len(history[-1]) > 1:
        try:
            today_minutes = int(history[-1][1])
        except (TypeError, ValueError):
            today_minutes = 0
    return {
        "pet": {
            "name": str(getattr(settings, "petname", "") or ""),
            "hp": int(getattr(pet, "hp", 0) or 0),
            "hp_tier": int(getattr(pet, "hp_tier", 2) or 0),
            "favorability": int(getattr(pet, "fv", 0) or 0),
            "favorability_level": int(getattr(pet, "fv_lvl", 0) or 0),
            "coins": int(getattr(pet, "coins", 0) or 0),
            "days": int(getattr(pet, "days", 0) or 0),
        },
        "tasks": {
            "todo": [
                {"id": str(task_id), "text": str(text)}
                for task_id, text in todo.items()
            ],
            "done_count": len(done),
            "completed_total": int(task_data.get("n_tasks", 0) or 0),
        },
        "progress": {
            "today_minutes": today_minutes,
            "goal_minutes": int(task_data.get("goal", 0) or 0),
            "streak_days": int(task_data.get("n_days", 0) or 0),
        },
        "now": datetime.now().isoformat(timespec="minutes"),
    }


def _direct_add(text: str) -> str:
    task_id = str(uuid.uuid4())
    settings.task_data.taskData["tasks_todo"][task_id] = text
    settings.task_data.save_data()
    return task_id


def _direct_complete(task_id: str) -> int:
    todo = settings.task_data.taskData["tasks_todo"]
    done = settings.task_data.taskData["tasks_done"]
    text = todo.pop(task_id, None)
    if text is None:
        return 0
    done[task_id] = text
    settings.task_data.taskData["n_tasks"] = (
        int(settings.task_data.taskData.get("n_tasks", 0) or 0) + 1
    )
    settings.task_data.save_data()
    reward = int(settings.SINGLETASK_REWARD)
    if settings.task_data.taskData["n_tasks"] % 5 == 0:
        reward += int(settings.FIVETASK_REWARD)
    settings.pet_data.change_coin(reward)
    return reward


def _direct_delete(task_id: str) -> Optional[str]:
    for key in ("tasks_todo", "tasks_done"):
        bucket = settings.task_data.taskData[key]
        if task_id in bucket:
            text = bucket.pop(task_id)
            settings.task_data.save_data()
            return str(text)
    return None


def execute_action(action: Dict[str, Any], task_panel: Any = None) -> Dict[str, Any]:
    """Run one task action, preferring the live dashboard panel when present."""

    kind = action.get("kind")
    tasks = settings.task_data.taskData
    if kind == "add_task":
        text = str(action.get("text", "")).strip()
        if not text:
            return {"ok": False, "reason": "empty"}
        if task_panel is not None and hasattr(task_panel, "add_task_external"):
            task_id = task_panel.add_task_external(text)
        else:
            task_id = _direct_add(text)
            bridge.tasks_changed.emit()
        return {"ok": True, "kind": kind, "task": text, "id": task_id}

    items = task_items({"tasks": {"todo": tasks.get("tasks_todo", {})}})
    item = find_task(items, str(action.get("query", "")))
    if item is None:
        return {"ok": False, "reason": "not_found", "query": action.get("query", "")}
    task_id = item["id"]
    text = item["text"]

    if kind == "complete_task":
        if task_id in tasks.get("tasks_done", {}):
            return {"ok": True, "kind": kind, "task": text, "reward": 0, "already": True}
        if task_panel is not None and hasattr(task_panel, "complete_task_external"):
            reward = task_panel.complete_task_external(task_id)
        else:
            reward = _direct_complete(task_id)
            bridge.tasks_changed.emit()
        return {"ok": True, "kind": kind, "task": text, "reward": int(reward or 0)}

    if kind == "delete_task":
        if task_panel is not None and hasattr(task_panel, "delete_task_external"):
            removed = task_panel.delete_task_external(task_id, False)
            text = removed or text
        else:
            _direct_delete(task_id)
            bridge.tasks_changed.emit()
        return {"ok": True, "kind": kind, "task": text}

    return {"ok": False, "reason": "unknown"}
