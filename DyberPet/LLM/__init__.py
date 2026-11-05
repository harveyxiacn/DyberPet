# -*- coding: utf-8 -*-
"""Local assistant for DyberPet: deterministic answers plus an optional LLM."""

from .assistant import (
    AssistantReply,
    BUILTIN_ONLY_INTENTS,
    answer_question,
    build_context,
    find_task,
    task_added_text,
    task_completed_text,
    task_deleted_text,
    task_items,
    task_not_found_text,
)
from .local_client import (
    DEFAULT_SERVERS,
    LocalLlmError,
    ProbeResult,
    chat,
    normalize_base_url,
    probe,
)

__all__ = [
    "AssistantReply",
    "BUILTIN_ONLY_INTENTS",
    "answer_question",
    "build_context",
    "find_task",
    "task_added_text",
    "task_completed_text",
    "task_deleted_text",
    "task_items",
    "task_not_found_text",
    "DEFAULT_SERVERS",
    "LocalLlmError",
    "ProbeResult",
    "chat",
    "normalize_base_url",
    "probe",
]
