# -*- coding: utf-8 -*-
"""Deterministic assistant for the DyberPet desktop pet.

This module is intentionally free of Qt, network and file access so it can be
unit-tested and used from any front end. It turns a question plus a small,
credential-free snapshot of the game state into an answer, and optionally into
one task action that the caller executes.

The built-in answers always work. A local language model is an optional
rephrasing layer; see ``local_client.py``.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Dict, List, Optional

MAX_QUESTION_CHARS = 500
MAX_CONTEXT_CHARS = 12000
MAX_TASK_CHARS = 120
MAX_LISTED_TASKS = 8
MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 400

# Intents that must stay deterministic: an action cannot be invented by a
# model, and greetings/help should be instant.
BUILTIN_ONLY_INTENTS = frozenset(
    {"action", "greeting", "thanks", "help", "add_task", "complete_task", "delete_task"}
)

_CJK_RE = re.compile(r"[㐀-鿿]")
_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT = " \t\r\n.,;:!?，。；：！？、"


@dataclass
class AssistantReply:
    """One answer plus, for task intents, an action for the caller to run."""

    text: str
    intent: str
    action: Optional[Dict[str, Any]] = None
    suggestions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Small helpers


def _clean_question(question: str) -> str:
    text = _SPACE_RE.sub(" ", (question or "").strip())
    return text[:MAX_QUESTION_CHARS]


def _clean_task(text: str) -> str:
    cleaned = (text or "").strip().strip(_TRAILING_PUNCT)
    cleaned = re.sub(r"^(?:to|that|about)\s+", "", cleaned, flags=re.I)
    return cleaned[:MAX_TASK_CHARS].strip()


def _language(question: str, language: Optional[str]) -> str:
    if language:
        return "zh" if str(language).lower().startswith("zh") else "en"
    return "zh" if _CJK_RE.search(question or "") else "en"


def _is_zh(language: str) -> bool:
    return language == "zh"


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pet(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    pet = snapshot.get("pet") if isinstance(snapshot, dict) else None
    return pet if isinstance(pet, dict) else {}


def _progress(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    progress = snapshot.get("progress") if isinstance(snapshot, dict) else None
    return progress if isinstance(progress, dict) else {}


def _tasks(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    tasks = snapshot.get("tasks") if isinstance(snapshot, dict) else None
    return tasks if isinstance(tasks, dict) else {}


def task_items(snapshot: Dict[str, Any]) -> List[Dict[str, str]]:
    """Return the pending tasks as ``[{'id': ..., 'text': ...}]``.

    Accepts both the canonical list form and a mapping, so a caller can pass
    ``settings.task_data.taskData['tasks_todo']`` directly.
    """

    raw = _tasks(snapshot).get("todo", [])
    items: List[Dict[str, str]] = []
    if isinstance(raw, dict):
        for task_id, text in raw.items():
            items.append({"id": str(task_id), "text": str(text)})
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                items.append(
                    {
                        "id": str(entry.get("id", "")),
                        "text": str(entry.get("text", "")),
                    }
                )
            else:
                items.append({"id": "", "text": str(entry)})
    return [item for item in items if item["text"].strip()]


def find_task(
    items: List[Dict[str, str]], query: str
) -> Optional[Dict[str, str]]:
    """Find a task by 1-based number, exact text, or a case-insensitive part."""

    q = (query or "").strip().strip(_TRAILING_PUNCT)
    if not q:
        return None
    if q.isdigit():
        index = int(q) - 1
        if 0 <= index < len(items):
            return items[index]
        return None

    needle = q.lower()
    for item in items:
        if item["text"].strip().lower() == needle:
            return item
    for item in items:
        if needle in item["text"].lower():
            return item
    return None


def _hp_word(tier: int, zh: bool) -> str:
    words_en = ["starving", "hungry", "doing fine", "full of energy"]
    words_zh = ["饿坏了", "有点饿", "状态正常", "精力充沛"]
    index = max(0, min(3, tier))
    return words_zh[index] if zh else words_en[index]


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


# ---------------------------------------------------------------------------
# Context for an optional local model


def build_context(
    snapshot: Dict[str, Any],
    history: Optional[List[Dict[str, str]]] = None,
    now: Optional[str] = None,
) -> str:
    """Build a compact, identity-free JSON context.

    Task ids and file paths are deliberately omitted: the model only needs the
    text. History is the visible chat, never logs or prompts from other apps.
    """

    pet = _pet(snapshot)
    progress = _progress(snapshot)
    tasks = _tasks(snapshot)
    todo = [item["text"][:MAX_TASK_CHARS] for item in task_items(snapshot)][:20]
    done = tasks.get("done", [])
    done_count = len(done) if isinstance(done, (list, dict)) else _as_int(tasks.get("done_count"))

    payload: Dict[str, Any] = {
        "now": now,
        "pet": {
            "name": str(pet.get("name", ""))[:40],
            "satiety": _as_int(pet.get("hp")),
            "satiety_tier": _as_int(pet.get("hp_tier")),
            "favorability": _as_int(pet.get("favorability", pet.get("fv"))),
            "favorability_level": _as_int(pet.get("favorability_level", pet.get("fv_lvl"))),
            "coins": _as_int(pet.get("coins")),
            "days_together": _as_int(pet.get("days")),
        },
        "tasks": {"todo": todo, "done_count": done_count},
        "progress": {
            "today_minutes": _as_int(progress.get("today_minutes")),
            "goal_minutes": _as_int(progress.get("goal_minutes")),
            "streak_days": _as_int(progress.get("streak_days")),
        },
    }
    if history:
        cleaned: List[Dict[str, str]] = []
        for message in history[-MAX_HISTORY_MESSAGES:]:
            role = "assistant" if message.get("role") == "assistant" else "user"
            text = _SPACE_RE.sub(" ", str(message.get("text", "")).strip())
            if text:
                cleaned.append({"role": role, "text": text[:MAX_HISTORY_CHARS]})
        payload["conversation"] = cleaned
    encoded = json.dumps(payload, ensure_ascii=False)
    return encoded[:MAX_CONTEXT_CHARS]


# ---------------------------------------------------------------------------
# Answer text builders


def task_added_text(task: str, language: str) -> str:
    return (
        f"已添加待办：{task}"
        if _is_zh(language)
        else f"Added to your to-do list: {task}"
    )


def task_completed_text(task: str, reward: int, language: str) -> str:
    if _is_zh(language):
        return f"已完成：{task}，获得 {reward} 金币。"
    return f"Completed: {task}. You earned {reward} coins."


def task_deleted_text(task: str, language: str) -> str:
    return f"已删除：{task}" if _is_zh(language) else f"Deleted: {task}"


def task_not_found_text(query: str, language: str) -> str:
    if _is_zh(language):
        return f"没有找到和“{query}”匹配的待办。可以说“查看任务”看看列表。"
    return f'I could not find a task matching "{query}". Ask "show my tasks" to see the list.'


def _status_text(snapshot: Dict[str, Any], language: str) -> str:
    pet = _pet(snapshot)
    zh = _is_zh(language)
    name = str(pet.get("name") or ("桌宠" if zh else "your pet"))[:40]
    tier = _as_int(pet.get("hp_tier"), 2)
    hp = _as_int(pet.get("hp"), 0)
    fv = _as_int(pet.get("favorability", pet.get("fv")))
    fv_lvl = _as_int(pet.get("favorability_level", pet.get("fv_lvl")))
    coins = _as_int(pet.get("coins"))
    days = _as_int(pet.get("days"))
    if zh:
        return (
            f"{name}现在{_hp_word(tier, True)}（饱腹度 {hp}/100），"
            f"好感度 {fv}（Lv.{fv_lvl}），有 {coins} 金币，已经陪你 {days} 天。"
        )
    return (
        f"{name} is {_hp_word(tier, False)} (satiety {hp}/100), "
        f"favorability {fv} (Lv. {fv_lvl}), {coins} coins, "
        f"{days} {_plural(days, 'day', 'days')} together."
    )


def _task_list_text(snapshot: Dict[str, Any], language: str) -> str:
    items = task_items(snapshot)
    zh = _is_zh(language)
    if not items:
        return (
            "现在没有待办任务。可以说“添加任务 写报告”来加一个。"
            if zh
            else 'You have no to-do tasks. Say "add task write the report" to add one.'
        )
    lines = [
        f"{index}. {item['text']}"
        for index, item in enumerate(items[:MAX_LISTED_TASKS], start=1)
    ]
    more = len(items) - MAX_LISTED_TASKS
    header = (
        f"你有 {len(items)} 个待办："
        if zh
        else f"You have {len(items)} {_plural(len(items), 'task', 'tasks')}:"
    )
    footer = (
        "说“完成 1”或“完成 写报告”即可勾选。"
        if zh
        else 'Say "complete 1" or "complete write the report" to finish one.'
    )
    body = "\n".join(lines)
    if more > 0:
        body += ("\n还有 %d 个未显示。" if zh else "\n…and %d more.") % more
    return f"{header}\n{body}\n{footer}"


def _progress_text(snapshot: Dict[str, Any], language: str) -> str:
    progress = _progress(snapshot)
    today = _as_int(progress.get("today_minutes"))
    goal = _as_int(progress.get("goal_minutes"))
    streak = _as_int(progress.get("streak_days"))
    zh = _is_zh(language)
    if goal > 0:
        percent = max(0, min(100, round(today * 100 / goal)))
        if zh:
            state = "已达成" if today >= goal else f"还差 {goal - today} 分钟"
            return (
                f"今天专注了 {today} 分钟，目标 {goal} 分钟（{percent}%），{state}；"
                f"已连续 {streak} 天。"
            )
        state = "goal reached" if today >= goal else f"{goal - today} minutes to go"
        return (
            f"Today: {today}/{goal} focus minutes ({percent}%), {state}; "
            f"streak {streak} {_plural(streak, 'day', 'days')}."
        )
    return (
        f"今天专注了 {today} 分钟，已连续 {streak} 天。"
        if zh
        else f"Today: {today} focus minutes; streak {streak} {_plural(streak, 'day', 'days')}."
    )


def _coins_text(snapshot: Dict[str, Any], language: str) -> str:
    coins = _as_int(_pet(snapshot).get("coins"))
    return f"你现在有 {coins} 金币。" if _is_zh(language) else f"You have {coins} coins."


def _summary_text(snapshot: Dict[str, Any], language: str) -> str:
    items = task_items(snapshot)
    zh = _is_zh(language)
    parts = [_status_text(snapshot, language), _progress_text(snapshot, language)]
    if items:
        first = (
            "；".join(item["text"] for item in items[:3])
            if zh
            else "; ".join(item["text"] for item in items[:3])
        )
        parts.append(
            f"待办 {len(items)} 个，最前面是：{first}。"
            if zh
            else f"{len(items)} to-do {_plural(len(items), 'task', 'tasks')}, starting with: {first}."
        )
    else:
        parts.append("待办清单是空的。" if zh else "Your to-do list is empty.")
    return "\n".join(parts)


def _advice_text(snapshot: Dict[str, Any], language: str) -> str:
    pet = _pet(snapshot)
    progress = _progress(snapshot)
    items = task_items(snapshot)
    zh = _is_zh(language)
    tier = _as_int(pet.get("hp_tier"), 2)
    today = _as_int(progress.get("today_minutes"))
    goal = _as_int(progress.get("goal_minutes"))
    if tier <= 1:
        return (
            "它有点饿了，先喂点东西吧。打开仪表盘的任务页或背包即可。"
            if zh
            else "Your pet is hungry — feed it something first. Open the dashboard's task or inventory page."
        )
    if goal > 0 and today < goal:
        remaining = goal - today
        return (
            f"今天还差 {remaining} 分钟专注目标。可以先挑一个待办，开一轮番茄钟。"
            if zh
            else f"You are {remaining} minutes short of today's focus goal. Pick one to-do and start a pomodoro."
        )
    if items:
        return (
            f"状态不错。下一步可以完成“{items[0]['text']}”，或者陪它玩一会儿。"
            if zh
            else f"Things look good. Next, finish “{items[0]['text']}” or spend a moment playing with your pet."
        )
    return (
        "一切正常。可以加一个今天想完成的小任务，或者摸摸它。"
        if zh
        else "Everything looks fine. Add one small task for today, or give your pet a pat."
    )


def _help_text(language: str) -> str:
    if _is_zh(language):
        return (
            "我可以帮你：\n"
            "• 查看待办、添加任务、完成或删除任务\n"
            "• 查看今日专注进度、连续天数和金币\n"
            "• 查看桌宠的饱腹度、好感度和状态\n"
            "• 给出下一步建议\n"
            "例如：“添加任务 写周报”“完成 1”“今天进度怎么样？”\n"
            "内置回答不需要联网；也可以在设置里接一个本机小模型来闲聊。"
        )
    return (
        "I can help with:\n"
        "• Listing, adding, completing and deleting to-do tasks\n"
        "• Today's focus progress, streak and coins\n"
        "• Your pet's satiety, favorability and mood\n"
        "• A suggestion for what to do next\n"
        'Try "add task write the weekly report", "complete 1", or "how is my progress?"\n'
        "Built-in answers need no network. You can optionally connect a local model for free-form chat in Settings."
    )


def _suggestions(intent: str, language: str) -> List[str]:
    zh = [
        "查看任务",
        "今天进度怎么样？",
        "你怎么样？",
        "添加任务 喝水",
    ]
    en = [
        "show my tasks",
        "how is my progress?",
        "how are you?",
        "add task drink water",
    ]
    base = zh if _is_zh(language) else en
    if intent in ("tasks", "add_task", "complete_task", "delete_task"):
        return base[:3]
    return base


# ---------------------------------------------------------------------------
# Intent parsing


_EN_ADD = re.compile(
    r"^(?:please\s+)?(?:add|create|new)\s+(?:a\s+|an\s+)?(?:new\s+)?(?:task|to-?do)\s*(?:to|:|-)?\s*(.+)$",
    re.I,
)
_EN_REMIND = re.compile(r"^remind\s+me\s+to\s+(.+)$", re.I)
_EN_TODO = re.compile(r"^(?:to-?do|task)\s*[:\-]\s*(.+)$", re.I)
_EN_COMPLETE = re.compile(
    r"^(?:please\s+)?(?:complete|finish|done|mark(?:\s+as)?(?:\s+done)?)\s+(?:task\s+)?(.+)$",
    re.I,
)
_EN_COMPLETE_SUFFIX = re.compile(
    r"^(?:task\s+)?(.+?)\s+(?:is\s+)?(?:done|completed|finished)$", re.I
)
_EN_DELETE = re.compile(
    r"^(?:please\s+)?(?:delete|remove|cancel|drop)\s+(?:task\s+)?(.+)$", re.I
)

_ZH_ADD = re.compile(
    r"^(?:请|麻烦)?(?:帮我)?(?:添加|加|新建|新增|记一下|记住|提醒我)(?:一个|个)?(?:新的)?(?:待办|任务)?\s*[：:，,]?\s*(.+)$"
)
_ZH_COMPLETE = re.compile(
    r"^(?:请|帮我)?(?:完成|做完|搞定|勾选|标记完成)(?:任务)?\s*[：:，,]?\s*(.+)$"
)
_ZH_COMPLETE_SUFFIX = re.compile(r"^(.+?)(?:已经)?(?:完成了|做完了|搞定了)$")
_ZH_DELETE = re.compile(
    r"^(?:请|帮我)?(?:删除|删掉|移除|取消)(?:任务)?\s*[：:，,]?\s*(.+)$"
)


def _match_action(question: str, language: str) -> Optional[Dict[str, Any]]:
    text = question.strip()
    if _is_zh(language):
        for pattern, kind in (
            (_ZH_ADD, "add_task"),
            (_ZH_COMPLETE, "complete_task"),
            (_ZH_COMPLETE_SUFFIX, "complete_task"),
            (_ZH_DELETE, "delete_task"),
        ):
            match = pattern.match(text)
            if match:
                value = _clean_task(match.group(1))
                if value:
                    return {"kind": kind, "query": value, "text": value}
        return None

    for pattern, kind in (
        (_EN_ADD, "add_task"),
        (_EN_REMIND, "add_task"),
        (_EN_TODO, "add_task"),
        (_EN_COMPLETE, "complete_task"),
        (_EN_COMPLETE_SUFFIX, "complete_task"),
        (_EN_DELETE, "delete_task"),
    ):
        match = pattern.match(text)
        if match:
            value = _clean_task(match.group(1))
            if value:
                return {"kind": kind, "query": value, "text": value}
    return None


def _has(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _detect_intent(question: str, language: str) -> str:
    text = question.strip()
    lower = text.lower()
    action = _match_action(text, language)
    if action:
        return action["kind"]

    if _is_zh(language):
        if re.match(r"^(你好|您好|嗨|哈喽|早上好|晚上好|早安|晚安|在吗)", text):
            return "greeting"
        if _has(text, "谢谢", "多谢", "感谢", "辛苦了"):
            return "thanks"
        if _has(text, "你能做什么", "你会什么", "帮助", "怎么用", "有什么功能", "命令", "使用说明"):
            return "help"
        if _has(text, "总结", "汇总", "概况", "报告", "简报"):
            return "summary"
        if _has(text, "待办", "任务", "清单"):
            return "tasks"
        if _has(text, "进度", "目标", "专注", "连续", "今天"):
            return "progress"
        if _has(text, "金币", "硬币", "余额", "多少钱", "钱"):
            return "coins"
        if _has(text, "建议", "怎么办", "该做什么", "帮我安排", "下一步"):
            return "advice"
        if _has(text, "状态", "你好吗", "怎么样", "饿", "累", "好感", "等级", "血量", "生命", "心情"):
            return "status"
        return "unknown"

    if re.match(r"^(hi|hello|hey|yo|good\s+(morning|afternoon|evening))\b", lower):
        return "greeting"
    if re.search(r"\b(thanks|thank\s+you|thx|cheers)\b", lower):
        return "thanks"
    if re.search(r"\b(what\s+can\s+you\s+do|help|commands?|how\s+(do|can)\s+i\s+use)\b", lower):
        return "help"
    if re.search(r"\b(summary|summarise|summarize|overview|status\s+report|brief)\b", lower):
        return "summary"
    if re.search(r"\b(tasks?|to-?do)\b", lower):
        return "tasks"
    if re.search(r"\b(progress|goal|focus|streak|today)\b", lower):
        return "progress"
    if re.search(r"\b(coins?|money|balance)\b", lower):
        return "coins"
    if re.search(r"\b(advice|suggest|what\s+should\s+i\s+do|next\s+step|plan)\b", lower):
        return "advice"
    if re.search(
        r"\b(status|how\s+are\s+you|hp|health|satiety|hungry|tired|favorability|level|mood)\b",
        lower,
    ):
        return "status"
    return "unknown"


def answer_question(
    question: str,
    snapshot: Dict[str, Any],
    language: Optional[str] = None,
    now: Optional[str] = None,
) -> AssistantReply:
    """Answer one question from the built-in rules.

    ``snapshot`` is the output of ``runtime.collect_snapshot()``; it may also be
    assembled by hand in tests. No network or file access happens here.
    """

    cleaned = _clean_question(question)
    lang = _language(cleaned, language)
    if not cleaned:
        return AssistantReply(
            text=_help_text(lang), intent="help", suggestions=_suggestions("help", lang)
        )

    intent = _detect_intent(cleaned, lang)
    if intent == "greeting":
        name = str(_pet(snapshot).get("name") or ("桌宠" if lang == "zh" else "your pet"))[:40]
        text = (
            f"嗨，我是{name}。今天想先看任务、进度，还是陪我玩一会儿？"
            if lang == "zh"
            else f"Hi, I'm {name}. Want to look at tasks, progress, or just spend a moment together?"
        )
    elif intent == "thanks":
        text = "不客气，我会一直在这里。" if lang == "zh" else "You're welcome — I'm right here."
    elif intent == "help":
        text = _help_text(lang)
    elif intent == "summary":
        text = _summary_text(snapshot, lang)
    elif intent == "tasks":
        text = _task_list_text(snapshot, lang)
    elif intent == "progress":
        text = _progress_text(snapshot, lang)
    elif intent == "coins":
        text = _coins_text(snapshot, lang)
    elif intent == "advice":
        text = _advice_text(snapshot, lang)
    elif intent == "status":
        text = _status_text(snapshot, lang)
    else:
        if lang == "zh":
            text = (
                "这个问题我暂时没有把握。我可以帮你管理待办、查看专注进度、金币和桌宠状态。"
                "试试“查看任务”或“今天进度怎么样？”。"
            )
        else:
            text = (
                "I'm not sure about that one yet. I can manage to-dos, show focus progress, coins "
                'and your pet\'s status. Try "show my tasks" or "how is my progress?".'
            )

    action = _match_action(cleaned, lang)
    if intent in ("add_task", "complete_task", "delete_task"):
        assert action is not None
        if intent == "add_task":
            text = (
                f"准备添加待办：{action['text']}"
                if lang == "zh"
                else f"Adding to your to-do list: {action['text']}"
            )
        elif intent == "complete_task":
            text = (
                f"正在查找并完成：{action['query']}"
                if lang == "zh"
                else f"Looking for and completing: {action['query']}"
            )
        else:
            text = (
                f"正在删除：{action['query']}"
                if lang == "zh"
                else f"Deleting: {action['query']}"
            )
    else:
        action = None

    return AssistantReply(
        text=text,
        intent=intent,
        action=action,
        suggestions=_suggestions(intent, lang),
    )
