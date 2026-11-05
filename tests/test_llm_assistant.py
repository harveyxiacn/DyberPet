# -*- coding: utf-8 -*-
"""Unit tests for the pure assistant core (no Qt, no network)."""

import json
import unittest

from DyberPet.LLM.assistant import (
    BUILTIN_ONLY_INTENTS,
    answer_question,
    build_context,
    find_task,
    task_items,
)


def snapshot(**overrides):
    data = {
        "pet": {
            "name": "Mochi",
            "hp": 73,
            "hp_tier": 2,
            "favorability": 12,
            "favorability_level": 1,
            "coins": 320,
            "days": 4,
        },
        "tasks": {
            "todo": [
                {"id": "a", "text": "write the report"},
                {"id": "b", "text": "buy milk"},
            ],
            "done_count": 3,
            "completed_total": 11,
        },
        "progress": {"today_minutes": 45, "goal_minutes": 180, "streak_days": 2},
    }
    for key, value in overrides.items():
        data[key] = value
    return data


class AssistantTest(unittest.TestCase):
    def test_status_uses_snapshot_numbers(self):
        reply = answer_question("how are you?", snapshot(), "en")
        self.assertEqual(reply.intent, "status")
        self.assertIn("Mochi", reply.text)
        self.assertIn("satiety 73/100", reply.text)
        self.assertIn("320 coins", reply.text)

    def test_tasks_list_is_numbered_and_actionable(self):
        reply = answer_question("show my tasks", snapshot(), "en")
        self.assertEqual(reply.intent, "tasks")
        self.assertIn("1. write the report", reply.text)
        self.assertIn("2. buy milk", reply.text)
        self.assertIn("complete 1", reply.text)

    def test_add_task_parses_english_and_chinese(self):
        en = answer_question("add task write the weekly report", snapshot(), "en")
        self.assertEqual(en.intent, "add_task")
        self.assertEqual(en.action["kind"], "add_task")
        self.assertEqual(en.action["text"], "write the weekly report")

        zh = answer_question("帮我添加一个任务：写周报", snapshot(), "zh")
        self.assertEqual(zh.intent, "add_task")
        self.assertEqual(zh.action["text"], "写周报")

    def test_remind_me_and_todo_shorthand(self):
        self.assertEqual(
            answer_question("remind me to stretch", snapshot(), "en").action["text"],
            "stretch",
        )
        self.assertEqual(
            answer_question("todo: call mom", snapshot(), "en").action["text"],
            "call mom",
        )

    def test_complete_and_delete_actions(self):
        done = answer_question("complete 2", snapshot(), "en")
        self.assertEqual(done.intent, "complete_task")
        self.assertEqual(done.action["query"], "2")

        done_zh = answer_question("完成写报告", snapshot(), "zh")
        self.assertEqual(done_zh.intent, "complete_task")
        self.assertEqual(done_zh.action["query"], "写报告")

        removed = answer_question("delete task buy milk", snapshot(), "en")
        self.assertEqual(removed.intent, "delete_task")
        self.assertEqual(removed.action["query"], "buy milk")

    def test_completed_suffix_is_recognised(self):
        reply = answer_question("buy milk is done", snapshot(), "en")
        self.assertEqual(reply.intent, "complete_task")
        self.assertEqual(reply.action["query"], "buy milk")

    def test_progress_and_coins(self):
        progress = answer_question("how is my progress?", snapshot(), "en")
        self.assertIn("45/180", progress.text)
        self.assertIn("streak 2 days", progress.text)
        coins = answer_question("how many coins do I have?", snapshot(), "en")
        self.assertEqual(coins.intent, "coins")
        self.assertIn("320", coins.text)

    def test_summary_and_help(self):
        summary = answer_question("summarise my day", snapshot(), "en")
        self.assertEqual(summary.intent, "summary")
        self.assertIn("45/180", summary.text)
        self.assertIn("write the report", summary.text)

        help_reply = answer_question("what can you do?", snapshot(), "en")
        self.assertEqual(help_reply.intent, "help")
        self.assertIn("to-do", help_reply.text)
        self.assertIn("local model", help_reply.text)

    def test_advice_prioritises_a_hungry_pet(self):
        hungry = snapshot(pet={"name": "Mochi", "hp_tier": 0, "hp": 5, "coins": 1})
        reply = answer_question("what should I do?", hungry, "en")
        self.assertEqual(reply.intent, "advice")
        self.assertIn("hungry", reply.text)

    def test_chinese_questions(self):
        self.assertEqual(answer_question("你好", snapshot(), "zh").intent, "greeting")
        self.assertEqual(answer_question("今天进度怎么样？", snapshot(), "zh").intent, "progress")
        self.assertEqual(answer_question("查看任务", snapshot(), "zh").intent, "tasks")
        self.assertEqual(answer_question("谢谢", snapshot(), "zh").intent, "thanks")

    def test_language_is_detected_when_not_given(self):
        reply = answer_question("你好", snapshot())
        self.assertIn("嗨", reply.text)
        self.assertEqual(reply.intent, "greeting")

    def test_empty_task_list_does_not_invent_one(self):
        empty = snapshot(tasks={"todo": [], "done_count": 0})
        reply = answer_question("show my tasks", empty, "en")
        self.assertIn("no to-do tasks", reply.text)

    def test_unknown_is_honest_and_offers_help(self):
        reply = answer_question("what is the meaning of life?", snapshot(), "en")
        self.assertEqual(reply.intent, "unknown")
        self.assertIn("not sure", reply.text)
        self.assertIn("show my tasks", reply.text)

    def test_action_intents_are_builtin_only(self):
        for intent in ("add_task", "complete_task", "delete_task", "help", "greeting", "thanks"):
            self.assertIn(intent, BUILTIN_ONLY_INTENTS)

    def test_snapshot_is_not_mutated(self):
        data = snapshot()
        before = json.loads(json.dumps(data))
        answer_question("complete 1", data, "en")
        self.assertEqual(data, before)

    def test_input_is_clamped(self):
        reply = answer_question("a" * 2000, snapshot(), "en")
        self.assertEqual(reply.intent, "unknown")
        self.assertLessEqual(len(reply.text), 1000)


class TaskHelpersTest(unittest.TestCase):
    def test_mapping_and_list_forms(self):
        mapping = snapshot(tasks={"todo": {"id-1": "one", "id-2": "two"}, "done_count": 0})
        self.assertEqual(
            task_items(mapping),
            [{"id": "id-1", "text": "one"}, {"id": "id-2", "text": "two"}],
        )
        listed = snapshot(tasks={"todo": ["bare"], "done_count": 0})
        self.assertEqual(task_items(listed), [{"id": "", "text": "bare"}])

    def test_find_by_number_text_and_part(self):
        items = task_items(snapshot())
        self.assertEqual(find_task(items, "1")["text"], "write the report")
        self.assertEqual(find_task(items, "buy milk")["text"], "buy milk")
        self.assertEqual(find_task(items, "REPORT")["text"], "write the report")
        self.assertIsNone(find_task(items, "9"))
        self.assertIsNone(find_task(items, "nothing here"))


class ContextTest(unittest.TestCase):
    def test_context_is_small_and_free_of_ids_and_identity(self):
        context = build_context(snapshot(), now="2026-11-06T12:00")
        payload = json.loads(context)
        self.assertEqual(payload["pet"]["name"], "Mochi")
        self.assertEqual(payload["tasks"]["todo"][0], "write the report")
        self.assertNotIn('"id"', context)
        self.assertNotIn("a@b.com", context)
        self.assertNotIn("/home/", context)
        self.assertNotIn("C:\\", context)

    def test_context_caps_history(self):
        history = [
            {"role": "user", "text": "x" * 1000},
            {"role": "assistant", "text": "y" * 1000},
        ] * 5
        context = build_context(snapshot(), history=history)
        payload = json.loads(context)
        self.assertLessEqual(len(payload["conversation"]), 6)
        self.assertLessEqual(len(payload["conversation"][0]["text"]), 400)
        self.assertLessEqual(len(context), 12000)


if __name__ == "__main__":
    unittest.main()
