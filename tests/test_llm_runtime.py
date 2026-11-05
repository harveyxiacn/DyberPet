# -*- coding: utf-8 -*-
"""Tests for the runtime glue, with the game state replaced by fakes."""

import unittest

import DyberPet.settings as settings
from DyberPet.LLM.runtime import (
    LlmConfig,
    collect_snapshot,
    execute_action,
    load_llm_config,
    save_llm_config,
)


class FakePet:
    def __init__(self, hp=73, coins=0):
        self.hp = hp
        self.hp_tier = 2
        self.fv = 12
        self.fv_lvl = 1
        self.coins = coins
        self.days = 4

    def change_coin(self, amount):
        self.coins += amount
        return self.coins


class FakeTaskData:
    def __init__(self, todo=None, n_tasks=0):
        self.taskData = {
            "tasks_todo": dict(todo or {}),
            "tasks_done": {},
            "n_tasks": n_tasks,
            "goal": 180,
            "n_days": 2,
            "history": [["2026-11-05", 45]],
        }
        self.saved = 0

    def save_data(self):
        self.saved += 1


class FakePanel:
    def __init__(self):
        self.calls = []

    def add_task_external(self, text):
        self.calls.append(("add", text))
        return "panel-id"

    def complete_task_external(self, task_id):
        self.calls.append(("complete", task_id))
        return 250

    def delete_task_external(self, task_id, done):
        self.calls.append(("delete", task_id, done))
        return "write the report"


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self._pet = getattr(settings, "pet_data", None)
        self._tasks = getattr(settings, "task_data", None)
        self._save = getattr(settings, "save_settings", None)
        self._name = getattr(settings, "petname", "")
        settings.pet_data = FakePet()
        settings.task_data = FakeTaskData({"a": "write the report", "b": "buy milk"})
        settings.petname = "Mochi"
        settings.save_settings = lambda: None

    def tearDown(self):
        settings.pet_data = self._pet
        settings.task_data = self._tasks
        settings.save_settings = self._save
        settings.petname = self._name

    def test_snapshot_reads_game_state(self):
        snapshot = collect_snapshot()
        self.assertEqual(snapshot["pet"]["name"], "Mochi")
        self.assertEqual(snapshot["tasks"]["todo"][0]["text"], "write the report")
        self.assertEqual(snapshot["progress"]["today_minutes"], 45)
        self.assertEqual(snapshot["tasks"]["todo"][0]["id"], "a")

    def test_add_uses_panel_and_skips_direct_write(self):
        panel = FakePanel()
        outcome = execute_action({"kind": "add_task", "text": "call mom"}, panel)
        self.assertTrue(outcome["ok"])
        self.assertEqual(panel.calls, [("add", "call mom")])
        self.assertNotIn("panel-id", settings.task_data.taskData["tasks_todo"])

    def test_add_without_panel_writes_and_saves(self):
        outcome = execute_action({"kind": "add_task", "text": "call mom"})
        self.assertTrue(outcome["ok"])
        self.assertIn("call mom", settings.task_data.taskData["tasks_todo"].values())
        self.assertEqual(settings.task_data.saved, 1)

    def test_complete_uses_panel_return_value(self):
        panel = FakePanel()
        outcome = execute_action({"kind": "complete_task", "query": "1"}, panel)
        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["task"], "write the report")
        self.assertEqual(outcome["reward"], 250)
        self.assertEqual(panel.calls, [("complete", "a")])

    def test_complete_without_panel_moves_task_and_rewards(self):
        outcome = execute_action({"kind": "complete_task", "query": "report"})
        self.assertEqual(outcome["reward"], 200)
        data = settings.task_data.taskData
        self.assertNotIn("a", data["tasks_todo"])
        self.assertIn("a", data["tasks_done"])
        self.assertEqual(settings.pet_data.coins, 200)

    def test_fifth_task_adds_the_bonus(self):
        settings.task_data = FakeTaskData({"a": "write the report"}, n_tasks=4)
        outcome = execute_action({"kind": "complete_task", "query": "1"})
        self.assertEqual(outcome["reward"], 200 + 1500)

    def test_delete_returns_the_removed_text(self):
        panel = FakePanel()
        outcome = execute_action({"kind": "delete_task", "query": "buy milk"}, panel)
        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["task"], "write the report")
        self.assertEqual(panel.calls, [("delete", "b", False)])

    def test_delete_without_panel_removes_and_saves(self):
        outcome = execute_action({"kind": "delete_task", "query": "buy milk"})
        self.assertTrue(outcome["ok"])
        self.assertNotIn("b", settings.task_data.taskData["tasks_todo"])
        self.assertEqual(settings.task_data.saved, 1)

    def test_unknown_task_is_a_normal_result(self):
        outcome = execute_action({"kind": "complete_task", "query": "nothing"})
        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["reason"], "not_found")

    def test_load_config_has_safe_defaults(self):
        for name in ("llm_enabled", "llm_base_url", "llm_model"):
            if hasattr(settings, name):
                delattr(settings, name)
        config = load_llm_config()
        self.assertFalse(config.enabled)
        self.assertEqual(config.base_url, "http://127.0.0.1:11434/v1")
        self.assertEqual(config.model, "")

    def test_save_config_updates_settings_and_persists(self):
        saved = []
        settings.save_settings = lambda: saved.append(True)
        save_llm_config(LlmConfig(True, "http://127.0.0.1:1234/v1", "qwen3:1.7b"))
        self.assertTrue(settings.llm_enabled)
        self.assertEqual(settings.llm_base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(settings.llm_model, "qwen3:1.7b")
        self.assertEqual(saved, [True])


if __name__ == "__main__":
    unittest.main()
