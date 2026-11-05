# -*- coding: utf-8 -*-
"""Unit tests for the loopback-only local model client."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from DyberPet.LLM.local_client import (
    LocalLlmError,
    chat,
    models_from_json,
    normalize_base_url,
    probe,
    reply_from_json,
)


class _Handler(BaseHTTPRequestHandler):
    requests = []
    redirect_models = False

    def log_message(self, format, *args):
        return

    def _send(self, code, payload, headers=None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        type(self).requests.append(
            {"method": "GET", "path": self.path, "headers": dict(self.headers)}
        )
        if self.path.endswith("/v1/models"):
            if self.redirect_models:
                self.send_response(302)
                self.send_header("Location", "http://example.com/v1/models")
                self.end_headers()
                return
            self._send(200, {"object": "list", "data": [{"id": "qwen3:1.7b"}]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        type(self).requests.append(
            {
                "method": "POST",
                "path": self.path,
                "headers": dict(self.headers),
                "body": raw,
            }
        )
        if self.path.endswith("/v1/chat/completions"):
            self._send(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "You have 2 tasks and 320 coins.",
                                "reasoning": "private",
                            }
                        }
                    ]
                },
            )
            return
        self._send(404, {"error": "not found"})


class ClientTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = "http://127.0.0.1:%d/v1" % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self):
        _Handler.requests = []
        _Handler.redirect_models = False

    def test_loopback_urls_are_accepted_and_normalised(self):
        self.assertEqual(
            normalize_base_url("http://127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1",
        )
        self.assertEqual(
            normalize_base_url("http://localhost:1234/v1/"),
            "http://localhost:1234/v1",
        )
        self.assertEqual(
            normalize_base_url("https://[::1]:8443/v1"),
            "https://[::1]:8443/v1",
        )
        self.assertEqual(
            normalize_base_url("http://127.0.0.1:8080/chat/completions"),
            "http://127.0.0.1:8080/v1",
        )

    def test_remote_credentials_and_malformed_urls_are_rejected(self):
        for raw in (
            "http://192.168.1.4:11434",
            "https://example.com/v1",
            "http://user:pass@127.0.0.1:11434/v1",
            "file:///etc/passwd",
            "http://127.0.0.1:11434/v1?x=1",
            "http://127.0.0.1:11434/v1#frag",
            "http://localhost.evil.test/v1",
            "not a url",
            "",
        ):
            with self.assertRaises(LocalLlmError, msg=raw):
                normalize_base_url(raw)

    def test_probe_lists_models(self):
        result = probe(self.base, timeout=2)
        self.assertTrue(result.reachable)
        self.assertEqual(result.models, ["qwen3:1.7b"])
        self.assertIsNone(result.issue)

    def test_probe_reports_missing_server_without_raising(self):
        result = probe("http://127.0.0.1:9", timeout=0.5)
        self.assertFalse(result.reachable)
        self.assertIn(result.issue, ("not_running", "rejected"))

    def test_redirects_are_not_followed(self):
        _Handler.redirect_models = True
        result = probe(self.base, timeout=2)
        self.assertFalse(result.reachable)
        self.assertEqual(result.issue, "rejected")

    def test_chat_sends_no_credentials_and_keeps_qwen_headroom(self):
        text = chat(
            self.base,
            "qwen3:1.7b",
            "How is my day?",
            '{"tasks":{"todo":["write"]}}',
            "You have 1 task.",
        )
        self.assertEqual(text, "You have 2 tasks and 320 coins.")
        post = [r for r in _Handler.requests if r["method"] == "POST"][0]
        self.assertFalse(any(key.lower() == "authorization" for key in post["headers"]))
        body = json.loads(post["body"].decode("utf-8"))
        self.assertEqual(body["max_tokens"], 700)
        self.assertEqual(body["model"], "qwen3:1.7b")
        system = body["messages"][0]["content"]
        self.assertIn("/no_think", system)
        user = body["messages"][1]["content"]
        self.assertIn("REFERENCE ANSWER", user)
        self.assertIn("You have 1 task.", user)

    def test_models_null_is_an_empty_list(self):
        self.assertEqual(models_from_json(b'{"object":"list","data":null}'), [])
        self.assertEqual(models_from_json(b'{"models":null}'), [])
        self.assertEqual(
            models_from_json(b'{"data":[{"id":"a"},{"id":"a"},{"id":2}]}'),
            ["a"],
        )
        with self.assertRaises(LocalLlmError):
            models_from_json(b"not json")

    def test_reply_uses_first_non_empty_message_and_strips_reasoning(self):
        open_tag = "\x3cthink\x3e"
        close_tag = "\x3c/think\x3e"
        body = json.dumps(
            {
                "choices": [
                    {"message": {"content": "   "}},
                    {"message": {"content": open_tag + "private" + close_tag + " hello  "}},
                ]
            }
        ).encode("utf-8")
        self.assertEqual(reply_from_json(body), "hello")
        with self.assertRaises(LocalLlmError):
            reply_from_json(b'{"choices":[]}')

    def test_empty_question_is_rejected_before_network(self):
        with self.assertRaises(LocalLlmError):
            chat(self.base, "qwen3:1.7b", "   ", "{}")


if __name__ == "__main__":
    unittest.main()
