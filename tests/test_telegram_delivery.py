from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import alttez.reports.telegram_delivery as td
from alttez.reports.daily_brief.delivery import _dispatch_daily_brief_telegram
from alttez.reports.telegram_delivery import chunk_messages, send_message


class _FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestChunkMessages(unittest.TestCase):
    def test_short_text_is_single_chunk(self) -> None:
        text = "hola\n\nmundo"
        self.assertEqual(chunk_messages(text), [text])

    def test_splits_on_section_boundaries_within_limit(self) -> None:
        blocks = [f"<b>Bloque {i}</b>\n" + ("x" * 200) for i in range(20)]
        text = "\n\n".join(blocks)
        chunks = chunk_messages(text, limit=900)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 900)
        # No content lost (modulo the join separators).
        rejoined = "".join(chunk.replace("\n", "") for chunk in chunks)
        original = text.replace("\n", "")
        self.assertEqual(rejoined, original)

    def test_oversized_single_line_is_hard_split(self) -> None:
        text = "y" * 5000
        chunks = chunk_messages(text, limit=1000)
        self.assertTrue(all(len(chunk) <= 1000 for chunk in chunks))
        self.assertEqual("".join(chunks), text)


class TestSendRetry(unittest.TestCase):
    def setUp(self) -> None:
        # No real backoff sleeps; keep tests instant.
        sleep_patch = mock.patch.object(td.time, "sleep", lambda *_: None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def test_retries_on_timeout_then_succeeds(self) -> None:
        calls = {"n": 0}

        def fake_urlopen(req, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise TimeoutError("read timed out")
            return _FakeResponse(200)

        with mock.patch.object(td.request, "urlopen", side_effect=fake_urlopen):
            ok, error_message = send_message("token", "chat", "hola")

        self.assertTrue(ok)
        self.assertIsNone(error_message)
        self.assertEqual(calls["n"], 2)  # one retry

    def test_gives_up_after_max_attempts(self) -> None:
        with mock.patch.object(td, "delivery_max_attempts", return_value=3):
            with mock.patch.object(td.request, "urlopen", side_effect=TimeoutError("nope")) as opener:
                ok, error_message = send_message("token", "chat", "hola")
        self.assertFalse(ok)
        self.assertIn("nope", error_message)
        self.assertEqual(opener.call_count, 3)

    def test_no_retry_on_permanent_4xx(self) -> None:
        with mock.patch.object(td.request, "urlopen", return_value=_FakeResponse(400)) as opener:
            ok, error_message = send_message("token", "chat", "hola")
        self.assertFalse(ok)
        self.assertEqual(opener.call_count, 1)  # 400 is permanent, no retry

    def test_daily_brief_retries_timeout_and_sends_attachments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            html_path = Path(tmp) / "brief.html"
            json_path = Path(tmp) / "brief.json"
            html_path.write_text("brief", encoding="utf-8")
            json_path.write_text("{}", encoding="utf-8")
            responses = [TimeoutError("timeout"), _FakeResponse(), _FakeResponse(), _FakeResponse()]
            with mock.patch("alttez.reports.daily_brief.delivery.telegram_config", return_value=("token", "chat")):
                with mock.patch("alttez.reports.daily_brief.delivery._build_telegram_summary_text", return_value="summary"):
                    with mock.patch.object(td.request, "urlopen", side_effect=responses) as opener:
                        result = _dispatch_daily_brief_telegram({"target_date": "2026-01-01"}, html_path, json_path)
        self.assertEqual(result, (True, None))
        self.assertEqual(opener.call_count, 4)


if __name__ == "__main__":
    unittest.main()
