from __future__ import annotations

import unittest

from alttez.reports.telegram_delivery import chunk_messages


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


if __name__ == "__main__":
    unittest.main()
