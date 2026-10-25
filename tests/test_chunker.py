import unittest

from texti.index.chunker import chunk_text, summarize
from texti.tokens import estimate_tokens


def sentences(n):
    return " ".join(f"Sentence number {i} talks about trees and search." for i in range(n))


class ChunkerTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("Hello world. Second sentence."), ["Hello world. Second sentence."])

    def test_empty_text_gives_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n  "), [])

    def test_chunks_respect_the_maximum(self):
        text = sentences(400)
        chunks = chunk_text(text, target_tokens=100, max_tokens=130, overlap_tokens=20)
        self.assertGreater(len(chunks), 3)
        for c in chunks:
            self.assertLessEqual(estimate_tokens(c), 130 + 1)

    def test_default_sizes_are_in_the_planned_range(self):
        chunks = chunk_text(sentences(600))
        self.assertGreater(len(chunks), 1)
        for c in chunks[:-1]:
            self.assertGreaterEqual(estimate_tokens(c), 300)
            self.assertLessEqual(estimate_tokens(c), 500)

    def test_overlap_carries_context(self):
        chunks = chunk_text(sentences(200), target_tokens=100, max_tokens=140, overlap_tokens=30)
        for a, b in zip(chunks, chunks[1:]):
            last_sentence_of_a = a.split(". ")[-1].strip()
            self.assertIn(last_sentence_of_a.rstrip("."), b)

    def test_no_text_is_lost(self):
        text = sentences(150)
        chunks = chunk_text(text, target_tokens=80, max_tokens=110, overlap_tokens=20)
        joined = " ".join(chunks)
        for i in range(150):
            self.assertIn(f"Sentence number {i} ", joined)

    def test_no_chunk_is_only_overlap(self):
        chunks = chunk_text(sentences(60), target_tokens=60, max_tokens=90, overlap_tokens=30)
        for a, b in zip(chunks, chunks[1:]):
            self.assertNotEqual(a, b)
            self.assertFalse(b in a)  # a later chunk always adds something new

    def test_one_giant_line_is_hard_split(self):
        text = "word " * 3000
        chunks = chunk_text(text, target_tokens=100, max_tokens=150, overlap_tokens=10)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertLessEqual(estimate_tokens(c), 150 + 1)

    def test_target_above_max_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("x", target_tokens=600, max_tokens=500)

    def test_telugu_text_is_chunked(self):
        telugu = " ".join("ఇది ఒక పరీక్ష వాక్యం." for _ in range(300))
        chunks = chunk_text(telugu, target_tokens=100, max_tokens=150, overlap_tokens=10)
        self.assertGreater(len(chunks), 1)

    def test_summary_is_first_meaningful_line(self):
        self.assertEqual(summarize("\n\nLecture 4: Trees\nbody text here"), "Lecture 4: Trees")

    def test_summary_is_trimmed(self):
        s = summarize("x" * 500, max_chars=50)
        self.assertLessEqual(len(s), 50)
        self.assertTrue(s.endswith("…"))


if __name__ == "__main__":
    unittest.main()
