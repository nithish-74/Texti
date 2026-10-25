import unittest

import numpy as np

from texti.index.store import Candidate
from texti.retrieve.budget import estimate_raw_tokens, select_within_budget


def cand(chunk_id, file_id, level, tokens, offset=0, tier=1):
    return Candidate(tier, chunk_id, file_id, level, tokens, offset, "eng", 1.0, np.zeros(2, dtype=np.float32))


class BudgetTests(unittest.TestCase):
    def test_summaries_come_first_when_full_text_does_not_fit(self):
        scored = [(cand(1, "A", "summary", 10), 0.9), (cand(2, "A", "full", 500), 0.9),
                  (cand(3, "B", "summary", 10), 0.8), (cand(4, "B", "full", 500), 0.8)]
        out = select_within_budget(scored, budget=100)
        self.assertEqual([(c.file_id, c.level) for c, _ in out], [("A", "summary"), ("B", "summary")])

    def test_summary_is_dropped_when_full_text_is_included(self):
        scored = [(cand(1, "A", "summary", 10), 0.9), (cand(2, "A", "full", 100), 0.9)]
        out = select_within_budget(scored, budget=500)
        self.assertEqual([c.level for c, _ in out], ["full"])

    def test_never_exceeds_the_budget(self):
        scored = [(cand(i, f"F{i}", "full", 90), 0.5 + i / 100) for i in range(10)]
        out = select_within_budget(scored, budget=300, max_files=10)
        self.assertLessEqual(sum(c.token_count for c, _ in out), 300)
        self.assertEqual(len(out), 3)

    def test_best_scoring_chunks_are_chosen_first(self):
        scored = [(cand(1, "A", "full", 100), 0.3), (cand(2, "B", "full", 100), 0.9), (cand(3, "C", "full", 100), 0.6)]
        out = select_within_budget(scored, budget=200, max_files=5)
        self.assertEqual({c.file_id for c, _ in out}, {"B", "C"})

    def test_a_too_big_chunk_does_not_block_smaller_ones(self):
        scored = [(cand(1, "A", "full", 900), 0.9), (cand(2, "B", "full", 50), 0.5)]
        out = select_within_budget(scored, budget=100)
        self.assertEqual([c.file_id for c, _ in out], ["B"])

    def test_irrelevant_files_are_left_out(self):
        scored = [(cand(1, "A", "full", 10), 0.05), (cand(2, "B", "full", 10), 0.5)]
        out = select_within_budget(scored, budget=500, min_score=0.1)
        self.assertEqual([c.file_id for c, _ in out], ["B"])

    def test_max_files_is_respected(self):
        scored = [(cand(i, f"F{i}", "full", 5), 0.5) for i in range(10)]
        self.assertEqual(len({c.file_id for c, _ in select_within_budget(scored, 500, max_files=3)}), 3)

    def test_chunks_of_a_file_come_in_document_order(self):
        scored = [(cand(1, "A", "full", 10, offset=3), 0.9), (cand(2, "A", "full", 10, offset=1), 0.5)]
        out = select_within_budget(scored, budget=100)
        self.assertEqual([c.source_offset for c, _ in out], [1, 3])

    def test_same_file_id_in_different_tiers_is_kept_apart(self):
        scored = [(cand(1, "A", "full", 10, tier=1), 0.9), (cand(1, "A", "full", 10, tier=3), 0.8)]
        self.assertEqual(len(select_within_budget(scored, budget=100)), 2)

    def test_empty_input(self):
        self.assertEqual(select_within_budget([], 800), [])

    def test_raw_token_estimates(self):
        self.assertEqual(estimate_raw_tokens(".PNG", 1, 765), 765)
        self.assertEqual(estimate_raw_tokens(".pdf", 4, 765), 4 * 765)
        self.assertEqual(estimate_raw_tokens(".pdf", 0, 765), 765)
        self.assertEqual(estimate_raw_tokens(".txt", 1, 765), 0)


if __name__ == "__main__":
    unittest.main()
