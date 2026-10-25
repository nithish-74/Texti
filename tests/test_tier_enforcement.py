import re
import unittest
from pathlib import Path

from texti.index.store import ChunkRecord
from texti.retrieve.search import search
from tests import fixtures as fx
from tests.helpers import Env

SRC = Path(__file__).resolve().parent.parent / "src" / "texti"


def add(env, tier, file_id, text, level="full", path="x/y.png"):
    emb = env.embedder.embed([text])[0]
    env.store.add_chunks(tier, [ChunkRecord(file_id, level, text, path, emb, 20)])
    env.store.upsert_file(file_id, "h", tier, 765, 0.1, "done")


class TierEnforcementTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        add(self.env, 1, "f1", "binary search tree lecture notes about insertion")
        add(self.env, 2, "f2", "binary search tree personal diary entry about a trip")
        add(self.env, 3, "f3", "binary search tree secret bank statement details")

    def tearDown(self):
        self.env.close()

    def run_search(self, **kw):
        return search(self.env.store, self.env.embedder, "binary search tree", record_usage=False, **kw)

    def test_default_search_returns_only_tier_1(self):
        res = self.run_search()
        self.assertEqual({h.tier for h in res.hits}, {1})

    def test_tier_2_without_approver_is_skipped_with_a_note(self):
        res = self.run_search(tiers=(1, 2))
        self.assertEqual({h.tier for h in res.hits}, {1})
        self.assertTrue(any("Tier 2" in n and "approval" in n for n in res.notes))

    def test_tier_3_without_approver_is_skipped(self):
        res = self.run_search(tiers=(1, 3))
        self.assertEqual({h.tier for h in res.hits}, {1})
        self.assertTrue(any("Tier 3" in n for n in res.notes))

    def test_tier_2_is_returned_after_approval(self):
        seen = []

        def approver(hits):
            seen.extend(hits)
            return hits

        res = self.run_search(tiers=(1, 2), approver=approver)
        self.assertEqual({h.tier for h in res.hits}, {1, 2})
        self.assertTrue(seen and all(h.tier == 2 for h in seen))  # approver only sees gated hits

    def test_rejected_results_are_left_out(self):
        res = self.run_search(tiers=(1, 2, 3), approver=lambda hits: [])
        self.assertEqual({h.tier for h in res.hits}, {1})
        self.assertTrue(any("not approved" in n for n in res.notes))

    def test_partial_approval_keeps_only_accepted(self):
        res = self.run_search(tiers=(1, 2, 3), approver=lambda hits: [h for h in hits if h.tier == 3])
        self.assertEqual({h.tier for h in res.hits}, {1, 3})

    def test_tier_3_search_carries_a_warning(self):
        res = self.run_search(tiers=(3,), approver=lambda hits: hits)
        self.assertEqual({h.tier for h in res.hits}, {3})
        self.assertTrue(any(n.startswith("Warning") for n in res.notes))

    def test_unknown_tier_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_search(tiers=(4,))

    def test_tier_totals_only_count_returned_hits(self):
        res = self.run_search(tiers=(1, 2), approver=lambda hits: [])
        self.assertEqual(res.tokens_returned, sum(h.token_count for h in res.hits))

    def test_only_search_module_reads_the_store(self):
        """Nothing but retrieve/search.py may load candidates or read chunk text."""
        offenders = []
        for path in SRC.rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            if rel in ("retrieve/search.py", "index/store.py"):
                continue
            body = path.read_text(encoding="utf-8")
            if re.search(r"\.(load_candidates|read_chunk)\(", body):
                offenders.append(rel)
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
