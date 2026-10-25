import unittest

from texti.index.embedder import HashingEmbedder
from texti.index.store import ChunkRecord
from texti.retrieve.bm25 import compute_bm25_scores, tokenize
from texti.retrieve.search import search
from tests.helpers import Env


class BM25UnitTests(unittest.TestCase):
    def test_tokenize(self):
        tokens = tokenize("Invoice INV-2024 for Account_No: 987654!")
        self.assertEqual(tokens, ["invoice", "inv", "2024", "for", "account_no", "987654"])

    def test_bm25_scoring_prioritizes_matching_terms(self):
        docs = [
            "This is a general note about machine learning.",
            "Invoice number INV-9988 for consulting services rendered.",
            "Another document about database optimization and indexing.",
        ]
        scores = compute_bm25_scores("INV-9988", docs)
        self.assertEqual(len(scores), 3)
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)

    def test_bm25_empty_query_and_docs(self):
        self.assertEqual(compute_bm25_scores("", ["Some doc"]), [0.0])
        self.assertEqual(compute_bm25_scores("query", []), [])


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()

    def tearDown(self):
        self.env.close()

    def test_hybrid_search_boosts_exact_code_matches(self):
        # Doc 1: Has specific transaction code TXN-8842 but neutral context
        text1 = "Receipt voucher TXN-8842 payment confirmed for client."
        # Doc 2: Generic text about payment and banking
        text2 = "General banking overview regarding payments, loans, and financial statements."

        emb1 = self.env.embedder.embed([text1])[0]
        emb2 = self.env.embedder.embed([text2])[0]

        self.env.store.add_chunks(
            1,
            [
                ChunkRecord("f1", "full", text1, "Docs/receipt.png", emb1, 20),
                ChunkRecord("f2", "full", text2, "Docs/overview.png", emb2, 20),
            ],
        )
        self.env.store.upsert_file("f1", "h1", 1, 765, 0.1, "done")
        self.env.store.upsert_file("f2", "h2", 1, 765, 0.1, "done")

        # Query searching for the exact voucher number TXN-8842
        res = search(
            self.env.store,
            self.env.embedder,
            "TXN-8842",
            budget=800,
            tiers=(1,),
            hybrid_weight=0.5,
            record_usage=False,
        )

        self.assertTrue(len(res.hits) > 0)
        self.assertEqual(res.hits[0].file_id, "f1")
        self.assertIn("TXN-8842", res.hits[0].text)

    def test_pure_vector_when_hybrid_weight_zero(self):
        text = "Lecture slides on sorting algorithms."
        emb = self.env.embedder.embed([text])[0]
        self.env.store.add_chunks(1, [ChunkRecord("f1", "full", text, "path.png", emb, 15)])
        self.env.store.upsert_file("f1", "h1", 1, 765, 0.1, "done")

        res = search(
            self.env.store,
            self.env.embedder,
            "sorting algorithms",
            budget=800,
            tiers=(1,),
            hybrid_weight=0.0,
            record_usage=False,
        )
        self.assertTrue(len(res.hits) > 0)


if __name__ == "__main__":
    unittest.main()
