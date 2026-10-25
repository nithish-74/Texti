import time
import unittest

import numpy as np

from texti.index.embedder import HashingEmbedder
from texti.index.store import ChunkRecord
from texti.retrieve.search import search
from tests.helpers import Env

N_CHUNKS = 5000
SEARCH_LIMIT_SECONDS = 2.0


class SearchSpeedTests(unittest.TestCase):
    def test_search_on_a_5000_chunk_index_is_under_two_seconds(self):
        env = Env()
        try:
            rng = np.random.default_rng(7)
            dim = env.embedder.dim
            vecs = rng.normal(size=(N_CHUNKS, dim)).astype(np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
            words = ["tree", "graph", "heap", "stack", "queue", "sort", "hash", "list"]
            records = [
                ChunkRecord(f"file{i // 5}", "full", f"{words[i % 8]} chunk number {i} " * 20, "Docs/x.png",
                            vecs[i], 80, source_offset=i % 5)
                for i in range(N_CHUNKS)
            ]
            env.store.add_chunks(1, records)
            for i in range(N_CHUNKS // 5):
                env.store.upsert_file(f"file{i}", "h", 1, 765, 0.1, "done")
            self.assertEqual(env.store.stats()["chunks_by_tier"][1], N_CHUNKS)

            times = []
            for _ in range(3):
                t0 = time.perf_counter()
                res = search(env.store, env.embedder, "binary tree heap", record_usage=False)
                times.append(time.perf_counter() - t0)
            best = min(times)
            print(f"\n  search over {N_CHUNKS} chunks: {', '.join(f'{t:.2f}s' for t in times)}")
            self.assertLess(max(times), SEARCH_LIMIT_SECONDS)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
