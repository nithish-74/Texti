import unittest

from texti.pipeline import process_file
from texti.retrieve.search import search
from tests.helpers import Env, text_image


class DuplicateCopyTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()

    def tearDown(self):
        self.env.close()

    def _ingest(self, name, file_id, rel_path, content_hash):
        p = self.env.write(name, text_image(["Lecture 3: Binary Search Trees", "Definition of a tree node."]))
        process_file(self.env.store, self.env.embedder, self.env.cfg, path=p, file_id=file_id,
                     rel_path=rel_path, content_hash=content_hash, rules=[])

    def test_identical_copies_are_returned_once(self):
        self._ingest("a.png", "id-a", "Pictures/a.png", "h1")
        self._ingest("b.png", "id-b", "Pictures - Copy/a.png", "h2")
        res = search(self.env.store, self.env.embedder, "tree node definition", record_usage=False)
        texts = [h.text for h in res.hits]
        self.assertTrue(texts)
        self.assertEqual(len(texts), len(set(texts)))
        self.assertEqual(res.tokens_raw_estimate, self.env.cfg.image_tokens)
        self.assertTrue(any("duplicate" in n for n in res.notes))


if __name__ == "__main__":
    unittest.main()
