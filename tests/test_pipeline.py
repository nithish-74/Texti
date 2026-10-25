import logging
import unittest

from texti.privacy.classifier import PathRule
from texti.pipeline import process_file
from texti.retrieve.search import search
from tests import fixtures as fx
from tests.helpers import Env, text_image


def all_db_bytes(env) -> bytes:
    return b"".join(p.read_bytes() for p in sorted((env.dir / "data").glob("*.db*")))


class CaptureLogs(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.env = Env()
        self.logs = CaptureLogs()
        self.logger = logging.getLogger("texti")
        self.logger.addHandler(self.logs)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        self.logger.removeHandler(self.logs)
        self.env.close()

    def ingest(self, name, lines, rel_path=None, rules=None, file_id=None, content_hash="h1"):
        p = self.env.write(name, text_image(lines))
        return process_file(
            self.env.store, self.env.embedder, self.env.cfg, path=p, file_id=file_id or f"id-{abs(hash(name)) % 10**8}",
            rel_path=rel_path or f"Docs/{name}", content_hash=content_hash, rules=rules or [],
        )

    # -- real OCR through the whole pipeline ----------------------------------------

    def test_lecture_screenshot_becomes_searchable_tier_1_text(self):
        r = self.ingest("slide.png", ["Lecture 3: Binary Search Trees",
                                       "A binary search tree keeps smaller keys on the left.",
                                       "Assignment: implement insert and delete."])
        self.assertEqual(r.status, "done")
        self.assertEqual(r.max_tier, 1)
        res = search(self.env.store, self.env.embedder, "binary search tree", record_usage=False)
        self.assertTrue(res.hits)
        self.assertIn("binary search tree", " ".join(h.text for h in res.hits).lower())

    def test_text_costs_far_fewer_tokens_than_the_image(self):
        self.ingest("slide.png", ["Lecture 3: Binary Search Trees", "Definition of a tree node."])
        res = search(self.env.store, self.env.embedder, "tree node definition")
        self.assertEqual(res.tokens_raw_estimate, self.env.cfg.image_tokens)
        self.assertLess(res.tokens_returned, res.tokens_raw_estimate / 5)
        self.assertEqual(res.tokens_saved_estimate, res.tokens_raw_estimate - res.tokens_returned)

    # -- redaction ----------------------------------------------------------------------

    def test_aadhaar_never_reaches_storage_search_or_logs(self):
        r = self.ingest("id.png", ["Name Test Person", f"Aadhaar number {fx.AADHAAR_SPACED}"],
                        rules=[PathRule("Docs", 1)])
        self.assertEqual(r.max_tier, 3, "content must raise a Tier 1 folder rule to Tier 3")
        res = search(self.env.store, self.env.embedder, "aadhaar number", tiers=(3,),
                     approver=lambda hits: hits, record_usage=False)
        self.assertTrue(res.hits)
        shown = " ".join(h.text for h in res.hits)
        self.assertIn(f"XXXX XXXX {fx.AADHAAR[-4:]}", shown)
        for form in (fx.AADHAAR, fx.AADHAAR_SPACED):
            self.assertNotIn(form, shown)
            self.assertFalse(form.encode() in all_db_bytes(self.env), "full number found in database files")
            self.assertFalse(any(form in line for line in self.logs.lines))

    def test_ocr_confusable_digits_in_other_ids_are_masked_too(self):
        self.ingest("pan.png", ["Permanent Account Number", fx.PAN, f"Card {fx.CARD_SPACED}"])
        res = search(self.env.store, self.env.embedder, "account number", tiers=(1, 2, 3),
                     approver=lambda hits: hits, record_usage=False)
        shown = " ".join(h.text for h in res.hits)
        self.assertNotIn(fx.PAN, shown)
        self.assertNotIn(fx.CARD_SPACED, shown)

    def test_logs_never_contain_text_or_paths(self):
        self.ingest("secret-name.png", ["Confidential unique marker zebra-quartz-1729"],
                    rel_path="Private/secret-name.png")
        joined = "\n".join(self.logs.lines)
        self.assertTrue(joined)  # something was logged
        self.assertNotIn("zebra-quartz", joined)
        self.assertNotIn("secret-name", joined)
        self.assertNotIn("Private", joined)

    # -- encryption ----------------------------------------------------------------------

    def test_database_files_contain_no_plaintext(self):
        self.ingest("note.png", ["Unique marker word: zebra-quartz-1729", "binary search tree lecture"],
                    rel_path="College/Maths/note.png")
        raw = all_db_bytes(self.env)
        self.assertGreater(len(raw), 1000)
        for needle in (b"zebra", b"quartz", b"binary search", b"College", b"Maths", b"note.png"):
            self.assertFalse(needle in raw, f"plaintext {needle!r} found in the database files")

    def test_each_tier_has_its_own_key(self):
        self.ingest("a.png", ["Lecture notes about trees and the syllabus"], rules=[PathRule("Docs", 1)])
        self.ingest("b.png", ["Bank Statement for March"], rules=[PathRule("Docs", 1)])
        keys = {self.env.secrets.get(f"tier{t}-key") for t in (1, 3)}
        self.assertEqual(len(keys), 2)
        self.assertNotIn(None, keys)

    def test_data_is_unreadable_with_a_different_key(self):
        from texti.index.store import CryptoError, Store
        from texti.secrets_store import MemorySecretStore

        self.ingest("a.png", ["Lecture notes about trees and the syllabus"], rules=[PathRule("Docs", 1)])
        self.env.store.close()
        other = Store(self.env.dir / "data", MemorySecretStore())  # fresh, different keys
        with self.assertRaises(CryptoError):
            other.load_candidates(1)
        other.close()

    # -- tiers and re-processing --------------------------------------------------------------

    def test_bank_statement_phrase_makes_whole_file_tier_3(self):
        r = self.ingest("bs.png", ["Bank Statement", "Salary credit 45,000", "Rent payment 15,000"],
                        rules=[PathRule("Docs", 1)])
        self.assertEqual(r.max_tier, 3)
        self.assertEqual(set(r.chunks_by_tier), {3})

    def test_unknown_plain_file_defaults_to_tier_2_and_is_not_in_default_search(self):
        r = self.ingest("n.png", ["Shopping list: milk, bread, tomatoes"])
        self.assertEqual(r.max_tier, 2)
        res = search(self.env.store, self.env.embedder, "shopping milk bread", record_usage=False)
        self.assertEqual(res.hits, [])

    def test_unchanged_file_is_skipped(self):
        self.ingest("a.png", ["Lecture about trees, the syllabus and chapter one"])
        again = self.ingest("a.png", ["Lecture about trees, the syllabus and chapter one"])
        self.assertEqual(again.status, "unchanged")

    def test_changed_file_replaces_old_chunks(self):
        self.ingest("a.png", ["Lecture about trees, the syllabus and chapter one"], file_id="f", content_hash="v1")
        self.ingest("a.png", ["Lecture about graphs, the syllabus and chapter two"], file_id="f", content_hash="v2")
        stats = self.env.store.stats()
        self.assertEqual(sum(stats["chunks_by_tier"].values()), 2)  # one summary + one chunk, not four
        res = search(self.env.store, self.env.embedder, "graphs", record_usage=False)
        self.assertIn("graphs", " ".join(h.text for h in res.hits).lower())
        self.assertNotIn("trees", " ".join(h.text for h in res.hits).lower())

    def test_raising_the_tier_removes_copies_from_lower_tiers(self):
        self.ingest("a.png", ["Lecture about trees, the syllabus and chapter one"], file_id="f",
                    content_hash="v1", rules=[PathRule("Docs", 1)])
        self.assertGreater(self.env.store.stats()["chunks_by_tier"][1], 0)
        self.ingest("a.png", ["Lecture about trees, the syllabus and chapter one"], file_id="f",
                    content_hash="v1", rules=[PathRule("Docs", 3)])  # same hash, tier rule changed
        # unchanged hash is skipped, so a tier change needs force; check that path explicitly
        p = self.env.dir / "a.png"
        process_file(self.env.store, self.env.embedder, self.env.cfg, path=p, file_id="f",
                     rel_path="Docs/a.png", content_hash="v1", rules=[PathRule("Docs", 3)], force=True)
        stats = self.env.store.stats()
        self.assertEqual(stats["chunks_by_tier"][1], 0)
        self.assertGreater(stats["chunks_by_tier"][3], 0)

    def test_pages_are_classified_separately(self):
        from texti.extract import ExtractionResult
        from texti.extract.ocr import Segment
        import texti.pipeline as pl

        pages = [Segment("Lecture on trees. Definition of a node. The syllabus covers chapter one.", 1, 1.0),
                 Segment(f"Scanned ID. Aadhaar {fx.AADHAAR_SPACED}", 2, 0.9),
                 Segment("Lecture on graphs. Assignment: define an edge. See the course textbook.", 3, 1.0)]
        original = pl.extract_file
        pl.extract_file = lambda path, lang: ExtractionResult(pages, 3)
        try:
            p = self.env.write("mixed.pdf", b"%PDF-fake")
            r = process_file(self.env.store, self.env.embedder, self.env.cfg, path=p, file_id="m",
                             rel_path="Docs/mixed.pdf", content_hash="h", rules=[PathRule("Docs", 1)])
        finally:
            pl.extract_file = original
        self.assertEqual(r.max_tier, 3)
        self.assertGreater(r.chunks_by_tier.get(1, 0), 0)  # lecture pages stay Tier 1
        self.assertGreater(r.chunks_by_tier.get(3, 0), 0)  # only the ID page is Tier 3
        self.assertEqual(self.env.store.stats()["chunks_by_tier"][2], 0)

    def test_empty_image_is_recorded_without_chunks(self):
        r = self.ingest("blank.png", [""])
        self.assertEqual(r.status, "empty")
        self.assertEqual(r.chunks, 0)


if __name__ == "__main__":
    unittest.main()
