import io
import os
import tempfile
import unittest
from pathlib import Path

from texti.cli import main
from texti.index.embedder import HashingEmbedder
from texti.secrets_store import MemorySecretStore
from tests import fixtures as fx
from tests.helpers import text_image


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.shots = Path(self.tmp.name) / "College"
        self.shots.mkdir()
        (self.shots / "slide.png").write_bytes(text_image(
            ["Lecture 3: Binary Search Trees", "A binary search tree keeps smaller keys on the left.",
             "Definition of a node. Assignment: implement insert."]))
        (self.shots / "id.png").write_bytes(text_image(["Government ID", f"Aadhaar number {fx.AADHAAR_SPACED}"]))
        (self.shots / "readme.txt").write_text("ignored")
        self._old = os.environ.get("TEXTI_HOME")
        os.environ["TEXTI_HOME"] = str(self.home)
        self.secrets = MemorySecretStore()

    def tearDown(self):
        if self._old is None:
            os.environ.pop("TEXTI_HOME", None)
        else:
            os.environ["TEXTI_HOME"] = self._old
        self.tmp.cleanup()

    def run_cli(self, *argv, answers=()):
        out, answers = io.StringIO(), iter(answers)
        code = main(list(argv), secrets=self.secrets, embedder=HashingEmbedder(), out=out,
                    ask=lambda prompt: next(answers))
        return code, out.getvalue()

    def test_ingest_search_report_forget_flow(self):
        code, out = self.run_cli("ingest", str(self.shots))
        self.assertEqual(code, 0)
        self.assertIn("2 processed", out)

        code, out = self.run_cli("search", "binary search tree node")
        self.assertIn("College/slide.png", out)
        self.assertEqual(code, 0)
        self.assertIn("[Tier 1]", out)
        self.assertIn("Tokens returned", out)
        self.assertIn("(estimate)", out)
        self.assertNotIn(fx.AADHAAR, out)

        code, out = self.run_cli("report")
        self.assertIn("Files processed: 2", out)
        self.assertIn("Tier 3: 1", out)  # the ID scan

        code, out = self.run_cli("forget", "--all", answers=["DELETE"])
        self.assertIn("deleted", out)
        self.assertEqual(list((self.home / "data").glob("*.db")) if (self.home / "data").exists() else [], [])
        self.assertIsNone(self.secrets.get("tier1-key"))

    def test_ingested_paths_are_relative_to_the_given_folder_without_doubling(self):
        (self.shots / "Maths").mkdir()
        (self.shots / "Maths" / "deep.png").write_bytes(text_image(
            ["Lecture 4: Heaps", "Definition of a heap. See the course textbook."]))
        self.run_cli("ingest", str(self.shots))
        _, out = self.run_cli("search", "heap definition")
        self.assertIn("College/Maths/deep.png", out)
        self.assertNotIn("College/College", out)

    def test_second_ingest_skips_unchanged_files(self):
        self.run_cli("ingest", str(self.shots))
        _, out = self.run_cli("ingest", str(self.shots))
        self.assertIn("0 processed, 2 unchanged", out)

    def test_tier_3_search_asks_for_approval_and_can_be_declined(self):
        self.run_cli("ingest", str(self.shots))
        _, out = self.run_cli("search", "aadhaar number", "--tiers", "3", answers=["n"])
        self.assertIn("NOT been sent anywhere", out)
        self.assertIn("No matching results", out)
        self.assertNotIn(fx.AADHAAR, out)

    def test_tier_3_search_shows_only_masked_values_when_approved(self):
        self.run_cli("ingest", str(self.shots))
        _, out = self.run_cli("search", "aadhaar number", "--tiers", "3", answers=["y"])
        self.assertIn(f"XXXX XXXX {fx.AADHAAR[-4:]}", out)
        self.assertNotIn(fx.AADHAAR, out)
        self.assertNotIn(fx.AADHAAR_SPACED, out)
        self.assertIn("Warning", out)

    def test_forget_needs_all_and_confirmation(self):
        self.run_cli("ingest", str(self.shots))
        code, _ = self.run_cli("forget")
        self.assertEqual(code, 1)
        _, out = self.run_cli("forget", "--all", answers=["no"])
        self.assertIn("Nothing was deleted", out)
        _, out = self.run_cli("report")
        self.assertIn("Files processed: 2", out)

    def test_bad_arguments_give_clean_errors(self):
        self.assertEqual(self.run_cli("search", "x", "--tiers", "9")[0], 1)
        self.assertEqual(self.run_cli("search", "x", "--tiers", "a")[0], 1)
        self.assertEqual(self.run_cli("ingest", "/does/not/exist")[0], 1)
        self.assertEqual(self.run_cli("ingest", str(self.shots), "--tier", "7")[0], 1)

    def test_ingest_tier_flag_forces_a_starting_tier(self):
        self.run_cli("ingest", str(self.shots), "--tier", "2")
        _, out = self.run_cli("report")
        self.assertIn("Tier 1: 0", out)

    def test_sync_without_folders_explains_what_to_do(self):
        code, _ = self.run_cli("sync")
        self.assertEqual(code, 1)

    def test_sync_without_login_explains_what_to_do(self):
        from texti.config import Config, FolderConfig

        cfg = Config(folders=[FolderConfig("F1", "College")])
        cfg.save(self.home)
        os.environ["TEXTI_CLIENT_ID"] = "cid"
        try:
            code, _ = self.run_cli("sync")
        finally:
            os.environ.pop("TEXTI_CLIENT_ID")
        self.assertEqual(code, 1)

    def test_legacy_fallback_notice(self):
        import texti.config as cfg_mod
        from unittest.mock import patch
        fake_home = Path(self.tmp.name) / "fake_home"
        fake_home.mkdir(parents=True, exist_ok=True)
        old_folder_name = ".media" + "2text"
        old_dir = fake_home / old_folder_name
        old_dir.mkdir(parents=True, exist_ok=True)
        new_dir = fake_home / ".texti"

        cfg_mod._migration_warned = False
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TEXTI_HOME", None)
            with patch.object(Path, "home", return_value=fake_home):
                f = io.StringIO()
                with patch("sys.stdout", f):
                    d = cfg_mod.data_dir()
                self.assertEqual(d, new_dir)
                self.assertFalse(new_dir.exists())
                self.assertIn("Please move", f.getvalue())


if __name__ == "__main__":
    unittest.main()
