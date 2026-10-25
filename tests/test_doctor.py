import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from texti.cli import App, cmd_doctor
from texti.config import Config
from texti.extract.ocr import OcrError


class FakeFailingKeyringStore:
    def get(self, name: str):
        return None

    def set(self, name: str, value: str):
        raise RuntimeError("Keyring locked")

    def delete(self, name: str):
        pass


class FakeWorkingKeyringStore:
    def __init__(self):
        self._d = {}

    def get(self, name: str):
        return self._d.get(name)

    def set(self, name: str, value: str):
        self._d[name] = value

    def delete(self, name: str):
        self._d.pop(name, None)


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.tmp.name) / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.out = io.StringIO()
        self.secrets = FakeWorkingKeyringStore()
        self.app = App(self.secrets, self.out)
        self.app.dir = self.data_dir
        self.app.cfg = Config()

    def tearDown(self):
        self.app.close()
        self.tmp.cleanup()

    def test_doctor_all_pass(self):
        with patch.dict(os.environ, {"TEXTI_CLIENT_ID": "test-client-id"}):
            with patch("texti.extract.ocr.installed_languages", return_value=["eng", "tel", "hin", "osd"]):
                with patch("texti.cli.get_embedder", return_value=None):
                    cmd_doctor(self.app, None)

        output = self.out.getvalue()
        self.assertIn("Client ID: PASS", output)
        self.assertIn("Tesseract: PASS", output)
        self.assertIn("Language data (eng, tel, hin): PASS", output)
        self.assertIn("Embedding model: PASS", output)
        self.assertIn("Keychain: PASS", output)
        self.assertIn("Data folder: PASS", output)
        self.assertNotIn("test-client-id", output, "Must not print client ID secret")

    def test_doctor_missing_client_id(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TEXTI_CLIENT_ID", None)
            with patch("texti.extract.ocr.installed_languages", return_value=["eng", "tel", "hin"]):
                with patch("texti.cli.get_embedder", return_value=None):
                    cmd_doctor(self.app, None)

        output = self.out.getvalue()
        self.assertIn("Client ID: FAIL", output)
        self.assertIn("Fix:", output)

    def test_doctor_missing_tesseract(self):
        with patch.dict(os.environ, {"TEXTI_CLIENT_ID": "some-id"}):
            with patch("texti.extract.ocr.installed_languages", side_effect=OcrError("Tesseract is not installed")):
                with patch("texti.cli.get_embedder", return_value=None):
                    cmd_doctor(self.app, None)

        output = self.out.getvalue()
        self.assertIn("Tesseract: FAIL", output)
        self.assertIn("Fix:", output)
        self.assertIn("Language data (eng, tel, hin): FAIL", output)

    def test_doctor_missing_some_languages(self):
        with patch.dict(os.environ, {"TEXTI_CLIENT_ID": "some-id"}):
            with patch("texti.extract.ocr.installed_languages", return_value=["eng"]):
                with patch("texti.cli.get_embedder", return_value=None):
                    cmd_doctor(self.app, None)

        output = self.out.getvalue()
        self.assertIn("Tesseract: PASS", output)
        self.assertIn("Language data (eng, tel, hin): FAIL", output)
        self.assertIn("tel", output)
        self.assertIn("hin", output)
        self.assertIn("Fix:", output)

    def test_doctor_missing_embedding_model(self):
        with patch.dict(os.environ, {"TEXTI_CLIENT_ID": "some-id"}):
            with patch("texti.extract.ocr.installed_languages", return_value=["eng", "tel", "hin"]):
                with patch("texti.cli.get_embedder", side_effect=RuntimeError("Model not cached")):
                    cmd_doctor(self.app, None)

        output = self.out.getvalue()
        self.assertIn("Embedding model: FAIL", output)
        self.assertIn("texti fetch-model", output)

    def test_doctor_failing_keychain(self):
        app = App(FakeFailingKeyringStore(), self.out)
        app.dir = self.data_dir
        with patch.dict(os.environ, {"TEXTI_CLIENT_ID": "some-id"}):
            with patch("texti.extract.ocr.installed_languages", return_value=["eng", "tel", "hin"]):
                with patch("texti.cli.get_embedder", return_value=None):
                    cmd_doctor(app, None)

        output = self.out.getvalue()
        self.assertIn("Keychain: FAIL", output)
        self.assertIn("Fix:", output)

    def test_doctor_unwritable_data_folder(self):
        with patch.dict(os.environ, {"TEXTI_CLIENT_ID": "some-id"}):
            with patch("texti.extract.ocr.installed_languages", return_value=["eng", "tel", "hin"]):
                with patch("texti.cli.get_embedder", return_value=None):
                    with patch.object(Path, "mkdir", side_effect=PermissionError("Read-only filesystem")):
                        cmd_doctor(self.app, None)

        output = self.out.getvalue()
        self.assertIn("Data folder: FAIL", output)
        self.assertIn("Fix:", output)


if __name__ == "__main__":
    unittest.main()
