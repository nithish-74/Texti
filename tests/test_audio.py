import io
import math
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from texti.config import Config
from texti.extract import extract_file, is_supported
from texti.extract.audio import (
    AUDIO_EXTENSIONS,
    MEDIA_AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS,
    extract_audio,
    read_audio_frames,
)
from texti.index.embedder import HashingEmbedder
from texti.index.store import Store
from texti.pipeline import process_file
from texti.retrieve.budget import estimate_raw_tokens


def generate_test_wav(duration_sec: float = 1.0, samplerate: int = 16000, freq: float = 440.0) -> bytes:
    """Generate in-memory WAV bytes with a sine tone."""
    t = np.linspace(0, duration_sec, int(samplerate * duration_sec), endpoint=False)
    data = (np.sin(2 * np.pi * freq * t) * 16384).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


class AudioExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_audio_and_video_extensions_defined(self):
        self.assertIn(".wav", AUDIO_EXTENSIONS)
        self.assertIn(".mp3", AUDIO_EXTENSIONS)
        self.assertIn(".m4a", AUDIO_EXTENSIONS)
        self.assertIn(".mp4", VIDEO_EXTENSIONS)
        self.assertIn(".mkv", VIDEO_EXTENSIONS)
        self.assertTrue(MEDIA_AUDIO_EXTENSIONS.issuperset(AUDIO_EXTENSIONS))
        self.assertTrue(MEDIA_AUDIO_EXTENSIONS.issuperset(VIDEO_EXTENSIONS))

    def test_read_audio_frames_wav(self):
        wav_file = self.tmp_path / "tone.wav"
        wav_file.write_bytes(generate_test_wav(duration_sec=0.5, samplerate=16000))
        samples, duration = read_audio_frames(wav_file, target_sr=16000)
        self.assertAlmostEqual(duration, 0.5, places=2)
        self.assertEqual(len(samples), 8000)
        self.assertIsInstance(samples, np.ndarray)

    def test_extract_audio_silent(self):
        wav_file = self.tmp_path / "silence.wav"
        # Zero amplitude
        wav_file.write_bytes(generate_test_wav(duration_sec=1.0, freq=0.0))
        segments, pages = extract_audio(wav_file)
        self.assertEqual(segments, [])
        self.assertEqual(pages, 1)

    def test_extract_audio_with_mocked_asr(self):
        wav_file = self.tmp_path / "speech.wav"
        wav_file.write_bytes(generate_test_wav(duration_sec=2.0))

        mock_asr = MagicMock()
        mock_asr.return_value = {
            "text": "Hello world welcome to texti speech extraction.",
            "chunks": [
                {"timestamp": (0.0, 1.2), "text": "Hello world"},
                {"timestamp": (1.2, 2.0), "text": "welcome to texti speech extraction."},
            ],
        }

        segments, pages = extract_audio(wav_file, asr_fn=mock_asr)
        self.assertEqual(pages, 1)
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "Hello world")
        self.assertEqual(segments[0].offset, 0)
        self.assertEqual(segments[1].text, "welcome to texti speech extraction.")
        self.assertEqual(segments[1].offset, 1)

    def test_extract_file_dispatches_audio(self):
        wav_file = self.tmp_path / "test.mp3"
        # Even with wav data written to .mp3, test dispatch mechanism
        wav_file.write_bytes(generate_test_wav(duration_sec=1.5))

        with patch("texti.extract.extract_audio") as mock_extract:
            from texti.extract.ocr import Segment

            mock_extract.return_value = ([Segment("Transcribed text", offset=0, confidence=0.95)], 1)
            res = extract_file(wav_file)
            self.assertEqual(res.pages, 1)
            self.assertEqual(res.segments[0].text, "Transcribed text")
            mock_extract.assert_called_once()

    def test_raw_token_estimation_for_audio_and_video(self):
        self.assertEqual(estimate_raw_tokens(".wav", 1, 765), 1500)
        self.assertEqual(estimate_raw_tokens(".mp4", 3, 765), 4500)
        self.assertEqual(estimate_raw_tokens(".png", 1, 765), 765)

    def test_pipeline_process_audio_file(self):
        from texti.retrieve.search import search
        from texti.secrets_store import MemorySecretStore

        wav_file = self.tmp_path / "recorded_meeting.wav"
        wav_file.write_bytes(generate_test_wav(duration_sec=1.0))

        db_path = self.tmp_path / "store"
        db_path.mkdir(parents=True, exist_ok=True)
        store = Store(db_path, MemorySecretStore())
        embedder = HashingEmbedder(dim=64)
        cfg = Config()

        fake_asr = MagicMock()
        fake_asr.return_value = {
            "text": "Discussion on project milestones and engineering goals.",
            "chunks": [
                {"timestamp": (0.0, 1.0), "text": "Discussion on project milestones and engineering goals."}
            ],
        }

        with patch("texti.extract.audio.get_asr_pipeline", return_value=fake_asr):
            result = process_file(
                store,
                embedder,
                cfg,
                path=wav_file,
                file_id="local-audio-1",
                rel_path="recordings/recorded_meeting.wav",
                content_hash="hash123",
            )

        try:
            self.assertEqual(result.status, "done")
            self.assertGreater(result.chunks, 0)
            # Search the stored audio chunk
            res = search(store, embedder, "project milestones", budget=800, tiers=(1, 2, 3), approver=lambda hits: hits)
            self.assertTrue(len(res.hits) > 0)

            self.assertIn("project milestones", res.hits[0].text)
        finally:
            store.close()



if __name__ == "__main__":
    unittest.main()

