"""Audio and video transcription using local Whisper and PyAV/soundfile."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .ocr import Segment

log = logging.getLogger("texti")

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
MEDIA_AUDIO_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

DEFAULT_WHISPER_MODEL = "openai/whisper-tiny"
TARGET_SAMPLE_RATE = 16000

_asr_pipeline: Any = None
_asr_model_name: str | None = None


class AudioExtractionError(RuntimeError):
    pass


def read_audio_frames(path: Path | str, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Read an audio or video file, returning (samples, duration_seconds) resampled to target_sr mono float32."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Fast path: soundfile for standard audio formats
    ext = p.suffix.lower()
    if ext in {".wav", ".mp3", ".ogg", ".flac"}:
        try:
            import soundfile as sf

            data, sr = sf.read(str(p), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)  # convert stereo to mono
            duration = float(len(data) / sr) if sr > 0 else 0.0
            if sr != target_sr and len(data) > 0:
                # Resample using linear interpolation if rates differ
                num_target_samples = int(len(data) * target_sr / sr)
                data = np.interp(
                    np.linspace(0, len(data), num_target_samples, endpoint=False),
                    np.arange(len(data)),
                    data,
                ).astype(np.float32)
            return data.astype(np.float32), duration
        except Exception as e:
            log.debug("soundfile could not decode this file (%s), falling back to PyAV", type(e).__name__)

    # PyAV path: decodes container audio streams from video or complex audio formats
    try:
        import av
    except ImportError as e:
        raise AudioExtractionError("PyAV is required for audio/video decoding: pip install av") from e

    try:
        container = av.open(str(p))
    except Exception as e:
        raise AudioExtractionError(f"Failed to open the media container ({type(e).__name__})") from e

    audio_stream = next((s for s in container.streams if s.type == "audio"), None)
    if audio_stream is None:
        container.close()
        return np.zeros(0, dtype=np.float32), 0.0

    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)
    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio_stream):
            for r_frame in resampler.resample(frame):
                chunks.append(r_frame.to_ndarray())
        for r_frame in resampler.resample(None):
            chunks.append(r_frame.to_ndarray())
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32), 0.0

    raw_arr = np.concatenate(chunks, axis=1)
    samples = (raw_arr[0] / 32768.0).astype(np.float32)
    duration = float(len(samples) / target_sr)
    return samples, duration


def get_asr_pipeline(model_name: str = DEFAULT_WHISPER_MODEL, allow_download: bool = False):
    """Obtain or cache the Whisper speech-to-text pipeline."""
    global _asr_pipeline, _asr_model_name
    if _asr_pipeline is not None and _asr_model_name == model_name:
        return _asr_pipeline

    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        from transformers import pipeline
    except ImportError as e:
        raise AudioExtractionError("transformers is required for speech recognition: pip install transformers") from e

    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            return_timestamps=True,
        )
    except Exception as e:
        if not allow_download:
            raise AudioExtractionError(
                f"Whisper model {model_name!r} is not on this machine. "
                "Run `texti fetch-model` once to download the public model weights."
            ) from e
        raise

    _asr_pipeline = pipe
    _asr_model_name = model_name
    return _asr_pipeline


def extract_audio(
    path: Path | str,
    language: str = "eng",
    model_name: str = DEFAULT_WHISPER_MODEL,
    allow_download: bool = False,
    asr_fn: Any = None,
) -> tuple[list[Segment], int]:
    """Transcribe an audio or video file into timestamped segments.

    Returns (segments, estimated_pages_equivalent), where each segment's offset
    represents the timestamp in seconds.
    """
    samples, duration = read_audio_frames(path)
    pages = max(1, math.ceil(duration / 60.0))
    if len(samples) == 0:
        return [], pages

    # If all silence (e.g. max amplitude negligible)
    if float(np.max(np.abs(samples))) < 1e-4:
        return [], pages

    if asr_fn is not None:
        pipe = asr_fn
    else:
        pipe = get_asr_pipeline(model_name=model_name, allow_download=allow_download)

    generate_kwargs: dict[str, Any] = {}
    if language and language not in ("eng", "auto"):
        # map common 3-letter codes to 2-letter if needed (e.g. hin -> hi, tel -> te)
        lang_code = language.split("+")[0].strip()
        lang_map = {"hin": "hi", "tel": "te", "tam": "ta", "kan": "kn", "mar": "mr", "ben": "bn"}
        generate_kwargs["language"] = lang_map.get(lang_code, lang_code)

    try:
        result = pipe({"sampling_rate": TARGET_SAMPLE_RATE, "raw": samples}, generate_kwargs=generate_kwargs)
    except Exception as e:
        log.warning("Whisper transcription failed (%s)", type(e).__name__)
        return [], pages

    segments: list[Segment] = []
    chunks = result.get("chunks") if isinstance(result, dict) else None
    if chunks:
        for chunk in chunks:
            text = (chunk.get("text") or "").strip()
            if not text:
                continue
            ts = chunk.get("timestamp") or (0.0, 0.0)
            start_sec = int(ts[0]) if ts[0] is not None else 0
            segments.append(
                Segment(
                    text=text,
                    offset=start_sec,
                    confidence=0.92,
                    language=language,
                )
            )
    else:
        full_text = (result.get("text") or "").strip() if isinstance(result, dict) else str(result).strip()
        if full_text:
            segments.append(
                Segment(
                    text=full_text,
                    offset=0,
                    confidence=0.90,
                    language=language,
                )
            )

    return segments, pages
