"""
transcriber.py
--------------
Local speech-to-text using faster-whisper.
Returns timestamped Segment objects for diarization alignment.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

DEFAULT_MODEL  = "base"
DEVICE         = "cpu"
COMPUTE_TYPE   = "int8"


@dataclass
class Segment:
    """One speech segment from the transcription."""
    start:   float          # seconds
    end:     float          # seconds
    text:    str
    speaker: str = "SPEAKER_?"

    def timestamp(self) -> str:
        """Format start time as MM:SS."""
        m, s = divmod(int(self.start), 60)
        return f"{m:02d}:{s:02d}"

    def timestamp_range(self) -> str:
        """Format start→end as MM:SS → MM:SS."""
        def fmt(t: float) -> str:
            m, s = divmod(int(t), 60)
            return f"{m:02d}:{s:02d}"
        return f"{fmt(self.start)} → {fmt(self.end)}"


class Transcriber:
    """
    Wraps faster-whisper.
    Model is lazy-loaded on first use and cached in ~/.cache/huggingface/.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        on_status: Callable[[str], None] = print,
    ):
        self.model_size = model_size
        self.on_status  = on_status
        self._model: WhisperModel | None = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> WhisperModel:
        if self._model is None:
            self.on_status(f"Loading Whisper '{self.model_size}' model…")
            self._model = WhisperModel(
                self.model_size,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
            )
            self.on_status("Whisper model ready.")
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def transcribe_segments(self, wav_path: Path | str) -> list[Segment]:
        """
        Transcribe and return a list of Segments with start/end timestamps.
        Speaker field defaults to 'SPEAKER_?' — set by Diarizer afterwards.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        self.on_status(f"Transcribing {wav_path.name}…")
        model = self._load_model()

        raw_segments, info = model.transcribe(
            str(wav_path),
            beam_size=5,
            language=None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,   # segment-level is enough for diarization
        )

        segments = [
            Segment(start=s.start, end=s.end, text=s.text.strip())
            for s in raw_segments
            if s.text.strip()
        ]

        total_words = sum(len(s.text.split()) for s in segments)
        self.on_status(
            f"Transcription complete — {total_words} words "
            f"({info.language}, {info.duration:.0f}s audio)"
        )
        return segments

    def transcribe_live_chunk(self, audio_array) -> str:
        """
        Transcribes a raw float32 numpy array on the fly.
        """
        import numpy as np
        audio_array = np.asarray(audio_array).flatten().astype(np.float32)
        
        model = self._load_model()
        raw_segments, _ = model.transcribe(
            audio_array,
            beam_size=5,
            language=None,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
        )
        return " ".join(s.text.strip() for s in raw_segments if s.text.strip())

    def transcribe(self, wav_path: Path | str) -> str:
        """Plain-text transcript (backwards-compatible)."""
        return " ".join(s.text for s in self.transcribe_segments(wav_path))
