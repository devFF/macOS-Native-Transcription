"""
transcriber.py
--------------
Local speech-to-text using Parakeet v3 on Apple Silicon via MLX.
Returns timestamped Segment objects for diarization alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Callable

from hf_cache import resolve_snapshot

DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
DEFAULT_CHUNK_DURATION = 120.0
DEFAULT_OVERLAP_DURATION = 10.0


@dataclass
class Segment:
    """One speech segment from the transcription."""

    start: float  # seconds
    end: float  # seconds
    text: str
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
    Wraps Parakeet v3 via `parakeet-mlx`.
    Model is lazy-loaded on first use and cached by Hugging Face locally.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        on_status: Callable[[str], None] = print,
    ):
        self.model_id = model_id
        self.on_status = on_status
        self._model = None
        self._decoding_config = None
        self._inference_lock = threading.RLock()

    def _load_model(self):
        with self._inference_lock:
            if self._model is not None:
                return self._model

            try:
                from parakeet_mlx import DecodingConfig, SentenceConfig, from_pretrained
            except ImportError as exc:
                raise RuntimeError(
                    "Parakeet MLX is not installed. Run: pip install parakeet-mlx"
                ) from exc

            self.on_status(f"Loading Parakeet '{self.model_id}'…")
            model_path = resolve_snapshot(
                self.model_id,
                allow_patterns=["config.json", "model.safetensors"],
                on_status=self.on_status,
                status_name=f"Parakeet model '{self.model_id}'",
            )
            self._model = from_pretrained(model_path)
            self._decoding_config = DecodingConfig(
                sentence=SentenceConfig(
                    max_words=30,
                    silence_gap=1.2,
                    max_duration=20.0,
                )
            )

            encoder = getattr(self._model, "encoder", None)
            set_attention_model = getattr(encoder, "set_attention_model", None)
            if callable(set_attention_model):
                # Local attention lowers memory pressure on longer recordings.
                set_attention_model("rel_pos_local_attn", (256, 256))

            self.on_status("Parakeet model ready.")
            return self._model

    @staticmethod
    def _result_to_segments(result) -> list[Segment]:
        segments: list[Segment] = []
        for sentence in getattr(result, "sentences", []) or []:
            text = getattr(sentence, "text", "").strip()
            if not text:
                continue
            start = float(getattr(sentence, "start", 0.0) or 0.0)
            end = float(getattr(sentence, "end", start) or start)
            segments.append(Segment(start=start, end=max(start, end), text=text))
        return segments

    def _collect_language(self, result) -> str:
        language = getattr(result, "language", None)
        return language or "auto"

    def transcribe_segments(self, wav_path: Path | str) -> list[Segment]:
        """
        Transcribe and return a list of Segments with start/end timestamps.
        Speaker field defaults to 'SPEAKER_?' — set by Diarizer afterwards.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV not found: {wav_path}")

        self.on_status(f"Transcribing {wav_path.name} with Parakeet…")
        with self._inference_lock:
            model = self._load_model()
            result = model.transcribe(
                str(wav_path),
                chunk_duration=DEFAULT_CHUNK_DURATION,
                overlap_duration=DEFAULT_OVERLAP_DURATION,
                decoding_config=self._decoding_config,
            )
        segments = self._result_to_segments(result)

        total_words = sum(len(s.text.split()) for s in segments)
        self.on_status(
            f"Transcription complete — {total_words} words "
            f"({self._collect_language(result)}, {segments[-1].end:.0f}s audio)"
            if segments
            else "Transcription complete — 0 words"
        )
        return segments

    def transcribe_live_chunk(self, audio_array) -> str:
        """Transcribes a raw float32 numpy array on the fly."""
        import numpy as np

        audio = np.asarray(audio_array).flatten().astype(np.float32)
        if audio.size == 0:
            return ""

        try:
            import mlx.core as mx
            from parakeet_mlx.audio import get_logmel
        except ImportError as exc:
            raise RuntimeError(
                "Parakeet MLX runtime is incomplete. Reinstall with: pip install parakeet-mlx"
            ) from exc

        with self._inference_lock:
            model = self._load_model()
            mel = get_logmel(mx.array(audio), model.preprocessor_config)
            alignments = model.generate(mel, decoding_config=self._decoding_config)
        result = alignments[0] if isinstance(alignments, list) else alignments
        text = getattr(result, "text", "") or ""
        return text.strip()

    def transcribe(self, wav_path: Path | str) -> str:
        """Plain-text transcript (backwards-compatible)."""
        return " ".join(s.text for s in self.transcribe_segments(wav_path))
