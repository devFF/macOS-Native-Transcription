"""
diarizer.py
-----------
Speaker diarization using pyannote.audio (fully local, one-time model download).

Optimized for Apple Silicon by:
  * preprocessing audio to mono 16 kHz before inference
  * preferring the newer `community-1` diarization pipeline
  * tuning CPU thread counts for M-series chips
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
from hf_cache import get_hf_token, resolve_snapshot

# Lazy imports so the app still launches if pyannote isn't installed
_pyannote_available = False
try:
    from pyannote.audio import Pipeline as _Pipeline

    _pyannote_available = True
except ImportError:
    pass

_DEFAULT_PIPELINE_ID = "pyannote/speaker-diarization-community-1"
_APPLE_CPU_THREADS = 8
_DEPENDENCY_MODELS = [
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
]


class Diarizer:
    """
    Assigns speaker labels to a list of Segment objects.

    Parameters
    ----------
    on_status : callable
        Status callback (same signature as in AudioRecorder / Transcriber).
    num_speakers : int | None
        Hint for pyannote. None = auto-detect.
    pipeline_id : str
        Hugging Face diarization pipeline id or local pipeline path.
    """

    def __init__(
        self,
        on_status: Callable[[str], None] = print,
        num_speakers: int | None = None,
        pipeline_id: str = _DEFAULT_PIPELINE_ID,
    ):
        self.on_status = on_status
        self.num_speakers = num_speakers
        self.pipeline_id = pipeline_id
        self._pipeline = None
        self._available = False

        if not _pyannote_available:
            on_status(
                "⚠ pyannote.audio not installed — speaker diarization disabled. "
                "Run: pip install pyannote.audio"
            )
            return

        token = get_hf_token()
        if not token:
            on_status(
                "⚠ No HF_TOKEN found — speaker diarization disabled. "
                "See src/diarizer.py for setup instructions."
            )
            return

        self._token = token
        self._available = True

    def assign_speakers(self, wav_path: Path | str, segments: list) -> list:
        """
        Run diarization on *wav_path* and assign .speaker to each Segment.
        Returns the same list (mutated in place) for chaining.

        If diarization is unavailable, labels remain 'SPEAKER_?'.
        """
        if not self._available or not segments:
            return segments

        wav_path = Path(wav_path)
        self.on_status("Running speaker diarization…")

        try:
            import soundfile as sf
            import torch

            self._configure_torch_runtime(torch)
            pipeline = self._load_pipeline()

            waveform, sample_rate = self._load_audio_for_diarization(sf, wav_path)
            input_dict = {
                "waveform": torch.from_numpy(waveform),
                "sample_rate": sample_rate,
            }

            diarization = pipeline(input_dict, num_speakers=self.num_speakers)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            self.on_status(f"⚠ Diarization failed ({exc}) — using timestamps only")
            return segments

        tracks = self._get_tracks(diarization)
        if not tracks:
            self.on_status("⚠ Diarization returned no speaker turns")
            return segments

        self._assign_tracks_to_segments(segments, tracks)

        n_speakers = len({s.speaker for s in segments} - {"SPEAKER_?"})
        self.on_status(f"Diarization complete — {n_speakers} speaker(s) detected")
        return segments

    def _configure_torch_runtime(self, torch) -> None:
        thread_count = max(1, min(_APPLE_CPU_THREADS, os.cpu_count() or 1))
        try:
            torch.set_num_threads(thread_count)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    def _load_pipeline(self):
        """Lazy-load pyannote pipeline."""
        if self._pipeline is None:
            self.on_status("Loading pyannote speaker-diarization model…")
            for dependency in _DEPENDENCY_MODELS:
                resolve_snapshot(
                    dependency,
                    on_status=self.on_status,
                    status_name=f"pyannote dependency '{dependency}'",
                )
            pipeline_source = resolve_snapshot(
                self.pipeline_id,
                on_status=self.on_status,
                status_name=f"pyannote pipeline '{self.pipeline_id}'",
            )
            self._pipeline = _Pipeline.from_pretrained(
                pipeline_source,
                token=self._token,
            )
            self.on_status("Diarization model ready.")
        return self._pipeline

    def _load_audio_for_diarization(self, sf, wav_path: Path) -> tuple[np.ndarray, int]:
        waveform, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        if sample_rate != 16000:
            waveform = self._resample_mono(waveform, sample_rate, 16000)
            sample_rate = 16000
        waveform = np.ascontiguousarray(waveform[np.newaxis, :], dtype=np.float32)
        return waveform, sample_rate

    @staticmethod
    def _resample_mono(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr or audio.size == 0:
            return audio.astype(np.float32, copy=False)

        duration = audio.shape[0] / float(orig_sr)
        target_len = max(1, int(round(duration * target_sr)))
        x_old = np.linspace(0.0, duration, audio.shape[0], endpoint=False)
        x_new = np.linspace(0.0, duration, target_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)

    @staticmethod
    def _get_tracks(diarization) -> list[tuple[float, float, str]]:
        track_source = getattr(diarization, "exclusive_speaker_diarization", None)
        if track_source is None and hasattr(diarization, "speaker_diarization"):
            speaker_diarization = diarization.speaker_diarization
            track_source = getattr(
                diarization, "exclusive_speaker_diarization", speaker_diarization
            )
        if track_source is None:
            track_source = diarization

        tracks = [
            (float(turn.start), float(turn.end), speaker)
            for turn, _, speaker in track_source.itertracks(yield_label=True)
        ]
        tracks.sort(key=lambda item: (item[0], item[1]))
        return tracks

    @staticmethod
    def _assign_tracks_to_segments(segments: list, tracks: list[tuple[float, float, str]]) -> None:
        track_index = 0
        track_count = len(tracks)

        for seg in segments:
            best_speaker = "SPEAKER_?"
            best_overlap = 0.0

            while track_index < track_count and tracks[track_index][1] <= seg.start:
                track_index += 1

            probe = track_index
            while probe < track_count and tracks[probe][0] < seg.end:
                t_start, t_end, speaker = tracks[probe]
                overlap = max(0.0, min(seg.end, t_end) - max(seg.start, t_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
                probe += 1

            seg.speaker = best_speaker
