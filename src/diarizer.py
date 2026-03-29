"""
diarizer.py
-----------
Speaker diarization using pyannote.audio (fully local, one-time model download).

Prerequisites (one-time setup):
  1. pip install pyannote.audio          (installs torch — ~1-2 GB)
  2. Accept license at:
       https://huggingface.co/pyannote/speaker-diarization-3.1
  3. Create an HF token at:
       https://huggingface.co/settings/tokens
  4. Save the token in ONE of:
       a) export HF_TOKEN=hf_...       (environment variable)
       b) echo "hf_..." > .hf_token   (file in project root)

If pyannote is not installed OR no token is found, the diarizer
falls back silently: all segments keep their "SPEAKER_?" label.
"""

import os
from pathlib import Path
from typing import Callable
import numpy as np

# Lazy imports so the app still launches if pyannote isn't installed
_pyannote_available = False
try:
    from pyannote.audio import Pipeline as _Pipeline
    _pyannote_available = True
except ImportError:
    pass

# Project root is one level up from src/
_PROJECT_ROOT = Path(__file__).parent.parent
_TOKEN_FILE   = _PROJECT_ROOT / ".hf_token"


def _get_hf_token() -> str | None:
    """Return the HuggingFace token or None if not configured."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if _TOKEN_FILE.exists():
        token = _TOKEN_FILE.read_text().strip()
        if token:
            return token
    return None


class Diarizer:
    """
    Assigns speaker labels to a list of Segment objects.

    Parameters
    ----------
    on_status : callable
        Status callback (same signature as in AudioRecorder / Transcriber).
    num_speakers : int | None
        Hint for pyannote. None = auto-detect.
    """

    def __init__(
        self,
        on_status: Callable[[str], None] = print,
        num_speakers: int | None = None,
    ):
        self.on_status    = on_status
        self.num_speakers = num_speakers
        self._pipeline    = None
        self._available   = False

        if not _pyannote_available:
            on_status(
                "⚠ pyannote.audio not installed — speaker diarization disabled. "
                "Run: pip install pyannote.audio"
            )
            return

        token = _get_hf_token()
        if not token:
            on_status(
                "⚠ No HF_TOKEN found — speaker diarization disabled. "
                "See src/diarizer.py for setup instructions."
            )
            return

        self._token     = token
        self._available = True

    # ── Public API ────────────────────────────────────────────────────────────

    def assign_speakers(self, wav_path: Path | str, segments: list) -> list:
        """
        Run diarization on *wav_path* and assign .speaker to each Segment.
        Returns the same list (mutated in place) for chaining.

        If diarization is unavailable, labels remain 'SPEAKER_?'.
        """
        if not self._available:
            return segments

        wav_path = Path(wav_path)
        self.on_status("Running speaker diarization…")

        try:
            import torch
            import soundfile as sf
            
            pipeline = self._load_pipeline()
            
            # Manually load the audio to bypass Pyannote's buggy torchaudio/ffmpeg internals
            waveform, sample_rate = sf.read(str(wav_path))
            if waveform.ndim == 1:
                waveform = waveform[np.newaxis, :]
            else:
                waveform = waveform.T
                
            input_dict = {
                "waveform": torch.from_numpy(waveform).float(),
                "sample_rate": sample_rate
            }

            diarization = pipeline(
                input_dict,
                num_speakers=self.num_speakers,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.on_status(f"⚠ Diarization failed ({exc}) — using timestamps only")
            return segments

        # Build a sortable list of (start, end, speaker) turns
        # Pyannote 3.1+ can return a DiarizeOutput dataclass when given raw waveforms
        if hasattr(diarization, "speaker_diarization"):
            diarization = diarization.speaker_diarization

        turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # Assign each whisper segment to the speaker that covers its midpoint
        for seg in segments:
            mid = (seg.start + seg.end) / 2.0
            best_speaker = "SPEAKER_?"
            best_overlap = 0.0
            for t_start, t_end, speaker in turns:
                # Calculate overlap between segment and turn
                overlap = max(0.0, min(seg.end, t_end) - max(seg.start, t_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
            seg.speaker = best_speaker

        n_speakers = len({s.speaker for s in segments} - {"SPEAKER_?"})
        self.on_status(f"Diarization complete — {n_speakers} speaker(s) detected")
        return segments

    # ── Private ───────────────────────────────────────────────────────────────

    def _load_pipeline(self):
        """Lazy-load pyannote pipeline (downloads model on first use ~750 MB)."""
        if self._pipeline is None:
            self.on_status("Loading pyannote speaker-diarization model…")
            self._pipeline = _Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=self._token,
            )
            self.on_status("Diarization model ready.")
        return self._pipeline
