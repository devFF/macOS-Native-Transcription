"""
audio_importer.py
-----------------
Imports existing audio files via FFmpeg and normalizes them to mono 16 kHz WAV
so the rest of the app can transcribe and diarize them consistently.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

from audio_recorder import SAMPLE_RATE
from config import get_recordings_dir


class AudioImportError(RuntimeError):
    """Raised when an external audio file cannot be imported."""


def import_audio_to_wav(
    source_path: Path | str,
    on_status: Callable[[str], None] = print,
) -> Path:
    source_path = Path(source_path).expanduser().resolve()
    if not source_path.exists():
        raise AudioImportError(f"Audio file not found: {source_path}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise AudioImportError("FFmpeg not found. Install it with: brew install ffmpeg")

    recordings_dir = get_recordings_dir()
    recordings_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = _sanitize_stem(source_path.stem)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = recordings_dir / f"{timestamp}_{safe_stem}.wav"

    on_status(f"Importing {source_path.name}…")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        error_text = (exc.stderr or "").strip().splitlines()
        detail = error_text[-1] if error_text else "ffmpeg conversion failed"
        raise AudioImportError(f"Could not import audio: {detail}") from exc

    return output_path


def _sanitize_stem(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return sanitized or "imported_audio"
