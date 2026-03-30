"""
hf_cache.py
-----------
Helpers for offline-first Hugging Face model resolution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).parent.parent
_TOKEN_FILE = _PROJECT_ROOT / ".hf_token"


def get_hf_token() -> str | None:
    """Return the Hugging Face token or None if not configured."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if _TOKEN_FILE.exists():
        token = _TOKEN_FILE.read_text().strip()
        if token:
            return token
    return None


def resolve_snapshot(
    repo_id_or_path: str | Path,
    *,
    allow_patterns: list[str] | None = None,
    on_status: Callable[[str], None] | None = None,
    status_name: str | None = None,
) -> str:
    """
    Resolve a Hugging Face repo to a local snapshot path.

    This is offline-first: if the repo is already cached locally, no network request
    is made. If not cached yet, it is downloaded once and reused afterwards.
    """
    path = Path(repo_id_or_path).expanduser()
    if path.exists():
        return str(path.resolve())

    from huggingface_hub import snapshot_download

    token = get_hf_token()
    snapshot_kwargs = {
        "repo_id": str(repo_id_or_path),
        "allow_patterns": allow_patterns,
        "token": token,
    }

    try:
        return snapshot_download(local_files_only=True, **snapshot_kwargs)
    except Exception:
        if on_status and status_name:
            on_status(f"Downloading {status_name}…")
        return snapshot_download(local_files_only=False, **snapshot_kwargs)
