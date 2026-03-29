"""
tray_app.py
-----------
All-in-one macOS tray app for AI Note-Taker.
No GUI window — everything controlled via the menu bar.

Menu:
  ▶  Start Recording
  ⏹  Stop & Transcribe
  ──────────────────
  Status: Idle
  ──────────────────
  Quit

Transcript output (with timestamps + speaker labels) is saved to
summaries/ and auto-opened in your default Markdown viewer.

Run:
    python src/tray_app.py
"""

import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import rumps

sys.path.insert(0, str(Path(__file__).parent))

from audio_recorder import AudioRecorder
from transcriber import Transcriber
from diarizer import Diarizer
from config import get_summaries_dir, set_output_dir, get_autostart, set_autostart
from autostart import enable_autostart, disable_autostart


def _format_transcript(segments: list) -> str:
    """
    Render a list of Segments into readable Markdown.

    Output example:
        **[00:00 → 00:05] SPEAKER_1:** Hello everyone, let's get started.
        **[00:05 → 00:12] SPEAKER_2:** Thanks for having me.
    """
    lines = []
    lines.append("## Transcript\n")
    for seg in segments:
        lines.append(
            f"**[{seg.timestamp_range()}] {seg.speaker}:** {seg.text}"
        )

    return "\n".join(lines)


class NoteTakerApp(rumps.App):

    def __init__(self, whisper_size: str = "base"):
        super().__init__(name="macOS-Native-Transcription", title="🎙", quit_button=None)

        # ── Backend services ──────────────────────────────────────────────────
        self.recorder    = AudioRecorder(on_status=self._set_status)
        self.transcriber = Transcriber(model_size=whisper_size, on_status=self._set_status)
        self.diarizer    = Diarizer(on_status=self._set_status)

        # ── State ─────────────────────────────────────────────────────────────
        self._state   = "IDLE"
        self._elapsed = 0

        # ── Menu items ────────────────────────────────────────────────────────
        self.start_item  = rumps.MenuItem("▶  Start Recording",  callback=self._start)
        self.stop_item   = rumps.MenuItem("⏹  Stop & Transcribe", callback=self._stop)
        self.config_item = rumps.MenuItem("⚙  Set Output Directory…", callback=self._set_output_dir)
        self.autostart_item = rumps.MenuItem("🚀 Launch at Login", callback=self._toggle_autostart)
        
        self.status_item = rumps.MenuItem("● Idle")
        self.status_item.set_callback(None)
        self.quit_item   = rumps.MenuItem("Quit", callback=rumps.quit_application)

        self.menu = [
            self.start_item,
            self.stop_item,
            self.config_item,
            self.autostart_item,
            None,
            self.status_item,
            None,
            self.quit_item,
        ]

        self._update_autostart_menu()
        self._apply_state("IDLE")
        self._timer = rumps.Timer(self._tick, 1)

    # ── Menu callbacks ────────────────────────────────────────────────────────

    def _toggle_autostart(self, _):
        new_state = not get_autostart()
        set_autostart(new_state)
        if new_state:
            enable_autostart()
            self._set_status("🚀 Autostart enabled")
        else:
            disable_autostart()
            self._set_status("🚀 Autostart disabled")
        self._update_autostart_menu()

    def _update_autostart_menu(self):
        state = get_autostart()
        self.autostart_item.state = 1 if state else 0

    def _set_output_dir(self, _):
        script = 'tell app "System Events" to activate\n' \
                 'tell app "System Events" to return POSIX path of (choose folder with prompt "Select Output Directory for Notes & Audio:")'
        try:
            res = subprocess.check_output(["osascript", "-e", script]).decode("utf-8").strip()
            if res:
                set_output_dir(res)
                self._set_status(f"✓ Outputs → {Path(res).name}")
        except subprocess.CalledProcessError:
            pass # User cancelled dialog

    def _start(self, _):
        if self._state != "IDLE":
            return
        self._elapsed = 0
        
        summaries_dir = get_summaries_dir()
        summaries_dir.mkdir(parents=True, exist_ok=True)
        self._ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._live_out = summaries_dir / f"{self._ts}.md"
        
        self._live_out.write_text(f"# Meeting Transcript — {self._ts}\n\n*Recording in progress...*\n\n## Live Text\n\n", encoding="utf-8")
        subprocess.call(["open", str(self._live_out)])
        
        self._processed_frames = 0
        self._accumulated_transcript = ""

        self._apply_state("RECORDING")
        self._timer.start()
        self.recorder.start()
        threading.Thread(target=self._live_transcription_worker, daemon=True).start()

    def _live_transcription_worker(self):
        import time
        import numpy as np
        
        max_chunk = 16000 * 25
        min_chunk = 16000 * 5
        
        while True:
            is_recording = (self._state == "RECORDING")
            
            full_audio = self.recorder.get_current_audio()
            if full_audio is not None:
                unprocessed = full_audio[self._processed_frames:]
                frames = len(unprocessed)
                
                should_process = False
                if not is_recording and frames > 0:
                    should_process = True
                elif is_recording and frames >= max_chunk:
                    should_process = True
                elif is_recording and frames >= min_chunk:
                    recent_energy = np.mean(np.abs(unprocessed[-8000:]))
                    if recent_energy < 0.005: 
                        should_process = True
                        
                if should_process:
                    chunk = unprocessed.copy()
                    self._processed_frames += len(chunk)
                    text = self.transcriber.transcribe_live_chunk(chunk)
                    if text:
                        self._accumulated_transcript += " " + text
                        with open(self._live_out, "a", encoding="utf-8") as f:
                            f.write(text + " ")
                        
            if not is_recording:
                break
                
            time.sleep(1.0)

    def _stop(self, _):
        """Stop recording; hand off pipeline to background thread immediately."""
        if self._state != "RECORDING":
            return
        self._timer.stop()
        self._apply_state("PROCESSING")
        threading.Thread(
            target=self._pipeline, daemon=True
        ).start()

    # ── Pipeline (background thread) ──────────────────────────────────────────

    def _pipeline(self):
        # ── Step 1: Stop recorder ─────────────────────────────────────────────
        print("[pipeline] Stopping recorder…", flush=True)
        try:
            wav = self.recorder.stop()
            print(f"[pipeline] WAV: {wav}", flush=True)
        except Exception:
            traceback.print_exc()
            self._set_status("❌ Failed to save recording — see terminal")
            self._apply_state("IDLE")
            return

        # ── Step 2: Transcribe (with timestamps) ──────────────────────────────
        try:
            segments = self.transcriber.transcribe_segments(wav)
            total_words = sum(len(s.text.split()) for s in segments)
            print(f"[pipeline] Transcript: {total_words} words, {len(segments)} segments", flush=True)
        except Exception:
            traceback.print_exc()
            self._set_status("❌ Transcription failed — see terminal")
            self._apply_state("IDLE")
            return

        # ── Step 3: Speaker diarization (optional) ────────────────────────────
        try:
            self.diarizer.assign_speakers(wav, segments)
        except Exception:
            traceback.print_exc()
            self._set_status("⚠ Diarization failed — using timestamps only")

        # ── Step 4: Format & save ─────────────────────────────────────────────
        out = getattr(self, "_live_out", None)
        if out is None:
            summaries_dir = get_summaries_dir()
            summaries_dir.mkdir(parents=True, exist_ok=True)
            ts  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            out = summaries_dir / f"{ts}.md"

        body = _format_transcript(segments)
        total_words = sum(len(s.text.split()) for s in segments)
        
        # Overwrite the live text file with the final polished diarized format
        out.write_text(
            f"# Meeting Transcript — {self._ts if hasattr(self, '_ts') else ts}\n\n"
            f"*{total_words} words · {len(segments)} segments*\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        print(f"[pipeline] Saved/Overwritten: {out}", flush=True)
        self._set_status(f"✓ Done — {out.name}")
        subprocess.call(["open", str(out)])
        self._apply_state("IDLE")

    # ── Timer ─────────────────────────────────────────────────────────────────

    def _tick(self, _):
        self._elapsed += 1
        m, s = divmod(self._elapsed, 60)
        self.title = f"🔴 {m:02d}:{s:02d}"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_state(self, state: str):
        self._state = state
        print(f"[state] → {state}", flush=True)
        if state == "IDLE":
            self.title = "🎙"
            self.start_item.set_callback(self._start)
            self.stop_item.set_callback(None)
        elif state == "RECORDING":
            self.start_item.set_callback(None)
            self.stop_item.set_callback(self._stop)
        elif state == "PROCESSING":
            self.title = "⏳"
            self.start_item.set_callback(None)
            self.stop_item.set_callback(None)

    def _set_status(self, msg: str):
        print(f"[status] {msg}", flush=True)
        if hasattr(self, 'status_item'):
            self.status_item.title = msg


if __name__ == "__main__":
    NoteTakerApp().run()
