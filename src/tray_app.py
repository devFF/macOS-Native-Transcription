"""
tray_app.py
-----------
All-in-one macOS tray app for AI Note-Taker.
No GUI window — everything controlled via the menu bar.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import rumps

sys.path.insert(0, str(Path(__file__).parent))

from audio_importer import AudioImportError, import_audio_to_wav
from audio_recorder import AudioRecorder
from autostart import disable_autostart, enable_autostart
from config import get_autostart, get_summaries_dir, set_autostart, set_output_dir
from diarizer import Diarizer
from transcriber import DEFAULT_MODEL as DEFAULT_ASR_MODEL, Transcriber


def _format_transcript(segments: list) -> str:
    """Render a list of segments into readable Markdown."""
    lines = ["## Transcript\n"]
    for seg in segments:
        lines.append(f"**[{seg.timestamp_range()}] {seg.speaker}:** {seg.text}")
    return "\n".join(lines)


class NoteTakerApp(rumps.App):
    def __init__(self, asr_model: str = DEFAULT_ASR_MODEL):
        super().__init__(name="macOS-Native-Transcription", title="🎙", quit_button=None)

        self.recorder = AudioRecorder(on_status=self._set_status)
        self.transcriber = Transcriber(model_id=asr_model, on_status=self._set_status)
        self.diarizer = Diarizer(on_status=self._set_status)

        self._state = "IDLE"
        self._elapsed = 0
        self._live_thread = None
        self._source_label = None

        self.start_item = rumps.MenuItem("▶  Start Recording", callback=self._start)
        self.stop_item = rumps.MenuItem("⏹  Stop & Transcribe", callback=self._stop)
        self.import_item = rumps.MenuItem("📂 Import Audio File…", callback=self._import_audio)
        self.config_item = rumps.MenuItem("⚙  Set Output Directory…", callback=self._set_output_dir)
        self.autostart_item = rumps.MenuItem("🚀 Launch at Login", callback=self._toggle_autostart)
        self.status_item = rumps.MenuItem("● Idle")
        self.status_item.set_callback(None)
        self.quit_item = rumps.MenuItem("Quit", callback=rumps.quit_application)

        self.menu = [
            self.start_item,
            self.stop_item,
            self.import_item,
            self.config_item,
            self.autostart_item,
            None,
            self.status_item,
            None,
            self.quit_item,
        ]

        self._update_autostart_menu()
        self._timer = rumps.Timer(self._tick, 1)
        self._apply_state("IDLE")

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
        self.autostart_item.state = 1 if get_autostart() else 0

    def _set_output_dir(self, _):
        script = (
            'tell app "System Events" to activate\n'
            'tell app "System Events" to return POSIX path of '
            '(choose folder with prompt "Select Output Directory for Notes & Audio:")'
        )
        try:
            res = subprocess.check_output(["osascript", "-e", script]).decode("utf-8").strip()
            if res:
                set_output_dir(res)
                self._set_status(f"✓ Outputs → {Path(res).name}")
        except subprocess.CalledProcessError:
            pass

    def _start(self, _):
        if self._state != "IDLE":
            return
        self._elapsed = 0
        self._source_label = None
        self._live_out = self._prepare_summary_shell(
            title_prefix="Meeting Transcript",
            body_intro="*Recording in progress...*\n\n## Live Text\n\n",
        )
        subprocess.call(["open", str(self._live_out)])

        self._processed_frames = 0
        self._accumulated_transcript = ""

        self._apply_state("RECORDING")
        self._timer.start()
        self.recorder.start()
        self._live_thread = threading.Thread(target=self._live_transcription_worker, daemon=True)
        self._live_thread.start()

    def _live_transcription_worker(self):
        import time
        import numpy as np

        max_chunk = 16000 * 25
        min_chunk = 16000 * 5

        while True:
            is_recording = self._state == "RECORDING"

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
                        with open(self._live_out, "a", encoding="utf-8") as handle:
                            handle.write(text + " ")

            if not is_recording:
                break

            time.sleep(1.0)

    def _stop(self, _):
        if self._state != "RECORDING":
            return
        self._apply_state("PROCESSING")
        threading.Thread(target=self._pipeline, daemon=True).start()

    def _import_audio(self, _):
        if self._state != "IDLE":
            return

        script = (
            'tell app "System Events" to activate\n'
            'tell app "System Events" to return POSIX path of '
            '(choose file with prompt "Select an audio file to transcribe and diarize:")'
        )
        try:
            selected = subprocess.check_output(["osascript", "-e", script]).decode("utf-8").strip()
        except subprocess.CalledProcessError:
            return

        source_path = Path(selected)
        self._elapsed = 0
        self._source_label = source_path.name
        self._live_out = self._prepare_summary_shell(
            title_prefix="Imported Audio Transcript",
            body_intro=(
                f"*Importing and processing `{source_path.name}`...*\n\n"
                "## Transcript\n\n"
            ),
        )
        subprocess.call(["open", str(self._live_out)])

        self._apply_state("PROCESSING")
        threading.Thread(target=self._import_pipeline, args=(source_path,), daemon=True).start()

    def _pipeline(self):
        print("[pipeline] Stopping recorder…", flush=True)
        try:
            wav = self.recorder.stop()
            print(f"[pipeline] WAV: {wav}", flush=True)
        except Exception:
            traceback.print_exc()
            self._set_status("❌ Failed to save recording — see terminal")
            self._apply_state("IDLE")
            return

        self._wait_for_live_transcription()
        self._process_audio_pipeline(wav)

    def _import_pipeline(self, source_path: Path):
        try:
            wav = import_audio_to_wav(source_path, on_status=self._set_status)
            print(f"[import] WAV: {wav}", flush=True)
        except AudioImportError as exc:
            self._set_status(f"❌ {exc}")
            self._apply_state("IDLE")
            return
        except Exception:
            traceback.print_exc()
            self._set_status("❌ Audio import failed — see terminal")
            self._apply_state("IDLE")
            return

        self._process_audio_pipeline(wav)

    def _process_audio_pipeline(self, wav: Path):
        try:
            segments = self.transcriber.transcribe_segments(wav)
            total_words = sum(len(s.text.split()) for s in segments)
            print(f"[pipeline] Transcript: {total_words} words, {len(segments)} segments", flush=True)
        except Exception:
            traceback.print_exc()
            self._set_status("❌ Transcription failed — see terminal")
            self._apply_state("IDLE")
            return

        try:
            self.diarizer.assign_speakers(wav, segments)
        except Exception:
            traceback.print_exc()
            self._set_status("⚠ Diarization failed — using timestamps only")

        out = getattr(self, "_live_out", None)
        if out is None:
            out = self._prepare_summary_shell(
                title_prefix="Meeting Transcript",
                body_intro="## Transcript\n\n",
                open_file=False,
            )

        body = _format_transcript(segments)
        total_words = sum(len(s.text.split()) for s in segments)
        source_meta = f"*Source: {self._source_label}*\n\n" if self._source_label else ""

        out.write_text(
            f"# Transcript — {self._ts}\n\n"
            f"{source_meta}"
            f"*{total_words} words · {len(segments)} segments*\n\n"
            f"{body}\n",
            encoding="utf-8",
        )
        print(f"[pipeline] Saved/Overwritten: {out}", flush=True)
        self._set_status(f"✓ Done — {out.name}")
        subprocess.call(["open", str(out)])
        self._apply_state("IDLE")

    def _tick(self, _):
        self._elapsed += 1
        m, s = divmod(self._elapsed, 60)
        if self._state == "RECORDING":
            self.title = f"🔴 {m:02d}:{s:02d}"

    def _apply_state(self, state: str):
        self._state = state
        print(f"[state] → {state}", flush=True)
        if state == "IDLE":
            self._timer.stop()
            self.title = "🎙"
            self.start_item.set_callback(self._start)
            self.stop_item.set_callback(None)
            self.import_item.set_callback(self._import_audio)
            self._set_status("● Idle")
        elif state == "RECORDING":
            self.start_item.set_callback(None)
            self.stop_item.set_callback(self._stop)
            self.import_item.set_callback(None)
            self._set_status("● Recording…")
        elif state == "PROCESSING":
            self._elapsed = 0
            self.title = "⏳"
            self.start_item.set_callback(None)
            self.stop_item.set_callback(None)
            self.import_item.set_callback(None)
            if not self._timer.is_alive():
                self._timer.start()

    def _set_status(self, msg: str):
        print(f"[status] {msg}", flush=True)
        if hasattr(self, "status_item"):
            self.status_item.title = msg

    def _wait_for_live_transcription(self):
        live_thread = getattr(self, "_live_thread", None)
        if not live_thread:
            return
        if live_thread.is_alive():
            self._set_status("Finishing live transcription…")
            live_thread.join()
        self._live_thread = None

    def _prepare_summary_shell(
        self,
        title_prefix: str,
        body_intro: str,
        open_file: bool = False,
    ) -> Path:
        summaries_dir = get_summaries_dir()
        summaries_dir.mkdir(parents=True, exist_ok=True)
        self._ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = summaries_dir / f"{self._ts}.md"
        out.write_text(
            f"# {title_prefix} — {self._ts}\n\n{body_intro}",
            encoding="utf-8",
        )
        if open_file:
            subprocess.call(["open", str(out)])
        return out


if __name__ == "__main__":
    NoteTakerApp().run()
