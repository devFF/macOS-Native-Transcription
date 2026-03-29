"""
audio_recorder.py
-----------------
Captures mic via sounddevice + system audio via native ScreenCaptureKit CLI.
Mixes both streams in real-time and writes a single mono WAV file.
"""

import os
import queue
import threading
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from config import get_recordings_dir

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"


class AudioRecorder:
    def __init__(self, on_status: Callable[[str], None] = print):
        self.on_status = on_status
        self._mic_queue: queue.Queue = queue.Queue()
        self._sys_queue: queue.Queue = queue.Queue()
        self._recording = False
        self._start_time: Optional[float] = None
        self._wav_path: Optional[Path] = None
        
        self._mix_thread: Optional[threading.Thread] = None
        self._sck_process: Optional[subprocess.Popen] = None
        self._sck_thread: Optional[threading.Thread] = None
        
        # Buffer for live transcription polling
        self._audio_lock = threading.Lock()
        self._audio_data = []

    def get_current_audio(self) -> Optional[np.ndarray]:
        """Returns a flat copy of all accumulated audio for live processing."""
        with self._audio_lock:
            if not self._audio_data:
                return None
            return np.concatenate(self._audio_data, axis=0)

    def start(self) -> None:
        if self._recording:
            return

        r_dir = get_recordings_dir()
        r_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._wav_path = r_dir / f"{timestamp}.wav"
        
        self._recording = True

        # Clear queues
        while not self._mic_queue.empty(): self._mic_queue.get_nowait()
        while not self._sys_queue.empty(): self._sys_queue.get_nowait()

        # Start mixer thread
        with self._audio_lock:
            self._audio_data = []
            
        self._mix_thread = threading.Thread(target=self._mix_and_write, daemon=True)
        self._mix_thread.start()

        # Start native Mic stream
        try:
            self._mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._mic_callback,
            )
            self._mic_stream.start()
        except Exception as e:
            self.on_status(f"⚠ Mic failed: {e}")
            self._recording = False
            return

        # Start native System Audio stream (ScreenCaptureKit)
        sck_bin = Path(__file__).parent / "sck_audio"
        if sck_bin.exists():
            try:
                self._sck_process = subprocess.Popen(
                    [str(sck_bin)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self._sck_thread = threading.Thread(target=self._sck_reader, daemon=True)
                self._sck_thread.start()
                self.on_status("Native macOS System Audio linked.")
            except Exception as e:
                self.on_status(f"⚠ System audio capture failed: {e}")
        else:
            self.on_status("⚠ sck_audio executable missing — Mic only")

        self._start_time = time.time()
        self.on_status("● Recording…")

    def stop(self) -> Path:
        if not self._recording:
            raise RuntimeError("Not recording.")
        self._recording = False

        self._mic_stream.stop()
        self._mic_stream.close()

        if self._sck_process:
            self._sck_process.terminate()
            try:
                self._sck_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._sck_process.kill()

        self._mic_queue.put(None)
        self._mix_thread.join(timeout=10)

        duration = time.time() - self._start_time
        self.on_status(f"Recording saved → {self._wav_path.name}  ({duration:.0f}s)")
        return self._wav_path

    @property
    def elapsed(self) -> float:
        if self._recording and self._start_time:
            return time.time() - self._start_time
        return 0.0

    def _mic_callback(self, indata, frames, time_info, status):
        if self._recording:
            self._mic_queue.put(indata.copy())

    def _sck_reader(self):
        """Reads raw 32-bit float PCM from the Swift CLI stdout and queues it."""
        try:
            rate_line = self._sck_process.stdout.readline().decode('utf-8', 'ignore').strip()
            chan_line = self._sck_process.stdout.readline().decode('utf-8', 'ignore').strip()
            bits_line = self._sck_process.stdout.readline().decode('utf-8', 'ignore').strip()
            float_line = self._sck_process.stdout.readline().decode('utf-8', 'ignore').strip()
            
            orig_sr = int(rate_line.split(":")[1])
            orig_ch = int(chan_line.split(":")[1])
            orig_bits = int(bits_line.split(":")[1])
            is_float = int(float_line.split(":")[1]) == 1
            
            if is_float and orig_bits == 32:
                sck_dtype = np.float32
            elif not is_float and orig_bits == 16:
                sck_dtype = np.int16
            elif not is_float and orig_bits == 32:
                sck_dtype = np.int32
            else:
                sck_dtype = np.float32
            
            print(f"[recorder] SCK Linked: {orig_sr}Hz {orig_ch}ch {orig_bits}bit {'Float' if is_float else 'Int'}", flush=True)
            
            bytes_per_frame = (orig_bits // 8) * orig_ch
        except Exception as e:
            print(f"[sck_reader] Failed to read header: {e}", flush=True)
            return

        while self._recording and self._sck_process:
            try:
                # Read chunks aligned to frames
                chunk = self._sck_process.stdout.read(4096 * bytes_per_frame)
                if not chunk:
                    break
                rem = len(chunk) % bytes_per_frame
                if rem > 0:
                    chunk = chunk[:-rem]
                if len(chunk) > 0:
                    audio = np.frombuffer(chunk, dtype=sck_dtype)
                    if sck_dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    elif sck_dtype == np.int32:
                        audio = audio.astype(np.float32) / 2147483648.0
                    
                    audio = audio.reshape(-1, orig_ch)
                    
                    # Convert to mono
                    if orig_ch > 1:
                        audio = audio.mean(axis=1, keepdims=True)
                    else:
                        audio = audio.reshape(-1, 1)
                        
                    # Resample to 16000 Hz if necessary
                    if orig_sr != SAMPLE_RATE:
                        orig_len = len(audio)
                        duration = orig_len / orig_sr
                        target_len = int(duration * SAMPLE_RATE)
                        x_old = np.linspace(0, duration, orig_len)
                        x_new = np.linspace(0, duration, target_len)
                        audio = np.interp(x_new, x_old, audio.flatten()).reshape(-1, 1)
                        
                    self._sys_queue.put(audio)
            except Exception as e:
                print(f"[sck_reader] Error: {e}", flush=True)
                break

    def _mix_and_write(self):
        try:
            chunks = []
            sys_buffer = np.zeros((0, 1), dtype=np.float32)
            
            while True:
                # Master Clock: Microphone Pulse
                mic_chunk = self._mic_queue.get()
                if mic_chunk is None:
                    break
                
                # Ingest any arrived SCK burst frames
                while not self._sys_queue.empty():
                    try:
                        burst = self._sys_queue.get_nowait()
                        sys_buffer = np.concatenate([sys_buffer, burst], axis=0)
                    except queue.Empty:
                        break
                
                mic_len = len(mic_chunk)
                
                if len(sys_buffer) >= mic_len:
                    # Plentiful sys frames: pop exactly mic_len
                    sys_mix = sys_buffer[:mic_len]
                    sys_buffer = sys_buffer[mic_len:]  # Persist remainder overflow
                else:
                    # Starved sys frames (or paused SCK): pad with silence
                    sys_mix = np.zeros((mic_len, 1), dtype=np.float32)
                    if len(sys_buffer) > 0:
                        sys_mix[:len(sys_buffer)] = sys_buffer
                    sys_buffer = np.zeros((0, 1), dtype=np.float32)
                
                # Combine flawlessly aligned frames
                mixed = (mic_chunk + sys_mix) / 2.0
                chunks.append(mixed)
                with self._audio_lock:
                    self._audio_data.append(mixed)

            if chunks:
                audio = np.concatenate(chunks, axis=0)
                sf.write(str(self._wav_path), audio, SAMPLE_RATE)
            else:
                sf.write(str(self._wav_path), np.zeros((SAMPLE_RATE, 1), dtype=DTYPE), SAMPLE_RATE)
                
            print(f"[recorder] WAV written: {self._wav_path.name} ({self._wav_path.stat().st_size // 1024} KB)", flush=True)

        except Exception:
            import traceback
            traceback.print_exc()
