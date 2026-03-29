# macOS-Native-Transcription

A fully local, offline macOS menu bar application designed to seamlessly record system and microphone audio, transcribe it in real-time, and perform speaker diarization.

---

## Core Features

- **Native System Audio Capture**: Utilizes macOS 13+ ScreenCaptureKit via a custom Swift CLI to intercept raw PCM float32 system frames continuously synchronized with the hardware microphone clock.
- **On-The-Fly Transcription**: Features a VAD-based chunking framework that streams transcribed text locally to an active Markdown document while the meeting is still in progress.
- **Speaker Diarization**: Uses the `pyannote.audio` post-processing pipeline to isolate distinct speakers and correctly attribute finalized transcription segments natively.
- **Offline First**: All audio extraction and deep learning models execute entirely locally on your hardware. No audio is ever transmitted off-device.
- **Headless Menu Bar UI**: The entire application is managed via a native macOS menu bar status icon, eliminating floating interface clutter.

## Architecture

The application is composed of multiple concurrent components:

1. `tray_app.py`: The entry point `rumps` menu bar process.
2. `audio_recorder.py`: Manages blocking microphone pulses and parses asynchronous bursts from the Swift CLI, routing them into an overlapping `numpy` alignment buffer to mathematically ensure zero dropped hardware frames.
3. `transcriber.py`: Wraps `faster-whisper` for both live array stringing and final WAV file offline processing.
4. `diarizer.py`: Wraps the HuggingFace `pyannote/speaker-diarization-3.1` and segmentation pipelines.

## Installation

### Prerequisites
- macOS 13 (Ventura) or newer (mandatory requirement for ScreenCaptureKit libraries).
- Python 3.10+
- **Homebrew** with **FFmpeg** installed (required for audio decoding libraries):
  ```bash
  brew install ffmpeg
  ```
- A Hugging Face account with data access rights to the `pyannote` dependencies.

### Step 1: Environment Setup
Clone the repository and install the standard dependencies. This step will install approx 2GB of data including `torch`, `faster-whisper`, and `pyannote.audio`:

```bash
git clone https://github.com/far-light/macOS-Native-Transcription.git
cd macOS-Native-Transcription
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Hugging Face Token (Pyannote)
The application utilizes `faster-whisper` (base model) which operates without authentication. However, the speaker diarization model strictly requires an authenticated Hugging Face token.

1. Visit https://hf.co/pyannote/speaker-diarization-3.1 and https://hf.co/pyannote/segmentation-3.0 to review and accept the official user conditions.
2. Create an access token inside your Hugging Face portal settings.
3. Create a plaintext file named `.hf_token` at the absolute root of this project and place your literal token inside it:

```bash
echo "hf_YOUR_TOKEN_KEY" > .hf_token
```

### Step 3: Compiling the Swift Capture Interceptor
If the compiled executable `sck_audio` does not currently exist inside the `src/` directory, you must compile it utilizing the native macOS Swift compiler:

```bash
swiftc src/sck_audio.swift -o src/sck_audio
```

## Usage

You must execute the menu bar application as the primary thread in standard standard contexts:

```bash
.venv/bin/python src/tray_app.py
```

A generic microphone icon will appear inside your macOS menu bar. 

1. Click **Start Recording**. A Markdown file will be generated and automatically trigger your default Markdown viewer to open it natively. Transcribed text will securely stream into this document in real-time overlapping blocks.
2. Click **Set Output Directory...** to invoke a native macOS folder selection dialog. All ensuing recordings and markdown summaries will seamlessly route to your selected workspace path.
3. Click **Launch at Login** to toggle automatic application startup. This creates a native macOS LaunchAgent (.plist) linked to your current virtual environment.
4. Click **Stop & Transcribe**. The application will gracefully drain the recording buffer, initiate the offline diarization model to compute speaker permutations, and systematically overwrite the active Markdown document with precise timestamp metrics and final structured formatting.

## Operating System Limitations

Due to the fundamental structure of the `rumps` application frame buffer, the script must be run as the absolute foreground process on macOS, or as a native bundled Application. Launching it via IDE background diagnostic tasks may result in silent permission blocking when it attempts to query Screen Recording infrastructure mechanisms. 

Make sure to explicitly grant "Screen & System Audio Recording" and "Microphone" permissions when standard macOS security interfaces populate during the initial recording session.
