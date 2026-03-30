# macOS-Native-Transcription

A native macOS menu bar app for real-time meeting transcription and speaker diarization. 100% local, private, and powered by ScreenCaptureKit, Parakeet v3, and pyannote.audio.

---

## Core Features

- **Native System Audio Capture**: Utilizes macOS 13+ ScreenCaptureKit via a custom Swift CLI to intercept raw PCM float32 system frames continuously synchronized with the hardware microphone clock.
- **On-The-Fly Transcription**: Features a VAD-based chunking framework that streams transcribed text locally to an active Markdown document while the meeting is still in progress.
- **Existing Audio Import**: Can ingest pre-recorded audio files such as `mp3`, `aac`, `wav`, `m4a`, and any other FFmpeg-supported format, convert them locally, and run the same transcript + diarization pipeline.
- **Speaker Diarization**: Uses the `pyannote.audio` post-processing pipeline to isolate distinct speakers and correctly attribute finalized transcription segments natively.
- **Offline First**: All audio extraction and deep learning models execute entirely locally on your hardware. No audio is ever transmitted off-device.
- **Offline-First Model Cache**: Hugging Face models are now resolved from local cache first and only downloaded once if they are missing.
- **Headless Menu Bar UI**: The entire application is managed via a native macOS menu bar status icon, eliminating floating interface clutter.

## Architecture

The application is composed of multiple concurrent components:

1. `tray_app.py`: The entry point `rumps` menu bar process.
2. `audio_recorder.py`: Manages blocking microphone pulses and parses asynchronous bursts from the Swift CLI, routing them into an overlapping `numpy` alignment buffer to mathematically ensure zero dropped hardware frames.
3. `audio_importer.py`: Uses `ffmpeg` to normalize imported audio files into mono 16 kHz WAV before they enter the transcription pipeline.
4. `transcriber.py`: Wraps `parakeet-mlx` with the `mlx-community/parakeet-tdt-0.6b-v3` model for both live chunks and final WAV processing on Apple Silicon.
5. `diarizer.py`: Wraps the HuggingFace `pyannote/speaker-diarization-community-1` pipeline with Apple Silicon-oriented CPU preprocessing.

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
Clone the repository and install the standard dependencies. On first run, this will pull the MLX Parakeet v3 weights plus the diarization pipeline:

```bash
git clone https://github.com/far-light/macOS-Native-Transcription.git
cd macOS-Native-Transcription
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Hugging Face Token (Pyannote)
The application utilizes `parakeet-mlx` with `mlx-community/parakeet-tdt-0.6b-v3`, which runs locally on Apple Silicon without authentication. Speaker diarization still requires an authenticated Hugging Face token.

1. Visit https://hf.co/pyannote/speaker-diarization-community-1 to review and accept the official user conditions.
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
2. Click **Import Audio File...** to select an existing `mp3`, `aac`, `wav`, `m4a`, or any other FFmpeg-supported audio file. The app will convert it locally and run transcription plus speaker diarization into a Markdown summary.
3. Click **Set Output Directory...** to invoke a native macOS folder selection dialog. All ensuing recordings and markdown summaries will seamlessly route to your selected workspace path.
4. Click **Launch at Login** to toggle automatic application startup. This creates a native macOS LaunchAgent (.plist) linked to your current virtual environment.
5. Click **Stop & Transcribe**. The application will gracefully drain the recording buffer, initiate the offline diarization model to compute speaker permutations, and systematically overwrite the active Markdown document with precise timestamp metrics and final structured formatting.

## Operating System Limitations

Due to the fundamental structure of the `rumps` application frame buffer, the script must be run as the absolute foreground process on macOS, or as a native bundled Application. Launching it via IDE background diagnostic tasks may result in silent permission blocking when it attempts to query Screen Recording infrastructure mechanisms. 

Make sure to explicitly grant "Screen & System Audio Recording" and "Microphone" permissions when standard macOS security interfaces populate during the initial recording session.
