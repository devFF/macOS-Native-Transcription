import os
import sys
from pathlib import Path

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.far-light.macos-native-transcription</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_executable}</string>
        <string>{script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>WorkingDirectory</key>
    <string>{working_dir}</string>
    <key>StandardOutPath</key>
    <string>/tmp/macos-native-transcription.stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/macos-native-transcription.stderr.log</string>
</dict>
</plist>
"""

def get_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / "com.far-light.macos-native-transcription.plist"

def enable_autostart():
    plist_path = get_plist_path()
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    
    python_executable = sys.executable
    script_path = str(Path(__file__).parent / "tray_app.py")
    working_dir = str(Path(__file__).parent.parent)
    
    content = PLIST_TEMPLATE.format(
        python_executable=python_executable,
        script_path=script_path,
        working_dir=working_dir
    )
    
    plist_path.write_text(content, encoding="utf-8")
    print(f"Autostart enabled: {plist_path}")

def disable_autostart():
    plist_path = get_plist_path()
    if plist_path.exists():
        plist_path.unlink()
        print(f"Autostart disabled: {plist_path}")
