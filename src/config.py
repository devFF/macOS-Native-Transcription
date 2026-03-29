import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent.parent / "config.json"

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text("utf-8"))
        except:
            return {}
    return {}

def save_config(config: dict):
    CONFIG_FILE.write_text(json.dumps(config, indent=4), "utf-8")

def get_base_dir() -> Path:
    cfg = load_config()
    custom_path = cfg.get("output_dir")
    if custom_path:
        p = Path(custom_path)
        if p.exists() and p.is_dir():
            return p
    return Path(__file__).parent.parent

def get_recordings_dir() -> Path:
    return get_base_dir() / "recordings"

def get_summaries_dir() -> Path:
    return get_base_dir() / "summaries"

def set_output_dir(path: str):
    cfg = load_config()
    cfg["output_dir"] = path
    save_config(cfg)

def get_autostart() -> bool:
    return load_config().get("autostart", False)

def set_autostart(enabled: bool):
    cfg = load_config()
    cfg["autostart"] = enabled
    save_config(cfg)
