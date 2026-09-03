"""Reading and writing saved UI presets.

A preset is one JSON file under PRESETS_DIR holding the state of every
control in the app, plus a small _meta block recording the format version
and when it was last used. This module owns the naming rules, the on-disk
layout and the last-used pointer -- nothing about what the values mean or
which widget each one belongs to.

Filesystem only: no gradio import, and the caller supplies and consumes
plain dicts, so the store can be exercised against a temp directory.
Split out of webui.py.
"""

import json
import os
from datetime import datetime
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))

PRESETS_DIR = os.path.join(current_dir, "presets")
os.makedirs(PRESETS_DIR, exist_ok=True)

UI_PRESET_VERSION = "1.0"
UI_PRESET_FORMAT = "indextts2_premium_ui"
DEFAULT_UI_PRESET_NAME = "default"
_LAST_USED_UI_PRESET_FILE = ".last_used_ui_preset.txt"
def _sanitize_preset_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))
    return safe.strip("._") or "default"


def _ui_preset_path(preset_name: str) -> Path:
    return Path(PRESETS_DIR) / f"{_sanitize_preset_name(preset_name)}.json"


def _list_ui_presets() -> List[str]:
    root = Path(PRESETS_DIR)
    saved = sorted(
        p.stem
        for p in root.glob("*.json")
        if p.is_file() and p.stem != DEFAULT_UI_PRESET_NAME
    )
    return [DEFAULT_UI_PRESET_NAME] + saved


def _set_last_used_ui_preset(preset_name: str) -> None:
    try:
        Path(PRESETS_DIR).mkdir(parents=True, exist_ok=True)
        (Path(PRESETS_DIR) / _LAST_USED_UI_PRESET_FILE).write_text(
            _sanitize_preset_name(preset_name),
            encoding="utf-8",
        )
    except OSError as exc:
        # If this write fails silently, the user's preset just stops being
        # remembered across restarts with no indication anywhere.
        print(
            f"Presets: could not record last-used preset {preset_name!r} ({exc}).",
            flush=True,
        )


def _get_last_used_ui_preset() -> Optional[str]:
    path = Path(PRESETS_DIR) / _LAST_USED_UI_PRESET_FILE
    if not path.exists():
        return None

    try:
        name = path.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    return name if name in _list_ui_presets() else None


def _save_ui_preset(preset_name: str, config: Dict[str, Any], now: Optional[Callable[[], datetime]] = None) -> str:
    # `now` is an injectable clock so the _meta timestamps are testable;
    # the default is the real one. §3.2.
    if now is None:
        now = datetime.now
    if not preset_name or not str(preset_name).strip():
        raise ValueError("Preset name cannot be empty.")

    safe_name = _sanitize_preset_name(preset_name)
    if safe_name == DEFAULT_UI_PRESET_NAME:
        raise ValueError(f"Preset name '{DEFAULT_UI_PRESET_NAME}' is reserved.")

    cfg = dict(config)
    cfg.setdefault("_meta", {})
    cfg["_meta"]["version"] = UI_PRESET_VERSION
    cfg["_meta"]["format"] = UI_PRESET_FORMAT
    cfg["_meta"]["last_modified"] = now().isoformat()
    if "created_at" not in cfg["_meta"]:
        cfg["_meta"]["created_at"] = cfg["_meta"]["last_modified"]

    out_path = _ui_preset_path(safe_name)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)
    _set_last_used_ui_preset(safe_name)
    return safe_name


def _load_ui_preset(preset_name: str) -> Optional[Dict[str, Any]]:
    if not preset_name or str(preset_name).strip() == DEFAULT_UI_PRESET_NAME:
        return None

    path = _ui_preset_path(preset_name)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    _set_last_used_ui_preset(preset_name)
    return data


def _delete_ui_preset(preset_name: str) -> bool:
    if not preset_name or str(preset_name).strip() == DEFAULT_UI_PRESET_NAME:
        return False

    path = _ui_preset_path(preset_name)
    if not path.exists():
        return False

    try:
        path.unlink()
        return True
    except Exception:
        return False
