"""Coercing saved preset values back into something a widget will accept.

A preset is JSON, so everything in it arrives as whatever json.load felt
like producing, and it may have been written by an older version of the
app with different fields. These turn a stored value into one the matching
control will take, falling back to the field's default rather than raising
when a value is missing, the wrong type, or no longer a valid choice.

Deliberately free of any reference to the widgets themselves -- that is
what lets them live here instead of inside the UI declaration. The parts
of the preset system that hold live components stayed behind in webui.py.

Split out of webui.py.
"""

from typing import Any, Dict, Optional

import gradio as gr

from subtitle_utils import parse_subtitle_file
from subtitle_render import build_subtitle_status_message
from webui_runtime import EMO_CHOICES_ALL

def _normalize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(default)

def _normalize_int(value: Any, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        normalized = int(float(value))
    except Exception:
        normalized = int(default)
    if min_value is not None:
        normalized = max(min_value, normalized)
    if max_value is not None:
        normalized = min(max_value, normalized)
    return normalized

def _normalize_float(value: Any, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        normalized = float(value)
    except Exception:
        normalized = float(default)
    if min_value is not None:
        normalized = max(min_value, normalized)
    if max_value is not None:
        normalized = min(max_value, normalized)
    return normalized

def _normalize_text(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    return str(value)

def _normalize_emotion_method(value: Any, default: int = 0) -> int:
    if hasattr(value, "value"):
        value = value.value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in EMO_CHOICES_ALL:
            return EMO_CHOICES_ALL.index(stripped)
    normalized = _normalize_int(value, default, 0, len(EMO_CHOICES_ALL) - 1)
    return normalized

def _normalize_field_value(field: Dict[str, Any], value: Any) -> Any:
    kind = field["kind"]
    default = field["default"]
    min_value = field.get("min")
    max_value = field.get("max")

    if kind == "str":
        return _normalize_text(value, default)
    if kind == "bool":
        return _normalize_bool(value, default)
    if kind == "int":
        return _normalize_int(value, default, min_value, max_value)
    if kind == "float":
        return _normalize_float(value, default, min_value, max_value)
    if kind == "int_text":
        return str(_normalize_int(value, int(default), min_value, max_value))
    if kind == "choice":
        normalized = _normalize_text(value, default)
        return normalized if normalized in field.get("choices", []) else default
    if kind == "emotion_method":
        return _normalize_emotion_method(value, default)
    return value if value is not None else default

def _component_output_value(field: Dict[str, Any], value: Any) -> Any:
    if field["kind"] == "emotion_method":
        index_value = _normalize_emotion_method(value, field["default"])
        return EMO_CHOICES_ALL[index_value]
    return value

def _build_subtitle_status_for_preset(subtitle_mode_value: bool, current_subtitle_file: Optional[str]):
    if not subtitle_mode_value or not current_subtitle_file:
        return gr.update(value="", visible=False)
    try:
        cues = parse_subtitle_file(current_subtitle_file)
        return gr.update(
            value=build_subtitle_status_message(cues, subtitle_file=current_subtitle_file),
            visible=True,
        )
    except Exception as e:
        return gr.update(value=f"Failed to load caption file: {str(e)}", visible=True)
