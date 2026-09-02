"""Progress reporting for the VoiceForge web UI.

Two audiences, one module. `print_console_progress` and the duration
formatters write to the terminal the app was launched from;
`render_progress_bar` builds the HTML bar the browser shows, and
`_drain_progress_file` tails the progress file that the generation
subprocess writes, which is how progress crosses the process boundary.

No gradio import -- render_progress_bar returns a plain HTML string.
Split out of webui.py.
"""

import html
import json
import os
import time

import numpy as np

def current_timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def format_elapsed_duration(elapsed_seconds):
    elapsed_ms = max(0, int(round(float(elapsed_seconds) * 1000.0)))
    hours, remainder_ms = divmod(elapsed_ms, 3600000)
    minutes, remainder_ms = divmod(remainder_ms, 60000)
    seconds, milliseconds = divmod(remainder_ms, 1000)

    if hours:
        return f"{hours}h {minutes}m {seconds}.{milliseconds:03d}s"
    if minutes:
        return f"{minutes}m {seconds}.{milliseconds:03d}s"
    return f"{seconds}.{milliseconds:03d}s"


def print_console_progress(label, completed, total, started_at, processed_audio_seconds=None, item_label="item"):
    total = max(1, int(total))
    completed = min(total, max(0, int(completed)))
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    eta_seconds = ((elapsed_seconds / completed) * (total - completed)) if completed else None
    percent = 100.0 * completed / total
    parts = [
        f">> {label} {completed}/{total} {item_label}{'' if completed == 1 else 's'} ({percent:.1f}%)",
        f"elapsed {format_elapsed_duration(elapsed_seconds)}",
    ]
    if eta_seconds is not None:
        parts.append(f"eta {format_elapsed_duration(eta_seconds)}")
    if processed_audio_seconds is not None and processed_audio_seconds > 0 and elapsed_seconds > 0:
        parts.append(f"speed {processed_audio_seconds / elapsed_seconds:.2f}x RT")
        parts.append(f"audio {processed_audio_seconds:.2f}s")
    print(" | ".join(parts))


def audio_duration_ms(audio, sampling_rate):
    matrix = np.asarray(audio)
    if matrix.ndim == 0:
        return 0
    return int(round(matrix.shape[0] * 1000.0 / sampling_rate))
def render_progress_bar(fraction, message, done=False, failed=False):
    """Render an explicit progress bar.

    Gradio's own progress overlay proved easy to miss, especially on a phone,
    so the tab draws its own bar as part of the normal page flow. Colours are
    inline and theme-neutral so it reads on both light and dark.
    """
    percent = max(0, min(100, int(round((fraction or 0.0) * 100))))
    if failed:
        fill, label = "#dc2626", f"Failed - {message}"
    elif done:
        fill, label = "#16a34a", f"Done - {message}"
    else:
        fill, label = "#2563eb", f"{percent}% - {message}"
    safe = html.escape(label)
    return (
        '<div style="width:100%;font-family:system-ui,sans-serif;font-size:0.85rem;">'
        f'<div style="margin-bottom:4px;opacity:0.85;">{safe}</div>'
        '<div style="width:100%;height:14px;border-radius:7px;'
        'background:rgba(128,128,128,0.25);overflow:hidden;">'
        f'<div style="width:{percent}%;height:100%;background:{fill};'
        'transition:width 0.25s ease;"></div></div></div>'
    )
GENERATION_PROGRESS_POLL_SECONDS = 0.4
def _drain_progress_file(path, consumed, last_fraction):
    """Read progress lines written since `consumed` bytes.

    Returns (new_consumed, newest_fraction, newest_description). The
    description is None when nothing new arrived. Malformed or partially
    written lines are skipped rather than raised: the worker appends while we
    read, so a torn final line is expected and is not an error.
    """
    try:
        size = os.path.getsize(path)
    except OSError:
        return consumed, last_fraction, None
    if size <= consumed:
        return consumed, last_fraction, None

    # Binary, because `consumed` is a byte offset. In text mode on Windows a
    # seek offset is an opaque cookie and \n is stored as \r\n, so byte
    # arithmetic silently slips and events get skipped or replayed.
    try:
        with open(path, "rb") as handle:
            handle.seek(consumed)
            chunk = handle.read()
    except OSError as exc:
        print(f"could not read progress file: {type(exc).__name__}: {exc}")
        return consumed, last_fraction, None

    # Only whole lines are safe to parse; a trailing partial line is expected
    # because the worker appends while we read, so leave it for the next poll.
    complete, newline, _partial = chunk.rpartition(b"\n")
    if not newline:
        return consumed, last_fraction, None

    newest = None
    for raw in complete.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        value = event.get("value")
        if isinstance(value, (int, float)):
            last_fraction = min(max(float(value), 0.0), 1.0)
        newest = str(event.get("desc") or "Working...")

    return consumed + len(complete) + len(newline), last_fraction, newest
