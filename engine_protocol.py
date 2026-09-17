"""The line protocol between the UI process and the persistent engine worker.

Both halves import this so the framing cannot drift. One request is one JSON
line written to the worker's stdin; one reply is one sentinel line on its
stdout, printed only after the result file is closed.

Payloads stay in files rather than on the pipe. The worker's stdout also carries
the engine's own logging, and a multi-kilobyte JSON blob interleaved with
progress bars is not something a line reader can frame reliably. Keeping the
pipe to fixed single-token lines means a stray print can never be mistaken for a
result.
"""
from __future__ import annotations

import json

# Printed once when the worker is up and its stdin loop is running. The model is
# NOT loaded at this point: it loads on the first request, so that starting the
# worker never costs VRAM on its own.
READY_SENTINEL = "@@INDEXTTS_WORKER_READY@@"

# Printed after the result file for the current request has been written and
# closed. Seeing it means the file is safe to read.
DONE_SENTINEL = "@@INDEXTTS_WORKER_DONE@@"

# Asks the worker to drop the model and exit its loop cleanly.
SHUTDOWN_COMMAND = "shutdown"

# Asks the worker what device it is actually running on, without touching the
# loaded model. Used by the health endpoint to report a GPU that is verified
# from the process doing the work, not guessed at from the host.
DEVICE_QUERY_COMMAND = "device_query"


def encode_request(request_file: str, result_file: str, progress_file: str | None) -> str:
    """Return the single stdin line that asks the worker to run one generation."""
    return json.dumps(
        {
            "command": "generate",
            "request_file": request_file,
            "result_file": result_file,
            "progress_file": progress_file,
        },
        ensure_ascii=False,
    ) + "\n"


def encode_shutdown() -> str:
    """Return the single stdin line that asks the worker to exit."""
    return json.dumps({"command": SHUTDOWN_COMMAND}, ensure_ascii=False) + "\n"


def encode_device_query(result_file: str) -> str:
    """Return the single stdin line that asks the worker to report its device."""
    return json.dumps(
        {"command": DEVICE_QUERY_COMMAND, "result_file": result_file},
        ensure_ascii=False,
    ) + "\n"


def decode_message(line: str) -> dict:
    """Parse one stdin line into a message dict.

    Raises ValueError on anything unparseable, which the worker reports rather
    than guessing at: a malformed line means the two halves disagree, and
    continuing would silently drop a generation the UI is still waiting on.
    """
    text = (line or "").strip()
    if not text:
        raise ValueError("empty message")
    try:
        message = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed message line: {exc}") from exc
    if not isinstance(message, dict):
        raise ValueError(f"expected a JSON object, got {type(message).__name__}")
    return message
