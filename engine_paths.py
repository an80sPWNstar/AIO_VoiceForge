"""Where the IndexTTS-2.5 engine lives, and what runs it.

The engine is a separate checkout with its own interpreter, not a package inside
this repo. It needs numpy 2.x and Python 3.11 while this app runs on numpy 1.26
and Python 3.10, so the two cannot share a process: the UI stays in this
interpreter and every generation happens in a subprocess under the engine's own.

Both sides of that boundary need the same four paths, so they are defined once
here rather than spelled out in webui.py and in the worker.

Set INDEXTTS25_ROOT to point at a different checkout.
"""
from __future__ import annotations

import os
import sys

DEFAULT_ENGINE_ROOT = r"D:\Index_TTS_v4\index-tts-2.5"

ENGINE_ROOT = os.environ.get("INDEXTTS25_ROOT") or DEFAULT_ENGINE_ROOT
ENGINE_PYTHON = os.path.join(ENGINE_ROOT, ".venv", "Scripts", "python.exe")
ENGINE_CHECKPOINTS = os.path.join(ENGINE_ROOT, "checkpoints")
ENGINE_CFG = os.path.join(ENGINE_CHECKPOINTS, "config.yaml")

# The environment variable the parent hands to the worker, so the worker resolves
# the same checkout even when the default has been overridden.
ENGINE_ROOT_ENV = "INDEXTTS25_ROOT"


def missing_engine_parts() -> list[str]:
    """Return the engine paths that do not exist, newest install problems first.

    Returning the list rather than raising lets the caller decide whether a
    missing engine is a startup error or a message in the UI.
    """
    return [
        path
        for path in (ENGINE_ROOT, ENGINE_PYTHON, ENGINE_CHECKPOINTS, ENGINE_CFG)
        if not os.path.exists(path)
    ]


def prepend_engine_to_sys_path() -> None:
    """Make `import indextts` resolve to the 2.5 checkout, not this repo's copy.

    Python puts the running script's own directory at sys.path[0], and this repo
    still carries an `indextts` package from the 2.0 engine, so appending is not
    enough — the engine root has to go in front of it.
    """
    if ENGINE_ROOT not in sys.path:
        sys.path.insert(0, ENGINE_ROOT)
    elif sys.path.index(ENGINE_ROOT) != 0:
        sys.path.remove(ENGINE_ROOT)
        sys.path.insert(0, ENGINE_ROOT)
