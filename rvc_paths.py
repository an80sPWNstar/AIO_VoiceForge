"""Where the Applio RVC install lives, and what runs it.

Same arrangement as engine_paths.py for the same reason: Applio is a separate
checkout with its own interpreter (torch 2.11/cu128 against this app's older
stack), so conversion and training run as subprocesses under Applio's python
and never share a process with the UI.

Set VOICEFORGE_APPLIO_ROOT to point at a different install.
"""
from __future__ import annotations

import os

DEFAULT_APPLIO_ROOT = r"D:\Applio"

APPLIO_ROOT = os.environ.get("VOICEFORGE_APPLIO_ROOT") or DEFAULT_APPLIO_ROOT
APPLIO_PYTHON = os.path.join(APPLIO_ROOT, "env", "Scripts", "python.exe")
APPLIO_CORE = os.path.join(APPLIO_ROOT, "core.py")
# Where Applio writes trained models: logs/<model_name>/ holds the checkpoints
# and the .index; the final weights land as <model_name>.pth inside it.
APPLIO_LOGS = os.path.join(APPLIO_ROOT, "logs")


def missing_applio_parts() -> list[str]:
    """The Applio paths that do not exist. Empty list means ready.

    A list rather than an exception, matching engine_paths: whether a missing
    trainer is a startup error or a line in the UI is the caller's decision.
    """
    return [
        path
        for path in (APPLIO_ROOT, APPLIO_PYTHON, APPLIO_CORE)
        if not os.path.exists(path)
    ]
