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
# and the .index; the final weights land as <model_name>_<epoch>e_<step>s.pth
# inside it.
APPLIO_LOGS = os.path.join(APPLIO_ROOT, "logs")

# Applio's settings file. It ships only as config_template.json and is created
# by their web UI on first launch -- which a CLI-only install never does. If
# it is missing, extract_model catches the read failure and SKIPS exporting
# the final weights, so a whole training run completes "successfully" and
# leaves nothing usable. Seed it from the template:
#   copy assets\config_template.json assets\config.json
APPLIO_SETTINGS = os.path.join(APPLIO_ROOT, "assets", "config.json")


def missing_applio_parts() -> list[str]:
    """The Applio paths that do not exist. Empty list means ready.

    A list rather than an exception, matching engine_paths: whether a missing
    trainer is a startup error or a line in the UI is the caller's decision.
    """
    return [
        path
        for path in (APPLIO_ROOT, APPLIO_PYTHON, APPLIO_CORE, APPLIO_SETTINGS)
        if not os.path.exists(path)
    ]
