"""Where the IndexTTS engine lives, and what runs it.

The engine is a separate checkout with its own interpreter, not a package inside
this repo. It needs numpy 2.x and a newer Python while this app runs on numpy
1.26 and Python 3.10, so the two cannot share a process: the UI stays in this
interpreter and every generation happens in a subprocess under the engine's own.

Both sides of that boundary need the same four paths, so they are defined once
here rather than spelled out in webui.py and in the worker.

The default is the SECourses V5 install (swapped 2026-09-05); set
INDEXTTS25_ROOT to point at a different checkout — the V4-era engine at
D:\\Index_TTS_v4\\index-tts-2.5 still works as a rollback target. Layout
differences between the two are detected rather than configured: V5 keeps its
venv in `venv/` and models in `models/`, the older checkout uses `.venv/` and
`checkpoints/`.
"""
from __future__ import annotations

import os
import sys

DEFAULT_ENGINE_ROOT = r"G:\Index_TTS_v5\Premium_IndexTTS2_SECourses"

ENGINE_ROOT = os.environ.get("INDEXTTS25_ROOT") or DEFAULT_ENGINE_ROOT


def _first_existing(*candidates: str) -> str:
    """The first candidate that exists, else the first candidate.

    Falling back to the first (rather than raising) keeps import safe on a
    machine without the engine; missing_engine_parts() is what reports it.
    """
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


ENGINE_PYTHON = _first_existing(
    os.path.join(ENGINE_ROOT, "venv", "Scripts", "python.exe"),
    os.path.join(ENGINE_ROOT, ".venv", "Scripts", "python.exe"),
    os.path.join(ENGINE_ROOT, "venv", "bin", "python"),
    os.path.join(ENGINE_ROOT, ".venv", "bin", "python"),
)
def _checkpoints_dir(root: str) -> str:
    """The model directory, decided by where config.yaml actually is.

    Bare directory existence is not enough: a V4-layout checkout can carry a
    stray models/ folder of auxiliary downloads beside the real checkpoints/,
    and picking it would point ENGINE_CFG at a config.yaml that isn't there.
    """
    candidates = [os.path.join(root, "models"), os.path.join(root, "checkpoints")]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "config.yaml")):
            return candidate
    return _first_existing(*candidates)


ENGINE_CHECKPOINTS = _checkpoints_dir(ENGINE_ROOT)
ENGINE_CFG = os.path.join(ENGINE_CHECKPOINTS, "config.yaml")

# The engine downloads auxiliary models (w2v-bert, CAMPPlus, BigVGAN) into an
# HF cache beside its checkpoints. The worker's environment must point there,
# or the first generation re-downloads gigabytes into the default cache.
ENGINE_HF_CACHE = os.path.join(ENGINE_CHECKPOINTS, "hf_cache")

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
    """Make `import indextts` resolve to the engine checkout, not this repo's copy.

    Python puts the running script's own directory at sys.path[0], and this repo
    still carries an `indextts` package from the 2.0 engine, so appending is not
    enough — the engine root has to go in front of it.
    """
    if ENGINE_ROOT not in sys.path:
        sys.path.insert(0, ENGINE_ROOT)
    elif sys.path.index(ENGINE_ROOT) != 0:
        sys.path.remove(ENGINE_ROOT)
        sys.path.insert(0, ENGINE_ROOT)
