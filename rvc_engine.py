"""Run RVC voice conversion and training through the Applio install.

Every call is a subprocess under Applio's own interpreter (see rvc_paths for
why the processes cannot be shared). This module owns building the command
lines and reading the results honestly; it knows nothing about gradio or the
character library.

Applio's CLI prints progress to stdout and exits nonzero on failure, so the
contract here is simple: capture everything, and when a call fails, raise
with the tail of what it said -- a conversion that failed with no visible
reason is the unattended failure mode this repo keeps paying for.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

import rvc_paths

CONVERT_TIMEOUT_SECONDS = 600.0
TRAIN_STEP_TIMEOUT_SECONDS = 6 * 3600.0


class RVCError(RuntimeError):
    """An Applio call failed; the message carries what it printed."""


def _run_core(args: list[str], timeout: float, applio_root: Optional[str] = None) -> str:
    root = applio_root or rvc_paths.APPLIO_ROOT
    python = os.path.join(root, "env", "Scripts", "python.exe")
    core = os.path.join(root, "core.py")
    missing = [p for p in (python, core) if not os.path.exists(p)]
    if missing:
        raise RVCError(f"Applio install incomplete, missing: {', '.join(missing)}")

    result = subprocess.run(
        [python, core, *args],
        cwd=root,  # core.py resolves models and logs relative to its checkout
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise RVCError(f"Applio {args[0]} failed (exit {result.returncode}): {tail}")
    return result.stdout


def convert(input_path: str, output_path: str, model_path: str, index_path: str,
            transpose: int = 0, applio_root: Optional[str] = None) -> str:
    """Convert one audio file through a trained RVC model. Returns output_path.

    `transpose` is the character's stored pitch shift in semitones (rvc.transpose
    in character.json). rmvpe for pitch: Applio's default and the robust choice
    for speech.
    """
    for path, what in ((input_path, "input audio"), (model_path, "model .pth"),
                       (index_path, "index file")):
        if not os.path.isfile(path):
            raise RVCError(f"No such {what}: {path}")

    _run_core(
        [
            "infer",
            "--input-path", os.path.abspath(input_path),
            "--output-path", os.path.abspath(output_path),
            "--pth-path", os.path.abspath(model_path),
            "--index-path", os.path.abspath(index_path),
            "--pitch", str(int(transpose)),
            "--f0-method", "rmvpe",
            "--export-format", "WAV",
        ],
        timeout=CONVERT_TIMEOUT_SECONDS,
        applio_root=applio_root,
    )
    if not os.path.isfile(output_path):
        # Applio can exit 0 and write nothing when its own model loading is
        # misconfigured; the file existing is the claim that matters.
        raise RVCError(f"Applio infer exited cleanly but produced no file at {output_path}")
    return output_path


def train(model_name: str, dataset_dir: str, *, sample_rate: int = 40000,
          total_epochs: int = 200, batch_size: int = 8, gpu: str = "0",
          applio_root: Optional[str] = None) -> dict:
    """Run the full training chain: preprocess, extract, train, index.

    Blocks for the duration -- callers wanting a live UI run it on a worker
    thread or subprocess. Returns the trained artifact paths, verified to
    exist rather than assumed from a clean exit.
    """
    if not os.path.isdir(dataset_dir):
        raise RVCError(f"No dataset directory at {dataset_dir}")

    _run_core(["preprocess", "--model-name", model_name,
               "--dataset-path", os.path.abspath(dataset_dir),
               "--sample-rate", str(sample_rate)],
              timeout=TRAIN_STEP_TIMEOUT_SECONDS, applio_root=applio_root)
    _run_core(["extract", "--model-name", model_name,
               "--sample-rate", str(sample_rate), "--gpu", gpu],
              timeout=TRAIN_STEP_TIMEOUT_SECONDS, applio_root=applio_root)
    _run_core(["train", "--model-name", model_name,
               "--sample-rate", str(sample_rate),
               "--total-epoch", str(total_epochs),
               "--save-every-epoch", "10", "--save-only-latest",
               "--batch-size", str(batch_size), "--gpu", gpu,
               "--pretrained", "--vocoder", "HiFi-GAN"],
              timeout=TRAIN_STEP_TIMEOUT_SECONDS, applio_root=applio_root)
    _run_core(["index", "--model-name", model_name],
              timeout=TRAIN_STEP_TIMEOUT_SECONDS, applio_root=applio_root)

    return trained_artifacts(model_name, applio_root=applio_root)


def trained_artifacts(model_name: str, applio_root: Optional[str] = None) -> dict:
    """The .pth and .index a finished training run left behind.

    Verified on disk: a training chain that exited cleanly but wrote no
    weights must be reported as the failure it is.
    """
    root = applio_root or rvc_paths.APPLIO_ROOT
    model_dir = os.path.join(root, "logs", model_name)
    pth = None
    index = None
    if os.path.isdir(model_dir):
        for name in sorted(os.listdir(model_dir)):
            # Exported inference weights are {model_name}_{epoch}e_{step}s.pth;
            # G_*/D_* are training checkpoints and are not loadable for infer.
            # sorted() + last-wins keeps the highest epoch when several exist.
            if name.startswith(f"{model_name}_") and name.endswith(".pth"):
                pth = os.path.join(model_dir, name)
            elif name.endswith(".index"):
                index = os.path.join(model_dir, name)
    if pth is None or index is None:
        raise RVCError(
            f"Training left no usable artifacts for {model_name!r} in "
            f"{model_dir}: expected {model_name}_<epoch>e_<step>s.pth and a "
            f".index file. (A missing assets/config.json in the Applio "
            f"install makes it skip the final weight export -- see rvc_paths.)"
        )
    return {"model_path": pth, "index_path": index}
