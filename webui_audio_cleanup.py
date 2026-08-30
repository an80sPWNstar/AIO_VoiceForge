"""Drives the audio cleanup worker from the web UI process.

The cleanup stack (audio-separator, speechbrain, silero-vad and their own
torch) lives in a SIDECAR virtualenv next to the app, not in the app's venv.
This app's environment is pinned hard -- gradio 6.17.3, transformers 4.52.1,
huggingface-hub<1.0 -- and has already been broken twice by a dependency
upgrade that arrived as a side effect of installing something else. The cleanup
stack wants huggingface-hub 1.29 and numpy 2.x, both of which would break the
TTS engine. Keeping it in another interpreter makes that collision impossible
rather than merely unlikely.

This module speaks to that interpreter over a subprocess boundary, and mirrors
the callback shape of webui_media_fetch so the UI code for both tabs looks the
same.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import audio_cleanup_shared as shared
import webui_media_fetch as media_fetch


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# The sidecar venv sits beside the repo, in the installer directory, so a
# `git clean` in the repo cannot delete several GB of environment.
SIDECAR_VENV_DIRNAME = "venv_audioclean"
SIDECAR_PYTHON_RELATIVE = os.path.join("Scripts", "python.exe")
SIDECAR_PYTHON_POSIX_RELATIVE = os.path.join("bin", "python")

WORKER_SCRIPT = "audio_cleanup_worker.py"

# Environment variable that overrides venv discovery, for a non-standard layout.
SIDECAR_PYTHON_ENV_VAR = "AUDIO_CLEANUP_PYTHON"

# How often the parent re-reads the worker's progress file.
PROGRESS_POLL_SECONDS = 0.25

# Cleanup of a long clip is slow but bounded; a run past this is hung.
CLEANUP_TIMEOUT_SECONDS = 7200

SETUP_HINT = (
    "The audio cleanup environment is not installed. Run "
    "install_audio_cleanup.bat from the app folder "
    "(a one-time ~4 GB download), then restart the app."
)


class CleanupError(RuntimeError):
    """A cleanup run failed for a reason worth showing the user verbatim."""


@dataclass(frozen=True)
class CleanupProgress:
    """One progress event from the worker. `fraction` is 0..1 or None."""

    stage: Optional[str]
    message: str
    fraction: Optional[float] = None


ProgressCallback = Callable[[CleanupProgress], None]


# --------------------------------------------------------------------------
# Environment discovery
# --------------------------------------------------------------------------

def _installer_root() -> str:
    """Return the directory holding the installer scripts and the sidecar venv."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sidecar_python() -> Optional[str]:
    """Return the sidecar interpreter path, or None when it is not installed."""
    override = os.environ.get(SIDECAR_PYTHON_ENV_VAR)
    if override:
        return override if os.path.isfile(override) else None

    venv = os.path.join(_installer_root(), SIDECAR_VENV_DIRNAME)
    for relative in (SIDECAR_PYTHON_RELATIVE, SIDECAR_PYTHON_POSIX_RELATIVE):
        candidate = os.path.join(venv, relative)
        if os.path.isfile(candidate):
            return candidate
    return None


def model_cache_dir() -> str:
    """Return the directory separation models are downloaded into."""
    return os.path.join(_installer_root(), shared.MODEL_CACHE_DIRNAME)


def cleanup_available() -> bool:
    """True when a cleanup run could actually start right now."""
    return sidecar_python() is not None and media_fetch.ffmpeg_available()


def cleanup_environment_note() -> str:
    """One-line readiness note for the UI."""
    python_path = sidecar_python()
    if python_path is None:
        return "Audio cleanup **not installed** - " + SETUP_HINT
    parts = ["Audio cleanup **ready**"]
    device = _probe_device(python_path)
    parts.append(f"compute: **{device}**")
    return " | ".join(parts)


def _probe_device(python_path: str) -> str:
    """Ask the sidecar interpreter whether it can see a GPU.

    Reported rather than assumed: the sidecar has its own torch build, so the
    app's CUDA availability says nothing about the cleanup environment's.
    """
    probe = (
        "import torch;"
        "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
    )
    try:
        result = subprocess.run(
            [python_path, "-c", probe],
            capture_output=True, text=True, timeout=media_fetch.FFPROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unknown ({type(exc).__name__})"
    if result.returncode != 0:
        return f"unknown (probe exited {result.returncode})"
    return result.stdout.strip() or "unknown"


# --------------------------------------------------------------------------
# Progress file tailing
# --------------------------------------------------------------------------

class _ProgressTail:
    """Reads newly-appended JSON lines from the worker's progress file.

    Opened in binary mode and tracked by byte offset because the worker writes
    with newline="" -- text mode would translate line endings back and desync
    the offsets by a byte per line.
    """

    def __init__(self, path: str):
        self._path = path
        self._offset = 0
        self._partial = b""

    def drain(self) -> List[CleanupProgress]:
        if not os.path.exists(self._path):
            return []
        with open(self._path, "rb") as handle:
            handle.seek(self._offset)
            chunk = handle.read()
            self._offset = handle.tell()

        events = []
        buffer = self._partial + chunk
        *lines, self._partial = buffer.split(b"\n")
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                events.append(CleanupProgress(
                    None, f"Unreadable progress line skipped: {exc}"))
                continue
            events.append(CleanupProgress(
                stage=payload.get("stage"),
                message=payload.get("message", ""),
                fraction=payload.get("fraction"),
            ))
        return events


# --------------------------------------------------------------------------
# Running a cleanup
# --------------------------------------------------------------------------

def build_worker_command(
    python_path: str,
    input_path: str,
    output_path: str,
    stages: List[str],
    progress_path: str,
    vocal_model: str,
    dereverb_model: str,
    denoise_model: str,
    speaker_mode: str,
    speaker_sample: Optional[str],
    speaker_threshold: float,
    sample_rate: int,
    channel_mode: str,
    device: str,
    keep_intermediates: bool,
) -> List[str]:
    """Build the worker argv. Pure: builds a list, runs nothing."""
    command = [
        python_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKER_SCRIPT),
        "--input", input_path,
        "--output", output_path,
        "--stages", ",".join(stages),
        "--progress-file", progress_path,
        "--model-dir", model_cache_dir(),
        "--device", device,
        "--vocal-model", vocal_model,
        "--dereverb-model", dereverb_model,
        "--denoise-model", denoise_model,
        "--speaker-mode", speaker_mode,
        "--speaker-threshold", str(speaker_threshold),
        "--sample-rate", str(sample_rate),
        "--channel-mode", channel_mode,
    ]
    if speaker_sample:
        command.extend(["--speaker-sample", speaker_sample])
    if keep_intermediates:
        command.append("--keep-intermediates")
    return command


def cleaned_output_path(source_path: str, output_root: str) -> str:
    """Return a unique path for the cleaned version of `source_path`."""
    directory = os.path.join(output_root, shared.CLEANUP_SUBDIR)
    os.makedirs(directory, exist_ok=True)
    stem = media_fetch.sanitize_filename(
        os.path.splitext(os.path.basename(source_path))[0] + shared.CLEANUP_SUFFIX
    )
    return media_fetch.unique_path(directory, stem, shared.CLEANUP_EXTENSION)


def run_cleanup(
    input_path: str,
    output_root: str,
    stages: List[str],
    vocal_model: str = shared.DEFAULT_VOCAL_MODEL,
    dereverb_model: str = shared.DEFAULT_DEREVERB_MODEL,
    denoise_model: str = shared.DEFAULT_DENOISE_MODEL,
    speaker_mode: str = shared.DEFAULT_SPEAKER_MODE,
    speaker_sample: Optional[str] = None,
    speaker_threshold: float = shared.DEFAULT_SPEAKER_THRESHOLD,
    sample_rate: int = media_fetch.DEFAULT_SAMPLE_RATE,
    channel_mode: str = media_fetch.CHANNEL_MONO,
    device: str = shared.DEFAULT_DEVICE,
    keep_intermediates: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict:
    """Clean `input_path` and return {"audio_path", "notes", "log"}.

    Blocking. Emits CleanupProgress events through `progress_callback` as the
    worker reports them.
    """
    if not input_path or not os.path.isfile(input_path):
        raise CleanupError("Download or extract an audio file first.")

    ordered = shared.ordered_stages(stages)
    if not ordered:
        raise CleanupError("Pick at least one cleanup step.")

    python_path = sidecar_python()
    if python_path is None:
        raise CleanupError(SETUP_HINT)
    if not media_fetch.ffmpeg_available():
        raise CleanupError("ffmpeg is not on PATH, so the cleaned file cannot be written.")
    if speaker_mode == shared.SPEAKER_MODE_SAMPLE and shared.STAGE_SPEAKER in ordered:
        if not speaker_sample or not os.path.isfile(speaker_sample):
            raise CleanupError(
                "Speaker mode is set to 'match a voice sample' but no sample "
                "file was provided."
            )

    output_path = cleaned_output_path(input_path, output_root)
    scratch = tempfile.mkdtemp(prefix="audio_cleanup_")
    progress_path = os.path.join(scratch, "progress.jsonl")
    stdout_path = os.path.join(scratch, "stdout.txt")
    stderr_path = os.path.join(scratch, "stderr.txt")

    command = build_worker_command(
        python_path=python_path,
        input_path=os.path.abspath(input_path),
        output_path=output_path,
        stages=ordered,
        progress_path=progress_path,
        vocal_model=vocal_model,
        dereverb_model=dereverb_model,
        denoise_model=denoise_model,
        speaker_mode=speaker_mode,
        speaker_sample=os.path.abspath(speaker_sample) if speaker_sample else None,
        speaker_threshold=speaker_threshold,
        sample_rate=sample_rate,
        channel_mode=channel_mode,
        device=device,
        keep_intermediates=keep_intermediates,
    )

    def report(event: CleanupProgress) -> None:
        if progress_callback is not None:
            progress_callback(event)

    report(CleanupProgress(None, "Starting the cleanup engine", 0.0))

    tail = _ProgressTail(progress_path)
    log: List[str] = []

    # Both streams go to FILES rather than pipes: the separation library logs
    # steadily, and a pipe nobody is draining fills its buffer and deadlocks
    # the worker at some unpredictable point mid-run.
    try:
        with open(stdout_path, "wb") as out, open(stderr_path, "wb") as err:
            process = subprocess.Popen(
                command,
                stdout=out,
                stderr=err,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            deadline = time.monotonic() + CLEANUP_TIMEOUT_SECONDS
            while process.poll() is None:
                for event in tail.drain():
                    log.append(event.message)
                    report(event)
                if time.monotonic() > deadline:
                    process.kill()
                    raise CleanupError(
                        f"Cleanup did not finish within "
                        f"{CLEANUP_TIMEOUT_SECONDS // 60} minutes and was stopped."
                    )
                time.sleep(PROGRESS_POLL_SECONDS)

            for event in tail.drain():
                log.append(event.message)
                report(event)

        result = _read_result(stdout_path, stderr_path, process.returncode)
        notes = result.get("notes", [])
        report(CleanupProgress(None, "Cleanup finished", 1.0))
        return {"audio_path": result["output"], "notes": notes, "log": log}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _read_result(stdout_path: str, stderr_path: str, return_code: int) -> Dict:
    """Parse the worker's final JSON line, or raise with the real reason."""
    payload = None
    try:
        with open(stdout_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped.startswith("{"):
                    continue
                try:
                    candidate = json.loads(stripped)
                except ValueError:
                    continue
                if isinstance(candidate, dict) and "ok" in candidate:
                    payload = candidate
    except OSError as exc:
        raise CleanupError(f"Could not read the cleanup result: {exc}") from exc

    if payload is None:
        raise CleanupError(
            f"The cleanup worker exited with code {return_code} without "
            f"reporting a result. Last error output: {_tail_text(stderr_path)}"
        )
    if not payload.get("ok"):
        raise CleanupError(payload.get("error", "Cleanup failed for an unstated reason."))
    if not os.path.isfile(payload.get("output", "")):
        raise CleanupError(
            "The cleanup worker reported success but wrote no file to "
            f"{payload.get('output')!r}."
        )
    return payload


def _tail_text(path: str, limit: int = 800) -> str:
    """Return the last `limit` characters of a text file, for error messages."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read().strip()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    return text[-limit:] if text else "(none)"
