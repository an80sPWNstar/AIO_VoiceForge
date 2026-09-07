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

import itertools
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import audio_cleanup_shared as shared
import webui_media_fetch as media_fetch
import webui_media_utils as media_utils


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

# How often to emit a heartbeat when no real progress event arrives.
# The heartbeat proves the run has not exited; it does not prove progress is being made.
HEARTBEAT_SECONDS = 15

SETUP_HINT = (
    "The audio cleanup environment is not installed. Run "
    "install_audio_cleanup.bat from the app folder "
    "(a one-time ~4 GB download), then restart the app."
)


def _format_duration(seconds: int) -> str:
    """Format a duration in seconds as a human-readable string.

    Returns a string like "1m 15s" or "8s". Minutes and seconds only.
    """
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    remaining_seconds = seconds % 60
    if remaining_seconds == 0:
        return f"{minutes}m"
    return f"{minutes}m {remaining_seconds}s"


class CleanupError(RuntimeError):
    """A cleanup run failed for a reason worth showing the user verbatim."""


class CleanupCancelled(CleanupError):
    """Raised when a run was stopped by the user rather than by a fault."""


# --------------------------------------------------------------------------
# Cancellation state: token-based to prevent stale-flag issues
# --------------------------------------------------------------------------

# A plain cancel_requested boolean cannot tell "stop this run" from
# "a cancel left over from a run that already ended"; using a token
# ensures a cancelled job cannot make the NEXT job cancel itself.
_CANCEL_LOCK = threading.Lock()
_RUN_COUNTER = itertools.count(1)
_ACTIVE_TOKEN = None        # Token of the run in progress, or None.
_ACTIVE_PROCESS = None      # Its Popen once started, or None.
_CANCELLED_TOKEN = None     # Token the user asked to cancel, or None.

# How long to wait after tree-kill for the process to be gone.
PROCESS_WAIT_GRACE_SECONDS = 3.0


@dataclass(frozen=True)
class CleanupProgress:
    """One progress event from the worker. `fraction` is 0..1 or None."""

    stage: Optional[str]
    message: str
    fraction: Optional[float] = None
    heartbeat: bool = False


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


def cleanup_environment_note(device: str = shared.DEFAULT_DEVICE) -> str:
    """One-line readiness note for the UI, for the card cleanup would use."""
    python_path = sidecar_python()
    if python_path is None:
        return "Audio cleanup **not installed** - " + SETUP_HINT
    parts = ["Audio cleanup **ready**"]
    name = _probe_device(python_path, device)
    parts.append(f"compute: **{name}**")
    return " | ".join(parts)


# Probed once per process: the answer costs a sidecar torch import (seconds)
# and cannot change without swapping hardware or reinstalling the sidecar.
_USABLE_DEVICES_CACHE: Optional[List[Dict]] = None


def usable_cleanup_devices() -> List[Dict]:
    """Cards the SIDECAR torch can actually run kernels on.

    Asked of the sidecar, not the app: the two have different torch builds,
    and a card the app can enumerate may be newer than the sidecar's compiled
    kernel set (measured 2026-09-05: app lists a 50-series card, sidecar
    torch 2.6/cu124 fails on it with "no kernel image"). Returns
    [{"index", "name", "ok"}]; empty when the sidecar is missing or the probe
    fails, which callers treat as "offer CPU only".
    """
    global _USABLE_DEVICES_CACHE
    if _USABLE_DEVICES_CACHE is not None:
        return _USABLE_DEVICES_CACHE
    python_path = sidecar_python()
    if python_path is None:
        _USABLE_DEVICES_CACHE = []
        return _USABLE_DEVICES_CACHE
    probe = (
        "import json, torch\n"
        "rows = []\n"
        "if torch.cuda.is_available():\n"
        "    archs = set(torch.cuda.get_arch_list())\n"
        "    for i in range(torch.cuda.device_count()):\n"
        "        cap = torch.cuda.get_device_capability(i)\n"
        "        rows.append({'index': i,\n"
        "                     'name': torch.cuda.get_device_name(i),\n"
        "                     'ok': f'sm_{cap[0]}{cap[1]}' in archs})\n"
        "print(json.dumps(rows))\n"
    )
    try:
        result = subprocess.run(
            [python_path, "-c", probe],
            capture_output=True, text=True,
            timeout=media_fetch.FFPROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Failures are cached too: three UI-build callers each re-running a
        # 60s-timeout probe turns one broken sidecar into minutes of hang.
        print(f"Cleanup device probe failed: {type(exc).__name__}: {exc}")
        _USABLE_DEVICES_CACHE = []
        return _USABLE_DEVICES_CACHE
    if result.returncode != 0:
        print(f"Cleanup device probe exited {result.returncode}: "
              f"{(result.stderr or '').strip()[-300:]}")
        _USABLE_DEVICES_CACHE = []
        return _USABLE_DEVICES_CACHE
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            _USABLE_DEVICES_CACHE = json.loads(line)
            return _USABLE_DEVICES_CACHE
        except ValueError:
            continue
    print("Cleanup device probe returned no parseable result.")
    _USABLE_DEVICES_CACHE = []
    return _USABLE_DEVICES_CACHE


def _probe_device(python_path: str, device: str = shared.DEFAULT_DEVICE) -> str:
    """Ask the sidecar interpreter what `device` resolves to.

    Reported rather than assumed: the sidecar has its own torch build, so the
    app's CUDA availability says nothing about the cleanup environment's.
    """
    if device == shared.DEVICE_CPU:
        return "CPU"
    index = shared.cuda_index(device) or 0
    probe = (
        "import torch;"
        f"print(torch.cuda.get_device_name({index})"
        f" if torch.cuda.is_available() and torch.cuda.device_count() > {index}"
        " else 'CPU')"
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
# Cancellation API
# --------------------------------------------------------------------------

def request_cancel() -> bool:
    """Ask the running cleanup job to stop. True if there was one to stop."""
    global _ACTIVE_TOKEN, _ACTIVE_PROCESS, _CANCELLED_TOKEN

    with _CANCEL_LOCK:
        if _ACTIVE_TOKEN is None:
            return False
        _CANCELLED_TOKEN = _ACTIVE_TOKEN
        process_ref = _ACTIVE_PROCESS

    # Never hold the lock across the kill, following engine_worker's pattern.
    if process_ref is not None:
        _kill_process(process_ref)
    return True


def cancel_is_complete() -> bool:
    """True when no cleanup process is still running."""
    with _CANCEL_LOCK:
        return _ACTIVE_TOKEN is None


def _kill_process(process: subprocess.Popen) -> None:
    """Kill a process and everything it spawned.

    Mirrors engine_worker's tree-kill: taskkill /T /F on Windows,
    os.killpg elsewhere. Never call this holding the cancellation lock.
    """
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            import signal

            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        process.wait(timeout=PROCESS_WAIT_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


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


def build_analyze_command(
    python_path: str,
    input_path: str,
    output_path: str,
    voices_dir: str,
    stages: List[str],
    progress_path: str,
    vocal_model: str,
    dereverb_model: str,
    denoise_model: str,
    device: str,
) -> List[str]:
    """Build the worker argv for analyze mode. Pure: builds a list, runs nothing."""
    command = [
        python_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKER_SCRIPT),
        "--mode", "analyze",
        "--input", input_path,
        "--output", output_path,
        "--voices-dir", voices_dir,
        "--stages", ",".join(stages),
        "--progress-file", progress_path,
        "--model-dir", model_cache_dir(),
        "--device", device,
        "--vocal-model", vocal_model,
        "--dereverb-model", dereverb_model,
        "--denoise-model", denoise_model,
    ]
    return command


def build_extract_command(
    python_path: str,
    input_path: str,
    output_path: str,
    reference_output: str,
    voices_dir: str,
    voice_id: int,
    stages: List[str],
    progress_path: str,
    speaker_threshold: float,
    sample_rate: int,
    channel_mode: str,
    device: str,
) -> List[str]:
    """Build the worker argv for extract mode. Pure: builds a list, runs nothing."""
    command = [
        python_path,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), WORKER_SCRIPT),
        "--mode", "extract",
        "--input", input_path,
        "--output", output_path,
        "--reference-output", reference_output,
        "--voices-dir", voices_dir,
        "--voice-id", str(voice_id),
        "--stages", ",".join(stages),
        "--progress-file", progress_path,
        # argparse marks --model-dir required even though extract mode only
        # needs it for the encoder cache; omitting it kills the worker at
        # argv parsing, before any error a user could act on.
        "--model-dir", model_cache_dir(),
        "--speaker-threshold", str(speaker_threshold),
        "--sample-rate", str(sample_rate),
        "--channel-mode", channel_mode,
        "--device", device,
    ]
    return command


def _run_worker(
    command: List[str],
    progress_path: str,
    stdout_path: str,
    stderr_path: str,
    progress_callback: Optional[ProgressCallback],
    finished_message: str,
) -> Dict:
    """Run the worker subprocess and collect its result.

    Handles Popen, tail, deadline, drain, and _read_result. Returns the
    parsed result dict from the worker's final JSON line. Checks for
    cancellation signals and raises CleanupCancelled if stopped by the user.
    """
    global _ACTIVE_TOKEN, _ACTIVE_PROCESS, _CANCELLED_TOKEN

    def report(event: CleanupProgress) -> None:
        if progress_callback is not None:
            progress_callback(event)

    # Claim a token before Popen so that a cancellation arriving during
    # startup (model loading, etc.) is not lost.
    run_token = next(_RUN_COUNTER)
    with _CANCEL_LOCK:
        _ACTIVE_TOKEN = run_token
        _CANCELLED_TOKEN = None

    report(CleanupProgress(None, finished_message, 0.0))

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
            # Register the process and check if it was already cancelled during startup.
            should_cancel = False
            with _CANCEL_LOCK:
                _ACTIVE_PROCESS = process
                if _CANCELLED_TOKEN == run_token:
                    should_cancel = True
            if should_cancel:
                _kill_process(process)
                raise CleanupCancelled("Cleanup stopped.")

            deadline = time.monotonic() + CLEANUP_TIMEOUT_SECONDS
            last_event_stage: Optional[str] = None
            last_event_fraction: Optional[float] = None
            last_event_message = ""
            last_real_event_time = time.monotonic()
            last_heartbeat_time = time.monotonic()

            while process.poll() is None:
                # Check each polling pass for a user cancellation.
                should_cancel = False
                with _CANCEL_LOCK:
                    if _CANCELLED_TOKEN == run_token:
                        should_cancel = True
                if should_cancel:
                    _kill_process(process)
                    raise CleanupCancelled("Cleanup stopped.")

                # Drain real events from the worker and track the latest one.
                for event in tail.drain():
                    log.append(event.message)
                    report(event)
                    last_event_stage = event.stage
                    last_event_fraction = event.fraction
                    last_event_message = event.message
                    last_real_event_time = time.monotonic()
                    last_heartbeat_time = last_real_event_time

                # Emit a heartbeat if enough time has passed since the last event.
                # The heartbeat proves the run has not exited; it does not assert progress.
                # So it reports elapsed time in the stage, not a "still working" message.
                # Gate on the last HEARTBEAT, not the last real event. Gating on
                # the event would satisfy the condition on every pass once the
                # stage went quiet, and this loop polls four times a second --
                # a heartbeat meant to reassure would become a flood of updates
                # for the whole of the stage it exists to describe.
                now = time.monotonic()
                if now - last_heartbeat_time >= HEARTBEAT_SECONDS:
                    elapsed_seconds = int(now - last_real_event_time)
                    # Nothing has been reported yet while the models load, and
                    # " (15s)" with no subject reads like a bug.
                    prefix = last_event_message or "Working"
                    heartbeat_message = f"{prefix} ({_format_duration(elapsed_seconds)})"
                    heartbeat = CleanupProgress(
                        stage=last_event_stage,
                        message=heartbeat_message,
                        fraction=last_event_fraction,
                        heartbeat=True,
                    )
                    report(heartbeat)
                    last_heartbeat_time = now

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
        report(CleanupProgress(None, "Finished", 1.0))
        result["log"] = log
        return result
    except CleanupError:
        raise
    finally:
        # Clear the global state only if this run's token still matches.
        # A run that has already been superseded must not clobber a newer run's state.
        with _CANCEL_LOCK:
            if _ACTIVE_TOKEN == run_token:
                _ACTIVE_TOKEN = None
                _ACTIVE_PROCESS = None
                _CANCELLED_TOKEN = None


def cleaned_output_path(source_path: str, output_root: str) -> str:
    """Return a unique path for the cleaned version of `source_path`."""
    directory = os.path.join(output_root, shared.CLEANUP_SUBDIR)
    os.makedirs(directory, exist_ok=True)
    stem = media_fetch.sanitize_filename(
        os.path.splitext(os.path.basename(source_path))[0] + shared.CLEANUP_SUFFIX
    )
    return media_fetch.unique_path(directory, stem, shared.CLEANUP_EXTENSION)


def analysis_voices_dir(output_root: str, source_path: str) -> str:
    """Return a unique directory for storing analyzed voices.

    Creates the directory and returns its path. Same source stem will get
    a numeric suffix to ensure uniqueness.
    """
    base_dir = os.path.join(output_root, shared.CLEANUP_SUBDIR, shared.VOICES_SUBDIR)
    os.makedirs(base_dir, exist_ok=True)

    source_stem = media_fetch.sanitize_filename(
        os.path.splitext(os.path.basename(source_path))[0]
    )

    counter = 1
    while True:
        if counter == 1:
            candidate_dir = os.path.join(base_dir, source_stem)
        else:
            candidate_dir = os.path.join(base_dir, f"{source_stem}_{counter}")

        try:
            os.makedirs(candidate_dir, exist_ok=False)
            return candidate_dir
        except FileExistsError:
            counter += 1


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

    # Extraction lands in the scratch dir so the finally-rmtree reclaims it:
    # a long video's PCM wav is gigabytes and used only for this one run.
    worker_input = input_path
    if media_utils.is_video_file(input_path):
        try:
            worker_input = media_utils.ensure_audio_file(
                input_path, output_dir=scratch)
        except ValueError as exc:
            shutil.rmtree(scratch, ignore_errors=True)
            raise CleanupError(str(exc))
    progress_path = os.path.join(scratch, "progress.jsonl")
    stdout_path = os.path.join(scratch, "stdout.txt")
    stderr_path = os.path.join(scratch, "stderr.txt")

    command = build_worker_command(
        python_path=python_path,
        input_path=os.path.abspath(worker_input),
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

    try:
        result = _run_worker(
            command=command,
            progress_path=progress_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            progress_callback=progress_callback,
            finished_message="Starting the cleanup engine",
        )
        notes = result.get("notes", [])
        log = result.get("log", [])
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


def run_voice_analysis(
    input_path: str,
    output_root: str,
    stages: List[str],
    vocal_model: str = shared.DEFAULT_VOCAL_MODEL,
    dereverb_model: str = shared.DEFAULT_DEREVERB_MODEL,
    denoise_model: str = shared.DEFAULT_DENOISE_MODEL,
    device: str = shared.DEFAULT_DEVICE,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict:
    """Analyze `input_path` to extract individual voices.

    Returns {"voices_dir", "duration", "voices", "notes", "log"} where each
    voice in the list has {"id", "talk_seconds", "share", "preview"}
    (centroids are stripped and kept only in voices.json on disk).
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

    voices_dir = analysis_voices_dir(output_root, input_path)
    output_path = os.path.join(voices_dir, shared.VOICES_PROCESSED_FILENAME)
    scratch = tempfile.mkdtemp(prefix="audio_analyze_")

    # Same as run_cleanup: extraction lives and dies with the scratch dir. The
    # worker copies what it needs into voices_dir as processed.wav.
    worker_input = input_path
    if media_utils.is_video_file(input_path):
        try:
            worker_input = media_utils.ensure_audio_file(
                input_path, output_dir=scratch)
        except ValueError as exc:
            shutil.rmtree(scratch, ignore_errors=True)
            raise CleanupError(str(exc))
    progress_path = os.path.join(scratch, "progress.jsonl")
    stdout_path = os.path.join(scratch, "stdout.txt")
    stderr_path = os.path.join(scratch, "stderr.txt")

    command = build_analyze_command(
        python_path=python_path,
        input_path=os.path.abspath(worker_input),
        output_path=output_path,
        voices_dir=voices_dir,
        stages=ordered,
        progress_path=progress_path,
        vocal_model=vocal_model,
        dereverb_model=dereverb_model,
        denoise_model=denoise_model,
        device=device,
    )

    try:
        result = _run_worker(
            command=command,
            progress_path=progress_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            progress_callback=progress_callback,
            finished_message="Starting voice analysis",
        )
        log = result.get("log", [])

        voices = result.get("voices", [])
        for voice in voices:
            if "centroid" in voice:
                del voice["centroid"]

        return {
            "voices_dir": voices_dir,
            "duration": result.get("duration", 0.0),
            "voices": voices,
            "notes": result.get("notes", []),
            "log": log,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def run_voice_extract(
    voices_dir: str,
    voice_id: int,
    source_name: str,
    output_root: str,
    stages: List[str],
    speaker_threshold: float = shared.DEFAULT_SPEAKER_THRESHOLD,
    sample_rate: int = media_fetch.DEFAULT_SAMPLE_RATE,
    channel_mode: str = media_fetch.CHANNEL_MONO,
    device: str = shared.DEFAULT_DEVICE,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict:
    """Extract a selected voice from a processed clip.

    Returns {"audio_path", "reference_path", "notes", "log"}.
    Raises CleanupError if the processed cache is missing or extraction fails.
    """
    processed_path = os.path.join(voices_dir, shared.VOICES_PROCESSED_FILENAME)
    if not os.path.isfile(processed_path):
        raise CleanupError(
            "Process a clip in manual mode first — the cached voices are gone."
        )

    ordered = shared.ordered_stages(stages)

    python_path = sidecar_python()
    if python_path is None:
        raise CleanupError(SETUP_HINT)
    if not media_fetch.ffmpeg_available():
        raise CleanupError("ffmpeg is not on PATH, so the extracted file cannot be written.")

    source_stem = os.path.splitext(source_name)[0]
    directory = os.path.join(output_root, shared.CLEANUP_SUBDIR)
    os.makedirs(directory, exist_ok=True)

    output_stem = media_fetch.sanitize_filename(
        f"{source_stem}_voice{voice_id}{shared.CLEANUP_SUFFIX}"
    )
    audio_path = media_fetch.unique_path(directory, output_stem, shared.CLEANUP_EXTENSION)

    reference_stem = media_fetch.sanitize_filename(
        f"{source_stem}_voice{voice_id}_reference{shared.CLEANUP_SUFFIX}"
    )
    reference_path = media_fetch.unique_path(directory, reference_stem, shared.CLEANUP_EXTENSION)

    scratch = tempfile.mkdtemp(prefix="audio_extract_")
    progress_path = os.path.join(scratch, "progress.jsonl")
    stdout_path = os.path.join(scratch, "stdout.txt")
    stderr_path = os.path.join(scratch, "stderr.txt")

    command = build_extract_command(
        python_path=python_path,
        input_path=processed_path,
        output_path=audio_path,
        reference_output=reference_path,
        voices_dir=voices_dir,
        voice_id=voice_id,
        stages=ordered,
        progress_path=progress_path,
        speaker_threshold=speaker_threshold,
        sample_rate=sample_rate,
        channel_mode=channel_mode,
        device=device,
    )

    try:
        result = _run_worker(
            command=command,
            progress_path=progress_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            progress_callback=progress_callback,
            finished_message="Extracting voice",
        )
        log = result.get("log", [])

        if not os.path.isfile(result.get("reference", "")):
            raise CleanupError(
                "The extraction worker reported success but did not write "
                f"the reference file to {result.get('reference')!r}."
            )

        return {
            "audio_path": result.get("output", audio_path),
            "reference_path": result.get("reference", reference_path),
            "notes": result.get("notes", []),
            "log": log,
        }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
