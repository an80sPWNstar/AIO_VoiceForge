"""Drive the V5 LoRA/DoRA training pipeline from the app.

Like rvc_engine.py, this wraps the subprocess boundary: the UI stays in this
interpreter, every training stage runs in a subprocess under the engine's own
python. Progress is polled from status.json files written atomically by the
worker. Failures raise LoraError with detail from stderr and status.json.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import engine_paths

PREP_MODULE = "indextts.training.prep_worker"
TRAIN_MODULE = "indextts.training.train_worker"
CACHE_TOOL_RELATIVE = os.path.join("tools", "cache_dataset_features.py")
STATUS_FILENAME = "status.json"
STOP_FLAG_FILENAME = "stop.flag"

POLL_INTERVAL_SECONDS = 0.5
PREP_TIMEOUT_SECONDS = 2 * 3600.0
CACHE_TIMEOUT_SECONDS = 2 * 3600.0
TRAIN_TIMEOUT_SECONDS = 12 * 3600.0


class LoraError(RuntimeError):
    """A LoRA stage failed; the message carries what it printed and logged."""


@dataclass(frozen=True)
class LoraProgress:
    """Progress event from a LoRA stage."""
    stage: str            # "prep" | "cache" | "train"
    message: str
    fraction: Optional[float] = None


def build_prep_config(clips_dir: str, name: str, output_root: str) -> dict:
    """Build a DatasetPrepConfig for the given clips.

    keys: `name`, `inputs: [clips_dir]`, `output_root`, `language: "EN"`,
    `whisper_device: "cuda:0"`. Everything else stays on V5's measured defaults.
    """
    return {
        "name": name,
        "inputs": [clips_dir],
        "output_root": output_root,
        "language": "EN",
        "whisper_device": "cuda:0",
    }


def build_train_config(dataset_dir: str, name: str, output_dir: str, *,
                       epochs: Optional[int] = None,
                       device: Optional[str] = None) -> dict:
    """Build a TrainConfig for the given dataset.

    keys: `dataset_dir`, `name`, `output_dir`,
    `model_dir: <abs ENGINE_CHECKPOINTS>`,
    `model_config: <abs ENGINE_CHECKPOINTS>/config.yaml`, plus `epochs`/`device`
    only when given (None = V5 default). Absolute model paths because the
    subprocess may not inherit assumptions about cwd.
    """
    config = {
        "dataset_dir": dataset_dir,
        "name": name,
        "output_dir": output_dir,
        "model_dir": os.path.abspath(engine_paths.ENGINE_CHECKPOINTS),
        "model_config": os.path.abspath(engine_paths.ENGINE_CFG),
    }
    if epochs is not None:
        config["epochs"] = epochs
    if device is not None:
        config["device"] = device
    return config


def build_prep_command(config_path: str, state_dir: str) -> list[str]:
    """Build argv for the prep subprocess."""
    return [
        engine_paths.ENGINE_PYTHON,
        "-m", PREP_MODULE,
        "--config", config_path,
        "--state-dir", state_dir,
    ]


def build_cache_command(dataset_dir: str) -> list[str]:
    """Build argv for the feature cache subprocess."""
    cache_tool = os.path.join(engine_paths.ENGINE_ROOT, CACHE_TOOL_RELATIVE)
    return [
        engine_paths.ENGINE_PYTHON,
        cache_tool,
        dataset_dir,
    ]


def build_train_command(config_path: str, state_dir: str) -> list[str]:
    """Build argv for the training subprocess."""
    return [
        engine_paths.ENGINE_PYTHON,
        "-m", TRAIN_MODULE,
        "--config", config_path,
        "--state-dir", state_dir,
    ]


def adapter_output_path(output_dir: str, name: str) -> str:
    """Return the best adapter output path, or the final one, or raise LoraError.

    Prefers best/<name>.safetensors if it exists, else <name>.safetensors.
    Raises LoraError naming both paths when neither exists.
    """
    best_path = os.path.join(output_dir, name, "best", f"{name}.safetensors")
    final_path = os.path.join(output_dir, name, f"{name}.safetensors")

    if os.path.isfile(best_path):
        return best_path
    if os.path.isfile(final_path):
        return final_path

    raise LoraError(
        f"Training produced no adapter weights in {output_dir}/{name}/: "
        f"expected {best_path} or {final_path}"
    )


def _read_status_json(state_dir: str) -> dict:
    """Read status.json, tolerating missing/partial file and JSON errors silently."""
    status_path = os.path.join(state_dir, STATUS_FILENAME)
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _run_stage(command: list[str], state_dir: str, stage: str, total_hint: Optional[int],
               progress_callback: Optional[Callable[[LoraProgress], None]],
               timeout: float) -> None:
    """Run a stage subprocess with file-captured stdout/stderr and progress polling.

    Reads status.json while running and emits LoraProgress with fraction
    step/total_steps when both present. Nonzero exit raises LoraError with the
    last ~20 lines of stderr AND the last `message` from status.json when
    phase == "failed".
    """
    os.makedirs(state_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as scratch:
        stdout_path = os.path.join(scratch, "stdout.txt")
        stderr_path = os.path.join(scratch, "stderr.txt")

        with open(stdout_path, "w") as stdout_file, open(stderr_path, "w") as stderr_file:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["HF_HOME"] = engine_paths.ENGINE_HF_CACHE

            process = subprocess.Popen(
                command,
                cwd=engine_paths.ENGINE_ROOT,
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
            )

        start = time.monotonic()
        deadline = start + timeout

        while process.poll() is None:
            if time.monotonic() > deadline:
                process.kill()
                raise LoraError(
                    f"Stage {stage} timed out after {timeout:.0f} seconds"
                )

            status = _read_status_json(state_dir)
            if progress_callback and status:
                step = status.get("step", 0)
                total = status.get("total_steps", total_hint)
                fraction = None
                if total and step is not None:
                    fraction = min(1.0, step / total)
                message = status.get("message", "")
                progress_callback(LoraProgress(
                    stage=stage,
                    message=message,
                    fraction=fraction,
                ))

            time.sleep(POLL_INTERVAL_SECONDS)

        if process.returncode != 0:
            # The engine prints non-ASCII; decode permissively or the error
            # report itself dies with UnicodeDecodeError.
            with open(stderr_path, "r", encoding="utf-8", errors="replace") as f:
                stderr_text = f.read().strip()
            tail = stderr_text[-2000:] if stderr_text else ""

            status = _read_status_json(state_dir)
            status_msg = ""
            if status.get("phase") == "failed":
                status_msg = f" | status: {status.get('message', '')}"

            raise LoraError(
                f"Stage {stage} failed (exit {process.returncode}): {tail}{status_msg}"
            )


def run_dataset_prep(clips_dir: str, name: str, output_root: str,
                     progress_callback: Optional[Callable[[LoraProgress], None]] = None) -> str:
    """Run dataset prep and return the dataset directory.

    Writes the config JSON to a temp dir, runs prep, verifies manifest.jsonl
    exists in the returned dataset dir, returns the dataset dir.
    """
    config = build_prep_config(clips_dir, name, output_root)

    with tempfile.TemporaryDirectory() as scratch:
        config_path = os.path.join(scratch, "prep_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # State lives WITH the dataset, not in scratch: status.json and log.txt
        # are worth keeping, and V5's own webui uses the same arrangement.
        dataset_dir = os.path.join(output_root, name)
        command = build_prep_command(config_path, dataset_dir)

        _run_stage(command, dataset_dir, "prep", None, progress_callback, PREP_TIMEOUT_SECONDS)

        manifest_path = os.path.join(dataset_dir, "manifest.jsonl")

        if not os.path.isfile(manifest_path):
            raise LoraError(
                f"Dataset prep exited cleanly but produced no manifest.jsonl "
                f"at {manifest_path}"
            )

        return dataset_dir


def run_feature_cache(dataset_dir: str,
                      progress_callback: Optional[Callable[[LoraProgress], None]] = None) -> None:
    """Run the feature cache tool and verify cache/index.jsonl exists after."""
    with tempfile.TemporaryDirectory() as scratch:
        state_dir = os.path.join(scratch, "state")
        command = build_cache_command(dataset_dir)

        _run_stage(command, state_dir, "cache", None, progress_callback, CACHE_TIMEOUT_SECONDS)

        cache_index_path = os.path.join(dataset_dir, "cache", "index.jsonl")

        if not os.path.isfile(cache_index_path):
            raise LoraError(
                f"Feature cache exited cleanly but produced no cache/index.jsonl "
                f"at {cache_index_path}"
            )


def run_training(dataset_dir: str, name: str, output_dir: str, *,
                 epochs: Optional[int] = None,
                 device: Optional[str] = None,
                 progress_callback: Optional[Callable[[LoraProgress], None]] = None) -> Dict[str, str]:
    """Run training and return {adapter_path, status_path, samples_dir}.

    Verifies manifest.jsonl and cache/index.jsonl exist in the dataset dir
    before running (training refuses to start without them).
    """
    manifest_path = os.path.join(dataset_dir, "manifest.jsonl")
    cache_index_path = os.path.join(dataset_dir, "cache", "index.jsonl")

    if not os.path.isfile(manifest_path):
        raise LoraError(
            f"Dataset missing manifest.jsonl at {manifest_path}. "
            f"Run dataset prep first."
        )

    if not os.path.isfile(cache_index_path):
        raise LoraError(
            f"Dataset missing cache/index.jsonl at {cache_index_path}. "
            f"Run feature cache first."
        )

    config = build_train_config(dataset_dir, name, output_dir, epochs=epochs, device=device)

    with tempfile.TemporaryDirectory() as scratch:
        config_path = os.path.join(scratch, "train_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        # State dir IS the adapter dir, exactly as V5's own webui runs it:
        # status.json, metrics.jsonl, log.txt and stop.flag belong beside the
        # checkpoints they describe, and request_stop() must be able to find
        # the live run after this function's scratch is gone.
        state_dir = os.path.join(output_dir, name)
        command = build_train_command(config_path, state_dir)

        _run_stage(command, state_dir, "train", None, progress_callback, TRAIN_TIMEOUT_SECONDS)

        adapter_path = adapter_output_path(output_dir, name)
        status_path = os.path.join(state_dir, STATUS_FILENAME)
        samples_dir = os.path.join(state_dir, "samples")

        return {
            "adapter_path": adapter_path,
            "status_path": status_path,
            "samples_dir": samples_dir,
        }


def request_stop(state_dir: str) -> None:
    """Write stop.flag to request graceful stop of a running stage."""
    os.makedirs(state_dir, exist_ok=True)
    stop_flag_path = os.path.join(state_dir, STOP_FLAG_FILENAME)
    Path(stop_flag_path).touch()


def train_lora_full(clips_dir: str, name: str, work_root: str,
                    progress_callback: Optional[Callable[[LoraProgress], None]] = None) -> Dict[str, str]:
    """The one-call chain: prep -> cache -> train.

    Dataset under <work_root>/dataset (pass output_root=<work_root>, then the
    dataset dir prep returns) and adapters under <work_root>/adapters.
    Maps fractions: prep to 0.00-0.25, cache to 0.25-0.35, train to 0.35-1.0.
    """
    def make_progress_callback(stage_name: str, fraction_start: float, fraction_end: float):
        def callback(progress: LoraProgress):
            if progress.fraction is not None:
                adjusted_fraction = fraction_start + (fraction_end - fraction_start) * progress.fraction
            else:
                adjusted_fraction = None
            if progress_callback:
                progress_callback(LoraProgress(
                    stage=stage_name,
                    message=progress.message,
                    fraction=adjusted_fraction,
                ))
        return callback

    dataset_output_root = os.path.join(work_root, "dataset")
    adapters_output_dir = os.path.join(work_root, "adapters")

    dataset_dir = run_dataset_prep(
        clips_dir, name, dataset_output_root,
        progress_callback=make_progress_callback("prep", 0.00, 0.25)
    )

    run_feature_cache(
        dataset_dir,
        progress_callback=make_progress_callback("cache", 0.25, 0.35)
    )

    return run_training(
        dataset_dir, name, adapters_output_dir,
        progress_callback=make_progress_callback("train", 0.35, 1.0)
    )


def engine_supports_lora() -> bool:
    """True when the engine has the LoRA training module."""
    train_worker_path = os.path.join(
        engine_paths.ENGINE_ROOT, "indextts", "training", "train_worker.py"
    )
    return os.path.isfile(train_worker_path)
