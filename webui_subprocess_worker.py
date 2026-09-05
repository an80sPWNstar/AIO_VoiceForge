from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

import engine_paths
import engine_protocol

# Import OUR runner while this repo is still first on sys.path: the V5 engine
# root is a full app carrying its own webui_generation_runner.py, so once the
# engine root is prepended, this import would resolve to Furkan's module
# instead (measured 2026-09-05: his infer call collides with our request's
# `lang` kwarg). Binding ours into sys.modules first makes it unshadowable.
from webui_generation_runner import create_tts, run_generation_request

# Prepended AFTER the runner import, and before any request runs: the runner's
# only `indextts` import is lazy (inside create_tts), and it has to land in
# the engine checkout rather than the 2.0 package still sitting in this repo.
engine_paths.prepend_engine_to_sys_path()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS WebUI subprocess worker")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Stay alive and read one generation request per line from stdin. "
             "The model is loaded once and reused, which is how the UI runs it.",
    )
    parser.add_argument("--request-file", help="Path to the generation request JSON file")
    parser.add_argument("--result-file", help="Path to the result JSON file")
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Optional path to append progress events to, one JSON object per line. "
             "A gradio Progress object cannot cross a process boundary, so the parent "
             "tails this file instead.",
    )
    args = parser.parse_args()
    if not args.serve and not (args.request_file and args.result_file):
        parser.error("--request-file and --result-file are required without --serve")
    return args


def make_progress_writer(path: str):
    """Return a callable matching IndexTTS2's gr_progress contract.

    IndexTTS2 calls it as gr_progress(value, desc=...). Each call appends one
    JSON line. A failure to write is reported but never propagated: progress is
    cosmetic and must not be able to kill a generation that is otherwise fine.
    """

    def write_progress(value, desc: str = "") -> None:
        try:
            # newline="" stops Windows translating \n into \r\n, so the byte
            # offsets the parent uses to tail this file stay exact.
            with open(path, "a", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps({
                    "value": float(value or 0.0),
                    "desc": str(desc or ""),
                }) + "\n")
                handle.flush()
        except (OSError, TypeError, ValueError) as exc:
            print(f"progress write failed ({type(exc).__name__}: {exc})", file=sys.stderr)

    return write_progress


def write_result(result_file: str, payload: dict) -> None:
    parent = os.path.dirname(result_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_one(request: dict, tts, progress_file: str | None) -> dict:
    progress_callback = make_progress_writer(progress_file) if progress_file else None
    try:
        result = run_generation_request(request, tts, progress_callback=progress_callback)
        return {"status": "ok", **result}
    except Exception as exc:
        traceback.print_exc()
        return {"status": "error", "error": str(exc)}


def release_cuda_cache() -> None:
    """Hand back the allocator's spare blocks between generations.

    The model itself stays resident; this only returns what a single generation
    grew the reservation by, which over a long session is the difference between
    steady VRAM and a slow climb into an out-of-memory.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        # A failure here means VRAM is not being released; say so.
        print(f"Engine worker: could not release CUDA cache ({exc}).", flush=True)


class ModelHolder:
    """Holds the loaded engine and rebuilds it when the runtime options change.

    Device and precision are chosen in the UI and travel with each request, so
    the worker cannot assume the model it already has is the one being asked
    for. The old model is dropped before the new one is built: loading a second
    copy alongside the first is how a 24 GB card runs out on a device switch.
    """

    def __init__(self) -> None:
        self._tts = None
        self._key: str | None = None

    def get(self, runtime: dict):
        key = json.dumps(runtime, sort_keys=True, default=str)
        if self._tts is not None and key == self._key:
            return self._tts

        if self._tts is not None:
            print("Runtime options changed; reloading the engine.", flush=True)
            self.release()

        self._tts = create_tts(runtime)
        self._key = key
        return self._tts

    def release(self) -> None:
        self._tts = None
        self._key = None
        release_cuda_cache()


def serve() -> int:
    """Read one request per stdin line until stdin closes or shutdown arrives."""
    holder = ModelHolder()

    # Announced before any model work: the parent waits on this to know the
    # process is alive, and loading here would make starting the worker cost
    # VRAM even when no one has asked for audio yet.
    print(engine_protocol.READY_SENTINEL, flush=True)

    # readline() rather than `for line in sys.stdin`: iterating a text stream
    # reads ahead into an internal buffer, so a single line written by the
    # parent can sit unread until more arrives -- which, with one request per
    # line and the parent waiting for the reply, is a hang.
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        if not line.strip():
            continue

        try:
            message = engine_protocol.decode_message(line)
        except ValueError as exc:
            print(f"worker: {exc}", file=sys.stderr, flush=True)
            continue

        if message.get("command") == engine_protocol.SHUTDOWN_COMMAND:
            break

        result_file = message.get("result_file")
        try:
            with open(message["request_file"], "r", encoding="utf-8") as handle:
                request = json.load(handle)
            tts = holder.get(request["runtime"])
            payload = run_one(request, tts, message.get("progress_file"))
        except Exception as exc:
            traceback.print_exc()
            payload = {"status": "error", "error": str(exc)}

        try:
            write_result(result_file, payload)
        except OSError as exc:
            print(f"worker: could not write result file: {exc}", file=sys.stderr, flush=True)

        release_cuda_cache()

        # Last, and only after the result file is closed: the parent treats this
        # line as permission to read that file.
        print(engine_protocol.DONE_SENTINEL, flush=True)

    holder.release()
    return 0


def run_once(args: argparse.Namespace) -> int:
    """Single-shot mode: one generation, then exit. Kept for scripted runs."""
    with open(args.request_file, "r", encoding="utf-8") as handle:
        request = json.load(handle)

    try:
        tts = create_tts(request["runtime"])
        payload = run_one(request, tts, args.progress_file)
    except Exception as exc:
        traceback.print_exc()
        payload = {"status": "error", "error": str(exc)}

    write_result(args.result_file, payload)
    return 0 if payload.get("status") == "ok" else 1


def main() -> int:
    args = parse_args()
    return serve() if args.serve else run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
