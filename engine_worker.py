"""Keeps one IndexTTS-2.5 worker process alive across generations.

Loading the engine costs roughly thirty seconds and most of a GPU's working set,
and the one-shot subprocess paid that on every click: a four second clip spent
longer loading the model than synthesising with it. This holds a single worker
open and feeds it requests over its stdin, so the model loads once and every
generation after the first starts immediately.

The trade is that the model then occupies VRAM between generations, which
matters on a machine whose GPUs are also doing other work. Three things give it
back: an idle timer whose limit the UI can change and which shuts the worker
down after it expires, shutdown() behind the UI's unload button, and an atexit
hook so closing the app never leaves a process sitting on a card.

Stdlib only, and no gradio import. This runs in the UI interpreter, but keeping
it free of both means the process handling can be exercised by a plain script
without standing up a web server.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time

import engine_paths
import engine_protocol


# How long the worker may sit idle before it drops the model and exits, unless
# the UI changes it for the session. These GPUs also run other work, so the
# worker gives a card back rather than holding one indefinitely -- but half an
# hour is long enough that a normal working session never pays a reload.
# 0 disables it. The environment variable sets the startup default; the dropdown
# in the UI overrides it until the app restarts.
DEFAULT_IDLE_SECONDS = float(os.environ.get("INDEXTTS25_IDLE_SECONDS", "1800"))

# How long a graceful shutdown is given before the process tree is killed.
SHUTDOWN_GRACE_SECONDS = 10.0

# How long to wait for the worker to announce itself before deciding the spawn
# failed. Interpreter start plus imports, not model loading, which happens later.
READY_TIMEOUT_SECONDS = 120.0


def _kill_process(process: subprocess.Popen | None) -> None:
    """Kill a process and everything it spawned. Never call this holding a lock."""
    if process is None or process.poll() is not None:
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
        process.wait(timeout=SHUTDOWN_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass


class EngineWorkerError(RuntimeError):
    """Raised when the worker cannot be started or is asked to do two things."""


class EngineWorker:
    """One long-lived engine process, and the state needed to talk to it."""

    def __init__(self) -> None:
        # Reentrant because shutdown() is reached both directly and from the
        # idle timer, which may already hold it.
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._ready = threading.Event()
        self._done = threading.Event()
        self._busy = False
        self._requests_served = 0
        self._started_at: float | None = None
        self._last_activity: float | None = None
        self._idle_timer: threading.Timer | None = None
        self._idle_seconds = DEFAULT_IDLE_SECONDS
        self._exit_registered = False

    # -- lifecycle ---------------------------------------------------------

    def _is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _spawn(self) -> None:
        """Start the worker and block until it says its loop is running."""
        missing = engine_paths.missing_engine_parts()
        if missing:
            raise EngineWorkerError(
                "The IndexTTS-2.5 engine is not where it is expected. Missing:\n  "
                + "\n  ".join(missing)
            )

        cmd = [
            engine_paths.ENGINE_PYTHON,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "webui_subprocess_worker.py"),
            "--serve",
        ]
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            # The engine prints non-ASCII; without this the child's stdout uses
            # the console codepage and a print can kill it with UnicodeError.
            "PYTHONIOENCODING": "utf-8",
            engine_paths.ENGINE_ROOT_ENV: engine_paths.ENGINE_ROOT,
        }
        # The launcher .bat exports an HF_HOME for whichever engine it was
        # written against; the worker must use THIS engine's cache or the aux
        # models (w2v-bert, CAMPPlus, BigVGAN) are re-downloaded on first use.
        if os.path.isdir(engine_paths.ENGINE_HF_CACHE):
            env["HF_HOME"] = engine_paths.ENGINE_HF_CACHE
        # CUDA_VISIBLE_DEVICES is deliberately left exactly as this process
        # inherited it, so the child enumerates the same cards, in the same
        # order, as the host. VOICEFORGE_TTS_DEVICE used to be written here
        # instead, which pinned the engine by hiding every other GPU from it:
        # the host saw three devices, the child saw one and called it cuda:0,
        # and a device picker cannot work when the two sides disagree about
        # what an index names. That variable now seeds the device selection in
        # webui_runtime, which travels down with each request, so the card is
        # chosen per load rather than fixed for the life of the process.
        popen_kwargs = {
            "cwd": os.path.dirname(os.path.abspath(__file__)),
            "env": env,
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            # stderr is left attached to this console: tqdm bars and tracebacks
            # go there, and they are for a human to read, not for us to parse.
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True

        # Fresh events per process: see _reader_loop.
        ready = threading.Event()
        done = threading.Event()
        self._ready = ready
        self._done = done
        process = subprocess.Popen(cmd, **popen_kwargs)
        self._process = process
        self._started_at = time.time()
        self._requests_served = 0

        reader = threading.Thread(
            target=self._reader_loop,
            args=(process, ready, done),
            name="engine-worker-reader",
            daemon=True,
        )
        reader.start()
        self._reader = reader

        if not self._exit_registered:
            atexit.register(self._atexit_shutdown)
            self._exit_registered = True

        if not ready.wait(READY_TIMEOUT_SECONDS):
            self._process = None
            _kill_process(process)
            raise EngineWorkerError(
                "The engine worker did not start within "
                f"{int(READY_TIMEOUT_SECONDS)} seconds. Check the console for its output."
            )

    def _reader_loop(
        self,
        process: subprocess.Popen,
        ready: threading.Event,
        done: threading.Event,
    ) -> None:
        """Forward the worker's stdout, picking the two sentinels out of it.

        Everything that is not a sentinel is the engine talking, and it goes to
        this process's stdout so the console reads exactly as it did when the
        worker was spawned per generation.

        The events are passed in rather than read off self: this thread outlives
        its process by however long the final flush takes, and by then self may
        already point at a replacement worker whose state it must not touch.
        """
        try:
            for line in process.stdout:
                stripped = line.rstrip("\r\n")
                if stripped == engine_protocol.DONE_SENTINEL:
                    done.set()
                    continue
                if stripped == engine_protocol.READY_SENTINEL:
                    ready.set()
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()
        except (ValueError, OSError):
            # The pipe was closed under us, which a kill does. EOF handling below
            # is what matters; there is nothing useful to report here.
            pass
        finally:
            # EOF means the process is gone. Anything waiting on this request has
            # to be released, or the UI would poll a dead worker forever.
            ready.set()
            done.set()

    # -- requests ----------------------------------------------------------

    def submit(self, request_file: str, result_file: str, progress_file: str | None) -> None:
        """Hand one generation to the worker, starting it if it is not running."""
        with self._lock:
            # A caller that abandoned its generator (a browser tab closed
            # mid-generation) leaves busy set with the work already finished.
            # Clearing it here means one dropped request cannot wedge the worker
            # for the rest of the session.
            if self._busy and self._done.is_set():
                self.finish()

            if self._busy:
                raise EngineWorkerError("A generation is already running.")

            self._cancel_idle_timer()
            if not self._is_running():
                self._spawn()

            self._done.clear()
            self._busy = True
            try:
                self._process.stdin.write(
                    engine_protocol.encode_request(request_file, result_file, progress_file)
                )
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._busy = False
                dead = self._process
                self._process = None
                _kill_process(dead)
                raise EngineWorkerError(
                    f"The engine worker stopped accepting work ({type(exc).__name__}). "
                    "It has been shut down and will restart on the next generation."
                ) from exc
            self._last_activity = time.time()

    def finished(self) -> bool:
        """True once the worker has answered, or died trying."""
        return self._done.is_set()

    def query_device(self, result_file: str, timeout: float = 15.0) -> bool:
        """Ask an idle, running worker to report its device. Returns False
        without asking anything if the worker is busy or not running: a
        health probe must never compete with a generation for the worker's
        stdin, and must never spawn a worker just to answer a health check.
        """
        with self._lock:
            if self._busy or not self._is_running():
                return False
            self._cancel_idle_timer()
            self._done.clear()
            self._busy = True
            try:
                self._process.stdin.write(engine_protocol.encode_device_query(result_file))
                self._process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError):
                self._busy = False
                self._start_idle_timer()
                return False

        answered = self._done.wait(timeout)
        self.finish()
        return answered

    def finish(self) -> str:
        """Close out the current request. Returns 'ok' or 'died'.

        'died' means the process is gone: either it crashed, or a cancel killed
        it. Either way the caller reports it, and the next submit() respawns.
        """
        with self._lock:
            self._busy = False
            self._last_activity = time.time()
            if not self._is_running():
                outcome = "died"
            else:
                outcome = "ok"
                self._requests_served += 1
            self._start_idle_timer()
            return outcome

    # -- teardown ----------------------------------------------------------

    def _cancel_idle_timer(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def set_idle_seconds(self, seconds: float) -> float:
        """Change the idle limit for this session. 0 means never unload.

        Restarts a pending timer against the new value, so shortening the limit
        takes effect on the worker already sitting idle rather than only on the
        next one.
        """
        with self._lock:
            self._idle_seconds = max(0.0, float(seconds))
            if not self._busy:
                self._start_idle_timer()
            return self._idle_seconds

    def _start_idle_timer(self) -> None:
        self._cancel_idle_timer()
        if self._idle_seconds <= 0 or not self._is_running():
            return
        timer = threading.Timer(self._idle_seconds, self._on_idle)
        timer.daemon = True
        timer.start()
        self._idle_timer = timer

    def _on_idle(self) -> None:
        with self._lock:
            if self._busy or not self._is_running():
                return
            minutes = self._idle_seconds / 60.0
            print(
                f"Engine worker idle for {minutes:.0f} minutes; unloading to free VRAM.",
                flush=True,
            )
            self.shutdown()

    def _detach(self) -> subprocess.Popen | None:
        """Take the running process off this object and return it.

        Everything slow -- draining stdin, waiting for exit, killing a tree --
        then happens without the lock, because status() takes the same lock and
        the UI calls status() from its own thread. Claiming the process here
        also stops a concurrent submit() from adopting one that is on its way
        out: it sees no worker and starts a fresh one.

        That does mean a generate click landing during an unload can leave two
        processes alive for the seconds the old one takes to die, and briefly
        double the VRAM. Deliberate: the alternative is making submit() wait on
        a shutdown, which is the multi-second UI stall this method exists to
        avoid.
        """
        with self._lock:
            self._cancel_idle_timer()
            process = self._process if self._is_running() else None
            self._process = None
            self._busy = False
            self._requests_served = 0
            self._started_at = None
            return process

    def shutdown(self, timeout: float = SHUTDOWN_GRACE_SECONDS) -> bool:
        """Ask the worker to drop the model and exit. True if one was running.

        Falls back to killing the process tree: a worker wedged inside a CUDA
        call will not read its stdin again, and leaving it holding a card is
        worse than an ungraceful exit.
        """
        process = self._detach()
        if process is None:
            return False

        try:
            process.stdin.write(engine_protocol.encode_shutdown())
            process.stdin.flush()
            process.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass

        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process(process)
        return True

    def kill(self) -> bool:
        """Kill the worker outright. Used by cancel, where not waiting is the point."""
        process = self._detach()
        if process is None:
            return False
        _kill_process(process)
        return True

    def _atexit_shutdown(self) -> None:
        # Never let closing the app orphan a process holding a GPU.
        try:
            self.shutdown(timeout=3.0)
        except Exception:
            pass

    # -- reporting ---------------------------------------------------------

    def status(self, now: float | None = None) -> dict:
        # `now` is an injectable clock reading so the idle/unload policy --
        # the thing that decides whether to release the GPU -- is testable at
        # this boundary. The default is the real clock.
        if now is None:
            now = time.time()
        with self._lock:
            running = self._is_running()
            return {
                "running": running,
                "pid": self._process.pid if running else None,
                "busy": self._busy,
                "requests_served": self._requests_served,
                # The worker loads the model on its first request, so having
                # served one is what proves weights are resident.
                "model_loaded": running and self._requests_served > 0,
                "uptime_seconds": (now - self._started_at) if self._started_at else 0.0,
                "idle_seconds": (
                    (now - self._last_activity)
                    if (self._last_activity and running and not self._busy)
                    else 0.0
                ),
                "idle_limit_seconds": self._idle_seconds,
            }

    def describe(self) -> str:
        """One human-readable line for the UI."""
        state = self.status()
        if not state["running"]:
            return "Engine worker: not running. It starts on the next generation."

        if state["busy"]:
            return f"Engine worker: generating (pid {state['pid']})."

        if not state["model_loaded"]:
            return f"Engine worker: started, model not loaded yet (pid {state['pid']})."

        idle_note = ""
        if state["idle_limit_seconds"] > 0:
            remaining = max(0.0, state["idle_limit_seconds"] - state["idle_seconds"])
            idle_note = f" Unloads in {remaining / 60.0:.1f} min if unused."
        return (
            f"Engine worker: model loaded, holding VRAM (pid {state['pid']}, "
            f"{state['requests_served']} generated).{idle_note}"
        )


# One worker per app process.
WORKER = EngineWorker()
