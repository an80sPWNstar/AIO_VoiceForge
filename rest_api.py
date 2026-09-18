"""A small REST surface mounted on the same FastAPI app Gradio already serves.

Gradio's Blocks.launch() builds and returns the FastAPI app it is about to run
uvicorn against (`demo.app`). webui.py launches with prevent_thread_lock=True,
mounts this module's router onto that app, then blocks the main thread itself
-- so this never becomes a second process and never opens a second port.

Every route below reuses the exact code path the Gradio "Generate" button
calls (`webui_generation.gen_single`), which is what already talks to the
persistent engine worker over the file-based protocol in engine_protocol.py.
Nothing here imports IndexTTS or touches the GPU directly. See engine_worker.py
for the worker lifecycle and webui_runtime.selected_device for which card the
next model load uses.
"""
from __future__ import annotations

import os
import tempfile
import threading

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
import gradio as gr

import character_store
import engine_worker
from webui_character_handlers import CHARACTER_LIBRARY_ROOT
from webui_generation import DEFAULT_EMOTION_BIASES, gen_single
from webui_handlers import ENGINE_IDLE_CHOICES, available_devices
from webui_preview import PREVIEW_MAX_TEXT_TOKENS
from webui_runtime import (
    DEFAULT_ENGINE_LANGUAGE,
    DEVICE_AUTO,
    DEVICE_CPU,
    cmd_args,
    selected_device,
)
from webui_tone_presets import TONE_SPEED_MAX, TONE_SPEED_MIN

router = APIRouter(prefix="/api/v1")

# The engine is one persistent subprocess; two /api/v1/tts calls landing at
# once must not interleave the request files they build or race the shared
# _SUBPROCESS_STATE / selected_lora globals gen_single reads. engine_worker
# itself refuses a second submit() while busy, but this queues politely
# instead of bouncing the second caller with a 503.
_TTS_LOCK = threading.Lock()

# How long a health check's device query waits for the worker to answer
# before giving up and falling back to the host's own view of the GPU.
_DEVICE_QUERY_TIMEOUT_SECONDS = 10.0

# How long /unload waits for an in-flight REST generation before giving up
# and telling the caller the model is still resident. Short on purpose: the
# caller wants VRAM now or not at all, and blocking a ComfyUI graph for the
# length of someone else's synthesis is worse than reporting "busy".
_UNLOAD_LOCK_TIMEOUT_SECONDS = 2.0

_DEFAULT_TEXT_SEGMENT_TOKENS = str(max(20, min(PREVIEW_MAX_TEXT_TOKENS, cmd_args.gui_seg_tokens)))


def _default_gen_args(text: str, prompt_path: str, speed_factor: float) -> tuple:
    """The full positional argument list gen_single expects, mirroring the
    default value every corresponding control in the UI is built with (see
    webui.py). Only text, the reference audio, and speed vary per request --
    everything else is what a fresh session would already send.
    """
    return (
        0,  # emo_control_method: "Same as speaker voice"
        prompt_path,
        text,
        False,  # subtitle_mode
        None,  # subtitle_file
        False,  # save_used_audio
        "",  # output_filename
        None,  # image_input
        None,  # emo_ref_path
        0.65,  # emo_weight
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # vec1..vec8 (unused at method 0)
        "",  # emo_text
        False,  # emo_random
        _DEFAULT_TEXT_SEGMENT_TOKENS,
        speed_factor,
        DEFAULT_ENGINE_LANGUAGE,
        False,  # save_as_mp3
        25,  # diffusion_steps
        0.7,  # inference_cfg_rate
        200,  # interval_silence
        30,  # max_speaker_audio_length
        30,  # max_emotion_audio_length
        1,  # autoregressive_batch_size
        True,  # apply_emo_bias
        0.8,  # max_emotion_sum
        1.72,  # latent_multiplier
        0,  # max_consecutive_silence
        "256k",  # mp3_bitrate
        True,  # do_sample
        0.8,  # top_p
        30,  # top_k
        0.8,  # temperature
        0.0,  # length_penalty
        3,  # num_beams
        10.0,  # repetition_penalty
        1500,  # max_mel_tokens
        False,  # low_memory_mode
        False,  # prevent_vram_accumulation
        17,  # semantic_layer
        8192,  # cfm_cache_length
        *DEFAULT_EMOTION_BIASES,
    )


def _run_tts_sync(text: str, prompt_path: str, speed_factor: float) -> str:
    """Drive gen_single to completion and return the final wav path.

    gen_single is a generator (it streams progress to the UI); a REST caller
    just wants the last frame, which carries the finished audio path in the
    same gr.update(value=...) shape the audio component receives.
    """
    args = _default_gen_args(text, prompt_path, speed_factor)
    last_update = None
    for update in gen_single(*args):
        last_update = update
    if not last_update:
        raise RuntimeError("Generation finished without producing an output.")
    audio_update = last_update[1]
    wav_path = audio_update.get("value") if isinstance(audio_update, dict) else None
    if not wav_path or not os.path.isfile(wav_path):
        raise RuntimeError("Generation finished without a readable audio file.")
    return wav_path


def _query_gpu_name() -> tuple[str | None, str | None]:
    """(device, gpu_name), verified from the live engine process when one is
    reachable; falls back to the host's own CUDA view otherwise. Never starts
    a worker just to answer this -- see EngineWorker.query_device.
    """
    result_fd, result_path = tempfile.mkstemp(prefix="indextts_device_", suffix=".json")
    os.close(result_fd)
    try:
        if engine_worker.WORKER.query_device(result_path, timeout=_DEVICE_QUERY_TIMEOUT_SECONDS):
            import json

            try:
                with open(result_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload.get("status") == "ok":
                    return payload.get("device"), payload.get("gpu_name")
            except (OSError, ValueError):
                pass
    finally:
        try:
            os.remove(result_path)
        except OSError:
            pass

    # Worker not running, busy, or unreachable: report the card the next load
    # would use, named from the host's own CUDA view. Host and engine now
    # enumerate identically (see engine_worker._spawn), so an index means the
    # same card either way -- just not proof the engine is holding it now.
    chosen = selected_device.get()
    if chosen == DEVICE_CPU:
        return DEVICE_CPU, None
    try:
        import torch

        if torch.cuda.is_available():
            index = int(chosen.split(":", 1)[1]) if chosen.startswith("cuda:") else 0
            if index >= torch.cuda.device_count():
                index = 0
            return f"cuda:{index}", torch.cuda.get_device_name(index)
    except Exception:
        pass
    return None, None


def _resolve_device(value) -> str:
    """Validate a device string against the pickable devices, or raise ValueError.

    Shared by /device and /tts so one list decides what is acceptable and both
    reject the same values with the same message. Enumerated per call rather
    than cached: this is the UI's own list, and a card that appeared or went
    away since startup should be reflected.
    """
    if not isinstance(value, str):
        raise ValueError("device must be a string.")
    candidate = value.strip()
    valid = [device for _, device in available_devices()]
    if candidate not in valid:
        raise ValueError(f"Unknown device {value!r}. Valid values: {', '.join(valid)}.")
    return candidate


def _resolve_idle_seconds(value) -> int:
    """Validate an idle limit in seconds, or raise ValueError.

    0 is valid and means the engine never unloads itself. bools are rejected
    before the int check because True is an int in Python and would otherwise
    be accepted as a one-second idle limit.
    """
    message = "idle_seconds must be a whole, non-negative number of seconds (0 means never unload)."
    if isinstance(value, bool):
        raise ValueError(message)
    if isinstance(value, float):
        if not value.is_integer():  # also rejects nan and inf
            raise ValueError(message)
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(message)
    if value < 0:
        raise ValueError(message)
    return value


@router.get("/health")
def health():
    device, gpu_name = _query_gpu_name()
    status = engine_worker.WORKER.status()
    return {
        "status": "ok",
        "engine": "indextts-2.5",
        "model": cmd_args.model_dir,
        "device": device or ("auto" if selected_device.get() == DEVICE_AUTO else selected_device.get()),
        "gpu_name": gpu_name,
        "worker_running": bool(status["running"]),
        # Whether weights are actually resident right now, which is what
        # a caller deciding to free the card cares about. The worker
        # loads on its first request, so a running worker that has
        # served nothing is holding no VRAM.
        "model_loaded": bool(status["model_loaded"]),
        "busy": bool(status["busy"]),
    }


@router.get("/voices")
def voices():
    summaries = character_store.list_characters(CHARACTER_LIBRARY_ROOT, mode=character_store.MODE_ONESHOT)
    return {
        "voices": [
            {
                "id": summary["slug"],
                "name": summary["name"],
                "has_sample": summary["clip_count"] > 0,
            }
            for summary in summaries
            if not summary.get("unreadable")
        ]
    }


@router.post("/tts")
async def tts(payload: dict):
    text = (payload.get("text") or "").strip()
    if not text:
        return JSONResponse({"error": "text must not be empty."}, status_code=400)

    voice = payload.get("voice")
    if not voice:
        return JSONResponse({"error": "voice is required."}, status_code=400)

    prompt_path = character_store.resolve_clip_path(CHARACTER_LIBRARY_ROOT, voice)
    if not prompt_path:
        return JSONResponse({"error": f"Unknown voice {voice!r}."}, status_code=400)

    fmt = (payload.get("format") or "wav").strip().lower()
    if fmt != "wav":
        return JSONResponse({"error": "Only format 'wav' is supported."}, status_code=400)

    try:
        speed = float(payload.get("speed", 1.0))
    except (TypeError, ValueError):
        return JSONResponse({"error": "speed must be a number."}, status_code=400)
    if not TONE_SPEED_MIN <= speed <= TONE_SPEED_MAX:
        return JSONResponse(
            {"error": f"speed must be between {TONE_SPEED_MIN} and {TONE_SPEED_MAX}."},
            status_code=400,
        )

    # `seed` is accepted for forward compatibility but is not currently wired
    # to anything: no RNG seed hook exists anywhere in the synthesis path this
    # reuses (checked engine_worker.py, webui_generation_runner.py, and the
    # UI controls -- none of them read or set one), and adding one would mean
    # touching the engine-side runner, which is out of scope for this change.
    # A caller that passes a seed today gets a valid, but not reproducible,
    # generation.

    # Optional overrides for the two session-wide settings /device and /idle
    # own. Validated here, before the lock, so a bad value is a 400 rather than
    # a half-applied state. Both outlive the request, exactly as they would if
    # they had been set through their own endpoints.
    device = None
    idle_seconds = None
    try:
        if "device" in payload:
            device = _resolve_device(payload["device"])
        if "idle_seconds" in payload:
            idle_seconds = _resolve_idle_seconds(payload["idle_seconds"])
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    with _TTS_LOCK:
        # Under the lock so a concurrent caller cannot move the device between
        # here and the generation it was meant for. It only binds if this
        # request is the one that loads the model: weights already resident
        # stay on the card they loaded onto until an /unload, which this
        # deliberately does not do on the caller's behalf.
        if device is not None:
            selected_device.set(device)
        if idle_seconds is not None:
            engine_worker.WORKER.set_idle_seconds(idle_seconds)
        try:
            wav_path = _run_tts_sync(text, prompt_path, speed)
        except gr.Error as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except engine_worker.EngineWorkerError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))


@router.post("/unload")
def unload():
    """Drop the TTS model and release the engine's VRAM.

    The weights live in the persistent engine subprocess (engine_worker.py),
    not in this process, so freeing them means asking that worker to exit. It
    respawns on the next request and reloads on first use -- that reload is the
    whole cost, and the persistent worker is exactly why a second generation is
    much faster than the first. So this is opt-in, for when another process on
    the same card needs the memory more than this one needs the warm start.

    Refuses while a generation is in flight instead of killing it. The Gradio
    Generate button does not take _TTS_LOCK, so holding the lock is not by
    itself proof the engine is idle -- the worker's own busy flag is.
    """
    if not _TTS_LOCK.acquire(timeout=_UNLOAD_LOCK_TIMEOUT_SECONDS):
        return JSONResponse(
            {"error": "A TTS request is in progress; model left loaded.", "unloaded": False},
            status_code=409,
        )
    try:
        if engine_worker.WORKER.status()["busy"]:
            return JSONResponse(
                {"error": "The engine is generating; model left loaded.", "unloaded": False},
                status_code=409,
            )
        unloaded = engine_worker.WORKER.shutdown()
    except Exception as exc:
        return JSONResponse({"error": str(exc), "unloaded": False}, status_code=500)
    finally:
        _TTS_LOCK.release()

    return {
        "unloaded": bool(unloaded),
        "worker_running": bool(engine_worker.WORKER.status()["running"]),
    }


@router.get("/devices")
def devices():
    """Every device that may be selected, and the one the next load will use.

    Built from the UI's own enumeration so this list and the dropdown cannot
    drift apart.
    """
    return {
        "devices": [{"value": value, "label": label} for label, value in available_devices()],
        "current": selected_device.get(),
    }


@router.post("/device")
def set_device(payload: dict):
    """Choose the device the next model load uses.

    The weights live in the engine subprocess and the choice travels down with
    the request that loads them, so this cannot move a model that is already
    resident. When one is, `reload_required` says so and the caller decides
    whether to spend an /unload on it -- dropping the model here would take it
    out from under whoever is mid-session on the Gradio page.
    """
    try:
        device = _resolve_device(payload.get("device"))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    changed = selected_device.set(device)
    response = {
        "device": device,
        "changed": bool(changed),
        "note": "Applies to the next model load.",
    }
    if changed and engine_worker.WORKER.status()["model_loaded"]:
        response["reload_required"] = True
    return response


@router.get("/idle")
def idle():
    """The engine's current idle-unload limit, and the limits the UI offers."""
    return {
        "idle_seconds": engine_worker.WORKER.status()["idle_limit_seconds"],
        "choices": [
            {"label": label, "seconds": seconds} for label, seconds in ENGINE_IDLE_CHOICES
        ],
    }


@router.post("/idle")
def set_idle(payload: dict):
    """Set how long the engine may sit idle before it unloads itself.

    0 means never: the model then holds VRAM until /unload, the UI's unload
    button, or the app exits. A shortened limit applies to the worker already
    sitting idle, not only to the next one (see EngineWorker.set_idle_seconds).
    """
    try:
        seconds = _resolve_idle_seconds(payload.get("idle_seconds"))
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    return {"idle_seconds": engine_worker.WORKER.set_idle_seconds(seconds)}


def mount(app) -> None:
    """Attach this router to the FastAPI app Gradio is already running."""
    app.include_router(router)
