"""A small REST surface mounted on the same FastAPI app Gradio already serves.

Gradio's Blocks.launch() builds and returns the FastAPI app it is about to run
uvicorn against (`demo.app`). webui.py launches with prevent_thread_lock=True,
mounts this module's router onto that app, then blocks the main thread itself
-- so this never becomes a second process and never opens a second port.

Every route below reuses the exact code path the Gradio "Generate" button
calls (`webui_generation.gen_single`), which is what already talks to the
persistent engine worker over the file-based protocol in engine_protocol.py.
Nothing here imports IndexTTS or touches the GPU directly. See engine_worker.py
for the worker lifecycle and VOICEFORGE_TTS_DEVICE for GPU pinning.
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
from webui_preview import PREVIEW_MAX_TEXT_TOKENS
from webui_runtime import DEFAULT_ENGINE_LANGUAGE, DEVICE_AUTO, cmd_args, selected_device
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

    # Worker not running, busy, or unreachable: report what the host process
    # itself sees. Same physical machine and driver, so a pinned index (see
    # VOICEFORGE_TTS_DEVICE) means the same card either way -- just not proof
    # the engine process is actually holding it right now.
    try:
        import torch

        if torch.cuda.is_available():
            pinned = os.environ.get("VOICEFORGE_TTS_DEVICE")
            index = int(pinned) if pinned and pinned.isdigit() else 0
            if index >= torch.cuda.device_count():
                index = 0
            return f"cuda:{index}", torch.cuda.get_device_name(index)
    except Exception:
        pass
    return None, None


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

    with _TTS_LOCK:
        try:
            wav_path = _run_tts_sync(text, prompt_path, speed)
        except gr.Error as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except engine_worker.EngineWorkerError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    return FileResponse(wav_path, media_type="audio/wav", filename=os.path.basename(wav_path))


def mount(app) -> None:
    """Attach this router to the FastAPI app Gradio is already running."""
    app.include_router(router)
