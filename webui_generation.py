"""Turning what is on screen into a finished audio file.

Three jobs, in order. _prepare_generation_request packs every control on
the page into one request dict and writes the task folder and its metadata.
_run_generation_subprocess hands that to the engine worker, tails its
progress file and streams updates back to the browser.
cancel_generation_process interrupts it.

Generation runs in a separate process under the engine's own interpreter,
so _SUBPROCESS_STATE is the only thing connecting a running job to the
cancel button. It is mutated in place under a lock rather than rebound,
because the UI thread and the generation thread both reach it.

Split out of webui.py.
"""

import datetime
import json
import os
import shutil
import tempfile
import threading
import time

import gradio as gr

import engine_worker
from subtitle_utils import (
    build_subtitle_render_units,
    get_subtitle_extension,
    get_subtitle_format_label,
    parse_subtitle_file,
)
from task_output_utils import (
    abs_path_or_none,
    create_task_output_layout,
    normalize_file_extension,
    write_metadata_file,
)
from webui_media_utils import FFMPEG_AVAILABLE, MP3_AVAILABLE
from webui_preview import resolve_max_text_tokens
from webui_progress import (
    GENERATION_PROGRESS_POLL_SECONDS,
    _drain_progress_file,
    current_timestamp,
    format_elapsed_duration,
    render_progress_bar,
)
from webui_runtime import (
    DEFAULT_ENGINE_LANGUAGE,
    EMO_CHOICES_ALL,
    _build_tts_runtime_options,
    cmd_args,
)

SUBTITLE_TIMING_INTERVAL_SILENCE_MS = 0
def resolve_optional_image_path(image_input):
    if not image_input:
        return None

    if isinstance(image_input, dict):
        image_path = image_input.get("path") or image_input.get("name")
    elif hasattr(image_input, "path"):
        image_path = image_input.path
    elif hasattr(image_input, "name"):
        image_path = image_input.name
    else:
        image_path = image_input

    image_path = str(image_path).strip() if image_path else ""
    if not image_path:
        return None
    if not os.path.isfile(image_path):
        raise gr.Error(f"Image file not found: {image_path}")
    if not FFMPEG_AVAILABLE:
        raise gr.Error("FFmpeg is required to generate MP4 output from an image.")
    return image_path


def get_image_copy_extension(image_path):
    _, extension = os.path.splitext(image_path or "")
    return normalize_file_extension(extension or ".png")


# Tuned per-channel weights applied before the emotion vector is normalized.
# The order is POSITIONAL and bound to the eight emo_bias_* sliders on the
# Advanced Parameters tab (webui.py), which read their defaults from here:
# joy, anger, sadness, fear, disgust, depression, surprise, calm.
# Reordering either side without the other silently mis-applies every bias.
DEFAULT_EMOTION_BIASES = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]


def normalize_emo_vector(emo_vector, apply_bias=True, max_emotion_sum=0.8, custom_biases=None):
    if apply_bias:
        emo_bias = custom_biases or DEFAULT_EMOTION_BIASES
        emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]

    emo_sum = sum(emo_vector)
    if emo_sum > max_emotion_sum and emo_sum > 0:
        scale_factor = max_emotion_sum / emo_sum
        emo_vector = [vec * scale_factor for vec in emo_vector]

    return emo_vector


# The engine worker owns the process and answers "is one running"; this state
# only carries what a cancel needs about the generation currently in flight.
_SUBPROCESS_STATE_LOCK = threading.Lock()
_SUBPROCESS_STATE = {
    "active": False,
    "metadata_path": None,
    "task_id": None,
    "canceled": False,
    "cancel_reason": None,
}
_IDLE_SUBPROCESS_STATE = dict(_SUBPROCESS_STATE)


def _clear_subprocess_state():
    with _SUBPROCESS_STATE_LOCK:
        snapshot = dict(_SUBPROCESS_STATE)
        _SUBPROCESS_STATE.update(_IDLE_SUBPROCESS_STATE)
        return snapshot


def _register_subprocess_state(metadata_path, task_id):
    with _SUBPROCESS_STATE_LOCK:
        _SUBPROCESS_STATE.update(
            {
                "active": True,
                "metadata_path": metadata_path,
                "task_id": task_id,
                "canceled": False,
                "cancel_reason": None,
            }
        )


def _cleanup_temp_file(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            # A file still held open is expected here; anything else would be
            # a programming fault and should raise.
            pass


def _parse_started_at_timestamp(started_at):
    if not started_at:
        return None
    try:
        return datetime.datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        # A stamp another process wrote in a format this one does not know.
        # (The old broad except here also hid an AttributeError -- this module
        # imports the datetime MODULE, and `datetime.strptime` does not exist
        # -- which silently kept elapsed time out of every canceled task.)
        return None


def _mark_metadata_canceled(metadata_path, reason, now=None):
    # `now` is an injectable clock (datetime.now-shaped) so elapsed-time
    # behaviour is testable; the default is the real one. §3.2.
    if now is None:
        now = datetime.datetime.now
    if not metadata_path or not os.path.exists(metadata_path):
        return

    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except Exception:
        return

    if metadata.get("status") == "completed":
        return

    ended_at = current_timestamp()
    metadata["status"] = "canceled"
    metadata["updated_at"] = ended_at
    metadata["error"] = reason
    metadata.setdefault("processing", {})
    metadata["processing"]["ended_at"] = ended_at
    started_at = _parse_started_at_timestamp(metadata["processing"].get("started_at"))
    if started_at is not None:
        elapsed_seconds = max(0.0, (now(started_at.tzinfo) - started_at).total_seconds())
        metadata["processing"]["elapsed_ms"] = int(round(elapsed_seconds * 1000.0))
        metadata["processing"]["elapsed_seconds"] = round(elapsed_seconds, 3)
        metadata["processing"]["elapsed_human"] = format_elapsed_duration(elapsed_seconds)
    write_metadata_file(metadata_path, metadata)


def _prepare_generation_request(
    emo_control_method,
    prompt,
    text,
    subtitle_mode,
    subtitle_file,
    save_used_audio,
    output_filename,
    image_input,
    emo_ref_path,
    emo_weight,
    vec1,
    vec2,
    vec3,
    vec4,
    vec5,
    vec6,
    vec7,
    vec8,
    emo_text,
    emo_random,
    max_text_tokens_per_segment,
    speed_factor,
    language,
    save_as_mp3,
    diffusion_steps,
    inference_cfg_rate,
    interval_silence,
    max_speaker_audio_length,
    max_emotion_audio_length,
    autoregressive_batch_size,
    apply_emo_bias,
    max_emotion_sum,
    latent_multiplier,
    max_consecutive_silence,
    mp3_bitrate,
    do_sample,
    top_p,
    top_k,
    temperature,
    length_penalty,
    num_beams,
    repetition_penalty,
    max_mel_tokens,
    low_memory_mode,
    prevent_vram_accumulation,
    semantic_layer,
    cfm_cache_length,
    emo_bias_joy,
    emo_bias_anger,
    emo_bias_sad,
    emo_bias_fear,
    emo_bias_disgust,
    emo_bias_depression,
    emo_bias_surprise,
    emo_bias_calm,
):
    subtitle_mode = bool(subtitle_mode)
    if not prompt:
        raise gr.Error("Speaker reference audio is required before you can generate speech.")

    processing_started_at = current_timestamp()
    subtitle_extension = get_subtitle_extension(subtitle_file) if subtitle_mode else None
    source_image_path = resolve_optional_image_path(image_input)
    task_layout = create_task_output_layout(
        # Absolute, because every path derived from this root is written by
        # THIS process and read back by the engine subprocess, which runs a
        # different interpreter and does not reliably inherit this working
        # directory. A relative root works only while both happen to sit in
        # the app folder, which nothing enforces.
        output_root=os.path.abspath("outputs"),
        filename=output_filename,
        subtitle_mode=subtitle_mode,
        subtitle_extension=subtitle_extension,
        image_extension=get_image_copy_extension(source_image_path) if source_image_path else None,
    )
    metadata_path = task_layout["metadata_path"]
    task_folder = task_layout["task_folder"]
    original_source_image_path = source_image_path
    if source_image_path and task_layout.get("source_image_copy_path"):
        shutil.copy2(source_image_path, task_layout["source_image_copy_path"])
        source_image_path = task_layout["source_image_copy_path"]

    if not isinstance(emo_control_method, int) and hasattr(emo_control_method, "value"):
        emo_control_method = emo_control_method.value
    emo_control_method = int(emo_control_method)

    if emo_control_method == 0:
        emo_ref_path = None
    if emo_control_method == 2:
        vec = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        custom_emo_biases = [
            emo_bias_joy,
            emo_bias_anger,
            emo_bias_sad,
            emo_bias_fear,
            emo_bias_disgust,
            emo_bias_depression,
            emo_bias_surprise,
            emo_bias_calm,
        ]
        vec = normalize_emo_vector(
            vec,
            apply_bias=apply_emo_bias,
            max_emotion_sum=max_emotion_sum,
            custom_biases=custom_emo_biases if apply_emo_bias else None,
        )
    else:
        vec = None

    if emo_text == "":
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")

    max_tokens = resolve_max_text_tokens(max_text_tokens_per_segment)
    infer_kwargs = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
        "emo_audio_prompt": emo_ref_path,
        "emo_alpha": float(emo_weight),
        "emo_vector": vec,
        "use_emo_text": (emo_control_method == 3),
        "emo_text": emo_text,
        "use_random": bool(emo_random),
        "verbose": bool(cmd_args.verbose),
        "max_text_tokens_per_segment": max_tokens,
        "interval_silence": int(interval_silence),
        # Speaking rate: above 1.0 is slower, below is faster. New in 2.5, and
        # the reason the Speed control exists.
        "duration_factor": float(speed_factor),
        "lang": str(language or DEFAULT_ENGINE_LANGUAGE),
    }
    # Everything the 2.0 fork accepted and 2.5 does not — diffusion_steps,
    # inference_cfg_rate, the reference-length caps, section_batch_size,
    # latent_multiplier, max_consecutive_silence, semantic_layer,
    # cfm_cache_length and reset_beam_cache_per_segment — is deliberately not
    # sent: passing an unknown keyword reaches the GPT sampler as a generation
    # argument and fails there instead of here. max_emotion_sum is still read,
    # but by normalize_emo_vector above rather than by the engine.

    subtitle_cues = parse_subtitle_file(subtitle_file) if subtitle_mode else []
    subtitle_render_units = build_subtitle_render_units(subtitle_cues) if subtitle_mode else []
    resolved_settings = {
        "emotion_control_method_index": emo_control_method,
        "emotion_control_method_label": EMO_CHOICES_ALL[emo_control_method]
        if 0 <= emo_control_method < len(EMO_CHOICES_ALL)
        else str(emo_control_method),
        "save_used_audio": bool(save_used_audio),
        "save_as_mp3_requested": bool(save_as_mp3),
        "save_as_mp3_enabled": bool(save_as_mp3 and MP3_AVAILABLE),
        "generate_mp4": bool(source_image_path),
        "mp3_bitrate": mp3_bitrate,
        "subtitle_mode": subtitle_mode,
        "subtitle_format": get_subtitle_format_label(subtitle_file) if subtitle_mode and subtitle_file else None,
        "low_memory_mode": bool(low_memory_mode),
        "prevent_vram_accumulation": bool(prevent_vram_accumulation),
        "use_subprocess_system": True,
        "execution_mode": "subprocess",
        "resolved_generation_kwargs": infer_kwargs,
        "subtitle_timing_overrides": (
            {
                "interval_silence": SUBTITLE_TIMING_INTERVAL_SILENCE_MS,
                "ffmpeg_timing_fit_enabled": bool(FFMPEG_AVAILABLE),
                "timing_fit_background_workers": max(1, min(4, os.cpu_count() or 1)),
                "duration_retry_enabled": False,
            }
            if subtitle_mode
            else None
        ),
        "normalized_emotion_vector": vec,
    }

    metadata = {
        "status": "in_progress",
        "created_at": processing_started_at,
        "updated_at": processing_started_at,
        "task": {
            "id": task_layout["task_id"],
            "folder": abs_path_or_none(task_folder),
            "mode": "subtitle" if subtitle_mode else "text",
            "requested_output_filename": output_filename or "",
            "resolved_output_basename": task_layout["final_basename"],
        },
        "inputs": {
            "text": text,
            "speaker_reference_audio": abs_path_or_none(prompt),
            "emotion_reference_audio": abs_path_or_none(emo_ref_path),
            "subtitle_file": abs_path_or_none(subtitle_file),
            "source_image": abs_path_or_none(original_source_image_path),
        },
        "settings": resolved_settings,
        "outputs": {
            "final_audio_path": None,
            "final_video_path": None,
            "final_wav_path": abs_path_or_none(task_layout["final_wav_path"]),
            "final_mp3_path": abs_path_or_none(task_layout["final_mp3_path"]) if save_as_mp3 and MP3_AVAILABLE else None,
            "final_mp4_path": abs_path_or_none(task_layout["final_mp4_path"]) if source_image_path else None,
            "metadata_path": abs_path_or_none(metadata_path),
            "segments_dir": abs_path_or_none(task_layout["segments_dir"]),
            "speaker_reference_copy_path": None,
            "source_image_copy_path": abs_path_or_none(source_image_path),
            "subtitle_copy_path": None,
        },
        "processing": {
            "started_at": processing_started_at,
            "ended_at": None,
            "elapsed_ms": None,
            "elapsed_seconds": None,
            "elapsed_human": None,
        },
        "subtitle": None,
        "error": None,
    }

    if subtitle_mode:
        metadata["subtitle"] = {
            "format": get_subtitle_format_label(subtitle_file) if subtitle_file else None,
            "cue_count": len(subtitle_cues),
            "render_unit_count": len(subtitle_render_units),
            "timeline_end_ms": subtitle_cues[-1].end_ms if subtitle_cues else 0,
            "cues": [
                {
                    "index": cue.index,
                    "start_ms": cue.start_ms,
                    "end_ms": cue.end_ms,
                    "duration_ms": cue.duration_ms,
                    "text": cue.text,
                    "segment_file": None,
                    "generated_duration_ms": None,
                }
                for cue in subtitle_cues
            ],
            "render_units": [
                {
                    "index": unit.index,
                    "start_ms": unit.start_ms,
                    "end_ms": unit.end_ms,
                    "duration_ms": unit.duration_ms,
                    "text": unit.text,
                    "source_cue_indices": list(unit.cue_indices),
                    "segment_file": None,
                    "natural_duration_ms": None,
                    "generated_duration_ms": None,
                    "duration_delta_before_fit_ms": None,
                    "fit_method": None,
                    "fit_stretch_rate": None,
                    "selected_latent_multiplier": float(infer_kwargs["latent_multiplier"]),
                    "retry_attempted": False,
                    "retry_selected": False,
                    "retry_latent_multiplier": None,
                }
                for unit in subtitle_render_units
            ],
            "timing_issues": [],
        }

    if subtitle_mode and subtitle_file and task_layout["subtitle_copy_path"]:
        shutil.copy2(subtitle_file, task_layout["subtitle_copy_path"])
        metadata["outputs"]["subtitle_copy_path"] = abs_path_or_none(task_layout["subtitle_copy_path"])

    write_metadata_file(metadata_path, metadata)

    return {
        "runtime": _build_tts_runtime_options(),
        "task_layout": task_layout,
        "metadata_path": metadata_path,
        "task_id": task_layout["task_id"],
        "prompt": prompt,
        "text": text,
        "subtitle_mode": subtitle_mode,
        "subtitle_file": subtitle_file,
        "save_used_audio": bool(save_used_audio),
        "save_as_mp3": bool(save_as_mp3),
        "mp3_bitrate": mp3_bitrate,
        "image_path": source_image_path,
        "infer_kwargs": infer_kwargs,
        "low_memory_mode": bool(low_memory_mode),
        "max_text_tokens": max_tokens,
    }
def _run_generation_subprocess(request):
    request_fd, request_path = tempfile.mkstemp(prefix="indextts_request_", suffix=".json")
    os.close(request_fd)
    result_fd, result_path = tempfile.mkstemp(prefix="indextts_result_", suffix=".json")
    os.close(result_fd)
    progress_fd, progress_path = tempfile.mkstemp(prefix="indextts_progress_", suffix=".jsonl")
    os.close(progress_fd)

    try:
        with open(request_path, "w", encoding="utf-8") as handle:
            json.dump(request, handle, indent=2, ensure_ascii=False)

        _register_subprocess_state(request["metadata_path"], request["task_id"])
        try:
            # The worker is long-lived and runs under the ENGINE's interpreter,
            # which needs a different numpy major than this one. It starts on
            # first use and keeps the model loaded afterwards, so only the first
            # generation of a session pays the thirty second load.
            engine_worker.WORKER.submit(request_path, result_path, progress_path)
        except engine_worker.EngineWorkerError as exc:
            _clear_subprocess_state()
            raise gr.Error(str(exc)) from exc

        # A gradio Progress object cannot cross a process boundary, so the
        # worker appends progress events to progress_path and we tail it while
        # the generation runs. Falling behind is harmless: only the newest event
        # is rendered, and the loop always drains what is there before exiting.
        consumed = 0
        last_fraction = 0.0
        while not engine_worker.WORKER.finished():
            consumed, last_fraction, event = _drain_progress_file(
                progress_path, consumed, last_fraction
            )
            if event is not None:
                yield (
                    gr.update(value=render_progress_bar(last_fraction, event)),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )
            time.sleep(GENERATION_PROGRESS_POLL_SECONDS)

        consumed, last_fraction, event = _drain_progress_file(
            progress_path, consumed, last_fraction
        )

        outcome = engine_worker.WORKER.finish()
        state_snapshot = _clear_subprocess_state() or {}
        canceled = bool(state_snapshot.get("canceled"))
        cancel_reason = state_snapshot.get("cancel_reason") or "Generation was canceled."

        if canceled:
            raise gr.Error(cancel_reason)

        # The result file is checked before the worker's health: a worker that
        # wrote a good result and then died on the way out still produced the
        # audio, and throwing that away would be the wrong answer.
        result = None
        if os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as handle:
                    result = json.load(handle)
            except (OSError, json.JSONDecodeError):
                result = None

        if result is None:
            if outcome == "died":
                raise gr.Error(
                    "The engine worker stopped during generation. Check the console "
                    "output; a fresh one starts on the next generation."
                )
            raise gr.Error("The engine worker finished without writing a readable result.")

        if result.get("status") != "ok":
            raise gr.Error(result.get("error") or "Generation failed.")

        subtitle_status_message = result.get("subtitle_status") or ""
        video_path = result.get("video_path")
        yield (
            gr.update(value=render_progress_bar(
                1.0, os.path.basename(result["output_path"]), done=True)),
            gr.update(value=result["output_path"], visible=True),
            gr.update(value=video_path, visible=bool(video_path)),
            gr.update(value=subtitle_status_message, visible=bool(subtitle_status_message)),
        )
    finally:
        _cleanup_temp_file(request_path)
        _cleanup_temp_file(result_path)
        _cleanup_temp_file(progress_path)

def cancel_generation_process(use_subprocess_system, cancel_confirmed):
    if not cancel_confirmed:
        return gr.update()

    with _SUBPROCESS_STATE_LOCK:
        active = bool(_SUBPROCESS_STATE.get("active"))
        metadata_path = _SUBPROCESS_STATE.get("metadata_path")
        task_id = _SUBPROCESS_STATE.get("task_id")
        if not active:
            _SUBPROCESS_STATE.update(_IDLE_SUBPROCESS_STATE)
            return gr.update(value="No generation is currently running.")

        _SUBPROCESS_STATE["canceled"] = True
        _SUBPROCESS_STATE["cancel_reason"] = "Generation canceled by user."

    _mark_metadata_canceled(metadata_path, "Generation canceled by user.")

    # Killing the worker is the only way to stop a generation mid-flight: the
    # engine offers no cooperative cancel. It costs the loaded model, so the
    # next generation pays a reload -- which is the right trade for a stop
    # button, but is why cancel is not a free way to reset.
    engine_worker.WORKER.kill()
    task_label = f" task {task_id}" if task_id else ""
    return gr.update(
        value=f"Canceled{task_label}. The engine reloads on the next generation."
    )


def gen_single(emo_control_method,prompt, text, subtitle_mode, subtitle_file, save_used_audio, output_filename, image_input,
               emo_ref_path, emo_weight,
               vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
               emo_text,emo_random,
               max_text_tokens_per_segment,
               speed_factor,
               language,
               save_as_mp3,
               # Expert params (in order from expert_params list)
               diffusion_steps,
               inference_cfg_rate,
               interval_silence,
               max_speaker_audio_length,
               max_emotion_audio_length,
               autoregressive_batch_size,
               apply_emo_bias,
               max_emotion_sum,
               latent_multiplier,
               max_consecutive_silence,
               mp3_bitrate,
               # Advanced params (in order from advanced_params list)
               do_sample,
               top_p,
               top_k,
               temperature,
               length_penalty,
               num_beams,
               repetition_penalty,
               max_mel_tokens,
               low_memory_mode,
               prevent_vram_accumulation,
               # Model params (semantic layer, cache, emotion biases)
               semantic_layer,
               cfm_cache_length,
               emo_bias_joy,
               emo_bias_anger,
               emo_bias_sad,
               emo_bias_fear,
               emo_bias_disgust,
               emo_bias_depression,
               emo_bias_surprise,
               emo_bias_calm,
               use_subprocess_system=True,
               progress=gr.Progress()):
    request = _prepare_generation_request(
        emo_control_method,
        prompt,
        text,
        subtitle_mode,
        subtitle_file,
        save_used_audio,
        output_filename,
        image_input,
        emo_ref_path,
        emo_weight,
        vec1,
        vec2,
        vec3,
        vec4,
        vec5,
        vec6,
        vec7,
        vec8,
        emo_text,
        emo_random,
        max_text_tokens_per_segment,
        speed_factor,
        language,
        save_as_mp3,
        diffusion_steps,
        inference_cfg_rate,
        interval_silence,
        max_speaker_audio_length,
        max_emotion_audio_length,
        autoregressive_batch_size,
        apply_emo_bias,
        max_emotion_sum,
        latent_multiplier,
        max_consecutive_silence,
        mp3_bitrate,
        do_sample,
        top_p,
        top_k,
        temperature,
        length_penalty,
        num_beams,
        repetition_penalty,
        max_mel_tokens,
        low_memory_mode,
        prevent_vram_accumulation,
        semantic_layer,
        cfm_cache_length,
        emo_bias_joy,
        emo_bias_anger,
        emo_bias_sad,
        emo_bias_fear,
        emo_bias_disgust,
        emo_bias_depression,
        emo_bias_surprise,
        emo_bias_calm,
    )

    # There is no in-process alternative: IndexTTS-2.5 needs numpy 2.x and
    # Python 3.11 while this UI runs on numpy 1.26 and Python 3.10, so the model
    # can only be built in a child under the engine's own interpreter. The
    # use_subprocess_system flag is accepted for saved presets and for the
    # cancel button, but it can no longer select anything.
    # _run_generation_subprocess is a generator: it streams progress while the
    # child runs, then yields the final result. yield from, not return.
    yield from _run_generation_subprocess(request)
