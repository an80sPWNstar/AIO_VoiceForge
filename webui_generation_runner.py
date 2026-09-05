from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import numpy as np

# This module runs in the ENGINE's interpreter (spawned by engine_worker), so
# everything it imports from the repo must stay gradio-free. The helpers below
# used to be duplicated here for that reason; the modules they live in are
# gradio-free, so the copies were deleted and the originals imported instead.
from subtitle_render import (
    build_subtitle_status_message,
    finalize_subtitle_segment_audio,
)
from subtitle_utils import (
    assemble_subtitle_audio,
    build_subtitle_render_units,
    parse_subtitle_file,
)
from task_output_utils import (
    abs_path_or_none,
    build_segment_output_path,
    write_metadata_file,
)
from webui_media_utils import FFMPEG_AVAILABLE, save_pcm16_wav
from webui_progress import (
    current_timestamp,
    format_elapsed_duration,
    print_console_progress,
)

try:
    from pydub import AudioSegment

    MP3_AVAILABLE = True
except ImportError:
    AudioSegment = None
    MP3_AVAILABLE = False


SUBTITLE_TIMING_INTERVAL_SILENCE_MS = 0


def create_tts(runtime_options: Dict[str, Any]):
    # IndexTTS-2.5 lives in its own checkout with its own interpreter, because it
    # needs numpy 2.x against this app's numpy 1.26. The worker puts that checkout
    # first on sys.path, so `indextts` here is 2.5's package, not the one vendored
    # beside this file.
    from indextts.infer_v2_5 import IndexTTS2

    # device is None for "auto", which is what IndexTTS2 already expected: it
    # then picks cuda:0 / xpu / mps / cpu itself, exactly as before this option
    # existed. A string like "cuda:1" or "cpu" pins it instead.
    return IndexTTS2(
        model_dir=runtime_options["model_dir"],
        cfg_path=runtime_options["cfg_path"],
        use_bf16=bool(runtime_options.get("use_fp16")),
        use_deepspeed=bool(runtime_options.get("use_deepspeed")),
        use_cuda_kernel=bool(runtime_options.get("use_cuda_kernel")),
        device=runtime_options.get("device"),
    )


def _apply_lora(tts, request: Dict[str, Any]) -> None:
    """Apply — or remove — the request's LoRA adapter on a V5 engine.

    Called on EVERY request, empty path included: the engine worker is
    persistent, so the previous request's adapter must never leak into this
    one. V5's set_lora short-circuits when the path has not changed, so the
    call is cheap in the steady state. Engines without set_lora (the pre-V5
    checkout) only error when a LoRA was actually asked for.
    """
    lora_path = request.get("lora_path") or ""
    strength = request.get("lora_strength")
    strength = 1.0 if strength is None else float(strength)

    if not hasattr(tts, "set_lora"):
        if lora_path:
            raise ValueError(
                "This engine cannot load a LoRA adapter. Point INDEXTTS25_ROOT "
                "at the V5 install to speak through trained voices."
            )
        return
    if lora_path and not os.path.isfile(lora_path):
        raise ValueError(
            f"The character's LoRA adapter is missing on disk: {lora_path}. "
            "Retrain the voice, or turn the trained voice off."
        )
    tts.set_lora(lora_path, strength)


def convert_wav_to_mp3(
    wav_path: str,
    mp3_path: str,
    bitrate: str = "256k",
    remove_source: bool = True,
) -> str:
    if not MP3_AVAILABLE or AudioSegment is None:
        return wav_path

    try:
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        if remove_source:
            os.remove(wav_path)
        return mp3_path
    except Exception as exc:
        print(f"Error converting to MP3: {exc}")
        return wav_path


def create_mp4_from_image_audio(image_path: str, audio_path: str, mp4_path: str) -> str:
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("FFmpeg is required to generate MP4 output from an image.")
    if not image_path or not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not audio_path or not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if os.path.dirname(mp4_path):
        os.makedirs(os.path.dirname(mp4_path), exist_ok=True)
    if os.path.isfile(mp4_path):
        os.remove(mp4_path)

    video_filter = (
        "scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "setsar=1,format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-framerate",
        "30",
        "-i",
        image_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-tune",
        "stillimage",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        "-movflags",
        "+faststart",
        mp4_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"FFmpeg MP4 generation failed: {stderr[-2000:] or 'unknown error'}")

    print(">> mp4 file saved to:", mp4_path)
    return mp4_path


def _load_metadata(metadata_path: str) -> Dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _emit_progress(progress_callback, value: float, desc: str) -> None:
    if progress_callback is None:
        return
    progress_callback(value, desc=desc)


def run_generation_request(
    request: Dict[str, Any],
    tts,
    progress_callback: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    prompt = request["prompt"]
    text = request["text"]
    subtitle_mode = bool(request["subtitle_mode"])
    subtitle_file = request.get("subtitle_file")
    save_used_audio = bool(request["save_used_audio"])
    save_as_mp3 = bool(request["save_as_mp3"])
    mp3_bitrate = request["mp3_bitrate"]
    image_path = request.get("image_path")
    infer_kwargs = dict(request["infer_kwargs"])
    low_memory_mode = bool(request["low_memory_mode"])
    task_layout = dict(request["task_layout"])
    metadata_path = request["metadata_path"]
    output_path = task_layout["final_wav_path"]
    task_folder = task_layout["task_folder"]

    processing_started_perf = time.perf_counter()
    metadata = _load_metadata(metadata_path)
    subtitle_status_message = ""
    output = None
    video_output = None

    tts.gr_progress = progress_callback
    # Low-memory mode was a fork-only attribute on the 2.0 engine. IndexTTS-2.5
    # has no equivalent, so setting it is a no-op rather than an error: assigning
    # it blindly would silently create an attribute the engine never reads.
    if hasattr(tts, "hybrid_model_device"):
        tts.hybrid_model_device = low_memory_mode

    _apply_lora(tts, request)

    try:
        subtitle_cues = parse_subtitle_file(subtitle_file) if subtitle_mode else []
        subtitle_render_units = build_subtitle_render_units(subtitle_cues) if subtitle_mode else []

        if subtitle_mode:
            if not subtitle_cues:
                raise ValueError("No caption cues were found in the selected file.")
            # Caption timing renders each cue separately through infer_texts, which
            # was added by the 2.0 fork and has no counterpart in IndexTTS-2.5.
            # Say so here rather than failing several minutes into a render.
            if not hasattr(tts, "infer_texts"):
                raise ValueError(
                    "Caption timing mode is not available on the IndexTTS-2.5 engine: "
                    "it needs the per-cue infer_texts call the previous engine provided. "
                    "Turn caption timing off to generate this text."
                )

            rendered_units = []
            original_progress = tts.gr_progress
            sampling_rate = 22050
            subtitle_console_started_at = time.perf_counter()
            assembled_audio_seconds = 0.0
            timing_executor = None
            subtitle_temp_dir = os.path.join(task_folder, "_subtitle_timing_tmp")

            try:
                tts.gr_progress = None
                non_empty_units = [unit for unit in subtitle_render_units if unit.text.strip()]
                unit_render_metadata = {}
                subtitle_infer_kwargs = dict(infer_kwargs)
                subtitle_infer_kwargs["interval_silence"] = SUBTITLE_TIMING_INTERVAL_SILENCE_MS
                subtitle_infer_kwargs["console_progress_enabled"] = True
                subtitle_infer_kwargs["console_progress_label"] = "Subtitle synthesis"
                subtitle_infer_kwargs["console_progress_item_label"] = "section"
                base_latent_multiplier = float(subtitle_infer_kwargs["latent_multiplier"])
                processing_sections = sum(
                    len(tts.tokenizer.split_segments(tts.tokenizer.tokenize(unit.text), request["max_text_tokens"]))
                    for unit in non_empty_units
                )
                unit_render_futures = {}
                print(
                    f">> Subtitle timing started | cues {len(subtitle_cues)} | "
                    f"timing units {len(subtitle_render_units)} | sections {processing_sections} | "
                    f"batch size {subtitle_infer_kwargs['section_batch_size']}"
                )
                if non_empty_units:
                    worker_count = max(1, min(4, os.cpu_count() or 1))
                    timing_executor = ThreadPoolExecutor(
                        max_workers=worker_count,
                        thread_name_prefix="subtitle_timing",
                    )
                    print(
                        f">> Subtitle timing workers started | workers {worker_count} | "
                        f"mode {'ffmpeg' if FFMPEG_AVAILABLE else 'python_fallback'}"
                    )

                    def schedule_subtitle_unit_timing(result_idx, result):
                        unit = non_empty_units[result_idx]
                        unit_sampling_rate, unit_audio = result
                        segment_path = build_segment_output_path(task_layout["segments_dir"], unit.index)
                        unit_render_futures[unit.index] = timing_executor.submit(
                            finalize_subtitle_segment_audio,
                            unit=unit,
                            unit_audio=unit_audio,
                            sampling_rate=unit_sampling_rate,
                            segment_path=segment_path,
                            temp_dir=subtitle_temp_dir,
                        )

                    tts.infer_texts(
                        spk_audio_prompt=prompt,
                        texts=[unit.text for unit in non_empty_units],
                        on_text_complete=schedule_subtitle_unit_timing,
                        **subtitle_infer_kwargs,
                    )

                for unit_idx, unit in enumerate(subtitle_render_units):
                    _emit_progress(
                        progress_callback,
                        0.05 + 0.8 * unit_idx / max(len(subtitle_render_units), 1),
                        f"subtitle unit {unit_idx + 1}/{len(subtitle_render_units)}...",
                    )
                    if unit.text.strip():
                        unit_result = unit_render_futures[unit.index].result()
                        unit_audio = unit_result["audio"]
                        segment_path = unit_result["segment_path"]
                        fit_info = unit_result["fit_info"]
                        natural_duration_ms = unit_result["natural_duration_ms"]
                    else:
                        unit_audio = np.zeros((0, 1), dtype=np.int16)
                        segment_path = build_segment_output_path(task_layout["segments_dir"], unit_idx + 1)
                        save_pcm16_wav(unit_audio, sampling_rate, segment_path)
                        fit_info = {
                            "method": "target_silence",
                            "stretch_rate": 1.0,
                            "output_duration_ms": 0,
                        }
                        natural_duration_ms = 0

                    unit_render_metadata[unit.index] = {
                        "natural_duration_ms": int(natural_duration_ms),
                        "duration_delta_before_fit_ms": int(natural_duration_ms - unit.duration_ms),
                        "fit_method": fit_info["method"],
                        "fit_stretch_rate": float(fit_info["stretch_rate"]),
                        "selected_latent_multiplier": float(base_latent_multiplier),
                        "retry_attempted": False,
                        "retry_selected": False,
                        "retry_latent_multiplier": None,
                    }

                    metadata["subtitle"]["render_units"][unit_idx]["segment_file"] = abs_path_or_none(segment_path)
                    metadata["subtitle"]["render_units"][unit_idx]["natural_duration_ms"] = (
                        unit_render_metadata.get(unit.index, {}).get("natural_duration_ms")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["generated_duration_ms"] = int(
                        round(unit_audio.shape[0] * 1000.0 / sampling_rate)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["duration_delta_before_fit_ms"] = (
                        unit_render_metadata.get(unit.index, {}).get("duration_delta_before_fit_ms")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["fit_method"] = (
                        unit_render_metadata.get(unit.index, {}).get("fit_method")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["fit_stretch_rate"] = (
                        unit_render_metadata.get(unit.index, {}).get("fit_stretch_rate")
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["selected_latent_multiplier"] = (
                        unit_render_metadata.get(unit.index, {}).get("selected_latent_multiplier")
                        or float(base_latent_multiplier)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["retry_attempted"] = (
                        unit_render_metadata.get(unit.index, {}).get("retry_attempted", False)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["retry_selected"] = (
                        unit_render_metadata.get(unit.index, {}).get("retry_selected", False)
                    )
                    metadata["subtitle"]["render_units"][unit_idx]["retry_latent_multiplier"] = (
                        unit_render_metadata.get(unit.index, {}).get("retry_latent_multiplier")
                    )
                    metadata["updated_at"] = current_timestamp()
                    write_metadata_file(metadata_path, metadata)

                    rendered_units.append((unit, unit_audio))
                    assembled_audio_seconds += unit_audio.shape[0] / float(sampling_rate) if sampling_rate else 0.0
                    print_console_progress(
                        "Subtitle timeline",
                        unit_idx + 1,
                        len(subtitle_render_units),
                        subtitle_console_started_at,
                        processed_audio_seconds=assembled_audio_seconds,
                        item_label="unit",
                    )
            finally:
                if timing_executor is not None:
                    timing_executor.shutdown(wait=True)
                if os.path.isdir(subtitle_temp_dir):
                    shutil.rmtree(subtitle_temp_dir, ignore_errors=True)
                tts.gr_progress = original_progress

            _emit_progress(progress_callback, 0.92, "assembling subtitle timeline...")
            print(">> Subtitle timeline assembly started")
            combined_audio, subtitle_issues = assemble_subtitle_audio(rendered_units, sampling_rate=sampling_rate)
            if combined_audio.shape[0] == 0:
                raise ValueError("The caption file does not contain any spoken text to synthesize.")

            print(
                f">> Subtitle timeline complete | duration {combined_audio.shape[0] / float(sampling_rate):.2f}s | "
                f"timing issues {len(subtitle_issues)}"
            )
            output = save_pcm16_wav(combined_audio, sampling_rate, output_path)
            metadata["subtitle"]["timing_issues"] = subtitle_issues
            subtitle_status_message = build_subtitle_status_message(
                subtitle_cues,
                issues=subtitle_issues,
                sample_count=combined_audio.shape[0],
                sampling_rate=sampling_rate,
                task_folder=task_folder,
                segments_dir=task_layout["segments_dir"],
                subtitle_file=subtitle_file,
            )
        else:
            output = tts.infer(
                spk_audio_prompt=prompt,
                text=text,
                output_path=output_path,
                **infer_kwargs,
            )

        if save_used_audio and prompt:
            try:
                shutil.copy2(prompt, task_layout["speaker_reference_copy_path"])
                metadata["outputs"]["speaker_reference_copy_path"] = abs_path_or_none(
                    task_layout["speaker_reference_copy_path"]
                )
                print(f"Saved used reference audio to: {task_layout['speaker_reference_copy_path']}")
            except Exception as exc:
                print(f"Error saving used audio: {exc}")

        if image_path:
            _emit_progress(progress_callback, 0.96, "rendering mp4...")
            video_output = create_mp4_from_image_audio(
                image_path,
                output,
                task_layout["final_mp4_path"],
            )

        if save_as_mp3 and MP3_AVAILABLE:
            output = convert_wav_to_mp3(
                output,
                task_layout["final_mp3_path"],
                bitrate=mp3_bitrate,
                remove_source=not bool(image_path),
            )

        processing_elapsed_seconds = time.perf_counter() - processing_started_perf
        metadata["status"] = "completed"
        metadata["updated_at"] = current_timestamp()
        metadata["error"] = None
        metadata["outputs"]["final_audio_path"] = abs_path_or_none(output)
        metadata["outputs"]["final_video_path"] = abs_path_or_none(video_output)
        metadata["outputs"]["final_wav_exists"] = bool(
            task_layout["final_wav_path"] and os.path.exists(task_layout["final_wav_path"])
        )
        metadata["outputs"]["final_mp3_exists"] = bool(
            task_layout["final_mp3_path"] and os.path.exists(task_layout["final_mp3_path"])
        )
        metadata["outputs"]["final_mp4_exists"] = bool(
            task_layout["final_mp4_path"] and os.path.exists(task_layout["final_mp4_path"])
        )
        metadata["processing"]["ended_at"] = metadata["updated_at"]
        metadata["processing"]["elapsed_ms"] = int(round(processing_elapsed_seconds * 1000.0))
        metadata["processing"]["elapsed_seconds"] = round(processing_elapsed_seconds, 3)
        metadata["processing"]["elapsed_human"] = format_elapsed_duration(processing_elapsed_seconds)
        write_metadata_file(metadata_path, metadata)

        return {
            "output_path": output,
            "video_path": video_output,
            "subtitle_status": subtitle_status_message,
        }
    except Exception as exc:
        processing_elapsed_seconds = time.perf_counter() - processing_started_perf
        metadata["status"] = "failed"
        metadata["updated_at"] = current_timestamp()
        metadata["error"] = str(exc)
        metadata["outputs"]["final_audio_path"] = abs_path_or_none(output)
        metadata["outputs"]["final_video_path"] = abs_path_or_none(video_output)
        metadata["outputs"]["final_wav_exists"] = bool(
            task_layout["final_wav_path"] and os.path.exists(task_layout["final_wav_path"])
        )
        metadata["outputs"]["final_mp3_exists"] = bool(
            task_layout["final_mp3_path"] and os.path.exists(task_layout["final_mp3_path"])
        )
        metadata["outputs"]["final_mp4_exists"] = bool(
            task_layout["final_mp4_path"] and os.path.exists(task_layout["final_mp4_path"])
        )
        metadata["processing"]["ended_at"] = metadata["updated_at"]
        metadata["processing"]["elapsed_ms"] = int(round(processing_elapsed_seconds * 1000.0))
        metadata["processing"]["elapsed_seconds"] = round(processing_elapsed_seconds, 3)
        metadata["processing"]["elapsed_human"] = format_elapsed_duration(processing_elapsed_seconds)
        write_metadata_file(metadata_path, metadata)
        raise
