"""Subtitle rendering shared by BOTH interpreters.

`finalize_subtitle_segment_audio` and `build_subtitle_status_message` used to
exist twice: once in webui_generation.py (the app process) and once in
webui_generation_runner.py (executed by the engine's Python, spawned at
engine_worker.py). The bodies were identical, but both write into the same
metadata.json, so a change to either copy's output format would have broken
the other process's reading of it in a way that looks like a data bug.

This module is the single copy. It must stay importable by the engine
interpreter: nothing here, and nothing it imports, may import gradio.
"""

import os

from subtitle_utils import (
    build_subtitle_render_units,
    ensure_audio_matrix,
    fit_audio_to_duration,
    format_srt_timestamp,
    get_subtitle_format_label,
    read_pcm16_wav,
    retime_audio_file_with_ffmpeg,
)
from webui_media_utils import FFMPEG_AVAILABLE, save_pcm16_wav
from webui_progress import audio_duration_ms


def finalize_subtitle_segment_audio(unit, unit_audio, sampling_rate, segment_path, temp_dir=None):
    natural_audio = ensure_audio_matrix(unit_audio)
    natural_duration_ms = audio_duration_ms(natural_audio, sampling_rate)

    fit_info = {
        "method": "copy",
        "source_duration_ms": int(natural_duration_ms),
        "target_duration_ms": int(unit.duration_ms),
        "delta_ms_before_fit": int(natural_duration_ms - unit.duration_ms),
        "stretch_rate": 1.0,
        "output_duration_ms": int(natural_duration_ms),
    }

    if natural_audio.shape[0] == 0 and unit.duration_ms <= 0:
        final_audio = natural_audio
        save_pcm16_wav(final_audio, sampling_rate, segment_path)
    elif natural_duration_ms == unit.duration_ms:
        final_audio = natural_audio
        save_pcm16_wav(final_audio, sampling_rate, segment_path)
    elif FFMPEG_AVAILABLE:
        if temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            raw_path = os.path.join(temp_dir, f"{unit.index:04d}_raw.wav")
        else:
            raw_path = f"{segment_path}.raw.wav"
        try:
            save_pcm16_wav(natural_audio, sampling_rate, raw_path)
            fit_info = retime_audio_file_with_ffmpeg(
                input_path=raw_path,
                output_path=segment_path,
                target_duration_ms=unit.duration_ms,
            )
            fitted_sampling_rate, final_audio = read_pcm16_wav(segment_path)
            if fitted_sampling_rate != sampling_rate:
                raise ValueError(
                    f"Subtitle unit {unit.index} retimed at {fitted_sampling_rate} Hz instead of {sampling_rate} Hz"
                )
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        final_audio, fit_info = fit_audio_to_duration(
            natural_audio,
            sampling_rate=sampling_rate,
            target_duration_ms=unit.duration_ms,
            return_info=True,
        )
        fit_info = dict(fit_info)
        fit_info["method"] = f"python_{fit_info['method']}"
        fit_info["output_duration_ms"] = audio_duration_ms(final_audio, sampling_rate)
        save_pcm16_wav(final_audio, sampling_rate, segment_path)

    print(
        f">> Subtitle unit saved | unit {unit.index} | "
        f"target {unit.duration_ms / 1000.0:.2f}s | natural {natural_duration_ms / 1000.0:.2f}s | "
        f"final {fit_info['output_duration_ms'] / 1000.0:.2f}s | method {fit_info['method']} | "
        f"stretch {fit_info['stretch_rate']:.4f}"
    )
    return {
        "audio": final_audio,
        "segment_path": segment_path,
        "natural_duration_ms": int(natural_duration_ms),
        "fit_info": fit_info,
    }


def build_subtitle_status_message(cues, issues=None, sample_count=None, sampling_rate=None,
                                  task_folder=None, segments_dir=None, subtitle_file=None):
    if not cues:
        return "No subtitle cues loaded."

    format_label = get_subtitle_format_label(subtitle_file)
    render_units = build_subtitle_render_units(cues)
    message_parts = [
        f"Loaded {len(cues)} {format_label} cue(s).",
        f"Timeline end: {format_srt_timestamp(cues[-1].end_ms)}.",
        f"Synthesis units: {len(render_units)}.",
        (
            "Subtitle timing uses cue start times, merges overlapping cues into larger render units when needed, "
            "avoids extra section-gap silence inside each unit, and retimes each finished unit to its target slot."
            if FFMPEG_AVAILABLE
            else "Subtitle timing uses cue start times, merges overlapping cues into larger render units when needed, "
            "avoids extra section-gap silence inside each unit, and falls back to in-process duration fitting because FFmpeg is unavailable."
        ),
    ]

    if len(render_units) < len(cues):
        message_parts.append(
            f"Detected {len(cues) - len(render_units)} overlapping cue transition(s); overlapping cues will be synthesized as merged units."
        )

    if sample_count is not None and sampling_rate:
        message_parts.append(f"Generated output length: {sample_count / float(sampling_rate):.2f}s.")

    if issues is not None:
        late_starts = [issue["delta_ms"] for issue in issues if issue["type"] == "late_start"]
        overruns = [issue["delta_ms"] for issue in issues if issue["type"] == "slot_overrun"]
        if late_starts:
            message_parts.append(
                f"{len(late_starts)} cue(s) started late because earlier speech ran long. Max late start: {max(late_starts)}ms."
            )
        if overruns:
            message_parts.append(
                f"{len(overruns)} cue(s) exceeded their subtitle duration. Max overrun: {max(overruns)}ms."
            )
        if not late_starts and not overruns:
            message_parts.append("All subtitle cue starts were preserved without timing overruns.")

    if task_folder:
        message_parts.append(f"Task folder: {os.path.abspath(task_folder)}.")
    if segments_dir:
        message_parts.append(f"Separate cue WAVs: {os.path.abspath(segments_dir)}.")

    return " ".join(message_parts)
