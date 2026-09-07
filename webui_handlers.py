"""The functions the widgets call, and the choices the widgets offer.

Everything the UI declaration hands to `fn=`, plus the lists it hands to
`choices=`. Three groups: which graphics card the engine loads on and how
long it stays resident; downloading and extracting audio from a link; and
running the clean-up chain over a clip.

The two long ones -- media_fetch_run_ui and cleanup_run_ui -- run their
real work on a worker thread and yield progress back as it arrives, so
the page updates while a download or a clean-up pass is still going. The
queue and the thread are local to each call; nothing is shared between
them at module level.

These return gradio update objects, so this module imports gradio. It does
not import webui, which is what lets webui.py import it. Split out of
webui.py.
"""

import json
import os
import platform
import queue
import subprocess
import threading
import time

import gradio as gr

import audio_cleanup_shared as cleanup_shared
import character_store as store
import engine_worker
import webui_audio_cleanup as audio_cleanup
import webui_character_handlers as characters
import webui_media_fetch as media_fetch
import html
import tempfile

import webui_tone_presets as tone_presets
import webui_voice_shaping as voice_shaping
from subtitle_utils import parse_subtitle_file, subtitle_cues_to_text
from subtitle_render import build_subtitle_status_message
from webui_media_utils import extract_audio_from_media, extract_time_ranges
from webui_preview import build_section_count_message, get_preview_rows
from webui_runtime import EMO_CHOICES_ALL
from webui_progress import render_progress_bar
from webui_runtime import (
    DEVICE_AUTO,
    DEVICE_AUTO_LABEL,
    DEVICE_CPU,
    DEVICE_CPU_LABEL,
    selected_device,
)


def update_prompt_audio():
    update_button = gr.update(interactive=True)
    return update_button




# --------------------------------------------------------------------------
# Media Fetch tab -- handlers
# --------------------------------------------------------------------------

MEDIA_FETCH_OUTPUT_ROOT = os.path.join("outputs", "media_fetch")
MEDIA_FETCH_EMPTY_INFO = "Paste a URL and press **Fetch Info** to see what is there."
MEDIA_FETCH_GAIN_MIN_DB = -20.0
MEDIA_FETCH_GAIN_MAX_DB = 20.0
MEDIA_FETCH_GAIN_STEP_DB = 0.5
MEDIA_FETCH_LOG_LINES = 12
# The worker thread has already signalled completion by the time we join, so
# this only bounds a pathological hang rather than normal waiting.
MEDIA_FETCH_THREAD_JOIN_SECONDS = 30


# --------------------------------------------------------------------------
# Emotion presets
# --------------------------------------------------------------------------



# Index of "Use emotion text description". The emo_text field is only read in
# this mode, so a tone preset has to switch here for the description to matter.
EMOTION_TEXT_MODE_INDEX = 3

# How many group components on_method_change toggles. Kept as a constant so a
# handler that wants to leave them all alone can emit the right number of
# no-op updates without hardcoding it in two places.
EMOTION_GROUP_COUNT = 5


# --------------------------------------------------------------------------
# Compute device selection
# --------------------------------------------------------------------------

def available_devices():
    """Return [(label, value)] of every device the user may pick.

    Always includes auto and CPU. Enumerates CUDA devices by index so the
    label names the actual card rather than a bare number.
    """
    choices = [(DEVICE_AUTO_LABEL, DEVICE_AUTO)]
    try:
        import torch
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                total_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
                choices.append((f"cuda:{index} - {name} ({total_gb:.0f} GB)", f"cuda:{index}"))
    except Exception as exc:  # noqa: BLE001 - reported, never silently swallowed
        print(f"Could not enumerate CUDA devices: {type(exc).__name__}: {exc}")
    choices.append((DEVICE_CPU_LABEL, DEVICE_CPU))
    return choices


# Offered idle limits, in seconds. 0 means the worker stays loaded until it is
# unloaded by hand or the app closes.
ENGINE_IDLE_CHOICES = [
    ("5 minutes", 300),
    ("10 minutes", 600),
    ("15 minutes", 900),
    ("30 minutes", 1800),
    ("1 hour", 3600),
    ("2 hours", 7200),
    ("Never unload", 0),
]


def engine_idle_choices():
    """The offered limits, plus the current one when it is not among them.

    INDEXTTS25_IDLE_SECONDS can be set to anything, and a dropdown whose value
    is not one of its choices renders blank.
    """
    current = int(engine_worker.DEFAULT_IDLE_SECONDS)
    choices = list(ENGINE_IDLE_CHOICES)
    if current not in [seconds for _, seconds in choices]:
        label = "Never unload" if current <= 0 else f"{current / 60.0:g} minutes"
        choices.append((f"{label} (from INDEXTTS25_IDLE_SECONDS)", current))
    return choices


def on_engine_idle_change(seconds):
    """Apply a new idle limit to the running worker."""
    try:
        applied = engine_worker.WORKER.set_idle_seconds(float(seconds))
    except (TypeError, ValueError):
        return gr.update(value="That is not a valid idle limit.")

    if applied <= 0:
        note = "The engine will stay loaded until you unload it or close the app."
    else:
        note = f"The engine will unload after {applied / 60.0:g} minutes idle."
    return gr.update(value=f"{engine_worker.WORKER.describe()} {note}")


def describe_engine_worker():
    return gr.update(value=engine_worker.WORKER.describe())


def unload_engine_worker():
    """Drop the loaded model so the GPU is free for something else."""
    if engine_worker.WORKER.status()["busy"]:
        return gr.update(
            value="A generation is running. Cancel it first, or wait for it to finish."
        )
    was_running = engine_worker.WORKER.shutdown()
    if not was_running:
        return gr.update(value="Engine worker: not running, nothing to unload.")
    return gr.update(
        value="Engine worker unloaded and VRAM released. "
              "The next generation starts a fresh one."
    )


def on_device_change(device_value):
    """Record the device the next generation subprocess should load onto."""
    if not selected_device.set(device_value):
        return gr.update(value=f"Already using {device_value}.", visible=True)
    if device_value == DEVICE_CPU:
        detail = "CPU selected - generation will be much slower, and fp16 is disabled there."
    else:
        detail = f"{device_value} selected."
    return gr.update(value=detail + " It applies to the next generation.", visible=True)

# Label of the tab that owns the reference voice, and the client-side hop that
# focuses it after a hand-off. Gradio's own tab selection needs an explicit
# gr.Tabs(id=...) parent, which this file does not have.
MEDIA_FETCH_GENERATION_TAB_LABEL = "Audio Generation"
MEDIA_FETCH_FOCUS_GENERATION_TAB_JS = (
    "() => {"
    f"  const target = {json.dumps(MEDIA_FETCH_GENERATION_TAB_LABEL)};"
    "  const all = Array.from(document.querySelectorAll('button'))"
    "      .filter(b => b.textContent.trim() === target);"
    "  const visible = all.filter(b => b.offsetParent !== null);"
    "  const tab = visible.length ? visible[visible.length - 1] : all[0];"
    "  if (tab) tab.click();"
    "}"
)


def media_fetch_format_choices():
    """Dropdown choices for the output format, filtered to what ffmpeg can do."""
    return [(fmt.label, fmt.key) for fmt in media_fetch.usable_formats()]


def media_fetch_default_quality():
    """Return the preset the tab starts on.

    Falls back to the Manual entry rather than raising, so a bad
    DEFAULT_QUALITY_KEY degrades to the plain module defaults instead of
    breaking the whole tab at build time.
    """
    preset = media_fetch.quality_preset_by_key(media_fetch.DEFAULT_QUALITY_KEY)
    if preset is not None:
        return preset
    return media_fetch.quality_preset_by_key(media_fetch.QUALITY_MANUAL_KEY)


def media_fetch_quality_choices():
    """Radio choices for the audio quality presets."""
    return [(preset.label, preset.key) for preset in media_fetch.QUALITY_PRESETS]


def media_fetch_quality_description(quality_key):
    """Blurb shown under the quality radio."""
    preset = media_fetch.quality_preset_by_key(quality_key)
    return preset.description if preset else ""


def on_media_fetch_quality_change(quality_key):
    """Fill the manual format/rate/channel controls from the chosen preset.

    One-way, like the cleanup presets: the manual controls stay the single
    source of truth for what actually runs, and the preset only seeds them.
    Picking Manual leaves them untouched.
    """
    preset = media_fetch.quality_preset_by_key(quality_key)
    if preset is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    description = gr.update(value=preset.description)
    if preset.key == media_fetch.QUALITY_MANUAL_KEY:
        return gr.update(), gr.update(), gr.update(), description
    return (
        gr.update(value=preset.format_key),
        gr.update(value=preset.sample_rate),
        gr.update(value=preset.channel_mode),
        description,
    )


def media_fetch_environment_note():
    """One-line readiness note shown under the tab heading."""
    ytdlp = media_fetch.ytdlp_version()
    parts = []
    parts.append(f"yt-dlp **{ytdlp}**" if ytdlp else "yt-dlp **not installed**")
    parts.append("ffmpeg **found**" if media_fetch.ffmpeg_available() else "ffmpeg **missing**")
    return " | ".join(parts)


def media_fetch_probe_ui(url, cookies_browser):
    """Fetch Info button: read metadata without downloading anything."""
    try:
        info = media_fetch.probe_media_info(url, cookies_browser)
    except media_fetch.MediaFetchError as exc:
        return gr.update(value=f"WARNING: {exc}")

    duration = info.get("duration_seconds")
    if duration:
        minutes, seconds = divmod(int(duration), 60)
        duration_text = f"{minutes}:{seconds:02d}"
    else:
        duration_text = "unknown"

    lines = [
        f"**{info['title']}**",
        f"- Uploader: {info['uploader']}",
        f"- Duration: {duration_text}",
        f"- Source: {info['extractor']}",
    ]
    return gr.update(value="\n".join(lines))


def media_fetch_run_ui(
    url,
    cookies_browser,
    download_mode,
    format_key,
    sample_rate,
    channel_mode,
    start_time,
    end_time,
    normalize,
    gain_db,
    keep_source,
    progress=gr.Progress(),
):
    """Download & Extract button: stream progress, then report the result.

    This is a generator so the status and log boxes update live. The pipeline
    itself is blocking, so it runs on a worker thread and pushes FetchProgress
    events onto a queue that this generator drains.
    """
    events = queue.Queue()
    outcome = {}
    finished = object()

    def worker():
        """Run the blocking pipeline, funnelling every event onto the queue."""
        try:
            start_seconds = media_fetch.parse_time_to_seconds(start_time)
            end_seconds = media_fetch.parse_time_to_seconds(end_time)
            outcome["result"] = media_fetch.fetch_and_extract(
                url=url,
                output_root=MEDIA_FETCH_OUTPUT_ROOT,
                format_key=format_key,
                sample_rate=int(sample_rate),
                channel_mode=channel_mode,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                normalize=bool(normalize),
                gain_db=float(gain_db),
                download_mode=download_mode,
                cookies_from_browser=cookies_browser,
                keep_source_file=bool(keep_source),
                progress_callback=events.put,
            )
        except media_fetch.MediaFetchError as exc:
            outcome["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            outcome["error"] = f"Unexpected {type(exc).__name__}: {exc}"
        finally:
            events.put(finished)

    thread = threading.Thread(target=worker, daemon=True, name="media-fetch")
    thread.start()

    lines = []
    last_fraction = 0.0
    while True:
        event = events.get()
        if event is finished:
            break
        lines.append(event.message)
        if event.fraction is not None:
            progress(event.fraction, desc=event.message)
            last_fraction = event.fraction
        yield (
            gr.update(value=render_progress_bar(last_fraction, event.message)),
            gr.update(),
            gr.update(),
            gr.update(value=event.message, visible=True),
            gr.update(value="\n".join(lines[-MEDIA_FETCH_LOG_LINES:])),
        )

    thread.join(timeout=MEDIA_FETCH_THREAD_JOIN_SECONDS)

    if "error" in outcome:
        lines.append(outcome["error"])
        yield (
            gr.update(value=render_progress_bar(
                last_fraction, outcome["error"], failed=True)),
            gr.update(value=None),
            gr.update(value=""),
            gr.update(value=f"WARNING: {outcome['error']}", visible=True),
            gr.update(value="\n".join(lines[-MEDIA_FETCH_LOG_LINES:])),
        )
        return

    result = outcome["result"]
    audio_path = result["audio_path"]
    status = f"Saved {os.path.basename(audio_path)} from \"{result['title']}\"."
    progress(1.0, desc=status)
    yield (
        gr.update(value=render_progress_bar(1.0, status, done=True)),
        gr.update(value=audio_path),
        gr.update(value=audio_path),
        gr.update(value=status, visible=True),
        gr.update(value="\n".join(result["log"][-MEDIA_FETCH_LOG_LINES:])),
    )


def media_fetch_open_folder():
    """Open the media_fetch output directory in the system file browser."""
    target = os.path.abspath(MEDIA_FETCH_OUTPUT_ROOT)
    os.makedirs(target, exist_ok=True)
    if platform.system() == "Windows":
        os.startfile(target)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", target])
    else:
        subprocess.Popen(["xdg-open", target])
    return gr.update(value=f"Opened {target}", visible=True)


# --------------------------------------------------------------------------
# Audio cleanup -- second half of the Download & Extract tab
# --------------------------------------------------------------------------

CLEANUP_PROGRESS_IDLE = render_progress_bar(0.0, "Idle")
CLEANUP_LOG_LINES = MEDIA_FETCH_LOG_LINES
CLEANUP_THREAD_JOIN_SECONDS = MEDIA_FETCH_THREAD_JOIN_SECONDS
CLEANUP_NO_RESULT = ""
# How long to wait for a cleanup cancellation to finish (process tree to die),
# in seconds. The GPU is held until all subprocesses exit; returning success too
# early leads to contention when the user starts another run.
CANCEL_CLEANUP_WAIT_SECONDS = 10

# Speaker mode radio labels. The values are the shared constants; only the
# wording lives here.
CLEANUP_SPEAKER_MODE_CHOICES = [
    ("Keep whoever talks the most", cleanup_shared.SPEAKER_MODE_DOMINANT),
    ("Keep whoever matches a voice sample", cleanup_shared.SPEAKER_MODE_SAMPLE),
]

def cleanup_device_choices():
    """Per-card choices for the cleanup tab, plus CPU.

    Asked of the SIDECAR, which is what actually runs the models: a card the
    app can see may still be newer than the sidecar torch's kernels (a
    50-series card on torch 2.6/cu124 fails with "no kernel image"), so those
    are listed but not selectable-as-usable. The cuda:N values line up across
    the process boundary because both sides enumerate in the same default
    CUDA order; the worker narrows CUDA_VISIBLE_DEVICES to the chosen index.
    """
    choices = []
    for card in audio_cleanup.usable_cleanup_devices():
        if card["ok"]:
            choices.append((f"{card['name']} (cuda:{card['index']})",
                            f"cuda:{card['index']}"))
        else:
            print(f"Cleanup cannot use {card['name']}: the cleanup engine's "
                  "torch build has no kernels for it.")
    choices.append(("CPU (slow)", cleanup_shared.DEVICE_CPU))
    return choices


def cleanup_default_device():
    """The first card, or CPU when there is none."""
    return cleanup_device_choices()[0][1]


def cleanup_device_note():
    """Names any card the cleanup engine cannot use.

    Without this, a card missing from the list reads as a bug rather than a
    fact about the engine's torch build.
    """
    unusable = sorted({card["name"]
                       for card in audio_cleanup.usable_cleanup_devices()
                       if not card["ok"]})
    if not unusable:
        return ""
    return (f"_{', '.join(unusable)}: not selectable — the cleanup engine's "
            "PyTorch build has no kernels for this card generation yet. "
            "Voice generation on the other tab is unaffected._")


def on_cleanup_device_change(device_value):
    """Refresh the banner for the card cleanup will actually use.

    Resolved from the startup probe's cache — a radio click must not cost a
    sidecar torch-import subprocess to learn a name the cache already holds.
    """
    if device_value == cleanup_shared.DEVICE_CPU:
        name = "CPU"
    else:
        index = cleanup_shared.cuda_index(device_value)
        name = next(
            (card["name"] for card in audio_cleanup.usable_cleanup_devices()
             if card["index"] == index),
            device_value,
        )
    return gr.update(value=f"Audio cleanup **ready** | compute: **{name}**")

VOICE_MODE_AUTO = "Auto (keep the loudest voice)"
VOICE_MODE_MANUAL = "Manual (pick from a list)"
VOICE_MODE_CHOICES = [VOICE_MODE_AUTO, VOICE_MODE_MANUAL]
MAX_VOICE_ROWS = cleanup_shared.MAX_VOICES_LISTED


def cleanup_stage_choices():
    """Checkbox choices for the individual cleanup stages, in pipeline order."""
    return [(cleanup_shared.STAGE_LABELS[stage], stage)
            for stage in cleanup_shared.STAGE_ORDER]


def cleanup_preset_choices():
    """Radio choices for the cleanup presets."""
    return [(preset.label, preset.key) for preset in cleanup_shared.CLEANUP_PRESETS]


def cleanup_default_stages():
    """Stages ticked when the tab first renders."""
    preset = cleanup_shared.preset_by_key(cleanup_shared.DEFAULT_PRESET_KEY)
    return list(preset.stages) if preset else []


def cleanup_preset_description(preset_key):
    """Blurb shown under the preset radio."""
    preset = cleanup_shared.preset_by_key(preset_key)
    return preset.description if preset else ""


def on_cleanup_preset_change(preset_key):
    """Fill the advanced stage checkboxes from the chosen preset.

    One-way on purpose: the checkboxes are the single source of truth for what
    runs, and the preset only seeds them. Wiring the reverse direction as well
    would make the two components update each other in a loop.
    """
    preset = cleanup_shared.preset_by_key(preset_key)
    if preset is None:
        return gr.update(), gr.update()
    description = gr.update(value=preset.description)
    if preset.key == cleanup_shared.PRESET_CUSTOM_KEY:
        return gr.update(), description
    return gr.update(value=list(preset.stages)), description


def on_cleanup_speaker_mode_change(mode):
    """Show the voice-sample box only in the mode that reads it."""
    return gr.update(visible=(mode == cleanup_shared.SPEAKER_MODE_SAMPLE))


def voice_choice_label(voice) -> str:
    """Format a voice dict as a label for display."""
    voice_id = voice.get("id", 0)
    share = voice.get("share", 0.0)
    talk_seconds = voice.get("talk_seconds", 0.0)
    return f"Voice {voice_id} — {share * 100:.0f}% of the speech ({talk_seconds:.0f}s)"


def voice_row_updates(voices) -> list:
    """Return a list of MAX_VOICE_ROWS gr.update()s for the preview audio slots."""
    updates = []
    for i in range(MAX_VOICE_ROWS):
        if i < len(voices):
            voice = voices[i]
            preview_path = voice.get("preview")
            updates.append(gr.update(value=preview_path, visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates


def voice_label_updates(voices) -> list:
    """Return a list of MAX_VOICE_ROWS gr.update()s for the per-row markdown labels."""
    updates = []
    for i in range(MAX_VOICE_ROWS):
        if i < len(voices):
            voice = voices[i]
            label = voice_choice_label(voice)
            updates.append(gr.update(value=label, visible=True))
        else:
            updates.append(gr.update(value="", visible=False))
    return updates


def voice_id_from_choice(state, choice: str):
    """Map a radio choice string back to the voice id via state["voices"]."""
    if state is None or choice is None:
        return None
    voices = state.get("voices", [])
    for voice in voices:
        if voice_choice_label(voice) == choice:
            return voice.get("id")
    return None


def cancel_cleanup_ui() -> str:
    """Stop the running cleanup job and confirm it actually died."""
    try:
        was_running = audio_cleanup.request_cancel()
        if not was_running:
            return "No cleanup was running."

        # Poll for confirmation that the process tree has exited, guarding
        # against the race where the UI reports "stopped" while the GPU is
        # still held. A user who reads success starts another run and the two
        # fight for the card.
        start_time = time.time()
        sleep_interval = 0.5  # seconds
        while time.time() - start_time < CANCEL_CLEANUP_WAIT_SECONDS:
            if audio_cleanup.cancel_is_complete():
                return "Cleanup stopped."
            time.sleep(sleep_interval)

        # Timeout elapsed without confirmation. Report honestly so the user
        # knows the GPU is still in use.
        return "Cleanup is still shutting down. The GPU may not be fully free yet."
    except Exception as exc:  # noqa: BLE001 - UI handler, must not raise
        return f"Error stopping cleanup: {exc}"


def cleanup_run_ui(
    fetched_path,
    uploaded_path,
    stages,
    vocal_model,
    dereverb_model,
    denoise_model,
    speaker_mode,
    speaker_sample,
    speaker_threshold,
    sample_rate,
    channel_mode,
    device,
    keep_intermediates,
    voice_mode=VOICE_MODE_AUTO,
    progress=gr.Progress(),
):
    """Clean Up Audio button: stream progress, then report the cleaned file.

    Same shape as media_fetch_run_ui -- a generator draining a queue that a
    worker thread fills, so the log and bar update while the subprocess runs.

    In AUTO mode, extraction runs once; in MANUAL mode, analysis extracts
    available voices and waits for selection.
    """
    source = uploaded_path or fetched_path
    events = queue.Queue()
    outcome = {}
    finished = object()

    is_manual_mode = voice_mode == VOICE_MODE_MANUAL

    def worker():
        try:
            if is_manual_mode:
                outcome["result"] = audio_cleanup.run_voice_analysis(
                    input_path=source,
                    output_root=MEDIA_FETCH_OUTPUT_ROOT,
                    stages=list(stages or []),
                    vocal_model=vocal_model,
                    dereverb_model=dereverb_model,
                    denoise_model=denoise_model,
                    device=device,
                    progress_callback=events.put,
                )
            else:
                outcome["result"] = audio_cleanup.run_cleanup(
                    input_path=source,
                    output_root=MEDIA_FETCH_OUTPUT_ROOT,
                    stages=list(stages or []),
                    vocal_model=vocal_model,
                    dereverb_model=dereverb_model,
                    denoise_model=denoise_model,
                    speaker_mode=speaker_mode,
                    speaker_sample=speaker_sample,
                    speaker_threshold=float(speaker_threshold),
                    sample_rate=int(sample_rate),
                    channel_mode=channel_mode,
                    device=device,
                    keep_intermediates=bool(keep_intermediates),
                    progress_callback=events.put,
                )
        except audio_cleanup.CleanupCancelled:
            outcome["cancelled"] = True
        except audio_cleanup.CleanupError as exc:
            outcome["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            outcome["error"] = f"Unexpected {type(exc).__name__}: {exc}"
        finally:
            events.put(finished)

    thread = threading.Thread(target=worker, daemon=True, name="audio-cleanup")
    thread.start()

    lines = []
    last_fraction = 0.0
    while True:
        event = events.get()
        if event is finished:
            break
        lines.append(event.message)
        if event.fraction is not None:
            progress(event.fraction, desc=event.message)
            last_fraction = event.fraction

        voice_state_update = gr.update()
        voice_radio_update = gr.update()
        voices_group_update = gr.update(visible=False)
        voice_audio_updates = [gr.update(value=None, visible=False) for _ in range(MAX_VOICE_ROWS)]
        voice_label_updates_list = [gr.update(value="", visible=False) for _ in range(MAX_VOICE_ROWS)]

        yield (
            gr.update(value=render_progress_bar(last_fraction, event.message)),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
            voice_state_update,
            voice_radio_update,
            voices_group_update,
            *voice_audio_updates,
            *voice_label_updates_list,
        )

    thread.join(timeout=CLEANUP_THREAD_JOIN_SECONDS)

    if "cancelled" in outcome:
        # User stopped the run; report calmly rather than as a crash.
        status = "Cleanup was cancelled."
        lines.append(status)

        voice_state_update = gr.update()
        voice_radio_update = gr.update()
        voices_group_update = gr.update(visible=False)
        voice_audio_updates = [gr.update(value=None, visible=False) for _ in range(MAX_VOICE_ROWS)]
        voice_label_updates_list = [gr.update(value="", visible=False) for _ in range(MAX_VOICE_ROWS)]

        yield (
            gr.update(value=render_progress_bar(last_fraction, status, done=True)),
            gr.update(value=None),
            gr.update(value=CLEANUP_NO_RESULT),
            gr.update(value=status),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
            voice_state_update,
            voice_radio_update,
            voices_group_update,
            *voice_audio_updates,
            *voice_label_updates_list,
        )
        return

    if "error" in outcome:
        lines.append(outcome["error"])

        voice_state_update = gr.update()
        voice_radio_update = gr.update()
        voices_group_update = gr.update(visible=False)
        voice_audio_updates = [gr.update(value=None, visible=False) for _ in range(MAX_VOICE_ROWS)]
        voice_label_updates_list = [gr.update(value="", visible=False) for _ in range(MAX_VOICE_ROWS)]

        yield (
            gr.update(value=render_progress_bar(
                last_fraction, outcome["error"], failed=True)),
            gr.update(value=None),
            gr.update(value=CLEANUP_NO_RESULT),
            gr.update(value=f"WARNING: {outcome['error']}"),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
            voice_state_update,
            voice_radio_update,
            voices_group_update,
            *voice_audio_updates,
            *voice_label_updates_list,
        )
        return

    result = outcome["result"]

    if is_manual_mode:
        voices = result.get("voices", [])
        choices = [voice_choice_label(v) for v in voices]
        voice_state_value = {
            "voices_dir": result.get("voices_dir"),
            "voices": voices,
            "source_name": os.path.basename(source),
            "choices": choices,
        }
        voice_state_update = gr.update(value=voice_state_value)
        voice_radio_update = gr.update(choices=choices, value=None)
        voices_group_update = gr.update(visible=True)
        voice_audio_updates = voice_row_updates(voices)
        voice_label_updates_list = voice_label_updates(voices)

        # The worker's notes carry reliability warnings (rough-guess split,
        # unmatched speech) — they must reach the user, not just the log.
        note_lines = list(result.get("notes") or [])
        note_lines.append(
            f"Found {len(voices)} voice(s). Play the previews, pick one, "
            "then press Extract.")
        notes = "\n".join(note_lines)
        status = f"Analyzed {os.path.basename(source)}"
        progress(1.0, desc=status)

        yield (
            gr.update(value=render_progress_bar(1.0, status, done=True)),
            gr.update(value=None),
            gr.update(value=CLEANUP_NO_RESULT),
            gr.update(value=notes),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
            voice_state_update,
            voice_radio_update,
            voices_group_update,
            *voice_audio_updates,
            *voice_label_updates_list,
        )
    else:
        audio_path = result["audio_path"]
        notes = "\n".join(result.get("notes", [])) or "Cleaned."
        status = f"Saved {os.path.basename(audio_path)}"
        progress(1.0, desc=status)

        voice_state_update = gr.update()
        voice_radio_update = gr.update()
        voices_group_update = gr.update(visible=False)
        voice_audio_updates = [gr.update(value=None, visible=False) for _ in range(MAX_VOICE_ROWS)]
        voice_label_updates_list = [gr.update(value="", visible=False) for _ in range(MAX_VOICE_ROWS)]

        yield (
            gr.update(value=render_progress_bar(1.0, status, done=True)),
            gr.update(value=audio_path),
            gr.update(value=audio_path),
            gr.update(value=notes),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
            voice_state_update,
            voice_radio_update,
            voices_group_update,
            *voice_audio_updates,
            *voice_label_updates_list,
        )


def voice_extract_run_ui(
    voices_state,
    choice,
    stages,
    speaker_threshold,
    sample_rate,
    channel_mode,
    device,
    progress=gr.Progress(),
):
    """Extract the selected voice and produce both full and reference clips.

    Same shape as cleanup_run_ui -- a generator draining a queue that a
    worker thread fills, so the log and bar update while the subprocess runs.
    """
    events = queue.Queue()
    outcome = {}
    finished = object()

    def worker():
        try:
            if voices_state is None:
                outcome["error"] = "Process a clip in manual mode first."
                return

            voice_id = voice_id_from_choice(voices_state, choice)
            if voice_id is None:
                outcome["error"] = "Pick a voice from the list first."
                return

            voices_dir = voices_state.get("voices_dir")
            source_name = voices_state.get("source_name", "audio")

            outcome["result"] = audio_cleanup.run_voice_extract(
                voices_dir=voices_dir,
                voice_id=voice_id,
                source_name=source_name,
                output_root=MEDIA_FETCH_OUTPUT_ROOT,
                stages=list(stages or []),
                speaker_threshold=float(speaker_threshold),
                sample_rate=int(sample_rate),
                channel_mode=channel_mode,
                device=device,
                progress_callback=events.put,
            )
        except audio_cleanup.CleanupCancelled:
            # Extraction runs the same worker as a cleanup pass, so Stop stops
            # this too. Caught before CleanupError -- it is a subclass, and the
            # base would otherwise swallow it and report a stop as a failure.
            outcome["cancelled"] = True
        except audio_cleanup.CleanupError as exc:
            outcome["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            outcome["error"] = f"Unexpected {type(exc).__name__}: {exc}"
        finally:
            events.put(finished)

    thread = threading.Thread(target=worker, daemon=True, name="voice-extract")
    thread.start()

    lines = []
    last_fraction = 0.0
    while True:
        event = events.get()
        if event is finished:
            break
        lines.append(event.message)
        if event.fraction is not None:
            progress(event.fraction, desc=event.message)
            last_fraction = event.fraction
        yield (
            gr.update(value=render_progress_bar(last_fraction, event.message)),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
        )

    thread.join(timeout=CLEANUP_THREAD_JOIN_SECONDS)

    if "cancelled" in outcome:
        # User stopped the run; report calmly rather than as a crash.
        status = "Extraction was cancelled."
        lines.append(status)
        yield (
            gr.update(value=render_progress_bar(
                last_fraction, status, done=True)),
            gr.update(value=None),
            gr.update(value=CLEANUP_NO_RESULT),
            gr.update(value=None),
            gr.update(value=CLEANUP_NO_RESULT),
            gr.update(value=status),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
        )
        return

    if "error" in outcome:
        lines.append(outcome["error"])
        yield (
            gr.update(value=render_progress_bar(
                last_fraction, outcome["error"], failed=True)),
            gr.update(value=None),
            gr.update(value=CLEANUP_NO_RESULT),
            gr.update(value=None),
            gr.update(value=CLEANUP_NO_RESULT),
            gr.update(value=f"WARNING: {outcome['error']}"),
            gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
        )
        return

    result = outcome["result"]
    audio_path = result.get("audio_path")
    reference_path = result.get("reference_path")
    notes = "\n".join(result.get("notes", [])) or "Extracted."
    status = f"Saved {os.path.basename(audio_path)}"
    progress(1.0, desc=status)
    yield (
        gr.update(value=render_progress_bar(1.0, status, done=True)),
        gr.update(value=audio_path),
        gr.update(value=audio_path),
        gr.update(value=reference_path),
        gr.update(value=reference_path),
        gr.update(value=notes),
        gr.update(value="\n".join(lines[-CLEANUP_LOG_LINES:])),
    )


def save_voice_reference_ui(mode, slug, reference_path, label, root=None):
    """Save Reference To Voice pressed: file a reference clip under a voice.

    Mirrors save_segment_to_voice_ui in webui_segmentation_handlers.py.
    Returns the standard four panel outputs.
    """
    if mode == store.MODE_RVC:
        return characters.refresh_panel(
            mode, slug, "An RVC voice holds a model, not clips.", root)
    if not slug:
        return characters.refresh_panel(mode, slug, "Select a voice first.", root)
    if not reference_path or not os.path.isfile(reference_path):
        return characters.refresh_panel(
            mode, slug, "Extract a voice first.", root)

    return characters.add_reference_to_character_ui(
        mode, slug, reference_path,
        label or os.path.basename(reference_path), root)


# ---------------------------------------------------------------------------
# Audio Generation tab: getting a reference voice in, and steering delivery.
# ---------------------------------------------------------------------------

def process_media_to_reference(media_path, time_ranges="", require_time_ranges=False):
    if not media_path:
        if require_time_ranges:
            return None, "Upload an audio or video file first."
        return None, ""

    try:
        temp_audio = tempfile.mktemp(suffix=".wav")
        extracted_audio = extract_audio_from_media(media_path, temp_audio)
        if not extracted_audio:
            return None, f"Failed to read audio from {os.path.basename(media_path)}."

        has_ranges = bool(time_ranges and time_ranges.strip())
        if has_ranges:
            segments_audio = extract_time_ranges(extracted_audio, time_ranges)
            if segments_audio:
                if os.path.exists(extracted_audio):
                    os.remove(extracted_audio)
                extracted_audio = segments_audio
                return (
                    extracted_audio,
                    f"Loaded extracted reference audio from {os.path.basename(media_path)} using ranges: {time_ranges.strip()}."
                )
            if os.path.exists(extracted_audio):
                os.remove(extracted_audio)
            return None, "No valid time ranges were found. Use a format like 1:3; 3:7; 11:15."

        if require_time_ranges:
            if os.path.exists(extracted_audio):
                os.remove(extracted_audio)
            return None, "Enter time ranges like 1:3; 3:7 before extracting segments."

        return extracted_audio, f"Loaded reference audio from {os.path.basename(media_path)}."
    except Exception as e:
        print(f"Error processing media: {e}")
        return None, f"Error while processing media: {str(e)}"

def process_media_upload(media_file, time_ranges):
    """Process uploaded media file and extract audio."""
    extracted_audio, status = process_media_to_reference(media_file, time_ranges, require_time_ranges=False)
    if not extracted_audio:
        if not status:
            return gr.update(), gr.update(value="", visible=False)
        return gr.update(), gr.update(value=status, visible=True)
    return gr.update(value=extracted_audio), gr.update(value=status, visible=True)

def extract_audio_segments(media_file, time_ranges):
    """Extract specific time segments from uploaded media."""
    extracted_audio, status = process_media_to_reference(media_file, time_ranges, require_time_ranges=True)
    if not extracted_audio:
        return gr.update(), gr.update(value=status, visible=True)
    return gr.update(value=extracted_audio), gr.update(value=status, visible=True)

def clear_reference_audio():
    """Clear the merged reference-media inputs."""
    return (
        gr.update(value=None),
        gr.update(value=None),
        gr.update(value=""),
        gr.update(value="", visible=False),
    )

def load_audio_from_path_ui(audio_path, time_ranges):
    """Load audio from the specified file path."""
    if not audio_path:
        return gr.update(), gr.update(value="Please enter a file path", visible=True)

    audio_path = audio_path.strip()
    if not os.path.exists(audio_path):
        return gr.update(), gr.update(value=f"File not found: {audio_path}", visible=True)

    extracted_audio, status = process_media_to_reference(audio_path, time_ranges, require_time_ranges=False)
    if extracted_audio:
        return gr.update(value=extracted_audio), gr.update(value=status, visible=True)
    return gr.update(), gr.update(value=status or "Failed to load audio file", visible=True)

def load_subtitle_file(subtitle_file_path, current_text, subtitle_mode, max_text_tokens_per_segment):
    if not subtitle_file_path:
        preview_rows = get_preview_rows(current_text, max_text_tokens_per_segment, False, None)
        section_count = build_section_count_message(current_text, max_text_tokens_per_segment, False, None)
        return (
            current_text,
            gr.update(value=False),
            gr.update(value="", visible=False),
            gr.update(value=preview_rows, visible=True, type="array"),
            gr.update(value=section_count),
        )

    try:
        cues = parse_subtitle_file(subtitle_file_path)
        subtitle_text = subtitle_cues_to_text(cues)
        use_subtitle_timing = bool(subtitle_mode)
        preview_rows = get_preview_rows(
            subtitle_text,
            max_text_tokens_per_segment,
            use_subtitle_timing,
            subtitle_file_path,
        )
        section_count = build_section_count_message(
            subtitle_text,
            max_text_tokens_per_segment,
            use_subtitle_timing,
            subtitle_file_path,
        )
        return (
            subtitle_text,
            gr.update(value=use_subtitle_timing),
            gr.update(value=build_subtitle_status_message(cues, subtitle_file=subtitle_file_path), visible=True),
            gr.update(value=preview_rows, visible=True, type="array"),
            gr.update(value=section_count),
        )
    except Exception as e:
        preview_rows = [[0, "Caption Error", str(e), ""]]
        return (
            current_text,
            gr.update(value=False),
            gr.update(value=f"Failed to load caption file: {str(e)}", visible=True),
            gr.update(value=preview_rows, visible=True, type="array"),
            gr.update(value=f"**Current Sections:** Unable to read subtitle file: {html.escape(str(e))}"),
        )

def on_segmentation_inputs_change(text, max_text_tokens_per_segment, subtitle_mode, subtitle_file_path):
    data = get_preview_rows(text, max_text_tokens_per_segment, subtitle_mode, subtitle_file_path)
    section_count = build_section_count_message(text, max_text_tokens_per_segment, subtitle_mode, subtitle_file_path)
    return (
        gr.update(value=data, visible=True, type="array"),
        gr.update(value=section_count),
    )

def on_method_change(emo_control_method):
    if emo_control_method == 1:  # emotion reference audio
        return (gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True)
                )
    elif emo_control_method == 2:  # emotion vectors
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True)
                )
    elif emo_control_method == 3:  # emotion text description
        return (gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True)
                )
    else:  # 0: same as speaker voice
        return (gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
                )

def send_fetched_to_reference(fetched_path):
    """Load the freshly extracted audio into the reference voice slot."""
    if not fetched_path:
        return (
            gr.update(),
            gr.update(value="Download and extract an audio file first.", visible=True),
        )
    extracted_audio, status = process_media_to_reference(fetched_path)
    if not extracted_audio:
        return (
            gr.update(),
            gr.update(value=status or "Could not load that file.", visible=True),
        )
    return gr.update(value=extracted_audio), gr.update(value=status, visible=True)

def send_cleaned_to_reference(cleaned_path):
    """Hand the cleaned file to the generation tab's reference voice box."""
    if not cleaned_path:
        return (
            gr.update(),
            gr.update(value="Clean up an audio file first.", visible=True),
        )
    reference_audio, status = process_media_to_reference(cleaned_path)
    if not reference_audio:
        return (
            gr.update(),
            gr.update(value=status or "Could not load the cleaned file.",
                      visible=True),
        )
    return gr.update(value=reference_audio), gr.update(value=status, visible=True)

def apply_tone_preset_ui(preset_name):
    """Fill the emotion-description box and switch to text-description mode.

    That field is only read in mode 3, so selecting a tone has to move the
    radio as well or the description is silently ignored.
    """
    description = tone_presets.tone_description(preset_name)
    # The preset seeds the Speed control; the control stays the value the
    # engine is given, so a later drag always wins over the preset.
    speed = gr.update(value=tone_presets.tone_speed(preset_name))
    if not description:
        # "None" clears the box but leaves the radio and the visible groups
        # alone, so picking it does not yank the user out of the mode they
        # were already working in.
        return (gr.update(value=""), gr.update(), speed) + tuple(
            gr.update() for _ in range(EMOTION_GROUP_COUNT)
        )
    return (
        gr.update(value=description),
        gr.update(value=EMO_CHOICES_ALL[EMOTION_TEXT_MODE_INDEX]),
        speed,
    ) + on_method_change(EMOTION_TEXT_MODE_INDEX)

def apply_voice_shaping_ui(audio_path, speed, semitones):
    """Reshape the generated clip in place in the player."""
    try:
        shaped = voice_shaping.shape_audio(
            audio_path,
            speed=speed,
            semitones=semitones,
            output_dir=os.path.join(MEDIA_FETCH_OUTPUT_ROOT, voice_shaping.SHAPED_SUBDIR),
        )
    except voice_shaping.VoiceShapingError as exc:
        return gr.update(), gr.update(value=f"WARNING: {exc}", visible=True)

    if shaped == audio_path:
        return gr.update(), gr.update(
            value=voice_shaping.describe_shaping(speed, semitones), visible=True
        )
    summary = voice_shaping.describe_shaping(speed, semitones)
    return (
        gr.update(value=shaped),
        gr.update(value=f"Applied {summary}.", visible=True),
    )

def reset_voice_shaping_ui():
    """Send both sliders back to their neutral values."""
    return (
        gr.update(value=voice_shaping.SPEED_DEFAULT),
        gr.update(value=voice_shaping.PITCH_DEFAULT_SEMITONES),
        gr.update(value="Speed and pitch reset.", visible=True),
    )
