import html
import json
import os
import re
import sys
import queue
import threading
import time
from datetime import datetime
import glob
from pathlib import Path
import platform
import subprocess
import tempfile
import shutil
from typing import Any, Dict, List, Optional

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# pandas removed - not needed, using native list format instead

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

# This process never imports the engine — it only needs to know where the engine
# lives and which interpreter runs it, so that it can spawn the worker.
import engine_paths
import engine_worker

from webui_runtime import (
    DEFAULT_ENGINE_LANGUAGE,
    EMO_CHOICES_ALL,
    ENGINE_LANGUAGES,
    DEVICE_AUTO,
    DEVICE_AUTO_LABEL,
    DEVICE_CPU,
    DEVICE_CPU_LABEL,
    DeviceSelection,
    _build_tts_runtime_options,
    cmd_args,
    selected_device,
)


if not os.path.exists(cmd_args.model_dir):
    print(f"Model directory {cmd_args.model_dir} does not exist. Please download the model first.")
    sys.exit(1)

# bpe.model is deliberately absent from this list: IndexTTS-2.5 tokenizes from a
# tiktoken vocabulary and never ships one, even though its config still names it.
for file in [
    "gpt.pth",
    "config.yaml",
    "s2mel.pth",
    "wav2vec2bert_stats.pt"
]:
    file_path = os.path.join(cmd_args.model_dir, file)
    if not os.path.exists(file_path):
        print(f"Required file {file_path} does not exist. Please download it.")
        sys.exit(1)

# Generation happens in a subprocess under the engine's own interpreter, so a
# missing engine has to be caught here rather than at the first generate click.
_missing_engine = engine_paths.missing_engine_parts()
if _missing_engine:
    print("IndexTTS-2.5 engine is not installed. Missing:")
    for path in _missing_engine:
        print(f"  {path}")
    sys.exit(1)

import gradio as gr
from omegaconf import OmegaConf
from indextts.utils.front import TextNormalizer, TextTokenizer
from subtitle_utils import (
    SUPPORTED_SUBTITLE_EXTENSIONS,
    assemble_subtitle_audio,
    build_subtitle_render_units,
    ensure_audio_matrix,
    fit_audio_to_duration,
    format_srt_timestamp,
    get_subtitle_extension,
    get_subtitle_format_label,
    parse_subtitle_file,
    read_pcm16_wav,
    retime_audio_file_with_ffmpeg,
    subtitle_cues_to_text,
    write_pcm16_wav,
)
from task_output_utils import (
    build_segment_output_path,
    create_task_output_layout,
    normalize_file_extension,
    write_metadata_file,
)
import webui_media_fetch as media_fetch
import webui_audio_cleanup as audio_cleanup
import audio_cleanup_shared as cleanup_shared
import webui_voice_shaping as voice_shaping
import webui_tone_presets as tone_presets
from webui_assets import (
    APP_ASSETS_DIR,
    APP_CSS,
    APP_FAVICON_PATH,
    APP_HEAD,
    APP_TITLE,
    CAPTION_TIMING_HELP,
    MEDIA_FILE_TYPES,
)
from webui_media_utils import (
    FFMPEG_AVAILABLE,
    MP3_AVAILABLE,
    convert_wav_to_mp3,
    extract_audio_from_media,
    extract_time_ranges,
    generate_output_path,
    get_next_file_number,
    load_audio_from_path,
    open_outputs_folder,
    save_pcm16_wav,
)
from webui_progress import (
    GENERATION_PROGRESS_POLL_SECONDS,
    _drain_progress_file,
    audio_duration_ms,
    current_timestamp,
    format_elapsed_duration,
    print_console_progress,
    render_progress_bar,
)
from webui_preset_store import (
    DEFAULT_UI_PRESET_NAME,
    PRESETS_DIR,
    UI_PRESET_FORMAT,
    UI_PRESET_VERSION,
    _delete_ui_preset,
    _get_last_used_ui_preset,
    _list_ui_presets,
    _load_ui_preset,
    _save_ui_preset,
    _set_last_used_ui_preset,
)
from webui_preview import (
    MODEL_VERSION,
    PREVIEW_MAX_TEXT_TOKENS,
    build_section_count_message,
    get_preview_rows,
    get_text_processing_sections,
    resolve_max_text_tokens,
)
from webui_generation import (
    build_subtitle_status_message,
    cancel_generation_process,
    gen_single,
    normalize_emo_vector,
    resolve_optional_image_path,
)
from webui_handlers import (
    CLEANUP_DEVICE_CHOICES,
    CLEANUP_LOG_LINES,
    CLEANUP_PROGRESS_IDLE,
    CLEANUP_SPEAKER_MODE_CHOICES,
    EMOTION_GROUP_COUNT,
    EMOTION_TEXT_MODE_INDEX,
    MEDIA_FETCH_EMPTY_INFO,
    MEDIA_FETCH_FOCUS_GENERATION_TAB_JS,
    MEDIA_FETCH_GAIN_MAX_DB,
    MEDIA_FETCH_GAIN_MIN_DB,
    MEDIA_FETCH_GAIN_STEP_DB,
    MEDIA_FETCH_LOG_LINES,
    MEDIA_FETCH_OUTPUT_ROOT,
    available_devices,
    cleanup_default_stages,
    cleanup_preset_choices,
    cleanup_preset_description,
    cleanup_run_ui,
    cleanup_stage_choices,
    describe_engine_worker,
    engine_idle_choices,
    media_fetch_default_quality,
    media_fetch_environment_note,
    media_fetch_format_choices,
    media_fetch_open_folder,
    media_fetch_probe_ui,
    media_fetch_quality_choices,
    media_fetch_quality_description,
    media_fetch_run_ui,
    on_cleanup_preset_change,
    on_cleanup_speaker_mode_change,
    on_device_change,
    on_engine_idle_change,
    on_media_fetch_quality_change,
    unload_engine_worker,
    update_prompt_audio,
)






# The languages IndexTTS-2.5 is trained for. The engine lowercases these and
# falls back to a generic token for anything it does not know, so a wrong value
os.makedirs("outputs/tasks",exist_ok=True)
os.makedirs("prompts",exist_ok=True)
os.makedirs("outputs/used_audios",exist_ok=True)



REFERENCE_WAVEFORM_OPTIONS = gr.WaveformOptions(
    waveform_color="#f7c0cb",
    waveform_progress_color="#a11436",
    trim_region_color="#e23a5e",
    sample_rate=24000,
)








MEDIA_FETCH_PROGRESS_IDLE = render_progress_bar(0.0, "Idle")


GENERATION_PROGRESS_IDLE = render_progress_bar(0.0, "Idle")






theme = gr.themes.Soft()
theme.font = [gr.themes.GoogleFont("Inter"), "Tahoma", "ui-sans-serif", "system-ui", "sans-serif"]
with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown("## Index TTS2 Premium SECourses App V4.1 : https://www.patreon.com/posts/139297407")

    with gr.Tab("Audio Generation"):
        with gr.Row(equal_height=False):
            os.makedirs("prompts",exist_ok=True)

            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes="top-input-panel"):
                    gr.Markdown("### Reference Media (Mandatory!) (Recommended 15 seconds)")
                    with gr.Group(elem_classes="reference-subsection"):
                        media_upload = gr.File(
                            label="Upload Speaker Reference / Audio / Video",
                            file_count="single",
                            file_types=MEDIA_FILE_TYPES,
                            type="filepath",
                            height=120,
                            elem_classes="top-section-flat",
                        )
                        time_ranges_input = gr.Textbox(
                            label="Extract Audio Segments (optional)",
                            placeholder="e.g., 1:3; 3:7; 11:15",
                            value="",
                            info="Optional. Extract and merge only these time ranges from the uploaded audio/video before using it as the speaker reference."
                        )
                        extract_button = gr.Button(
                            "Extract and Use Audio",
                            variant="secondary",
                            elem_id="extract-audio-button",
                            elem_classes=["action-button"],
                        )
                        gr.Markdown("##### Load Audio from Path")
                        with gr.Row():
                            audio_path_input = gr.Textbox(
                                label="Audio File Path",
                                placeholder="Enter full path to audio or video file",
                                value=""
                            )
                            load_audio_button = gr.Button(
                                "Load Audio",
                                variant="secondary",
                                elem_id="load-audio-button",
                                elem_classes=["action-button"],
                            )
                    gr.Markdown("#### Record Speaker Audio", elem_classes="reference-subsection-title")
                    with gr.Group(elem_classes="reference-subsection"):
                        prompt_audio = gr.Audio(
                            label="Active Speaker Reference Audio (Required, 3-90 seconds)",
                            key="prompt_audio",
                            # "upload" as well as the microphone: a phone
                            # cannot use the mic here at all (browsers block
                            # getUserMedia on a plain http:// origin), so
                            # microphone-only left mobile with no way to supply
                            # a reference voice.
                            sources=["upload", "microphone"],
                            type="filepath",
                            format="wav",
                            elem_id="speaker-reference-audio",
                            waveform_options=REFERENCE_WAVEFORM_OPTIONS,
                            elem_classes="top-section-flat",
                        )
                    reference_status = gr.Textbox(
                        label="Reference Audio Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

                prompt_list = os.listdir("prompts")
                default = ''
                if prompt_list:
                    default = prompt_list[0]

            with gr.Column(scale=1, min_width=280):
                with gr.Group(elem_classes=["subtitle-controls-group", "top-input-panel"]):
                    gr.Markdown("### Captions")
                    subtitle_file = gr.File(
                        label="Caption File (.srt, .vtt, .sbv)",
                        file_count="single",
                        file_types=list(SUPPORTED_SUBTITLE_EXTENSIONS),
                        type="filepath",
                        height=120,
                        elem_classes="top-section-flat",
                    )
                    subtitle_mode = gr.Checkbox(
                        label="Use Caption Cue Timing",
                        value=False,
                        info="Generate separate caption timing units, then auto-retime each finished unit to the caption duration before timeline assembly."
                    )
                    gr.Markdown(CAPTION_TIMING_HELP, elem_classes="caption-timing-help")
                    subtitle_status = gr.Textbox(
                        label="Caption Timing Status",
                        value="",
                        interactive=False,
                        visible=False
                    )

            with gr.Column(scale=1, min_width=300):
                use_subprocess_system = gr.Checkbox(
                    label="Use Subprocess System",
                    value=True,
                    interactive=False,
                    info="Always on for IndexTTS-2.5: the engine runs under its own Python, so every generation happens in a separate process. Ending that process also frees its RAM and VRAM."
                )
                input_text_single = gr.TextArea(
                    label="Text to Synthesize",
                    key="input_text_single",
                    elem_id="input-text-source",
                    placeholder="Enter the text you want to convert to speech",
                    info=f"Model v{MODEL_VERSION} | Supports multiple languages. Long texts are automatically segmented. Upload a caption file (.srt/.vtt/.sbv) when you want cue-by-cue timing."
                )
                section_count_refresh_signal = gr.Textbox(
                    value="",
                    show_label=False,
                    container=False,
                    elem_id="section-count-refresh-signal",
                    elem_classes=["ui-hidden-signal"],
                )
                with gr.Row():
                    gen_button = gr.Button(
                        "Generate Speech",
                        key="gen_button",
                        elem_id="generate-speech-button",
                        elem_classes=["action-button"],
                        interactive=True,
                        variant="primary"
                    )
                    open_outputs_button = gr.Button(
                        "Open Outputs Folder",
                        key="open_outputs_button",
                        elem_id="open-outputs-button",
                        elem_classes=["action-button"],
                    )

                section_count_label = gr.Markdown("**Current Sections:** 0")
                autoregressive_batch_size = gr.Slider(
                    interactive=False,
                    label="Section Batch Size",
                    value=1,
                    minimum=1,
                    maximum=8,
                    step=1,
                    info="Not used by IndexTTS-2.5. Real micro-batch size for processing multiple text/subtitle sections together with shared reference conditioning. Higher values increase throughput with a smaller VRAM increase than parallel runs, but still use more memory. Start with 2."
                )

                # Output filename and save used audio options
                with gr.Row():
                    with gr.Column():
                        output_filename = gr.Textbox(
                            label="Output Filename (optional)",
                            placeholder="Optional final filename inside the numbered task folder",
                            value=""
                        )
                        mp4_image_input = gr.Image(
                            label="Image for MP4 (optional)",
                            type="filepath",
                            height=160,
                        )
                    save_used_audio = gr.Checkbox(
                        label="Save Used Reference Audio",
                        value=False,
                        info="Copy the speaker reference audio into this generation's numbered task folder"
                    )

            with gr.Column(scale=1, min_width=280):
                gen_progress = gr.HTML(value=GENERATION_PROGRESS_IDLE)
                output_audio = gr.Audio(
                    label="Generated Result (click to play/download)",
                    visible=True,
                    # filepath, not the gr.Audio default of numpy: the Speed &
                    # Pitch controls take this component as an input and need a
                    # path to hand ffmpeg, not a (sample_rate, array) tuple.
                    type="filepath",
                    key="output_audio"
                )
                with gr.Accordion("Speed & Pitch", open=False):
                    gr.Markdown(
                        "Applied to the generated clip above. The engine has no "
                        "speed or pitch setting, so this reshapes the audio "
                        "afterwards without changing who it sounds like."
                    )
                    shaping_speed = gr.Slider(
                        label="Speed",
                        minimum=voice_shaping.SPEED_MIN,
                        maximum=voice_shaping.SPEED_MAX,
                        step=voice_shaping.SPEED_STEP,
                        value=voice_shaping.SPEED_DEFAULT,
                        info="1.0 is unchanged. Below 1 is slower, above is faster.",
                    )
                    shaping_pitch = gr.Slider(
                        label="Pitch (semitones)",
                        minimum=voice_shaping.PITCH_MIN_SEMITONES,
                        maximum=voice_shaping.PITCH_MAX_SEMITONES,
                        step=voice_shaping.PITCH_STEP_SEMITONES,
                        value=voice_shaping.PITCH_DEFAULT_SEMITONES,
                        info="Negative is deeper, positive is higher. 12 = one octave.",
                    )
                    with gr.Row():
                        shaping_apply_btn = gr.Button("Apply", variant="secondary")
                        shaping_reset_btn = gr.Button("Reset", variant="secondary")
                    shaping_status = gr.Textbox(
                        label="Speed & pitch status",
                        value="",
                        visible=False,
                        interactive=False,
                        lines=1,
                    )
                output_video = gr.Video(
                    label="Generated MP4",
                    visible=False,
                    key="output_video",
                )
                with gr.Accordion("Config Presets (Save / Load)", open=True):
                    gr.Markdown(
                        "Saves and loads tunable controls from the Audio Generation and Advanced Parameters tabs. Working content like Text to Synthesize, uploaded subtitle/reference files, and active reference-media inputs is intentionally not included."
                    )
                    ui_preset_dropdown = gr.Dropdown(
                        label="Select Preset",
                        choices=_list_ui_presets(),
                        value=(_get_last_used_ui_preset() or DEFAULT_UI_PRESET_NAME),
                        allow_custom_value=False,
                    )
                    ui_preset_name = gr.Textbox(
                        label="New Preset Name",
                        placeholder="Enter a preset name to save",
                        value="",
                    )
                    with gr.Row():
                        ui_preset_save_btn = gr.Button(
                            "Save",
                            variant="primary",
                            elem_id="preset-save-button",
                            elem_classes=["action-button"],
                        )
                        ui_preset_load_btn = gr.Button(
                            "Load Selected",
                            elem_id="preset-load-button",
                            elem_classes=["action-button"],
                        )
                    with gr.Row():
                        ui_preset_reset_btn = gr.Button(
                            "Reset Defaults",
                            variant="secondary",
                            elem_id="preset-reset-button",
                            elem_classes=["action-button"],
                        )
                        ui_preset_delete_btn = gr.Button(
                            "Delete",
                            variant="stop",
                            elem_id="preset-delete-button",
                            elem_classes=["action-button"],
                        )
                    ui_preset_status = gr.Markdown("")
                cancel_process_button = gr.Button(
                    "Cancel Running Process",
                    variant="stop",
                    elem_id="cancel-generation-button",
                    elem_classes=["action-button"],
                )
                cancel_process_note = gr.Markdown("Small note: works only when subprocess mode is enabled.")
                cancel_confirm_signal = gr.Checkbox(value=False, visible=False)
                cancel_process_status = gr.Markdown("")

        with gr.Accordion("Function Settings"):
            # 情感控制选项部分 - now showing ALL options including experimental
            with gr.Row():
                emo_control_method = gr.Radio(
                    choices=EMO_CHOICES_ALL,
                    type="index",
                    value=EMO_CHOICES_ALL[0],
                    label="Emotion Control Method",
                    info="Choose how to control emotions: Speaker's natural emotion, reference audio emotion, manual vector control, or text description"
                )
                with gr.Column(min_width=230):
                    tone_preset = gr.Dropdown(
                        label="Tone / delivery preset",
                        choices=tone_presets.TONE_PRESET_NAMES,
                        value=tone_presets.TONE_PRESET_NONE,
                        info="Describes HOW to say it (low and deep, whispered...). Never spoken aloud.",
                    )
                    # Not just "Speed": the voice-shaping panel already has a
                    # slider by that name, which retimes the finished clip. This
                    # one changes how the engine speaks in the first place.
                    speed_factor = gr.Slider(
                        label="Speaking speed",
                        value=tone_presets.DEFAULT_TONE_SPEED,
                        minimum=tone_presets.TONE_SPEED_MIN,
                        maximum=tone_presets.TONE_SPEED_MAX,
                        step=0.05,
                        info="Drag left to speak faster or right to speak slower. "
                             "Choosing a tone preset sets this to suit it, and you "
                             "can change it afterwards.",
                    )
                    language_choice = gr.Dropdown(
                        label="Language",
                        choices=ENGINE_LANGUAGES,
                        value=DEFAULT_ENGINE_LANGUAGE,
                        info="The language the text is written in.",
                    )
                    gr.Markdown(
                        "**For emphasis, CAPITALISE the word** you want stressed "
                        "in the text itself - `I told you NOT to do that`, or part "
                        "of a word, `absoLUTEly`. That works where the emotion "
                        "controls do not, because stress is decided by the text, "
                        "not by tone.",
                        elem_classes="top-section-flat",
                    )
        # 情感参考音频部分
        with gr.Group(visible=False) as emotion_reference_group:
            with gr.Row():
                emo_upload = gr.Audio(
                    label="Upload Emotion Reference Audio",
                    type="filepath"
                )

        # 情感随机采样
        with gr.Row(visible=False) as emotion_randomize_group:
            emo_random = gr.Checkbox(
                label="Random Emotion Sampling",
                value=False,
                info="Enable random sampling from emotion matrix for more varied emotional expression"
            )

        # 情感向量控制部分
        with gr.Group(visible=False) as emotion_vector_group:
            with gr.Row():
                with gr.Column():
                    vec1 = gr.Slider(label="Joy", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Happiness and cheerfulness in voice")
                    vec2 = gr.Slider(label="Anger", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Aggressive and forceful tone")
                    vec3 = gr.Slider(label="Sadness", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Melancholic and sorrowful expression")
                    vec4 = gr.Slider(label="Fear", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Anxious and worried tone")
                with gr.Column():
                    vec5 = gr.Slider(label="Disgust", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Repulsed and disgusted expression")
                    vec6 = gr.Slider(label="Depression", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Low energy and melancholic mood")
                    vec7 = gr.Slider(label="Surprise", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Shocked and amazed reaction")
                    vec8 = gr.Slider(label="Calm", minimum=0.0, maximum=1.0, value=0.0, step=0.05, info="Neutral and peaceful tone")

        with gr.Group(visible=False) as emo_text_group:
            with gr.Row():
                emo_text = gr.Textbox(label="Emotion Description Text",
                                      placeholder="Enter emotion description (or leave empty to automatically use target text as emotion description)",
                                      value="",
                                      info="e.g.: feeling wronged, danger is approaching quietly")

        with gr.Row(visible=False) as emo_weight_group:
            emo_weight = gr.Slider(
                label="Emotion Weight",
                minimum=0.0,
                maximum=1.0,
                value=0.65,
                step=0.01,
                info="Controls the strength of emotion blending. 0 = no emotion, 1 = full emotion from reference. Default: 0.65"
            )

        with gr.Accordion("Advanced Generation Parameter Settings", open=True, visible=True) as advanced_settings_group:
            # Row 1: Diffusion Steps and CFG Rate
            with gr.Row():
                diffusion_steps = gr.Slider(
                    interactive=False,
                    label="Diffusion Steps",
                    value=25,
                    minimum=10,
                    maximum=100,
                    step=1,
                    info="Not used by IndexTTS-2.5. Number of denoising steps in the diffusion model. Higher = better quality but slower. Default: 25"
                )
                inference_cfg_rate = gr.Slider(
                    interactive=False,
                    label="CFG Rate (Classifier-Free Guidance)",
                    value=0.7,
                    minimum=0.0,
                    maximum=2.0,
                    step=0.05,
                    info="Not used by IndexTTS-2.5. Controls how strongly the model follows the voice, emotion, and style characteristics from reference audio. Higher values = stricter adherence to reference, lower = more variation. 0.0 = no guidance (random), 0.7 = balanced (default), >1.0 = very strong adherence to reference characteristics."
                )

            # Row 2: Reference Audio Processing Limits
            with gr.Row():
                with gr.Column():
                    max_speaker_audio_length = gr.Slider(
                        interactive=False,
                        label="Max Speaker Reference Length (seconds)",
                        value=30,
                        minimum=3,
                        maximum=90,
                        step=1,
                        info="Not used by IndexTTS-2.5. How much of the speaker reference audio to use. Model works best with 5-15 seconds. Maximum set to 90 seconds for safety. Default: 30s"
                    )
                with gr.Column():
                    max_emotion_audio_length = gr.Slider(
                        interactive=False,
                        label="Max Emotion Reference Length (seconds)",
                        value=30,
                        minimum=3,
                        maximum=90,
                        step=1,
                        info="Not used by IndexTTS-2.5. How much of the emotion reference audio to use. Model works best with 5-15 seconds. Maximum set to 90 seconds for safety. Default: 30s"
                    )

            # Row 3: Enable Sampling and Temperature
            with gr.Row():
                do_sample = gr.Checkbox(
                    label="Enable Sampling",
                    value=True,
                    info="When ON: Uses random sampling for natural, varied speech. When OFF: Always picks most likely tokens for consistent but potentially robotic output. Keep ON for natural speech."
                )
                temperature = gr.Slider(
                    label="Temperature",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.8,
                    step=0.1,
                    info="Controls speech expressiveness. Higher (0.9-1.2) = more varied intonation and expression. Lower (0.3-0.7) = flatter but more stable speech. Default 0.8 is balanced."
                )

            # Row 4: Beam Search Beams and Max Tokens per Segment
            with gr.Row():
                num_beams = gr.Slider(
                    label="Beam Search Beams",
                    value=3,
                    minimum=1,
                    maximum=10,
                    step=1,
                    info="Explores multiple generation paths simultaneously. Higher (5-10) = better quality but slower. Lower (1-3) = faster but potentially worse quality. Default 3 balances speed and quality. Bigger also uses more VRAM."
                )
                initial_value = max(20, min(PREVIEW_MAX_TEXT_TOKENS, cmd_args.gui_seg_tokens))
                max_text_tokens_per_segment = gr.Textbox(
                    label="Max Tokens per Segment",
                    value=str(initial_value),
                    key="max_text_tokens_per_segment",
                    elem_id="max-tokens-segment-source",
                    info=f"Splits long text into chunks for processing. Valid range: 20-{PREVIEW_MAX_TEXT_TOKENS}. Smaller (80-120) = more natural pauses and consistent quality but slower. Larger (150-200) = faster but may have quality variations. Default: {initial_value}. Bigger value uses more VRAM."
                )

            # Row 5: Save as MP3 and Low Memory Mode
            with gr.Row():
                save_as_mp3 = gr.Checkbox(
                    label="Save as MP3",
                    value=False,
                    visible=MP3_AVAILABLE,
                    info="Save audio as MP3 format instead of WAV" if MP3_AVAILABLE else "Requires pydub: pip install pydub"
                )
                low_memory_mode = gr.Checkbox(
                    interactive=False,
                    label="Low Memory Mode",
                    value=False,
                    info="Not used by IndexTTS-2.5. Enable low memory mode for systems with limited GPU memory (inference will be slower)"
                )
                prevent_vram_accumulation = gr.Checkbox(
                    interactive=False,
                    label="Prevent VRAM Accumulation",
                    value=False,
                    info="Not used by IndexTTS-2.5. Reset beam search cache after each segment. Helps avoid VRAM growth at higher beams (e.g., 8). Slight performance impact."
                )

        with gr.Accordion("Preview Sentence Segmentation Results", open=True) as segments_settings:
            segments_preview = gr.Dataframe(
                headers=["Index", "Type", "Content", "Details"],
                key="segments_preview",
                wrap=True,
            )

    with gr.Tab("Advanced Parameters"):
        gr.Markdown("### 🎯 Advanced Audio Generation Parameters")
        gr.Markdown("_Fine-tune generation parameters for expert control over audio synthesis._")

        with gr.Group():
            gr.Markdown("#### Compute Device")
            with gr.Row():
                device_dropdown = gr.Dropdown(
                    label="Run the model on",
                    choices=available_devices(),
                    value=DEVICE_AUTO,
                    info="The engine picks this up on the next generation, reloading the model onto it.",
                )
                device_status = gr.Textbox(
                    label="Device status",
                    value="",
                    visible=False,
                    interactive=False,
                    lines=1,
                )
            with gr.Row():
                engine_status = gr.Textbox(
                    label="Engine worker",
                    value="Engine worker: not running. It starts on the next generation.",
                    interactive=False,
                    lines=2,
                    scale=3,
                    info="The engine stays loaded between generations so only the "
                         "first one waits for the model. That means it holds VRAM "
                         "while loaded.",
                )
                with gr.Column(scale=1, min_width=200):
                    engine_idle_choice = gr.Dropdown(
                        label="Unload after",
                        choices=engine_idle_choices(),
                        value=int(engine_worker.DEFAULT_IDLE_SECONDS),
                        info="How long the engine may sit idle before it gives "
                             "the GPU back. Set INDEXTTS25_IDLE_SECONDS to change "
                             "the startup default.",
                    )
                    engine_refresh_btn = gr.Button("Refresh status", size="sm")
                    engine_unload_btn = gr.Button(
                        "Unload engine (free VRAM)", size="sm", variant="stop"
                    )

        with gr.Row():
            with gr.Column():
                mp3_bitrate = gr.Dropdown(
                    label="MP3 Bitrate",
                    choices=["128k", "192k", "256k", "320k"],
                    value="256k",
                    info="Audio quality when saving as MP3. 128k = smaller files but lower quality. 320k = best quality but larger files. 256k = good balance for most uses."
                )
            with gr.Column():
                latent_multiplier = gr.Slider(
                    interactive=False,
                    label="Latent Length Multiplier",
                    value=1.72,
                    minimum=1.0,
                    maximum=3.0,
                    step=0.01,
                    info="Not used by IndexTTS-2.5. Controls speech pacing speed. Higher (2.0-3.0) = slower, more stretched speech. Lower (1.0-1.5) = faster, more compressed speech. Default 1.72 is natural pacing."
                )

        with gr.Row():
            with gr.Column():
                top_p = gr.Slider(
                    label="Top-p (Nucleus Sampling)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.01,
                    info="Limits token selection to most probable options. Higher (0.9-1.0) = more varied and expressive speech. Lower (0.3-0.7) = more predictable, conservative speech. Default 0.8 balances variety and stability."
                )
            with gr.Column():
                top_k = gr.Slider(
                    label="Top-k",
                    minimum=0,
                    maximum=100,
                    value=30,
                    step=1,
                    info="Limits selection to k most probable tokens. Higher (50-100) = more speech variety. Lower (10-30) = more consistent speech. 0 = disabled. Default 30 avoids unlikely tokens while maintaining variety."
                )

        with gr.Row():
            with gr.Column():
                repetition_penalty = gr.Number(
                    label="Repetition Penalty",
                    precision=None,
                    value=10.0,
                    minimum=1.0,
                    maximum=20.0,
                    step=0.1,
                    info="Prevents speech from getting stuck in loops. Higher (10-15) = strongly avoids repetition. Lower (1-5) = allows natural repetition. Default 10.0 effectively prevents stuttering."
                )
            with gr.Column():
                length_penalty = gr.Number(
                    label="Length Penalty",
                    precision=None,
                    value=0.0,
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.1,
                    info="Influences speech segment length. Positive (0.5-2.0) = longer segments. Negative (-2.0 to -0.5) = shorter segments. Zero = natural length based on content."
                )

        with gr.Row():
            with gr.Column():
                max_consecutive_silence = gr.Slider(
                    interactive=False,
                    label="Max Consecutive Silent Tokens (0=disabled)",
                    value=0,
                    minimum=0,
                    maximum=100,
                    step=5,
                    info="Not used by IndexTTS-2.5. Removes long pauses in speech. Higher (30-50) = allows longer natural pauses. Lower (5-20) = tighter, more continuous speech. 0 = no pause removal. Try 30 if output has awkward long silences."
                )
            with gr.Column():
                interval_silence = gr.Slider(
                    label="Silence Between Segments (ms)",
                    value=200,
                    minimum=0,
                    maximum=1000,
                    step=50,
                    info="Pause length between text segments. Higher (500-1000ms) = formal presentation style with clear breaks. Lower (50-200ms) = conversational flow. Default 200ms is natural for most content."
                )

        gr.Markdown("### 🎯 Emotion Control Parameters")
        with gr.Row():
            with gr.Column():
                apply_emo_bias = gr.Checkbox(
                    label="Apply Emotion Bias Correction",
                    value=True,
                    info="Prevents emotions from becoming too extreme or unnatural. Keeps emotional expression balanced and realistic. Recommended: Keep ON."
                )
            with gr.Column():
                max_emotion_sum = gr.Slider(
                    label="Max Total Emotion Strength",
                    value=0.8,
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    info="Limits overall emotional intensity. Higher (1.0-2.0) = stronger emotions allowed. Lower (0.3-0.7) = more subtle emotions. Default 0.8 keeps emotions natural."
                )

        with gr.Row():
            with gr.Column():
                max_mel_tokens = gr.Slider(
                    label="Max Mel Tokens",
                    value=1500,
                    minimum=50,
                    maximum=1815,
                    step=10,
                    info=f"Maximum speech length per segment. 1815 tokens ≈ 84 seconds. Lower values may cut off long segments. Default 1500 works for most content.",
                    key="max_mel_tokens"
                )

        gr.Markdown("### 🧠 Advanced Model Architecture Settings (Expert Only!)")
        gr.Markdown("⚠️ **WARNING**: These settings directly affect model internals. Only change if you understand the architecture!")

        with gr.Row():
            with gr.Column():
                semantic_layer = gr.Slider(
                    interactive=False,
                    label="Semantic Feature Extraction Layer",
                    value=17,
                    minimum=1,
                    maximum=24,
                    step=1,
                    info="Not used by IndexTTS-2.5. Which layer of the semantic model to use. Higher layers (15-20) = more expressive, emotion-aware speech. Lower layers (5-12) = clearer pronunciation. Default 17 balances both."
                )
            with gr.Column():
                cfm_cache_length = gr.Slider(
                    interactive=False,
                    label="CFM Max Cache Sequence Length",
                    value=8192,
                    minimum=1024,
                    maximum=16384,
                    step=512,
                    info="Not used by IndexTTS-2.5. Memory allocation for processing speech. Higher (12000-16000) = handles longer segments better but uses more VRAM. Lower (4000-8000) = less memory usage. Default 8192 works for most."
                )

        gr.Markdown("### 🎛️ Custom Emotion Bias Weights")
        gr.Markdown("Fine-tune individual emotion channel biases in normalize_emo_vec() when Apply Emotion Bias is enabled:")
        with gr.Row():
            emo_bias_joy = gr.Slider(label="Joy Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                     info="Adjusts how much joy/happiness is expressed. <1.0 = less joyful, >1.0 = more joyful")
            emo_bias_anger = gr.Slider(label="Anger Bias", value=0.875, minimum=0.5, maximum=1.5, step=0.0625,
                                       info="Adjusts anger intensity. <1.0 = less angry, >1.0 = more angry")
            emo_bias_sad = gr.Slider(label="Sadness Bias", value=1.0, minimum=0.5, maximum=1.5, step=0.0625,
                                     info="Adjusts sadness expression. <1.0 = less sad, >1.0 = more sad")
            emo_bias_fear = gr.Slider(label="Fear Bias", value=1.0, minimum=0.5, maximum=1.5, step=0.0625,
                                      info="Adjusts fear/anxiety expression. <1.0 = less fearful, >1.0 = more fearful")
        with gr.Row():
            emo_bias_disgust = gr.Slider(label="Disgust Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                         info="Adjusts disgust expression. <1.0 = less disgusted, >1.0 = more disgusted")
            emo_bias_depression = gr.Slider(label="Depression Bias", value=0.9375, minimum=0.5, maximum=1.5, step=0.0625,
                                           info="Adjusts melancholic/depressed tone. <1.0 = less depressed, >1.0 = more depressed")
            emo_bias_surprise = gr.Slider(label="Surprise Bias", value=0.6875, minimum=0.5, maximum=1.5, step=0.0625,
                                          info="Adjusts surprise/amazement expression. <1.0 = less surprised, >1.0 = more surprised")
            emo_bias_calm = gr.Slider(label="Calm Bias", value=0.5625, minimum=0.5, maximum=1.5, step=0.0625,
                                      info="Adjusts calm/neutral tone. <1.0 = less calm, >1.0 = more calm and peaceful")

        # Define parameter lists for function calls
        advanced_params = [
            do_sample, top_p, top_k, temperature,
            length_penalty, num_beams, repetition_penalty, max_mel_tokens,
            low_memory_mode, prevent_vram_accumulation,
        ]

        expert_params = [
            diffusion_steps, inference_cfg_rate, interval_silence,
            max_speaker_audio_length, max_emotion_audio_length,
            autoregressive_batch_size, apply_emo_bias, max_emotion_sum,
            latent_multiplier, max_consecutive_silence, mp3_bitrate
        ]

        model_params = [semantic_layer, cfm_cache_length,
                       emo_bias_joy, emo_bias_anger, emo_bias_sad, emo_bias_fear,
                       emo_bias_disgust, emo_bias_depression, emo_bias_surprise, emo_bias_calm]

        _CONFIG_FIELDS = [
            {"section": "audio_generation", "key": "autoregressive_batch_size", "component": autoregressive_batch_size, "default": 1, "kind": "int", "min": 1, "max": 8},
            {"section": "audio_generation", "key": "output_filename", "component": output_filename, "default": "", "kind": "str"},
            {"section": "audio_generation", "key": "save_used_audio", "component": save_used_audio, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "use_subprocess_system", "component": use_subprocess_system, "default": True, "kind": "bool"},
            {"section": "audio_generation", "key": "emo_control_method", "component": emo_control_method, "default": 0, "kind": "emotion_method"},
            {"section": "audio_generation", "key": "emo_random", "component": emo_random, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "vec1", "component": vec1, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec2", "component": vec2, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec3", "component": vec3, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec4", "component": vec4, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec5", "component": vec5, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec6", "component": vec6, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec7", "component": vec7, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "vec8", "component": vec8, "default": 0.0, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "emo_text", "component": emo_text, "default": "", "kind": "str"},
            {"section": "audio_generation", "key": "emo_weight", "component": emo_weight, "default": 0.65, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "audio_generation", "key": "diffusion_steps", "component": diffusion_steps, "default": 25, "kind": "int", "min": 10, "max": 100},
            {"section": "audio_generation", "key": "inference_cfg_rate", "component": inference_cfg_rate, "default": 0.7, "kind": "float", "min": 0.0, "max": 2.0},
            {"section": "audio_generation", "key": "max_speaker_audio_length", "component": max_speaker_audio_length, "default": 30, "kind": "int", "min": 3, "max": 90},
            {"section": "audio_generation", "key": "max_emotion_audio_length", "component": max_emotion_audio_length, "default": 30, "kind": "int", "min": 3, "max": 90},
            {"section": "audio_generation", "key": "do_sample", "component": do_sample, "default": True, "kind": "bool"},
            {"section": "audio_generation", "key": "temperature", "component": temperature, "default": 0.8, "kind": "float", "min": 0.1, "max": 2.0},
            {"section": "audio_generation", "key": "num_beams", "component": num_beams, "default": 3, "kind": "int", "min": 1, "max": 10},
            {
                "section": "audio_generation",
                "key": "max_text_tokens_per_segment",
                "component": max_text_tokens_per_segment,
                "default": str(initial_value),
                "kind": "int_text",
                "min": 20,
                "max": PREVIEW_MAX_TEXT_TOKENS,
            },
            {"section": "audio_generation", "key": "save_as_mp3", "component": save_as_mp3, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "low_memory_mode", "component": low_memory_mode, "default": False, "kind": "bool"},
            {"section": "audio_generation", "key": "prevent_vram_accumulation", "component": prevent_vram_accumulation, "default": False, "kind": "bool"},
            {
                "section": "advanced_parameters",
                "key": "mp3_bitrate",
                "component": mp3_bitrate,
                "default": "256k",
                "kind": "choice",
                "choices": ["128k", "192k", "256k", "320k"],
            },
            {"section": "advanced_parameters", "key": "latent_multiplier", "component": latent_multiplier, "default": 1.72, "kind": "float", "min": 1.0, "max": 3.0},
            {"section": "advanced_parameters", "key": "top_p", "component": top_p, "default": 0.8, "kind": "float", "min": 0.0, "max": 1.0},
            {"section": "advanced_parameters", "key": "top_k", "component": top_k, "default": 30, "kind": "int", "min": 0, "max": 100},
            {"section": "advanced_parameters", "key": "repetition_penalty", "component": repetition_penalty, "default": 10.0, "kind": "float", "min": 1.0, "max": 20.0},
            {"section": "advanced_parameters", "key": "length_penalty", "component": length_penalty, "default": 0.0, "kind": "float", "min": -2.0, "max": 2.0},
            {"section": "advanced_parameters", "key": "max_consecutive_silence", "component": max_consecutive_silence, "default": 0, "kind": "int", "min": 0, "max": 100},
            {"section": "advanced_parameters", "key": "interval_silence", "component": interval_silence, "default": 200, "kind": "int", "min": 0, "max": 1000},
            {"section": "advanced_parameters", "key": "apply_emo_bias", "component": apply_emo_bias, "default": True, "kind": "bool"},
            {"section": "advanced_parameters", "key": "max_emotion_sum", "component": max_emotion_sum, "default": 0.8, "kind": "float", "min": 0.1, "max": 2.0},
            {"section": "advanced_parameters", "key": "max_mel_tokens", "component": max_mel_tokens, "default": 1500, "kind": "int", "min": 50, "max": 1815},
            {"section": "advanced_parameters", "key": "semantic_layer", "component": semantic_layer, "default": 17, "kind": "int", "min": 1, "max": 24},
            {"section": "advanced_parameters", "key": "cfm_cache_length", "component": cfm_cache_length, "default": 8192, "kind": "int", "min": 1024, "max": 16384},
            {"section": "advanced_parameters", "key": "emo_bias_joy", "component": emo_bias_joy, "default": 0.9375, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_anger", "component": emo_bias_anger, "default": 0.875, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_sad", "component": emo_bias_sad, "default": 1.0, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_fear", "component": emo_bias_fear, "default": 1.0, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_disgust", "component": emo_bias_disgust, "default": 0.9375, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_depression", "component": emo_bias_depression, "default": 0.9375, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_surprise", "component": emo_bias_surprise, "default": 0.6875, "kind": "float", "min": 0.5, "max": 1.5},
            {"section": "advanced_parameters", "key": "emo_bias_calm", "component": emo_bias_calm, "default": 0.5625, "kind": "float", "min": 0.5, "max": 1.5},
        ]
        _CONFIG_COMPONENTS = [field["component"] for field in _CONFIG_FIELDS]
        _CONFIG_SECTIONS = tuple(dict.fromkeys(field["section"] for field in _CONFIG_FIELDS))
        _PRESET_AUX_OUTPUTS = [
            emotion_reference_group,
            emotion_randomize_group,
            emotion_vector_group,
            emo_text_group,
            emo_weight_group,
            segments_preview,
            section_count_label,
            subtitle_status,
        ]

        def _default_ui_config() -> Dict[str, Any]:
            cfg: Dict[str, Any] = {
                "_meta": {
                    "version": UI_PRESET_VERSION,
                    "format": UI_PRESET_FORMAT,
                }
            }
            for section in _CONFIG_SECTIONS:
                cfg[section] = {}
            for field in _CONFIG_FIELDS:
                cfg[field["section"]][field["key"]] = field["default"]
            return cfg

        def _merge_ui_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            merged = _default_ui_config()
            if not isinstance(cfg, dict):
                return merged

            meta = cfg.get("_meta")
            if isinstance(meta, dict):
                merged["_meta"].update(meta)

            for section in _CONFIG_SECTIONS:
                section_data = cfg.get(section)
                if isinstance(section_data, dict):
                    merged[section].update(section_data)

            return merged

        def _normalize_bool(value: Any, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return bool(default)
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "yes", "y", "on"}:
                    return True
                if normalized in {"0", "false", "no", "n", "off", ""}:
                    return False
            return bool(default)

        def _normalize_int(value: Any, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
            try:
                normalized = int(float(value))
            except Exception:
                normalized = int(default)
            if min_value is not None:
                normalized = max(min_value, normalized)
            if max_value is not None:
                normalized = min(max_value, normalized)
            return normalized

        def _normalize_float(value: Any, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
            try:
                normalized = float(value)
            except Exception:
                normalized = float(default)
            if min_value is not None:
                normalized = max(min_value, normalized)
            if max_value is not None:
                normalized = min(max_value, normalized)
            return normalized

        def _normalize_text(value: Any, default: str) -> str:
            if value is None:
                return str(default)
            return str(value)

        def _normalize_emotion_method(value: Any, default: int = 0) -> int:
            if hasattr(value, "value"):
                value = value.value
            if isinstance(value, str):
                stripped = value.strip()
                if stripped in EMO_CHOICES_ALL:
                    return EMO_CHOICES_ALL.index(stripped)
            normalized = _normalize_int(value, default, 0, len(EMO_CHOICES_ALL) - 1)
            return normalized

        def _normalize_field_value(field: Dict[str, Any], value: Any) -> Any:
            kind = field["kind"]
            default = field["default"]
            min_value = field.get("min")
            max_value = field.get("max")

            if kind == "str":
                return _normalize_text(value, default)
            if kind == "bool":
                return _normalize_bool(value, default)
            if kind == "int":
                return _normalize_int(value, default, min_value, max_value)
            if kind == "float":
                return _normalize_float(value, default, min_value, max_value)
            if kind == "int_text":
                return str(_normalize_int(value, int(default), min_value, max_value))
            if kind == "choice":
                normalized = _normalize_text(value, default)
                return normalized if normalized in field.get("choices", []) else default
            if kind == "emotion_method":
                return _normalize_emotion_method(value, default)
            return value if value is not None else default

        def _normalize_ui_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            merged = _merge_ui_config(cfg)
            for field in _CONFIG_FIELDS:
                section = field["section"]
                key = field["key"]
                merged[section][key] = _normalize_field_value(field, merged[section].get(key))
            return merged

        def _values_to_ui_config(*values: Any) -> Dict[str, Any]:
            cfg = _default_ui_config()
            for field, value in zip(_CONFIG_FIELDS, values):
                cfg[field["section"]][field["key"]] = _normalize_field_value(field, value)
            return cfg

        def _ui_config_to_values(cfg: Optional[Dict[str, Any]]) -> List[Any]:
            normalized = _normalize_ui_config(cfg)
            return [normalized[field["section"]][field["key"]] for field in _CONFIG_FIELDS]

        def _component_output_value(field: Dict[str, Any], value: Any) -> Any:
            if field["kind"] == "emotion_method":
                index_value = _normalize_emotion_method(value, field["default"])
                return EMO_CHOICES_ALL[index_value]
            return value

        def _build_subtitle_status_for_preset(subtitle_mode_value: bool, current_subtitle_file: Optional[str]):
            if not subtitle_mode_value or not current_subtitle_file:
                return gr.update(value="", visible=False)
            try:
                cues = parse_subtitle_file(current_subtitle_file)
                return gr.update(
                    value=build_subtitle_status_message(cues, subtitle_file=current_subtitle_file),
                    visible=True,
                )
            except Exception as e:
                return gr.update(value=f"Failed to load caption file: {str(e)}", visible=True)

        def _preset_component_updates(
            cfg: Optional[Dict[str, Any]],
            current_text: str,
            current_subtitle_mode: bool,
            current_subtitle_file: Optional[str],
        ) -> List[Any]:
            normalized = _normalize_ui_config(cfg)
            values = [
                _component_output_value(field, normalized[field["section"]][field["key"]])
                for field in _CONFIG_FIELDS
            ]
            text_value = current_text
            subtitle_mode_value = bool(current_subtitle_mode)
            max_tokens_value = normalized["audio_generation"]["max_text_tokens_per_segment"]
            preview_rows = get_preview_rows(
                text_value,
                max_tokens_value,
                subtitle_mode_value,
                current_subtitle_file,
            )
            section_count = build_section_count_message(
                text_value,
                max_tokens_value,
                subtitle_mode_value,
                current_subtitle_file,
            )
            return values + list(on_method_change(normalized["audio_generation"]["emo_control_method"])) + [
                gr.update(value=preview_rows, visible=True, type="array"),
                gr.update(value=section_count),
                _build_subtitle_status_for_preset(subtitle_mode_value, current_subtitle_file),
            ]

        def _save_preset_ui(preset_name: str, *values: Any):
            try:
                saved = _save_ui_preset(preset_name, _values_to_ui_config(*values))
                return (
                    gr.update(choices=_list_ui_presets(), value=saved),
                    gr.update(value=saved),
                    f"✅ Saved preset **{saved}**",
                )
            except Exception as e:
                return (
                    gr.update(choices=_list_ui_presets()),
                    gr.update(),
                    f"[ERROR] Save failed: {e}",
                )

        def _load_preset_ui(
            preset_name: str,
            current_text: str,
            current_subtitle_mode: bool,
            current_subtitle_file: Optional[str],
        ):
            requested = (preset_name or "").strip()
            if not requested or requested == DEFAULT_UI_PRESET_NAME:
                _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
                return (
                    *_preset_component_updates(
                        _default_ui_config(),
                        current_text,
                        current_subtitle_mode,
                        current_subtitle_file,
                    ),
                    "INFO: Loaded default settings.",
                )

            cfg = _load_ui_preset(requested)
            if not cfg:
                _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
                return (
                    *_preset_component_updates(
                        _default_ui_config(),
                        current_text,
                        current_subtitle_mode,
                        current_subtitle_file,
                    ),
                    f"WARNING: Preset **{requested}** not found (loaded defaults).",
                )

            return (
                *_preset_component_updates(
                    cfg,
                    current_text,
                    current_subtitle_mode,
                    current_subtitle_file,
                ),
                f"✅ Loaded preset **{requested}**",
            )

        def _reset_defaults_ui(current_text: str, current_subtitle_mode: bool, current_subtitle_file: Optional[str]):
            _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
            return (
                gr.update(choices=_list_ui_presets(), value=DEFAULT_UI_PRESET_NAME),
                *_preset_component_updates(
                    _default_ui_config(),
                    current_text,
                    current_subtitle_mode,
                    current_subtitle_file,
                ),
                "✅ Reset to defaults",
            )

        def _delete_preset_ui(
            preset_name: str,
            current_text: str,
            current_subtitle_mode: bool,
            current_subtitle_file: Optional[str],
        ):
            requested = (preset_name or "").strip()
            if not requested or requested == DEFAULT_UI_PRESET_NAME:
                _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
                return (
                    gr.update(choices=_list_ui_presets(), value=DEFAULT_UI_PRESET_NAME),
                    *_preset_component_updates(
                        _default_ui_config(),
                        current_text,
                        current_subtitle_mode,
                        current_subtitle_file,
                    ),
                    f"INFO: Built-in preset **{DEFAULT_UI_PRESET_NAME}** cannot be deleted",
                )

            ok = _delete_ui_preset(requested)
            _set_last_used_ui_preset(DEFAULT_UI_PRESET_NAME)
            return (
                gr.update(choices=_list_ui_presets(), value=DEFAULT_UI_PRESET_NAME),
                *_preset_component_updates(
                    _default_ui_config(),
                    current_text,
                    current_subtitle_mode,
                    current_subtitle_file,
                ),
                f"✅ Deleted preset **{requested}**" if ok else f"WARNING: Could not delete preset **{requested}**",
            )

    with gr.Tab("Download & Extract Audio"):
        gr.Markdown("### Download from YouTube or any other site yt-dlp supports, then extract the audio")
        gr.Markdown(media_fetch_environment_note())
        gr.Markdown(
            "Only download material you own or otherwise have the right to use."
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    gr.Markdown("#### Source")
                    mf_url = gr.Textbox(
                        label="Video / audio URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1,
                    )
                    with gr.Row():
                        mf_fetch_info_btn = gr.Button("Fetch Info", variant="secondary")
                        mf_cookies_browser = gr.Dropdown(
                            label="Cookies from browser",
                            choices=media_fetch.COOKIES_BROWSERS,
                            value=media_fetch.COOKIES_NONE,
                            info="Needed for age-restricted or members-only media.",
                        )
                    mf_download_mode = gr.Radio(
                        label="Download mode",
                        choices=media_fetch.DOWNLOAD_MODES,
                        value=media_fetch.DEFAULT_DOWNLOAD_MODE,
                    )
                    mf_keep_source = gr.Checkbox(
                        label="Keep the downloaded source file",
                        value=True,
                    )
                    mf_info = gr.Markdown(MEDIA_FETCH_EMPTY_INFO)

            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    gr.Markdown("#### Audio output")
                    mf_quality = gr.Radio(
                        label="Audio quality",
                        choices=media_fetch_quality_choices(),
                        value=media_fetch.DEFAULT_QUALITY_KEY,
                    )
                    mf_quality_note = gr.Markdown(
                        media_fetch_quality_description(
                            media_fetch.DEFAULT_QUALITY_KEY)
                    )
                    with gr.Row():
                        mf_start_time = gr.Textbox(
                            label="Trim start",
                            placeholder="0:15 or 15",
                            lines=1,
                        )
                        mf_end_time = gr.Textbox(
                            label="Trim end",
                            placeholder="0:45 or 45",
                            lines=1,
                        )

                with gr.Accordion("Manual audio settings", open=False):
                    gr.Markdown(media_fetch.QUALITY_GUIDANCE)
                    # Seeded from the default preset, not from the module
                    # defaults: .change only fires on user interaction, so a
                    # control built with a different value would contradict the
                    # radio above it until the user touched the radio.
                    mf_format = gr.Dropdown(
                        label="Format",
                        choices=media_fetch_format_choices(),
                        value=media_fetch_default_quality().format_key,
                    )
                    with gr.Row():
                        mf_sample_rate = gr.Dropdown(
                            label="Sample rate (Hz)",
                            choices=media_fetch.SAMPLE_RATES,
                            value=media_fetch_default_quality().sample_rate,
                        )
                        mf_channels = gr.Radio(
                            label="Channels",
                            choices=media_fetch.CHANNEL_MODES,
                            value=media_fetch_default_quality().channel_mode,
                        )
                    mf_normalize = gr.Checkbox(
                        label="Normalise loudness (EBU R128)",
                        value=False,
                    )
                    mf_gain = gr.Slider(
                        label="Extra gain (dB)",
                        minimum=MEDIA_FETCH_GAIN_MIN_DB,
                        maximum=MEDIA_FETCH_GAIN_MAX_DB,
                        step=MEDIA_FETCH_GAIN_STEP_DB,
                        value=0.0,
                    )

        with gr.Row():
            mf_run_btn = gr.Button("Download & Extract", variant="primary", scale=2)
            mf_send_btn = gr.Button("Send to Reference Voice", variant="secondary", scale=1)
            mf_open_folder_btn = gr.Button("Open Output Folder", variant="secondary", scale=1)

        mf_progress = gr.HTML(value=MEDIA_FETCH_PROGRESS_IDLE)
        mf_status = gr.Textbox(label="Status", value="", visible=False, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                mf_result_audio = gr.Audio(label="Extracted audio", type="filepath")
                mf_result_path = gr.Textbox(
                    label="Saved to",
                    value="",
                    interactive=False,
                    lines=1,
                )
            with gr.Column(scale=1):
                mf_log = gr.Textbox(
                    label="Log",
                    value="",
                    interactive=False,
                    lines=MEDIA_FETCH_LOG_LINES,
                    max_lines=MEDIA_FETCH_LOG_LINES,
                )


        gr.Markdown("---")
        gr.Markdown("### Clean up the audio")
        gr.Markdown(
            "Removes music, background noise and other people's voices, so what "
            "is left is one clean speaker. Runs on the extracted file above, or "
            "on a file you upload."
        )
        cl_env_note = gr.Markdown(audio_cleanup.cleanup_environment_note())

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    cl_preset = gr.Radio(
                        label="How thorough?",
                        choices=cleanup_preset_choices(),
                        value=cleanup_shared.DEFAULT_PRESET_KEY,
                    )
                    cl_preset_note = gr.Markdown(
                        cleanup_preset_description(cleanup_shared.DEFAULT_PRESET_KEY)
                    )
            with gr.Column(scale=1, min_width=320):
                with gr.Group():
                    cl_upload = gr.Audio(
                        label="Clean a different file instead (optional)",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )

        with gr.Row():
            cl_run_btn = gr.Button("Clean Up Audio", variant="primary", scale=2)
            cl_send_btn = gr.Button(
                "Send Cleaned to Reference Voice", variant="secondary", scale=1)

        cl_progress = gr.HTML(value=CLEANUP_PROGRESS_IDLE)

        with gr.Accordion("Advanced cleanup settings", open=False):
            gr.Markdown(
                "These checkboxes are what actually runs. The preset above just "
                "fills them in."
            )
            cl_stages = gr.CheckboxGroup(
                label="Steps",
                choices=cleanup_stage_choices(),
                value=cleanup_default_stages(),
            )
            with gr.Row():
                cl_vocal_model = gr.Dropdown(
                    label="Vocal isolation model",
                    choices=cleanup_shared.VOCAL_MODELS,
                    value=cleanup_shared.DEFAULT_VOCAL_MODEL,
                    info="Higher SDR is cleaner but slower. First use downloads it.",
                )
                cl_dereverb_model = gr.Dropdown(
                    label="De-reverb model",
                    choices=cleanup_shared.DEREVERB_MODELS,
                    value=cleanup_shared.DEFAULT_DEREVERB_MODEL,
                )
                cl_denoise_model = gr.Dropdown(
                    label="Denoise model",
                    choices=cleanup_shared.DENOISE_MODELS,
                    value=cleanup_shared.DEFAULT_DENOISE_MODEL,
                )
            with gr.Row():
                cl_speaker_mode = gr.Radio(
                    label="Which speaker to keep",
                    choices=CLEANUP_SPEAKER_MODE_CHOICES,
                    value=cleanup_shared.DEFAULT_SPEAKER_MODE,
                )
                cl_speaker_threshold = gr.Slider(
                    label="Voice match threshold",
                    minimum=cleanup_shared.SPEAKER_THRESHOLD_MIN,
                    maximum=cleanup_shared.SPEAKER_THRESHOLD_MAX,
                    step=cleanup_shared.SPEAKER_THRESHOLD_STEP,
                    value=cleanup_shared.DEFAULT_SPEAKER_THRESHOLD,
                    info="Lower keeps more; raise it if another voice slips in.",
                )
            cl_speaker_sample = gr.Audio(
                label="Voice sample of the speaker to keep",
                type="filepath",
                sources=["upload", "microphone"],
                visible=False,
            )
            with gr.Row():
                cl_sample_rate = gr.Dropdown(
                    label="Output sample rate (Hz)",
                    choices=media_fetch.SAMPLE_RATES,
                    value=media_fetch.DEFAULT_SAMPLE_RATE,
                )
                cl_channels = gr.Radio(
                    label="Output channels",
                    choices=[media_fetch.CHANNEL_MONO, media_fetch.CHANNEL_STEREO],
                    value=media_fetch.CHANNEL_MONO,
                )
                cl_device = gr.Radio(
                    label="Run on",
                    choices=CLEANUP_DEVICE_CHOICES,
                    value=cleanup_shared.DEFAULT_DEVICE,
                )
            cl_keep_intermediates = gr.Checkbox(
                label="Keep each step's output file (for comparing)",
                value=False,
            )

        with gr.Row():
            with gr.Column(scale=1):
                cl_result_audio = gr.Audio(label="Cleaned audio", type="filepath")
                cl_result_path = gr.Textbox(
                    label="Saved to", value="", interactive=False, lines=1)
            with gr.Column(scale=1):
                cl_notes = gr.Textbox(
                    label="What it did", value="", interactive=False, lines=4)
                cl_log = gr.Textbox(
                    label="Cleanup log",
                    value="",
                    interactive=False,
                    lines=CLEANUP_LOG_LINES,
                    max_lines=CLEANUP_LOG_LINES,
                )

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

    emo_control_method.change(on_method_change,
        inputs=[emo_control_method],
        outputs=[emotion_reference_group,
                 emotion_randomize_group,
                 emotion_vector_group,
                 emo_text_group,
                 emo_weight_group]
    )


    section_count_refresh_signal.change(
        on_segmentation_inputs_change,
        inputs=[input_text_single, max_text_tokens_per_segment, subtitle_mode, subtitle_file],
        outputs=[segments_preview, section_count_label],
        queue=False,
        show_progress="hidden"
    )

    subtitle_mode.change(
        on_segmentation_inputs_change,
        inputs=[input_text_single, max_text_tokens_per_segment, subtitle_mode, subtitle_file],
        outputs=[segments_preview, section_count_label]
    )

    subtitle_file.change(
        load_subtitle_file,
        inputs=[subtitle_file, input_text_single, subtitle_mode, max_text_tokens_per_segment],
        outputs=[input_text_single, subtitle_mode, subtitle_status, segments_preview, section_count_label]
    )

    prompt_audio.upload(update_prompt_audio,
                         inputs=[],
                         outputs=[gen_button],
                         queue=False,
                         show_progress="hidden")

    media_upload.upload(
        process_media_upload,
        inputs=[media_upload, time_ranges_input],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last"
    )

    extract_button.click(
        extract_audio_segments,
        inputs=[media_upload, time_ranges_input],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden"
    )

    prompt_audio.clear(
        clear_reference_audio,
        outputs=[media_upload, prompt_audio, audio_path_input, reference_status],
        queue=False,
        show_progress="hidden"
    )

    load_audio_button.click(
        load_audio_from_path_ui,
        inputs=[audio_path_input, time_ranges_input],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden"
    )

    gen_button.click(gen_single,
                     inputs=[emo_control_method,prompt_audio, input_text_single, subtitle_mode, subtitle_file, save_used_audio, output_filename, mp4_image_input, emo_upload, emo_weight,
                              vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8,
                               emo_text,emo_random,
                               max_text_tokens_per_segment,
                              speed_factor,
                              language_choice,
                              save_as_mp3,
                              *expert_params,
                              *advanced_params,
                              *model_params,
                               use_subprocess_system,
                       ],
                       outputs=[gen_progress, output_audio, output_video, subtitle_status])

    open_outputs_button.click(open_outputs_folder)

    cancel_process_button.click(
        fn=cancel_generation_process,
        inputs=[use_subprocess_system, cancel_confirm_signal],
        outputs=[cancel_process_status],
        queue=False,
        show_progress="hidden",
        js="""
        (use_subprocess_system, _signal) => [
            use_subprocess_system,
            window.confirm("Cancel the running generation subprocess?")
        ]
        """,
    )

    ui_preset_save_btn.click(
        fn=_save_preset_ui,
        inputs=[ui_preset_name] + _CONFIG_COMPONENTS,
        outputs=[ui_preset_dropdown, ui_preset_name, ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_load_btn.click(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=_CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_dropdown.change(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=_CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_reset_btn.click(
        fn=_reset_defaults_ui,
        inputs=[input_text_single, subtitle_mode, subtitle_file],
        outputs=[ui_preset_dropdown] + _CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    ui_preset_delete_btn.click(
        fn=_delete_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=[ui_preset_dropdown] + _CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )
    demo.load(
        fn=_load_preset_ui,
        inputs=[ui_preset_dropdown, input_text_single, subtitle_mode, subtitle_file],
        outputs=_CONFIG_COMPONENTS + _PRESET_AUX_OUTPUTS + [ui_preset_status],
        queue=False,
        show_progress="hidden",
    )



    # ----------------------------------------------------------------------
    # Media Fetch tab -- event wiring
    # ----------------------------------------------------------------------

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

    mf_quality.change(
        on_media_fetch_quality_change,
        inputs=[mf_quality],
        outputs=[mf_format, mf_sample_rate, mf_channels, mf_quality_note],
        queue=False,
        show_progress="hidden",
    )

    mf_fetch_info_btn.click(
        media_fetch_probe_ui,
        inputs=[mf_url, mf_cookies_browser],
        outputs=[mf_info],
        show_progress="minimal",
    )

    mf_run_btn.click(
        media_fetch_run_ui,
        inputs=[
            mf_url,
            mf_cookies_browser,
            mf_download_mode,
            mf_format,
            mf_sample_rate,
            mf_channels,
            mf_start_time,
            mf_end_time,
            mf_normalize,
            mf_gain,
            mf_keep_source,
        ],
        outputs=[mf_progress, mf_result_audio, mf_result_path, mf_status, mf_log],
        show_progress="minimal",
    )

    mf_send_btn.click(
        send_fetched_to_reference,
        inputs=[mf_result_path],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        # Jump to the generation tab so the hand-off is visible. Done in the
        # browser because selecting a tab from Python needs an explicit
        # gr.Tabs(id=...) parent, which would mean re-indenting every existing
        # tab in this file and conflicting with every upstream pull.
        js=MEDIA_FETCH_FOCUS_GENERATION_TAB_JS,
    )

    mf_open_folder_btn.click(
        media_fetch_open_folder,
        inputs=[],
        outputs=[mf_status],
        queue=False,
        show_progress="hidden",
    )

    cl_preset.change(
        on_cleanup_preset_change,
        inputs=[cl_preset],
        outputs=[cl_stages, cl_preset_note],
        queue=False,
        show_progress="hidden",
    )

    cl_speaker_mode.change(
        on_cleanup_speaker_mode_change,
        inputs=[cl_speaker_mode],
        outputs=[cl_speaker_sample],
        queue=False,
        show_progress="hidden",
    )

    cl_run_btn.click(
        cleanup_run_ui,
        inputs=[
            mf_result_path,
            cl_upload,
            cl_stages,
            cl_vocal_model,
            cl_dereverb_model,
            cl_denoise_model,
            cl_speaker_mode,
            cl_speaker_sample,
            cl_speaker_threshold,
            cl_sample_rate,
            cl_channels,
            cl_device,
            cl_keep_intermediates,
        ],
        outputs=[cl_progress, cl_result_audio, cl_result_path, cl_notes, cl_log],
        show_progress="minimal",
    )

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

    cl_send_btn.click(
        send_cleaned_to_reference,
        inputs=[cl_result_path],
        outputs=[prompt_audio, reference_status],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js=MEDIA_FETCH_FOCUS_GENERATION_TAB_JS,
    )

    # ----------------------------------------------------------------------
    # Emotion presets and compute device
    # ----------------------------------------------------------------------

    _EMOTION_GROUPS = [
        emotion_reference_group,
        emotion_randomize_group,
        emotion_vector_group,
        emo_text_group,
        emo_weight_group,
    ]

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

    tone_preset.change(
        apply_tone_preset_ui,
        inputs=[tone_preset],
        outputs=[emo_text, emo_control_method, speed_factor] + _EMOTION_GROUPS,
        queue=False,
        show_progress="hidden",
    )

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

    shaping_apply_btn.click(
        apply_voice_shaping_ui,
        inputs=[output_audio, shaping_speed, shaping_pitch],
        outputs=[output_audio, shaping_status],
        show_progress="minimal",
    )

    shaping_reset_btn.click(
        reset_voice_shaping_ui,
        inputs=[],
        outputs=[shaping_speed, shaping_pitch, shaping_status],
        queue=False,
        show_progress="hidden",
    )

    device_dropdown.change(
        on_device_change,
        inputs=[device_dropdown],
        outputs=[device_status],
        queue=False,
        show_progress="hidden",
    )

    engine_idle_choice.change(
        on_engine_idle_change,
        inputs=[engine_idle_choice],
        outputs=[engine_status],
        queue=False,
        show_progress="hidden",
    )

    engine_refresh_btn.click(
        describe_engine_worker,
        outputs=[engine_status],
        queue=False,
        show_progress="hidden",
    )

    engine_unload_btn.click(
        unload_engine_worker,
        outputs=[engine_status],
        queue=False,
        show_progress="hidden",
    )

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(
        # --host and --port were parsed but never forwarded, so the app always
        # bound 127.0.0.1 and no other device on the network could reach it.
        server_name=cmd_args.host,
        server_port=cmd_args.port,
        share=cmd_args.share,
        inbrowser=True,
        theme=theme,
        css=APP_CSS,
        head=APP_HEAD,
        favicon_path=APP_FAVICON_PATH,
    )
