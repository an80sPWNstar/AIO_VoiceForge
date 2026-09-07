"""The audio cleanup panel: remove music, noise, and background voices.

Stage 2 of the ingestion pipeline. Reads the previous stage's output path
(media-fetch result) from upstream_path, with manual upload winning over it
when both are present.

The presets are named for what they do to the audio -- just remove the music,
a standard clean-up, an aggressive studio clean, or custom -- and each one
simply fills in the stage checkboxes, which are what actually runs. Manual
voice selection stops after analysis and lists every voice it heard, so the
one to extract can be picked by ear.
Once extraction is done, a best reference clip can be saved to the Voice
Library for use in generation.
"""

from __future__ import annotations

from typing import Any, Dict

import gradio as gr

import audio_cleanup_shared as cleanup_shared
import webui_audio_cleanup as audio_cleanup
import webui_character_handlers as character_handlers
import webui_handlers
import webui_media_fetch as media_fetch
import webui_segmentation_handlers as segmentation_handlers
from webui_assets import MEDIA_FILE_TYPES
from webui_panel_context import PanelContext


def build_cleanup_panel(ctx: PanelContext, upstream_path: Any) -> Dict[str, Any]:
    """Build the cleanup panel's components and wire its events."""

    # The save-target caption names the voice a saved clip would land on, and
    # it has to say so from the first render -- the character events below only
    # refresh it once the library is touched. Same seed the library panel uses.
    _char_choices0, _char_first0, _char_name0, _char_desc0 = (
        character_handlers.initial_state()
    )

    gr.Markdown("### Clean up the audio")
    gr.Markdown(
        "Removes music, background noise and other people's voices, so what "
        "is left is one clean speaker. Runs on the extracted file above, or "
        "on a file you upload."
    )

    # Initial environment note shows the device info.
    cl_env_note = gr.Markdown(
        audio_cleanup.cleanup_environment_note(
            webui_handlers.cleanup_default_device()
        )
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            with gr.Group():
                cl_preset = gr.Radio(
                    label="How thorough?",
                    choices=webui_handlers.cleanup_preset_choices(),
                    value=cleanup_shared.DEFAULT_PRESET_KEY,
                )
                cl_preset_note = gr.Markdown(
                    webui_handlers.cleanup_preset_description(
                        cleanup_shared.DEFAULT_PRESET_KEY
                    )
                )
                cl_device = gr.Radio(
                    label="Run on",
                    choices=webui_handlers.cleanup_device_choices(),
                    value=webui_handlers.cleanup_default_device(),
                )
                _cl_device_note = webui_handlers.cleanup_device_note()
                if _cl_device_note:
                    gr.Markdown(_cl_device_note)
        with gr.Column(scale=1, min_width=320):
            with gr.Group():
                cl_upload = gr.File(
                    label="Clean a different file instead (optional, audio or video)",
                    file_count="single",
                    file_types=MEDIA_FILE_TYPES,
                    type="filepath",
                )
                cl_voice_mode = gr.Radio(
                    label="Voice selection",
                    choices=webui_handlers.VOICE_MODE_CHOICES,
                    value=webui_handlers.VOICE_MODE_AUTO,
                    info="Manual stops after analysis and lists every "
                    "voice it hears, so you pick the one to extract.",
                )

    with gr.Row():
        cl_run_btn = gr.Button("Clean Up Audio", variant="primary", scale=2)
        # Scales are set against the LONGEST label, not evenly: "Send Cleaned
        # to Reference Voice" is far wider than its neighbours and wraps to two
        # lines if it gets an equal share. Adding a third key to this row is
        # what first broke it.
        cl_send_btn = gr.Button(
            "Send Cleaned to Reference Voice", variant="secondary", scale=3
        )
        cl_cancel_btn = gr.Button("Stop Cleanup", variant="stop", scale=1)

    # The status line sits BELOW the row, not in it. A Markdown in a row of
    # keys takes width from them, and the first casualty was "Send Cleaned to
    # Reference Voice" wrapping onto two lines -- the wrapping this project
    # already ruled out once. Keys keep the row; prose goes underneath.
    cl_cancel_status = gr.Markdown("")

    cl_progress = gr.HTML(value=webui_handlers.CLEANUP_PROGRESS_IDLE)

    with gr.Accordion("Advanced cleanup settings", open=False):
        gr.Markdown(
            "These checkboxes are what actually runs. The preset above just "
            "fills them in."
        )
        cl_stages = gr.CheckboxGroup(
            label="Steps",
            choices=webui_handlers.cleanup_stage_choices(),
            value=webui_handlers.cleanup_default_stages(),
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
                choices=webui_handlers.CLEANUP_SPEAKER_MODE_CHOICES,
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
                choices=[
                    media_fetch.CHANNEL_MONO,
                    media_fetch.CHANNEL_STEREO,
                ],
                value=media_fetch.CHANNEL_MONO,
            )
        cl_keep_intermediates = gr.Checkbox(
            label="Keep each step's output file (for comparing)",
            value=False,
        )

    # Manual voice selection. The whole group stays hidden until an
    # analysis run fills it; its state survives until the next analysis or
    # a page reload, so several voices can be extracted from one clip.
    cl_voices_state = gr.State(None)
    with gr.Group(visible=False) as cl_voices_group:
        gr.Markdown("#### Voices heard in this clip")
        cl_voice_labels = []
        cl_voice_players = []
        for _voice_slot in range(webui_handlers.MAX_VOICE_ROWS):
            with gr.Row():
                cl_voice_labels.append(gr.Markdown(visible=False))
                cl_voice_players.append(
                    gr.Audio(
                        type="filepath",
                        visible=False,
                        show_label=False,
                        scale=2,
                    )
                )
        with gr.Row():
            cl_voice_pick = gr.Radio(
                label="Voice to extract", choices=[], value=None, scale=2
            )
            cl_extract_btn = gr.Button("Extract Selected Voice", variant="primary", scale=1)
        with gr.Row():
            with gr.Column(scale=1):
                cl_reference_audio = gr.Audio(
                    label="Best reference clip (15s or less)",
                    type="filepath",
                )
                cl_reference_path = gr.Textbox(
                    label="Reference saved to", value="", interactive=False, lines=1
                )
            with gr.Column(scale=1):
                cl_voice_save_target = gr.Markdown(
                    segmentation_handlers.describe_save_target(_char_name0)
                )
                cl_voice_save_name = gr.Textbox(
                    label="Label for the saved clip (optional)",
                    value="",
                    lines=1,
                )
                cl_voice_save_btn = gr.Button("Save Reference to Voice")
                cl_voice_save_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    lines=2,
                    visible=False,
                )

    with gr.Row():
        with gr.Column(scale=1):
            cl_result_audio = gr.Audio(label="Cleaned audio", type="filepath")
            cl_result_path = gr.Textbox(
                label="Saved to", value="", interactive=False, lines=1
            )
        with gr.Column(scale=1):
            cl_notes = gr.Textbox(
                label="What it did", value="", interactive=False, lines=4
            )
            cl_log = gr.Textbox(
                label="Cleanup log",
                value="",
                interactive=False,
                lines=webui_handlers.CLEANUP_LOG_LINES,
                max_lines=webui_handlers.CLEANUP_LOG_LINES,
            )

    # Event wiring
    cl_preset.change(
        webui_handlers.on_cleanup_preset_change,
        inputs=[cl_preset],
        outputs=[cl_stages, cl_preset_note],
        queue=False,
        show_progress="hidden",
    )

    cl_speaker_mode.change(
        webui_handlers.on_cleanup_speaker_mode_change,
        inputs=[cl_speaker_mode],
        outputs=[cl_speaker_sample],
        queue=False,
        show_progress="hidden",
    )

    cl_device.change(
        webui_handlers.on_cleanup_device_change,
        inputs=[cl_device],
        outputs=[cl_env_note],
        queue=False,
        show_progress="hidden",
    )

    cl_run_btn.click(
        webui_handlers.cleanup_run_ui,
        inputs=[
            upstream_path,
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
            cl_voice_mode,
        ],
        outputs=[
            cl_progress,
            cl_result_audio,
            cl_result_path,
            cl_notes,
            cl_log,
            cl_voices_state,
            cl_voice_pick,
            cl_voices_group,
        ]
        + cl_voice_players
        + cl_voice_labels,
        show_progress="minimal",
        concurrency_limit=1,  # Only one cleanup job at a time; a second click queues behind the first, which is why orphaned runs looked like a UI freeze.
    )

    cl_cancel_btn.click(
        webui_handlers.cancel_cleanup_ui,
        inputs=[],
        outputs=[cl_cancel_status],
        queue=False,  # LOAD-BEARING: queue=False means this runs immediately, not queued behind the cleanup job it is meant to cancel. Removing this would break cancellation.
        show_progress="hidden",
    )

    cl_extract_btn.click(
        webui_handlers.voice_extract_run_ui,
        inputs=[
            cl_voices_state,
            cl_voice_pick,
            cl_stages,
            cl_speaker_threshold,
            cl_sample_rate,
            cl_channels,
            cl_device,
        ],
        outputs=[
            cl_progress,
            cl_result_audio,
            cl_result_path,
            cl_reference_audio,
            cl_reference_path,
            cl_notes,
            cl_log,
        ],
        show_progress="minimal",
    )

    cl_voice_save_btn.click(
        webui_handlers.save_voice_reference_ui,
        inputs=[
            ctx.character.select,
            cl_reference_path,
            cl_voice_save_name,
        ],
        outputs=[
            ctx.character.select,
            ctx.character.name,
            ctx.character.summary,
            cl_voice_save_status,
        ],
        queue=False,
    )

    cl_send_btn.click(
        webui_handlers.send_cleaned_to_reference,
        inputs=[cl_result_path],
        outputs=[ctx.reference.audio, ctx.reference.status],
        queue=False,
        show_progress="hidden",
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js=ctx.focus_generation_tab_js,
    )

    # Update the save target caption when the character selection changes.
    for _character_event in (ctx.character.name.change, ctx.character.select.change):
        _character_event(
            segmentation_handlers.describe_save_target,
            inputs=[ctx.character.name],
            outputs=[cl_voice_save_target],
            queue=False,
            show_progress="hidden",
        )

    # Return a dict mapping each component's variable name to the component.
    return {
        "cl_env_note": cl_env_note,
        "cl_preset": cl_preset,
        "cl_preset_note": cl_preset_note,
        "cl_device": cl_device,
        "cl_upload": cl_upload,
        "cl_voice_mode": cl_voice_mode,
        "cl_run_btn": cl_run_btn,
        "cl_send_btn": cl_send_btn,
        "cl_cancel_btn": cl_cancel_btn,
        "cl_cancel_status": cl_cancel_status,
        "cl_progress": cl_progress,
        "cl_stages": cl_stages,
        "cl_vocal_model": cl_vocal_model,
        "cl_dereverb_model": cl_dereverb_model,
        "cl_denoise_model": cl_denoise_model,
        "cl_speaker_mode": cl_speaker_mode,
        "cl_speaker_threshold": cl_speaker_threshold,
        "cl_speaker_sample": cl_speaker_sample,
        "cl_sample_rate": cl_sample_rate,
        "cl_channels": cl_channels,
        "cl_keep_intermediates": cl_keep_intermediates,
        "cl_voices_state": cl_voices_state,
        "cl_voices_group": cl_voices_group,
        "cl_voice_labels": cl_voice_labels,
        "cl_voice_players": cl_voice_players,
        "cl_voice_pick": cl_voice_pick,
        "cl_extract_btn": cl_extract_btn,
        "cl_reference_audio": cl_reference_audio,
        "cl_reference_path": cl_reference_path,
        "cl_voice_save_target": cl_voice_save_target,
        "cl_voice_save_name": cl_voice_save_name,
        "cl_voice_save_btn": cl_voice_save_btn,
        "cl_voice_save_status": cl_voice_save_status,
        "cl_result_audio": cl_result_audio,
        "cl_result_path": cl_result_path,
        "cl_notes": cl_notes,
        "cl_log": cl_log,
    }
