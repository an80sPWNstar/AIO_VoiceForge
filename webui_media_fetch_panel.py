"""The Media Fetch tab: download audio from YouTube or other sites, then extract to WAV.

Downloads a video/audio stream from YouTube or any other site yt-dlp supports,
then transcodes the audio track into a format suitable for use as a TTS
reference voice. All media-fetch UI lives here; the handlers and constants
stay in webui_handlers, where they can be tested independently without
a Blocks context.
"""

from __future__ import annotations

from typing import Any, Dict

import gradio as gr

import webui_media_fetch as media_fetch
from webui_handlers import (
    on_media_fetch_quality_change,
    media_fetch_probe_ui,
    media_fetch_run_ui,
    send_fetched_to_reference,
    media_fetch_open_folder,
    media_fetch_environment_note,
    media_fetch_quality_choices,
    media_fetch_quality_description,
    media_fetch_default_quality,
    media_fetch_format_choices,
    MEDIA_FETCH_EMPTY_INFO,
    MEDIA_FETCH_GAIN_MIN_DB,
    MEDIA_FETCH_GAIN_MAX_DB,
    MEDIA_FETCH_GAIN_STEP_DB,
    MEDIA_FETCH_LOG_LINES,
)
from webui_progress import render_progress_bar
from webui_panel_context import PanelContext


# The progress bar displayed during a fetch/extract run.
MEDIA_FETCH_PROGRESS_IDLE = render_progress_bar(0.0, "Idle")


def build_media_fetch_panel(ctx: PanelContext) -> Dict[str, Any]:
    """Build the media-fetch panel's components and wire its events."""

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

    # Wire events
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
        outputs=[ctx.reference.audio, ctx.reference.status],
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
        js=ctx.focus_generation_tab_js,
    )

    mf_open_folder_btn.click(
        media_fetch_open_folder,
        inputs=[],
        outputs=[mf_status],
        queue=False,
        show_progress="hidden",
    )

    return {
        "mf_url": mf_url,
        "mf_fetch_info_btn": mf_fetch_info_btn,
        "mf_cookies_browser": mf_cookies_browser,
        "mf_download_mode": mf_download_mode,
        "mf_keep_source": mf_keep_source,
        "mf_info": mf_info,
        "mf_quality": mf_quality,
        "mf_quality_note": mf_quality_note,
        "mf_start_time": mf_start_time,
        "mf_end_time": mf_end_time,
        "mf_format": mf_format,
        "mf_sample_rate": mf_sample_rate,
        "mf_channels": mf_channels,
        "mf_normalize": mf_normalize,
        "mf_gain": mf_gain,
        "mf_run_btn": mf_run_btn,
        "mf_send_btn": mf_send_btn,
        "mf_open_folder_btn": mf_open_folder_btn,
        "mf_progress": mf_progress,
        "mf_status": mf_status,
        "mf_result_audio": mf_result_audio,
        "mf_result_path": mf_result_path,
        "mf_log": mf_log,
    }
