"""Build and wire the reference-clip selection panel.

This is stage 3 of the ingestion tab's three-stage pipeline (fetch -> clean ->
pick a clip). It reads the previous stage's result path, which is passed in as
`upstream_path` — the cleanup panel's `cl_result_path` — with the optional
upload winning over it when both are present.

Unlike the older tabs, this panel both builds and wires itself in the same
module. A new tab has no re-indent cost, and this file is already past the split
threshold the September review recorded.
"""

from __future__ import annotations

from typing import Any, Dict

import gradio as gr

import audio_segmentation as segmentation
import webui_character_handlers as character_handlers
import webui_segmentation_handlers as segmentation_handlers
from webui_assets import MEDIA_FILE_TYPES
from webui_panel_context import PanelContext


def build_segmentation_panel(ctx: PanelContext, upstream_path: Any) -> Dict[str, Any]:
    """Build the reference-clip panel's components and wire its events.

    Takes the cleanup panel's result path as `upstream_path` so the upload
    can win over it. Returns all created components for the caller to wire
    any cross-panel events.
    """

    # Get the initial character name for the save-target caption.
    _char_choices0, _char_first0, _char_name0, _char_desc0 = (
        character_handlers.initial_state()
    )

    gr.Markdown("### Find a reference clip")
    gr.Markdown(
        "Splits a recording where the speaker stops and measures every "
        "piece, so a usable reference can be found without listening to "
        "all of it. Runs on the cleaned audio above, or on a file you "
        "upload. Pitch range is the column to sort by if you want an "
        "animated delivery; a narrow range is a flat one."
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            with gr.Group():
                sg_upload = gr.File(
                    label="Scan a different file instead (optional, audio or video)",
                    file_count="single",
                    file_types=MEDIA_FILE_TYPES,
                    type="filepath",
                )
        with gr.Column(scale=1, min_width=320):
            with gr.Group():
                sg_best_first = gr.Checkbox(
                    label="Best candidates first",
                    value=True,
                    info="Off lists them in the order they were spoken.",
                )
                sg_min_seconds = gr.Slider(
                    label="Ignore anything shorter than (seconds)",
                    minimum=0.5,
                    maximum=10.0,
                    step=0.5,
                    value=segmentation.MIN_USEFUL_SECONDS,
                    info="A reference shorter than a couple of seconds "
                         "does not carry a delivery.",
                )

    sg_scan_btn = gr.Button("Scan for Reference Clips", variant="primary")

    sg_progress = gr.HTML(value=segmentation_handlers.SCAN_PROGRESS_IDLE)

    with gr.Accordion("Advanced splitting settings", open=False):
        gr.Markdown(
            "These decide where the cuts land. The defaults keep breaths "
            "and room tone out without cutting inside a quiet word."
        )
        with gr.Row():
            sg_top_db = gr.Slider(
                label="Silence threshold (dB below the loudest part)",
                minimum=15,
                maximum=60,
                step=1,
                value=segmentation.DEFAULT_SILENCE_TOP_DB,
                info="Lower splits more eagerly.",
            )
            sg_merge_gap = gr.Slider(
                label="Join pieces separated by less than (seconds)",
                minimum=0.0,
                maximum=2.0,
                step=0.05,
                value=segmentation.DEFAULT_MERGE_GAP_SECONDS,
                info="Pauses inside a sentence run 0.2-0.4s; cutting "
                     "there produces fragments that sound clipped.",
            )

    sg_table = gr.Dataframe(
        headers=segmentation_handlers.SEGMENT_TABLE_HEADERS,
        key="segment_scan_table",
        wrap=True,
    )

    with gr.Row():
        with gr.Column(scale=1):
            sg_select = gr.Dropdown(
                label="Segment",
                choices=[],
                value=None,
                info="Pick one to hear it, then use it as the reference "
                     "voice or save it to a character.",
            )
            sg_preview_audio = gr.Audio(label="Segment preview",
                                        type="filepath")
            # The completion action sits here, under the table and the
            # preview, rather than up beside Scan. Everything that makes
            # someone confident about which segment to take renders below
            # that button row, so a button placed there is scrolled out of
            # sight by the time the decision is made -- which is how this
            # panel's predecessor ended up with an action nobody found.
            sg_use_btn = gr.Button("Use Segment as Reference Voice",
                                   variant="primary")
        with gr.Column(scale=1):
            sg_status = gr.Textbox(
                label="Status", value="", interactive=False, lines=2,
                visible=False)
            sg_detail = gr.Textbox(
                label="Selected segment", value="", interactive=False,
                lines=3, visible=False)
            sg_target_voice = gr.Markdown(
                segmentation_handlers.describe_save_target(_char_name0))
            sg_save_name = gr.Textbox(
                label="Label for the saved clip (optional)",
                value="",
                lines=1,
                info="Left empty, the clip is labelled with the recording "
                     "and the segment number.",
            )
            sg_save_btn = gr.Button("Save Segment to Selected Voice",
                                    variant="secondary")

    sg_state = gr.State(None)

    # Wire the panel's events.

    # The scan is the only generator here; the three that follow it read the
    # state it left behind. Picking a segment cuts it out immediately so the
    # player has something to play -- that is a second of work on an already
    # decoded file, not another scan.
    sg_scan_btn.click(
        segmentation_handlers.scan_recording_ui,
        inputs=[
            upstream_path,
            sg_upload,
            sg_top_db,
            sg_merge_gap,
            sg_min_seconds,
            sg_best_first,
        ],
        outputs=[sg_table, sg_state, sg_select, sg_progress, sg_status],
        show_progress="minimal",
    )

    sg_select.change(
        segmentation_handlers.preview_segment_ui,
        inputs=[sg_state, sg_select],
        outputs=[sg_preview_audio, sg_detail],
        queue=False,
    )

    sg_use_btn.click(
        segmentation_handlers.use_segment_as_reference_ui,
        inputs=[sg_state, sg_select],
        outputs=[ctx.reference.audio, ctx.reference.status],
        queue=False,
        show_progress="hidden",
    ).then(
        # Handed the reference the click produced, so a press that failed its
        # preconditions leaves the user on this tab with the message beside
        # the button instead of bouncing them to another tab to read it.
        fn=None,
        inputs=[ctx.reference.audio],
        outputs=None,
        js=segmentation_handlers.FOCUS_GENERATION_TAB_IF_SET_JS,
    )

    # The voice a save would land on is chosen on the other tab, so its name
    # is echoed beside the save button. character_name already holds the
    # resolved name and is refreshed by every control that can change it.
    for _character_event in (ctx.character.name.change, ctx.character.select.change):
        _character_event(
            segmentation_handlers.describe_save_target,
            inputs=[ctx.character.name],
            outputs=[sg_target_voice],
            queue=False,
            show_progress="hidden",
        )

    sg_save_btn.click(
        segmentation_handlers.save_segment_to_voice_ui,
        inputs=[ctx.character.select, sg_state, sg_select,
                sg_save_name],
        # The first three land on the Character Library panel, which is on
        # the other tab and has to reflect the new clip. The fourth is the
        # status, and it goes to the box beside this button rather than the
        # library's own: a confirmation on a tab the user is not looking at
        # reads as the button having done nothing.
        outputs=[ctx.character.select, ctx.character.name, ctx.character.summary,
                 sg_status],
        queue=False,
    )

    return {
        "sg_upload": sg_upload,
        "sg_best_first": sg_best_first,
        "sg_min_seconds": sg_min_seconds,
        "sg_scan_btn": sg_scan_btn,
        "sg_progress": sg_progress,
        "sg_top_db": sg_top_db,
        "sg_merge_gap": sg_merge_gap,
        "sg_table": sg_table,
        "sg_select": sg_select,
        "sg_preview_audio": sg_preview_audio,
        "sg_use_btn": sg_use_btn,
        "sg_status": sg_status,
        "sg_detail": sg_detail,
        "sg_target_voice": sg_target_voice,
        "sg_save_name": sg_save_name,
        "sg_save_btn": sg_save_btn,
        "sg_state": sg_state,
    }
