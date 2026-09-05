"""Choosing a reference clip out of a long recording.

`audio_segmentation` measures; this module is what the panel does with the
measurements. It runs the scan off the UI thread, formats each segment into a
row a person can read, cuts the chosen span out to its own wav, and hands that
wav either to the reference slot or to a character in the library.

Three things shape the design.

The scan is slow enough to need reporting. A ten-minute recording takes about
a minute, almost all of it pitch tracking, and a minute of a page that looks
frozen reads as broken. So the scan runs on a worker thread reporting into a
queue and the handler is a generator that drains it -- the same shape as
`media_fetch_run_ui` and `cleanup_run_ui`, which solved this already.

The chosen segment is picked from a dropdown, not by clicking the table. The
table is for reading -- nine columns of measurements across a few hundred
rows -- and a click on a table reports a cell, which is one more thing to get
wrong. A dropdown says what is selected without the user having to remember
which row they clicked, and it is a control the panel already uses everywhere
else.

Measurements taken here follow the clip into the library, which is what
`add_reference_to_character_ui`'s duration note was waiting for: a clip cut
from a scan arrives with its loudness and pitch already known, rather than a
duration read off the wav header and nothing else.
"""

from __future__ import annotations

import hashlib
import math
import os
import queue
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import librosa
import numpy as np

import audio_segmentation as segmentation
import character_store as store
import webui_character_handlers as characters
import webui_media_utils as media_utils
from webui_progress import render_progress_bar

# The most recent video-audio extraction this tab made (see scan_recording_ui).
_LAST_EXTRACTED: Dict[str, Optional[str]] = {"path": None}

# The table's columns. "#" is the segment's place in the recording, kept even
# when the rows are sorted best-first, because it is how someone finds the
# moment again in the original file.
SEGMENT_TABLE_HEADERS = [
    "#", "Start", "Length", "Loudness", "Pitch", "Pitch range",
    "Voiced", "Pace", "Verdict",
]

# A measurement that could not be taken. Not "0" and not blank: a zero would
# read as measured-and-silent, and a blank cell as a formatting bug.
MISSING_MEASUREMENT = "—"

# A long recording can yield several hundred segments. Past this the table
# stops being readable and starts being a scroll, so it is truncated -- and
# the status line always says when it was, because a silently shortened table
# reads as a complete one. The dropdown is never truncated.
MAX_TABLE_ROWS = 200

# Long enough for a scan of a very long recording to finish reporting; the
# work itself is already done by the time this matters.
SCAN_THREAD_JOIN_SECONDS = 30

# What the bar shows before anything has been scanned.
SCAN_PROGRESS_IDLE = render_progress_bar(0.0, "Idle")

# The hand-off JS the other tabs use jumps to the generation tab every time it
# is chained, so a press that failed its preconditions teleports the user to
# another tab to read the error -- which reads as a navigation glitch rather
# than as a validation message. This variant is handed the reference path the
# handler produced and moves only when one was actually set.
FOCUS_GENERATION_TAB_IF_SET_JS = (
    "(reference) => {"
    "  if (!reference) return;"
    "  const target = 'Audio Generation';"
    "  const all = Array.from(document.querySelectorAll('button'))"
    "      .filter(b => b.textContent.trim() === target);"
    "  const visible = all.filter(b => b.offsetParent !== null);"
    "  const tab = visible.length ? visible[visible.length - 1] : all[0];"
    "  if (tab) tab.click();"
    "}"
)

# Where cut segments are written, under the output root. They are working
# files: a clip saved to a character is copied into the library by the store,
# so these are only what the preview player reads.
SEGMENT_EXPORT_DIRNAME = "segments"

# How much of the source path goes into an export's name. Long enough that two
# recordings will not collide, short enough to leave the name readable.
SOURCE_DIGEST_LENGTH = 8

# Ramped in and out over this at each end of a cut. The split boundary is not
# a zero crossing, and a hard edge is a click on every generation that uses
# the clip.
EDGE_FADE_SECONDS = 0.005

# Verdicts. The 15-second cap is the engine's, not ours; a segment over it
# still clones, using only its opening, which is worth saying rather than
# dropping the row.
VERDICT_USABLE = "good"
VERDICT_TOO_SHORT = f"under {segmentation.MIN_USEFUL_SECONDS:.0f}s"
VERDICT_OVER_CAP = f"over {segmentation.ENGINE_REFERENCE_SECONDS:.0f}s cap"


def default_export_root() -> str:
    """Where exported spans go by default.

    Absolute, deliberately. `outputs` resolved against the working directory
    put generated files wherever the app happened to be launched from, which
    is the bug commit 58095ce fixed for task output; anything else written
    next to it has the same trap.
    """
    return os.path.join(os.path.abspath("outputs"), SEGMENT_EXPORT_DIRNAME)


# --------------------------------------------------------------- formatting

def _format_clock(seconds: Any) -> str:
    """Seconds as m:ss.s, so a position in a long recording is scannable."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return MISSING_MEASUREMENT
    # Before the clamp, not after. An infinity survives float() and then kills
    # int(), because divmod(inf, 60) is (nan, nan); and clamping first would
    # hide it, since max(0.0, nan) is 0.0 and would report a position of zero
    # for a number that is not one. A real scan cannot produce either, but
    # this formats whatever is in the browser's copy of the state.
    if not math.isfinite(value):
        return MISSING_MEASUREMENT
    total = max(0.0, value)
    # Rounded before the split, not after: divmod first would render 59.96 as
    # "0:60.0", because the carry happens in the f-string where the minutes
    # are already fixed.
    minutes, remainder = divmod(round(total, 1), 60.0)
    return f"{int(minutes)}:{remainder:04.1f}"


def _format_measurement(value: Any, suffix: str = "", digits: int = 1) -> str:
    """A measured number, or the missing marker when it could not be taken."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return MISSING_MEASUREMENT
    return f"{float(value):.{digits}f}{suffix}"


def _format_percent(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return MISSING_MEASUREMENT
    return f"{float(value) * 100:.0f}%"


def verdict_for(segment: segmentation.Segment) -> str:
    """What is worth saying about this segment's length.

    Length is the only measurement with a right answer. There is no best
    loudness or pitch -- only the delivery someone wants -- so those are
    reported and not judged.
    """
    if not segment.is_long_enough:
        return VERDICT_TOO_SHORT
    if not segment.fits_the_reference_cap:
        return VERDICT_OVER_CAP
    return VERDICT_USABLE


def segment_row(segment: segmentation.Segment) -> List[str]:
    """One table row: everything measured about one candidate, formatted."""
    return [
        str(segment.index + 1),
        _format_clock(segment.start_s),
        _format_measurement(segment.duration_s, "s"),
        _format_measurement(segment.lufs, " LUFS"),
        _format_measurement(segment.pitch_hz_mean, " Hz", 0),
        _format_measurement(segment.pitch_hz_range, " Hz", 0),
        _format_percent(segment.voiced_fraction),
        _format_measurement(segment.onset_rate_hz, "/s"),
        verdict_for(segment),
    ]


def segment_rows(segments: Sequence[segmentation.Segment]) -> List[List[str]]:
    """The table, truncated to what stays readable. See MAX_TABLE_ROWS."""
    return [segment_row(segment) for segment in segments[:MAX_TABLE_ROWS]]


def segment_label(segment: segmentation.Segment) -> str:
    """One dropdown entry: enough to choose by without reading the table."""
    return "  ·  ".join([
        f"#{segment.index + 1}",
        _format_clock(segment.start_s),
        _format_measurement(segment.duration_s, "s"),
        _format_measurement(segment.lufs, " LUFS"),
        f"{_format_measurement(segment.pitch_hz_range, '', 0)} Hz range",
    ])


def segment_choices(segments: Sequence[segmentation.Segment]) -> List[Tuple[str, int]]:
    """Dropdown choices.

    The value is the segment's index in the recording, so it keeps meaning
    whichever order the rows happen to be shown in.
    """
    return [(segment_label(segment), segment.index) for segment in segments]


# ------------------------------------------------------------------- state

def scan_state(
    source_path: str,
    sample_rate: int,
    segments: Sequence[segmentation.Segment],
) -> Dict[str, Any]:
    """What a scan leaves behind for the buttons that come after it.

    Plain data rather than the dataclasses: this crosses into gr.State, and a
    dict of numbers survives anything gradio decides to do to it.
    """
    return {
        "source": source_path,
        "sample_rate": int(sample_rate),
        "segments": [segment.as_dict() for segment in segments],
    }


def segment_from_state(
    state: Optional[Dict[str, Any]], index: Any
) -> Optional[segmentation.Segment]:
    """The chosen segment, or None when the state and the choice disagree.

    They can: a scan replaces the state while a stale index is still sitting
    in the dropdown, and gradio hands back whatever the browser last had.
    """
    if not isinstance(state, dict):
        return None
    try:
        wanted = int(index)
    except (TypeError, ValueError):
        return None
    for raw in state.get("segments") or []:
        if not isinstance(raw, dict) or raw.get("index") != wanted:
            continue
        try:
            return segmentation.Segment(**raw)
        except TypeError:
            # State written by a different shape of the dataclass. A miss
            # rather than a raise: the fix is another scan, and the panel has
            # to stay usable enough to press the button again.
            return None
    return None


# ------------------------------------------------------------------ export

def segment_export_name(source_path: str, segment: segmentation.Segment) -> str:
    """The filename one span is written under.

    Named for the span it holds rather than given a fresh id, so scanning the
    same recording twice does not fill the folder with copies of the same
    seconds. The source's full path is folded in as a short digest because
    the exports share one flat folder and the basename alone is not unique --
    two different session folders both holding `take.wav` would otherwise
    overwrite each other's segments silently.
    """
    stem = os.path.splitext(os.path.basename(source_path))[0]
    digest = hashlib.sha1(
        os.path.abspath(source_path).encode("utf-8", "replace")
    ).hexdigest()[:SOURCE_DIGEST_LENGTH]
    name = f"{stem}_{digest}_{segment.start_s:.2f}-{segment.end_s:.2f}.wav"
    return name.replace(" ", "_")


def _fade_edges(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Ramp the first and last few milliseconds in and out.

    The cut lands wherever the silence split put it, which is not a zero
    crossing, and a hard edge on a speaker reference is an audible click at
    both ends of every generation. Short enough not to eat a word onset: the
    split already placed the boundary inside a quiet stretch.
    """
    ramp = int(sample_rate * EDGE_FADE_SECONDS)
    if ramp < 1 or audio.size < ramp * 2:
        return audio
    faded = audio.copy()
    shape = np.linspace(0.0, 1.0, ramp, dtype=faded.dtype)
    faded[:ramp] *= shape
    faded[-ramp:] *= shape[::-1]
    return faded


def export_segment(
    source_path: str,
    segment: segmentation.Segment,
    export_root: Optional[str] = None,
) -> str:
    """Cut one span out to its own wav and return the path."""
    import soundfile

    root = export_root or default_export_root()
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, segment_export_name(source_path, segment))

    audio, sample_rate = librosa.load(
        source_path,
        sr=None,
        mono=True,
        offset=segment.start_s,
        duration=max(0.0, segment.end_s - segment.start_s),
    )
    # Written under a temp name and moved, because the preview player reads
    # this path as soon as the handler returns and a half-written wav plays
    # as a click. os.replace is atomic within a volume.
    partial = f"{destination}.partial"
    # `format` is explicit because soundfile infers it from the extension,
    # and the temp name's extension is ".partial", which it does not know.
    soundfile.write(partial, _fade_edges(audio, sample_rate), sample_rate,
                    format="WAV")
    os.replace(partial, destination)
    return destination


# ---------------------------------------------------------------- the scan

def _scan_outputs(table: Any, state: Any, choices: Any, bar: str, status: str):
    """The five things a scan yields, in the order the wiring binds them."""
    return (
        table if isinstance(table, dict) else gr.update(value=table),
        state,
        choices,
        gr.update(value=bar),
        gr.update(value=status, visible=bool(status)),
    )


def _scan_failed(message: str):
    return _scan_outputs(
        [], None,
        gr.update(choices=[], value=characters.NO_SELECTION),
        render_progress_bar(0.0, message, failed=True),
        message,
    )


def scan_summary(segments: Sequence[segmentation.Segment]) -> str:
    """What the scan found, including anything the table is not showing."""
    usable = [s for s in segments if s.is_long_enough and s.fits_the_reference_cap]
    parts = [
        f"{len(segments)} segment{'s' if len(segments) != 1 else ''} found, "
        f"{len(usable)} within the "
        f"{segmentation.ENGINE_REFERENCE_SECONDS:.0f}s reference cap."
    ]
    if len(segments) > MAX_TABLE_ROWS:
        parts.append(
            f"The table shows the first {MAX_TABLE_ROWS}; "
            "the dropdown below has all of them."
        )
    return " ".join(parts)


def scan_recording_ui(
    cleaned_path: Optional[str] = None,
    uploaded_path: Optional[str] = None,
    top_db: Any = segmentation.DEFAULT_SILENCE_TOP_DB,
    merge_gap_s: Any = segmentation.DEFAULT_MERGE_GAP_SECONDS,
    min_seconds: Any = segmentation.MIN_USEFUL_SECONDS,
    best_first: bool = True,
    progress: Optional[Any] = None,
):
    """Scan pressed. A generator, because the scan is slow enough to report.

    Yields the table, the state, the dropdown, the progress bar and a status
    line -- repeatedly while the scan runs, then once more when it finishes.

    Takes both the cleaned file from the section above and an upload, and an
    upload wins, the same way `cleanup_run_ui` treats its two sources.

    `progress` is optional and defaults to nothing rather than to
    `gr.Progress()`: the in-page bar is the reporting that matters here (see
    `render_progress_bar`, which exists because gradio's own overlay proved
    easy to miss), and a plain default is what lets this be tested.
    """
    path = uploaded_path or cleaned_path
    if not path:
        yield _scan_failed("Load a recording to scan first.")
        return
    if not os.path.isfile(path):
        yield _scan_failed(f"That recording is no longer there: {path}")
        return

    if media_utils.is_video_file(path):
        try:
            path = media_utils.ensure_audio_file(path)
        except ValueError as exc:
            yield _scan_failed(str(exc))
            return
        # The extracted wav must outlive this scan — segments are cut from it
        # when the user picks one — so it cannot live in a scratch dir. Keep
        # exactly one: each new extraction deletes its predecessor, bounding
        # the temp-space cost to a single file instead of one per video.
        previous = _LAST_EXTRACTED.get("path")
        if previous and previous != path:
            try:
                os.remove(previous)
            except OSError:
                pass  # still open in the player, or already gone
        _LAST_EXTRACTED["path"] = path

    yield _scan_outputs(
        [], None,
        gr.update(choices=[], value=characters.NO_SELECTION),
        render_progress_bar(0.0, "Finding where the speaker pauses..."),
        "",
    )

    events: "queue.Queue[Any]" = queue.Queue()
    outcome: Dict[str, Any] = {}
    finished = object()

    def report(done: int, total: int) -> None:
        events.put((done, total))

    def worker() -> None:
        try:
            segments, sample_rate = segmentation.load_and_segment(
                path, int(top_db), float(merge_gap_s), float(min_seconds), report
            )
            outcome["segments"], outcome["sample_rate"] = segments, sample_rate
        except Exception as exc:            # noqa: BLE001 - surfaced, never swallowed
            # A missing codec, an unreadable file, a truncated download. It
            # belongs on the status line, not in a console nobody is reading.
            outcome["error"] = f"Unexpected {type(exc).__name__}: {exc}"
        finally:
            events.put(finished)

    thread = threading.Thread(target=worker, daemon=True, name="segment-scan")
    thread.start()

    # Throttled to whole percent, because one event arrives per segment and a
    # long recording has hundreds of them -- every one of which would
    # otherwise be a round trip to the browser.
    last_percent = -1
    while True:
        event = events.get()
        if event is finished:
            break
        done, total = event
        if not total:
            continue
        percent = int(100 * done / total)
        if percent == last_percent:
            continue
        last_percent = percent
        message = f"Measuring segment {done} of {total}"
        if progress is not None:
            progress(done / total, desc=message)
        yield _scan_outputs(
            gr.update(), gr.update(), gr.update(),
            render_progress_bar(done / total, message),
            "",
        )
    thread.join(timeout=SCAN_THREAD_JOIN_SECONDS)

    if outcome.get("error"):
        yield _scan_failed(f"Could not scan that recording. {outcome['error']}")
        return

    segments = list(outcome.get("segments") or [])
    if not segments:
        yield _scan_failed(
            "No usable speech in that recording: nothing longer than "
            f"{float(min_seconds):.0f}s between pauses."
        )
        return

    ordered = segmentation.rank_for_reference(segments) if best_first else segments
    choices = segment_choices(ordered)
    summary = scan_summary(segments)
    yield _scan_outputs(
        segment_rows(ordered),
        scan_state(path, outcome.get("sample_rate") or 0, segments),
        gr.update(choices=choices, value=choices[0][1]),
        render_progress_bar(1.0, summary, done=True),
        summary,
    )


# --------------------------------------------------------------- the picks

def describe_segment(segment: segmentation.Segment) -> str:
    """The chosen segment in words, for the line under the player."""
    return (
        f"Segment #{segment.index + 1}, {_format_clock(segment.start_s)} to "
        f"{_format_clock(segment.end_s)} · "
        f"{_format_measurement(segment.duration_s, 's')} · "
        f"{_format_measurement(segment.lufs, ' LUFS')} · "
        f"pitch {_format_measurement(segment.pitch_hz_mean, ' Hz', 0)} "
        f"(range {_format_measurement(segment.pitch_hz_range, ' Hz', 0)}) · "
        f"{_format_percent(segment.voiced_fraction)} voiced · "
        f"{verdict_for(segment)}"
    )


def preview_segment_ui(
    state: Optional[Dict[str, Any]],
    index: Any,
    export_root: Optional[str] = None,
):
    """A segment was chosen: cut it out so the player has something to play."""
    segment = segment_from_state(state, index)
    if segment is None:
        return gr.update(value=None), gr.update(value="", visible=False)
    try:
        path = export_segment((state or {}).get("source") or "", segment, export_root)
    except Exception as exc:            # noqa: BLE001 - surfaced, never swallowed
        # Broad on purpose. A failed decode has no stable type here: missing
        # files give OSError, libsndfile gives a RuntimeError, the audioread
        # fallback gives NoBackendError and a truncated file gives EOFError,
        # both plain Exceptions. This runs unattended on every dropdown
        # change, so an uncaught one takes the page down.
        return (
            gr.update(value=None),
            gr.update(value=f"Could not cut that segment out: {exc}", visible=True),
        )
    return (
        gr.update(value=path),
        gr.update(value=describe_segment(segment), visible=True),
    )


def use_segment_as_reference_ui(
    state: Optional[Dict[str, Any]],
    index: Any,
    export_root: Optional[str] = None,
):
    """Use As Reference pressed: put this span in the reference slot."""
    segment = segment_from_state(state, index)
    if segment is None:
        return gr.update(), gr.update(
            value="Scan a recording and pick a segment first.", visible=True)
    try:
        path = export_segment((state or {}).get("source") or "", segment, export_root)
    except Exception as exc:            # noqa: BLE001 - surfaced, never swallowed
        return gr.update(), gr.update(
            value=f"Could not cut that segment out: {exc}", visible=True)

    note = ""
    if not segment.fits_the_reference_cap:
        note = (
            f" Only its first {segmentation.ENGINE_REFERENCE_SECONDS:.0f} "
            "seconds will be used."
        )
    return (
        gr.update(value=path),
        gr.update(
            value=f"Reference voice set from segment #{segment.index + 1}.{note}",
            visible=True,
        ),
    )


def describe_save_target(voice_name: Optional[str]) -> str:
    """Which voice the save button will write to.

    The voice is chosen on the Character Library panel, which is on another
    tab, so without this the user cannot tell what they are about to add a
    clip to without leaving the page and coming back.
    """
    name = (voice_name or "").strip()
    if not name:
        return "_No voice selected in the Character Library._"
    return f"Saving to: **{name}**"


def segment_metrics(segment: segmentation.Segment) -> Dict[str, Any]:
    """What follows this clip into the library.

    Everything the scan measured, plus where in the recording it came from,
    so a clip in a character can be traced back to the moment it was cut
    from -- which a copied wav on its own cannot tell you. `index` is dropped
    because it means nothing once the clip is out of the scan that numbered
    it.
    """
    metrics = segment.as_dict()
    metrics.pop("index", None)
    return metrics


def save_segment_to_voice_ui(
    mode: str,
    slug: str,
    state: Optional[Dict[str, Any]],
    index: Any,
    label: str = "",
    root: Optional[str] = None,
    export_root: Optional[str] = None,
):
    """Save Segment To Voice pressed: cut the span, then file it under a voice.

    Returns the same four outputs every mutating character control returns,
    so this binds to the panel exactly like the others.
    """
    segment = segment_from_state(state, index)
    if segment is None:
        return characters.refresh_panel(
            mode, slug, "Scan a recording and pick a segment first.", root)
    # Checked before the export, not after: cutting a wav to disk and then
    # refusing to file it leaves a working file nothing points at.
    if mode == store.MODE_RVC:
        return characters.refresh_panel(
            mode, slug, "An RVC voice holds a model, not clips.", root)
    if not slug:
        return characters.refresh_panel(mode, slug, "Select a voice first.", root)

    source = (state or {}).get("source") or ""
    try:
        path = export_segment(source, segment, export_root)
    except Exception as exc:            # noqa: BLE001 - surfaced, never swallowed
        return characters.refresh_panel(
            mode, slug, f"Could not cut that segment out: {exc}", root)

    spoken = label or f"{os.path.basename(source)} #{segment.index + 1}"
    return characters.add_reference_to_character_ui(
        mode, slug, path, spoken, root, metrics=segment_metrics(segment)
    )
