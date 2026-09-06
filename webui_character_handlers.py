"""What the character-library controls do.

Two dropdowns drive this. The first picks the kind of voice -- a one-shot
reference or an RVC model -- and the second lists the voices of that kind.
Everything else on the panel reacts to which voice is selected: its name,
its notes, what it holds, and the buttons that load it into the reference
slot or add the clip currently loaded there.

Every handler returns gradio update objects and reports trouble as status
text. Nothing here raises: a library on a disconnected drive or a character
someone deleted in Explorer has to leave the page usable rather than
stopping a handler with a traceback the user cannot act on.

The library root is a parameter defaulting to the module constant, so the
tests point at a temp directory instead of the real library. That is the
same reason character_store takes one.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

import character_store as store

CHARACTER_LIBRARY_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "characters"
)

# Roughly what each trainer wants before a run is worth starting. Used only to
# tell the user where they are, never to block anything.
RVC_TARGET_SECONDS = 600.0
SOVITS_TARGET_SECONDS = 60.0

# gradio raises "Value: is not in the list of choices" when a Dropdown is
# preprocessed with a value absent from its choices, and an empty string is
# absent from an empty list. None is the value it accepts for "nothing
# selected", so an empty library must use that and not "".
NO_SELECTION = None


def mode_choices() -> List[Tuple[str, str]]:
    """The first dropdown: what kind of voice."""
    return [(store.MODE_LABELS[mode], mode) for mode in store.MODES]


def character_choices(mode: str, root: Optional[str] = None) -> List[Tuple[str, str]]:
    """The second dropdown: every voice of the selected kind.

    Names are shown, slugs are the values, because a rename must not strand
    a selection and two voices can be renamed to look alike mid-session.
    """
    summaries = store.list_characters(root or CHARACTER_LIBRARY_ROOT, mode=mode)
    return [(_dropdown_label(summary), summary["slug"]) for summary in summaries]


def _dropdown_label(summary: Dict[str, Any]) -> str:
    if summary.get("unreadable"):
        return f"{summary['name']}  (unreadable)"
    if summary.get("mode") == store.MODE_ONESHOT:
        count = summary.get("clip_count", 0)
        return f"{summary['name']}  ({count} clip{'s' if count != 1 else ''})"
    return summary["name"]


def describe_character(mode: str, slug: str, root: Optional[str] = None) -> str:
    """The markdown under the dropdowns: what this voice actually holds."""
    if not slug:
        return "_No voice selected._"
    library = root or CHARACTER_LIBRARY_ROOT
    document = store.load_character(library, slug)
    if document is None:
        return "_That voice is no longer in the library._"

    if document.get("mode") == store.MODE_RVC:
        rvc = document.get("rvc") or {}
        if not rvc.get("model_path"):
            return "**RVC voice.** No model attached yet."
        return f"**RVC voice.** Model: `{rvc['model_path']}`"

    clips = (document.get("oneshot") or {}).get("clips", [])
    if not clips:
        return ("**No clips yet.** Load a reference clip in the panel above, "
                'then press **Save Loaded Voice As** to store it here.')

    seconds = store.total_clip_seconds(document)
    headline = (
        f"**{len(clips)} clip{'s' if len(clips) != 1 else ''}, "
        f"{seconds:.0f}s total.** {_readiness(seconds)}"
    )
    lines = [headline]
    default_id = (document.get("oneshot") or {}).get("default_clip_id")
    for clip in clips[:8]:
        marker = "**>**" if clip.get("id") == default_id else "  -"
        lines.append(f"{marker} {_clip_line(clip)}")
    if len(clips) > 8:
        lines.append(f"  _...and {len(clips) - 8} more._")
    return "\n\n".join(lines)


def _clip_line(clip: Dict[str, Any]) -> str:
    parts = [clip.get("label") or clip.get("source") or clip.get("id", "clip")]
    duration = clip.get("duration_s")
    if isinstance(duration, (int, float)):
        parts.append(f"{float(duration):.1f}s")
    lufs = clip.get("lufs")
    if isinstance(lufs, (int, float)):
        parts.append(f"{float(lufs):.1f} LUFS")
    return " · ".join(parts)


def _readiness(seconds: float) -> str:
    """Where this voice stands against what the trainers want.

    Deliberately informational. A short character is perfectly usable for
    zero-shot generation, which is what most of them are for.
    """
    if seconds >= RVC_TARGET_SECONDS:
        return "Enough for RVC training."
    if seconds >= SOVITS_TARGET_SECONDS:
        return (
            f"Enough for GPT-SoVITS; RVC wants about "
            f"{RVC_TARGET_SECONDS / 60:.0f} minutes."
        )
    return "Fine for one-shot use. Not enough to train from yet."


def on_mode_change(mode: str, root: Optional[str] = None):
    """First dropdown changed: refill the second and clear what was shown."""
    choices = character_choices(mode, root)
    first = choices[0][1] if choices else NO_SELECTION
    return (
        gr.update(choices=choices, value=first),
        gr.update(value=_name_of(first, root)),
        gr.update(value=describe_character(mode, first, root)),
        gr.update(value="", visible=False),
    )


def on_character_change(mode: str, slug: str, root: Optional[str] = None):
    """Second dropdown changed: show that voice's name and contents."""
    return (
        gr.update(value=_name_of(slug, root)),
        gr.update(value=describe_character(mode, slug, root)),
        gr.update(value="", visible=False),
    )


def _name_of(slug: str, root: Optional[str] = None) -> str:
    if not slug:
        return ""
    document = store.load_character(root or CHARACTER_LIBRARY_ROOT, slug)
    return (document or {}).get("name", "") if document else ""


def refresh_panel(mode: str, slug: str, message: str, root: Optional[str] = None):
    """The four outputs every mutating button returns.

    Public because it is the panel's contract, not an implementation detail:
    anything that adds a control to this panel -- including the segmentation
    handlers, which live in their own module -- has to return these same four
    in this same order.
    """
    choices = character_choices(mode, root)
    values = [value for _, value in choices]
    selected = slug if slug in values else (values[0] if values else NO_SELECTION)
    return (
        gr.update(choices=choices, value=selected),
        gr.update(value=_name_of(selected, root)),
        gr.update(value=describe_character(mode, selected, root)),
        gr.update(value=message, visible=bool(message)),
    )


# The name the rest of this module was written against. Kept so the eight
# call sites below read as they did, rather than churning them to prove a
# rename happened.
_refresh = refresh_panel


def create_character_ui(mode: str, name: str, root: Optional[str] = None):
    """New Voice pressed."""
    if not (name or "").strip():
        return _refresh(mode, NO_SELECTION, "Type a name for the new voice first.", root)
    try:
        slug = store.create_character(root or CHARACTER_LIBRARY_ROOT, name, mode)
    except store.CharacterStoreError as exc:
        return _refresh(mode, NO_SELECTION, str(exc), root)
    except OSError as exc:
        return _refresh(mode, NO_SELECTION, f"Could not create that voice: {exc}", root)
    return _refresh(mode, slug, f"Created {name.strip()!r}.", root)


def rename_character_ui(mode: str, slug: str, new_name: str, root: Optional[str] = None):
    """Rename pressed. The name box is the source of the new name."""
    if not slug:
        return _refresh(mode, slug, "Select a voice first.", root)
    if not (new_name or "").strip():
        return _refresh(mode, slug, "A voice needs a name.", root)
    try:
        new_slug = store.rename_character(root or CHARACTER_LIBRARY_ROOT, slug, new_name)
    except store.CharacterStoreError as exc:
        return _refresh(mode, slug, str(exc), root)
    except OSError as exc:
        return _refresh(mode, slug, f"Could not rename that voice: {exc}", root)
    return _refresh(mode, new_slug, f"Renamed to {new_name.strip()!r}.", root)


def delete_character_ui(mode: str, slug: str, confirmed: bool, root: Optional[str] = None):
    """Delete pressed. `confirmed` comes from the same confirm-signal pattern
    the cancel button uses, so a misclick cannot destroy a library."""
    if not slug:
        return _refresh(mode, slug, "Select a voice first.", root)
    if not confirmed:
        return _refresh(mode, slug, "Press Delete again to confirm.", root)
    name = _name_of(slug, root) or slug
    if store.delete_character(root or CHARACTER_LIBRARY_ROOT, slug):
        return _refresh(mode, NO_SELECTION, f"Deleted {name!r}.", root)
    return _refresh(mode, slug, f"Could not delete {name!r}; it may be in use.", root)


def on_lora_speak_change(enabled: bool, slug: str, strength,
                         root: Optional[str] = None):
    """The 'Speak with trained voice' controls (or the selection) changed.

    Points the next generation at the character's adapter through the
    selected_lora holder — the same no-new-gen_single-argument route the
    device choice takes. Called with enabled=False this actively CLEARS the
    holder, so a stale adapter can never haunt later generations.
    """
    from webui_runtime import selected_lora

    strength = float(strength if strength is not None else 1.0)
    if not enabled:
        selected_lora.set("", strength)
        return gr.update(value="Trained voice off. Generations use the "
                               "reference audio alone.", visible=True)
    if not slug:
        selected_lora.set("", strength)
        return gr.update(value="Select a voice first.", visible=True)

    document = store.load_character(root or CHARACTER_LIBRARY_ROOT, slug)
    block = (document or {}).get("lora") or {}
    adapter_path = block.get("adapter_path")
    if not adapter_path or not os.path.isfile(adapter_path):
        selected_lora.set("", strength)
        return gr.update(
            value="This voice has no trained TTS model yet — press "
                  "Train TTS LoRA first. Generations use the reference "
                  "audio alone.", visible=True)

    selected_lora.set(adapter_path, strength)
    return gr.update(
        value=f"Next generations speak through {os.path.basename(adapter_path)} "
              f"(strength {strength:.2f}).", visible=True)


def use_character_ui(mode: str, slug: str, root: Optional[str] = None):
    """Use This Voice pressed: load the default clip into the reference slot."""
    if mode == store.MODE_RVC:
        return (
            gr.update(),
            gr.update(
                value="RVC voices are not wired into generation yet.",
                visible=True,
            ),
        )
    if not slug:
        return gr.update(), gr.update(value="Select a voice first.", visible=True)

    path = store.resolve_clip_path(root or CHARACTER_LIBRARY_ROOT, slug)
    if not path:
        return (
            gr.update(),
            gr.update(
                value="That voice has no usable clip. Add one below.",
                visible=True,
            ),
        )
    name = _name_of(slug, root) or slug
    return (
        gr.update(value=path),
        gr.update(value=f"Reference voice set from {name!r}.", visible=True),
    )


def add_reference_to_character_ui(mode: str, slug: str, reference_path: Optional[str],
                                  label: str = "", root: Optional[str] = None,
                                  metrics: Optional[Dict[str, Any]] = None):
    """Add Current Reference pressed: file the loaded clip under this voice.

    `metrics` is for a caller that already measured the clip -- a segment cut
    out of a scan arrives with its loudness and pitch known. When it is not
    given, the wav header is read for a duration and nothing else, which is
    all this handler can afford between two clicks.
    """
    if mode == store.MODE_RVC:
        return _refresh(mode, slug, "An RVC voice holds a model, not clips.", root)
    if not slug:
        return _refresh(mode, slug, "Select a voice first.", root)
    if not reference_path:
        return _refresh(mode, slug, "Load a reference clip first.", root)

    if metrics is None:
        metrics = {}
        duration = _wav_duration_seconds(reference_path)
        if duration is not None:
            metrics["duration_s"] = duration
    try:
        store.add_clip(root or CHARACTER_LIBRARY_ROOT, slug, reference_path,
                       metrics, label=label or "")
    except store.CharacterStoreError as exc:
        return _refresh(mode, slug, str(exc), root)
    except OSError as exc:
        return _refresh(mode, slug, f"Could not add that clip: {exc}", root)
    return _refresh(mode, slug, "Added the current reference to this voice.", root)


def save_reference_as_new_voice_ui(mode: str, name: str, reference_path: Optional[str],
                                   root: Optional[str] = None):
    """Save Loaded Voice As pressed: create a voice AND file the clip in one go.

    The three-step version -- name it, press New, then press Add -- is the
    thing people could not find. This is the same work behind one button,
    which is how the panel is usually reached for: a reference is already
    loaded and it wants a name.
    """
    if mode == store.MODE_RVC:
        return _refresh(mode, NO_SELECTION, "An RVC voice holds a model, not clips.", root)
    if not reference_path:
        return _refresh(mode, NO_SELECTION, "Load a reference clip first, then save it.", root)
    if not (name or "").strip():
        return _refresh(mode, NO_SELECTION, "Type a name for this voice first.", root)

    library = root or CHARACTER_LIBRARY_ROOT
    try:
        slug = store.create_character(library, name, mode)
    except store.CharacterStoreError as exc:
        # Most often "already exists", and adding to the existing voice is
        # almost certainly what was meant -- but say which happened.
        existing = store.slugify(name)
        if store.load_character(library, existing) is None:
            return _refresh(mode, NO_SELECTION, str(exc), root)
        return add_reference_to_character_ui(mode, existing, reference_path, "", root)
    except OSError as exc:
        return _refresh(mode, NO_SELECTION, f"Could not create that voice: {exc}", root)

    result = add_reference_to_character_ui(mode, slug, reference_path, "", root)
    select, name_box, summary, _ = result
    return (
        select, name_box, summary,
        gr.update(value=f"Saved the loaded voice as {name.strip()!r}.", visible=True),
    )


def _wav_duration_seconds(path: str) -> Optional[float]:
    """Duration of a wav, or None for anything else.

    Deliberately only the stdlib `wave` module: this runs on the UI thread
    between two clicks, and pulling in librosa to read a header would cost
    seconds. Anything that is not a plain wav is left unmeasured rather than
    guessed. A clip that came from a scan skips this entirely -- it arrives
    with `metrics` already measured, which is the properly-measured path.
    """
    import wave

    try:
        with wave.open(path, "rb") as handle:
            rate = handle.getframerate()
            if not rate:
                return None
            return handle.getnframes() / float(rate)
    except (wave.Error, OSError, EOFError):
        return None


def initial_state(root: Optional[str] = None):
    """What the panel shows on first load, before anything is clicked."""
    mode = store.MODE_ONESHOT
    choices = character_choices(mode, root)
    first = choices[0][1] if choices else NO_SELECTION
    return mode, choices, first, _name_of(first, root), describe_character(mode, first, root)
