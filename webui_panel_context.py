"""The handles a panel needs from outside its own tab.

Panels used to be built inline in webui.py, so a "send this to the reference
slot" button could simply name `prompt_audio` — a local variable a few hundred
lines up. Once a panel moves into its own module that name is gone, and the
choice is between reaching back into webui.py (a circular import, and the
coupling the split exists to remove) or being handed what it needs.

This is the second option, written down. webui.py builds the generation tab
first, packs the few components other panels touch into a `PanelContext`, and
passes it to each panel builder. A panel therefore declares its outside world
in its signature instead of inheriting the whole module scope, and a test can
build one from stand-ins without a Blocks context.

Components are typed `Any` on purpose: gradio's classes are only meaningful
inside a Blocks context, and the tests pass simple stand-ins.

What deliberately does NOT live here is the ingestion pipeline's own chaining.
Fetch, clean and pick-a-clip each read the previous stage's result path, and
that path is passed to the next builder as an ordinary `upstream_path`
argument instead. It is an edge between two adjacent stages rather than shared
app state, and keeping it in the signature is what makes the chain readable at
the call site — a context object would hide the ordering that matters most.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReferenceTargets:
    """Where a panel's "use this audio" hand-off lands.

    Both the media-fetch and cleanup panels end in a button that sends their
    result to the generation tab's reference slot; these are the two components
    that hand-off writes to.
    """

    audio: Any
    status: Any


@dataclass(frozen=True)
class CharacterTargets:
    """The Character Library controls a panel refreshes after saving a clip.

    Saving a clip from the cleanup or segmentation panel changes the library,
    so the library's own controls have to be refreshed from the other tab.
    `name` and `select` are also event SOURCES — the save-target captions on
    those panels follow whichever voice the library currently has selected.
    """

    mode: Any
    select: Any
    name: Any
    summary: Any


@dataclass(frozen=True)
class PanelContext:
    """Everything a panel on the ingestion tab needs from the generation tab."""

    reference: ReferenceTargets
    character: CharacterTargets
    focus_generation_tab_js: str
