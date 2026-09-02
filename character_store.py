"""Named voices, and the clips or model files that make them up.

A character is one folder: a `character.json` describing it, and a `clips/`
directory holding the audio that belongs to it. Two kinds exist and the
JSON says which:

  one-shot  a set of reference clips, each with the measurements that make
            it findable later (how loud, how wide the pitch, how fast, who
            it matched, what was said). One is marked default. This is what
            IndexTTS-2.5 and every other zero-shot engine consume, and the
            clip file itself is the portable artefact.

  rvc       a trained model, so a .pth and its .index. Nothing is stored
            here but the paths and the settings that go with them.

Deliberately holding MANY clips per character rather than one. A zero-shot
generation needs a single 15-second clip, but training an RVC or GPT-SoVITS
checkpoint wants ten minutes of clean audio paired with transcripts, and
both come out of the same collection.

Filesystem and plain dicts only -- no gradio, no torch, no audio libraries.
The root directory is a parameter rather than a module constant, so tests
run against a temp directory instead of the user's real library.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

CHARACTER_FORMAT = "voiceforge_character"
CHARACTER_VERSION = "1.0"

MODE_ONESHOT = "oneshot"
MODE_RVC = "rvc"
MODES = (MODE_ONESHOT, MODE_RVC)

MODE_LABELS = {
    MODE_ONESHOT: "One-shot reference",
    MODE_RVC: "RVC model",
}

CHARACTER_FILE = "character.json"
CLIPS_DIRNAME = "clips"

# A slug has to survive being a directory name on Windows and a dropdown
# value in gradio, so it is deliberately narrower than a filename needs.
_SLUG_STRIP = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_SLUG_LENGTH = 64


class CharacterStoreError(RuntimeError):
    """Raised for a request the store cannot honour.

    Distinct from an OSError: it means the caller asked for something
    inconsistent (a duplicate name, an unknown mode), not that the disk
    failed.
    """


def slugify(name: str) -> str:
    """Turn a display name into a directory-safe identifier."""
    slug = _SLUG_STRIP.sub("-", (name or "").strip()).strip("-._")
    slug = slug[:_MAX_SLUG_LENGTH]
    return slug.lower() or "unnamed"


def character_dir(root: str, slug: str) -> Path:
    return Path(root) / slug


def clips_dir(root: str, slug: str) -> Path:
    return character_dir(root, slug) / CLIPS_DIRNAME


def _character_file(root: str, slug: str) -> Path:
    return character_dir(root, slug) / CHARACTER_FILE


def _now(now: Optional[float] = None) -> str:
    """ISO-8601 timestamp. `now` is injectable so tests are not clock-bound."""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now if now is not None else time.time()))


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write via a temp file and rename.

    A half-written character.json is unreadable forever, and the read path
    would then have to guess whether that meant "new" or "corrupt". Renaming
    means a reader sees either the old file or the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def blank_character(name: str, mode: str, now: Optional[float] = None) -> Dict[str, Any]:
    """The document a brand-new character starts from."""
    if mode not in MODES:
        raise CharacterStoreError(f"Unknown voice type {mode!r}; expected one of {', '.join(MODES)}")
    stamp = _now(now)
    return {
        "_meta": {
            "format": CHARACTER_FORMAT,
            "version": CHARACTER_VERSION,
            "created_at": stamp,
            "updated_at": stamp,
        },
        "name": (name or "").strip() or "Unnamed",
        "mode": mode,
        "notes": "",
        "oneshot": {"clips": [], "default_clip_id": None},
        "rvc": {"model_path": None, "index_path": None, "transpose": 0},
    }


def list_characters(root: str, mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every character on disk, optionally filtered to one voice type.

    Returns summaries, not whole documents -- the dropdown needs a name and
    a slug, not the clip table. Unreadable folders are skipped and reported
    in the summary rather than raising, so one bad character does not empty
    the list.
    """
    root_path = Path(root)
    try:
        if not root_path.is_dir():
            return []
        entries = sorted(root_path.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        # Both the stat and the listing can fail, and on a network or removable
        # path they routinely do. An empty dropdown is recoverable; a traceback
        # out of a UI handler is not.
        print(f"Character library: cannot list {root_path} ({exc}).", flush=True)
        return []

    summaries: List[Dict[str, Any]] = []
    for entry in entries:
        try:
            if not entry.is_dir():
                continue
        except OSError as exc:
            print(f"Character library: cannot stat {entry} ({exc}); skipping it.", flush=True)
            continue
        document = _read_character_file(entry / CHARACTER_FILE)
        if document is None:
            summaries.append({
                "slug": entry.name,
                "name": entry.name,
                "mode": None,
                "clip_count": 0,
                "unreadable": True,
            })
            continue
        if mode is not None and document.get("mode") != mode:
            continue
        summaries.append({
            "slug": entry.name,
            "name": document.get("name") or entry.name,
            "mode": document.get("mode"),
            "clip_count": len(document.get("oneshot", {}).get("clips", [])),
            "unreadable": False,
        })
    return summaries


def _read_character_file(path: Path) -> Optional[Dict[str, Any]]:
    """Parse one character.json, or None when it is missing or unreadable.

    Narrow on purpose: a missing file is an ordinary outcome (the folder is
    not a character), a malformed one is worth knowing about, and anything
    else is a real fault and is allowed to propagate.
    """
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Character library: {path} is not readable JSON ({exc}); skipping it.", flush=True)
        return None
    except OSError as exc:
        # Denied permission, a broken symlink, a file that vanished between the
        # is_file() check and the open. The docstring promises one bad folder
        # does not empty the library, so it has to survive these too.
        print(f"Character library: cannot read {path} ({exc}); skipping it.", flush=True)
        return None


def load_character(root: str, slug: str) -> Optional[Dict[str, Any]]:
    """The whole document for one character, or None if there is no such folder."""
    return _read_character_file(_character_file(root, slug))


def save_character(root: str, slug: str, document: Dict[str, Any],
                   now: Optional[float] = None) -> Dict[str, Any]:
    """Persist a document, stamping updated_at. Returns what was written."""
    document = dict(document)
    meta = dict(document.get("_meta") or {})
    meta.setdefault("format", CHARACTER_FORMAT)
    meta.setdefault("version", CHARACTER_VERSION)
    meta.setdefault("created_at", _now(now))
    meta["updated_at"] = _now(now)
    document["_meta"] = meta
    _write_json_atomic(_character_file(root, slug), document)
    return document


def create_character(root: str, name: str, mode: str,
                     now: Optional[float] = None) -> str:
    """Make a new character folder. Returns its slug.

    Refuses to overwrite an existing one -- two voices sharing a slug would
    silently merge their clip folders.
    """
    slug = slugify(name)
    if _character_file(root, slug).exists():
        raise CharacterStoreError(
            f"A voice called {name!r} already exists (folder {slug!r}). Rename one of them."
        )
    clips_dir(root, slug).mkdir(parents=True, exist_ok=True)
    try:
        save_character(root, slug, blank_character(name, mode, now), now)
    except OSError:
        # Otherwise a failed save leaves a folder with no character.json, which
        # then shows up in the dropdown forever as an unreadable voice.
        shutil.rmtree(character_dir(root, slug), ignore_errors=True)
        raise
    return slug


def rename_character(root: str, slug: str, new_name: str,
                     now: Optional[float] = None) -> str:
    """Change a character's display name, and its folder with it.

    Returns the slug afterwards, which is unchanged when the new name
    slugifies the same way.
    """
    document = load_character(root, slug)
    if document is None:
        raise CharacterStoreError(f"No voice named {slug!r} to rename.")

    document["name"] = (new_name or "").strip() or document.get("name") or slug
    new_slug = slugify(document["name"])
    if new_slug != slug:
        if _character_file(root, new_slug).exists():
            raise CharacterStoreError(
                f"A voice called {new_name!r} already exists (folder {new_slug!r})."
            )
        os.replace(character_dir(root, slug), character_dir(root, new_slug))
    save_character(root, new_slug, document, now)
    return new_slug


def delete_character(root: str, slug: str) -> bool:
    """Remove a character and everything under it. True if one was there."""
    target = character_dir(root, slug)
    if not target.is_dir():
        return False
    try:
        shutil.rmtree(target)
    except OSError as exc:
        # Windows keeps a handle on an audio file that is still open in a
        # player or in the engine, and rmtree raises rather than skipping it.
        # Reporting False beats a traceback the user cannot act on.
        print(f"Character library: could not delete {target} ({exc}).", flush=True)
        return False
    return True


def add_clip(root: str, slug: str, source_path: str,
             metrics: Optional[Dict[str, Any]] = None,
             label: str = "", make_default: bool = False,
             now: Optional[float] = None) -> Dict[str, Any]:
    """Copy an audio file into a character and record what is known about it.

    `metrics` is whatever the analysis produced -- duration, loudness, pitch,
    speaker similarity, emotion, transcript. Nothing here computes or
    validates them; the store only has to keep them next to the audio.
    """
    document = load_character(root, slug)
    if document is None:
        raise CharacterStoreError(f"No voice named {slug!r} to add a clip to.")
    if not os.path.isfile(source_path):
        raise CharacterStoreError(f"No such audio file: {source_path}")

    clip_id = uuid.uuid4().hex[:12]
    extension = os.path.splitext(source_path)[1].lower() or ".wav"
    destination = clips_dir(root, slug) / f"{clip_id}{extension}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)

    clip: Dict[str, Any] = {
        "id": clip_id,
        "file": f"{CLIPS_DIRNAME}/{destination.name}",
        "label": label or "",
        "added_at": _now(now),
        "source": os.path.basename(source_path),
    }
    clip.update(metrics or {})

    oneshot = document.setdefault("oneshot", {"clips": [], "default_clip_id": None})
    oneshot.setdefault("clips", []).append(clip)
    if make_default or not oneshot.get("default_clip_id"):
        oneshot["default_clip_id"] = clip_id
    try:
        save_character(root, slug, document, now)
    except OSError:
        # The audio is already copied in. If the document describing it cannot
        # be written, that file is unreferenced for good, so take it back out
        # rather than growing the clips folder with audio nothing points at.
        destination.unlink(missing_ok=True)
        raise
    return clip


def remove_clip(root: str, slug: str, clip_id: str, now: Optional[float] = None) -> bool:
    """Delete one clip and its file. True if it was there.

    If it was the default, the next remaining clip becomes default rather
    than leaving the character pointing at something that is gone.
    """
    document = load_character(root, slug)
    if document is None:
        return False
    oneshot = document.setdefault("oneshot", {"clips": [], "default_clip_id": None})
    clips = oneshot.setdefault("clips", [])
    remaining = [clip for clip in clips if clip.get("id") != clip_id]
    if len(remaining) == len(clips):
        return False

    for clip in clips:
        if clip.get("id") == clip_id:
            path = character_dir(root, slug) / clip.get("file", "")
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                # The document still drops the clip: a file we could not delete
                # is a tidiness problem, but a clip the user removed reappearing
                # in the library is a correctness one.
                print(f"Character library: could not delete {path} ({exc}).", flush=True)

    oneshot["clips"] = remaining
    if oneshot.get("default_clip_id") == clip_id:
        oneshot["default_clip_id"] = remaining[0]["id"] if remaining else None
    save_character(root, slug, document, now)
    return True


def set_default_clip(root: str, slug: str, clip_id: str, now: Optional[float] = None) -> bool:
    """Mark which clip a plain 'use this voice' should reach for."""
    document = load_character(root, slug)
    if document is None:
        return False
    oneshot = document.setdefault("oneshot", {"clips": [], "default_clip_id": None})
    if not any(clip.get("id") == clip_id for clip in oneshot.get("clips", [])):
        return False
    oneshot["default_clip_id"] = clip_id
    save_character(root, slug, document, now)
    return True


def resolve_clip_path(root: str, slug: str, clip_id: Optional[str] = None) -> Optional[str]:
    """Absolute path to a clip, or to the character's default when no id given.

    Returns None rather than raising when the character, the clip or the file
    is missing -- the caller is usually a UI handler that has to say something
    useful instead of crashing.
    """
    document = load_character(root, slug)
    if document is None:
        return None
    oneshot = document.get("oneshot") or {}
    wanted = clip_id or oneshot.get("default_clip_id")
    if not wanted:
        return None
    for clip in oneshot.get("clips", []):
        if clip.get("id") == wanted:
            path = character_dir(root, slug) / clip.get("file", "")
            return str(path.resolve()) if path.is_file() else None
    return None


def total_clip_seconds(document: Dict[str, Any]) -> float:
    """How much audio a character holds.

    The number that decides whether a character is ready to train from:
    RVC wants roughly ten minutes, GPT-SoVITS rather less. Clips with no
    measured duration count as zero rather than guessing.
    """
    clips = (document.get("oneshot") or {}).get("clips", [])
    total = 0.0
    for clip in clips:
        try:
            total += float(clip.get("duration_s") or 0.0)
        except (TypeError, ValueError):
            # A hand-edited document, or a clip stored before the analysis ran.
            # One unparseable number must not take the whole library readout
            # down, and guessing a duration would misreport training readiness.
            print(
                f"Character library: clip {clip.get('id')} has an unusable "
                f"duration {clip.get('duration_s')!r}; counting it as zero.",
                flush=True,
            )
    return total
