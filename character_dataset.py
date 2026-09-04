"""Export a character's clips as a training dataset.

The character library holds many clips per voice on purpose: a zero-shot
generation needs one 15-second reference, but training wants minutes of
material, and both come out of the same collection (character_store's own
docstring). This module is the bridge: it lays a character's clips out in
the flat folder-of-wavs shape every RVC-family trainer preprocesses from,
with a manifest recording what came from where.

Deliberately dumb about audio: clips are stored as wavs and trainers run
their own resampling and slicing during preprocessing, so this copies bytes
and verifies each file is a readable wav rather than re-encoding anything.
Re-encoding here would mean a second audio stack to keep correct for zero
gain.

Filesystem only, no gradio, root and clock injectable -- same testability
rules as character_store.
"""

import json
import shutil
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import character_store

# What the trainers want to see, used for the readiness warning only. RVC
# quality reports commonly plateau around ten minutes; GPT-SoVITS asks for
# less. An export below the floor still exports -- a mechanics test on thin
# data is legitimate -- it just says so out loud in the manifest.
RVC_RECOMMENDED_SECONDS = 600.0
DATASET_MANIFEST_NAME = "dataset.json"


class DatasetExportError(RuntimeError):
    """The export cannot produce anything usable, and the caller should know."""


def _wav_duration_seconds(path: Path) -> Optional[float]:
    """Actual audio length read from the file, None for an unreadable one."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        return frames / float(rate) if rate else None
    except (OSError, wave.Error, EOFError):
        return None


def export_dataset(root: str, slug: str, dest_dir: str,
                   now: Optional[float] = None) -> Dict[str, Any]:
    """Copy every readable clip of one character into dest_dir, and write a
    manifest beside them. Returns the manifest.

    Unreadable clips are skipped and named in the manifest's warnings rather
    than failing the export: one corrupt download must not hold the other
    nine minutes hostage. Zero exportable clips IS a failure -- an empty
    dataset directory that looks ready to train from would waste a GPU run.
    """
    document = character_store.load_character(root, slug)
    if document is None:
        raise DatasetExportError(f"No character named {slug!r} in the library.")

    clips = (document.get("oneshot") or {}).get("clips", [])
    if not clips:
        raise DatasetExportError(f"Character {slug!r} has no clips to export.")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    exported = []
    warnings = []
    total_seconds = 0.0
    for clip in clips:
        clip_id = clip.get("id") or "unknown"
        source = character_store.character_dir(root, slug) / (clip.get("file") or "")
        if not source.is_file():
            warnings.append(f"clip {clip_id}: file missing, skipped")
            continue
        duration = _wav_duration_seconds(source)
        if duration is None:
            warnings.append(f"clip {clip_id}: not a readable wav, skipped")
            continue
        target = dest / f"{slug}_{clip_id}.wav"
        shutil.copyfile(source, target)
        total_seconds += duration
        exported.append({
            "id": clip_id,
            "file": target.name,
            "duration_s": round(duration, 3),
            "label": clip.get("label") or "",
            "source": clip.get("source") or "",
        })

    if not exported:
        raise DatasetExportError(
            f"Character {slug!r}: none of its {len(clips)} clips could be "
            f"exported ({'; '.join(warnings)})."
        )

    if total_seconds < RVC_RECOMMENDED_SECONDS:
        warnings.append(
            f"only {total_seconds:.0f}s of audio; RVC quality wants around "
            f"{RVC_RECOMMENDED_SECONDS:.0f}s -- fine for a mechanics run, "
            f"thin for a real voice"
        )

    manifest = {
        "character": document.get("name"),
        "slug": slug,
        "exported_at": character_store._now(now),
        "clips": exported,
        "total_seconds": round(total_seconds, 3),
        "warnings": warnings,
    }
    (dest / DATASET_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
