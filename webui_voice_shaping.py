"""Speed and pitch shaping for generated audio.

IndexTTS2 has no speed or pitch parameter -- neither appears anywhere in
infer_v2.py -- so both are applied afterwards with ffmpeg's rubberband filter,
which changes tempo and pitch independently. Chaining asetrate with atempo
would also work but shifts formants, giving the chipmunk / slowed-tape sound;
rubberband keeps the voice recognisably the same person.

The generic ffmpeg helpers live in webui_media_fetch rather than being copied
here. This module imports nothing from webui.py.
"""

import os
from typing import List, Optional

from webui_media_fetch import (
    FFMPEG_BIN,
    MediaFetchError,
    ffmpeg_available,
    run_ffmpeg,
    sanitize_filename,
    unique_path,
)

# Playback speed. 1.0 is unchanged. rubberband handles this range cleanly.
SPEED_MIN = 0.5
SPEED_MAX = 2.0
SPEED_STEP = 0.05
SPEED_DEFAULT = 1.0

# Pitch in semitones. 12 semitones is one octave.
PITCH_MIN_SEMITONES = -12.0
PITCH_MAX_SEMITONES = 12.0
PITCH_STEP_SEMITONES = 0.5
PITCH_DEFAULT_SEMITONES = 0.0

SEMITONES_PER_OCTAVE = 12.0

SHAPED_SUBDIR = "shaped"
SHAPED_SUFFIX = "shaped"

# Below this, a speed or pitch request is treated as "no change requested".
SHAPING_EPSILON = 1e-3


class VoiceShapingError(MediaFetchError):
    """Raised when shaping fails. Subclasses MediaFetchError so callers that
    already handle one error type keep working."""


def semitones_to_ratio(semitones: float) -> float:
    """Convert a semitone offset to the frequency ratio rubberband expects."""
    return 2.0 ** (float(semitones) / SEMITONES_PER_OCTAVE)


def is_noop(speed: float, semitones: float) -> bool:
    """True when neither control is meaningfully away from its default."""
    return (
        abs(float(speed) - SPEED_DEFAULT) < SHAPING_EPSILON
        and abs(float(semitones) - PITCH_DEFAULT_SEMITONES) < SHAPING_EPSILON
    )


def build_shaping_command(
    source_path: str,
    destination_path: str,
    speed: float,
    semitones: float,
) -> List[str]:
    """Build the ffmpeg argv that applies speed and pitch to `source_path`.

    tempo is the speed multiplier directly; pitch is a frequency ratio, so a
    semitone offset has to be converted first.
    """
    ratio = semitones_to_ratio(semitones)
    return [
        FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-filter:a",
        f"rubberband=tempo={float(speed):.4f}:pitch={ratio:.6f}",
        destination_path,
    ]


def shape_audio(
    source_path: str,
    speed: float = SPEED_DEFAULT,
    semitones: float = PITCH_DEFAULT_SEMITONES,
    output_dir: Optional[str] = None,
) -> str:
    """Apply speed and pitch to `source_path`, returning the new file path.

    Returns `source_path` unchanged when neither control is off its default,
    so a no-op costs nothing and never rewrites a good file. Raises
    VoiceShapingError on any failure.
    """
    if not source_path:
        raise VoiceShapingError("Generate some audio before shaping it.")
    if not os.path.exists(source_path):
        raise VoiceShapingError(f"Audio file no longer exists: {source_path}")
    if is_noop(speed, semitones):
        return source_path
    if not ffmpeg_available():
        raise VoiceShapingError(
            "ffmpeg was not found on PATH, so speed and pitch cannot be applied."
        )

    speed = min(max(float(speed), SPEED_MIN), SPEED_MAX)
    semitones = min(max(float(semitones), PITCH_MIN_SEMITONES), PITCH_MAX_SEMITONES)

    target_dir = output_dir or os.path.join(os.path.dirname(source_path), SHAPED_SUBDIR)
    os.makedirs(target_dir, exist_ok=True)

    stem = sanitize_filename(os.path.splitext(os.path.basename(source_path))[0])
    extension = os.path.splitext(source_path)[1].lstrip(".") or "wav"
    destination = unique_path(target_dir, f"{stem}_{SHAPED_SUFFIX}", extension)

    run_ffmpeg(build_shaping_command(source_path, destination, speed, semitones))

    if not os.path.exists(destination):
        raise VoiceShapingError(
            f"ffmpeg reported success but produced no file: {destination}"
        )
    return destination


def describe_shaping(speed: float, semitones: float) -> str:
    """Short human readable summary of what shaping will do."""
    if is_noop(speed, semitones):
        return "No change (speed 1.00x, pitch 0 semitones)"
    parts = []
    if abs(float(speed) - SPEED_DEFAULT) >= SHAPING_EPSILON:
        parts.append(f"speed {float(speed):.2f}x")
    if abs(float(semitones) - PITCH_DEFAULT_SEMITONES) >= SHAPING_EPSILON:
        parts.append(f"pitch {float(semitones):+.1f} semitones")
    return ", ".join(parts)
