"""Tone description presets for the Audio Generation tab.

These are fed to the engine's emotion-description field (emo_text), NOT to the
text being synthesized. That distinction is the whole point: a phrase like
"in a low, deep voice" typed into the prompt gets read aloud, because the model
treats the prompt as words to speak. Put the same phrase here and it steers the
delivery instead, and is never spoken.

Selecting one of these switches the Emotion Control Method to "Use emotion text
description", because that is the only mode in which the field is read.

This module is standalone: it imports nothing from webui.py.
"""

from typing import Dict, List

TONE_PRESET_NONE = "None"

# Descriptions name concrete vocal behaviour -- pitch, pace, breath, volume,
# articulation -- rather than abstract moods, which the model steers on far
# more reliably.
TONE_DESCRIPTION_PRESETS: Dict[str, str] = {
    # Empty string: the engine falls back to its own default behaviour.
    "None": "",
    "Low and deep": "low pitch with heavy chest resonance and slow pacing",
    "Soft and breathy": "quiet volume with heavy breath intake and close proximity",
    "Bright and energetic": "high pitch with rapid tempo and sharp articulation",
    "Slow and deliberate": "unhurried pace with heavy consonant emphasis and measured pauses",
    "Warm and intimate": "warm mid-range pitch with soft consonants and relaxed delivery",
    "Cold and detached": "flat pitch with clipped consonants and minimal breath variation",
    "Urgent and tense": "tight throat tension with rapid pacing and pressed volume",
    "Gentle and reassuring": "steady mid-range pitch with smooth transitions and calm pacing",
    "Commanding": "projected volume with firm onset and clear downward inflection",
    "Playful": "light pitch with quick tempo and varied intonation patterns",
    "Whispered": "hushed volume with minimal vocal fold vibration and close mic",
}

TONE_PRESET_NAMES: List[str] = list(TONE_DESCRIPTION_PRESETS)

# Longest description the field is expected to carry. A very long description
# starts competing with the text itself for the model's attention.
TONE_DESCRIPTION_MAX_WORDS = 14


def tone_description(name: str) -> str:
    """Return the description for `name`, or "" when unknown."""
    return TONE_DESCRIPTION_PRESETS.get(name, "")


def _self_check() -> None:
    """Fail loudly at import if the table violates its own contract."""
    if TONE_PRESET_NONE not in TONE_DESCRIPTION_PRESETS:
        raise ValueError(f"{TONE_PRESET_NONE!r} preset is required")
    if TONE_DESCRIPTION_PRESETS[TONE_PRESET_NONE] != "":
        raise ValueError(f"{TONE_PRESET_NONE!r} must map to an empty description")

    seen = {}
    for name, description in TONE_DESCRIPTION_PRESETS.items():
        if name == TONE_PRESET_NONE:
            continue
        if not description:
            raise ValueError(f"preset {name!r} has an empty description")
        if '"' in description or "'" in description:
            raise ValueError(f"preset {name!r} contains a quote mark: {description!r}")
        if description.endswith("."):
            raise ValueError(f"preset {name!r} ends with a full stop: {description!r}")
        words = len(description.split())
        if words > TONE_DESCRIPTION_MAX_WORDS:
            raise ValueError(
                f"preset {name!r} is {words} words, over the "
                f"{TONE_DESCRIPTION_MAX_WORDS} word limit"
            )
        if description in seen:
            raise ValueError(
                f"presets {seen[description]!r} and {name!r} share a description"
            )
        seen[description] = name


_self_check()
