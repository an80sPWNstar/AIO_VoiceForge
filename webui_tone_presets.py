"""Tone and delivery presets for the Audio Generation tab.

These are fed to the engine's emotion-description field (emo_text), NOT to the
text being synthesized. That distinction is the whole point: a phrase like
"in a low, deep voice" typed into the prompt gets read aloud, because the model
treats the prompt as words to speak. Put the same phrase here and it steers the
delivery instead, and is never spoken.

Selecting one switches the Emotion Control Method to "Use emotion text
description", because that is the only mode in which the field is read.

WORDING follows the guidance in Inworld's TTS prompting notes
(https://docs.inworld.ai/tts/best-practices/prompting-for-tts-2): a
single-dimension instruction such as "sound sad" gives weak control, while
layering several dimensions in one directive gives far more. Each description
below therefore combines at least three of:

    emotion . speed . pitch . volume . articulation . vocal style . intonation

and is phrased as a DIRECTION to the speaker ("speak slowly in a low voice
with...") rather than as a clinical label ("low pitch, slow pacing"). Avoid
mixing opposing directions in one description -- whispered plus loud produces
unpredictable results.

That guidance was written for a different engine, whose syntax puts bracketed
tags inline in the text. Do NOT copy that syntax here: IndexTTS2 has no such
tag parser, so a bracketed tag in the prompt is simply read aloud. Only the
wording style transfers, which is what this module borrows.

This module is standalone: it imports nothing from webui.py.
"""

from typing import Dict, List

TONE_PRESET_NONE = "None"

TONE_DESCRIPTION_PRESETS: Dict[str, str] = {
    # Empty string: the engine falls back to its own default behaviour.
    "None": "",

    # --- delivery styles -------------------------------------------------
    "Low and deep": "speak slowly in a low deep voice with heavy chest resonance",
    "Soft and breathy": "speak quietly and slowly in a soft breathy voice with low close delivery",
    "Whispered": "whisper very quietly and slowly in a low breathy voice with close intimate delivery",
    "Slow and deliberate": "speak very slowly with deliberate pauses and heavy emphasis on each word",
    "Bright and energetic": "speak quickly and brightly with high pitch and sharp clear articulation",
    "Commanding": "speak loudly and firmly with falling pitch and forceful clear articulation",
    "Cold and detached": "speak flatly with even pitch clipped words and no emotional colour",

    # --- warm / intimate -------------------------------------------------
    # Four neighbouring moods, kept apart on concrete axes so they do not
    # collapse into one another: Sultry is languid, Sexy is poised and even,
    # Flirty is quick and rising, Kinky is slow and firmly controlled.
    "Warm and intimate": "speak softly and warmly in a low close voice with relaxed pacing",
    "Sultry": "speak slowly in a low warm voice with breathy relaxed unhurried delivery",
    "Sexy": "speak smoothly and confidently in a low warm voice with deliberate unhurried pacing",
    "Flirty": "say playfully and softly with a light teasing lilt and quick rising pitch",
    "Kinky": "speak slowly and firmly in a low teasing voice with deliberate playful emphasis",
    "Gentle and reassuring": "speak gently and calmly with steady pitch and smooth soothing pacing",

    # --- emotional range -------------------------------------------------
    # These overlap what the emotion-vector sliders try to do. The description
    # field steers delivery far more audibly than the vector does on this
    # model, so the same ground is covered here as well.
    "Happy": "say happily with a bright rising pitch and quick lively pacing",
    "Excited": "say excitedly with high pitch fast pace and energetic emphasis",
    "Sad": "say sadly with a low quiet voice and slow heavy pacing",
    "Angry": "say angrily with forceful loud delivery and hard clipped consonants",
    "Fearful": "say fearfully with a tight tense voice quick breaths and rising pitch",
    "Urgent": "say urgently with a tense pressed voice and fast clipped pacing",
    "Playful": "say playfully with a light teasing tone quick pace and varied pitch",
}

TONE_PRESET_NAMES: List[str] = list(TONE_DESCRIPTION_PRESETS)

# Long enough for a genuinely multi-dimensional direction, short enough that
# the description does not start competing with the text for attention.
TONE_DESCRIPTION_MAX_WORDS = 18

# A description naming fewer dimensions than this is the weak single-axis
# phrasing the guidance warns about.
TONE_MIN_WORDS = 6

# Words that mark each steerable dimension. Used by the self-check to prove a
# description really is multi-dimensional rather than a relabelled single axis.
TONE_DIMENSION_MARKERS = {
    "speed": ("slow", "slowly", "quick", "quickly", "fast", "unhurried", "pace", "pacing", "pauses"),
    "pitch": ("low", "high", "deep", "rising", "falling", "pitch", "resonance"),
    "volume": ("quiet", "quietly", "loud", "loudly", "soft", "softly", "whisper", "hushed", "forceful", "pressed"),
    "articulation": ("clear", "clipped", "sharp", "articulation", "emphasis", "consonants", "deliberate", "even"),
    "manner": ("breathy", "warm", "warmly", "gently", "calmly", "tense", "teasing", "close",
               "intimate", "happily", "sadly", "angrily", "fearfully", "urgently", "playfully",
               "excitedly", "brightly", "firmly", "flatly", "smooth", "soothing", "energetic",
               "lively", "heavy", "relaxed", "steady", "colour"),
}

# Pairs that contradict each other. The guidance is explicit that opposing
# directions in one instruction produce unpredictable output.
TONE_CONTRADICTIONS = (
    ("whisper", "loud"),
    ("quietly", "loudly"),
    ("slowly", "quickly"),
)


def tone_description(name: str) -> str:
    """Return the description for `name`, or "" when unknown."""
    return TONE_DESCRIPTION_PRESETS.get(name, "")


def dimensions_covered(description: str) -> List[str]:
    """Return which steerable dimensions a description actually names."""
    words = set(description.lower().replace(",", " ").split())
    return [
        dimension
        for dimension, markers in TONE_DIMENSION_MARKERS.items()
        if words & set(markers)
    ]


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
        if "[" in description or "]" in description:
            raise ValueError(
                f"preset {name!r} contains a bracketed tag. This engine has no tag "
                "parser and would read it aloud."
            )

        words = description.split()
        if not TONE_MIN_WORDS <= len(words) <= TONE_DESCRIPTION_MAX_WORDS:
            raise ValueError(
                f"preset {name!r} is {len(words)} words, outside the "
                f"{TONE_MIN_WORDS}-{TONE_DESCRIPTION_MAX_WORDS} range"
            )

        covered = dimensions_covered(description)
        if len(covered) < 3:
            raise ValueError(
                f"preset {name!r} names only {len(covered)} dimension(s) ({covered}); "
                "single-axis directions steer the model weakly"
            )

        lowered = description.lower()
        for first, second in TONE_CONTRADICTIONS:
            if first in lowered and second in lowered:
                raise ValueError(
                    f"preset {name!r} mixes opposing directions "
                    f"({first!r} and {second!r})"
                )

        if description in seen:
            raise ValueError(
                f"presets {seen[description]!r} and {name!r} share a description"
            )
        seen[description] = name


_self_check()
