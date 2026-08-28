"""Emotion presets for the Audio Generation tab.

Order of every tuple is fixed and matches the vec1..vec8 sliders:
    (joy, anger, sadness, fear, disgust, depression, surprise, calm)

IMPORTANT: these are RAW slider values, and raw is not what the model receives.
webui.normalize_emo_vector multiplies each dimension by a fixed bias and only
then caps the total, so the useful strength of a preset is its POST-BIAS sum,
not its raw sum. Calm is cut to 0.5625 and surprise to 0.6875, so calm-heavy
presets need much larger raw numbers to land with the same force as an angry
or sad one.

Each preset is therefore tuned so its post-bias total approaches
EMOTION_POST_BIAS_CAP. Two cannot reach it at all -- a pure Calm or pure
Surprise mix runs out of raw range at 1.0 first -- which is a property of the
engine's bias table, not a mistake here.

This module is standalone: it imports nothing from webui.py.
"""

from typing import Dict, List, Tuple

EmotionVector = Tuple[float, float, float, float, float, float, float, float]

# Mirrors the default emo_bias list in webui.normalize_emo_vector. If that
# table ever changes upstream, the self-check below starts failing loudly
# rather than quietly producing weak presets.
EMOTION_BIAS: EmotionVector = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)

# Matches normalize_emo_vector(max_emotion_sum=0.8). Anything above this is
# rescaled down, so it is the ceiling worth aiming at.
EMOTION_POST_BIAS_CAP = 0.8

# Presets whose direction cannot reach the cap because their dominant
# dimension is biased down too hard to get there before raw 1.0.
EMOTION_CAPPED_BY_BIAS = ("Calm", "Surprised")

EMOTION_VECTOR_LENGTH = 8
EMOTION_PRESET_NEUTRAL = "Neutral"

EMOTION_DIMENSIONS: Tuple[str, ...] = (
    "joy", "anger", "sadness", "fear", "disgust", "depression", "surprise", "calm",
)

EMOTION_PRESETS: Dict[str, EmotionVector] = {
    # No emotional colouring: what the engine treats as a clean read.
    "Neutral": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Warm, ordinary happiness.
    "Happy": (0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    # High-energy delight, with a lift of surprise.
    "Excited": (0.77, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11, 0.0),
    # Gentle quiet sadness.
    "Sad": (0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Heavier and more resigned than Sad, carried by depression.
    "Melancholy": (0.0, 0.0, 0.23, 0.0, 0.0, 0.6, 0.0, 0.0),
    # Hot, direct anger.
    "Angry": (0.0, 0.91, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    # Contained irritation: anger soured with a little disgust.
    "Annoyed": (0.0, 0.67, 0.0, 0.0, 0.22, 0.0, 0.0, 0.0),
    # Frightened and tense.
    "Fearful": (0.0, 0.0, 0.11, 0.69, 0.0, 0.0, 0.0, 0.0),
    # Startled. Surprise is biased to 0.6875, so this tops out below the cap.
    "Surprised": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    # Revulsion, carried by disgust.
    "Disgusted": (0.0, 0.0, 0.0, 0.0, 0.85, 0.0, 0.0, 0.0),
    # Settled and serene. Calm is biased to 0.5625, the hardest cut of the
    # eight, so even at raw 1.0 this is the weakest preset available.
    "Calm": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    # Soothing and gentle: calm carried, with real warmth over it.
    "Tender": (0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85),
    # Hushed and close. Mostly calm, with a touch of sadness to soften and
    # slow the delivery rather than to sound unhappy.
    "Intimate": (0.15, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.9),
    # Warm and unhurried, more forward than Tender: confident low delivery.
    "Sultry": (0.55, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49),
}

EMOTION_PRESET_NAMES: List[str] = list(EMOTION_PRESETS)


def preset_vector(name: str) -> EmotionVector:
    """Return the raw vector for `name`, falling back to Neutral if unknown."""
    return EMOTION_PRESETS.get(name, EMOTION_PRESETS[EMOTION_PRESET_NEUTRAL])


def post_bias_strength(vector: EmotionVector) -> float:
    """Total strength the model actually receives, after the bias table."""
    return sum(value * bias for value, bias in zip(vector, EMOTION_BIAS))


def _self_check() -> None:
    """Fail loudly at import if the table violates its own contract."""
    for name, vector in EMOTION_PRESETS.items():
        if len(vector) != EMOTION_VECTOR_LENGTH:
            raise ValueError(
                f"preset {name!r} has {len(vector)} values, expected {EMOTION_VECTOR_LENGTH}"
            )
        if any(value < 0.0 or value > 1.0 for value in vector):
            raise ValueError(f"preset {name!r} has a value outside 0.0-1.0: {vector}")
        strength = post_bias_strength(vector)
        if strength > EMOTION_POST_BIAS_CAP + 1e-6:
            raise ValueError(
                f"preset {name!r} reaches {strength:.3f} after bias, above the "
                f"{EMOTION_POST_BIAS_CAP} cap; the engine would rescale and reshape it"
            )
        # A preset that lands far under the cap is the bug that made the first
        # version of this table inaudible. Two are known to be unable to reach it.
        if not any(vector):
            continue
        if strength < EMOTION_POST_BIAS_CAP * 0.85 and name not in EMOTION_CAPPED_BY_BIAS:
            raise ValueError(
                f"preset {name!r} only reaches {strength:.3f} after bias, well under "
                f"the {EMOTION_POST_BIAS_CAP} cap, so it will sound weak"
            )

    if EMOTION_PRESET_NEUTRAL not in EMOTION_PRESETS:
        raise ValueError(f"{EMOTION_PRESET_NEUTRAL!r} preset is required")
    if any(EMOTION_PRESETS[EMOTION_PRESET_NEUTRAL]):
        raise ValueError(f"{EMOTION_PRESET_NEUTRAL!r} must be all zeros")

    for name in EMOTION_CAPPED_BY_BIAS:
        if name not in EMOTION_PRESETS:
            raise ValueError(f"{name!r} is listed as bias-capped but is not a preset")


_self_check()
