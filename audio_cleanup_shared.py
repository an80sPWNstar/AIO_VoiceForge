"""Constants shared by the audio cleanup driver and its worker subprocess.

The cleanup pipeline runs in a SEPARATE virtualenv (see webui_audio_cleanup.py
for why), so the two halves cannot share objects -- only this module, which is
stdlib-only and importable from either interpreter.

Every tunable literal for cleanup lives here. Nothing below this file inlines
a stage name, a model filename or a threshold.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------

STAGE_ISOLATE = "isolate"        # strip music/instruments, keep the vocal stem
STAGE_DEREVERB = "dereverb"      # remove room reverb and echo
STAGE_DENOISE = "denoise"        # remove hiss, hum and broadband noise
STAGE_SPEAKER = "speaker"        # keep only one speaker's segments
STAGE_TRIM = "trim"              # drop leading/trailing/long internal silence
STAGE_NORMALIZE = "normalize"    # EBU R128 loudness normalisation

# Canonical execution order. The worker always runs stages in this order
# regardless of the order they arrive in, because the chain is not commutative:
# separation must precede speaker matching (embeddings are far more reliable on
# an isolated voice) and normalisation must come last (every earlier stage
# changes the level).
STAGE_ORDER: List[str] = [
    STAGE_ISOLATE,
    STAGE_DEREVERB,
    STAGE_DENOISE,
    STAGE_SPEAKER,
    STAGE_TRIM,
    STAGE_NORMALIZE,
]

STAGE_LABELS: Dict[str, str] = {
    STAGE_ISOLATE: "Remove music and background",
    STAGE_DEREVERB: "Remove reverb and echo",
    STAGE_DENOISE: "Remove noise and hiss",
    STAGE_SPEAKER: "Keep only one speaker",
    STAGE_TRIM: "Trim silence",
    STAGE_NORMALIZE: "Normalise loudness",
}

# Rough share of total runtime per stage, used to turn per-stage progress into
# one overall fraction. Only the ENABLED stages are considered, and the weights
# are renormalised over those, so these are relative rather than absolute.
STAGE_WEIGHTS: Dict[str, float] = {
    STAGE_ISOLATE: 0.40,
    STAGE_DEREVERB: 0.16,
    STAGE_DENOISE: 0.16,
    STAGE_SPEAKER: 0.22,
    STAGE_TRIM: 0.03,
    STAGE_NORMALIZE: 0.03,
}


# --------------------------------------------------------------------------
# Separation models
# --------------------------------------------------------------------------
#
# Only torch-backed models (.ckpt / .pth) are offered as defaults. The ONNX
# models in the UVR zoo are just as good, but running them on the GPU needs
# onnxruntime-gpu, which on Windows means matching CUDA and cuDNN DLLs by hand.
# The torch models reach the GPU through the torch already installed in the
# sidecar venv, so there is nothing extra to line up.

# Every filename below was read off `audio-separator --list_models` on this
# machine, with the SDR figures that listing reports. Do not edit them from
# memory -- re-run that command and copy the exact string.

VOCAL_MODELS: List[str] = [
    "vocals_mel_band_roformer.ckpt",              # vocals SDR 12.6, best listed
    "mel_band_roformer_kim_ft_unwa.ckpt",         # vocals SDR 12.4
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",  # vocals SDR 12.1
    "htdemucs_ft.yaml",                           # vocals SDR 10.8, 4-stem
]
DEFAULT_VOCAL_MODEL = VOCAL_MODELS[0]

DEREVERB_MODELS: List[str] = [
    "deverb_bs_roformer_8_384dim_10depth.ckpt",   # stems: noreverb*, reverb
    "UVR-DeEcho-DeReverb.pth",                    # stems: no reverb*, reverb
    "MDX23C-De-Reverb-aufr33-jarredou.ckpt",      # stems: dry, no dry
]
DEFAULT_DEREVERB_MODEL = DEREVERB_MODELS[0]

DENOISE_MODELS: List[str] = [
    "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",       # stems: dry*, other
    "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",  # same, harder
    "UVR-DeNoise.pth",                                         # stems: noise*, no noise
]
DEFAULT_DENOISE_MODEL = DENOISE_MODELS[0]

# Stem tags that identify the WANTED output of each stage, in preference order.
# These are matched against the "(...)" tag audio-separator puts in each output
# filename, NOT the whole filename -- the model name itself often contains a
# stem word ("denoise_mel_band_roformer...") and would match the wrong half.
#
# Two traps encoded here:
#   * The denoise roformers call the CLEAN stem "dry" and the discarded one
#     "other". Matching "other" would hand the noise residue down the chain.
#   * MDX23C-De-Reverb emits "dry" and "no dry", so an exact tag match is tried
#     before any substring match; otherwise "dry" also matches "no dry".
WANTED_STEM_KEYWORDS: Dict[str, List[str]] = {
    STAGE_ISOLATE: ["vocals"],
    STAGE_DEREVERB: ["noreverb", "no reverb", "no echo", "noecho", "dry"],
    STAGE_DENOISE: ["dry", "no noise", "nonoise"],
}


# --------------------------------------------------------------------------
# Speaker isolation
# --------------------------------------------------------------------------

SPEAKER_MODE_DOMINANT = "dominant"   # keep whoever talks the most
SPEAKER_MODE_SAMPLE = "sample"       # keep whoever matches a supplied clip
SPEAKER_MODES: List[str] = [SPEAKER_MODE_DOMINANT, SPEAKER_MODE_SAMPLE]
DEFAULT_SPEAKER_MODE = SPEAKER_MODE_DOMINANT

# speechbrain's ECAPA-TDNN speaker encoder. Apache-2.0 and ungated: unlike
# pyannote's diarization pipeline it needs no HuggingFace token and no
# accepting of model terms, which keeps first-run setup to a plain download.
SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Both the VAD and the speaker encoder are trained at 16 kHz.
SPEAKER_ANALYSIS_SAMPLE_RATE = 16000

# Cosine similarity above which a segment counts as the target speaker in
# sample mode. ECAPA cosine scores for same-speaker pairs typically sit well
# above 0.5; 0.25 is the conventional verification threshold, and the higher
# default here trades a little recall for not letting a second voice through.
DEFAULT_SPEAKER_THRESHOLD = 0.45
SPEAKER_THRESHOLD_MIN = 0.10
SPEAKER_THRESHOLD_MAX = 0.90
SPEAKER_THRESHOLD_STEP = 0.05

# Agglomerative clustering cut-off in cosine distance, used in dominant mode to
# decide how many distinct voices are present.
SPEAKER_CLUSTER_DISTANCE = 0.55

# Speech shorter than this is too little signal for a stable embedding, so it
# is judged by its neighbours rather than on its own.
MIN_SPEECH_SEGMENT_SECONDS = 0.60

# Padding kept either side of a retained segment so words are not clipped.
SEGMENT_PAD_SECONDS = 0.10

# Retained segments closer together than this are merged instead of being
# butt-joined, which avoids a crossfade in the middle of a continuous phrase.
SEGMENT_MERGE_GAP_SECONDS = 0.35

# Linear crossfade applied where two non-adjacent segments are joined. Long
# enough to hide the discontinuity, short enough not to swallow a syllable.
SEGMENT_CROSSFADE_SECONDS = 0.02


# --------------------------------------------------------------------------
# Silence trimming
# --------------------------------------------------------------------------

TRIM_SILENCE_THRESHOLD_DB = -40.0
TRIM_MIN_SILENCE_SECONDS = 0.60
# Silence deliberately left in place of a removed gap, so speech does not run
# together unnaturally.
TRIM_KEEP_SILENCE_SECONDS = 0.25


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

CLEANUP_SUBDIR = "cleaned"
CLEANUP_SUFFIX = "_clean"
# No leading dot: media_fetch.unique_path adds the separator itself.
CLEANUP_EXTENSION = "wav"
# Intermediates are written here when the user asks to keep them, so each
# stage's output can be listened to on its own.
STAGES_SUBDIR = "stages"

# Working format between stages: 32-bit float WAV, so repeated passes never
# accumulate quantisation error. The final file is written at the requested
# rate and bit depth.
INTERMEDIATE_SAMPLE_RATE = 44100

DEFAULT_DEVICE = "cuda"
DEVICE_CPU = "cpu"

# Model cache lives beside the venv rather than in the repo, since it is
# several GB of downloads and must survive a git clean.
MODEL_CACHE_DIRNAME = "audio_cleanup_models"


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CleanupPreset:
    """One entry in the preset radio: a name, a blurb, and the stages it runs."""

    key: str
    label: str
    description: str
    stages: List[str] = field(default_factory=list)


PRESET_CUSTOM_KEY = "custom"

CLEANUP_PRESETS: List[CleanupPreset] = [
    CleanupPreset(
        key="music",
        label="Just remove the music",
        description=(
            "Strips instruments and background, keeps every voice. Fastest, and "
            "the safest choice when the clip is already one person talking over "
            "a backing track."
        ),
        stages=[STAGE_ISOLATE, STAGE_NORMALIZE],
    ),
    CleanupPreset(
        key="standard",
        label="Standard clean-up",
        description=(
            "Removes music and noise, keeps only the main speaker, trims dead "
            "air. This is the one to use for a reference voice."
        ),
        stages=[
            STAGE_ISOLATE,
            STAGE_DENOISE,
            STAGE_SPEAKER,
            STAGE_TRIM,
            STAGE_NORMALIZE,
        ],
    ),
    CleanupPreset(
        key="studio",
        label="Aggressive - studio clean",
        description=(
            "Everything, including de-reverb. Best on echoey rooms and phone "
            "recordings; can sound slightly processed on already-clean audio."
        ),
        stages=list(STAGE_ORDER),
    ),
    CleanupPreset(
        key=PRESET_CUSTOM_KEY,
        label="Custom",
        description="Whatever is ticked under Advanced.",
        stages=[],
    ),
]

DEFAULT_PRESET_KEY = "standard"


def preset_by_key(key: str) -> Optional[CleanupPreset]:
    """Return the preset with this key, or None when there is no such preset."""
    for preset in CLEANUP_PRESETS:
        if preset.key == key:
            return preset
    return None


def ordered_stages(stages) -> List[str]:
    """Return `stages` deduplicated and sorted into canonical pipeline order."""
    wanted = set(stages or ())
    return [stage for stage in STAGE_ORDER if stage in wanted]


def progress_plan(stages) -> Dict[str, tuple]:
    """Map each enabled stage to its (start, end) slice of the 0..1 bar.

    Weights are renormalised over the enabled stages only, so a two-stage run
    still fills the whole bar.
    """
    active = ordered_stages(stages)
    total = sum(STAGE_WEIGHTS[stage] for stage in active)
    if total <= 0:
        return {}
    plan = {}
    cursor = 0.0
    for stage in active:
        span = STAGE_WEIGHTS[stage] / total
        plan[stage] = (cursor, cursor + span)
        cursor += span
    return plan
