"""Cutting a long recording into candidate reference clips, and measuring them.

The engine copies the delivery of whatever reference it is given -- it does
not analyse a clip for range, it imitates it. So picking a reference is
picking a moment: a calm one for narration, an animated one for dialogue.
That is what this module is for. It splits a recording where the speaker
stops, then measures each piece so the good ones can be found without
listening to all of them.

What it measures, and why these rather than an emotion label:

  duration        the 15-second cap is hard, so anything shorter is a
                  candidate and anything longer will be truncated
  loudness (LUFS) integrated loudness, the same unit the clean-up chain
                  normalises to
  pitch           mean and range in Hz. Range is the useful one: a wide
                  range is an animated delivery, a narrow one is flat
  voiced fraction how much of the span is actually speech rather than
                  breath and room
  onset rate      events per second, a proxy for pace

Deliberately no emotion classifier. Automatic speech-emotion labelling runs
around 60-75% on acted speech and worse on real recordings, and a label you
cannot trust is worse than a number you can. Pitch range and onset rate
answer "find me an animated 15 seconds" more reliably than a guess at
"excited" does.

Pure analysis: no gradio, no threads, no filesystem beyond reading the file
it is given. The UI layer owns progress reporting and the worker thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

# The engine truncates a speaker reference to this. A segment longer than it
# is still useful -- it just means only the opening will ever be heard.
ENGINE_REFERENCE_SECONDS = 15.0

# Below this a span is too short to clone from usefully; above the cap it is
# only partly used. Both are reported, neither is enforced.
MIN_USEFUL_SECONDS = 2.0

# librosa.effects.split threshold, in dB below the peak. 30 keeps breaths and
# room tone out while not cutting inside a quiet word.
DEFAULT_SILENCE_TOP_DB = 30

# Merge two spans separated by less than this. Sentence-internal pauses are
# routinely 200-400 ms and cutting there produces unusable fragments.
DEFAULT_MERGE_GAP_SECONDS = 0.35

# Pitch tracking is the expensive part. Speech f0 sits well inside this and a
# narrower band is markedly faster than librosa's default range.
PITCH_FLOOR_HZ = 65.0
PITCH_CEILING_HZ = 500.0

# pyin runs on a decimated copy: f0 up to 500 Hz needs nothing like 22 kHz,
# and the cost scales with sample rate.
PITCH_ANALYSIS_RATE = 16000

# Loudness needs a window; pyloudnorm's BS.1770 meter requires at least this
# much audio or it raises rather than returning a number.
MIN_LOUDNESS_SECONDS = 0.4

# An absolute floor, because librosa.effects.split thresholds RELATIVE to the
# peak: in a silent buffer every sample equals the peak, so it returns the
# whole thing as one non-silent span. Without this guard a silent recording
# reports a single segment covering all of it, which reads as "here is a fine
# ten-minute reference" rather than "there is nothing here".
SILENCE_FLOOR_DBFS = -60.0


@dataclass(frozen=True)
class Segment:
    """One candidate clip, with everything measured about it."""

    index: int
    start_s: float
    end_s: float
    duration_s: float
    lufs: Optional[float]
    peak_dbfs: Optional[float]
    pitch_hz_mean: Optional[float]
    pitch_hz_range: Optional[float]
    voiced_fraction: Optional[float]
    onset_rate_hz: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def fits_the_reference_cap(self) -> bool:
        return self.duration_s <= ENGINE_REFERENCE_SECONDS

    @property
    def is_long_enough(self) -> bool:
        return self.duration_s >= MIN_USEFUL_SECONDS


def find_speech_spans(
    audio: np.ndarray,
    sample_rate: int,
    top_db: int = DEFAULT_SILENCE_TOP_DB,
    merge_gap_s: float = DEFAULT_MERGE_GAP_SECONDS,
) -> List[Tuple[float, float]]:
    """Where the speaker is talking, in seconds, merged across short pauses.

    Returns an empty list for silence rather than one span covering it, so a
    caller can tell "nothing here" from "one long take".
    """
    if audio.size == 0 or sample_rate <= 0:
        return []

    # Absolute check first -- see SILENCE_FLOOR_DBFS. split() cannot make this
    # judgement because its threshold is relative to whatever the loudest
    # sample happens to be.
    peak = _peak_dbfs(audio)
    if peak is None or peak < SILENCE_FLOOR_DBFS:
        return []

    intervals = librosa.effects.split(audio, top_db=top_db)
    if len(intervals) == 0:
        return []

    spans = [(start / sample_rate, end / sample_rate) for start, end in intervals]
    return _merge_close_spans(spans, merge_gap_s)


def _merge_close_spans(
    spans: Sequence[Tuple[float, float]], merge_gap_s: float
) -> List[Tuple[float, float]]:
    """Join spans separated by less than merge_gap_s.

    Splitting inside a sentence produces fragments that sound clipped as a
    reference, so the gap between spans decides, not their length.
    """
    if not spans:
        return []
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def measure_span(audio: np.ndarray, sample_rate: int) -> Dict[str, Optional[float]]:
    """Everything measurable about one span of audio.

    Every field is independently optional: a span too short to meter for
    loudness can still have a pitch, and an unvoiced span can still have a
    peak. A measurement that could not be taken is None, never a guess.
    """
    return {
        "lufs": _integrated_loudness(audio, sample_rate),
        "peak_dbfs": _peak_dbfs(audio),
        **_pitch_statistics(audio, sample_rate),
        "onset_rate_hz": _onset_rate(audio, sample_rate),
    }


def _peak_dbfs(audio: np.ndarray) -> Optional[float]:
    if audio.size == 0:
        return None
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0:
        return None
    return 20.0 * math.log10(peak)


def _integrated_loudness(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """BS.1770 integrated loudness, or None when the span is too short.

    pyloudnorm raises on audio shorter than its block size rather than
    returning a rough number, which is the right call -- a loudness figure
    from 200 ms of audio would not mean anything.
    """
    if audio.size < int(MIN_LOUDNESS_SECONDS * sample_rate):
        return None
    try:
        import pyloudnorm
    except ImportError:
        return None
    try:
        meter = pyloudnorm.Meter(sample_rate)
        value = float(meter.integrated_loudness(audio.astype(np.float64)))
    except (ValueError, ZeroDivisionError):
        return None
    # Digital silence meters as -inf, which is true but not printable.
    return value if math.isfinite(value) else None


def _pitch_statistics(audio: np.ndarray, sample_rate: int) -> Dict[str, Optional[float]]:
    """Mean f0, the spread of f0, and how much of the span is voiced.

    Range is reported as the 5th-to-95th percentile rather than min-to-max:
    a single octave-error frame would otherwise dominate, and pitch trackers
    produce those on breaths and plosives.
    """
    empty: Dict[str, Optional[float]] = {
        "pitch_hz_mean": None,
        "pitch_hz_range": None,
        "voiced_fraction": None,
    }
    if audio.size == 0:
        return empty

    if sample_rate > PITCH_ANALYSIS_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=PITCH_ANALYSIS_RATE)
        sample_rate = PITCH_ANALYSIS_RATE

    # pyin needs at least a couple of frames at the lowest frequency it tracks.
    if audio.size < int(sample_rate / PITCH_FLOOR_HZ) * 4:
        return empty

    try:
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=PITCH_FLOOR_HZ,
            fmax=PITCH_CEILING_HZ,
            sr=sample_rate,
        )
    except Exception:
        # librosa raises several unrelated types here depending on input
        # length and dtype. An unmeasurable pitch is not a failure worth
        # taking the whole scan down for.
        return empty

    voiced = f0[np.isfinite(f0)]
    fraction = float(np.mean(voiced_flag)) if voiced_flag.size else None
    if voiced.size == 0:
        return {**empty, "voiced_fraction": fraction}

    low, high = np.percentile(voiced, [5, 95])
    return {
        "pitch_hz_mean": float(np.mean(voiced)),
        "pitch_hz_range": float(high - low),
        "voiced_fraction": fraction,
    }


def _onset_rate(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Onsets per second: a proxy for pace.

    Not a speaking rate -- that needs words, and words need a transcript.
    Named for what it measures.
    """
    if audio.size < sample_rate // 2:
        return None
    try:
        onsets = librosa.onset.onset_detect(y=audio, sr=sample_rate, units="time")
    except Exception:
        return None
    duration = audio.size / float(sample_rate)
    if duration <= 0:
        return None
    return float(len(onsets) / duration)


def segment_audio(
    audio: np.ndarray,
    sample_rate: int,
    top_db: int = DEFAULT_SILENCE_TOP_DB,
    merge_gap_s: float = DEFAULT_MERGE_GAP_SECONDS,
    min_seconds: float = MIN_USEFUL_SECONDS,
    progress: Optional[Any] = None,
) -> List[Segment]:
    """Split a recording and measure every piece.

    `progress` is an optional callable taking (done, total); the UI layer
    supplies one so a long scan can report itself. Nothing here prints.
    """
    spans = [
        (start, end)
        for start, end in find_speech_spans(audio, sample_rate, top_db, merge_gap_s)
        if (end - start) >= min_seconds
    ]

    segments: List[Segment] = []
    for index, (start, end) in enumerate(spans):
        piece = audio[int(start * sample_rate): int(end * sample_rate)]
        measured = measure_span(piece, sample_rate)
        segments.append(
            Segment(
                index=index,
                start_s=round(start, 3),
                end_s=round(end, 3),
                duration_s=round(end - start, 3),
                **measured,
            )
        )
        if progress is not None:
            progress(index + 1, len(spans))
    return segments


def load_and_segment(
    path: str,
    top_db: int = DEFAULT_SILENCE_TOP_DB,
    merge_gap_s: float = DEFAULT_MERGE_GAP_SECONDS,
    min_seconds: float = MIN_USEFUL_SECONDS,
    progress: Optional[Any] = None,
) -> Tuple[List[Segment], int]:
    """Read an audio file and segment it. Returns the segments and the rate.

    Mono, at the file's own rate -- resampling here would cost time and
    change nothing about where the speaker pauses.
    """
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    return segment_audio(audio, sample_rate, top_db, merge_gap_s, min_seconds, progress), sample_rate


def rank_for_reference(segments: Sequence[Segment]) -> List[Segment]:
    """Best candidates first, for someone who just wants a usable clip.

    Ordering is: fits the 15-second cap, is long enough to clone from, then
    most voiced. Loudness and pitch are deliberately NOT ranked on -- there
    is no single best loudness or pitch, only the one that matches the
    delivery wanted, and that is the user's call.
    """
    def key(segment: Segment):
        return (
            not segment.fits_the_reference_cap,
            not segment.is_long_enough,
            -(segment.voiced_fraction or 0.0),
        )

    return sorted(segments, key=key)
