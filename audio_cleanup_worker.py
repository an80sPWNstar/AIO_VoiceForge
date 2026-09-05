"""Audio cleanup worker -- runs INSIDE the sidecar venv, never in the app venv.

Reads one audio file, runs the requested cleanup stages over it, and writes one
cleaned file. Progress is appended as JSON lines to --progress-file; the final
result is printed to stdout as a single JSON object.

It is a separate process because the separation stack (audio-separator,
speechbrain, its own torch) cannot share an environment with the app's pinned
gradio 6.17.3 / transformers 4.52.1 / torch 2.8.0 without risking exactly the
kind of silent dependency drift that has broken this app twice already.

Invoked by webui_audio_cleanup.py. Runnable by hand for debugging:

    <sidecar-venv>\\Scripts\\python.exe audio_cleanup_worker.py \\
        --input clip.wav --output clean.wav --stages isolate,denoise,speaker
"""

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_cleanup_shared as shared  # noqa: E402
import webui_media_fetch as media_fetch  # noqa: E402  (stdlib-only at import)


class CleanupError(RuntimeError):
    """A cleanup stage failed for a reason worth showing the user verbatim."""


# Speaker encoder, built once per process: loading it costs seconds and both
# the segment pass and the reference pass need the same one.
_ENCODER = None


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------

class ProgressReporter:
    """Appends progress events to a file the parent process tails.

    The file is opened with newline="" on purpose: Python would otherwise
    translate "\\n" to "\\r\\n" on Windows, and the parent reads the file in
    binary mode by byte offset, so every translated line would slip its
    bookkeeping by a byte.
    """

    def __init__(self, path: Optional[str], stages: Sequence[str]):
        self._path = path
        self._plan = shared.progress_plan(stages)
        self._handle = None
        if path:
            self._handle = open(path, "a", encoding="utf-8", newline="")

    def stage(self, stage: str, within: float, message: str) -> None:
        """Report `within` (0..1) progress inside `stage`."""
        start, end = self._plan.get(stage, (0.0, 1.0))
        clamped = min(max(within, 0.0), 1.0)
        self._emit(start + (end - start) * clamped, stage, message)

    def note(self, message: str) -> None:
        """Report a message that does not move the bar."""
        self._emit(None, None, message)

    def _emit(self, fraction: Optional[float], stage: Optional[str], message: str) -> None:
        payload = {"fraction": fraction, "stage": stage, "message": message}
        line = json.dumps(payload, ensure_ascii=True)
        if self._handle is None:
            print(line, file=sys.stderr, flush=True)
            return
        self._handle.write(line + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


# --------------------------------------------------------------------------
# Separation stages (audio-separator / UVR models)
# --------------------------------------------------------------------------

def _stem_tag(filename: str) -> str:
    """Return the lowercased "(...)" stem tag audio-separator writes into a name.

    Output files look like `clip_(Vocals)_model_name.wav`. Only the tag may be
    matched against: the model name itself frequently contains a stem word
    (denoise_mel_band_roformer...), so matching the whole filename picks the
    wrong stem about as often as the right one.
    """
    tags = re.findall(r"\(([^)]+)\)", os.path.basename(filename))
    return tags[-1].strip().lower() if tags else ""


def _pick_stem(stage: str, produced: Sequence[str], output_dir: str) -> str:
    """Return the produced file that holds the stem this stage wants.

    Exact tag matches are tried before substring matches, because several
    models emit a complementary pair whose names nest -- MDX23C-De-Reverb
    produces both "dry" and "no dry", and a substring test would take whichever
    happened to come first.

    A model whose stems match nothing raises rather than letting the pipeline
    continue with, say, the instrumental track or the isolated noise.
    """
    keywords = shared.WANTED_STEM_KEYWORDS[stage]
    tagged = [(os.path.basename(name), _stem_tag(name)) for name in produced]

    for keyword in keywords:
        for name, tag in tagged:
            if tag == keyword:
                return os.path.join(output_dir, name)
    for keyword in keywords:
        for name, tag in tagged:
            if keyword in tag:
                return os.path.join(output_dir, name)
    raise CleanupError(
        f"The {shared.STAGE_LABELS[stage]} model produced "
        f"stems {[_stem_tag(n) or os.path.basename(n) for n in produced]}, none of which look like "
        f"the wanted stem (expected a stem tag matching one of {keywords}). "
        "Pick a different model for this stage."
    )


class SeparationRunner:
    """Lazily-built audio-separator wrapper, reused across separation stages."""

    def __init__(self, model_dir: str, work_dir: str):
        self._model_dir = model_dir
        self._work_dir = work_dir
        self._separator = None
        self._loaded_model = None

    def _ensure(self):
        if self._separator is not None:
            return self._separator
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:
            raise CleanupError(
                "audio-separator is not installed in the cleanup environment. "
                "Re-run the cleanup setup script. "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._work_dir, exist_ok=True)
        self._separator = Separator(
            model_file_dir=self._model_dir,
            output_dir=self._work_dir,
            output_format="WAV",
        )
        return self._separator

    def run(self, stage: str, model_filename: str, source_path: str,
            reporter: ProgressReporter) -> str:
        """Run one separation model over `source_path`, return the wanted stem."""
        separator = self._ensure()
        label = shared.STAGE_LABELS[stage]

        if self._loaded_model != model_filename:
            reporter.stage(stage, 0.05, f"{label}: loading {model_filename}")
            try:
                separator.load_model(model_filename=model_filename)
            except Exception as exc:
                raise CleanupError(
                    f"Could not load the model {model_filename!r} for "
                    f"{label.lower()}: {type(exc).__name__}: {exc}"
                ) from exc
            self._loaded_model = model_filename

        reporter.stage(stage, 0.25, f"{label}: processing")
        try:
            produced = separator.separate(source_path)
        except Exception as exc:
            raise CleanupError(
                f"{label} failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not produced:
            raise CleanupError(f"{label} produced no output files.")

        chosen = _pick_stem(stage, produced, self._work_dir)
        reporter.stage(stage, 1.0, f"{label}: done")
        return chosen


# --------------------------------------------------------------------------
# Speaker isolation
# --------------------------------------------------------------------------

def _detect_speech(path: str, reporter: ProgressReporter) -> Tuple[List[Tuple[float, float]], float]:
    """Return speech spans as (start, end) seconds, plus the clip duration."""
    import torch  # noqa: F401  (silero_vad needs it loaded)
    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
    except ImportError as exc:
        raise CleanupError(
            "silero-vad is not installed in the cleanup environment. "
            f"({type(exc).__name__}: {exc})"
        ) from exc

    rate = shared.SPEAKER_ANALYSIS_SAMPLE_RATE
    audio = read_audio(path, sampling_rate=rate)
    duration = float(len(audio)) / rate
    model = load_silero_vad()
    stamps = get_speech_timestamps(
        audio, model, sampling_rate=rate, return_seconds=True
    )
    spans = [(float(s["start"]), float(s["end"])) for s in stamps]
    reporter.stage(
        shared.STAGE_SPEAKER, 0.25,
        f"Found {len(spans)} speech segments in {duration:.0f}s of audio",
    )
    return spans, duration


def _load_encoder():
    """Build the ECAPA speaker encoder, cached for the life of the process.

    speechbrain rejects a bare "cuda" and falls back to device 0 with a
    warning, so the index is always spelled out.
    """
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER
    import torch
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise CleanupError(
            "speechbrain is not installed in the cleanup environment. "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    device = "cuda:0" if torch.cuda.is_available() else shared.DEVICE_CPU
    _ENCODER = EncoderClassifier.from_hparams(
        source=shared.SPEAKER_EMBEDDING_MODEL,
        savedir=os.path.join(
            os.environ.get("AUDIO_CLEANUP_MODEL_DIR", "."), "ecapa"),
        run_opts={"device": device},
    )
    return _ENCODER


def _embed_segments(path: str, spans: Sequence[Tuple[float, float]],
                    reporter: ProgressReporter):
    """Return an L2-normalised ECAPA embedding per span, as a numpy array."""
    import numpy as np
    import torch
    from silero_vad import read_audio

    rate = shared.SPEAKER_ANALYSIS_SAMPLE_RATE
    audio = read_audio(path, sampling_rate=rate)
    encoder = _load_encoder()

    vectors = []
    for index, (start, end) in enumerate(spans):
        chunk = audio[int(start * rate):int(end * rate)]
        with torch.no_grad():
            embedding = encoder.encode_batch(chunk.unsqueeze(0))
        vector = embedding.squeeze().detach().cpu().numpy()
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm > 0 else vector)
        if index % 10 == 0:
            reporter.stage(
                shared.STAGE_SPEAKER,
                0.25 + 0.45 * (index + 1) / max(len(spans), 1),
                f"Comparing voices ({index + 1}/{len(spans)})",
            )
    return np.stack(vectors) if vectors else None


def _embed_reference(path: str):
    """Return one L2-normalised embedding for a whole reference clip."""
    import numpy as np
    import torch
    from silero_vad import read_audio

    audio = read_audio(path, sampling_rate=shared.SPEAKER_ANALYSIS_SAMPLE_RATE)
    encoder = _load_encoder()
    with torch.no_grad():
        embedding = encoder.encode_batch(audio.unsqueeze(0))
    vector = embedding.squeeze().detach().cpu().numpy()
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _analysis_units(spans):
    """Merge nearby VAD spans into units long enough to embed stably.

    Sub-second spans produce embeddings so noisy that one speaker clusters as
    several voices (see VOICE_UNIT_TARGET_SECONDS in the shared constants).
    Only pauses up to VOICE_UNIT_MAX_GAP_SECONDS are bridged, and a unit stops
    growing once it reaches the target, so turn-taking between two speakers is
    unlikely to be fused into one unit.
    """
    units = []
    for start, end in spans:
        if units:
            last_start, last_end = units[-1]
            if (start - last_end <= shared.VOICE_UNIT_MAX_GAP_SECONDS
                    and (last_end - last_start) < shared.VOICE_UNIT_TARGET_SECONDS):
                units[-1] = (last_start, end)
                continue
        units.append((start, end))
    return units


def _merge_similar_clusters(vectors, labels, talk_time):
    """Fold clusters whose centroids read as the same voice into one.

    Returns (labels, talk_time) with the smaller cluster relabelled into the
    larger at each merge, repeated until no pair scores at or above
    VOICE_MERGE_SIMILARITY.
    """
    import numpy as np

    labels = np.array(labels)
    talk_time = dict(talk_time)
    while len(talk_time) > 1:
        ids = sorted(talk_time)
        centroids = {i: _normalised_centroid(vectors, labels, i) for i in ids}
        best_pair, best_similarity = None, shared.VOICE_MERGE_SIMILARITY
        for index, a in enumerate(ids):
            for b in ids[index + 1:]:
                similarity = float(centroids[a] @ centroids[b])
                if similarity >= best_similarity:
                    best_pair, best_similarity = (a, b), similarity
        if best_pair is None:
            break
        a, b = best_pair
        keep, fold = (a, b) if talk_time[a] >= talk_time[b] else (b, a)
        labels[labels == fold] = keep
        talk_time[keep] += talk_time.pop(fold)
    return labels, talk_time


def _cluster_voices(vectors, spans):
    """Group spans into voices. Returns (labels, talk_time) where talk_time
    maps int label -> total seconds.
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=shared.SPEAKER_CLUSTER_DISTANCE,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(vectors)

    talk_time = {}
    for label, (start, end) in zip(labels, spans):
        talk_time[int(label)] = talk_time.get(int(label), 0.0) + (end - start)

    return _merge_similar_clusters(vectors, labels, talk_time)


def _normalised_centroid(vectors, labels, label):
    """Mean of the label's vectors, L2-normalised."""
    import numpy as np

    centroid = vectors[labels == label].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def _dominant_centroid(vectors, spans):
    """Return (target_vector, note) for the speaker who talks the most.

    Clustering runs on whole VAD spans because long spans give stable
    embeddings, which is what choosing the right speaker needs. The result only
    picks a target; what actually gets kept is decided per window by
    _score_windows.
    """
    if len(spans) == 1:
        return vectors[0], "Only one usable speech segment; used it as the target voice."

    labels, talk_time = _cluster_voices(vectors, spans)
    dominant = max(talk_time, key=talk_time.get)
    centroid = _normalised_centroid(vectors, labels, dominant)

    total = sum(talk_time.values())
    share = talk_time[dominant] / total if total else 1.0
    note = (
        f"Heard {len(talk_time)} distinct voice(s); targeted the one talking "
        f"{talk_time[dominant]:.0f}s ({share * 100:.0f}% of the speech)."
    )
    return centroid, note


def _score_windows(audio, spans, target, encoder):
    """Score every window of speech against `target`.

    Returns [(start, end, similarity)]. Windows overlap, so a speaker change
    part-way through a long span is caught by whichever windows straddle it --
    exactly the case the previous per-span version could not see.
    """
    import numpy as np
    import torch

    rate = shared.SPEAKER_ANALYSIS_SAMPLE_RATE
    window = shared.SPEAKER_WINDOW_SECONDS
    hop = shared.SPEAKER_HOP_SECONDS
    minimum_samples = int(0.2 * rate)

    scored = []
    for start, end in spans:
        starts = []
        cursor = start
        while cursor + window <= end + 0.01:
            starts.append(cursor)
            cursor += hop
        if not starts:
            starts = [start]        # span shorter than one window; score it whole
        for begin in starts:
            finish = min(begin + window, end)
            chunk = audio[int(begin * rate):int(finish * rate)]
            if len(chunk) < minimum_samples:
                continue
            with torch.no_grad():
                embedding = encoder.encode_batch(chunk.unsqueeze(0))
            vector = embedding.squeeze().detach().cpu().numpy()
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            scored.append((begin, finish, float(vector @ target)))
    return scored


def _keep_intervals(scored, threshold, duration):
    """Turn window scores into (kept, dropped) interval lists.

    Overlapping windows vote on each slice and the mean decides it, so a
    transition is judged from both sides rather than by whichever window
    happened to come first.
    """
    import numpy as np

    slice_seconds = shared.SPEAKER_SLICE_SECONDS
    count = int(np.ceil(duration / slice_seconds)) + 1
    totals = np.zeros(count)
    votes = np.zeros(count)

    for begin, finish, similarity in scored:
        first = int(begin / slice_seconds)
        last = min(int(np.ceil(finish / slice_seconds)), count)
        totals[first:last] += similarity
        votes[first:last] += 1

    speech = votes > 0
    means = np.divide(totals, votes, out=np.zeros_like(totals), where=speech)
    keep_flags = speech & (means >= threshold)
    drop_flags = speech & ~keep_flags

    def runs(flags):
        found = []
        index = 0
        while index < count:
            if not flags[index]:
                index += 1
                continue
            start = index
            while index < count and flags[index]:
                index += 1
            found.append((start * slice_seconds,
                          min(index * slice_seconds, duration)))
        return found

    kept = [span for span in runs(keep_flags)
            if span[1] - span[0] >= shared.MIN_KEPT_RUN_SECONDS]
    return kept, runs(drop_flags)


def _pad_and_merge(kept, dropped, duration):
    """Pad and merge kept intervals WITHOUT ever re-including dropped speech.

    The previous version padded and merged in isolation, so a gap shorter than
    the merge threshold got bridged even when the audio inside it had just been
    classified as somebody else. Every extension here is clipped against the
    dropped spans.
    """
    def clip_forward(value, limit):
        """Largest position <= value that does not reach into dropped speech."""
        for start, end in dropped:
            if start >= limit and start < value:
                return start
        return value

    def clip_backward(value, limit):
        """Smallest position >= value that does not reach into dropped speech."""
        for start, end in dropped:
            if end <= limit and end > value:
                return end
        return value

    padded = []
    for start, end in kept:
        new_start = clip_backward(max(0.0, start - shared.SEGMENT_PAD_SECONDS), start)
        new_end = clip_forward(min(duration, end + shared.SEGMENT_PAD_SECONDS), end)
        padded.append((new_start, new_end))
    padded.sort()

    merged = []
    for start, end in padded:
        if not merged:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        gap_has_speech = any(
            drop_start < start and drop_end > previous_end
            for drop_start, drop_end in dropped
        )
        if start - previous_end <= shared.SEGMENT_MERGE_GAP_SECONDS and not gap_has_speech:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _concatenate(source_path: str, keep: Sequence[Tuple[float, float]],
                 destination_path: str) -> float:
    """Write only `keep` spans of the source, crossfaded at each join."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(source_path, dtype="float32", always_2d=True)
    fade_samples = max(1, int(shared.SEGMENT_CROSSFADE_SECONDS * rate))

    pieces = []
    for start, end in keep:
        chunk = data[int(start * rate):int(end * rate)].copy()
        if len(chunk) <= 2 * fade_samples:
            pieces.append(chunk)
            continue
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype="float32")[:, None]
        chunk[:fade_samples] *= ramp
        chunk[-fade_samples:] *= ramp[::-1]
        pieces.append(chunk)

    if not pieces:
        raise CleanupError("Speaker isolation kept nothing at all.")

    joined = np.concatenate(pieces, axis=0)
    sf.write(destination_path, joined, rate, subtype="FLOAT")
    return float(len(joined)) / rate


def _best_reference_span(scored, threshold, duration):
    """Pick the best <= REFERENCE_CLIP_SECONDS span of the target's speech.

    `scored` is _score_windows output: [(begin, end, similarity)].
    Returns (start, end).
    """
    import numpy as np

    if not scored:
        raise CleanupError("No speech windows to select a reference from.")

    kept = [w for w in scored if w[2] >= threshold]
    if not kept:
        kept = [max(scored, key=lambda w: w[2])]

    runs = []
    current_run = [kept[0]]
    for window in kept[1:]:
        if window[0] <= current_run[-1][1]:
            current_run.append(window)
        else:
            runs.append(current_run)
            current_run = [window]
    if current_run:
        runs.append(current_run)

    candidates = []
    for run in runs:
        run_start = run[0][0]
        run_end = run[-1][1]
        run_interval = run_end - run_start
        run_similarities = [w[2] for w in run]

        if run_interval <= shared.REFERENCE_CLIP_SECONDS:
            score = float(np.mean(run_similarities))
            candidates.append(((run_start, run_end), score, run_interval))
        else:
            cursor = run_start
            while cursor + shared.REFERENCE_CLIP_SECONDS <= run_end:
                window_end = cursor + shared.REFERENCE_CLIP_SECONDS
                window_sims = [w[2] for w in run
                               if w[0] >= cursor and w[1] <= window_end]
                if window_sims:
                    score = float(np.mean(window_sims))
                    candidates.append(((cursor, window_end), score, shared.REFERENCE_CLIP_SECONDS))
                cursor += shared.SPEAKER_HOP_SECONDS

    if not candidates:
        raise CleanupError("Could not find a suitable reference span.")

    # Longest first, similarity as the tie-break: every candidate already
    # cleared the match threshold, and the engine gets more out of 10 good
    # seconds than out of 1.2 slightly-better ones.
    best = max(candidates, key=lambda c: (c[2], c[1]))
    start, end = best[0]
    end = min(end, duration)
    return start, end


def _isolate_target(source_path: str, destination_path: str, target, threshold: float,
                    spans: Sequence[Tuple[float, float]], duration: float,
                    reporter: ProgressReporter):
    """Score, keep, pad, concatenate. Returns (kept_seconds, removed_seconds,
    scored) where scored is the _score_windows output.
    """
    from silero_vad import read_audio

    reporter.stage(shared.STAGE_SPEAKER, 0.70,
                   "Checking every second for other voices")
    audio = read_audio(source_path, sampling_rate=shared.SPEAKER_ANALYSIS_SAMPLE_RATE)
    scored = _score_windows(audio, spans, target, _load_encoder())
    if not scored:
        raise CleanupError("Speaker isolation could not score any speech.")

    kept, dropped = _keep_intervals(scored, threshold, duration)
    if not kept:
        best = max(similarity for _, _, similarity in scored)
        raise CleanupError(
            "No speech matched the target voice closely enough (best match "
            f"{best:.2f}, threshold {threshold:.2f}). Lower the voice match "
            "threshold, or check the voice sample is the right person."
        )

    final = _pad_and_merge(kept, dropped, duration)
    reporter.stage(shared.STAGE_SPEAKER, 0.85, "Rebuilding the clip")
    kept_seconds = _concatenate(source_path, final, destination_path)
    removed = sum(end - start for start, end in dropped)
    reporter.stage(shared.STAGE_SPEAKER, 1.0, "Speaker isolated")

    return kept_seconds, removed, scored


def keep_one_speaker(source_path: str, destination_path: str, mode: str,
                     sample_path: Optional[str], threshold: float,
                     reporter: ProgressReporter) -> str:
    """Run the speaker isolation stage. Returns a human-readable note.

    Two passes. The first decides WHO to keep, from whole speech spans, because
    long spans embed stably. The second decides WHAT to keep, scoring every
    1.5s window against that target -- which is what catches a second speaker
    buried inside one long unbroken span.
    """
    spans, duration = _detect_speech(source_path, reporter)
    if not spans:
        raise CleanupError(
            "No speech was detected in this clip, so there is no speaker to "
            "isolate. Turn that stage off, or check the earlier stages did not "
            "remove the voice."
        )

    if mode == shared.SPEAKER_MODE_SAMPLE:
        if not sample_path:
            raise CleanupError(
                "Speaker mode is 'match a sample' but no sample was supplied."
            )
        target = _embed_reference(sample_path)
        note = "Matched against the supplied voice sample."
    else:
        usable = [u for u in _analysis_units(spans)
                  if (u[1] - u[0]) >= shared.MIN_SPEECH_SEGMENT_SECONDS]
        if not usable:
            reporter.stage(shared.STAGE_SPEAKER, 1.0, "Only one voice present")
            _copy_audio(source_path, destination_path)
            return "Too little continuous speech to tell voices apart; kept everything."
        vectors = _embed_segments(source_path, usable, reporter)
        target, note = _dominant_centroid(vectors, usable)

    kept_seconds, removed, scored = _isolate_target(
        source_path, destination_path, target, threshold, spans, duration, reporter
    )
    return (
        f"{note} Removed {removed:.0f}s of other voices; "
        f"result is {kept_seconds:.0f}s long."
    )


def _copy_audio(source_path: str, destination_path: str) -> None:
    """Copy audio through soundfile so the destination is always a valid WAV."""
    import soundfile as sf
    data, rate = sf.read(source_path, dtype="float32", always_2d=True)
    sf.write(destination_path, data, rate, subtype="FLOAT")


def _found_voices(vectors, units):
    """Group units into listable voices, gated by embedding stability.

    Only units of at least VOICE_FOUNDER_MIN_SECONDS may found a voice:
    shorter ones embed too noisily and, clustered directly, split one speaker
    into many "voices" (the failure this replaces). Short units join the
    founded voice they best match, or stay unattributed (label -1) and off the
    list. Returns (labels, talk_time, notes); talk_time never has a -1 entry.
    """
    import numpy as np

    lengths = [unit[1] - unit[0] for unit in units]
    founder_indices = [i for i, length in enumerate(lengths)
                       if length >= shared.VOICE_FOUNDER_MIN_SECONDS]
    notes = []

    if not founder_indices:
        # Nothing stable to anchor on: cluster everything, but say the split
        # is unreliable rather than presenting it as confident.
        if len(units) == 1:
            labels = np.zeros(1, dtype=int)
            talk_time = {0: lengths[0]}
        else:
            labels, talk_time = _cluster_voices(vectors, units)
        notes.append(
            f"No speech ran longer than {shared.VOICE_FOUNDER_MIN_SECONDS:.0f}s "
            "unbroken, so the voice split is a rough guess — the same person "
            "may appear more than once."
        )
        return labels, talk_time, notes

    founder_vectors = vectors[founder_indices]
    founder_units = [units[i] for i in founder_indices]
    if len(founder_indices) == 1:
        founder_labels = np.zeros(1, dtype=int)
        talk_time = {0: lengths[founder_indices[0]]}
    else:
        founder_labels, talk_time = _cluster_voices(founder_vectors, founder_units)
    founder_labels = np.asarray(founder_labels)

    labels = np.full(len(units), -1, dtype=int)
    for position, index in enumerate(founder_indices):
        labels[index] = founder_labels[position]

    centroids = {
        label: _normalised_centroid(founder_vectors, founder_labels, label)
        for label in talk_time
    }
    unattributed = 0.0
    for i in range(len(units)):
        if labels[i] != -1:
            continue
        best_label, best_similarity = None, shared.VOICE_ATTACH_SIMILARITY
        for label, centroid in centroids.items():
            similarity = float(vectors[i] @ centroid)
            if similarity >= best_similarity:
                best_label, best_similarity = label, similarity
        if best_label is None:
            unattributed += lengths[i]
        else:
            labels[i] = best_label
            talk_time[best_label] += lengths[i]

    if unattributed > 0:
        notes.append(
            f"{unattributed:.0f}s of short speech snippets could not be "
            "matched to a listed voice and were left off the list."
        )
    return labels, talk_time, notes


def analyze_voices(source_path: str, voices_dir: str, reporter: ProgressReporter) -> Dict:
    """Analyze a clip into distinct voices. Returns metadata dict."""
    import json
    import numpy as np
    import soundfile as sf

    spans, duration = _detect_speech(source_path, reporter)
    if not spans:
        raise CleanupError("No speech was detected in this clip.")

    units = _analysis_units(spans)
    usable = [u for u in units
              if (u[1] - u[0]) >= shared.MIN_SPEECH_SEGMENT_SECONDS]
    if not usable:
        raise CleanupError("Too little continuous speech to tell voices apart.")

    vectors = _embed_segments(source_path, usable, reporter)

    labels, talk_time, notes = _found_voices(vectors, usable)

    ordered_labels = sorted(talk_time.keys(),
                           key=lambda l: talk_time[l], reverse=True)
    kept_labels = ordered_labels[:shared.MAX_VOICES_LISTED]
    dropped_count = len(ordered_labels) - len(kept_labels)

    if dropped_count > 0:
        notes.append(f"Found {len(ordered_labels)} voices; listed the top {len(kept_labels)}.")

    os.makedirs(voices_dir, exist_ok=True)

    data, rate = sf.read(source_path, dtype="float32", always_2d=True)

    voices_list = []
    for rank, label in enumerate(kept_labels, 1):
        centroid = _normalised_centroid(vectors, labels, label)
        label_spans = [usable[i] for i, l in enumerate(labels) if l == label]
        longest_span = max(label_spans, key=lambda s: s[1] - s[0])

        preview_duration = min(shared.VOICE_PREVIEW_SECONDS, longest_span[1] - longest_span[0])
        span_midpoint = (longest_span[0] + longest_span[1]) / 2
        preview_start = max(longest_span[0], span_midpoint - preview_duration / 2)
        preview_end = min(preview_start + preview_duration, longest_span[1])
        if preview_end - preview_start < preview_duration:
            preview_start = max(longest_span[0], preview_end - preview_duration)

        preview_path = os.path.join(voices_dir, f"preview_{rank}.wav")
        preview_chunk = data[int(preview_start * rate):int(preview_end * rate)]
        sf.write(preview_path, preview_chunk, rate, subtype="FLOAT")

        total_talk_time = sum(s[1] - s[0] for s in label_spans)
        # Share of the SPEECH, not of the clip: the UI says "% of the speech"
        # and silence should not dilute it.
        all_speech = sum(talk_time.values())
        share = total_talk_time / all_speech if all_speech > 0 else 0.0

        voices_list.append({
            "id": rank,
            "talk_seconds": float(total_talk_time),
            "share": float(share),
            "preview": os.path.abspath(preview_path),
            "centroid": centroid.tolist(),
        })

    processed_path = os.path.join(voices_dir, shared.VOICES_PROCESSED_FILENAME)
    _copy_audio(source_path, processed_path)

    metadata_path = os.path.join(voices_dir, shared.VOICES_METADATA_FILENAME)
    metadata = {
        "duration": float(duration),
        "processed": os.path.abspath(processed_path),
        "voices": voices_list,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "ok": True,
        "output": os.path.abspath(processed_path),
        "voices_dir": os.path.abspath(voices_dir),
        "duration": float(duration),
        "notes": notes,
        "voices": voices_list,
    }


# --------------------------------------------------------------------------
# Trim and normalise (ffmpeg)
# --------------------------------------------------------------------------

def finalise(source_path: str, destination_path: str, trim: bool,
             normalize: bool, sample_rate: int, channel_mode: str,
             reporter: ProgressReporter) -> None:
    """Write the final file, optionally trimming silence and normalising.

    Both are ffmpeg filters, so they run in one pass. The loudness target is
    taken from webui_media_fetch rather than restated here, so the cleaned file
    matches what the extract tab produces.
    """
    filters: List[str] = []
    if trim:
        filters.append(
            "silenceremove="
            "start_periods=1:start_duration=0:"
            f"start_threshold={shared.TRIM_SILENCE_THRESHOLD_DB}dB:"
            "stop_periods=-1:"
            f"stop_duration={shared.TRIM_MIN_SILENCE_SECONDS}:"
            f"stop_threshold={shared.TRIM_SILENCE_THRESHOLD_DB}dB:"
            f"stop_silence={shared.TRIM_KEEP_SILENCE_SECONDS}:"
            "detection=rms"
        )
    if normalize:
        filters.append(
            f"loudnorm=I={media_fetch.LOUDNORM_TARGET_LUFS}"
            f":TP={media_fetch.LOUDNORM_TRUE_PEAK_DB}"
            f":LRA={media_fetch.LOUDNORM_RANGE_LU}"
        )

    command = [media_fetch.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
               "-i", source_path]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend(["-ar", str(sample_rate)])
    if channel_mode == media_fetch.CHANNEL_MONO:
        command.extend(["-ac", "1"])
    elif channel_mode == media_fetch.CHANNEL_STEREO:
        command.extend(["-ac", "2"])
    command.extend(["-c:a", "pcm_s16le", destination_path])

    if trim:
        reporter.stage(shared.STAGE_TRIM, 0.5, "Trimming silence")
    if normalize:
        reporter.stage(shared.STAGE_NORMALIZE, 0.5, "Normalising loudness")

    result = subprocess.run(
        command, capture_output=True, text=True,
        timeout=media_fetch.SUBPROCESS_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise CleanupError(
            f"ffmpeg failed writing the cleaned file: {result.stderr.strip()}"
        )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def run_pipeline(args, reporter: ProgressReporter) -> Dict:
    """Run every requested stage in canonical order. Returns a result dict."""
    stages = shared.ordered_stages(args.stages.split(","))
    if not stages:
        raise CleanupError("No cleanup stages were selected.")

    work_dir = os.path.join(os.path.dirname(os.path.abspath(args.output)),
                            shared.STAGES_SUBDIR)
    os.makedirs(work_dir, exist_ok=True)

    notes: List[str] = []
    current = os.path.abspath(args.input)
    separation = SeparationRunner(args.model_dir, work_dir)

    separation_models = {
        shared.STAGE_ISOLATE: args.vocal_model,
        shared.STAGE_DEREVERB: args.dereverb_model,
        shared.STAGE_DENOISE: args.denoise_model,
    }

    for stage in stages:
        if stage in separation_models:
            current = separation.run(stage, separation_models[stage], current, reporter)
            notes.append(f"{shared.STAGE_LABELS[stage]}: done")
        elif stage == shared.STAGE_SPEAKER:
            target = os.path.join(work_dir, "speaker_isolated.wav")
            notes.append(keep_one_speaker(
                current, target, args.speaker_mode, args.speaker_sample,
                args.speaker_threshold, reporter,
            ))
            current = target

    finalise(
        current, args.output,
        trim=shared.STAGE_TRIM in stages,
        normalize=shared.STAGE_NORMALIZE in stages,
        sample_rate=args.sample_rate,
        channel_mode=args.channel_mode,
        reporter=reporter,
    )

    if not args.keep_intermediates:
        _clear_work_dir(work_dir, keep=os.path.abspath(args.output))

    return {"ok": True, "output": os.path.abspath(args.output), "notes": notes}


def _clear_work_dir(work_dir: str, keep: str) -> None:
    """Delete stage intermediates, reporting anything that will not delete."""
    for name in os.listdir(work_dir):
        path = os.path.join(work_dir, name)
        if os.path.abspath(path) == keep:
            continue
        try:
            os.remove(path)
        except OSError as exc:
            print(f"Could not remove intermediate {path}: {exc}", file=sys.stderr)
    try:
        os.rmdir(work_dir)
    except OSError:
        pass  # still holds files the user asked to keep, or a locked handle


def run_analyze(args, reporter: ProgressReporter) -> Dict:
    """Analyze clip into distinct voices; run only separation stages."""
    stages = shared.ordered_stages(args.stages.split(","))
    separation_stages = [s for s in stages if s in (shared.STAGE_ISOLATE, shared.STAGE_DEREVERB, shared.STAGE_DENOISE)]

    work_dir = os.path.join(args.voices_dir, shared.STAGES_SUBDIR)
    os.makedirs(work_dir, exist_ok=True)

    current = os.path.abspath(args.input)
    separation = SeparationRunner(args.model_dir, work_dir)

    separation_models = {
        shared.STAGE_ISOLATE: args.vocal_model,
        shared.STAGE_DEREVERB: args.dereverb_model,
        shared.STAGE_DENOISE: args.denoise_model,
    }

    for stage in separation_stages:
        current = separation.run(stage, separation_models[stage], current, reporter)

    # analyze_voices writes processed.wav into voices_dir itself; the work dir
    # (where `current` may live) is only cleared after that copy exists.
    result = analyze_voices(current, args.voices_dir, reporter)
    _clear_work_dir(work_dir, keep=result["output"])
    return result


def run_extract(args, reporter: ProgressReporter) -> Dict:
    """Extract one voice from analyzed clip."""
    import json
    import numpy as np

    voices_json_path = os.path.join(args.voices_dir, shared.VOICES_METADATA_FILENAME)
    try:
        with open(voices_json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError as exc:
        raise CleanupError(f"Voices metadata not found: {voices_json_path}") from exc

    voices = metadata.get("voices", [])
    voice_ids = [v["id"] for v in voices]
    if args.voice_id not in voice_ids:
        raise CleanupError(
            f"Voice ID {args.voice_id} not found. Available: {voice_ids}"
        )

    voice = next(v for v in voices if v["id"] == args.voice_id)
    target = np.asarray(voice["centroid"])

    source = os.path.join(args.voices_dir, shared.VOICES_PROCESSED_FILENAME)
    if not os.path.exists(source):
        raise CleanupError("Process a clip in manual mode first.")

    spans, duration = _detect_speech(source, reporter)

    work_dir = os.path.join(args.voices_dir, shared.STAGES_SUBDIR)
    os.makedirs(work_dir, exist_ok=True)

    isolated_path = os.path.join(work_dir, "isolated.wav")
    kept_seconds, removed, scored = _isolate_target(
        source, isolated_path, target, args.speaker_threshold, spans, duration, reporter
    )

    ref_start, ref_end = _best_reference_span(scored, args.speaker_threshold, duration)
    reference_path = os.path.join(work_dir, "reference.wav")
    _concatenate(source, [(ref_start, ref_end)], reference_path)

    finalise(
        isolated_path, args.output,
        trim=shared.STAGE_TRIM in shared.ordered_stages(args.stages.split(",")),
        normalize=shared.STAGE_NORMALIZE in shared.ordered_stages(args.stages.split(",")),
        sample_rate=args.sample_rate,
        channel_mode=args.channel_mode,
        reporter=reporter,
    )

    finalise(
        reference_path, args.reference_output,
        trim=False,
        normalize=shared.STAGE_NORMALIZE in shared.ordered_stages(args.stages.split(",")),
        sample_rate=args.sample_rate,
        channel_mode=args.channel_mode,
        reporter=reporter,
    )

    _clear_work_dir(work_dir, keep="")

    return {
        "ok": True,
        "output": os.path.abspath(args.output),
        "reference": os.path.abspath(args.reference_output),
        "notes": [
            f"Kept {kept_seconds:.0f}s of voice {voice['id']}; "
            f"removed {removed:.0f}s of other voices.",
            f"Reference clip: {ref_end - ref_start:.1f}s starting at {ref_start:.1f}s.",
        ],
    }


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Clean up an audio clip.")
    parser.add_argument("--mode", choices=("clean", "analyze", "extract"),
                        default="clean")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stages", required=True,
                        help="Comma-separated stage keys; order is ignored.")
    parser.add_argument("--progress-file", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default=shared.DEFAULT_DEVICE)
    parser.add_argument("--vocal-model", default=shared.DEFAULT_VOCAL_MODEL)
    parser.add_argument("--dereverb-model", default=shared.DEFAULT_DEREVERB_MODEL)
    parser.add_argument("--denoise-model", default=shared.DEFAULT_DENOISE_MODEL)
    parser.add_argument("--speaker-mode", default=shared.DEFAULT_SPEAKER_MODE,
                        choices=shared.SPEAKER_MODES)
    parser.add_argument("--speaker-sample", default=None)
    parser.add_argument("--speaker-threshold", type=float,
                        default=shared.DEFAULT_SPEAKER_THRESHOLD)
    parser.add_argument("--sample-rate", type=int,
                        default=media_fetch.DEFAULT_SAMPLE_RATE)
    parser.add_argument("--channel-mode", default=media_fetch.DEFAULT_CHANNEL_MODE)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--voices-dir", default=None)
    parser.add_argument("--voice-id", type=int, default=None)
    parser.add_argument("--reference-output", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Must happen before torch is imported anywhere, which is why every heavy
    # import in this file is inside a function.
    if args.device == shared.DEVICE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        # A "cuda:N" pick narrows the visible set to that one card, so the
        # hardcoded cuda:0 further down means "the chosen card". The indices
        # follow the same default CUDA ordering the app's picker enumerates.
        index = shared.cuda_index(args.device)
        if index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
    os.environ["AUDIO_CLEANUP_MODEL_DIR"] = args.model_dir

    reporter = ProgressReporter(args.progress_file, args.stages.split(","))
    try:
        if args.mode == "clean":
            result = run_pipeline(args, reporter)
        elif args.mode == "analyze":
            result = run_analyze(args, reporter)
        elif args.mode == "extract":
            result = run_extract(args, reporter)
        else:
            raise CleanupError(f"Unknown mode: {args.mode}")
    except CleanupError as exc:
        reporter.note(str(exc))
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    except Exception as exc:  # noqa: BLE001 - reported in full, never swallowed
        detail = f"Unexpected {type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
        reporter.note(detail)
        print(json.dumps({"ok": False, "error": detail}))
        return 1
    finally:
        reporter.close()

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
