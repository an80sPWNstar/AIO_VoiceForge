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


def _choose_dominant(vectors, spans) -> Tuple[List[int], str]:
    """Cluster the segments and return the indices of the longest-talking one."""
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    if len(spans) == 1:
        return [0], "Only one speech segment; kept it."

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=shared.SPEAKER_CLUSTER_DISTANCE,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(vectors)

    talk_time: Dict[int, float] = {}
    for label, (start, end) in zip(labels, spans):
        talk_time[int(label)] = talk_time.get(int(label), 0.0) + (end - start)

    dominant = max(talk_time, key=talk_time.get)
    kept = [i for i, label in enumerate(labels) if int(label) == dominant]
    total = sum(talk_time.values())
    share = talk_time[dominant] / total if total else 1.0
    note = (
        f"Heard {len(talk_time)} distinct voice(s); kept the one talking "
        f"{talk_time[dominant]:.0f}s ({share * 100:.0f}% of the speech)."
    )
    return kept, note


def _choose_by_sample(vectors, spans, sample_path: str,
                      threshold: float) -> Tuple[List[int], str]:
    """Return the indices of segments matching the supplied reference voice."""
    import numpy as np

    reference = _embed_reference(sample_path)
    scores = vectors @ reference
    kept = [i for i, score in enumerate(scores) if score >= threshold]
    if not kept:
        raise CleanupError(
            "No speech in this clip matched the voice sample "
            f"(best similarity {float(np.max(scores)):.2f}, threshold "
            f"{threshold:.2f}). Lower the match threshold, or check the "
            "sample is the right person."
        )
    matched = sum(spans[i][1] - spans[i][0] for i in kept)
    note = (
        f"Matched {len(kept)} of {len(spans)} segments ({matched:.0f}s) to the "
        f"voice sample; best similarity {float(np.max(scores)):.2f}."
    )
    return kept, note


def _merge_spans(spans: Sequence[Tuple[float, float]],
                 duration: float) -> List[Tuple[float, float]]:
    """Pad, clamp and merge nearly-touching spans into a tidy keep-list."""
    padded = []
    for start, end in spans:
        padded.append((
            max(0.0, start - shared.SEGMENT_PAD_SECONDS),
            min(duration, end + shared.SEGMENT_PAD_SECONDS),
        ))
    padded.sort()

    merged: List[Tuple[float, float]] = []
    for start, end in padded:
        if merged and start - merged[-1][1] <= shared.SEGMENT_MERGE_GAP_SECONDS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
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


def keep_one_speaker(source_path: str, destination_path: str, mode: str,
                     sample_path: Optional[str], threshold: float,
                     reporter: ProgressReporter) -> str:
    """Run the speaker isolation stage. Returns a human-readable note."""
    spans, duration = _detect_speech(source_path, reporter)
    if not spans:
        raise CleanupError(
            "No speech was detected in this clip, so there is no speaker to "
            "isolate. Turn that stage off, or check the earlier stages did not "
            "remove the voice."
        )

    usable = [s for s in spans if (s[1] - s[0]) >= shared.MIN_SPEECH_SEGMENT_SECONDS]
    if len(usable) < 2:
        reporter.stage(shared.STAGE_SPEAKER, 1.0, "Only one voice present")
        _copy_audio(source_path, destination_path)
        return "Too little separate speech to tell voices apart; kept everything."

    vectors = _embed_segments(source_path, usable, reporter)
    if mode == shared.SPEAKER_MODE_SAMPLE:
        if not sample_path:
            raise CleanupError(
                "Speaker mode is 'match a sample' but no sample was supplied."
            )
        kept_indices, note = _choose_by_sample(vectors, usable, sample_path, threshold)
    else:
        kept_indices, note = _choose_dominant(vectors, usable)

    keep = _merge_spans([usable[i] for i in kept_indices], duration)
    reporter.stage(shared.STAGE_SPEAKER, 0.85, "Rebuilding the clip")
    kept_seconds = _concatenate(source_path, keep, destination_path)
    reporter.stage(shared.STAGE_SPEAKER, 1.0, "Speaker isolated")
    return f"{note} Result is {kept_seconds:.0f}s long."


def _copy_audio(source_path: str, destination_path: str) -> None:
    """Copy audio through soundfile so the destination is always a valid WAV."""
    import soundfile as sf
    data, rate = sf.read(source_path, dtype="float32", always_2d=True)
    sf.write(destination_path, data, rate, subtype="FLOAT")


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


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Clean up an audio clip.")
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    # Must happen before torch is imported anywhere, which is why every heavy
    # import in this file is inside a function.
    if args.device == shared.DEVICE_CPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["AUDIO_CLEANUP_MODEL_DIR"] = args.model_dir

    reporter = ProgressReporter(args.progress_file, args.stages.split(","))
    try:
        result = run_pipeline(args, reporter)
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
