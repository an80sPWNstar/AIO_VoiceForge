"""Listen to a generated wav: transcribe it back and measure the sound.

The model can only be judged by ear, and an unattended session has none. This
is the substitute: faster-whisper turns the audio back into text so it can be
compared against what the model was ASKED to say (word error rate), and
librosa measures the qualities a listener would notice without parsing a word
of it -- pace, loudness, silence gaps, clipping.

What it can and cannot claim, so results are read honestly:
  - High WER on clean input text is a real defect: the speech is garbled,
    truncated, or the wrong words. This is the reliable signal.
  - WER ~0 proves intelligibility only. Tone, emotion and naturalness are
    invisible to it; a flat robotic read of the right words scores perfectly.
  - The acoustic numbers are for COMPARISON between runs of the same text,
    not absolute judgment. A pace of 2.5 words/sec means little alone; the
    same sentence dropping from 3.1 to 1.4 between settings means the second
    setting broke something.

Transcription runs in a different venv than analysis: faster-whisper lives in
the speech-service checkout's venv, librosa in the TTS install's. This module
must run under the TTS install's interpreter (same as tools/gate.py) and
shells out to ASR_PYTHON for the transcription half.

  D:\\Index_TTS_v4\\Premium_IndexTTS2_SECourses\\venv\\Scripts\\python.exe \\
      tools\\hear.py path\\to\\out.wav --expect "the text it was asked to say"

Exits 0 and prints one JSON object either way; judgment belongs to the caller.
"""

import argparse
import json
import re
import subprocess
import sys

import numpy as np

# faster-whisper (CPU int8, small.en) lives here; override with --asr-python.
# CPU deliberately: transcription must never compete with generation for VRAM.
ASR_PYTHON = r"E:\vs_code_projects\speech-service\.venv\Scripts\python.exe"

_ASR_SNIPPET = """
import json, sys
from faster_whisper import WhisperModel
model = WhisperModel("small.en", device="cpu", compute_type="int8")
segments, info = model.transcribe(sys.argv[1], beam_size=1)
text = " ".join(s.text.strip() for s in segments).strip()
print(json.dumps({"text": text, "duration": info.duration}))
"""


def transcribe(wav_path, asr_python=ASR_PYTHON):
    """Round-trip the audio through whisper. Raises on a broken ASR setup:
    an inaudible checker must never read as a passing one (same rule as the
    gate's ruff step)."""
    result = subprocess.run(
        [asr_python, "-c", _ASR_SNIPPET, str(wav_path)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ASR failed: {result.stderr.strip()[:400]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def normalize_words(text):
    """Lowercase words, punctuation stripped: WER should count wrong WORDS,
    not whisper's comma placement."""
    return re.findall(r"[a-z0-9']+", text.lower())


def word_error_rate(expected, heard):
    """Levenshtein distance on word lists over the expected length."""
    ref, hyp = normalize_words(expected), normalize_words(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    # One row at a time; texts here are sentences, not books.
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_word != hyp_word),
            ))
        previous = current
    return previous[-1] / len(ref)


def acoustic_metrics(wav_path):
    """The things a listener notices without parsing a word."""
    import librosa  # deferred: transcription-only callers may lack it

    samples, rate = librosa.load(str(wav_path), sr=None, mono=True)
    duration = len(samples) / float(rate)
    if duration == 0.0:
        return {"duration_seconds": 0.0}

    rms = librosa.feature.rms(y=samples)[0]
    frame_seconds = 512 / float(rate)  # librosa's default hop
    # Silence: frames under 1% of the loudest frame. Leading/trailing quiet is
    # normal; a long INTERNAL gap is a stall the ear would flag immediately.
    quiet = rms < (np.max(rms) * 0.01)
    longest_gap = 0
    run = 0
    for is_quiet in quiet:
        run = run + 1 if is_quiet else 0
        longest_gap = max(longest_gap, run)

    f0 = librosa.yin(samples, fmin=60, fmax=500, sr=rate)
    voiced = f0[(f0 > 65) & (f0 < 480)]

    return {
        "duration_seconds": round(duration, 2),
        "rms_mean": round(float(np.mean(rms)), 5),
        "clipping_fraction": round(float(np.mean(np.abs(samples) > 0.999)), 5),
        "longest_silence_seconds": round(longest_gap * frame_seconds, 2),
        "pitch_median_hz": round(float(np.median(voiced)), 1) if voiced.size else None,
        "pitch_spread_hz": round(float(np.percentile(voiced, 90) - np.percentile(voiced, 10)), 1) if voiced.size else None,
    }


def hear(wav_path, expected=None, asr_python=ASR_PYTHON):
    asr = transcribe(wav_path, asr_python)
    report = {"wav": str(wav_path), "heard": asr["text"]}
    report.update(acoustic_metrics(wav_path))
    if expected is not None:
        report["expected"] = expected
        report["wer"] = round(word_error_rate(expected, asr["text"]), 3)
        words = normalize_words(asr["text"])
        speech_seconds = max(0.1, report["duration_seconds"])
        report["words_per_second"] = round(len(words) / speech_seconds, 2)
    return report


def main():
    parser = argparse.ArgumentParser(description="Transcribe and measure a generated wav.")
    parser.add_argument("wav")
    parser.add_argument("--expect", default=None, help="The text the model was asked to say.")
    parser.add_argument("--asr-python", default=ASR_PYTHON)
    args = parser.parse_args()
    print(json.dumps(hear(args.wav, args.expect, args.asr_python), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
