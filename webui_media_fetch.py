"""Media fetch + audio extraction helpers for the IndexTTS2 web UI.

Downloads a video/audio stream from YouTube or any other site yt-dlp supports,
then transcodes the audio track into a format suitable for use as a TTS
reference voice.

This module is standalone: it imports nothing from webui.py.

REQUIRES gradio 6.17.3. Gradio 6.11.0 has an upstream regression where
switching to a tab whose contents were hidden sends Svelte into an infinite
effect loop (effect_update_depth_exceeded) and locks the browser tab up:
https://github.com/gradio-app/gradio/issues/13285
The tab this module backs is the one that triggers it. 6.17.3 is verified
clear of it; re-test tab switching before bumping gradio.

Do not jump to 6.26: it pulls huggingface-hub 1.29, which breaks
transformers 4.52.1 at model load while the app still serves a healthy
looking UI. 6.17.3 is the newest gradio that keeps huggingface-hub<1.0.
"""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# Constants -- every tunable literal lives here, nothing is inlined below.
# --------------------------------------------------------------------------

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

# Sample rates offered in the UI. 24000 is IndexTTS2's native rate.
SAMPLE_RATES: List[int] = [16000, 22050, 24000, 32000, 44100, 48000]
DEFAULT_SAMPLE_RATE = 24000

CHANNEL_MONO = "mono"
CHANNEL_STEREO = "stereo"
CHANNEL_SOURCE = "keep source"
CHANNEL_MODES: List[str] = [CHANNEL_MONO, CHANNEL_STEREO, CHANNEL_SOURCE]
DEFAULT_CHANNEL_MODE = CHANNEL_MONO

# Loudness normalisation target (EBU R128), used when normalise is enabled.
LOUDNORM_TARGET_LUFS = -16.0
LOUDNORM_TRUE_PEAK_DB = -1.5
LOUDNORM_RANGE_LU = 11.0

SUBPROCESS_TIMEOUT_SECONDS = 3600
FFPROBE_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class AudioFormat:
    """One selectable output format in the UI dropdown."""

    key: str            # stable id used by the UI
    label: str          # human readable dropdown entry
    extension: str      # file extension without the leading dot
    encoder: str        # ffmpeg encoder name, e.g. "libmp3lame"
    extra_args: Tuple[str, ...] = ()   # encoder-specific ffmpeg flags


# Ordered: the reference-voice friendly lossless entries first.
AUDIO_FORMATS: Tuple[AudioFormat, ...] = (
    AudioFormat("wav_pcm16", "WAV 16-bit PCM (recommended reference)", "wav", "pcm_s16le"),
    AudioFormat("wav_pcm24", "WAV 24-bit PCM", "wav", "pcm_s24le"),
    AudioFormat("wav_f32", "WAV 32-bit float", "wav", "pcm_f32le"),
    AudioFormat("flac", "FLAC (lossless, compressed)", "flac", "flac", ("-compression_level", "5")),
    AudioFormat("mp3_320", "MP3 320 kbps", "mp3", "libmp3lame", ("-b:a", "320k")),
    AudioFormat("mp3_192", "MP3 192 kbps", "mp3", "libmp3lame", ("-b:a", "192k")),
    AudioFormat("m4a_aac", "M4A / AAC 256 kbps", "m4a", "aac", ("-b:a", "256k")),
    AudioFormat("opus", "Opus 128 kbps", "opus", "libopus", ("-b:a", "128k")),
    AudioFormat("ogg_vorbis", "OGG Vorbis q6", "ogg", "libvorbis", ("-q:a", "6")),
)

FORMATS_BY_KEY: Dict[str, AudioFormat] = {fmt.key: fmt for fmt in AUDIO_FORMATS}
DEFAULT_FORMAT_KEY = "wav_pcm16"

# Characters that are unsafe in a Windows filename.
UNSAFE_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
FILENAME_MAX_LENGTH = 120


class MediaFetchError(RuntimeError):
    """Raised when a download or transcode step fails. Never swallowed."""


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------

PHASE_RESOLVE = "resolve"
PHASE_DOWNLOAD = "download"
PHASE_EXTRACT = "extract"
PHASE_DONE = "done"

# (start, end) of the overall 0..1 bar that each phase owns. Downloading is by
# far the longest leg, so it gets most of the bar.
PHASE_SPANS: Dict[str, Tuple[float, float]] = {
    PHASE_RESOLVE: (0.00, 0.05),
    PHASE_DOWNLOAD: (0.05, 0.75),
    PHASE_EXTRACT: (0.75, 0.97),
    PHASE_DONE: (0.97, 1.00),
}


@dataclass(frozen=True)
class FetchProgress:
    """One progress event emitted while fetching and transcoding.

    `fraction` is progress across the WHOLE pipeline (0..1), already mapped
    through PHASE_SPANS, or None when this phase cannot measure itself.
    """

    phase: str
    message: str
    fraction: Optional[float] = None


ProgressCallback = Callable[[FetchProgress], None]


def _phase_fraction(phase: str, within_phase: float) -> float:
    """Map 0..1 progress inside `phase` onto the overall 0..1 bar."""
    start, end = PHASE_SPANS.get(phase, (0.0, 1.0))
    clamped = min(max(within_phase, 0.0), 1.0)
    return start + (end - start) * clamped


# --------------------------------------------------------------------------
# Environment probes
# --------------------------------------------------------------------------

def ffmpeg_available() -> bool:
    """Return True when the ffmpeg binary is resolvable on PATH."""
    return shutil.which(FFMPEG_BIN) is not None


def available_encoders() -> set:
    """Return the set of ffmpeg audio encoder names this build supports.

    Runs `ffmpeg -hide_banner -encoders`, parses the table, and collects the
    encoder name column for rows whose capability flags begin with "A"
    (audio). Returns an empty set when ffmpeg is missing or the call fails --
    callers treat an empty set as "unknown, do not filter".
    """
    try:
        result = subprocess.run(
            [FFMPEG_BIN, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return set()

    encoders = set()
    lines = result.stdout.splitlines()
    in_table = False
    for line in lines:
        if line.startswith(" ------"):
            in_table = True
            continue
        if not in_table:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("A"):
            encoders.add(parts[1])
    return encoders


def usable_formats() -> List[AudioFormat]:
    """Return the AUDIO_FORMATS entries this ffmpeg build can actually encode.

    When available_encoders() returns an empty set (probe failed), return the
    full AUDIO_FORMATS list unchanged rather than hiding everything.
    """
    encoders = available_encoders()
    if not encoders:
        return list(AUDIO_FORMATS)
    return [fmt for fmt in AUDIO_FORMATS if fmt.encoder in encoders]


# --------------------------------------------------------------------------
# Filename helpers
# --------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Strip characters that are illegal in a Windows filename.

    Replaces each unsafe character with an underscore, collapses runs of
    whitespace to a single space, strips leading/trailing dots and spaces, and
    truncates to FILENAME_MAX_LENGTH. Returns "media" when nothing survives.
    """
    name = re.sub(UNSAFE_FILENAME_CHARS, "_", name)
    name = re.sub(r'\s+', ' ', name)
    name = name.strip(" .")
    name = name[:FILENAME_MAX_LENGTH].strip(" .")
    return name if name else "media"


def unique_path(directory: str, stem: str, extension: str) -> str:
    """Return a path inside `directory` that does not yet exist.

    First candidate is "<stem>.<extension>"; on collision append "_2", "_3"
    and so on until a free name is found. `extension` has no leading dot.
    """
    candidate = f"{stem}.{extension}"
    path = os.path.join(directory, candidate)
    counter = 2
    while os.path.exists(path):
        candidate = f"{stem}_{counter}.{extension}"
        path = os.path.join(directory, candidate)
        counter += 1
    return path


# --------------------------------------------------------------------------
# ffmpeg command construction
# --------------------------------------------------------------------------

def build_ffmpeg_command(
    source_path: str,
    destination_path: str,
    audio_format: AudioFormat,
    sample_rate: int,
    channel_mode: str,
    start_seconds: Optional[float],
    end_seconds: Optional[float],
    normalize: bool,
    gain_db: float,
) -> List[str]:
    """Build the ffmpeg argv list that transcodes `source_path`.

    Rules:
      * Always start with [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y"].
      * A start offset is an INPUT seek: "-ss <start>" placed BEFORE "-i".
      * An end offset is "-to <end>", also placed BEFORE "-i". Both are
        input-side, so the range is absolute in the source timeline.
      * Skip "-ss" when start_seconds is None or 0. Skip "-to" when
        end_seconds is None or not greater than start_seconds.
      * Then "-i", source_path.
      * "-vn" to drop any video stream.
      * Audio filters are joined with "," into a single "-af" argument, in this
        order, skipping any that are not requested:
          - loudnorm=I=<LOUDNORM_TARGET_LUFS>:TP=<LOUDNORM_TRUE_PEAK_DB>:LRA=<LOUDNORM_RANGE_LU>
            when `normalize` is True
          - volume=<gain_db>dB when gain_db is non-zero
        Emit no "-af" flag at all when the filter list is empty.
      * "-ar", str(sample_rate).
      * "-ac", "1" for CHANNEL_MONO, "-ac", "2" for CHANNEL_STEREO, and no
        "-ac" flag at all for CHANNEL_SOURCE.
      * "-c:a", audio_format.encoder, then *audio_format.extra_args.
      * destination_path last.

    Returns the argv list. Does not run anything.
    """
    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y"]

    if start_seconds is not None and start_seconds != 0:
        cmd.extend(["-ss", str(start_seconds)])
    if end_seconds is not None and (start_seconds is None or end_seconds > start_seconds):
        cmd.extend(["-to", str(end_seconds)])

    cmd.extend(["-i", source_path])
    cmd.append("-vn")

    filters = []
    if normalize:
        filters.append(
            f"loudnorm=I={LOUDNORM_TARGET_LUFS}:TP={LOUDNORM_TRUE_PEAK_DB}:LRA={LOUDNORM_RANGE_LU}"
        )
    if gain_db != 0:
        filters.append(f"volume={gain_db}dB")

    if filters:
        cmd.extend(["-af", ",".join(filters)])

    cmd.extend(["-ar", str(sample_rate)])

    if channel_mode == CHANNEL_MONO:
        cmd.extend(["-ac", "1"])
    elif channel_mode == CHANNEL_STEREO:
        cmd.extend(["-ac", "2"])

    cmd.extend(["-c:a", audio_format.encoder])
    cmd.extend(audio_format.extra_args)
    cmd.append(destination_path)

    return cmd


def run_ffmpeg(command: List[str]) -> None:
    """Run an ffmpeg argv list, raising MediaFetchError on failure.

    Uses subprocess.run with capture_output=True, text=True and
    timeout=SUBPROCESS_TIMEOUT_SECONDS. On a non-zero return code raise
    MediaFetchError whose message contains the last 20 lines of stderr. On
    subprocess.TimeoutExpired raise MediaFetchError naming the timeout.
    Never swallow an exception silently.
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        raise MediaFetchError(f"ffmpeg timed out after {SUBPROCESS_TIMEOUT_SECONDS} seconds")

    if result.returncode != 0:
        stderr_lines = result.stderr.splitlines()
        last_20 = "\n".join(stderr_lines[-20:])
        raise MediaFetchError(f"ffmpeg failed with return code {result.returncode}: {last_20}")


def probe_duration_seconds(path: str) -> Optional[float]:
    """Return the duration of `path` in seconds via ffprobe, or None.

    Runs ffprobe with:
      -v error -show_entries format=duration -of json <path>
    Parses the JSON and returns float(data["format"]["duration"]). Returns
    None when ffprobe is missing, the call fails, or the field is absent --
    a missing duration is informational only, so it must not raise.
    """
    probe_path = shutil.which(FFPROBE_BIN)
    if probe_path is None:
        return None

    try:
        result = subprocess.run(
            [probe_path, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    if result.returncode != 0:
        return None

    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


# --------------------------------------------------------------------------
# Download layer (yt-dlp)
# --------------------------------------------------------------------------

YTDLP_INSTALL_HINT = (
    "yt-dlp is not installed in this environment. Install it with:\n"
    "    venv\\Scripts\\python.exe -m pip install -U yt-dlp"
)

DOWNLOAD_AUDIO_ONLY = "Audio only (fastest)"
DOWNLOAD_WITH_VIDEO = "Video + audio (keeps the video file too)"
DOWNLOAD_MODES: List[str] = [DOWNLOAD_AUDIO_ONLY, DOWNLOAD_WITH_VIDEO]
DEFAULT_DOWNLOAD_MODE = DOWNLOAD_AUDIO_ONLY

# yt-dlp format selectors, one per download mode.
FORMAT_SELECTOR_AUDIO = "bestaudio/best"
FORMAT_SELECTOR_VIDEO = "bestvideo+bestaudio/best"
MERGE_CONTAINER = "mp4"

COOKIES_NONE = "none"
COOKIES_BROWSERS: List[str] = [
    COOKIES_NONE,
    "chrome",
    "edge",
    "firefox",
    "brave",
    "chromium",
    "opera",
    "vivaldi",
]

SUPPORTED_URL_SCHEMES = ("http://", "https://")

# Base used when folding "hh:mm:ss" fields down into seconds.
SECONDS_PER_UNIT = 60

# yt-dlp writes into a temporary staging directory before we rename the result.
DOWNLOAD_SUBDIR = "downloads"
EXTRACTED_SUBDIR = "extracted"


def parse_time_to_seconds(value: Any) -> Optional[float]:
    """Parse a trim field into seconds.

    Accepts a bare number of seconds ("83", "12.5"), "mm:ss" ("1:23") or
    "hh:mm:ss" ("1:02:03"). Returns None for empty input, so "no trim" and
    "trim at zero" stay distinguishable. Raises MediaFetchError on garbage.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    parts = text.split(":")
    if len(parts) > 3:
        raise MediaFetchError(f"Could not read the time value: {text}")

    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise MediaFetchError(f"Could not read the time value: {text}") from exc

    total = 0.0
    for number in numbers:
        total = total * SECONDS_PER_UNIT + number
    if total < 0:
        raise MediaFetchError(f"Time value cannot be negative: {text}")
    return total


def _import_ytdlp():
    """Import yt_dlp lazily so a missing dependency does not break webui start.

    Raises MediaFetchError carrying YTDLP_INSTALL_HINT when the import fails.
    """
    try:
        import yt_dlp  # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:
        raise MediaFetchError(f"{YTDLP_INSTALL_HINT}\n\nOriginal error: {exc}") from exc
    return yt_dlp


def ytdlp_version() -> Optional[str]:
    """Return the installed yt-dlp version string, or None when unavailable."""
    try:
        return _import_ytdlp().version.__version__
    except MediaFetchError:
        return None
    except AttributeError:
        return "unknown"


def validate_url(url: str) -> str:
    """Return the trimmed URL, or raise MediaFetchError when it is unusable."""
    cleaned = (url or "").strip()
    if not cleaned:
        raise MediaFetchError("No URL was supplied.")
    if not cleaned.lower().startswith(SUPPORTED_URL_SCHEMES):
        raise MediaFetchError(
            f"URL must start with http:// or https:// -- got: {cleaned[:80]}"
        )
    return cleaned


def _cookie_option(cookies_from_browser: str) -> Dict[str, Any]:
    """Build the yt-dlp cookiesfrombrowser option, empty when not requested."""
    if not cookies_from_browser or cookies_from_browser == COOKIES_NONE:
        return {}
    return {"cookiesfrombrowser": (cookies_from_browser,)}


def probe_media_info(url: str, cookies_from_browser: str = COOKIES_NONE) -> Dict[str, Any]:
    """Fetch metadata for `url` without downloading the media.

    Returns a dict with the keys: title, uploader, duration_seconds, extractor,
    webpage_url. Raises MediaFetchError when extraction fails.
    """
    yt_dlp = _import_ytdlp()
    options: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }
    options.update(_cookie_option(cookies_from_browser))

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
    except Exception as exc:
        raise MediaFetchError(f"Could not read media info: {exc}") from exc

    if info is None:
        raise MediaFetchError("yt-dlp returned no media info for that URL.")
    if "entries" in info and info["entries"]:
        info = info["entries"][0]

    return {
        "title": info.get("title") or "untitled",
        "uploader": info.get("uploader") or info.get("channel") or "unknown",
        "duration_seconds": info.get("duration"),
        "extractor": info.get("extractor_key") or info.get("extractor") or "unknown",
        "webpage_url": info.get("webpage_url") or url,
    }


def download_media(
    url: str,
    destination_dir: str,
    download_mode: str = DEFAULT_DOWNLOAD_MODE,
    cookies_from_browser: str = COOKIES_NONE,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Download `url` into `destination_dir`.

    Returns (downloaded_file_path, media_info_dict). Raises MediaFetchError on
    any failure. `progress_callback` receives FetchProgress events.
    """
    yt_dlp = _import_ytdlp()
    os.makedirs(destination_dir, exist_ok=True)

    def _report(message: str, fraction: Optional[float] = None) -> None:
        if progress_callback is not None:
            progress_callback(FetchProgress(PHASE_DOWNLOAD, message, fraction))

    def _hook(status: Dict[str, Any]) -> None:
        state = status.get("status")
        if state == "downloading":
            done = status.get("downloaded_bytes")
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            fraction = None
            if isinstance(done, (int, float)) and isinstance(total, (int, float)) and total > 0:
                fraction = _phase_fraction(PHASE_DOWNLOAD, min(done / total, 1.0))
            percent = status.get("_percent_str", "").strip()
            speed = status.get("_speed_str", "").strip()
            eta = status.get("_eta_str", "").strip()
            detail = " ".join(part for part in (percent, f"at {speed}" if speed else "",
                                                f"ETA {eta}" if eta else "") if part)
            _report(f"Downloading {detail}".rstrip(), fraction)
        elif state == "finished":
            _report("Download finished, post-processing...",
                    _phase_fraction(PHASE_DOWNLOAD, 1.0))

    wants_video = download_mode == DOWNLOAD_WITH_VIDEO
    options: Dict[str, Any] = {
        "outtmpl": os.path.join(destination_dir, "%(title).100s.%(ext)s"),
        "format": FORMAT_SELECTOR_VIDEO if wants_video else FORMAT_SELECTOR_AUDIO,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_hook],
        "windowsfilenames": True,
        "restrictfilenames": False,
        "overwrites": True,
    }
    if wants_video:
        options["merge_output_format"] = MERGE_CONTAINER
    options.update(_cookie_option(cookies_from_browser))

    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
            if info is None:
                raise MediaFetchError("yt-dlp returned no result for that URL.")
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            downloaded_path = downloader.prepare_filename(info)
    except MediaFetchError:
        raise
    except Exception as exc:
        raise MediaFetchError(f"Download failed: {exc}") from exc

    # prepare_filename reports the pre-merge extension; the merged container
    # may differ, so fall back to whichever sibling file actually exists.
    if not os.path.exists(downloaded_path):
        downloaded_path = _find_actual_download(downloaded_path)

    media_info = {
        "title": info.get("title") or "untitled",
        "uploader": info.get("uploader") or info.get("channel") or "unknown",
        "duration_seconds": info.get("duration"),
        "extractor": info.get("extractor_key") or info.get("extractor") or "unknown",
        "webpage_url": info.get("webpage_url") or url,
    }
    return downloaded_path, media_info


def _find_actual_download(expected_path: str) -> str:
    """Locate the real output file when the merged container renamed it.

    Looks for any sibling sharing the same stem. Raises MediaFetchError when
    nothing matches, because a missing download must never pass silently.
    """
    directory = os.path.dirname(expected_path) or "."
    stem = os.path.splitext(os.path.basename(expected_path))[0]
    if os.path.isdir(directory):
        for candidate in sorted(os.listdir(directory)):
            if os.path.splitext(candidate)[0] == stem:
                return os.path.join(directory, candidate)
    raise MediaFetchError(
        f"Download reported success but no output file was found for: {stem}"
    )


# --------------------------------------------------------------------------
# Top-level pipeline used by the web UI
# --------------------------------------------------------------------------

def fetch_and_extract(
    url: str,
    output_root: str,
    format_key: str = DEFAULT_FORMAT_KEY,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channel_mode: str = DEFAULT_CHANNEL_MODE,
    start_seconds: Optional[float] = None,
    end_seconds: Optional[float] = None,
    normalize: bool = False,
    gain_db: float = 0.0,
    download_mode: str = DEFAULT_DOWNLOAD_MODE,
    cookies_from_browser: str = COOKIES_NONE,
    keep_source_file: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Download `url` and transcode its audio track.

    Returns a dict with keys: audio_path, source_path, title, log (list of str).
    Raises MediaFetchError on any failure. `progress_callback` receives
    FetchProgress events covering the whole pipeline.
    """
    log: List[str] = []

    def _emit(event: FetchProgress) -> None:
        log.append(event.message)
        if progress_callback is not None:
            progress_callback(event)

    def _report(message: str, phase: str = PHASE_RESOLVE,
                within_phase: Optional[float] = None) -> None:
        fraction = None if within_phase is None else _phase_fraction(phase, within_phase)
        _emit(FetchProgress(phase, message, fraction))

    clean_url = validate_url(url)
    if not ffmpeg_available():
        raise MediaFetchError(
            "ffmpeg was not found on PATH. Install it and restart the app."
        )
    audio_format = FORMATS_BY_KEY.get(format_key)
    if audio_format is None:
        raise MediaFetchError(f"Unknown output format: {format_key}")

    download_dir = os.path.join(output_root, DOWNLOAD_SUBDIR)
    extracted_dir = os.path.join(output_root, EXTRACTED_SUBDIR)
    os.makedirs(extracted_dir, exist_ok=True)

    _report(f"Resolving {clean_url}", PHASE_RESOLVE, 0.5)
    source_path, media_info = download_media(
        clean_url,
        download_dir,
        download_mode=download_mode,
        cookies_from_browser=cookies_from_browser,
        # download_media emits FetchProgress objects directly, so it gets the
        # raw sink -- _report builds events, it does not consume them.
        progress_callback=_emit,
    )
    _report(f"Downloaded: {os.path.basename(source_path)}", PHASE_DOWNLOAD, 1.0)

    stem = sanitize_filename(media_info["title"])
    destination = unique_path(extracted_dir, stem, audio_format.extension)

    _report(f"Extracting audio -> {audio_format.label}", PHASE_EXTRACT, 0.1)
    command = build_ffmpeg_command(
        source_path=source_path,
        destination_path=destination,
        audio_format=audio_format,
        sample_rate=sample_rate,
        channel_mode=channel_mode,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        normalize=normalize,
        gain_db=gain_db,
    )
    run_ffmpeg(command)

    if not os.path.exists(destination):
        raise MediaFetchError(f"ffmpeg reported success but produced no file: {destination}")

    _report("Transcode complete.", PHASE_EXTRACT, 1.0)

    duration = probe_duration_seconds(destination)
    if duration is not None:
        _report(f"Output duration: {duration:.2f}s", PHASE_DONE, 0.3)

    if not keep_source_file and os.path.exists(source_path):
        try:
            os.remove(source_path)
            _report("Removed the downloaded source file.", PHASE_DONE, 0.6)
        except OSError as exc:
            _report(f"Could not remove source file: {exc}", PHASE_DONE, 0.6)

    _report(f"Done: {destination}", PHASE_DONE, 1.0)
    return {
        "audio_path": destination,
        "source_path": source_path if os.path.exists(source_path) else None,
        "title": media_info["title"],
        "uploader": media_info["uploader"],
        "duration_seconds": duration,
        "log": log,
    }
