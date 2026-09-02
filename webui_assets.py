"""Static UI assets for the VoiceForge web UI.

Page chrome only -- title, favicon path, injected <head> markup, the file-type
filter for the media picker, the caption timing help text, and the stylesheet.
Nothing here imports gradio or touches the engine, so it stays cheap to import
and easy to edit without reasoning about the app.

Split out of webui.py, where these 318 lines of markup sat between the
generation helpers and made the module hard to read.
"""

import os

current_dir = os.path.dirname(os.path.abspath(__file__))

APP_TITLE = "Index TTS2 Premium SECourses App"
APP_ASSETS_DIR = os.path.join(current_dir, "ui_assets")
APP_FAVICON_PATH = os.path.join(APP_ASSETS_DIR, "indextts_premium_favicon.svg")
APP_HEAD = """
<meta name="theme-color" content="#a11236">
<script>
(() => {
  let sectionCountTimer = null;

  function scheduleSectionCountRefresh() {
    const signal = document.querySelector("#section-count-refresh-signal textarea, #section-count-refresh-signal input");
    if (!signal) {
      return;
    }
    if (sectionCountTimer) {
      clearTimeout(sectionCountTimer);
    }
    sectionCountTimer = setTimeout(() => {
      signal.value = String(Date.now());
      signal.dispatchEvent(new Event("input", { bubbles: true }));
      signal.dispatchEvent(new Event("change", { bubbles: true }));
    }, 500);
  }

  document.addEventListener("input", (event) => {
    const target = event.target;
    if (!target) {
      return;
    }
    if (target.closest("#input-text-source") || target.closest("#max-tokens-segment-source")) {
      scheduleSectionCountRefresh();
    }
  }, true);
})();
</script>
"""
MEDIA_FILE_TYPES = [
    # The "audio" and "video" shorthands become audio/* and video/* in the
    # picker's accept attribute. Android resolves accept by MIME type and
    # largely ignores bare extensions, so an extension-only list greys out
    # perfectly valid files on a phone. Extensions stay for desktop browsers,
    # which do honour them.
    "audio", "video",
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".opus",
]
CAPTION_TIMING_HELP = """
**What cue timing does**

The app generates separate caption timing units, then auto-retimes each finished unit to the matching caption duration before timeline assembly. In most files, each caption block becomes its own unit. If cues overlap, overlapping cues are merged into a larger timing unit first.

**Example**

`00:00:01.000 --> 00:00:03.000   Hello there.`

`00:00:04.500 --> 00:00:06.000   Welcome back.`

With cue timing **off**, the app treats the text like normal paragraphs and decides pacing on its own.

With cue timing **on**, `Hello there.` is generated as its own timing unit, retimed to fit the `2.0s` cue slot, the `1.5s` gap is preserved, and `Welcome back.` starts at `4.5s`.

**Impact on the result**

This is useful for subtitle-aligned narration, dubbing, and scene-matched timing. Because each finished timing unit is retimed to its target slot before assembly, subtitle timing stays aligned much more reliably than the old cue-fitting approach. If cues overlap, they are synthesized as merged timing units so the final timeline still matches the caption file structure.
"""
APP_CSS = """
.top-input-panel {
    border: 0 !important;
    border-radius: 0;
    padding: 0;
    background: transparent !important;
    box-shadow: none !important;
}

.ui-hidden-signal {
    display: none !important;
}

.top-input-panel > div {
    border: 1px solid var(--block-border-color, rgba(255, 255, 255, 0.08)) !important;
    border-radius: var(--radius-lg, 18px) !important;
    padding: 0.95rem !important;
    background: var(--block-background-fill, transparent) !important;
    box-shadow: none !important;
}

.top-input-panel h3,
.top-input-panel .prose h3 {
    margin-top: 0 !important;
    margin-bottom: 0.8rem !important;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--block-border-color, rgba(255, 255, 255, 0.08));
    color: var(--body-text-color, inherit) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

.caption-timing-help > div,
.caption-timing-help .prose {
    padding: 0.45rem 0 0.3rem !important;
}

.reference-subsection-title > div,
.reference-subsection-title .prose {
    padding: 0.1rem 0 0.35rem !important;
}

.reference-subsection h4,
.reference-subsection-title h4,
.reference-subsection-title .prose h4 {
    color: var(--body-text-color, inherit) !important;
    font-weight: 700 !important;
    margin: 0 !important;
}

.reference-subsection {
    margin: 0 0 0.9rem;
}

.reference-subsection > div {
    border: 1px solid var(--block-border-color, rgba(255, 255, 255, 0.08)) !important;
    border-radius: calc(var(--radius-lg, 18px) - 4px) !important;
    padding: 0.85rem !important;
    background: var(--background-fill-secondary, transparent) !important;
    box-shadow: none !important;
}

.top-section-flat {
    border-width: 0 !important;
    border-style: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

.top-section-flat > .wrap,
.top-section-flat > .block,
.top-section-flat .block {
    border-width: 0 !important;
    border-style: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

:is(button.action-button, .action-button button) {
    position: relative;
    overflow: hidden;
    min-height: 48px;
    border-radius: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    color: #fdf8ff !important;
    font-weight: 700 !important;
    letter-spacing: 0.01em;
    text-shadow: 0 1px 0 rgba(15, 23, 42, 0.28);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}

:is(button.action-button, .action-button button)::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 18%, rgba(255, 255, 255, 0.22) 38%, transparent 56%);
    transform: translateX(-160%);
    transition: transform 0.55s ease;
    pointer-events: none;
}

:is(button.action-button, .action-button button):hover {
    transform: translateY(-1px);
    filter: saturate(1.06) brightness(1.03);
}

:is(button.action-button, .action-button button):hover::before {
    transform: translateX(160%);
}

:is(button.action-button, .action-button button):active {
    transform: translateY(1px);
}

:is(button.action-button, .action-button button):focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.82);
    outline-offset: 2px;
}

:is(button#extract-audio-button, #extract-audio-button button) {
    background: linear-gradient(180deg, #ffd4a4 0%, #ffb05f 18%, #e67c26 58%, #9a4a0d 100%) !important;
    border-color: rgba(160, 77, 14, 0.65) !important;
    box-shadow:
        0 14px 28px rgba(230, 124, 38, 0.28),
        0 1px 0 rgba(255, 247, 234, 0.34) inset,
        0 -3px 0 rgba(113, 50, 5, 0.28) inset !important;
}

:is(button#load-audio-button, #load-audio-button button) {
    background: linear-gradient(180deg, #9df2db 0%, #47d6ab 16%, #0f9f7f 56%, #0a5f4e 100%) !important;
    border-color: rgba(8, 100, 82, 0.65) !important;
    box-shadow:
        0 14px 28px rgba(15, 159, 127, 0.25),
        0 1px 0 rgba(229, 255, 248, 0.34) inset,
        0 -3px 0 rgba(5, 73, 60, 0.28) inset !important;
}

:is(button#generate-speech-button, #generate-speech-button button) {
    min-height: 54px;
    letter-spacing: 0.03em;
    color: #ffe7eb !important;
    text-shadow:
        0 0 8px rgba(255, 222, 228, 0.65),
        0 0 18px rgba(255, 131, 157, 0.48),
        0 1px 0 rgba(107, 13, 35, 0.9);
    background:
        radial-gradient(circle at 18% 0%, rgba(255, 197, 210, 0.36), transparent 34%),
        linear-gradient(180deg, #ff9aae 0%, #ff6383 16%, #d91f4d 55%, #7f102c 100%) !important;
    border-color: rgba(135, 17, 48, 0.72) !important;
    box-shadow:
        0 0 0 1px rgba(255, 180, 196, 0.12),
        0 16px 34px rgba(217, 31, 77, 0.34),
        0 0 26px rgba(255, 77, 116, 0.26),
        0 1px 0 rgba(255, 234, 239, 0.34) inset,
        0 -3px 0 rgba(95, 9, 31, 0.34) inset !important;
    animation: premium-button-glow 2.8s ease-in-out infinite;
}

:is(button#generate-speech-button, #generate-speech-button button)::before {
    background: linear-gradient(120deg, transparent 15%, rgba(255, 255, 255, 0.28) 36%, transparent 58%);
    animation: premium-button-sheen 3.6s ease-in-out infinite;
}

:is(button#generate-speech-button, #generate-speech-button button):hover {
    box-shadow:
        0 0 0 1px rgba(255, 188, 202, 0.18),
        0 20px 38px rgba(217, 31, 77, 0.42),
        0 0 34px rgba(255, 77, 116, 0.34),
        0 1px 0 rgba(255, 238, 242, 0.38) inset,
        0 -3px 0 rgba(95, 9, 31, 0.38) inset !important;
}

:is(button#generate-speech-button, #generate-speech-button button):active {
    animation-play-state: paused;
}

:is(button#open-outputs-button, #open-outputs-button button) {
    background: linear-gradient(180deg, #b5e8ff 0%, #69c8ff 18%, #238cd8 58%, #12518d 100%) !important;
    border-color: rgba(19, 85, 145, 0.68) !important;
    box-shadow:
        0 14px 28px rgba(35, 140, 216, 0.26),
        0 1px 0 rgba(234, 248, 255, 0.34) inset,
        0 -3px 0 rgba(14, 60, 106, 0.28) inset !important;
}

:is(button#preset-save-button, #preset-save-button button) {
    background: linear-gradient(180deg, #d6bcff 0%, #b084ff 18%, #7a41d8 58%, #4c1f96 100%) !important;
    border-color: rgba(80, 31, 151, 0.68) !important;
    box-shadow:
        0 14px 28px rgba(122, 65, 216, 0.27),
        0 1px 0 rgba(246, 239, 255, 0.34) inset,
        0 -3px 0 rgba(60, 20, 118, 0.28) inset !important;
}

:is(button#preset-load-button, #preset-load-button button) {
    background: linear-gradient(180deg, #c1cbff 0%, #8ea2ff 18%, #4c65e2 58%, #2c3a97 100%) !important;
    border-color: rgba(42, 58, 151, 0.7) !important;
    box-shadow:
        0 14px 28px rgba(76, 101, 226, 0.26),
        0 1px 0 rgba(241, 244, 255, 0.34) inset,
        0 -3px 0 rgba(28, 40, 112, 0.3) inset !important;
}

:is(button#preset-reset-button, #preset-reset-button button) {
    background: linear-gradient(180deg, #ffe9b0 0%, #ffd463 18%, #e0a61f 58%, #8f6200 100%) !important;
    border-color: rgba(145, 99, 1, 0.68) !important;
    color: #fffdf5 !important;
    box-shadow:
        0 14px 28px rgba(224, 166, 31, 0.26),
        0 1px 0 rgba(255, 251, 231, 0.34) inset,
        0 -3px 0 rgba(109, 74, 2, 0.28) inset !important;
}

:is(button#preset-delete-button, #preset-delete-button button) {
    background: linear-gradient(180deg, #ffbdd1 0%, #ff7aa2 18%, #d62f6b 58%, #7b163d 100%) !important;
    border-color: rgba(125, 20, 62, 0.72) !important;
    box-shadow:
        0 14px 28px rgba(214, 47, 107, 0.28),
        0 1px 0 rgba(255, 238, 244, 0.34) inset,
        0 -3px 0 rgba(92, 10, 43, 0.32) inset !important;
}

@keyframes premium-button-glow {
    0%, 100% {
        box-shadow:
            0 14px 30px rgba(226, 58, 94, 0.26),
            0 1px 0 rgba(255, 255, 255, 0.35) inset,
            0 -3px 0 rgba(104, 10, 30, 0.32) inset;
    }
    50% {
        box-shadow:
            0 18px 38px rgba(226, 58, 94, 0.4),
            0 1px 0 rgba(255, 255, 255, 0.38) inset,
            0 -3px 0 rgba(104, 10, 30, 0.36) inset;
    }
}

@keyframes premium-button-sheen {
    0%, 100% {
        transform: translateX(-150%);
    }
    45%, 55% {
        transform: translateX(150%);
    }
}

@media (prefers-reduced-motion: reduce) {
    :is(button#generate-speech-button, #generate-speech-button button),
    :is(button#generate-speech-button, #generate-speech-button button)::before {
        animation: none !important;
    }
}
"""
