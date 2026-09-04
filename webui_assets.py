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

APP_TITLE = "AIO VoiceForge"
APP_ASSETS_DIR = os.path.join(current_dir, "ui_assets")
APP_FAVICON_PATH = os.path.join(APP_ASSETS_DIR, "indextts_premium_favicon.svg")
APP_HEAD = """
<meta name="theme-color" content="#d1310b">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700;12..96,800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
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

/* ------------------------------------------------------------------ */
/* The console-key system.                                             */
/*                                                                     */
/* One quiet key treatment for every action, with a thin colored edge  */
/* naming its function group -- intake, files, presets, destructive -- */
/* and ONE hot key: Generate Speech, the forge strike. The page spends */
/* all of its heat in that single element; everything else is tool     */
/* steel. (This replaced seven unrelated candy gradients.)             */
/* ------------------------------------------------------------------ */

:root {
    --vf-ember-hi: #ff7a3d;
    --vf-ember: #e6470f;
    --vf-ember-lo: #a02804;
    --vf-key-top: #fdfcfa;
    --vf-key-bottom: #eceae5;
    --vf-key-border: rgba(60, 56, 50, 0.28);
    --vf-key-text: #2b2a27;
    --vf-edge: transparent;
    --vf-edge-intake: #0f9f7f;
    --vf-edge-files: #2c7fd8;
    --vf-edge-presets: #7a5cd6;
    --vf-edge-warn: #c8912a;
    --vf-danger: #c23b52;
}

.dark {
    --vf-key-top: #3d424b;
    --vf-key-bottom: #262a31;
    --vf-key-border: rgba(255, 255, 255, 0.14);
    --vf-key-text: #eef0f3;
}

:is(button.action-button, .action-button button) {
    position: relative;
    overflow: hidden;
    min-height: 48px;
    border-radius: 12px !important;
    border: 1px solid var(--vf-key-border) !important;
    border-bottom-width: 3px !important;
    background: linear-gradient(180deg, var(--vf-key-top) 0%, var(--vf-key-bottom) 100%) !important;
    box-shadow: 0 1px 2px rgba(20, 18, 16, 0.12) !important;
    color: var(--vf-key-text) !important;
    font-weight: 650 !important;
    letter-spacing: 0.01em;
    text-shadow: none;
    transition: transform 0.12s ease, box-shadow 0.12s ease, filter 0.12s ease !important;
}

/* The function edge: a thin strip of meaning, not a costume. */
:is(button.action-button, .action-button button)::after {
    content: "";
    position: absolute;
    left: 10px;
    right: 10px;
    bottom: 5px;
    height: 3px;
    border-radius: 2px;
    background: var(--vf-edge);
    pointer-events: none;
}

:is(button.action-button, .action-button button):hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 8px rgba(20, 18, 16, 0.16) !important;
}

:is(button.action-button, .action-button button):active {
    transform: translateY(1px);
    box-shadow: 0 0 1px rgba(20, 18, 16, 0.2) !important;
}

:is(button.action-button, .action-button button):focus-visible {
    outline: 2px solid var(--vf-ember);
    outline-offset: 2px;
}

/* Intake keys: audio coming into the forge. */
:is(button#extract-audio-button, #extract-audio-button button),
:is(button#load-audio-button, #load-audio-button button) {
    --vf-edge: var(--vf-edge-intake);
}

/* File keys. */
:is(button#open-outputs-button, #open-outputs-button button) {
    --vf-edge: var(--vf-edge-files);
}

/* Preset keys. */
:is(button#preset-save-button, #preset-save-button button),
:is(button#preset-load-button, #preset-load-button button) {
    --vf-edge: var(--vf-edge-presets);
}

:is(button#preset-reset-button, #preset-reset-button button) {
    --vf-edge: var(--vf-edge-warn);
}

/* Destructive: the one key that is a different key. */
:is(button#preset-delete-button, #preset-delete-button button) {
    --vf-edge: transparent;
    background: transparent !important;
    border: 1px solid var(--vf-danger) !important;
    border-bottom-width: 3px !important;
    color: var(--vf-danger) !important;
    box-shadow: none !important;
}

/* ------------------------------------------------------------------ */
/* The forge strike.                                                   */
/* ------------------------------------------------------------------ */

:is(button#generate-speech-button, #generate-speech-button button) {
    min-height: 54px;
    letter-spacing: 0.02em;
    color: #fff4ec !important;
    font-weight: 750 !important;
    text-shadow: 0 1px 0 rgba(80, 20, 0, 0.55);
    background:
        radial-gradient(120% 90% at 50% 0%, rgba(255, 176, 122, 0.5), transparent 52%),
        linear-gradient(180deg, var(--vf-ember-hi) 0%, var(--vf-ember) 52%, var(--vf-ember-lo) 100%) !important;
    border: 1px solid rgba(122, 34, 4, 0.75) !important;
    border-bottom-width: 3px !important;
    box-shadow:
        0 10px 26px rgba(209, 49, 11, 0.32),
        0 1px 0 rgba(255, 226, 205, 0.45) inset !important;
}

:is(button#generate-speech-button, #generate-speech-button button)::after {
    display: none;
}

:is(button#generate-speech-button, #generate-speech-button button):hover {
    filter: brightness(1.05);
    box-shadow:
        0 14px 32px rgba(209, 49, 11, 0.42),
        0 1px 0 rgba(255, 226, 205, 0.5) inset !important;
}

/* ------------------------------------------------------------------ */
/* Masthead and typography.                                            */
/* ------------------------------------------------------------------ */

.vf-masthead {
    display: flex;
    align-items: baseline;
    gap: 0.9rem;
    flex-wrap: wrap;
    padding: 0.35rem 0 0.15rem;
}

.vf-wordmark {
    font-family: "Bricolage Grotesque", Inter, ui-sans-serif, sans-serif;
    font-weight: 800;
    font-size: 1.7rem;
    letter-spacing: -0.03em;
    line-height: 1;
    color: var(--body-text-color);
}

.vf-wordmark em {
    font-style: normal;
    color: var(--vf-ember);
}

.vf-tagline {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued, #8a8f98);
}

.vf-credit,
.vf-credit:visited {
    margin-left: auto;
    font-size: 0.78rem;
    color: var(--body-text-color-subdued, #8a8f98);
    text-decoration: none;
    border-bottom: 1px dotted currentColor;
}

.top-input-panel h3,
.top-input-panel .prose h3 {
    font-family: "Bricolage Grotesque", Inter, ui-sans-serif, sans-serif;
}

/* Paths, cue examples, and the scan table read in mono: these are the   */
/* page's instrument readouts, and lining digits keep columns honest.    */
.caption-timing-help code,
.top-input-panel code,
table.svelte-table, .table-wrap table, .gradio-dataframe table {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-variant-numeric: tabular-nums;
    font-size: 0.86em;
}

@media (prefers-reduced-motion: reduce) {
    :is(button.action-button, .action-button button),
    :is(button#generate-speech-button, #generate-speech-button button) {
        transition: none !important;
    }
}
"""
