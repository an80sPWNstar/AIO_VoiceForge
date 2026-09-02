"""The segment preview: how the app guesses where your text will be split.

The engine always does its own segmentation, so nothing here changes what
comes out of a generation. This only drives the preview table and the
"N sections" label, which exist so you can see a long script is about to
be cut into forty pieces before you press Generate.

It needs a sentencepiece model that IndexTTS-2.5 does not ship, so the
tokenizer is often absent and the split falls back to a character-count
estimate -- see the comment on PREVIEW_TEXT_TOKENIZER below.

Split out of webui.py. Imports webui_runtime for the checkpoint directory,
which is why the argument parsing had to move down first.
"""

import os

from omegaconf import OmegaConf

from indextts.utils.front import TextNormalizer, TextTokenizer
from subtitle_utils import (
    build_subtitle_render_units,
    format_srt_timestamp,
    get_subtitle_format_label,
    parse_subtitle_file,
)
from webui_runtime import cmd_args

PREVIEW_CFG = OmegaConf.load(os.path.join(cmd_args.model_dir, "config.yaml"))
PREVIEW_MAX_TEXT_TOKENS = int(PREVIEW_CFG.gpt.max_text_tokens)
MODEL_VERSION = str(getattr(PREVIEW_CFG, "version", "1.0"))
PREVIEW_BPE_PATH = os.path.join(cmd_args.model_dir, PREVIEW_CFG.dataset["bpe_model"])
PREVIEW_TEXT_NORMALIZER = TextNormalizer()
PREVIEW_TEXT_NORMALIZER.load()

# The segment preview is a UI convenience, and it needs a sentencepiece model
# that IndexTTS-2.5 does not ship: its config still names bpe.model, but the
# release tokenizes from a tiktoken vocabulary instead, so that file is absent.
# Without a tokenizer the preview falls back to a rough split; the engine itself
# always does its own segmentation, so an estimate here never changes output.
PREVIEW_TEXT_TOKENIZER = (
    TextTokenizer(PREVIEW_BPE_PATH, PREVIEW_TEXT_NORMALIZER)
    if os.path.exists(PREVIEW_BPE_PATH)
    else None
)
if PREVIEW_TEXT_TOKENIZER is None:
    print(
        f"Segment preview: no tokenizer at {PREVIEW_BPE_PATH}; "
        "showing an estimated split instead."
    )

# Average characters per token, used only when estimating the preview split.
PREVIEW_CHARS_PER_TOKEN = 4

def resolve_max_text_tokens(max_text_tokens_per_segment):
    if not max_text_tokens_per_segment:
        return 120

    try:
        max_tokens = int(float(str(max_text_tokens_per_segment).strip()))
        return max(20, min(max_tokens, PREVIEW_MAX_TEXT_TOKENS))
    except (ValueError, TypeError):
        return 120


def get_text_processing_sections(text, max_text_tokens_per_segment):
    if not text:
        return []

    max_tokens = resolve_max_text_tokens(max_text_tokens_per_segment)
    if PREVIEW_TEXT_TOKENIZER is None:
        return estimate_text_processing_sections(text, max_tokens)
    text_tokens_list = PREVIEW_TEXT_TOKENIZER.tokenize(text)
    return PREVIEW_TEXT_TOKENIZER.split_segments(text_tokens_list, max_text_tokens_per_segment=max_tokens)


def estimate_text_processing_sections(text, max_tokens):
    """Split text on sentence ends when no tokenizer is available.

    Roughly four characters per token, which is close enough for a preview whose
    only job is to show the user how many chunks their text will become.
    """
    budget = max(1, int(max_tokens * PREVIEW_CHARS_PER_TOKEN))
    sentences = re.findall(r"[^.!?。！？\n]+[.!?。！？]*\s*|\n+", text)
    sections, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) > budget:
            sections.append([current.strip()])
            current = sentence
        else:
            current += sentence
    if current.strip():
        sections.append([current.strip()])
    return sections


def build_section_count_message(text, max_text_tokens_per_segment, subtitle_mode=False, subtitle_file=None):
    if subtitle_mode and subtitle_file:
        try:
            cues = parse_subtitle_file(subtitle_file)
            render_units = build_subtitle_render_units(cues)
            processing_sections = 0
            for unit in render_units:
                if unit.text.strip():
                    processing_sections += len(get_text_processing_sections(unit.text, max_text_tokens_per_segment))

            if processing_sections == len(render_units):
                return (
                    f"**Current Sections:** {processing_sections} subtitle timing "
                    f"{'unit' if processing_sections == 1 else 'units'} from {len(cues)} cue"
                    f"{'' if len(cues) == 1 else 's'}"
                )

            return (
                f"**Current Sections:** {processing_sections} processing section"
                f"{'' if processing_sections == 1 else 's'} across {len(render_units)} subtitle timing unit"
                f"{'' if len(render_units) == 1 else 's'} from {len(cues)} cue"
                f"{'' if len(cues) == 1 else 's'}"
            )
        except Exception as e:
            return f"**Current Sections:** Unable to read subtitle file: {html.escape(str(e))}"

    sections = get_text_processing_sections(text, max_text_tokens_per_segment)
    return f"**Current Sections:** {len(sections)} text section{'' if len(sections) == 1 else 's'}"


def get_preview_rows(text, max_text_tokens_per_segment, subtitle_mode=False, subtitle_file=None):
    if subtitle_mode and subtitle_file:
        try:
            cues = parse_subtitle_file(subtitle_file)
            data = []
            cue_label = f"{get_subtitle_format_label(subtitle_file)} Cue"
            for cue in cues:
                details = f"{format_srt_timestamp(cue.start_ms)} -> {format_srt_timestamp(cue.end_ms)} ({cue.duration_ms} ms)"
                data.append([cue.index, cue_label, cue.text, details])
            return data
        except Exception as e:
            return [[0, "Caption Error", str(e), ""]]

    if not text:
        return []

    segments = get_text_processing_sections(text, max_text_tokens_per_segment)

    data = []
    for i, segment_tokens in enumerate(segments):
        segment_str = ''.join(segment_tokens)
        # Without a tokenizer the sections are estimated, so report the measure
        # that was actually used rather than calling characters "tokens".
        if PREVIEW_TEXT_TOKENIZER is None:
            details = f"~{len(segment_str)} characters"
        else:
            details = f"{len(segment_tokens)} tokens"
        data.append([i, "Text Segment", segment_str, details])
    return data
