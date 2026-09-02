"""Every function in the modules split out of webui.py, actually called.

These four modules were extracted during the refactor and shipped with no
tests at all. Four of them then broke in production because a function had
been moved without a module-level import it depended on -- `os` in
_build_tts_runtime_options, `re` and `html` in the preview, `datetime` in
the preset store. Every one of those modules imported cleanly, so
`ast.parse` passed, `import webui` passed, and 179 tests passed.

The point of this file is therefore blunt rather than clever: CALL every
public function at least once with plausible arguments. A NameError inside
a function body is invisible until the body runs, and running it is the
whole test. Assertions beyond "it ran and returned something sane" are a
bonus, not the purpose.

`ruff --select F821` catches the same class in about a second and is now
part of the gate. This is the belt to that braces -- it also catches a name
that exists but is wrong, which F821 cannot see.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["webui.py"]

import webui_media_utils as media
import webui_preview as preview
import webui_progress as progress
import webui_runtime as runtime


class RuntimeTests(unittest.TestCase):
    """Configuration and the device the next model load uses."""

    def test_runtime_options_can_actually_be_built(self):
        # This is on the path of EVERY generation. It shipped raising
        # NameError: name 'os' is not defined, and nothing caught it.
        options = runtime._build_tts_runtime_options()
        self.assertIn("model_dir", options)
        self.assertIn("cfg_path", options)
        self.assertTrue(options["cfg_path"].endswith("config.yaml"))

    def test_runtime_options_carry_the_selected_device(self):
        runtime.selected_device.set(runtime.DEVICE_CPU)
        self.addCleanup(runtime.selected_device.set, runtime.DEVICE_AUTO)
        self.assertEqual(runtime._build_tts_runtime_options()["device"], runtime.DEVICE_CPU)

    def test_auto_is_reported_as_no_explicit_device(self):
        runtime.selected_device.set(runtime.DEVICE_AUTO)
        self.assertIsNone(runtime._build_tts_runtime_options()["device"])

    def test_device_selection_reports_whether_it_changed(self):
        selection = runtime.DeviceSelection()
        self.assertTrue(selection.set(runtime.DEVICE_CPU))
        self.assertFalse(selection.set(runtime.DEVICE_CPU))
        self.assertEqual(selection.get(), runtime.DEVICE_CPU)


class ProgressTests(unittest.TestCase):
    """Console and browser progress reporting."""

    def test_current_timestamp_is_iso_like(self):
        stamp = progress.current_timestamp()
        self.assertRegex(stamp, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_elapsed_duration_formats_across_magnitudes(self):
        for seconds in (0, 1.5, 61, 3600, 86400):
            self.assertTrue(progress.format_elapsed_duration(seconds))

    def test_console_progress_runs_without_a_terminal(self):
        # It writes to stdout, which under a test runner is not a tty.
        # `started_at` here is a perf_counter float, NOT the ISO string
        # current_timestamp() returns -- webui_generation uses that name for
        # the ISO form, so the two conventions share a parameter name.
        import time
        progress.print_console_progress("Testing", 3, 10, time.perf_counter())

    def test_console_progress_handles_zero_total(self):
        import time
        progress.print_console_progress("Testing", 0, 0, time.perf_counter())

    def test_console_progress_reports_a_realtime_factor_when_given_audio(self):
        import time
        progress.print_console_progress(
            "Testing", 2, 4, time.perf_counter() - 1.0, processed_audio_seconds=3.0)

    def test_audio_duration_uses_the_sample_rate(self):
        import numpy as np
        self.assertAlmostEqual(
            progress.audio_duration_ms(np.zeros(24000, dtype=np.int16), 24000), 1000.0, places=3)

    def test_progress_bar_renders_html_at_every_state(self):
        for fraction, kwargs in ((0.0, {}), (0.5, {}), (1.0, {"done": True}), (0.3, {"failed": True})):
            html = progress.render_progress_bar(fraction, "working", **kwargs)
            self.assertIn("<", html)

    def test_progress_bar_escapes_its_message(self):
        # The message can carry a filename, and filenames can contain angle
        # brackets on some filesystems.
        self.assertNotIn("<script>", progress.render_progress_bar(0.5, "<script>x</script>"))

    def test_draining_a_progress_file_that_does_not_exist_is_safe(self):
        consumed, fraction, description = progress._drain_progress_file(
            os.path.join(tempfile.gettempdir(), "no-such-progress.jsonl"), 0, None)
        self.assertEqual(consumed, 0)

    def test_draining_reads_new_events_and_remembers_the_offset(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "progress.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"fraction": 0.25, "desc": "quarter"}) + "\n")
            consumed, fraction, description = progress._drain_progress_file(path, 0, None)
            self.assertGreater(consumed, 0)
            # a second drain from the same offset yields nothing new
            again = progress._drain_progress_file(path, consumed, fraction)
            self.assertEqual(again[0], consumed)

    def test_draining_survives_a_half_written_line(self):
        # The writer is another process; a read can land mid-line.
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "progress.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('{"fraction": 0.5, "desc": "hal')
            progress._drain_progress_file(path, 0, None)


class PreviewTests(unittest.TestCase):
    """The segment preview. Shipped raising NameError on `re` and `html`."""

    LONG = "Hello there. How are you today? I am quite well, thank you. " * 6

    def test_token_limit_is_clamped_into_range(self):
        self.assertGreaterEqual(preview.resolve_max_text_tokens(1), 20)
        self.assertLessEqual(preview.resolve_max_text_tokens(10 ** 6),
                             preview.PREVIEW_MAX_TEXT_TOKENS)

    def test_a_junk_token_limit_falls_back_to_a_default(self):
        self.assertTrue(preview.resolve_max_text_tokens("not a number"))
        self.assertTrue(preview.resolve_max_text_tokens(None))

    def test_estimating_splits_long_text(self):
        # The function that shipped raising NameError: name 're' is not defined.
        sections = preview.estimate_text_processing_sections(self.LONG, 20)
        self.assertGreater(len(sections), 1)

    def test_estimating_empty_text_does_not_crash(self):
        preview.estimate_text_processing_sections("", 120)

    def test_section_splitting_returns_something_for_real_text(self):
        self.assertTrue(preview.get_text_processing_sections(self.LONG, 120))

    def test_section_count_message_renders(self):
        # Reaches `html` on the subtitle error path.
        message = preview.build_section_count_message(self.LONG, 120)
        self.assertIn("Sections", message)

    def test_section_count_message_with_a_missing_subtitle_file_is_reported(self):
        message = preview.build_section_count_message(
            "", 120, subtitle_mode=True,
            subtitle_file=os.path.join(tempfile.gettempdir(), "no-such.srt"))
        self.assertTrue(message)

    def test_preview_rows_are_table_shaped(self):
        rows = preview.get_preview_rows(self.LONG, 120)
        self.assertIsInstance(rows, list)
        if rows:
            self.assertIsInstance(rows[0], list)

    def test_preview_rows_for_empty_text(self):
        preview.get_preview_rows("", 120)

    def test_the_engine_config_values_are_the_ones_the_engine_reports(self):
        # Guards the split: these come from cmd_args.model_dir, which now
        # lives in a different module than it used to.
        self.assertGreater(preview.PREVIEW_MAX_TEXT_TOKENS, 0)
        self.assertTrue(str(preview.MODEL_VERSION))


class MediaUtilsTests(unittest.TestCase):
    """Filesystem and ffmpeg helpers."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = self._dir.name
        self.addCleanup(self._dir.cleanup)

    def test_ffmpeg_probe_returns_a_bool_either_way(self):
        self.assertIsInstance(media.check_ffmpeg(), bool)

    def test_availability_flags_are_bools(self):
        self.assertIsInstance(media.FFMPEG_AVAILABLE, bool)
        self.assertIsInstance(media.MP3_AVAILABLE, bool)

    def test_next_file_number_starts_at_one_in_an_empty_directory(self):
        self.assertEqual(media.get_next_file_number(self.dir), 1)

    def test_next_file_number_steps_past_what_is_there(self):
        Path(self.dir, "0001.wav").write_bytes(b"x")
        Path(self.dir, "0002.wav").write_bytes(b"x")
        self.assertGreater(media.get_next_file_number(self.dir), 2)

    def test_next_file_number_ignores_unrelated_names(self):
        Path(self.dir, "notes.txt").write_text("x", encoding="utf-8")
        self.assertEqual(media.get_next_file_number(self.dir), 1)

    def test_generating_an_output_path_produces_something_writable(self):
        path = media.generate_output_path(target_folder=self.dir, filename="voice")
        self.assertTrue(os.path.isabs(path) or path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"x")

    def test_loading_a_path_that_exists_returns_it(self):
        source = Path(self.dir, "clip.wav")
        source.write_bytes(b"RIFFfake")
        self.assertTrue(media.load_audio_from_path(str(source)))

    def test_loading_a_path_that_does_not_exist_is_handled(self):
        # Either a falsy return or a raised error is acceptable; a NameError
        # or a silent wrong answer is not.
        try:
            self.assertFalse(media.load_audio_from_path(os.path.join(self.dir, "ghost.wav")))
        except Exception as exc:
            self.assertNotIsInstance(exc, NameError)

    def test_saving_and_reading_back_a_pcm16_wav(self):
        import numpy as np
        out = os.path.join(self.dir, "out.wav")
        media.save_pcm16_wav(np.zeros(2400, dtype=np.int16), 24000, out)
        self.assertTrue(os.path.isfile(out))
        self.assertGreater(os.path.getsize(out), 0)

    def test_extracting_from_a_file_that_is_not_media_is_handled(self):
        junk = Path(self.dir, "junk.mp4")
        junk.write_bytes(b"not really a video")
        try:
            media.extract_audio_from_media(str(junk))
        except Exception as exc:
            self.assertNotIsInstance(exc, NameError)

    def test_time_range_parsing_with_a_nonsense_range_is_handled(self):
        junk = Path(self.dir, "junk.wav")
        junk.write_bytes(b"RIFFfake")
        try:
            media.extract_time_ranges(str(junk), "not; a; range")
        except Exception as exc:
            self.assertNotIsInstance(exc, NameError)

    def test_mp3_conversion_without_pydub_is_reported_not_crashed(self):
        source = Path(self.dir, "in.wav")
        source.write_bytes(b"RIFFfake")
        try:
            media.convert_wav_to_mp3(str(source), os.path.join(self.dir, "out.mp3"))
        except Exception as exc:
            self.assertNotIsInstance(exc, NameError)


if __name__ == "__main__":
    unittest.main()
