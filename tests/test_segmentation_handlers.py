"""The reference-clip panel: what each control does.

These call the handlers the way the wiring does. Gradio update objects are
plain dicts, so assertions read the "value"/"choices" keys directly.

Two rules under test throughout.

No handler raises. Every one of them hangs off a control, and a traceback out
of a gradio handler shows the user a spinner that stops and nothing else.

Every public function is called at least once, with its real arguments. A
module can import cleanly and still raise NameError the first time a body
runs, which is how four bugs shipped out of the webui split; a test that only
imports would not have caught any of them.

The scan tests run the real librosa pipeline against a synthesised recording
rather than a fixture on disk, so the pauses are where the test says they are
and the expected segment count is arithmetic rather than a guess.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_segmentation as segmentation
import character_store as store
import webui_character_handlers as characters
import webui_segmentation_handlers as handlers

FIXTURE_RATE = 22050


def a_segment(index=0, start=0.0, duration=5.0, **overrides):
    """A measured segment, without going near an audio file."""
    fields = dict(
        lufs=-20.0, peak_dbfs=-3.0, pitch_hz_mean=150.0,
        pitch_hz_range=40.0, voiced_fraction=0.8, onset_rate_hz=3.0,
    )
    fields.update(overrides)
    return segmentation.Segment(
        index=index, start_s=start, end_s=start + duration,
        duration_s=duration, **fields,
    )


def speech_like(seconds, rate=FIXTURE_RATE, base_hz=150.0):
    """Something pyin will track a pitch in.

    A plain sine is not enough -- a pitch tracker wants harmonics and a
    little movement, and a dead-steady tone reports a zero range, which is
    the value several assertions here need to be non-zero.
    """
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    f0 = base_hz + 25.0 * np.sin(2 * np.pi * 0.8 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    wave = np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    return (0.4 * wave / np.max(np.abs(wave))).astype(np.float32)


def a_recording(path, bursts=3, burst_s=3.0, gap_s=1.0, rate=FIXTURE_RATE):
    """Bursts of speech separated by real silence, written to `path`."""
    silence = np.zeros(int(rate * gap_s), dtype=np.float32)
    pieces = []
    for number in range(bursts):
        pieces.append(speech_like(burst_s, rate, base_hz=140.0 + 20.0 * number))
        if number != bursts - 1:
            pieces.append(silence)
    soundfile.write(str(path), np.concatenate(pieces), rate)
    return str(path)


class _Scratch(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.scratch = self._dir.name
        self.exports = os.path.join(self.scratch, "exports")
        self.library = os.path.join(self.scratch, "library")
        os.makedirs(self.library, exist_ok=True)

    def a_state(self, segments, source=None):
        return handlers.scan_state(source or self.a_recording_path(), FIXTURE_RATE,
                                   segments)

    def a_recording_path(self, name="take.wav", **kwargs):
        return a_recording(Path(self.scratch) / name, **kwargs)


# ------------------------------------------------------------- formatting

class FormattingTests(_Scratch):
    def test_a_position_reads_as_minutes_and_seconds(self):
        self.assertEqual(handlers._format_clock(0.0), "0:00.0")
        self.assertEqual(handlers._format_clock(75.4), "1:15.4")

    def test_a_missing_measurement_is_marked_not_zeroed(self):
        # A zero would read as measured-and-silent, which is a different
        # thing from "this could not be measured".
        self.assertEqual(handlers._format_measurement(None), handlers.MISSING_MEASUREMENT)
        self.assertEqual(handlers._format_percent(None), handlers.MISSING_MEASUREMENT)
        self.assertNotIn("0", handlers._format_measurement(None))

    def test_a_boolean_is_not_treated_as_a_number(self):
        # bool is an int in Python, so an unguarded isinstance check would
        # format True as "1.0".
        self.assertEqual(handlers._format_measurement(True), handlers.MISSING_MEASUREMENT)
        self.assertEqual(handlers._format_percent(False), handlers.MISSING_MEASUREMENT)

    def test_a_clock_given_nonsense_does_not_raise(self):
        self.assertEqual(handlers._format_clock("later"), handlers.MISSING_MEASUREMENT)

    def test_a_voiced_fraction_is_shown_as_a_percentage(self):
        self.assertEqual(handlers._format_percent(0.802), "80%")


class VerdictTests(_Scratch):
    def test_a_short_segment_is_called_short(self):
        verdict = handlers.verdict_for(a_segment(duration=1.0))
        self.assertEqual(verdict, handlers.VERDICT_TOO_SHORT)

    def test_a_segment_over_the_engine_cap_says_so(self):
        verdict = handlers.verdict_for(a_segment(duration=40.0))
        self.assertEqual(verdict, handlers.VERDICT_OVER_CAP)

    def test_a_usable_segment_is_called_good(self):
        self.assertEqual(handlers.verdict_for(a_segment(duration=8.0)),
                         handlers.VERDICT_USABLE)

    def test_the_cap_in_the_verdict_is_the_engines_number(self):
        # Not a literal typed twice: if the engine's cap moves, this moves.
        self.assertIn(f"{segmentation.ENGINE_REFERENCE_SECONDS:.0f}",
                      handlers.VERDICT_OVER_CAP)


class TableTests(_Scratch):
    def test_a_row_has_one_cell_per_header(self):
        row = handlers.segment_row(a_segment())
        self.assertEqual(len(row), len(handlers.SEGMENT_TABLE_HEADERS))

    def test_the_first_column_counts_from_one_not_zero(self):
        # The index is a person-facing position in the recording.
        self.assertEqual(handlers.segment_row(a_segment(index=0))[0], "1")

    def test_an_unmeasurable_segment_still_produces_a_full_row(self):
        silent = a_segment(lufs=None, pitch_hz_mean=None, pitch_hz_range=None,
                           voiced_fraction=None, onset_rate_hz=None)
        row = handlers.segment_row(silent)
        self.assertEqual(len(row), len(handlers.SEGMENT_TABLE_HEADERS))
        self.assertEqual(row.count(handlers.MISSING_MEASUREMENT), 5)

    def test_the_table_is_truncated_but_the_dropdown_is_not(self):
        many = [a_segment(index=i) for i in range(handlers.MAX_TABLE_ROWS + 25)]
        self.assertEqual(len(handlers.segment_rows(many)), handlers.MAX_TABLE_ROWS)
        self.assertEqual(len(handlers.segment_choices(many)), len(many))

    def test_a_truncated_table_says_so_in_the_summary(self):
        # A silently shortened table reads as a complete one.
        many = [a_segment(index=i) for i in range(handlers.MAX_TABLE_ROWS + 1)]
        self.assertIn(str(handlers.MAX_TABLE_ROWS), handlers.scan_summary(many))

    def test_a_short_table_does_not_mention_truncation(self):
        summary = handlers.scan_summary([a_segment(), a_segment(index=1)])
        self.assertNotIn(str(handlers.MAX_TABLE_ROWS), summary)

    def test_one_segment_is_counted_in_the_singular(self):
        self.assertIn("1 segment found", handlers.scan_summary([a_segment()]))

    def test_two_segments_are_counted_in_the_plural(self):
        summary = handlers.scan_summary([a_segment(), a_segment(index=1)])
        self.assertIn("2 segments found", summary)

    def test_the_summary_counts_only_usable_segments_as_usable(self):
        segments = [a_segment(index=0, duration=8.0),
                    a_segment(index=1, duration=40.0),
                    a_segment(index=2, duration=0.5)]
        self.assertIn("3 segments found, 1 within", handlers.scan_summary(segments))

    def test_a_dropdown_value_is_the_position_in_the_recording(self):
        # Not the row number: the rows get re-sorted, the position does not.
        ranked = [a_segment(index=7), a_segment(index=2)]
        self.assertEqual([value for _, value in handlers.segment_choices(ranked)],
                         [7, 2])

    def test_a_dropdown_label_carries_enough_to_choose_by(self):
        label = handlers.segment_label(a_segment(index=4, start=62.0, duration=6.0))
        self.assertIn("#5", label)
        self.assertIn("1:02.0", label)
        self.assertIn("LUFS", label)


# ------------------------------------------------------------------ state

class StateTests(_Scratch):
    def test_a_segment_survives_the_round_trip_through_state(self):
        original = a_segment(index=3, start=10.0, duration=4.0)
        state = self.a_state([original])
        self.assertEqual(handlers.segment_from_state(state, 3), original)

    def test_an_index_that_is_not_in_the_state_is_a_miss_not_a_crash(self):
        # A scan replaces the state while a stale index sits in the dropdown.
        state = self.a_state([a_segment(index=0)])
        self.assertIsNone(handlers.segment_from_state(state, 99))

    def test_no_state_at_all_is_a_miss(self):
        self.assertIsNone(handlers.segment_from_state(None, 0))
        self.assertIsNone(handlers.segment_from_state("not a dict", 0))

    def test_a_nonsense_index_is_a_miss(self):
        state = self.a_state([a_segment(index=0)])
        self.assertIsNone(handlers.segment_from_state(state, None))
        self.assertIsNone(handlers.segment_from_state(state, "second one"))

    def test_state_written_by_a_different_dataclass_shape_is_a_miss(self):
        state = {"source": "x.wav", "sample_rate": 22050,
                 "segments": [{"index": 0, "start_s": 0.0, "mystery": 1}]}
        self.assertIsNone(handlers.segment_from_state(state, 0))

    def test_the_state_remembers_which_file_was_scanned(self):
        state = handlers.scan_state("/tmp/take.wav", 44100, [a_segment()])
        self.assertEqual(state["source"], "/tmp/take.wav")
        self.assertEqual(state["sample_rate"], 44100)


# ----------------------------------------------------------------- export

class ExportTests(_Scratch):
    def test_a_span_is_cut_out_at_the_length_it_claims(self):
        source = self.a_recording_path()
        segment = a_segment(start=4.0, duration=3.0)
        path = handlers.export_segment(source, segment, self.exports)
        written, rate = soundfile.read(path)
        self.assertAlmostEqual(len(written) / rate, 3.0, places=1)

    def test_the_cut_lands_where_the_segment_says_it_does(self):
        # Burst, gap, burst: a span inside the gap has to come out silent,
        # which proves the offset is applied and not ignored.
        source = self.a_recording_path(bursts=2, burst_s=3.0, gap_s=2.0)
        quiet = handlers.export_segment(source, a_segment(start=3.3, duration=1.4),
                                        self.exports)
        loud = handlers.export_segment(source, a_segment(start=0.5, duration=1.4),
                                       self.exports)
        self.assertLess(float(np.max(np.abs(soundfile.read(quiet)[0]))), 0.01)
        self.assertGreater(float(np.max(np.abs(soundfile.read(loud)[0]))), 0.1)

    def test_the_same_span_twice_is_the_same_file_not_two(self):
        source = self.a_recording_path()
        segment = a_segment(start=1.0, duration=2.0)
        first = handlers.export_segment(source, segment, self.exports)
        second = handlers.export_segment(source, segment, self.exports)
        self.assertEqual(first, second)
        self.assertEqual(len(os.listdir(self.exports)), 1)

    def test_no_half_written_file_is_left_beside_the_finished_one(self):
        # The preview player reads the path the moment the handler returns.
        source = self.a_recording_path()
        handlers.export_segment(source, a_segment(duration=2.0), self.exports)
        leftovers = [n for n in os.listdir(self.exports) if n.endswith(".partial")]
        self.assertEqual(leftovers, [])

    def test_the_export_root_is_created_if_it_is_not_there(self):
        source = self.a_recording_path()
        nested = os.path.join(self.exports, "deeper", "still")
        path = handlers.export_segment(source, a_segment(duration=2.0), nested)
        self.assertTrue(os.path.isfile(path))

    def test_the_default_export_root_is_absolute(self):
        # `outputs` against the working directory put files wherever the app
        # happened to be launched from; that is a bug this repo has fixed once.
        self.assertTrue(os.path.isabs(handlers.default_export_root()))

    def test_the_default_export_root_is_under_outputs(self):
        root = handlers.default_export_root()
        self.assertEqual(os.path.basename(root), handlers.SEGMENT_EXPORT_DIRNAME)
        self.assertEqual(os.path.basename(os.path.dirname(root)), "outputs")

    def test_no_export_root_given_falls_back_to_the_default(self):
        # The only caller that omits it is a mis-wiring, but the branch is
        # reachable and was not being run.
        source = self.a_recording_path()
        with mock.patch.object(handlers, "default_export_root",
                               return_value=self.exports):
            path = handlers.export_segment(source, a_segment(duration=2.0))
        self.assertEqual(os.path.dirname(path), self.exports)

    def test_the_filename_carries_the_span_to_two_decimals(self):
        # The name is what makes re-cutting the same span idempotent, so the
        # precision in it is load-bearing rather than cosmetic.
        source = self.a_recording_path("take.wav")
        name = os.path.basename(handlers.export_segment(
            source, a_segment(start=1.25, duration=2.5), self.exports))
        self.assertTrue(name.startswith("take_"))
        self.assertTrue(name.endswith("_1.25-3.75.wav"), name)

    def test_two_recordings_sharing_a_basename_do_not_overwrite_each_other(self):
        # One flat export folder, and "take.wav" out of two session folders is
        # an ordinary thing to have. The basename alone is not a unique name.
        first = os.path.join(self.scratch, "monday")
        second = os.path.join(self.scratch, "tuesday")
        os.makedirs(first), os.makedirs(second)
        segment = a_segment(start=0.5, duration=2.0)
        one = handlers.export_segment(
            a_recording(Path(first) / "take.wav"), segment, self.exports)
        two = handlers.export_segment(
            a_recording(Path(second) / "take.wav", bursts=2), segment, self.exports)
        self.assertNotEqual(one, two)
        self.assertEqual(len(os.listdir(self.exports)), 2)

    def test_the_same_recording_keeps_the_same_export_name(self):
        # The digest must come from the path, not from the file's contents or
        # a clock, or re-previewing a segment would write a new file each time.
        source = self.a_recording_path()
        segment = a_segment(start=1.0, duration=2.0)
        self.assertEqual(handlers.segment_export_name(source, segment),
                         handlers.segment_export_name(source, segment))

    def test_the_cut_is_ramped_in_and_out_so_it_does_not_click(self):
        # A boundary that is not a zero crossing is an audible click at both
        # ends of every generation that uses the clip.
        source = self.a_recording_path(bursts=1, burst_s=5.0)
        path = handlers.export_segment(
            source, a_segment(start=1.0, duration=3.0), self.exports)
        written, rate = soundfile.read(path)
        self.assertEqual(written[0], 0.0)
        self.assertEqual(written[-1], 0.0)
        # Only the edges: the middle must be untouched audio, not silence.
        self.assertGreater(float(np.max(np.abs(written[rate:-rate]))), 0.1)

    def test_a_clip_too_short_to_ramp_is_returned_unfaded_not_emptied(self):
        rate = 22050
        tiny = speech_like(0.004, rate)
        faded = handlers._fade_edges(tiny, rate)
        self.assertEqual(faded.size, tiny.size)
        self.assertTrue(np.array_equal(faded, tiny))

    def test_a_backwards_span_is_clamped_rather_than_raising(self):
        source = self.a_recording_path()
        backwards = segmentation.Segment(
            index=0, start_s=4.0, end_s=2.0, duration_s=-2.0, lufs=None,
            peak_dbfs=None, pitch_hz_mean=None, pitch_hz_range=None,
            voiced_fraction=None, onset_rate_hz=None)
        path = handlers.export_segment(source, backwards, self.exports)
        self.assertTrue(os.path.isfile(path))


# ------------------------------------------------------------------- scan

class ScanTests(_Scratch):
    def drain(self, **kwargs):
        """Run the generator to the end and hand back the last yield."""
        frames = list(handlers.scan_recording_ui(**kwargs))
        self.assertTrue(frames, "the scan yielded nothing at all")
        return frames[-1]

    def unpack(self, frame):
        table, state, choices, bar, status = frame
        return table, state, choices, bar, status

    def test_pressing_scan_with_nothing_loaded_says_so(self):
        _, state, _, _, status = self.unpack(self.drain(cleaned_path=None))
        self.assertIsNone(state)
        self.assertIn("Load a recording", status["value"])

    def test_a_file_that_has_since_gone_is_reported_not_raised(self):
        frame = self.drain(cleaned_path=os.path.join(self.scratch, "gone.wav"))
        self.assertIn("no longer there", self.unpack(frame)[4]["value"])

    def test_an_upload_wins_over_the_cleaned_file(self):
        # Same precedence as the cleanup section above it.
        upload = self.a_recording_path("upload.wav", bursts=2)
        frame = self.drain(cleaned_path=os.path.join(self.scratch, "gone.wav"),
                           uploaded_path=upload)
        self.assertEqual(self.unpack(frame)[1]["source"], upload)

    def test_a_recording_splits_at_its_pauses(self):
        path = self.a_recording_path(bursts=3, burst_s=3.0, gap_s=1.0)
        table, state, choices, bar, status = self.unpack(self.drain(cleaned_path=path))
        self.assertEqual(len(state["segments"]), 3)
        self.assertEqual(len(table["value"]), 3)
        self.assertEqual(len(choices["choices"]), 3)
        self.assertIn("3 segments found", status["value"])

    def test_the_scan_measures_what_it_found(self):
        path = self.a_recording_path(bursts=2, burst_s=3.0, gap_s=1.0)
        state = self.unpack(self.drain(cleaned_path=path))[1]
        first = state["segments"][0]
        self.assertIsNotNone(first["lufs"])
        self.assertIsNotNone(first["pitch_hz_mean"])
        # The fixture sweeps 25 Hz either side of its base frequency, so a
        # range of zero would mean the tracker reported nothing useful.
        self.assertGreater(first["pitch_hz_range"], 1.0)

    def test_a_silent_recording_is_reported_as_empty_not_as_one_long_take(self):
        # librosa.effects.split thresholds relative to the peak, so without
        # the absolute floor a silent file comes back as one fine segment
        # covering the whole thing.
        path = Path(self.scratch) / "silent.wav"
        soundfile.write(str(path), np.zeros(FIXTURE_RATE * 4, dtype=np.float32),
                        FIXTURE_RATE)
        _, state, _, _, status = self.unpack(self.drain(cleaned_path=str(path)))
        self.assertIsNone(state)
        self.assertIn("No usable speech", status["value"])

    def test_a_file_that_is_not_audio_is_reported_not_raised(self):
        path = Path(self.scratch) / "notaudio.wav"
        path.write_bytes(b"this is not a wav file at all")
        _, state, _, _, status = self.unpack(self.drain(cleaned_path=str(path)))
        self.assertIsNone(state)
        self.assertIn("Could not scan", status["value"])

    def test_the_failure_says_what_actually_went_wrong(self):
        # "Could not scan that recording." on its own is not actionable; the
        # exception type and message are the only clue the user gets.
        path = Path(self.scratch) / "notaudio.wav"
        path.write_bytes(b"this is not a wav file at all")
        status = self.unpack(self.drain(cleaned_path=str(path)))[4]["value"]
        self.assertIn("Unexpected", status)
        self.assertGreater(len(status), len("Could not scan that recording. "))

    def test_a_failed_scan_clears_the_table_and_the_state(self):
        path = Path(self.scratch) / "notaudio.wav"
        path.write_bytes(b"nope")
        table, state, _, _, _ = self.unpack(self.drain(cleaned_path=str(path)))
        self.assertEqual(table["value"], [])
        self.assertIsNone(state)

    def test_the_minimum_length_setting_drops_shorter_pieces(self):
        path = self.a_recording_path(bursts=3, burst_s=3.0, gap_s=1.0)
        frame = self.drain(cleaned_path=path, min_seconds=10.0)
        self.assertIsNone(self.unpack(frame)[1])

    def test_the_scan_reports_progress_once_per_segment(self):
        # One opening frame, one per segment measured, one final. A looser
        # assertion here would still pass if the throttle collapsed three
        # updates into one, which is the regression worth catching.
        path = self.a_recording_path(bursts=3, burst_s=3.0, gap_s=1.0)
        frames = list(handlers.scan_recording_ui(cleaned_path=path))
        self.assertEqual(len(frames), 5)

    def test_a_supplied_progress_callback_gets_the_whole_run(self):
        seen = []
        path = self.a_recording_path(bursts=3, burst_s=3.0, gap_s=1.0)
        list(handlers.scan_recording_ui(
            cleaned_path=path,
            progress=lambda fraction, desc=None: seen.append(round(fraction, 2))))
        self.assertEqual(seen, [0.33, 0.67, 1.0])

    def test_the_progress_message_names_the_segment_being_measured(self):
        seen = []
        path = self.a_recording_path(bursts=2, burst_s=3.0, gap_s=1.0)
        list(handlers.scan_recording_ui(
            cleaned_path=path,
            progress=lambda fraction, desc=None: seen.append(desc)))
        self.assertEqual(seen, ["Measuring segment 1 of 2",
                                "Measuring segment 2 of 2"])

    def a_mixed_recording(self):
        """Three spans where the middle one is over the engine's cap.

        A fixture of three equally usable bursts cannot tell a ranking from a
        shuffle, which is what an earlier version of this test could not do.
        """
        rate = FIXTURE_RATE
        silence = np.zeros(int(rate * 1.0), dtype=np.float32)
        over_cap = segmentation.ENGINE_REFERENCE_SECONDS + 5.0
        pieces = [
            speech_like(3.0, rate, 140.0), silence,
            speech_like(over_cap, rate, 160.0), silence,
            speech_like(3.0, rate, 180.0),
        ]
        path = Path(self.scratch) / "mixed.wav"
        soundfile.write(str(path), np.concatenate(pieces), rate)
        return str(path)

    def test_spoken_order_is_the_order_they_were_said_in(self):
        frame = self.drain(cleaned_path=self.a_mixed_recording(), best_first=False)
        self.assertEqual([v for _, v in self.unpack(frame)[2]["choices"]], [0, 1, 2])

    def test_best_first_sinks_the_segment_that_breaks_the_cap(self):
        # The middle span is over the 15s cap, so ranking must move it last
        # while leaving the two usable ones ahead of it.
        frame = self.drain(cleaned_path=self.a_mixed_recording(), best_first=True)
        order = [v for _, v in self.unpack(frame)[2]["choices"]]
        self.assertEqual(order[-1], 1)
        self.assertEqual(sorted(order[:2]), [0, 2])

    def test_best_first_and_spoken_order_actually_differ(self):
        # If they agree, neither of the two tests above is proving anything.
        path = self.a_mixed_recording()
        ranked = [v for _, v in self.unpack(
            self.drain(cleaned_path=path, best_first=True))[2]["choices"]]
        spoken = [v for _, v in self.unpack(
            self.drain(cleaned_path=path, best_first=False))[2]["choices"]]
        self.assertNotEqual(ranked, spoken)

    def test_the_over_cap_segment_is_labelled_as_over_the_cap(self):
        frame = self.drain(cleaned_path=self.a_mixed_recording(), best_first=False)
        rows = self.unpack(frame)[0]["value"]
        self.assertEqual(rows[1][-1], handlers.VERDICT_OVER_CAP)
        self.assertEqual(rows[0][-1], handlers.VERDICT_USABLE)

    def test_the_dropdown_comes_back_with_something_selected(self):
        # A Dropdown preprocessed with a value absent from its choices raises.
        path = self.a_recording_path(bursts=2)
        choices = self.unpack(self.drain(cleaned_path=path))[2]
        self.assertIn(choices["value"], [value for _, value in choices["choices"]])

    def test_a_failed_scan_clears_the_dropdown_to_nothing_selected(self):
        # An empty string is absent from an empty choice list, and gradio
        # raises on that; None is the value it accepts.
        choices = self.unpack(self.drain(cleaned_path=None))[2]
        self.assertEqual(choices["choices"], [])
        self.assertIs(choices["value"], characters.NO_SELECTION)


# ------------------------------------------------------------- the picks

class PreviewTests(_Scratch):
    def test_choosing_a_segment_produces_something_to_play(self):
        source = self.a_recording_path()
        state = self.a_state([a_segment(start=1.0, duration=2.0)], source)
        audio, detail = handlers.preview_segment_ui(state, 0, self.exports)
        self.assertTrue(os.path.isfile(audio["value"]))
        self.assertTrue(detail["visible"])

    def test_the_description_names_the_measurements(self):
        detail = handlers.describe_segment(a_segment(index=2, start=61.0))
        self.assertIn("#3", detail)
        self.assertIn("LUFS", detail)
        self.assertIn("voiced", detail)

    def test_a_stale_selection_clears_the_player_instead_of_raising(self):
        state = self.a_state([a_segment()])
        audio, detail = handlers.preview_segment_ui(state, 99, self.exports)
        self.assertIsNone(audio["value"])
        self.assertFalse(detail["visible"])

    def test_a_source_that_has_gone_is_reported_not_raised(self):
        state = handlers.scan_state(os.path.join(self.scratch, "gone.wav"),
                                    FIXTURE_RATE, [a_segment()])
        audio, detail = handlers.preview_segment_ui(state, 0, self.exports)
        self.assertIsNone(audio["value"])
        self.assertIn("Could not cut", detail["value"])


class UseAsReferenceTests(_Scratch):
    def test_the_reference_slot_gets_the_cut_segment(self):
        source = self.a_recording_path()
        state = self.a_state([a_segment(start=1.0, duration=2.0)], source)
        reference, status = handlers.use_segment_as_reference_ui(state, 0, self.exports)
        self.assertTrue(os.path.isfile(reference["value"]))
        self.assertIn("#1", status["value"])

    def test_an_over_long_segment_warns_that_only_its_opening_is_used(self):
        source = self.a_recording_path(bursts=1, burst_s=20.0)
        state = self.a_state([a_segment(start=0.0, duration=20.0)], source)
        _, status = handlers.use_segment_as_reference_ui(state, 0, self.exports)
        self.assertIn(f"{segmentation.ENGINE_REFERENCE_SECONDS:.0f}", status["value"])

    def test_a_segment_inside_the_cap_carries_no_warning(self):
        source = self.a_recording_path()
        state = self.a_state([a_segment(duration=6.0)], source)
        _, status = handlers.use_segment_as_reference_ui(state, 0, self.exports)
        self.assertNotIn("Only its first", status["value"])

    def test_pressing_it_with_nothing_scanned_says_so(self):
        reference, status = handlers.use_segment_as_reference_ui(None, 0, self.exports)
        self.assertNotIn("value", reference)
        self.assertIn("Scan a recording", status["value"])

    def test_a_source_that_has_gone_is_reported_not_raised(self):
        state = handlers.scan_state(os.path.join(self.scratch, "gone.wav"),
                                    FIXTURE_RATE, [a_segment()])
        reference, status = handlers.use_segment_as_reference_ui(state, 0, self.exports)
        self.assertNotIn("value", reference)
        self.assertIn("Could not cut", status["value"])

    def test_a_failed_press_leaves_the_reference_slot_alone(self):
        # The reference already loaded must survive a press that could not
        # produce a new one, which is what a bare gr.update() means.
        reference, _ = handlers.use_segment_as_reference_ui(None, 0, self.exports)
        self.assertEqual(reference, {"__type__": "update"})


class SaveToVoiceTests(_Scratch):
    def a_voice(self, name="Narrator"):
        return store.create_character(self.library, name, store.MODE_ONESHOT)

    def test_the_measurements_follow_the_clip_into_the_library(self):
        # This is the point of the whole stage: a clip saved from a scan
        # arrives knowing its own loudness and pitch, rather than a duration
        # read off a wav header and nothing else.
        slug = self.a_voice()
        source = self.a_recording_path()
        state = self.a_state([a_segment(start=1.0, duration=3.0, lufs=-18.5,
                                        pitch_hz_mean=142.0)], source)
        handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, slug, state, 0, "", self.library, self.exports)

        clip, = store.load_character(self.library, slug)["oneshot"]["clips"]
        self.assertAlmostEqual(clip["lufs"], -18.5)
        self.assertAlmostEqual(clip["pitch_hz_mean"], 142.0)
        self.assertAlmostEqual(clip["duration_s"], 3.0)
        self.assertAlmostEqual(clip["voiced_fraction"], 0.8)

    def test_the_scan_index_does_not_follow_the_clip(self):
        # It numbers a scan that no longer exists once the clip is filed.
        slug = self.a_voice()
        state = self.a_state([a_segment(index=4)], self.a_recording_path())
        handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, slug, state, 4, "", self.library, self.exports)
        clip, = store.load_character(self.library, slug)["oneshot"]["clips"]
        self.assertNotIn("index", clip)

    def test_the_clip_is_labelled_with_where_it_came_from(self):
        slug = self.a_voice()
        source = self.a_recording_path("interview.wav")
        state = self.a_state([a_segment(index=2)], source)
        handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, slug, state, 2, "", self.library, self.exports)
        clip, = store.load_character(self.library, slug)["oneshot"]["clips"]
        self.assertIn("interview.wav", clip["label"])
        self.assertIn("#3", clip["label"])

    def test_a_typed_label_wins_over_the_generated_one(self):
        slug = self.a_voice()
        state = self.a_state([a_segment()], self.a_recording_path())
        handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, slug, state, 0, "angry take", self.library,
            self.exports)
        clip, = store.load_character(self.library, slug)["oneshot"]["clips"]
        self.assertEqual(clip["label"], "angry take")

    def test_it_returns_the_four_outputs_the_panel_expects(self):
        slug = self.a_voice()
        state = self.a_state([a_segment()], self.a_recording_path())
        result = handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, slug, state, 0, "", self.library, self.exports)
        self.assertEqual(len(result), 4)
        select, name, summary, status = result
        self.assertEqual(name["value"], "Narrator")
        self.assertIn("1 clip", summary["value"])
        self.assertTrue(status["visible"])

    def test_saving_with_nothing_picked_says_so_and_files_nothing(self):
        slug = self.a_voice()
        result = handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, slug, None, 0, "", self.library, self.exports)
        self.assertIn("Scan a recording", result[3]["value"])
        self.assertEqual(store.load_character(self.library, slug)["oneshot"]["clips"], [])

    def test_a_source_that_has_gone_is_reported_not_raised(self):
        slug = self.a_voice()
        state = handlers.scan_state(os.path.join(self.scratch, "gone.wav"),
                                    FIXTURE_RATE, [a_segment()])
        result = handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, slug, state, 0, "", self.library, self.exports)
        self.assertIn("Could not cut", result[3]["value"])

    def test_an_rvc_voice_is_refused_rather_than_given_a_clip(self):
        slug = store.create_character(self.library, "Robot", store.MODE_RVC)
        state = self.a_state([a_segment()], self.a_recording_path())
        result = handlers.save_segment_to_voice_ui(
            store.MODE_RVC, slug, state, 0, "", self.library, self.exports)
        self.assertIn("model, not clips", result[3]["value"])

    def test_saving_with_no_voice_selected_says_so(self):
        state = self.a_state([a_segment()], self.a_recording_path())
        result = handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, "", state, 0, "", self.library, self.exports)
        self.assertIn("Select a voice", result[3]["value"])

    def test_a_refused_save_does_not_leave_a_cut_wav_behind(self):
        # Exporting first and refusing afterwards leaves a working file that
        # nothing points at, and the export folder is never cleaned.
        state = self.a_state([a_segment()], self.a_recording_path())
        handlers.save_segment_to_voice_ui(
            store.MODE_ONESHOT, "", state, 0, "", self.library, self.exports)
        self.assertFalse(os.path.isdir(self.exports))


class MetricsHandoffTests(_Scratch):
    """The change this stage needed from the character handlers."""

    def test_supplied_measurements_replace_the_wav_header_guess(self):
        slug = store.create_character(self.library, "Narrator", store.MODE_ONESHOT)
        source = self.a_recording_path()
        characters.add_reference_to_character_ui(
            store.MODE_ONESHOT, slug, source, "", self.library,
            metrics={"duration_s": 4.25, "lufs": -14.0})
        clip, = store.load_character(self.library, slug)["oneshot"]["clips"]
        self.assertAlmostEqual(clip["duration_s"], 4.25)
        self.assertAlmostEqual(clip["lufs"], -14.0)

    def test_without_measurements_it_still_reads_the_wav_header(self):
        # The existing button passes no metrics, and must keep working.
        slug = store.create_character(self.library, "Narrator", store.MODE_ONESHOT)
        source = self.a_recording_path(bursts=1, burst_s=3.0)
        characters.add_reference_to_character_ui(
            store.MODE_ONESHOT, slug, source, "", self.library)
        clip, = store.load_character(self.library, slug)["oneshot"]["clips"]
        self.assertAlmostEqual(clip["duration_s"], 3.0, places=1)
        self.assertNotIn("lufs", clip)

    def test_the_save_target_names_the_selected_voice(self):
        # The voice lives on another tab, so this line is the only thing
        # telling the user what a save would land on.
        self.assertIn("Narrator", handlers.describe_save_target("Narrator"))

    def test_the_save_target_says_so_when_no_voice_is_selected(self):
        for empty in (None, "", "   "):
            self.assertIn("No voice selected", handlers.describe_save_target(empty))

    def test_the_idle_progress_bar_is_rendered_html_not_a_placeholder(self):
        self.assertIn("<div", handlers.SCAN_PROGRESS_IDLE)
        self.assertIn("Idle", handlers.SCAN_PROGRESS_IDLE)

    def test_the_tab_jump_script_does_nothing_without_a_reference(self):
        # The failure path must not teleport the user to another tab.
        self.assertIn("if (!reference) return;",
                      handlers.FOCUS_GENERATION_TAB_IF_SET_JS)
        self.assertIn("Audio Generation", handlers.FOCUS_GENERATION_TAB_IF_SET_JS)

    def test_the_panel_refresh_contract_is_public_and_unchanged(self):
        self.assertIs(characters._refresh, characters.refresh_panel)
        result = characters.refresh_panel(store.MODE_ONESHOT, "", "hello",
                                          self.library)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[3]["value"], "hello")


if __name__ == "__main__":
    unittest.main()
