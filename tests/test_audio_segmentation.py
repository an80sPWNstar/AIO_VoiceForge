"""Splitting a recording into candidate reference clips, and measuring them.

Mostly synthetic audio: a tone at a known frequency has a known pitch, and
silence has a known absence of one, which makes the measurements checkable
rather than merely plausible. One test runs against the real fixture so the
whole path is exercised on something that is actually speech.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_segmentation as seg

RATE = 22050


def tone(seconds: float, hz: float = 200.0, amplitude: float = 0.5, rate: int = RATE) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * rate), endpoint=False)
    return (amplitude * np.sin(2 * math.pi * hz * t)).astype(np.float32)


def silence(seconds: float, rate: int = RATE) -> np.ndarray:
    return np.zeros(int(seconds * rate), dtype=np.float32)


class SpanFindingTests(unittest.TestCase):
    def test_pure_silence_yields_no_spans(self):
        # Not one span covering the file: a caller has to be able to tell
        # "nothing here" from "one long take".
        self.assertEqual(seg.find_speech_spans(silence(2.0), RATE), [])

    def test_empty_audio_yields_no_spans(self):
        self.assertEqual(seg.find_speech_spans(np.zeros(0, dtype=np.float32), RATE), [])

    def test_near_silence_yields_no_spans_either(self):
        # librosa.effects.split thresholds relative to the peak, so a buffer
        # of near-nothing looks like solid speech to it. Without an absolute
        # floor a silent recording reports one segment covering all of it.
        whisper_quiet = np.full(int(2.0 * RATE), 1e-6, dtype=np.float32)
        self.assertEqual(seg.find_speech_spans(whisper_quiet, RATE), [])

    def test_audible_audio_is_still_found_after_the_floor_check(self):
        # The guard must not swallow quiet but real speech.
        audio = np.concatenate([silence(0.5), tone(1.5, amplitude=0.02), silence(0.5)])
        self.assertEqual(len(seg.find_speech_spans(audio, RATE)), 1)

    def test_a_single_burst_is_one_span(self):
        audio = np.concatenate([silence(0.5), tone(1.0), silence(0.5)])
        spans = seg.find_speech_spans(audio, RATE)
        self.assertEqual(len(spans), 1)
        start, end = spans[0]
        self.assertAlmostEqual(end - start, 1.0, delta=0.15)

    def test_two_bursts_with_a_long_gap_stay_separate(self):
        audio = np.concatenate([tone(0.8), silence(1.5), tone(0.8)])
        self.assertEqual(len(seg.find_speech_spans(audio, RATE)), 2)

    def test_two_bursts_with_a_short_gap_are_merged(self):
        # A sentence-internal pause is routinely 200-400 ms. Cutting there
        # produces fragments that sound clipped as a reference.
        audio = np.concatenate([tone(0.8), silence(0.2), tone(0.8)])
        self.assertEqual(len(seg.find_speech_spans(audio, RATE)), 1)

    def test_the_merge_gap_is_the_thing_that_decides(self):
        audio = np.concatenate([tone(0.8), silence(0.5), tone(0.8)])
        self.assertEqual(len(seg.find_speech_spans(audio, RATE, merge_gap_s=0.1)), 2)
        self.assertEqual(len(seg.find_speech_spans(audio, RATE, merge_gap_s=1.0)), 1)


class MergeTests(unittest.TestCase):
    """_merge_close_spans is pure, so it can be checked exactly."""

    def test_nothing_merges_nothing(self):
        self.assertEqual(seg._merge_close_spans([], 0.5), [])

    def test_adjacent_spans_join(self):
        self.assertEqual(seg._merge_close_spans([(0.0, 1.0), (1.2, 2.0)], 0.5), [(0.0, 2.0)])

    def test_distant_spans_do_not_join(self):
        spans = [(0.0, 1.0), (5.0, 6.0)]
        self.assertEqual(seg._merge_close_spans(spans, 0.5), spans)

    def test_a_chain_of_close_spans_becomes_one(self):
        spans = [(0.0, 1.0), (1.1, 2.0), (2.1, 3.0)]
        self.assertEqual(seg._merge_close_spans(spans, 0.2), [(0.0, 3.0)])


class MeasurementTests(unittest.TestCase):
    def test_every_measurement_key_is_always_present(self):
        # A caller reading .get() on a missing key would silently get None
        # and not know whether it was unmeasurable or unmeasured.
        measured = seg.measure_span(tone(1.0), RATE)
        for key in ("lufs", "peak_dbfs", "pitch_hz_mean", "pitch_hz_range",
                    "voiced_fraction", "onset_rate_hz"):
            self.assertIn(key, measured)

    def test_peak_of_a_known_amplitude(self):
        # 0.5 amplitude is -6 dBFS.
        self.assertAlmostEqual(seg._peak_dbfs(tone(0.5, amplitude=0.5)), -6.02, delta=0.1)

    def test_peak_of_silence_is_none_not_negative_infinity(self):
        self.assertIsNone(seg._peak_dbfs(silence(0.5)))

    def test_peak_of_empty_audio_is_none(self):
        self.assertIsNone(seg._peak_dbfs(np.zeros(0, dtype=np.float32)))

    def test_a_louder_tone_measures_louder(self):
        quiet = seg._integrated_loudness(tone(2.0, amplitude=0.1), RATE)
        loud = seg._integrated_loudness(tone(2.0, amplitude=0.8), RATE)
        if quiet is not None and loud is not None:
            self.assertGreater(loud, quiet)

    def test_loudness_of_too_short_a_span_is_none_not_a_guess(self):
        self.assertIsNone(seg._integrated_loudness(tone(0.1), RATE))

    def test_pitch_of_a_known_tone_is_that_tone(self):
        stats = seg._pitch_statistics(tone(1.5, hz=220.0), RATE)
        self.assertIsNotNone(stats["pitch_hz_mean"])
        self.assertAlmostEqual(stats["pitch_hz_mean"], 220.0, delta=15.0)

    def test_a_steady_tone_has_a_narrow_pitch_range(self):
        stats = seg._pitch_statistics(tone(1.5, hz=220.0), RATE)
        self.assertLess(stats["pitch_hz_range"], 30.0)

    def test_pitch_range_ignores_a_brief_outlier(self):
        # Range is the 5th-to-95th percentile so a couple of stray frames --
        # what a plosive or a breath produces -- cannot report a steady voice
        # as an animated one.
        #
        # Threshold measured, not guessed: on this fixture the percentile
        # range is 0.0 Hz and min-to-max is 15.5 Hz. pyin's Viterbi smoothing
        # means a 60 ms jump to 430 Hz never shows as 230 Hz of spread, so an
        # assertion loose enough to allow 15 Hz would pass either way.
        steady = tone(1.0, hz=200.0)
        blip = tone(0.06, hz=430.0)
        stats = seg._pitch_statistics(np.concatenate([steady, blip, steady]), RATE)
        self.assertIsNotNone(stats["pitch_hz_range"])
        self.assertLess(stats["pitch_hz_range"], 5.0,
                        "a 60 ms outlier is being counted into the range")

    def test_pitch_of_silence_reports_no_pitch(self):
        stats = seg._pitch_statistics(silence(1.5), RATE)
        self.assertIsNone(stats["pitch_hz_mean"])

    def test_pitch_of_too_short_a_span_is_none(self):
        stats = seg._pitch_statistics(tone(0.01), RATE)
        self.assertIsNone(stats["pitch_hz_mean"])

    def test_pitch_of_empty_audio_is_none(self):
        stats = seg._pitch_statistics(np.zeros(0, dtype=np.float32), RATE)
        self.assertIsNone(stats["pitch_hz_mean"])

    def test_onset_rate_of_too_short_a_span_is_none(self):
        self.assertIsNone(seg._onset_rate(tone(0.1), RATE))

    def test_onset_rate_is_per_second_not_a_count(self):
        # Same pattern, twice the length: the RATE should be similar, which
        # a raw count would not be.
        pattern = np.concatenate([tone(0.15), silence(0.15)])
        short = seg._onset_rate(np.tile(pattern, 4), RATE)
        long = seg._onset_rate(np.tile(pattern, 8), RATE)
        if short and long:
            self.assertAlmostEqual(short, long, delta=max(short, long) * 0.5)


class SegmentAudioTests(unittest.TestCase):
    def test_segments_carry_their_position_in_the_source(self):
        audio = np.concatenate([silence(1.0), tone(2.0), silence(1.0), tone(2.0)])
        segments = seg.segment_audio(audio, RATE)
        self.assertEqual(len(segments), 2)
        self.assertGreater(segments[1].start_s, segments[0].end_s)
        self.assertEqual([s.index for s in segments], [0, 1])

    def test_spans_shorter_than_the_minimum_are_dropped(self):
        audio = np.concatenate([tone(0.3), silence(1.0), tone(3.0)])
        segments = seg.segment_audio(audio, RATE, min_seconds=1.0)
        self.assertEqual(len(segments), 1)
        self.assertGreater(segments[0].duration_s, 1.0)

    def test_progress_is_reported_once_per_segment(self):
        audio = np.concatenate([tone(2.0), silence(1.0), tone(2.0), silence(1.0), tone(2.0)])
        seen = []
        seg.segment_audio(audio, RATE, progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[-1], (3, 3))

    def test_silence_produces_no_segments_and_no_progress(self):
        seen = []
        segments = seg.segment_audio(silence(3.0), RATE,
                                     progress=lambda done, total: seen.append(done))
        self.assertEqual(segments, [])
        self.assertEqual(seen, [])

    def test_a_segment_serialises_to_a_plain_dict(self):
        # character_store keeps these beside the audio, so they have to be
        # JSON-able without special handling.
        import json
        segments = seg.segment_audio(np.concatenate([tone(2.5)]), RATE)
        json.dumps(segments[0].as_dict())


class ReferenceSuitabilityTests(unittest.TestCase):
    """The 15-second engine cap, reported rather than enforced."""

    def _segment(self, duration, voiced=0.5):
        return seg.Segment(index=0, start_s=0.0, end_s=duration, duration_s=duration,
                           lufs=-20.0, peak_dbfs=-6.0, pitch_hz_mean=180.0,
                           pitch_hz_range=40.0, voiced_fraction=voiced, onset_rate_hz=4.0)

    def test_a_short_clip_fits_the_cap(self):
        self.assertTrue(self._segment(9.0).fits_the_reference_cap)

    def test_a_long_clip_does_not(self):
        # Not an error -- only the first 15 seconds will ever be used.
        self.assertFalse(self._segment(40.0).fits_the_reference_cap)

    def test_a_very_short_clip_is_flagged_as_too_short(self):
        self.assertFalse(self._segment(1.0).is_long_enough)

    def test_ranking_puts_usable_clips_first(self):
        over = self._segment(40.0)
        tiny = self._segment(1.0)
        good = self._segment(10.0, voiced=0.9)
        ranked = seg.rank_for_reference([over, tiny, good])
        self.assertIs(ranked[0], good)

    def test_ranking_prefers_the_more_voiced_of_two_usable_clips(self):
        quiet = self._segment(10.0, voiced=0.3)
        talky = self._segment(10.0, voiced=0.9)
        self.assertIs(seg.rank_for_reference([quiet, talky])[0], talky)

    def test_the_cap_outranks_voicing(self):
        # A 40-second clip is mostly wasted whatever its voiced fraction --
        # only the first 15 seconds is ever heard. Comparing two clips that
        # differ ONLY in whether they fit isolates that rule from the others.
        over = self._segment(40.0, voiced=0.99)
        under = self._segment(10.0, voiced=0.40)
        self.assertIs(seg.rank_for_reference([over, under])[0], under)

    def test_ranking_an_empty_list_is_empty(self):
        self.assertEqual(seg.rank_for_reference([]), [])


class RealAudioTests(unittest.TestCase):
    """One pass over actual speech, so the whole path runs on real input."""

    FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_prompt.wav")

    @unittest.skipUnless(os.path.isfile(FIXTURE), "sample_prompt.wav not present")
    def test_a_real_recording_yields_measured_segments(self):
        segments, rate = seg.load_and_segment(self.FIXTURE)
        self.assertGreater(rate, 0)
        self.assertGreaterEqual(len(segments), 1)
        first = segments[0]
        self.assertGreater(first.duration_s, 0.0)
        # Speech should measure a pitch somewhere in the human range.
        self.assertIsNotNone(first.pitch_hz_mean)
        self.assertGreater(first.pitch_hz_mean, seg.PITCH_FLOOR_HZ)
        self.assertLess(first.pitch_hz_mean, seg.PITCH_CEILING_HZ)
        self.assertIsNotNone(first.lufs)

    @unittest.skipUnless(os.path.isfile(FIXTURE), "sample_prompt.wav not present")
    def test_a_missing_file_raises_rather_than_returning_nothing(self):
        # Silently returning no segments would look like "this recording has
        # no speech in it", which is a different and misleading answer.
        with self.assertRaises(Exception):
            seg.load_and_segment(os.path.join(os.path.dirname(self.FIXTURE), "no-such.wav"))


if __name__ == "__main__":
    unittest.main()
