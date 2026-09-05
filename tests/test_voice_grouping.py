"""Tests for voice grouping stability and cleanup device routing.

Covers the 2026-09-05 fixes: one speaker no longer lists as many "voices"
(units, founder gating, cluster merging), and the cleanup tab can target a
specific GPU.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["webui.py"]

import numpy as np

import audio_cleanup_shared as shared
import audio_cleanup_worker as worker
import webui_audio_cleanup as cleanup


def _unit_vector(seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=192)
    return v / np.linalg.norm(v)


class AnalysisUnitsTests(unittest.TestCase):

    def test_close_spans_merge_into_one_unit(self):
        spans = [(0.0, 1.0), (1.3, 2.2), (2.5, 3.4)]
        self.assertEqual(worker._analysis_units(spans), [(0.0, 3.4)])

    def test_a_unit_stops_growing_at_the_target(self):
        long_first = [(0.0, shared.VOICE_UNIT_TARGET_SECONDS + 1.0),
                      (shared.VOICE_UNIT_TARGET_SECONDS + 1.2,
                       shared.VOICE_UNIT_TARGET_SECONDS + 2.0)]
        self.assertEqual(len(worker._analysis_units(long_first)), 2)

    def test_big_gaps_stay_separate(self):
        spans = [(0.0, 1.0), (5.0, 6.0)]
        self.assertEqual(worker._analysis_units(spans), spans)


class MergeSimilarClustersTests(unittest.TestCase):

    def test_same_direction_clusters_fold_together(self):
        base = _unit_vector(1)
        vectors = np.stack([base, base, base])
        labels = np.array([0, 0, 1])
        talk = {0: 4.0, 1: 2.0}
        merged_labels, merged_talk = worker._merge_similar_clusters(
            vectors, labels, talk)
        self.assertEqual(len(merged_talk), 1)
        self.assertAlmostEqual(merged_talk[0], 6.0)
        self.assertTrue((merged_labels == 0).all())

    def test_distinct_clusters_stay_apart(self):
        a, b = _unit_vector(1), _unit_vector(2)  # random vectors: near-zero sim
        vectors = np.stack([a, b])
        labels = np.array([0, 1])
        talk = {0: 4.0, 1: 3.0}
        _, merged_talk = worker._merge_similar_clusters(vectors, labels, talk)
        self.assertEqual(len(merged_talk), 2)


class FoundVoicesTests(unittest.TestCase):

    def test_short_units_join_the_founder_they_match(self):
        founder = _unit_vector(1)
        vectors = np.stack([founder, founder])
        units = [(0.0, 3.0), (10.0, 11.0)]     # one founder, one short
        labels, talk, notes = worker._found_voices(vectors, units)
        self.assertEqual(list(labels), [0, 0])
        self.assertAlmostEqual(talk[0], 4.0)
        self.assertEqual(notes, [])

    def test_unmatched_short_units_stay_off_the_list(self):
        founder, stranger = _unit_vector(1), _unit_vector(2)
        vectors = np.stack([founder, stranger])
        units = [(0.0, 3.0), (10.0, 11.0)]
        labels, talk, notes = worker._found_voices(vectors, units)
        self.assertEqual(labels[1], -1)
        self.assertAlmostEqual(talk[0], 3.0)
        self.assertEqual(len(notes), 1)
        self.assertIn("left off the list", notes[0])

    def test_no_founders_falls_back_with_a_warning(self):
        vectors = np.stack([_unit_vector(1)])
        units = [(0.0, 1.0)]
        labels, talk, notes = worker._found_voices(vectors, units)
        self.assertEqual(list(labels), [0])
        self.assertEqual(len(notes), 1)
        self.assertIn("rough guess", notes[0])


class CudaIndexTests(unittest.TestCase):

    def test_indexed_device_parses(self):
        self.assertEqual(shared.cuda_index("cuda:1"), 1)

    def test_everything_else_is_none(self):
        for value in ("cuda", "cpu", "cuda:x", "", None):
            self.assertIsNone(shared.cuda_index(value))


class WorkerDeviceRoutingTests(unittest.TestCase):

    BASE = ["--input", "x", "--output", "y", "--stages", "denoise",
            "--model-dir", "m"]

    def _run_main(self, device):
        argv = self.BASE + ["--device", device]
        with mock.patch.object(worker, "run_pipeline",
                               return_value={"ok": True, "output": "y"}):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                worker.main(argv)
                return os.environ.get("CUDA_VISIBLE_DEVICES")

    def test_cuda_index_narrows_visible_devices(self):
        self.assertEqual(self._run_main("cuda:1"), "1")

    def test_cpu_hides_every_device(self):
        self.assertEqual(self._run_main("cpu"), "")

    def test_bare_cuda_leaves_the_default(self):
        self.assertIsNone(self._run_main("cuda"))


class CleanupDeviceChoicesTests(unittest.TestCase):

    def test_cpu_is_always_last_and_cards_are_indexed(self):
        import webui_handlers
        choices = webui_handlers.cleanup_device_choices()
        self.assertEqual(choices[-1], ("CPU (slow)", shared.DEVICE_CPU))
        for _, value in choices[:-1]:
            self.assertRegex(value, r"^cuda:\d+$")

    def test_default_is_the_first_choice(self):
        import webui_handlers
        self.assertEqual(webui_handlers.cleanup_default_device(),
                         webui_handlers.cleanup_device_choices()[0][1])


class ProbeDeviceTests(unittest.TestCase):

    def test_cpu_short_circuits_without_a_subprocess(self):
        with mock.patch.object(cleanup.subprocess, "run") as run:
            self.assertEqual(cleanup._probe_device("py", shared.DEVICE_CPU),
                             "CPU")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
