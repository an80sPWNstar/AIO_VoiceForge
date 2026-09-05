"""Tests for voice selection worker modes and helper functions."""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio_cleanup_worker as worker


class TestParseArgs(unittest.TestCase):
    """Test CLI argument parsing."""

    def test_default_mode_is_clean(self):
        """Default mode should be 'clean'."""
        args = worker.parse_args(["--input", "x", "--output", "y", "--stages", "isolate",
                                   "--model-dir", "m"])
        self.assertEqual(args.mode, "clean")

    def test_analyze_mode_parses(self):
        """Analyze mode and its flags should parse."""
        args = worker.parse_args(["--mode", "analyze", "--input", "x", "--output", "y",
                                   "--stages", "isolate", "--model-dir", "m",
                                   "--voices-dir", "v"])
        self.assertEqual(args.mode, "analyze")
        self.assertEqual(args.voices_dir, "v")

    def test_extract_mode_parses(self):
        """Extract mode and its flags should parse."""
        args = worker.parse_args(["--mode", "extract", "--input", "x", "--output", "y",
                                   "--stages", "isolate", "--model-dir", "m",
                                   "--voices-dir", "v", "--voice-id", "2",
                                   "--reference-output", "r"])
        self.assertEqual(args.mode, "extract")
        self.assertEqual(args.voice_id, 2)
        self.assertEqual(args.reference_output, "r")

    def test_bad_mode_exits(self):
        """Parsing a bad mode should raise SystemExit."""
        with self.assertRaises(SystemExit):
            worker.parse_args(["--mode", "invalid", "--input", "x", "--output", "y",
                               "--stages", "isolate", "--model-dir", "m"])


class TestBestReferenceSpan(unittest.TestCase):
    """Test _best_reference_span reference selection."""

    def test_short_run_above_threshold(self):
        """All windows above threshold in one short run -> whole run."""
        scored = [(0.0, 1.5, 0.8), (0.75, 2.25, 0.85), (1.5, 3.0, 0.82)]
        start, end = worker._best_reference_span(scored, 0.7, 100.0)
        self.assertEqual(start, 0.0)
        self.assertLessEqual(end, 3.0)

    def test_long_run_picks_best_window(self):
        """Long run > 15s -> picks best 15s window, higher-scoring end."""
        import audio_cleanup_shared as shared
        scored = [(i * 0.75, (i + 2) * 0.75, 0.5 + i * 0.02) for i in range(40)]
        start, end = worker._best_reference_span(scored, 0.7, 100.0)
        self.assertLessEqual(end - start, shared.REFERENCE_CLIP_SECONDS + 0.01)

    def test_nothing_above_threshold(self):
        """Nothing above threshold -> highest single window."""
        scored = [(0.0, 1.5, 0.2), (1.5, 3.0, 0.3), (3.0, 4.5, 0.25)]
        start, end = worker._best_reference_span(scored, 0.7, 100.0)
        self.assertEqual(start, 1.5)
        self.assertEqual(end, 3.0)

    def test_two_runs_second_scores_higher(self):
        """Two runs, second scores higher -> second chosen."""
        scored = [(0.0, 1.5, 0.6), (1.5, 3.0, 0.5),
                  (5.0, 6.5, 0.9), (6.25, 7.75, 0.85)]
        start, end = worker._best_reference_span(scored, 0.7, 100.0)
        self.assertGreaterEqual(start, 5.0)


class TestClusterVoices(unittest.TestCase):
    """Test _cluster_voices refactor."""

    def test_cluster_two_voices(self):
        """Two obviously separated groups should cluster into two labels."""
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            self.skipTest("sklearn not available")

        e1 = np.array([1.0, 0.0, 0.0])
        e2 = np.array([0.0, 1.0, 0.0])

        vectors = np.array([
            e1 / np.linalg.norm(e1),
            e1 / np.linalg.norm(e1),
            e1 / np.linalg.norm(e1),
            e1 / np.linalg.norm(e1),
            e2 / np.linalg.norm(e2),
            e2 / np.linalg.norm(e2),
        ])

        spans = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
                 (5.0, 6.0), (6.0, 7.0)]

        labels, talk_time = worker._cluster_voices(vectors, spans)

        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(len(talk_time), 2)
        self.assertAlmostEqual(sum(talk_time.values()), 6.0)


class TestMainModeDispatching(unittest.TestCase):
    """Test main() dispatches to correct function by mode."""

    @patch("audio_cleanup_worker.run_pipeline")
    def test_clean_mode_dispatches_to_run_pipeline(self, mock_pipeline):
        """Clean mode should call run_pipeline."""
        mock_pipeline.return_value = {"ok": True, "output": "x"}
        result = worker.main([
            "--mode", "clean",
            "--input", "x",
            "--output", "y",
            "--stages", "isolate",
            "--model-dir", "m",
        ])
        mock_pipeline.assert_called_once()
        self.assertEqual(result, 0)

    @patch("audio_cleanup_worker.run_analyze")
    def test_analyze_mode_dispatches_to_run_analyze(self, mock_analyze):
        """Analyze mode should call run_analyze."""
        mock_analyze.return_value = {"ok": True, "output": "x", "voices": []}
        result = worker.main([
            "--mode", "analyze",
            "--input", "x",
            "--output", "y",
            "--stages", "isolate",
            "--model-dir", "m",
            "--voices-dir", "v",
        ])
        mock_analyze.assert_called_once()
        self.assertEqual(result, 0)

    @patch("audio_cleanup_worker.run_extract")
    def test_extract_mode_dispatches_to_run_extract(self, mock_extract):
        """Extract mode should call run_extract."""
        mock_extract.return_value = {"ok": True, "output": "x", "reference": "r"}
        result = worker.main([
            "--mode", "extract",
            "--input", "x",
            "--output", "y",
            "--stages", "isolate",
            "--model-dir", "m",
            "--voices-dir", "v",
            "--voice-id", "1",
            "--reference-output", "r",
        ])
        mock_extract.assert_called_once()
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
