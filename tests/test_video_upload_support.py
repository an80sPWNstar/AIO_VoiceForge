"""Tests for video file upload support on cleanup and segmentation tabs."""

import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["webui.py"]

import webui_media_utils as media_utils
import webui_audio_cleanup as cleanup


class IsVideoFileTests(unittest.TestCase):
    """Tests for is_video_file helper."""

    def test_is_video_file_true_for_mp4(self):
        self.assertTrue(media_utils.is_video_file("test.mp4"))

    def test_is_video_file_true_for_mov(self):
        self.assertTrue(media_utils.is_video_file("test.mov"))

    def test_is_video_file_true_for_uppercase(self):
        self.assertTrue(media_utils.is_video_file("test.MP4"))

    def test_is_video_file_false_for_wav(self):
        self.assertFalse(media_utils.is_video_file("test.wav"))

    def test_is_video_file_false_for_empty_string(self):
        self.assertFalse(media_utils.is_video_file(""))

    def test_is_video_file_false_for_none(self):
        self.assertFalse(media_utils.is_video_file(None))


class EnsureAudioFileTests(unittest.TestCase):
    """Tests for ensure_audio_file helper."""

    def test_ensure_audio_file_returns_wav_unchanged(self):
        with mock.patch('webui_media_utils.extract_audio_from_media') as mock_extract:
            result = media_utils.ensure_audio_file("test.wav")
            self.assertEqual(result, "test.wav")
            mock_extract.assert_not_called()

    def test_ensure_audio_file_extracts_mp4(self):
        fake_wav = "/tmp/extracted.wav"
        with mock.patch('webui_media_utils.extract_audio_from_media', return_value=fake_wav) as mock_extract:
            result = media_utils.ensure_audio_file("test.mp4")
            self.assertEqual(result, fake_wav)
            mock_extract.assert_called_once()
            args, kwargs = mock_extract.call_args
            self.assertEqual(args[0], "test.mp4")
            self.assertIsNone(kwargs.get("channels", "missing"))

    def test_ensure_audio_file_raises_on_extraction_failure(self):
        with mock.patch('webui_media_utils.extract_audio_from_media', return_value=None):
            with self.assertRaises(ValueError) as cm:
                media_utils.ensure_audio_file("test.mp4")
            self.assertIn("Could not extract audio", str(cm.exception))

    def test_ensure_audio_file_uses_sample_rate_parameter(self):
        fake_wav = "/tmp/extracted.wav"
        with mock.patch('webui_media_utils.extract_audio_from_media', return_value=fake_wav) as mock_extract:
            media_utils.ensure_audio_file("test.mp4", sample_rate=48000)
            mock_extract.assert_called_once()
            args, kwargs = mock_extract.call_args
            self.assertEqual(kwargs.get("sample_rate"), 48000)

    def test_extract_command_keeps_source_channels_when_channels_is_none(self):
        with mock.patch('webui_media_utils.subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="stop here")
            media_utils.extract_audio_from_media("clip.mp4", channels=None)
            cmd = mock_run.call_args[0][0]
            self.assertNotIn("-ac", cmd)

    def test_extract_command_downmixes_to_mono_by_default(self):
        with mock.patch('webui_media_utils.subprocess.run') as mock_run:
            mock_run.return_value = mock.Mock(returncode=1, stderr="stop here")
            media_utils.extract_audio_from_media("clip.mp4")
            cmd = mock_run.call_args[0][0]
            self.assertIn("-ac", cmd)
            self.assertEqual(cmd[cmd.index("-ac") + 1], "1")


class RunCleanupVideoTests(unittest.TestCase):
    """Tests for run_cleanup with video files."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = self._dir.name
        self.addCleanup(self._dir.cleanup)

    def test_run_cleanup_raises_cleanup_error_when_video_extraction_fails(self):
        clip_path = os.path.join(self.dir, "clip.mov")
        Path(clip_path).write_bytes(b"fake video")

        with mock.patch('webui_audio_cleanup.sidecar_python', return_value="/fake/python"):
            with mock.patch('webui_audio_cleanup.media_fetch.ffmpeg_available', return_value=True):
                with mock.patch('webui_audio_cleanup.media_utils.ensure_audio_file', side_effect=ValueError("extraction failed")):
                    with self.assertRaises(cleanup.CleanupError) as cm:
                        cleanup.run_cleanup(
                            input_path=clip_path,
                            output_root=self.dir,
                            stages=["denoise"],
                        )
                    self.assertIn("extraction failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
