"""Tests for voice selection driver and UI handlers.

Unit tests for build_analyze_command, build_extract_command, analysis_voices_dir,
run_voice_analysis, run_voice_extract, and the related UI handlers.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# webui_handlers' import chain parses sys.argv at import time.
sys.argv = ["webui.py"]

import audio_cleanup_shared as cleanup_shared
import character_store as store
import webui_audio_cleanup as audio_cleanup
import webui_handlers
import webui_media_fetch as media_fetch


class TestCommandBuilders(unittest.TestCase):
    """Test build_analyze_command and build_extract_command."""

    def test_build_analyze_command_structure(self):
        """Verify analyze command has exactly the right flags."""
        command = audio_cleanup.build_analyze_command(
            python_path="python.exe",
            input_path="/input.wav",
            output_path="/output.wav",
            voices_dir="/voices",
            stages=["isolate", "dereverb"],
            progress_path="/progress.jsonl",
            vocal_model="vocals",
            dereverb_model="dereverb",
            denoise_model="denoise",
            device="cuda:0",
        )

        self.assertIn("python.exe", command)
        self.assertIn("--mode", command)
        mode_idx = command.index("--mode")
        self.assertEqual(command[mode_idx + 1], "analyze")
        self.assertIn("--input", command)
        self.assertIn("/input.wav", command)
        self.assertIn("--output", command)
        self.assertIn("/output.wav", command)
        self.assertIn("--voices-dir", command)
        self.assertIn("/voices", command)
        self.assertIn("--stages", command)
        self.assertIn("isolate,dereverb", command)

    def test_build_extract_command_structure(self):
        """Verify extract command has exactly the right flags."""
        command = audio_cleanup.build_extract_command(
            python_path="python.exe",
            input_path="/processed.wav",
            output_path="/output.wav",
            reference_output="/ref.wav",
            voices_dir="/voices",
            voice_id=2,
            stages=["trim", "normalize"],
            progress_path="/progress.jsonl",
            speaker_threshold=0.5,
            sample_rate=16000,
            channel_mode="mono",
            device="cpu",
        )

        self.assertIn("--mode", command)
        mode_idx = command.index("--mode")
        self.assertEqual(command[mode_idx + 1], "extract")
        self.assertIn("--voice-id", command)
        voice_id_idx = command.index("--voice-id")
        self.assertEqual(command[voice_id_idx + 1], "2")
        self.assertIn("--reference-output", command)
        ref_idx = command.index("--reference-output")
        self.assertEqual(command[ref_idx + 1], "/ref.wav")


class TestAnalysisVoicesDir(unittest.TestCase):
    """Test analysis_voices_dir directory creation."""

    def setUp(self):
        self.temp_root = tempfile.mkdtemp(prefix="test_voices_")

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_creates_directory(self):
        """Verify analysis_voices_dir creates the directory."""
        result = audio_cleanup.analysis_voices_dir(self.temp_root, "/path/to/audio.wav")
        self.assertTrue(os.path.isdir(result))

    def test_unique_on_duplicate_stem(self):
        """Same source stem should get numeric suffixes."""
        path1 = audio_cleanup.analysis_voices_dir(self.temp_root, "/path/to/audio.wav")
        path2 = audio_cleanup.analysis_voices_dir(self.temp_root, "/path/to/audio.wav")
        self.assertNotEqual(path1, path2)
        self.assertTrue(os.path.isdir(path1))
        self.assertTrue(os.path.isdir(path2))


class TestRunVoiceAnalysisValidation(unittest.TestCase):
    """Test run_voice_analysis validation errors."""

    def test_missing_input_file(self):
        """Should raise CleanupError when input file does not exist."""
        with self.assertRaises(audio_cleanup.CleanupError) as ctx:
            audio_cleanup.run_voice_analysis(
                input_path="/nonexistent.wav",
                output_root="/output",
                stages=["isolate"],
            )
        self.assertIn("Download or extract", str(ctx.exception))

    @patch("webui_audio_cleanup.sidecar_python")
    def test_missing_sidecar(self, mock_sidecar):
        """Should raise CleanupError with SETUP_HINT when sidecar is None."""
        mock_sidecar.return_value = None
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            with self.assertRaises(audio_cleanup.CleanupError) as ctx:
                audio_cleanup.run_voice_analysis(
                    input_path=f.name,
                    output_root="/output",
                    stages=["isolate"],
                )
            self.assertIn(audio_cleanup.SETUP_HINT, str(ctx.exception))


class TestRunVoiceExtractValidation(unittest.TestCase):
    """Test run_voice_extract validation errors."""

    def test_missing_processed_wav(self):
        """Should raise CleanupError when processed.wav is missing."""
        temp_voices_dir = tempfile.mkdtemp(prefix="test_extract_")
        try:
            with self.assertRaises(audio_cleanup.CleanupError) as ctx:
                audio_cleanup.run_voice_extract(
                    voices_dir=temp_voices_dir,
                    voice_id=0,
                    source_name="audio.wav",
                    output_root="/output",
                    stages=["trim"],
                )
            self.assertIn("cached voices are gone", str(ctx.exception))
        finally:
            shutil.rmtree(temp_voices_dir, ignore_errors=True)


class TestVoiceChoiceLabel(unittest.TestCase):
    """Test voice_choice_label formatting."""

    def test_formats_correctly(self):
        """Verify the label format."""
        voice = {"id": 2, "share": 0.31, "talk_seconds": 48.0}
        label = webui_handlers.voice_choice_label(voice)
        self.assertIn("Voice 2", label)
        self.assertIn("31%", label)
        self.assertIn("48s", label)


class TestVoiceRowUpdates(unittest.TestCase):
    """Test voice_row_updates hides/shows correctly."""

    def test_visible_for_real_voices(self):
        """Real voices should have visible=True and a preview path."""
        voices = [
            {"id": 0, "preview": "/path/preview0.wav"},
            {"id": 1, "preview": "/path/preview1.wav"},
        ]
        updates = webui_handlers.voice_row_updates(voices)
        self.assertEqual(len(updates), webui_handlers.MAX_VOICE_ROWS)
        for i in range(len(voices)):
            self.assertEqual(updates[i]["visible"], True)
            self.assertEqual(updates[i]["value"], voices[i]["preview"])
        for i in range(len(voices), webui_handlers.MAX_VOICE_ROWS):
            self.assertEqual(updates[i]["visible"], False)
            self.assertIsNone(updates[i]["value"])


class TestVoiceIdFromChoice(unittest.TestCase):
    """Test voice_id_from_choice mapping."""

    def test_roundtrip_mapping(self):
        """Should map label back to voice id."""
        voice = {"id": 3, "share": 0.5, "talk_seconds": 30.0}
        state = {"voices": [voice]}
        label = webui_handlers.voice_choice_label(voice)
        voice_id = webui_handlers.voice_id_from_choice(state, label)
        self.assertEqual(voice_id, 3)

    def test_unmapped_choice(self):
        """Should return None for unknown choice."""
        state = {"voices": [{"id": 0}]}
        result = webui_handlers.voice_id_from_choice(state, "unknown")
        self.assertIsNone(result)

    def test_none_state(self):
        """Should return None when state is None."""
        result = webui_handlers.voice_id_from_choice(None, "any")
        self.assertIsNone(result)


class TestSaveVoiceReferenceUi(unittest.TestCase):
    """Test save_voice_reference_ui guards and delegation."""

    @patch("webui_handlers.characters.refresh_panel")
    def test_rvc_mode_guard(self, mock_refresh):
        """Should refuse RVC mode."""
        mock_refresh.return_value = ("a", "b", "c", "d")
        webui_handlers.save_voice_reference_ui(
            store.MODE_RVC, "slug", "/path.wav", "label"
        )
        mock_refresh.assert_called_once()
        call_args = mock_refresh.call_args[0]
        self.assertIn("RVC voice holds a model", call_args[2])

    @patch("webui_handlers.characters.refresh_panel")
    def test_empty_slug_guard(self, mock_refresh):
        """Should refuse empty slug."""
        mock_refresh.return_value = ("a", "b", "c", "d")
        webui_handlers.save_voice_reference_ui(
            store.MODE_ONESHOT, "", "/path.wav", "label"
        )
        mock_refresh.assert_called_once()
        call_args = mock_refresh.call_args[0]
        self.assertIn("Select a voice", call_args[2])

    @patch("webui_handlers.characters.refresh_panel")
    def test_missing_path_guard(self, mock_refresh):
        """Should refuse missing or nonexistent reference_path."""
        mock_refresh.return_value = ("a", "b", "c", "d")
        webui_handlers.save_voice_reference_ui(
            store.MODE_ONESHOT, "slug", "/nonexistent.wav", "label"
        )
        mock_refresh.assert_called_once()
        call_args = mock_refresh.call_args[0]
        self.assertIn("Extract a voice", call_args[2])

    @patch("webui_handlers.characters.add_reference_to_character_ui")
    def test_calls_add_reference(self, mock_add_ref):
        """Should call add_reference_to_character_ui on success."""
        mock_add_ref.return_value = ("a", "b", "c", "d")
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            result = webui_handlers.save_voice_reference_ui(
                store.MODE_ONESHOT, "slug", f.name, "label"
            )
            mock_add_ref.assert_called_once()
            call_args = mock_add_ref.call_args[0]
            self.assertEqual(call_args[0], store.MODE_ONESHOT)
            self.assertEqual(call_args[1], "slug")
            self.assertEqual(call_args[2], f.name)
            self.assertEqual(call_args[3], "label")


class TestCleanupRunUiAuto(unittest.TestCase):
    """Test cleanup_run_ui in AUTO mode."""

    @patch("webui_handlers.audio_cleanup.run_cleanup")
    def test_auto_mode_still_works(self, mock_run_cleanup):
        """AUTO mode should still work end-to-end."""
        mock_run_cleanup.return_value = {
            "audio_path": "/output.wav",
            "notes": ["Note 1"],
            "log": ["Log 1"],
        }

        gen = webui_handlers.cleanup_run_ui(
            fetched_path="/input.wav",
            uploaded_path=None,
            stages=["isolate"],
            vocal_model="vocals",
            dereverb_model="dereverb",
            denoise_model="denoise",
            speaker_mode="dominant",
            speaker_sample=None,
            speaker_threshold=0.5,
            sample_rate=16000,
            channel_mode="mono",
            device="cpu",
            keep_intermediates=False,
            voice_mode=webui_handlers.VOICE_MODE_AUTO,
        )

        items = list(gen)
        self.assertGreater(len(items), 0)
        final_yield = items[-1]
        self.assertEqual(len(final_yield), 5 + 1 + 1 + 1 + webui_handlers.MAX_VOICE_ROWS + webui_handlers.MAX_VOICE_ROWS)
        voices_group_update = final_yield[7]
        self.assertEqual(voices_group_update["visible"], False)


class TestCleanupRunUiManual(unittest.TestCase):
    """Test cleanup_run_ui in MANUAL mode."""

    @patch("webui_handlers.audio_cleanup.run_voice_analysis")
    def test_manual_mode_analysis(self, mock_run_analysis):
        """MANUAL mode should call run_voice_analysis and show voices."""
        mock_run_analysis.return_value = {
            "voices_dir": "/voices_dir",
            "duration": 60.0,
            "voices": [
                {"id": 0, "share": 0.6, "talk_seconds": 36.0, "preview": "/preview0.wav"},
                {"id": 1, "share": 0.4, "talk_seconds": 24.0, "preview": "/preview1.wav"},
            ],
            "notes": ["Found 2 voices"],
            "log": ["Analysis done"],
        }

        gen = webui_handlers.cleanup_run_ui(
            fetched_path="/input.wav",
            uploaded_path=None,
            stages=["isolate"],
            vocal_model="vocals",
            dereverb_model="dereverb",
            denoise_model="denoise",
            speaker_mode="dominant",
            speaker_sample=None,
            speaker_threshold=0.5,
            sample_rate=16000,
            channel_mode="mono",
            device="cpu",
            keep_intermediates=False,
            voice_mode=webui_handlers.VOICE_MODE_MANUAL,
        )

        items = list(gen)
        self.assertGreater(len(items), 0)
        final_yield = items[-1]

        voices_state_update = final_yield[5]
        state_value = voices_state_update["value"]
        self.assertEqual(state_value["voices_dir"], "/voices_dir")
        self.assertEqual(len(state_value["voices"]), 2)
        self.assertEqual(len(state_value["choices"]), 2)
        self.assertIn("Voice 0", state_value["choices"][0])
        self.assertIn("60%", state_value["choices"][0])

        voices_group_update = final_yield[7]
        self.assertEqual(voices_group_update["visible"], True)

        voice_radio_update = final_yield[6]
        # Check that choices include voice labels
        self.assertGreater(len(voice_radio_update["choices"]), 0)
        self.assertIn("Voice 0", voice_radio_update["choices"][0])


class TestVoiceExtractRunUi(unittest.TestCase):
    """Test voice_extract_run_ui."""

    @patch("webui_handlers.audio_cleanup.run_voice_extract")
    def test_happy_path(self, mock_run_extract):
        """Should extract voice successfully."""
        mock_run_extract.return_value = {
            "audio_path": "/extracted.wav",
            "reference_path": "/reference.wav",
            "notes": ["Extracted"],
            "log": ["Extract log"],
        }

        voices = [
            {"id": 0, "share": 0.6, "talk_seconds": 36.0, "preview": "/preview0.wav"}
        ]
        state = {
            "voices_dir": "/voices_dir",
            "voices": voices,
            "source_name": "audio.wav",
            "choices": [webui_handlers.voice_choice_label(v) for v in voices],
        }
        choice = webui_handlers.voice_choice_label(voices[0])

        gen = webui_handlers.voice_extract_run_ui(
            voices_state=state,
            choice=choice,
            stages=["trim"],
            speaker_threshold=0.5,
            sample_rate=16000,
            channel_mode="mono",
            device="cpu",
        )

        items = list(gen)
        # Should have multiple yields (progress + final)
        self.assertGreater(len(items), 0)
        # Verify the mock was called with correct arguments
        mock_run_extract.assert_called_once()
        call_args = mock_run_extract.call_args
        self.assertEqual(call_args[1]["voice_id"], 0)
        self.assertEqual(call_args[1]["source_name"], "audio.wav")

    @patch("webui_handlers.audio_cleanup.run_voice_extract")
    def test_pick_voice_first(self, mock_run_extract):
        """Should show error when no voice is picked."""
        state = {
            "voices_dir": "/voices_dir",
            "voices": [{"id": 0}],
            "source_name": "audio.wav",
            "choices": ["Voice 0"],
        }

        gen = webui_handlers.voice_extract_run_ui(
            voices_state=state,
            choice=None,
            stages=["trim"],
            speaker_threshold=0.5,
            sample_rate=16000,
            channel_mode="mono",
            device="cpu",
        )

        items = list(gen)
        # Should have multiple yields (at least the error one)
        self.assertGreater(len(items), 0)
        final_yield = items[-1]
        notes = final_yield[5].get("value", "")
        self.assertIn("Pick a voice", notes)


if __name__ == "__main__":
    unittest.main()
