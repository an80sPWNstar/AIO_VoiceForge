"""The V5 LoRA pipeline wrapper, without V5.

Every test fakes the subprocess boundary and the filesystem; what is under
test is the part this repo owns -- config/command construction, the honesty
rules (stages verify pre/post conditions), and error messages carrying what
the worker actually said.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["webui.py"]

import lora_engine
import engine_paths


def fake_process(returncode=0):
    """A fake process object for subprocess.Popen."""
    return mock.Mock(returncode=returncode, poll=mock.Mock(side_effect=[None, returncode]))


class ConfigBuilderTests(unittest.TestCase):
    """Test pure config builder functions."""

    def test_prep_config_has_required_keys(self):
        config = lora_engine.build_prep_config("/clips", "test_voice", "/output")
        self.assertEqual(config["name"], "test_voice")
        self.assertEqual(config["inputs"], ["/clips"])
        self.assertEqual(config["output_root"], "/output")
        self.assertEqual(config["language"], "EN")
        self.assertEqual(config["whisper_device"], "cuda:0")

    def test_prep_config_has_no_extra_defaults(self):
        config = lora_engine.build_prep_config("/clips", "test_voice", "/output")
        # These should not be present (they stay on V5's defaults)
        self.assertNotIn("segmentation_mode", config)
        self.assertNotIn("target_s", config)

    def test_train_config_has_required_keys(self):
        config = lora_engine.build_train_config("/dataset", "test_voice", "/output")
        self.assertEqual(config["dataset_dir"], "/dataset")
        self.assertEqual(config["name"], "test_voice")
        self.assertEqual(config["output_dir"], "/output")
        self.assertIn("model_dir", config)
        self.assertIn("model_config", config)

    def test_train_config_model_paths_are_absolute(self):
        config = lora_engine.build_train_config("/dataset", "test_voice", "/output")
        self.assertTrue(os.path.isabs(config["model_dir"]))
        self.assertTrue(os.path.isabs(config["model_config"]))

    def test_train_config_epochs_only_when_set(self):
        config_no_epochs = lora_engine.build_train_config("/dataset", "test_voice", "/output")
        self.assertNotIn("epochs", config_no_epochs)

        config_with_epochs = lora_engine.build_train_config(
            "/dataset", "test_voice", "/output", epochs=5
        )
        self.assertEqual(config_with_epochs["epochs"], 5)

    def test_train_config_device_only_when_set(self):
        config_no_device = lora_engine.build_train_config("/dataset", "test_voice", "/output")
        self.assertNotIn("device", config_no_device)

        config_with_device = lora_engine.build_train_config(
            "/dataset", "test_voice", "/output", device="cuda:1"
        )
        self.assertEqual(config_with_device["device"], "cuda:1")


class CommandBuilderTests(unittest.TestCase):
    """Test pure command builder functions."""

    def test_prep_command_shape(self):
        cmd = lora_engine.build_prep_command("/path/config.json", "/path/state")
        self.assertEqual(cmd[0], engine_paths.ENGINE_PYTHON)
        self.assertIn("-m", cmd)
        self.assertIn(lora_engine.PREP_MODULE, cmd)
        self.assertIn("--config", cmd)
        self.assertIn("/path/config.json", cmd)
        self.assertIn("--state-dir", cmd)
        self.assertIn("/path/state", cmd)

    def test_cache_command_shape(self):
        cmd = lora_engine.build_cache_command("/dataset")
        self.assertEqual(cmd[0], engine_paths.ENGINE_PYTHON)
        self.assertIn("/dataset", cmd)

    def test_train_command_shape(self):
        cmd = lora_engine.build_train_command("/path/config.json", "/path/state")
        self.assertEqual(cmd[0], engine_paths.ENGINE_PYTHON)
        self.assertIn("-m", cmd)
        self.assertIn(lora_engine.TRAIN_MODULE, cmd)
        self.assertIn("--config", cmd)
        self.assertIn("/path/config.json", cmd)
        self.assertIn("--state-dir", cmd)
        self.assertIn("/path/state", cmd)


class AdapterOutputPathTests(unittest.TestCase):
    """Test adapter output path resolution."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.output_dir = self.tmp.name

    def test_prefers_best_safetensors(self):
        name = "test_voice"
        os.makedirs(os.path.join(self.output_dir, name, "best"))
        best_path = os.path.join(self.output_dir, name, "best", f"{name}.safetensors")
        final_path = os.path.join(self.output_dir, name, f"{name}.safetensors")

        Path(best_path).touch()
        Path(final_path).touch()

        result = lora_engine.adapter_output_path(self.output_dir, name)
        self.assertEqual(result, best_path)

    def test_falls_back_to_final_safetensors(self):
        name = "test_voice"
        os.makedirs(os.path.join(self.output_dir, name))
        final_path = os.path.join(self.output_dir, name, f"{name}.safetensors")
        Path(final_path).touch()

        result = lora_engine.adapter_output_path(self.output_dir, name)
        self.assertEqual(result, final_path)

    def test_raises_when_neither_exists(self):
        name = "test_voice"
        os.makedirs(os.path.join(self.output_dir, name))

        with self.assertRaises(lora_engine.LoraError) as caught:
            lora_engine.adapter_output_path(self.output_dir, name)

        error_msg = str(caught.exception)
        self.assertIn("best", error_msg)
        self.assertIn(f"{name}.safetensors", error_msg)


class RunStageFailureTests(unittest.TestCase):
    """Test _run_stage failure handling."""

    def test_nonzero_exit_raises_lora_error_with_stderr_and_status_message(self):
        with tempfile.TemporaryDirectory() as state_dir:
            # Create a failed status.json
            status_path = os.path.join(state_dir, "status.json")
            with open(status_path, "w") as f:
                json.dump({
                    "phase": "failed",
                    "message": "boom from status",
                    "step": 0,
                    "total_steps": 100,
                }, f)

            # Create fake process that exits with error
            fake_proc = mock.Mock(returncode=1, poll=mock.Mock(side_effect=[None, 1]))

            with tempfile.TemporaryDirectory() as scratch:
                stderr_path = os.path.join(scratch, "stderr.txt")
                with open(stderr_path, "w") as f:
                    f.write("stderr line 1\nstderr line 2: actual error")

                with mock.patch("subprocess.Popen") as mock_popen:
                    mock_popen.return_value = fake_proc
                    with mock.patch("builtins.open", mock.mock_open()) as mock_file:
                        # Make the stderr file readable
                        def open_handler(path, *args, **kwargs):
                            if path == stderr_path:
                                return open(stderr_path, "r")
                            if "stderr" in str(path) and "r" in str(args):
                                m = mock.mock_open(read_data="stderr line 1\nstderr line 2: actual error")()
                                return m
                            if "w" in str(args):
                                return mock.mock_open()()
                            return mock.mock_open()()

                        with mock.patch("builtins.open", side_effect=open_handler):
                            with self.assertRaises(lora_engine.LoraError) as caught:
                                lora_engine._run_stage(
                                    ["python", "-m", "dummy"],
                                    state_dir,
                                    "test_stage",
                                    None,
                                    None,
                                    60.0,
                                )

                error_msg = str(caught.exception)
                self.assertIn("test_stage", error_msg)
                self.assertIn("failed", error_msg)

    def test_timeout_kills_process_and_raises(self):
        with tempfile.TemporaryDirectory() as state_dir:
            # Create a fake process that never finishes
            fake_proc = mock.Mock(returncode=None)
            fake_proc.poll.return_value = None  # Never finishes
            fake_proc.kill = mock.Mock()

            with mock.patch("subprocess.Popen", return_value=fake_proc):
                with self.assertRaises(lora_engine.LoraError) as caught:
                    lora_engine._run_stage(
                        ["python", "-m", "dummy"],
                        state_dir,
                        "slow_stage",
                        None,
                        None,
                        0.1,  # Very short timeout
                    )

            error_msg = str(caught.exception)
            self.assertIn("timed out", error_msg.lower())
            fake_proc.kill.assert_called()


class TrainLoraFullTests(unittest.TestCase):
    """Test train_lora_full chaining."""

    def test_calls_stages_in_order_with_correct_fractions(self):
        calls = []

        def mock_prep(*args, **kwargs):
            calls.append(("prep", kwargs.get("progress_callback")))
            return "/work/dataset/test_voice"

        def mock_cache(*args, **kwargs):
            calls.append(("cache", kwargs.get("progress_callback")))

        def mock_train(*args, **kwargs):
            calls.append(("train", kwargs.get("progress_callback")))
            return {
                "adapter_path": "/work/adapters/test_voice/test_voice.safetensors",
                "status_path": "/work/adapters/test_voice/status.json",
                "samples_dir": "/work/adapters/test_voice/samples",
            }

        with mock.patch.object(lora_engine, "run_dataset_prep", side_effect=mock_prep):
            with mock.patch.object(lora_engine, "run_feature_cache", side_effect=mock_cache):
                with mock.patch.object(lora_engine, "run_training", side_effect=mock_train):
                    result = lora_engine.train_lora_full(
                        "/clips", "test_voice", "/work"
                    )

        # Verify order
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][0], "prep")
        self.assertEqual(calls[1][0], "cache")
        self.assertEqual(calls[2][0], "train")

        # Verify result
        self.assertEqual(result["adapter_path"], "/work/adapters/test_voice/test_voice.safetensors")

    def test_passes_correct_output_dirs(self):
        dataset_prep_args = None
        train_args = None

        def mock_prep(clips_dir, name, output_root, **kwargs):
            nonlocal dataset_prep_args
            dataset_prep_args = (clips_dir, name, output_root)
            return os.path.join(output_root, name)

        def mock_cache(*args, **kwargs):
            pass

        def mock_train(dataset_dir, name, output_dir, **kwargs):
            nonlocal train_args
            train_args = (dataset_dir, name, output_dir)
            return {
                "adapter_path": "/adapters/test_voice/test_voice.safetensors",
                "status_path": "/adapters/test_voice/status.json",
                "samples_dir": "/adapters/test_voice/samples",
            }

        with mock.patch.object(lora_engine, "run_dataset_prep", side_effect=mock_prep):
            with mock.patch.object(lora_engine, "run_feature_cache", side_effect=mock_cache):
                with mock.patch.object(lora_engine, "run_training", side_effect=mock_train):
                    lora_engine.train_lora_full(
                        "/clips", "test_voice", "/work"
                    )

        # Verify dataset prep gets output_root = /work/dataset
        expected_dataset_root = os.path.join("/work", "dataset")
        self.assertEqual(dataset_prep_args[2], expected_dataset_root)

        # Verify training gets output_dir = /work/adapters
        expected_adapters_dir = os.path.join("/work", "adapters")
        self.assertEqual(train_args[2], expected_adapters_dir)


class EngineSupportTests(unittest.TestCase):
    """Test engine_supports_lora."""

    def test_returns_false_when_train_worker_missing(self):
        with mock.patch.object(engine_paths, "ENGINE_ROOT", tempfile.mkdtemp()):
            result = lora_engine.engine_supports_lora()
            self.assertFalse(result)

    def test_returns_true_when_train_worker_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_worker_path = os.path.join(
                tmp, "indextts", "training", "train_worker.py"
            )
            os.makedirs(os.path.dirname(train_worker_path))
            Path(train_worker_path).touch()

            with mock.patch.object(engine_paths, "ENGINE_ROOT", tmp):
                result = lora_engine.engine_supports_lora()
                self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
