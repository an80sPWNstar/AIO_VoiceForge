"""The Applio subprocess wrapper, without Applio.

Every test fakes the subprocess boundary and the filesystem; what is under
test is the part this repo owns -- command construction, the honesty rules
(a clean exit with no output file is a failure, artifacts are verified on
disk), and the error messages carrying what Applio actually said.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rvc_engine


def fake_run(returncode=0, stdout="", stderr=""):
    return mock.Mock(returncode=returncode, stdout=stdout, stderr=stderr)


class FakeApplio(unittest.TestCase):
    """A temp directory shaped like an Applio install."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, "env", "Scripts"), exist_ok=True)
        for rel in (("env", "Scripts", "python.exe"), ("core.py",)):
            with open(os.path.join(self.root, *rel), "wb") as handle:
                handle.write(b"x")

    def touch(self, *rel, content=b"x"):
        path = os.path.join(self.root, *rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(content)
        return path


class ConvertTests(FakeApplio):
    def setUp(self):
        super().setUp()
        self.wav = self.touch("in.wav")
        self.pth = self.touch("model.pth")
        self.index = self.touch("model.index")
        self.out = os.path.join(self.root, "out.wav")

    def test_builds_the_infer_command_with_the_stored_transpose(self):
        def run_and_write(cmd, **kwargs):
            with open(self.out, "wb") as handle:
                handle.write(b"x")
            self.cmd = cmd
            return fake_run()
        with mock.patch.object(subprocess, "run", side_effect=run_and_write):
            rvc_engine.convert(self.wav, self.out, self.pth, self.index,
                               transpose=-2, applio_root=self.root)
        self.assertIn("infer", self.cmd)
        pitch_at = self.cmd.index("--pitch")
        self.assertEqual(self.cmd[pitch_at + 1], "-2")

    def test_a_clean_exit_with_no_output_file_is_a_failure(self):
        with mock.patch.object(subprocess, "run", return_value=fake_run()):
            with self.assertRaises(rvc_engine.RVCError) as caught:
                rvc_engine.convert(self.wav, self.out, self.pth, self.index,
                                   applio_root=self.root)
        self.assertIn("no file", str(caught.exception))

    def test_a_failed_call_carries_what_applio_said(self):
        with mock.patch.object(subprocess, "run",
                               return_value=fake_run(1, stderr="CUDA out of memory")):
            with self.assertRaises(rvc_engine.RVCError) as caught:
                rvc_engine.convert(self.wav, self.out, self.pth, self.index,
                                   applio_root=self.root)
        self.assertIn("CUDA out of memory", str(caught.exception))

    def test_a_missing_model_fails_before_any_subprocess_runs(self):
        with mock.patch.object(subprocess, "run") as run:
            with self.assertRaises(rvc_engine.RVCError):
                rvc_engine.convert(self.wav, self.out,
                                   os.path.join(self.root, "nope.pth"),
                                   self.index, applio_root=self.root)
        run.assert_not_called()

    def test_a_missing_install_names_what_is_missing(self):
        os.remove(os.path.join(self.root, "core.py"))
        with self.assertRaises(rvc_engine.RVCError) as caught:
            rvc_engine.convert(self.wav, self.out, self.pth, self.index,
                               applio_root=self.root)
        self.assertIn("core.py", str(caught.exception))


class TrainTests(FakeApplio):
    def setUp(self):
        super().setUp()
        self.dataset = os.path.join(self.root, "dataset")
        os.makedirs(self.dataset)

    def test_runs_the_four_steps_in_order_and_returns_verified_artifacts(self):
        # 90e beside 100e: the pick must compare epochs as numbers -- a
        # lexicographic sort chose 90e the first time a run crossed three
        # digits.
        self.touch("logs", "narrator", "narrator_90e_2070s.pth")
        self.touch("logs", "narrator", "narrator_200e_4600s.pth")
        self.touch("logs", "narrator", "trained.index")
        calls = []
        with mock.patch.object(subprocess, "run",
                               side_effect=lambda cmd, **k: calls.append(cmd) or fake_run()):
            artifacts = rvc_engine.train("narrator", self.dataset,
                                         applio_root=self.root)
        steps = [cmd[2] for cmd in calls]
        self.assertEqual(steps, ["preprocess", "extract", "train", "index"])
        self.assertTrue(artifacts["model_path"].endswith("narrator_200e_4600s.pth"))
        self.assertTrue(artifacts["index_path"].endswith("trained.index"))

    def test_a_clean_chain_with_no_weights_on_disk_is_a_failure(self):
        with mock.patch.object(subprocess, "run", return_value=fake_run()):
            with self.assertRaises(rvc_engine.RVCError) as caught:
                rvc_engine.train("narrator", self.dataset, applio_root=self.root)
        self.assertIn("no usable artifacts", str(caught.exception))

    def test_a_failing_step_stops_the_chain(self):
        with mock.patch.object(
                subprocess, "run",
                side_effect=[fake_run(), fake_run(2, stderr="extract blew up")]) as run:
            with self.assertRaises(rvc_engine.RVCError) as caught:
                rvc_engine.train("narrator", self.dataset, applio_root=self.root)
        self.assertEqual(run.call_count, 2)
        self.assertIn("extract blew up", str(caught.exception))

    def test_a_missing_dataset_fails_before_any_subprocess_runs(self):
        with mock.patch.object(subprocess, "run") as run:
            with self.assertRaises(rvc_engine.RVCError):
                rvc_engine.train("narrator",
                                 os.path.join(self.root, "no-dataset"),
                                 applio_root=self.root)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
