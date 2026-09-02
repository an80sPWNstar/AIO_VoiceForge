"""The cancel path, and the device the next generation will load on.

Generation happens in a separate process, so pressing Cancel cannot reach
into it -- the app tracks just enough about the running job in one dict to
be able to kill the worker and mark the task. That dict is the whole
mechanism, and these pin down what it does when there is nothing running,
when there is, and when a cancel and a completion race each other.

No GPU and no engine: WORKER.kill() returns False when nothing is running,
which is exactly the state every test here leaves it in.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["webui.py"]

import webui_generation as gen
from webui_runtime import DEVICE_AUTO, DEVICE_CPU, DeviceSelection


class SubprocessStateTests(unittest.TestCase):
    """Registering a running job, and clearing it again."""

    def tearDown(self):
        gen._clear_subprocess_state()

    def test_state_starts_idle(self):
        gen._clear_subprocess_state()
        self.assertFalse(gen._SUBPROCESS_STATE["active"])

    def test_registering_marks_it_active_and_records_the_task(self):
        gen._register_subprocess_state("/tmp/meta.json", "task-7")
        self.assertTrue(gen._SUBPROCESS_STATE["active"])
        self.assertEqual(gen._SUBPROCESS_STATE["metadata_path"], "/tmp/meta.json")
        self.assertEqual(gen._SUBPROCESS_STATE["task_id"], "task-7")

    def test_clearing_returns_what_was_there_and_resets(self):
        # The caller reads the snapshot to find out whether the job it just
        # finished had been canceled underneath it, so the return value
        # matters as much as the reset.
        gen._register_subprocess_state("/tmp/meta.json", "task-7")
        snapshot = gen._clear_subprocess_state()
        self.assertEqual(snapshot["task_id"], "task-7")
        self.assertFalse(gen._SUBPROCESS_STATE["active"])

    def test_clearing_does_not_rebind_the_dict(self):
        # Two threads hold this by reference. Rebinding it would leave the
        # cancel handler looking at a dict nobody updates any more.
        before = gen._SUBPROCESS_STATE
        gen._register_subprocess_state("/tmp/meta.json", "task-7")
        gen._clear_subprocess_state()
        self.assertIs(gen._SUBPROCESS_STATE, before)


class CancelTests(unittest.TestCase):
    def tearDown(self):
        gen._clear_subprocess_state()

    def test_an_unconfirmed_cancel_reports_nothing(self):
        # gr.update() is a plain dict, and the bare one carries no "value" key
        # at all. Reaching for .__dict__ here passes against anything, which is
        # how an earlier version of this test managed to grade nothing.
        update = gen.cancel_generation_process(True, False)
        self.assertNotIn("value", update)

    def test_an_unconfirmed_cancel_leaves_a_running_job_alone(self):
        # The button asks for confirmation first. Until it comes back, the job
        # must still be running and un-canceled.
        gen._register_subprocess_state(None, "task-99")
        gen.cancel_generation_process(True, False)
        self.assertTrue(gen._SUBPROCESS_STATE["active"])
        self.assertFalse(gen._SUBPROCESS_STATE["canceled"])

    def test_cancelling_with_nothing_running_says_so(self):
        gen._clear_subprocess_state()
        update = gen.cancel_generation_process(True, True)
        self.assertIn("No generation is currently running", str(update))

    def test_cancelling_a_running_job_reports_the_task_and_clears_active(self):
        gen._register_subprocess_state(None, "task-42")
        update = gen.cancel_generation_process(True, True)
        self.assertIn("task-42", str(update))
        self.assertTrue(gen._SUBPROCESS_STATE["canceled"])


class MetadataCancelTests(unittest.TestCase):
    """Marking the task record canceled, and when not to."""

    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_a_running_task_is_marked_canceled(self):
        path = self._write({"status": "processing", "processing": {}})
        gen._mark_metadata_canceled(path, "Generation canceled by user.")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["status"], "canceled")

    def test_a_completed_task_is_left_alone(self):
        # Cancel and completion race: the worker can finish between the click
        # and this call. Rewriting a finished task as canceled would lose a
        # result the user actually has on disk.
        path = self._write({"status": "completed", "processing": {}})
        gen._mark_metadata_canceled(path, "Generation canceled by user.")
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["status"], "completed")

    def test_a_missing_file_is_not_an_error(self):
        gen._mark_metadata_canceled("/definitely/not/here/metadata.json", "x")

    def test_no_path_at_all_is_not_an_error(self):
        gen._mark_metadata_canceled(None, "x")


class DeviceSelectionTests(unittest.TestCase):
    """Which card the next model load uses."""

    def test_it_starts_on_auto(self):
        self.assertEqual(DeviceSelection().get(), DEVICE_AUTO)

    def test_setting_a_new_device_reports_that_it_changed(self):
        selection = DeviceSelection()
        self.assertTrue(selection.set(DEVICE_CPU))
        self.assertEqual(selection.get(), DEVICE_CPU)

    def test_setting_the_same_device_reports_no_change(self):
        # on_device_change uses this to decide whether to tell the user the
        # next generation will be slow, so a repeat selection must be silent.
        selection = DeviceSelection()
        selection.set(DEVICE_CPU)
        self.assertFalse(selection.set(DEVICE_CPU))


if __name__ == "__main__":
    unittest.main()
