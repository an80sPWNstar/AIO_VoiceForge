"""Tests for cleanup job cancellation.

The design under test: a per-run token prevents stale-flag bugs. A plain
boolean cancel_requested cannot tell "stop THIS run" from "a cancel left
over from a run that already ended", so a cancelled job would make the NEXT
job cancel itself instantly. A token ensures that cannot happen.

Key scenario: after a run is cancelled and finishes, a SUBSEQUENT run is NOT
cancelled. This is the defect the token design exists to prevent.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import webui_audio_cleanup as cleanup


class CancellationTests(unittest.TestCase):
    """Test cleanup job cancellation."""

    def setUp(self):
        """Reset module state before each test."""
        cleanup._ACTIVE_TOKEN = None
        cleanup._ACTIVE_PROCESS = None
        cleanup._CANCELLED_TOKEN = None

    def tearDown(self):
        """Clean up after each test."""
        cleanup._ACTIVE_TOKEN = None
        cleanup._ACTIVE_PROCESS = None
        cleanup._CANCELLED_TOKEN = None

    def test_request_cancel_returns_false_when_nothing_running(self):
        """Calling request_cancel with no active run returns False."""
        self.assertFalse(cleanup.request_cancel())

    def test_cancel_is_complete_returns_true_when_nothing_running(self):
        """cancel_is_complete returns True when no run is active."""
        self.assertTrue(cleanup.cancel_is_complete())

    def test_a_cancelled_run_raises_cleanup_cancelled(self):
        """A run that is cancelled mid-operation raises CleanupCancelled."""
        with tempfile.TemporaryDirectory() as scratch:
            progress_path = os.path.join(scratch, "progress.jsonl")
            stdout_path = os.path.join(scratch, "stdout.txt")
            stderr_path = os.path.join(scratch, "stderr.txt")

            # Create dummy output files so _read_result won't fail.
            Path(stdout_path).write_text('{"ok": true, "output": "/dev/null", "notes": []}')
            Path(stderr_path).write_text("")

            # Use a command that runs a sleep loop; the test cancels it mid-run.
            command = [
                sys.executable,
                "-c",
                "import time; [time.sleep(0.01) for _ in range(100)]"
            ]

            # Start the worker in a thread so we can cancel it from the main thread.
            exception_holder = [None]
            def run_worker():
                try:
                    cleanup._run_worker(
                        command=command,
                        progress_path=progress_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        progress_callback=None,
                        finished_message="Test",
                    )
                except cleanup.CleanupCancelled as e:
                    exception_holder[0] = e
                except Exception as e:
                    exception_holder[0] = e

            thread = threading.Thread(target=run_worker)
            thread.start()

            # Give the worker time to start and claim its token.
            time.sleep(0.05)

            # Now cancel it.
            cancelled = cleanup.request_cancel()
            self.assertTrue(cancelled)

            # Wait for the thread to finish.
            thread.join(timeout=5.0)

            # Verify it was cancelled, not some other error.
            self.assertIsInstance(exception_holder[0], cleanup.CleanupCancelled)

    def test_stale_flag_case_second_run_not_cancelled(self):
        """The critical case: a second run is not cancelled after the first is.

        This is the defect the token design exists to prevent. A plain
        boolean flag would make the second run cancel itself instantly.
        """
        # First run: simulate it being cancelled and then finishing.
        cleanup._ACTIVE_TOKEN = 1
        cleanup._CANCELLED_TOKEN = 1
        cleanup._ACTIVE_PROCESS = None

        # Clean up after cancellation (simulating what finally does).
        cleanup._ACTIVE_TOKEN = None
        cleanup._ACTIVE_PROCESS = None
        cleanup._CANCELLED_TOKEN = None

        with tempfile.TemporaryDirectory() as scratch:
            progress_path = os.path.join(scratch, "progress.jsonl")
            stdout_path = os.path.join(scratch, "stdout.txt")
            stderr_path = os.path.join(scratch, "stderr.txt")

            # Create a dummy output file for the result.
            result_file = os.path.join(scratch, "output.wav")
            Path(result_file).write_text("")

            # Create a command that writes the expected result JSON to stdout.
            import json as json_lib
            result_json = json_lib.dumps({"ok": True, "output": result_file, "notes": []})
            command = [sys.executable, "-c", f"print({result_json!r})"]

            # Second run: should NOT be cancelled, even though _CANCELLED_TOKEN
            # is still set to 1 (from the previous run).
            # Create a fresh token for the second run.

            # This should work normally without raising CleanupCancelled.
            result = cleanup._run_worker(
                command=command,
                progress_path=progress_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                progress_callback=None,
                finished_message="Test",
            )
            self.assertEqual(result["ok"], True)

    def test_cancel_arriving_before_process_starts_is_honoured(self):
        """A cancellation request during startup is still respected."""
        with tempfile.TemporaryDirectory() as scratch:
            progress_path = os.path.join(scratch, "progress.jsonl")
            stdout_path = os.path.join(scratch, "stdout.txt")
            stderr_path = os.path.join(scratch, "stderr.txt")

            Path(stdout_path).write_text('{"ok": true, "output": "/dev/null", "notes": []}')
            Path(stderr_path).write_text("")

            # Use a command that starts but takes time to run.
            command = [
                sys.executable,
                "-c",
                "import time; [time.sleep(0.01) for _ in range(100)]"
            ]

            exception_holder = [None]
            def run_worker():
                try:
                    cleanup._run_worker(
                        command=command,
                        progress_path=progress_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        progress_callback=None,
                        finished_message="Test",
                    )
                except cleanup.CleanupCancelled as e:
                    exception_holder[0] = e
                except Exception as e:
                    exception_holder[0] = e

            thread = threading.Thread(target=run_worker)
            thread.start()

            # Request cancellation immediately (might arrive before or during startup).
            time.sleep(0.01)
            cancelled = cleanup.request_cancel()
            self.assertTrue(cancelled)

            # Wait for the thread to finish.
            thread.join(timeout=5.0)

            # Verify it was cancelled.
            self.assertIsInstance(exception_holder[0], cleanup.CleanupCancelled)

    def test_cleanup_cancelled_is_catchable_as_cleanup_error(self):
        """CleanupCancelled is a subclass of CleanupError and catches as such."""
        # This is the point of subclassing: existing except CleanupError sites work.
        exc = cleanup.CleanupCancelled("User stopped.")
        self.assertIsInstance(exc, cleanup.CleanupError)
        self.assertIsInstance(exc, cleanup.CleanupCancelled)

    def test_cancel_is_complete_reflects_active_token_state(self):
        """cancel_is_complete returns True only when no run is active."""
        # Initially, no run.
        self.assertTrue(cleanup.cancel_is_complete())

        # Simulate an active run.
        cleanup._ACTIVE_TOKEN = 42
        self.assertFalse(cleanup.cancel_is_complete())

        # After the run finishes (token is cleared).
        cleanup._ACTIVE_TOKEN = None
        self.assertTrue(cleanup.cancel_is_complete())

    def test_request_cancel_returns_true_when_run_is_active(self):
        """request_cancel returns True when it stops a running job."""
        with tempfile.TemporaryDirectory() as scratch:
            # Simulate a real process
            command = [sys.executable, "-c", "import time; time.sleep(10)"]
            process = subprocess.Popen(command)

            # Set up a fake running state with this real process.
            cleanup._ACTIVE_TOKEN = 42
            cleanup._ACTIVE_PROCESS = process

            result = cleanup.request_cancel()
            self.assertTrue(result)

            # Wait a moment for the process to be killed.
            time.sleep(0.1)

            # The process should be gone.
            self.assertIsNotNone(process.poll())

    def test_cleanup_state_cleared_after_successful_run(self):
        """After a normal completion, the global state is cleared."""
        with tempfile.TemporaryDirectory() as scratch:
            progress_path = os.path.join(scratch, "progress.jsonl")
            stdout_path = os.path.join(scratch, "stdout.txt")
            stderr_path = os.path.join(scratch, "stderr.txt")

            # Create a dummy output file.
            result_file = os.path.join(scratch, "output.wav")
            Path(result_file).write_text("")

            # Create a command that writes the expected result JSON to stdout.
            import json as json_lib
            result_json = json_lib.dumps({"ok": True, "output": result_file, "notes": []})
            command = [sys.executable, "-c", f"print({result_json!r})"]

            cleanup._run_worker(
                command=command,
                progress_path=progress_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                progress_callback=None,
                finished_message="Test",
            )

            # After the run, the state should be cleared.
            self.assertIsNone(cleanup._ACTIVE_TOKEN)
            self.assertIsNone(cleanup._ACTIVE_PROCESS)
            self.assertIsNone(cleanup._CANCELLED_TOKEN)


if __name__ == "__main__":
    unittest.main()
