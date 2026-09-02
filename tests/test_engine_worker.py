"""Lifecycle of the long-lived engine worker.

The worker holds a model in VRAM between generations, which is the whole
point of it, and that makes its state machine worth pinning down: when it
reports itself loaded, when it agrees to unload, and what it does when
asked to stop something that was never started.

None of this needs a GPU or the engine. Constructing an EngineWorker only
sets attributes, and every method here is exercised against a worker that
never spawned a process -- which is exactly the state the app is in until
someone presses Generate.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine_worker


class FreshWorkerStateTests(unittest.TestCase):
    """A worker that has never spawned anything reports itself honestly."""

    def setUp(self):
        self.worker = engine_worker.EngineWorker()

    def test_constructing_a_worker_starts_no_process(self):
        state = self.worker.status()
        self.assertFalse(state["running"])
        self.assertIsNone(state["pid"])
        self.assertFalse(state["busy"])
        self.assertEqual(state["requests_served"], 0)

    def test_model_is_not_loaded_until_a_request_has_been_served(self):
        # model_loaded is derived from requests_served rather than tracked
        # separately, because the worker loads lazily on its first request.
        self.assertFalse(self.worker.status()["model_loaded"])

    def test_describe_says_it_starts_on_the_next_generation(self):
        self.assertIn("not running", self.worker.describe())

    def test_uptime_and_idle_are_zero_before_it_ever_ran(self):
        state = self.worker.status()
        self.assertEqual(state["uptime_seconds"], 0.0)
        self.assertEqual(state["idle_seconds"], 0.0)


class StoppingSomethingThatIsNotRunningTests(unittest.TestCase):
    """The unload button and cancel are reachable before anything has started."""

    def setUp(self):
        self.worker = engine_worker.EngineWorker()

    def test_shutdown_reports_that_there_was_nothing_to_shut_down(self):
        self.assertFalse(self.worker.shutdown())

    def test_kill_reports_that_there_was_nothing_to_kill(self):
        self.assertFalse(self.worker.kill())

    def test_shutdown_twice_is_still_harmless(self):
        self.worker.shutdown()
        self.assertFalse(self.worker.shutdown())


class IdleLimitTests(unittest.TestCase):
    """The 'Unload after' control, and the timer behind it."""

    def setUp(self):
        self.worker = engine_worker.EngineWorker()

    def tearDown(self):
        self.worker._cancel_idle_timer()

    def test_setting_the_limit_reports_it_back_and_records_it(self):
        self.assertEqual(self.worker.set_idle_seconds(600), 600.0)
        self.assertEqual(self.worker.status()["idle_limit_seconds"], 600.0)

    def test_zero_means_never_unload(self):
        self.assertEqual(self.worker.set_idle_seconds(0), 0.0)
        self.assertEqual(self.worker.status()["idle_limit_seconds"], 0.0)

    def test_a_negative_limit_is_clamped_rather_than_rejected(self):
        # The UI offers a fixed list, but the value also arrives from an
        # environment variable, so a nonsense number has to land somewhere sane.
        self.assertEqual(self.worker.set_idle_seconds(-30), 0.0)

    def test_no_timer_is_armed_while_nothing_is_running(self):
        # Arming one against a worker with no process would fire _on_idle for a
        # process that does not exist, and would keep a Timer thread alive.
        self.worker.set_idle_seconds(1)
        self.assertIsNone(self.worker._idle_timer)

    def test_zero_leaves_no_timer_armed(self):
        self.worker.set_idle_seconds(0)
        self.assertIsNone(self.worker._idle_timer)

    def test_going_idle_with_no_process_does_nothing(self):
        # _on_idle is what the timer calls; reaching it after the process died
        # must not raise or try to shut down a corpse.
        self.worker._on_idle()
        self.assertFalse(self.worker.status()["running"])


class ModuleWorkerTests(unittest.TestCase):
    """The shared WORKER the app actually uses."""

    def test_module_worker_exists_and_is_an_engine_worker(self):
        self.assertIsInstance(engine_worker.WORKER, engine_worker.EngineWorker)

    def test_importing_engine_worker_does_not_start_the_engine(self):
        # engine_worker is imported at webui start-up, before gradio. If the
        # handle spawned eagerly, every launch would pay the model load and
        # hold a card whether or not anyone generated anything.
        self.assertFalse(engine_worker.WORKER.status()["running"])

    def test_the_default_idle_limit_is_a_number_of_seconds(self):
        self.assertIsInstance(engine_worker.DEFAULT_IDLE_SECONDS, float)
        self.assertGreaterEqual(engine_worker.DEFAULT_IDLE_SECONDS, 0.0)


if __name__ == "__main__":
    unittest.main()
