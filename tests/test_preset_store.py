"""The preset store against a temp directory: timestamps and the last-used pointer.

webui_preset_store is filesystem-only by design, so it can be pointed at a
temp directory by patching PRESETS_DIR and exercised without gradio. These
tests exist because two things here used to be untestable or silent: the
_meta timestamps came from a clock read inside _save_ui_preset, and a failed
write of the last-used pointer vanished without a trace.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import webui_preset_store as store


class SaveTimestampTests(unittest.TestCase):
    """_save_ui_preset(now=...) stamps _meta from the injected clock."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(store, "PRESETS_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clock = lambda: datetime(2026, 9, 3, 12, 0, 0)

    def _saved(self, name):
        with open(os.path.join(self.tmp.name, f"{name}.json"), encoding="utf-8") as handle:
            return json.load(handle)

    def test_last_modified_comes_from_the_injected_clock(self):
        store._save_ui_preset("clocked", {"a": 1}, now=self.clock)
        meta = self._saved("clocked")["_meta"]
        self.assertEqual(meta["last_modified"], "2026-09-03T12:00:00")

    def test_created_at_is_set_on_first_save_and_kept_on_the_second(self):
        store._save_ui_preset("kept", {"a": 1}, now=self.clock)
        later = lambda: datetime(2026, 9, 3, 13, 30, 0)
        cfg = self._saved("kept")
        store._save_ui_preset("kept", cfg, now=later)
        meta = self._saved("kept")["_meta"]
        self.assertEqual(meta["created_at"], "2026-09-03T12:00:00")
        self.assertEqual(meta["last_modified"], "2026-09-03T13:30:00")

    def test_the_default_clock_is_the_real_one(self):
        store._save_ui_preset("real", {"a": 1})
        year = int(self._saved("real")["_meta"]["last_modified"][:4])
        self.assertGreaterEqual(year, 2026)


class LastUsedPointerTests(unittest.TestCase):
    """The pointer write says something when it fails instead of nothing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(store, "PRESETS_DIR", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_successful_write_round_trips(self):
        # The getter only trusts the pointer if the preset file exists too.
        name = store._save_ui_preset("my preset", {"a": 1})
        store._set_last_used_ui_preset(name)
        self.assertEqual(store._get_last_used_ui_preset(), name)

    def test_a_failed_write_reports_which_preset_was_lost(self):
        # The preset silently not being remembered across restarts was the
        # original finding; the fix is one visible line naming the preset.
        out = io.StringIO()
        with mock.patch.object(store.Path, "write_text", side_effect=OSError("disk full")):
            with redirect_stdout(out):
                store._set_last_used_ui_preset("fragile")
        self.assertIn("fragile", out.getvalue())
        self.assertIn("disk full", out.getvalue())

    def test_a_failed_write_does_not_raise(self):
        with mock.patch.object(store.Path, "write_text", side_effect=OSError("denied")):
            with redirect_stdout(io.StringIO()):
                store._set_last_used_ui_preset("anything")


if __name__ == "__main__":
    unittest.main()
