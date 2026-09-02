import importlib
import sys
import unittest


class WebUIPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_argv = sys.argv[:]
        sys.argv = ["webui.py"]
        cls.webui = importlib.import_module("webui")

    @classmethod
    def tearDownClass(cls):
        sys.argv = cls._original_argv

    def test_import_does_not_start_the_engine(self):
        # This used to read `webui.tts.is_loaded()`. That attribute went away
        # when generation moved into a subprocess, and the assertion went with
        # it -- it had been failing rather than guarding anything.
        #
        # The intent is still the one worth keeping: opening the app must not
        # load a model. The worker starts on the first generation, so a launch
        # nobody generates with should never touch a graphics card.
        import engine_worker

        self.assertFalse(engine_worker.WORKER.status()["running"])
        self.assertFalse(engine_worker.WORKER.status()["model_loaded"])

    def test_subprocess_checkbox_is_in_default_preset_config(self):
        cfg = self.webui._default_ui_config()
        self.assertIn("use_subprocess_system", cfg["audio_generation"])
        self.assertTrue(cfg["audio_generation"]["use_subprocess_system"])

    def test_subprocess_checkbox_round_trips_through_preset_value_mapping(self):
        index = next(
            idx for idx, field in enumerate(self.webui._CONFIG_FIELDS) if field["key"] == "use_subprocess_system"
        )
        values = self.webui._ui_config_to_values({"audio_generation": {"use_subprocess_system": False}})
        self.assertFalse(values[index])


if __name__ == "__main__":
    unittest.main()
