import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import unittest.mock as mock
import tempfile
import json
import os
import pathlib
import gradio as gr
import character_store as store
import webui_training_handlers as training
import webui_character_handlers as characters
import lora_engine
import character_dataset
import rvc_engine


class _TempLibrary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.slug = store.create_character(self.root, "Test Narrator", "oneshot")
        self.doc = store.load_character(self.root, self.slug)

    def _add_clip(self, duration_s=None):
        clip = {"id": "clip1", "file": "clips/clip1.wav"}
        if duration_s is not None:
            clip["duration_s"] = duration_s
        self.doc["oneshot"]["clips"].append(clip)
        store.save_character(self.root, self.slug, self.doc)

    def _adapter_dir(self):
        d = training.adapter_dir(self.root, self.slug)
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        return d

    def _touch_checkpoint(self, filename, in_best=False):
        base = self._adapter_dir()
        if in_best:
            base = os.path.join(base, "best")
            pathlib.Path(base).mkdir(parents=True, exist_ok=True)
        p = os.path.join(base, filename)
        pathlib.Path(p).touch()
        return os.path.abspath(p)

    def _write_eval(self, rows, recommended):
        eval_dir = os.path.join(self._adapter_dir(), "analysis")
        pathlib.Path(eval_dir).mkdir(parents=True, exist_ok=True)
        data = {
            "adapter_dir": self._adapter_dir(),
            "dataset_dir": training.dataset_state_dir(self.root, self.slug),
            "rows": rows,
            "best_label": "best",
            "best_path": recommended,
            "recommended_checkpoint": recommended,
            "summary_markdown": "",
            "generated_at": "now"
        }
        with open(os.path.join(eval_dir, "checkpoint_eval.json"), "w") as f:
            json.dump(data, f)

    def _touch_sample(self, name):
        samples_dir = os.path.join(self._adapter_dir(), "samples")
        pathlib.Path(samples_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(samples_dir, name)).touch()


class PathTests(_TempLibrary):
    def test_work_root_matches_character_training_default(self):
        expected = os.path.join(self.root, self.slug, "lora_training")
        self.assertEqual(training.work_root(self.root, self.slug), expected)

    def test_adapter_dir_is_under_work_root(self):
        expected = os.path.join(self.root, self.slug, "lora_training", "adapters", f"voiceforge_{self.slug}")
        self.assertEqual(training.adapter_dir(self.root, self.slug), expected)

    def test_dataset_state_dir_is_under_work_root(self):
        expected = os.path.join(self.root, self.slug, "lora_training", "dataset", f"voiceforge_{self.slug}")
        self.assertEqual(training.dataset_state_dir(self.root, self.slug), expected)


class LaneTests(unittest.TestCase):
    def test_lane_choices_values_are_the_constants(self):
        with mock.patch("lora_engine.engine_supports_lora", return_value=True):
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                choices = training.lane_choices()
                # Assuming the docstring contract implies specific labels, but we check constants
                # The docstring says: [("TTS LoRA — the voice itself (V5)", LANE_LORA), ("RVC — post-conversion (Applio)", LANE_RVC)]
                self.assertEqual(choices[0][1], training.LANE_LORA)
                self.assertEqual(choices[1][1], training.LANE_RVC)

    def test_lane_change_shows_stop_only_for_lora(self):
        stop_lora, epochs_lora = training.on_lane_change(training.LANE_LORA)
        self.assertTrue(stop_lora["visible"])
        self.assertFalse(epochs_lora["visible"])

        stop_rvc, epochs_rvc = training.on_lane_change(training.LANE_RVC)
        self.assertFalse(stop_rvc["visible"])
        self.assertTrue(epochs_rvc["visible"])


class ReadinessTests(_TempLibrary):
    def _patch_engines(self):
        self.p_lora = mock.patch("lora_engine.engine_supports_lora", return_value=True)
        self.p_rvc = mock.patch("rvc_paths.missing_applio_parts", return_value=[])
        self.p_lora.start()
        self.p_rvc.start()
        self.addCleanup(self.p_lora.stop)
        self.addCleanup(self.p_rvc.stop)

    def test_no_slug_asks_for_a_selection(self):
        self._patch_engines()
        res = training.readiness_report("lora", None, self.root)
        self.assertIn("Select a voice", res)

    def test_unknown_slug_asks_for_a_selection(self):
        self._patch_engines()
        res = training.readiness_report("lora", "nonexistent", self.root)
        self.assertIn("Select a voice", res)

    def test_reports_clip_count_and_seconds(self):
        self._patch_engines()
        self._add_clip(300.0)
        self._add_clip(200.0)
        res = training.readiness_report("lora", self.slug, self.root)
        self.assertIn("2", res)
        self.assertIn("500", res)

    def test_missing_lora_engine_is_named(self):
        with mock.patch("lora_engine.engine_supports_lora", return_value=False):
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                res = training.readiness_report("lora", self.slug, self.root)
                self.assertIn("LoRA training pipeline", res)

    def test_missing_applio_parts_are_named(self):
        with mock.patch("lora_engine.engine_supports_lora", return_value=True):
            with mock.patch("rvc_paths.missing_applio_parts", return_value=["python.exe"]):
                res = training.readiness_report("rvc", self.slug, self.root)
                self.assertIn("python.exe", res)

    def test_oneshot_mode_reports_both_lanes(self):
        # The tab's mode dropdown carries character modes ("oneshot"/"rvc"),
        # never lane names; a oneshot voice must still see both lanes' status.
        self._patch_engines()
        self._add_clip(300.0)
        res = training.readiness_report("oneshot", self.slug, self.root)
        self.assertIn("LoRA", res)
        self.assertIn("RVC", res)

    def test_thin_dataset_gets_a_warning(self):
        self._patch_engines()
        self._add_clip(30.0)
        res = training.readiness_report("lora", self.slug, self.root)
        self.assertIn("thin", res)

    def test_already_trained_lane_is_reported(self):
        self._patch_engines()
        adapter = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        self.doc["lora"]["adapter_path"] = adapter
        store.save_character(self.root, self.slug, self.doc)
        res = training.readiness_report("lora", self.slug, self.root)
        self.assertIn("already trained", res)

    def test_unmeasured_clips_are_counted(self):
        self._patch_engines()
        self._add_clip(None)
        res = training.readiness_report("lora", self.slug, self.root)
        self.assertIn("no measured duration", res)


class CheckpointListTests(_TempLibrary):
    def test_missing_adapter_dir_is_empty(self):
        self.assertEqual(training.list_checkpoints(self.slug, self.root), [])

    def test_kinds_are_classified_from_filenames(self):
        ad = self._adapter_dir()
        self._touch_checkpoint(f"voiceforge_{self.slug}_epoch_003.safetensors")
        self._touch_checkpoint(f"voiceforge_{self.slug}_step_000450.safetensors")
        self._touch_checkpoint(f"voiceforge_{self.slug}_interrupted.safetensors")
        self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors", in_best=True)
        
        cps = training.list_checkpoints(self.slug, self.root)
        kinds = {c["kind"] for c in cps}
        self.assertIn("best", kinds)
        self.assertIn("epoch", kinds)
        self.assertIn("step", kinds)
        self.assertIn("interrupted", kinds)
        self.assertIn("final", kinds)
        
        epoch_entry = [c for c in cps if c["kind"] == "epoch"][0]
        self.assertEqual(epoch_entry["epoch"], 3)

    def test_sorted_best_then_epochs_ascending(self):
        ad = self._adapter_dir()
        self._touch_checkpoint(f"voiceforge_{self.slug}_epoch_010.safetensors")
        self._touch_checkpoint(f"voiceforge_{self.slug}_epoch_002.safetensors")
        self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors", in_best=True)
        self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        
        cps = training.list_checkpoints(self.slug, self.root)
        kinds = [c["kind"] for c in cps]
        # best, then epochs 2, 10, then final
        self.assertEqual(kinds[0], "best")
        self.assertEqual(cps[1]["kind"], "epoch")
        self.assertEqual(cps[1]["epoch"], 2)
        self.assertEqual(cps[2]["kind"], "epoch")
        self.assertEqual(cps[2]["epoch"], 10)
        self.assertEqual(kinds[-1], "final")

    def test_eval_report_attaches_measurements(self):
        ad = self._adapter_dir()
        epoch_path = self._touch_checkpoint(f"voiceforge_{self.slug}_epoch_003.safetensors")
        self._write_eval([{"path": epoch_path, "val_loss": 5.5, "phase": "train", "epoch": 3}], epoch_path)
        
        cps = training.list_checkpoints(self.slug, self.root)
        epoch_entry = [c for c in cps if c["kind"] == "epoch"][0]
        self.assertEqual(epoch_entry["val_loss"], 5.5)
        self.assertEqual(epoch_entry["phase"], "train")

    def test_recommended_flag_follows_the_report(self):
        ad = self._adapter_dir()
        best_path = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors", in_best=True)
        self._write_eval([{"path": best_path, "val_loss": 5.0, "phase": "best", "epoch": 1}], best_path)
        
        cps = training.list_checkpoints(self.slug, self.root)
        best_entry = [c for c in cps if c["kind"] == "best"][0]
        self.assertTrue(best_entry["recommended"])

    def test_corrupt_eval_json_degrades_to_unmeasured(self):
        ad = self._adapter_dir()
        eval_dir = os.path.join(ad, "analysis")
        pathlib.Path(eval_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(eval_dir, "checkpoint_eval.json"), "wb") as f:
            f.write(b"garbage")
        
        self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        cps = training.list_checkpoints(self.slug, self.root)
        self.assertEqual(cps[0]["val_loss"], None)


class CheckpointChoiceTests(_TempLibrary):
    def test_empty_library_yields_no_choices(self):
        choices, selected = training.checkpoint_choices(self.slug, self.root)
        self.assertEqual(choices, [])
        self.assertIsNone(selected)

    def test_labels_carry_epoch_val_and_recommended(self):
        ad = self._adapter_dir()
        best_path = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors", in_best=True)
        self._write_eval([{"path": best_path, "val_loss": 5.28, "phase": "best", "epoch": 5}], best_path)
        
        choices, selected = training.checkpoint_choices(self.slug, self.root)
        label = choices[0][0]
        self.assertIn("5", label)
        self.assertIn("5.28", label)
        self.assertIn("recommended", label)

    def test_selects_recommended_first(self):
        ad = self._adapter_dir()
        best_path = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors", in_best=True)
        self._write_eval([{"path": best_path, "val_loss": 5.28, "phase": "best", "epoch": 5}], best_path)
        
        _, selected = training.checkpoint_choices(self.slug, self.root)
        self.assertEqual(selected, best_path)

    def test_falls_back_to_best_kind_then_first(self):
        ad = self._adapter_dir()
        best_path = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors", in_best=True)
        final_path = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        
        _, selected = training.checkpoint_choices(self.slug, self.root)
        self.assertEqual(selected, best_path)


class SampleTests(_TempLibrary):
    def test_epoch_checkpoint_finds_its_padded_sample(self):
        ad = self._adapter_dir()
        cp = self._touch_checkpoint(f"voiceforge_{self.slug}_epoch_003.safetensors")
        self._touch_sample("epoch_003.wav")
        
        sample = training.sample_for_checkpoint(self.slug, cp, self.root)
        self.assertIsNotNone(sample)
        self.assertIn("epoch_003.wav", sample)

    def test_final_checkpoint_takes_highest_sample(self):
        ad = self._adapter_dir()
        cp = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        self._touch_sample("epoch_001.wav")
        self._touch_sample("epoch_007.wav")
        
        sample = training.sample_for_checkpoint(self.slug, cp, self.root)
        self.assertIsNotNone(sample)
        self.assertIn("epoch_007.wav", sample)

    def test_none_path_and_no_samples_return_none(self):
        sample = training.sample_for_checkpoint(self.slug, None, self.root)
        self.assertIsNone(sample)


class CheckpointDetailTests(_TempLibrary):
    def test_none_asks_for_a_selection(self):
        detail = training.checkpoint_detail(self.slug, None, self.root)
        self.assertIn("Select a checkpoint", detail)

    def test_detail_names_current_adapter(self):
        ad = self._adapter_dir()
        cp = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        self.doc["lora"]["adapter_path"] = cp
        store.save_character(self.root, self.slug, self.doc)
        
        detail = training.checkpoint_detail(self.slug, cp, self.root)
        self.assertIn("current", detail)

    def test_change_without_sample_says_so(self):
        ad = self._adapter_dir()
        cp = self._touch_checkpoint(f"voiceforge_{self.slug}_epoch_003.safetensors")
        # No sample created
        
        audio_update, detail = training.on_checkpoint_change("lora", self.slug, cp, self.root)
        self.assertIsNone(audio_update["value"])
        self.assertIn("no kept sample", detail)


class UseCheckpointTests(_TempLibrary):
    def test_refuses_without_slug(self):
        status, _ = training.use_checkpoint_ui("lora", None, "/fake/path", self.root)
        self.assertIn("Select a voice", status)
        self.assertIsNone(store.load_character(self.root, self.slug)["lora"]["adapter_path"])

    def test_refuses_without_selection(self):
        status, _ = training.use_checkpoint_ui("lora", self.slug, None, self.root)
        self.assertIn("Select a checkpoint", status)

    def test_refuses_a_missing_file(self):
        status, _ = training.use_checkpoint_ui("lora", self.slug, "/nonexistent/path.safetensors", self.root)
        self.assertIn("does not exist", status)

    def test_writes_adapter_path_and_preserves_strength(self):
        ad = self._adapter_dir()
        cp = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        self.doc["lora"]["strength"] = 1.5
        store.save_character(self.root, self.slug, self.doc)
        
        status, _ = training.use_checkpoint_ui("lora", self.slug, cp, self.root)
        self.assertIn("Speak with trained voice", status)
        
        new_doc = store.load_character(self.root, self.slug)
        self.assertEqual(new_doc["lora"]["adapter_path"], cp)
        self.assertEqual(new_doc["lora"]["strength"], 1.5)

    def test_preserves_the_rest_of_the_lora_block(self):
        ad = self._adapter_dir()
        cp = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        self.doc["lora"]["trained_from_seconds"] = 123.0
        self.doc["lora"]["samples_dir"] = "somewhere"
        store.save_character(self.root, self.slug, self.doc)

        training.use_checkpoint_ui("oneshot", self.slug, cp, self.root)

        new_doc = store.load_character(self.root, self.slug)
        self.assertEqual(new_doc["lora"]["trained_from_seconds"], 123.0)
        self.assertEqual(new_doc["lora"]["samples_dir"], "somewhere")

    def test_success_mentions_retoggling_speak(self):
        ad = self._adapter_dir()
        cp = self._touch_checkpoint(f"voiceforge_{self.slug}.safetensors")
        status, _ = training.use_checkpoint_ui("lora", self.slug, cp, self.root)
        self.assertIn("Speak with trained voice", status)


class StopTests(_TempLibrary):
    def test_no_slug_is_a_message(self):
        res = training.stop_training_ui("lora", None, self.root)
        self.assertIn("Select a voice", res)

    def test_touches_stop_flags_in_both_state_dirs(self):
        ad = self._adapter_dir()
        ds = training.dataset_state_dir(self.root, self.slug)
        pathlib.Path(ad).mkdir(parents=True, exist_ok=True)
        pathlib.Path(ds).mkdir(parents=True, exist_ok=True)
        
        res = training.stop_training_ui("lora", self.slug, self.root)
        self.assertTrue(os.path.exists(os.path.join(ad, "stop.flag")))
        self.assertTrue(os.path.exists(os.path.join(ds, "stop.flag")))


class LiveStatusTests(_TempLibrary):
    def test_missing_status_returns_fallback(self):
        res = training._format_live_status("/nonexistent", "fallback")
        self.assertEqual(res, "fallback")

    def test_fields_are_folded_in(self):
        ad = self._adapter_dir()
        status_path = os.path.join(ad, "status.json")
        with open(status_path, "w") as f:
            json.dump({"epoch": 3, "total_epochs": 10, "step": 120, "total_steps": 400, "loss": 5.91}, f)
        
        res = training._format_live_status(ad, "fallback")
        self.assertIn("3/10", res)
        self.assertIn("120/400", res)
        self.assertIn("5.91", res)

    def test_unreadable_status_returns_fallback(self):
        ad = self._adapter_dir()
        status_path = os.path.join(ad, "status.json")
        with open(status_path, "wb") as f:
            f.write(b"garbage")
        
        res = training._format_live_status(ad, "fallback")
        self.assertEqual(res, "fallback")


class StartTrainingGateTests(_TempLibrary):
    def test_no_slug_is_refused(self):
        gen = training.start_training_ui("lora", None, training.LANE_LORA, True, 200, self.root)
        yields = list(gen)
        self.assertEqual(len(yields), 1)
        self.assertIn("Select a voice", yields[0][1])

    def test_unconfirmed_press_arms(self):
        gen = training.start_training_ui("lora", self.slug, training.LANE_LORA, False, 200, self.root)
        yields = list(gen)
        self.assertEqual(len(yields), 1)
        self.assertIn("Confirm", yields[0][1])

    def test_missing_lora_engine_is_refused(self):
        with mock.patch("lora_engine.engine_supports_lora", return_value=False):
            gen = training.start_training_ui("lora", self.slug, training.LANE_LORA, True, 200, self.root)
            yields = list(gen)
            self.assertEqual(len(yields), 1)
            self.assertIn("LoRA", yields[0][1])
            # The refusal must carry the fix, same as the character panel's.
            self.assertIn("INDEXTTS25_ROOT", yields[0][1])

    def test_missing_applio_is_refused(self):
        with mock.patch("rvc_paths.missing_applio_parts", return_value=["python.exe"]):
            gen = training.start_training_ui("rvc", self.slug, training.LANE_RVC, True, 200, self.root)
            yields = list(gen)
            self.assertEqual(len(yields), 1)
            self.assertIn("python.exe", yields[0][1])


class StartTrainingLoraTests(_TempLibrary):
    def test_progress_reaches_the_bar_and_result_names_the_adapter(self):
        ad = self._adapter_dir()
        adapter_path = os.path.join(ad, f"voiceforge_{self.slug}.safetensors")
        status_path = os.path.join(ad, "status.json")
        samples_dir = os.path.join(ad, "samples")
        
        def fake_train(*args, **kwargs):
            cb = kwargs.get('progress_callback')
            if cb:
                cb(lora_engine.LoraProgress("prep", "segmenting", 0.1))
                cb(lora_engine.LoraProgress("train", "epoch", 0.6))
            return {
                "manifest": {"total_seconds": 500.0},
                "artifacts": {
                    "adapter_path": adapter_path,
                    "status_path": status_path,
                    "samples_dir": samples_dir
                }
            }

        with mock.patch("character_training.train_character_lora", fake_train):
            gen = training.start_training_ui("lora", self.slug, training.LANE_LORA, True, 200, self.root)
            yields = list(gen)
            
            # At least one intermediate yield must carry real progress: a bar
            # that is no longer the idle one.
            self.assertGreater(len(yields), 1)
            self.assertTrue(any(y[0] != training.TRAIN_PROGRESS_IDLE
                                for y in yields[:-1]))

            final = yields[-1]
            self.assertIn("voiceforge", final[1])
            self.assertIn("500", final[1])
            self.assertIn("choices", final[2])

    def test_root_defaults_to_the_library(self):
        # The live wiring passes no root; the handler must resolve the library
        # root itself before the worker thread ever sees it.
        seen = {}

        def fake_train(root_arg, slug_arg, **kwargs):
            seen["root"] = root_arg
            return {"manifest": {"total_seconds": 1.0},
                    "artifacts": {"adapter_path": "", "status_path": "",
                                  "samples_dir": ""}}

        with mock.patch.object(characters, "CHARACTER_LIBRARY_ROOT", self.root):
            with mock.patch("character_training.train_character_lora", fake_train):
                with mock.patch("lora_engine.engine_supports_lora", return_value=True):
                    gen = training.start_training_ui(
                        "oneshot", self.slug, training.LANE_LORA, True, 200)
                    list(gen)
        self.assertEqual(seen.get("root"), self.root)

    def test_live_status_reaches_intermediate_yields(self):
        # The point of the tab: epoch/loss off status.json while training runs.
        ad = self._adapter_dir()
        with open(os.path.join(ad, "status.json"), "w") as f:
            json.dump({"epoch": 3, "total_epochs": 10, "step": 120,
                       "total_steps": 400, "loss": 5.91}, f)

        def fake_train(*args, **kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb(lora_engine.LoraProgress("train", "training", 0.6))
            return {"manifest": {"total_seconds": 1.0},
                    "artifacts": {"adapter_path": "", "status_path": "",
                                  "samples_dir": ""}}

        with mock.patch("character_training.train_character_lora", fake_train):
            gen = training.start_training_ui(
                "oneshot", self.slug, training.LANE_LORA, True, 200, self.root)
            yields = list(gen)
        self.assertTrue(any("3/10" in (y[1] or "") for y in yields[:-1]))

    def test_bar_holds_last_fraction_when_none(self):
        # A LoraProgress with fraction=None must not snap the bar back to 0.
        def fake_train(*args, **kwargs):
            cb = kwargs.get("progress_callback")
            if cb:
                cb(lora_engine.LoraProgress("train", "at half", 0.5))
                cb(lora_engine.LoraProgress("train", "still at half", None))
            return {"manifest": {"total_seconds": 1.0},
                    "artifacts": {"adapter_path": "", "status_path": "",
                                  "samples_dir": ""}}

        with mock.patch("character_training.train_character_lora", fake_train):
            gen = training.start_training_ui(
                "oneshot", self.slug, training.LANE_LORA, True, 200, self.root)
            yields = list(gen)
        progress_frames = [y[0] for y in yields[:-1]]
        self.assertGreaterEqual(len(progress_frames), 2)
        from webui_progress import render_progress_bar
        self.assertNotIn(render_progress_bar(0.0, "train"), progress_frames[1:])

    def test_lora_error_reaches_the_status_line(self):
        def fake_train(*args, **kwargs):
            raise lora_engine.LoraError("boom")
        
        with mock.patch("character_training.train_character_lora", fake_train):
            gen = training.start_training_ui("lora", self.slug, training.LANE_LORA, True, 200, self.root)
            yields = list(gen)
            self.assertIn("boom", yields[-1][1])

    def test_dataset_error_reaches_the_status_line(self):
        def fake_train(*args, **kwargs):
            raise character_dataset.DatasetExportError("data fail")
        
        with mock.patch("character_training.train_character_lora", fake_train):
            gen = training.start_training_ui("lora", self.slug, training.LANE_LORA, True, 200, self.root)
            yields = list(gen)
            self.assertIn("data fail", yields[-1][1])


class StartTrainingRvcTests(_TempLibrary):
    def test_epochs_are_passed_through_as_int(self):
        with mock.patch("character_training.train_character") as fake:
            fake.return_value = {"manifest": {}, "artifacts": {}}
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                gen = training.start_training_ui("rvc", self.slug, training.LANE_RVC, True, 150.0, self.root)
                list(gen)
                fake.assert_called_once()
                call_kwargs = fake.call_args[1]
                self.assertEqual(call_kwargs["total_epochs"], 150)

    def test_rvc_yields_a_start_message_before_blocking(self):
        # train_character blocks for hours; the browser must hear that the run
        # started before the call, not only after it returns.
        with mock.patch("character_training.train_character") as fake:
            fake.return_value = {"manifest": {}, "artifacts": {}}
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                gen = training.start_training_ui(
                    "rvc", self.slug, training.LANE_RVC, True, 200, self.root)
                yields = list(gen)
        self.assertGreaterEqual(len(yields), 2)
        self.assertNotEqual(yields[0][1], "")

    def test_rvc_failure_reaches_the_status_line(self):
        def fake_train(*args, **kwargs):
            raise rvc_engine.RVCError("applio broke")
        
        with mock.patch("character_training.train_character", fake_train):
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                gen = training.start_training_ui("rvc", self.slug, training.LANE_RVC, True, 200, self.root)
                yields = list(gen)
                self.assertIn("applio broke", yields[-1][1])


class PanelRefreshTests(_TempLibrary):
    def test_refresh_returns_the_four_updates(self):
        with mock.patch("lora_engine.engine_supports_lora", return_value=True):
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                res = training.refresh_training_panel("lora", self.slug, self.root)
                self.assertEqual(len(res), 4)
                # readiness, ckpt update, sample update, detail
                self.assertIsInstance(res[0], str)
                self.assertIn("choices", res[1])
                self.assertIn("value", res[2])
                self.assertIsInstance(res[3], str)

    def test_mode_change_repopulates_voices(self):
        with mock.patch("lora_engine.engine_supports_lora", return_value=True):
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                res = training.on_mode_change("oneshot", self.root)
                self.assertEqual(len(res), 5)
                # voice update, readiness, ckpt update, sample update, detail
                self.assertIn("choices", res[0])
                self.assertIn(self.slug, [c[1] for c in res[0]["choices"]])

    def test_mode_with_no_voices_does_not_crash(self):
        # The library holds only a oneshot character; switching the tab to
        # rvc must land on "nothing selected", not a traceback.
        with mock.patch("lora_engine.engine_supports_lora", return_value=True):
            with mock.patch("rvc_paths.missing_applio_parts", return_value=[]):
                res = training.on_mode_change("rvc", self.root)
                self.assertEqual(len(res), 5)
                self.assertEqual(res[0]["choices"], [])
                self.assertIsNone(res[0]["value"])
                self.assertIsInstance(res[1], str)


class BuildTests(_TempLibrary):
    def test_tab_builds_inside_blocks(self):
        with gr.Blocks() as blocks:
            d = training.build_training_tab(root=self.root)
        
        self.assertIsInstance(d, dict)
        self.assertTrue(len(d) > 0)
        for v in d.values():
            self.assertTrue(isinstance(v, (gr.Component, type(None))))

    def test_initial_state_shape(self):
        res = training.initial_state(root=self.root)
        self.assertEqual(len(res), 8)
        self.assertIsInstance(res[0], str)
        self.assertIsInstance(res[1], list)


if __name__ == "__main__":
    unittest.main()
