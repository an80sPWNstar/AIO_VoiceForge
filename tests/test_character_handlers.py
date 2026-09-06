"""The character-library panel: what each control does.

These call the handlers exactly the way the wiring does, with a temp library
root injected. Gradio update objects are plain dicts, so assertions read the
"value"/"choices" keys directly.

The rule under test throughout is that no handler raises. Every one of them
hangs off a button, and a traceback out of a gradio handler shows the user a
spinner that stops and nothing else.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import character_store as store
import webui_character_handlers as handlers


class _Panel(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = self._dir.name
        self.addCleanup(self._dir.cleanup)
        self._scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self._scratch.cleanup)

    def a_clip(self, name="sample.wav"):
        # deliberately outside the library root: a test fixture must not
        # rely on how the store treats stray folders inside it
        path = Path(self._scratch.name) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"RIFFfake")
        return str(path)

    def a_voice(self, name="Narrator", mode=store.MODE_ONESHOT):
        return store.create_character(self.root, name, mode)

    # the four outputs every mutating button returns
    def unpack(self, result):
        select, name, summary, status = result
        return select, name, summary, status


class ModeDropdownTests(_Panel):
    def test_both_voice_types_are_offered(self):
        values = [value for _, value in handlers.mode_choices()]
        self.assertEqual(values, [store.MODE_ONESHOT, store.MODE_RVC])

    def test_the_labels_are_human_readable_not_slugs(self):
        labels = [label for label, _ in handlers.mode_choices()]
        self.assertIn("One-shot reference", labels)

    def test_choices_are_filtered_by_the_selected_type(self):
        self.a_voice("Narrator", store.MODE_ONESHOT)
        self.a_voice("Robot", store.MODE_RVC)
        oneshot = handlers.character_choices(store.MODE_ONESHOT, self.root)
        self.assertEqual([v for _, v in oneshot], ["narrator"])

    def test_the_dropdown_value_is_the_slug_not_the_name(self):
        # A rename must not strand the selection, and two voices can end up
        # with labels that look alike.
        self.a_voice("Narrator")
        (_, value), = handlers.character_choices(store.MODE_ONESHOT, self.root)
        self.assertEqual(value, "narrator")

    def test_the_label_shows_how_many_clips_a_voice_holds(self):
        slug = self.a_voice("Narrator")
        store.add_clip(self.root, slug, self.a_clip())
        (label, _), = handlers.character_choices(store.MODE_ONESHOT, self.root)
        self.assertIn("1 clip", label)

    def test_an_unreadable_voice_is_listed_and_marked(self):
        self.a_voice("Narrator")
        (Path(self.root) / "broken").mkdir()
        (Path(self.root) / "broken" / store.CHARACTER_FILE).write_text("{", encoding="utf-8")
        labels = [label for label, _ in handlers.character_choices(None, self.root)]
        self.assertTrue(any("unreadable" in label for label in labels))


class ModeChangeTests(_Panel):
    def test_switching_type_refills_the_voice_list_and_picks_the_first(self):
        self.a_voice("Robot", store.MODE_RVC)
        select, name, summary, status = handlers.on_mode_change(store.MODE_RVC, self.root)
        self.assertEqual([v for _, v in select["choices"]], ["robot"])
        self.assertEqual(select["value"], "robot")
        self.assertEqual(name["value"], "Robot")

    def test_switching_to_a_type_with_no_voices_clears_the_panel(self):
        self.a_voice("Narrator", store.MODE_ONESHOT)
        select, name, summary, status = handlers.on_mode_change(store.MODE_RVC, self.root)
        self.assertEqual(select["choices"], [])
        self.assertEqual(select["value"], handlers.NO_SELECTION)
        self.assertEqual(name["value"], "")

    def test_switching_type_hides_any_stale_status(self):
        self.a_voice("Narrator")
        *_, status = handlers.on_mode_change(store.MODE_ONESHOT, self.root)
        self.assertFalse(status["visible"])


class SummaryTests(_Panel):
    def test_no_selection_says_so(self):
        self.assertIn("No voice selected", handlers.describe_character(store.MODE_ONESHOT, "", self.root))

    def test_a_voice_deleted_underneath_the_dropdown_says_so(self):
        self.assertIn("no longer in the library",
                      handlers.describe_character(store.MODE_ONESHOT, "ghost", self.root))

    def test_an_empty_voice_tells_you_what_to_do_next(self):
        slug = self.a_voice("Narrator")
        self.assertIn("No clips yet", handlers.describe_character(store.MODE_ONESHOT, slug, self.root))

    def test_an_rvc_voice_with_no_model_says_so(self):
        slug = self.a_voice("Robot", store.MODE_RVC)
        self.assertIn("No model attached", handlers.describe_character(store.MODE_RVC, slug, self.root))

    def test_an_rvc_voice_shows_its_model_path(self):
        slug = self.a_voice("Robot", store.MODE_RVC)
        document = store.load_character(self.root, slug)
        document["rvc"]["model_path"] = r"D:\models\robot.pth"
        store.save_character(self.root, slug, document)
        self.assertIn("robot.pth", handlers.describe_character(store.MODE_RVC, slug, self.root))

    def test_clip_count_and_total_are_reported(self):
        slug = self.a_voice("Narrator")
        store.add_clip(self.root, slug, self.a_clip("a.wav"), {"duration_s": 10.0})
        store.add_clip(self.root, slug, self.a_clip("b.wav"), {"duration_s": 5.0})
        text = handlers.describe_character(store.MODE_ONESHOT, slug, self.root)
        self.assertIn("2 clips", text)
        self.assertIn("15s", text)

    def test_the_default_clip_is_marked(self):
        slug = self.a_voice("Narrator")
        store.add_clip(self.root, slug, self.a_clip("a.wav"), {"duration_s": 1.0}, label="first")
        self.assertIn("**>**", handlers.describe_character(store.MODE_ONESHOT, slug, self.root))

    def test_a_long_library_is_truncated_rather_than_flooding_the_panel(self):
        slug = self.a_voice("Narrator")
        for i in range(12):
            store.add_clip(self.root, slug, self.a_clip(f"c{i}.wav"), {"duration_s": 1.0})
        text = handlers.describe_character(store.MODE_ONESHOT, slug, self.root)
        self.assertIn("and 4 more", text)

    def test_readiness_is_reported_against_the_training_targets(self):
        slug = self.a_voice("Narrator")
        store.add_clip(self.root, slug, self.a_clip("a.wav"), {"duration_s": 5.0})
        self.assertIn("Not enough to train", handlers.describe_character(store.MODE_ONESHOT, slug, self.root))

        store.add_clip(self.root, slug, self.a_clip("b.wav"), {"duration_s": 100.0})
        self.assertIn("GPT-SoVITS", handlers.describe_character(store.MODE_ONESHOT, slug, self.root))

        store.add_clip(self.root, slug, self.a_clip("c.wav"), {"duration_s": 600.0})
        self.assertIn("Enough for RVC", handlers.describe_character(store.MODE_ONESHOT, slug, self.root))

    def test_a_clip_with_measurements_shows_them(self):
        slug = self.a_voice("Narrator")
        store.add_clip(self.root, slug, self.a_clip(), {"duration_s": 12.5, "lufs": -19.3},
                       label="calm narration")
        text = handlers.describe_character(store.MODE_ONESHOT, slug, self.root)
        self.assertIn("calm narration", text)
        self.assertIn("12.5s", text)
        self.assertIn("-19.3 LUFS", text)


class CreateTests(_Panel):
    def test_a_blank_name_is_refused_with_a_reason(self):
        _, _, _, status = handlers.create_character_ui(store.MODE_ONESHOT, "   ", self.root)
        self.assertTrue(status["visible"])
        self.assertIn("name", status["value"].lower())

    def test_creating_selects_the_new_voice(self):
        select, name, _, status = handlers.create_character_ui(store.MODE_ONESHOT, "Narrator", self.root)
        self.assertEqual(select["value"], "narrator")
        self.assertEqual(name["value"], "Narrator")
        self.assertIn("Created", status["value"])

    def test_a_duplicate_name_is_reported_not_raised(self):
        handlers.create_character_ui(store.MODE_ONESHOT, "Narrator", self.root)
        _, _, _, status = handlers.create_character_ui(store.MODE_ONESHOT, "narrator", self.root)
        self.assertIn("already exists", status["value"])

    def test_a_disk_failure_is_reported_not_raised(self):
        with mock.patch.object(store, "create_character", side_effect=OSError(28, "no space")):
            _, _, _, status = handlers.create_character_ui(store.MODE_ONESHOT, "Narrator", self.root)
        self.assertIn("Could not create", status["value"])

    def test_a_new_rvc_voice_lands_in_the_rvc_list(self):
        handlers.create_character_ui(store.MODE_RVC, "Robot", self.root)
        self.assertEqual([v for _, v in handlers.character_choices(store.MODE_RVC, self.root)], ["robot"])
        self.assertEqual(handlers.character_choices(store.MODE_ONESHOT, self.root), [])


class RenameTests(_Panel):
    def test_renaming_with_nothing_selected_is_refused(self):
        _, _, _, status = handlers.rename_character_ui(store.MODE_ONESHOT, "", "New", self.root)
        self.assertIn("Select a voice", status["value"])

    def test_renaming_to_blank_is_refused(self):
        slug = self.a_voice("Narrator")
        _, _, _, status = handlers.rename_character_ui(store.MODE_ONESHOT, slug, "  ", self.root)
        self.assertIn("needs a name", status["value"])

    def test_renaming_follows_the_selection_to_the_new_slug(self):
        slug = self.a_voice("Narrator")
        select, name, _, status = handlers.rename_character_ui(
            store.MODE_ONESHOT, slug, "Storyteller", self.root)
        self.assertEqual(select["value"], "storyteller")
        self.assertEqual(name["value"], "Storyteller")

    def test_renaming_onto_an_existing_name_is_reported_not_raised(self):
        self.a_voice("Narrator")
        other = self.a_voice("Robot", store.MODE_ONESHOT)
        _, _, _, status = handlers.rename_character_ui(store.MODE_ONESHOT, other, "Narrator", self.root)
        self.assertIn("already exists", status["value"])

    def test_a_disk_failure_while_renaming_is_reported(self):
        slug = self.a_voice("Narrator")
        with mock.patch.object(store, "rename_character", side_effect=OSError(13, "denied")):
            _, _, _, status = handlers.rename_character_ui(store.MODE_ONESHOT, slug, "X", self.root)
        self.assertIn("Could not rename", status["value"])


class DeleteTests(_Panel):
    def test_deleting_with_nothing_selected_is_refused(self):
        _, _, _, status = handlers.delete_character_ui(store.MODE_ONESHOT, "", True, self.root)
        self.assertIn("Select a voice", status["value"])

    def test_an_unconfirmed_delete_does_nothing(self):
        slug = self.a_voice("Narrator")
        _, _, _, status = handlers.delete_character_ui(store.MODE_ONESHOT, slug, False, self.root)
        self.assertIn("confirm", status["value"].lower())
        self.assertIsNotNone(store.load_character(self.root, slug))

    def test_a_confirmed_delete_removes_it_and_clears_the_selection(self):
        slug = self.a_voice("Narrator")
        select, name, _, status = handlers.delete_character_ui(store.MODE_ONESHOT, slug, True, self.root)
        self.assertIsNone(store.load_character(self.root, slug))
        self.assertEqual(select["value"], handlers.NO_SELECTION)
        self.assertIn("Deleted", status["value"])

    def test_a_locked_voice_reports_rather_than_raising(self):
        slug = self.a_voice("Narrator")
        with mock.patch.object(store, "delete_character", return_value=False):
            _, _, _, status = handlers.delete_character_ui(store.MODE_ONESHOT, slug, True, self.root)
        self.assertIn("Could not delete", status["value"])


class UseVoiceTests(_Panel):
    def test_using_a_voice_loads_its_default_clip(self):
        slug = self.a_voice("Narrator")
        store.add_clip(self.root, slug, self.a_clip())
        audio, status = handlers.use_character_ui(store.MODE_ONESHOT, slug, self.root)
        self.assertTrue(os.path.isfile(audio["value"]))
        self.assertIn("Narrator", status["value"])

    def test_using_a_voice_with_no_clips_says_so_and_leaves_the_slot_alone(self):
        slug = self.a_voice("Narrator")
        audio, status = handlers.use_character_ui(store.MODE_ONESHOT, slug, self.root)
        self.assertNotIn("value", audio)
        self.assertIn("no usable clip", status["value"])

    def test_using_with_nothing_selected_says_so(self):
        audio, status = handlers.use_character_ui(store.MODE_ONESHOT, "", self.root)
        self.assertNotIn("value", audio)
        self.assertIn("Select a voice", status["value"])

    def test_an_rvc_voice_says_it_is_not_wired_up_yet(self):
        # Honest rather than silently doing nothing: RVC generation is stage 7.
        slug = self.a_voice("Robot", store.MODE_RVC)
        audio, status = handlers.use_character_ui(store.MODE_RVC, slug, self.root)
        self.assertNotIn("value", audio)
        self.assertIn("not wired into generation yet", status["value"])

    def test_a_clip_whose_file_vanished_says_so(self):
        slug = self.a_voice("Narrator")
        clip = store.add_clip(self.root, slug, self.a_clip())
        (store.character_dir(self.root, slug) / clip["file"]).unlink()
        audio, status = handlers.use_character_ui(store.MODE_ONESHOT, slug, self.root)
        self.assertNotIn("value", audio)
        self.assertIn("no usable clip", status["value"])


class AddReferenceTests(_Panel):
    def test_adding_the_loaded_reference_files_it_under_the_voice(self):
        slug = self.a_voice("Narrator")
        _, _, summary, status = handlers.add_reference_to_character_ui(
            store.MODE_ONESHOT, slug, self.a_clip(), "", self.root)
        self.assertIn("Added", status["value"])
        self.assertEqual(len(store.load_character(self.root, slug)["oneshot"]["clips"]), 1)

    def test_adding_with_no_reference_loaded_says_so(self):
        slug = self.a_voice("Narrator")
        _, _, _, status = handlers.add_reference_to_character_ui(
            store.MODE_ONESHOT, slug, None, "", self.root)
        self.assertIn("Load a reference", status["value"])

    def test_adding_with_no_voice_selected_says_so(self):
        _, _, _, status = handlers.add_reference_to_character_ui(
            store.MODE_ONESHOT, "", self.a_clip(), "", self.root)
        self.assertIn("Select a voice", status["value"])

    def test_adding_to_an_rvc_voice_is_refused(self):
        slug = self.a_voice("Robot", store.MODE_RVC)
        _, _, _, status = handlers.add_reference_to_character_ui(
            store.MODE_RVC, slug, self.a_clip(), "", self.root)
        self.assertIn("model, not clips", status["value"])

    def test_a_disk_failure_while_adding_is_reported(self):
        slug = self.a_voice("Narrator")
        with mock.patch.object(store, "add_clip", side_effect=OSError(28, "no space")):
            _, _, _, status = handlers.add_reference_to_character_ui(
                store.MODE_ONESHOT, slug, self.a_clip(), "", self.root)
        self.assertIn("Could not add", status["value"])

    def test_a_missing_source_file_is_reported_not_raised(self):
        slug = self.a_voice("Narrator")
        _, _, _, status = handlers.add_reference_to_character_ui(
            store.MODE_ONESHOT, slug, os.path.join(self.root, "ghost.wav"), "", self.root)
        self.assertIn("No such audio file", status["value"])

    def test_a_real_wav_gets_its_duration_measured_on_the_way_in(self):
        fixture = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_prompt.wav")
        if not os.path.isfile(fixture):
            self.skipTest("sample_prompt.wav not present")
        slug = self.a_voice("Narrator")
        handlers.add_reference_to_character_ui(store.MODE_ONESHOT, slug, fixture, "", self.root)
        clip = store.load_character(self.root, slug)["oneshot"]["clips"][0]
        self.assertGreater(clip["duration_s"], 0.0)

    def test_a_file_that_is_not_a_wav_is_stored_without_a_guessed_duration(self):
        # Better unmeasured than wrong: the segmentation stage fills it in.
        slug = self.a_voice("Narrator")
        handlers.add_reference_to_character_ui(store.MODE_ONESHOT, slug, self.a_clip("x.mp3"), "", self.root)
        clip = store.load_character(self.root, slug)["oneshot"]["clips"][0]
        self.assertNotIn("duration_s", clip)


class WavDurationTests(_Panel):
    """Reading a duration off the header, on the UI thread, between clicks."""

    def test_a_wav_with_no_sample_rate_is_left_unmeasured(self):
        # A malformed header reporting rate 0 would divide by zero.
        class _Handle:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def getframerate(self): return 0
            def getnframes(self): return 1000
        with mock.patch('wave.open', return_value=_Handle()):
            self.assertIsNone(handlers._wav_duration_seconds('anything.wav'))

    def test_a_file_that_is_not_a_wav_is_left_unmeasured(self):
        self.assertIsNone(handlers._wav_duration_seconds(self.a_clip('x.mp3')))

    def test_a_missing_file_is_left_unmeasured(self):
        self.assertIsNone(handlers._wav_duration_seconds(os.path.join(self.root, 'ghost.wav')))


class SelectionConsistencyTests(_Panel):
    """The four outputs must always describe the same voice."""

    def test_the_name_box_and_summary_follow_the_selection(self):
        self.a_voice("Narrator")
        slug_b = self.a_voice("Robot", store.MODE_ONESHOT)
        store.add_clip(self.root, slug_b, self.a_clip(), {"duration_s": 3.0})
        name, summary, status = handlers.on_character_change(store.MODE_ONESHOT, slug_b, self.root)
        self.assertEqual(name["value"], "Robot")
        self.assertIn("1 clip", summary["value"])

    def test_deleting_the_selected_voice_falls_back_to_a_surviving_one(self):
        # Comparing the name box against _name_of(selection) is not enough:
        # if the dropdown keeps pointing at the deleted slug, both are "" and
        # the comparison passes while the panel shows a voice that is gone.
        # Assert the selection lands on something that actually exists.
        self.a_voice("Narrator")
        slug_b = self.a_voice("Robot", store.MODE_ONESHOT)
        select, name, summary, _ = handlers.delete_character_ui(
            store.MODE_ONESHOT, slug_b, True, self.root)
        self.assertEqual(select["value"], "narrator")
        self.assertEqual(name["value"], "Narrator")
        surviving = [v for _, v in select["choices"]]
        self.assertIn(select["value"], surviving)

    def test_deleting_the_only_voice_leaves_an_empty_but_coherent_panel(self):
        slug = self.a_voice("Narrator")
        select, name, summary, _ = handlers.delete_character_ui(
            store.MODE_ONESHOT, slug, True, self.root)
        self.assertEqual(select["choices"], [])
        self.assertEqual(select["value"], handlers.NO_SELECTION)
        self.assertEqual(name["value"], "")
        self.assertIn("No voice selected", summary["value"])

    def test_a_refused_rename_keeps_the_original_selection(self):
        # _refresh must hold the selection when the operation did not change it.
        self.a_voice("Narrator")
        other = self.a_voice("Robot", store.MODE_ONESHOT)
        select, name, _, _ = handlers.rename_character_ui(
            store.MODE_ONESHOT, other, "Narrator", self.root)
        self.assertEqual(select["value"], other)
        self.assertEqual(name["value"], "Robot")

    def test_initial_state_is_coherent_on_an_empty_library(self):
        mode, choices, first, name, summary = handlers.initial_state(self.root)
        self.assertEqual(mode, store.MODE_ONESHOT)
        self.assertEqual(choices, [])
        self.assertEqual(first, handlers.NO_SELECTION)
        self.assertEqual(name, "")
        self.assertIn("No voice selected", summary)

    def test_initial_state_picks_the_first_voice_when_there_is_one(self):
        self.a_voice("Narrator")
        mode, choices, first, name, summary = handlers.initial_state(self.root)
        self.assertEqual(first, "narrator")
        self.assertEqual(name, "Narrator")


class NoHandlerRaisesTests(_Panel):
    """A traceback out of any of these shows the user a stalled spinner."""

    def test_every_handler_survives_a_library_that_cannot_be_read(self):
        slug = self.a_voice("Narrator")
        with mock.patch.object(Path, "iterdir", side_effect=PermissionError(13, "denied")):
            handlers.character_choices(store.MODE_ONESHOT, self.root)
            handlers.on_mode_change(store.MODE_ONESHOT, self.root)
            handlers.on_character_change(store.MODE_ONESHOT, slug, self.root)
            handlers.create_character_ui(store.MODE_ONESHOT, "X", self.root)
            handlers.rename_character_ui(store.MODE_ONESHOT, slug, "Y", self.root)
            handlers.delete_character_ui(store.MODE_ONESHOT, slug, True, self.root)
            handlers.use_character_ui(store.MODE_ONESHOT, slug, self.root)
            handlers.add_reference_to_character_ui(
                store.MODE_ONESHOT, slug, self.a_clip(), "", self.root)
            handlers.initial_state(self.root)

    def test_every_handler_survives_a_slug_that_is_not_there(self):
        for call in (
            lambda: handlers.on_character_change(store.MODE_ONESHOT, "ghost", self.root),
            lambda: handlers.rename_character_ui(store.MODE_ONESHOT, "ghost", "X", self.root),
            lambda: handlers.delete_character_ui(store.MODE_ONESHOT, "ghost", True, self.root),
            lambda: handlers.use_character_ui(store.MODE_ONESHOT, "ghost", self.root),
            lambda: handlers.add_reference_to_character_ui(
                store.MODE_ONESHOT, "ghost", self.a_clip(), "", self.root),
            lambda: handlers.describe_character(store.MODE_ONESHOT, "ghost", self.root),
        ):
            call()


class SaveLoadedVoiceTests(_Panel):
    """The one-press path: a reference is loaded and it wants a name.

    Added after the three-step version -- name it, press New, press Add --
    turned out to be undiscoverable in use. A voice was created and left
    with zero clips because nothing on the panel connected the loaded
    reference to the selected voice.
    """

    def test_one_press_creates_the_voice_and_files_the_clip(self):
        select, name, summary, status = handlers.save_reference_as_new_voice_ui(
            store.MODE_ONESHOT, "Narrator", self.a_clip(), self.root)
        self.assertEqual(select["value"], "narrator")
        self.assertIn("Saved", status["value"])
        self.assertEqual(
            len(store.load_character(self.root, "narrator")["oneshot"]["clips"]), 1)

    def test_saving_again_under_the_same_name_adds_rather_than_refusing(self):
        # "Already exists" would be technically right and useless: the clip
        # in hand is almost certainly meant for that voice.
        handlers.save_reference_as_new_voice_ui(
            store.MODE_ONESHOT, "Narrator", self.a_clip("one.wav"), self.root)
        *_, status = handlers.save_reference_as_new_voice_ui(
            store.MODE_ONESHOT, "Narrator", self.a_clip("two.wav"), self.root)
        self.assertIn("Added", status["value"])
        self.assertEqual(
            len(store.load_character(self.root, "narrator")["oneshot"]["clips"]), 2)

    def test_saving_with_no_reference_loaded_says_so(self):
        *_, status = handlers.save_reference_as_new_voice_ui(
            store.MODE_ONESHOT, "Narrator", None, self.root)
        self.assertIn("Load a reference", status["value"])
        self.assertEqual(handlers.character_choices(store.MODE_ONESHOT, self.root), [])

    def test_saving_with_no_name_says_so_and_creates_nothing(self):
        *_, status = handlers.save_reference_as_new_voice_ui(
            store.MODE_ONESHOT, "   ", self.a_clip(), self.root)
        self.assertIn("name", status["value"].lower())
        self.assertEqual(handlers.character_choices(store.MODE_ONESHOT, self.root), [])

    def test_saving_into_rvc_mode_is_refused(self):
        *_, status = handlers.save_reference_as_new_voice_ui(
            store.MODE_RVC, "Robot", self.a_clip(), self.root)
        self.assertIn("model, not clips", status["value"])

    def test_a_disk_failure_while_saving_is_reported(self):
        with mock.patch.object(store, "create_character", side_effect=OSError(28, "no space")):
            *_, status = handlers.save_reference_as_new_voice_ui(
                store.MODE_ONESHOT, "Narrator", self.a_clip(), self.root)
        self.assertIn("Could not create", status["value"])


if __name__ == "__main__":
    unittest.main()
