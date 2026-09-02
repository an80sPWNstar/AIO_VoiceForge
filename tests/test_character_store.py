"""The character library: named voices and the clips that make them up.

Everything runs against a temp directory, which is the reason the store
takes its root as a parameter instead of reading a module constant. No
audio libraries and no gradio are involved -- the clips here are a few
bytes of nonsense, because the store never looks inside them.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import character_store as cs


class _TempLibrary(unittest.TestCase):
    """Base: a throwaway library root and a throwaway audio file."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = self._dir.name
        self.addCleanup(self._dir.cleanup)

    def a_clip(self, name="sample.wav", payload=b"RIFFfake"):
        path = Path(self.root) / "_incoming" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)


class SlugTests(unittest.TestCase):
    def test_spaces_and_punctuation_become_dashes(self):
        self.assertEqual(cs.slugify("Captain O'Hara!"), "captain-o-hara")

    def test_case_is_folded(self):
        self.assertEqual(cs.slugify("NarratorVoice"), "narratorvoice")

    def test_an_empty_name_still_yields_a_usable_folder(self):
        # A blank name must not produce "" and then a write to the library root.
        self.assertEqual(cs.slugify("   "), "unnamed")
        self.assertEqual(cs.slugify("!!!"), "unnamed")

    def test_a_very_long_name_is_bounded(self):
        self.assertLessEqual(len(cs.slugify("x" * 500)), 64)

    def test_path_separators_cannot_escape_the_library(self):
        # The slug becomes a directory name, so anything that could climb out
        # of the root has to be neutralised.
        for hostile in ("../../etc/passwd", r"..\..\windows", "a/b/c"):
            slug = cs.slugify(hostile)
            self.assertNotIn("/", slug)
            self.assertNotIn("\\", slug)
            self.assertFalse(slug.startswith("."), slug)


class CreateAndLoadTests(_TempLibrary):
    def test_a_new_character_round_trips(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        document = cs.load_character(self.root, slug)
        self.assertEqual(document["name"], "Narrator")
        self.assertEqual(document["mode"], cs.MODE_ONESHOT)
        self.assertEqual(document["oneshot"]["clips"], [])

    def test_creating_makes_the_clips_folder(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        self.assertTrue(cs.clips_dir(self.root, slug).is_dir())

    def test_a_duplicate_name_is_refused_rather_than_merged(self):
        # Two voices sharing a slug would share a clips folder, and the second
        # save would overwrite the first document.
        cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        with self.assertRaises(cs.CharacterStoreError):
            cs.create_character(self.root, "narrator", cs.MODE_RVC)

    def test_an_unknown_voice_type_is_refused(self):
        with self.assertRaises(cs.CharacterStoreError):
            cs.create_character(self.root, "Narrator", "sqlite")

    def test_loading_something_that_is_not_there_is_none_not_an_error(self):
        self.assertIsNone(cs.load_character(self.root, "nobody"))

    def test_timestamps_are_taken_from_the_injected_clock(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT, now=0)
        meta = cs.load_character(self.root, slug)["_meta"]
        self.assertEqual(meta["created_at"], meta["updated_at"])
        self.assertTrue(meta["created_at"].startswith("19") or meta["created_at"].startswith("20"))


class ListingTests(_TempLibrary):
    def setUp(self):
        super().setUp()
        cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        cs.create_character(self.root, "Robot", cs.MODE_RVC)

    def test_listing_returns_both(self):
        self.assertEqual(len(cs.list_characters(self.root)), 2)

    def test_listing_filters_by_voice_type(self):
        # This is what the second dropdown does when the first one changes.
        oneshot = cs.list_characters(self.root, mode=cs.MODE_ONESHOT)
        self.assertEqual([c["slug"] for c in oneshot], ["narrator"])

    def test_an_empty_library_lists_nothing_rather_than_failing(self):
        self.assertEqual(cs.list_characters(os.path.join(self.root, "nope")), [])

    def test_one_corrupt_character_does_not_empty_the_list(self):
        # A half-written or hand-edited file must not take the dropdown with it.
        (Path(self.root) / "broken").mkdir()
        (Path(self.root) / "broken" / cs.CHARACTER_FILE).write_text("{not json", encoding="utf-8")
        summaries = cs.list_characters(self.root)
        self.assertEqual(len(summaries), 3)
        self.assertTrue(any(s["unreadable"] for s in summaries))

    def test_a_stray_file_in_the_library_root_is_ignored(self):
        (Path(self.root) / "notes.txt").write_text("hello", encoding="utf-8")
        self.assertEqual(len(cs.list_characters(self.root)), 2)


class RenameTests(_TempLibrary):
    def test_renaming_within_the_same_slug_keeps_the_folder(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        new_slug = cs.rename_character(self.root, slug, "narrator")
        self.assertEqual(new_slug, slug)
        self.assertEqual(cs.load_character(self.root, new_slug)["name"], "narrator")

    def test_renaming_to_a_new_slug_moves_the_folder_and_its_clips(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        cs.add_clip(self.root, slug, self.a_clip(), {"duration_s": 4.0})
        new_slug = cs.rename_character(self.root, slug, "Storyteller")
        self.assertEqual(new_slug, "storyteller")
        self.assertFalse(cs.character_dir(self.root, slug).exists())
        self.assertEqual(len(cs.load_character(self.root, new_slug)["oneshot"]["clips"]), 1)
        self.assertIsNotNone(cs.resolve_clip_path(self.root, new_slug))

    def test_renaming_onto_an_existing_name_is_refused(self):
        cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        other = cs.create_character(self.root, "Robot", cs.MODE_ONESHOT)
        with self.assertRaises(cs.CharacterStoreError):
            cs.rename_character(self.root, other, "Narrator")

    def test_renaming_something_absent_is_refused(self):
        with self.assertRaises(cs.CharacterStoreError):
            cs.rename_character(self.root, "nobody", "Someone")


class ClipTests(_TempLibrary):
    def setUp(self):
        super().setUp()
        self.slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)

    def test_adding_a_clip_copies_the_file_in(self):
        source = self.a_clip()
        clip = cs.add_clip(self.root, self.slug, source, {"duration_s": 12.5})
        stored = cs.character_dir(self.root, self.slug) / clip["file"]
        self.assertTrue(stored.is_file())
        self.assertTrue(os.path.isfile(source), "the source must not be moved or consumed")

    def test_measurements_are_kept_beside_the_audio(self):
        clip = cs.add_clip(self.root, self.slug, self.a_clip(), {
            "duration_s": 12.5, "lufs": -19.3, "pitch_hz_mean": 118.0,
            "transcript": "hello there", "emotion": {"label": "neutral", "confidence": 0.6},
        })
        self.assertEqual(clip["lufs"], -19.3)
        self.assertEqual(clip["emotion"]["label"], "neutral")

    def test_the_first_clip_becomes_the_default(self):
        clip = cs.add_clip(self.root, self.slug, self.a_clip())
        self.assertEqual(cs.load_character(self.root, self.slug)["oneshot"]["default_clip_id"], clip["id"])

    def test_a_later_clip_does_not_steal_default_unless_asked(self):
        first = cs.add_clip(self.root, self.slug, self.a_clip("one.wav"))
        cs.add_clip(self.root, self.slug, self.a_clip("two.wav"))
        self.assertEqual(cs.load_character(self.root, self.slug)["oneshot"]["default_clip_id"], first["id"])

    def test_a_clip_can_claim_default_on_the_way_in(self):
        cs.add_clip(self.root, self.slug, self.a_clip("one.wav"))
        second = cs.add_clip(self.root, self.slug, self.a_clip("two.wav"), make_default=True)
        self.assertEqual(cs.load_character(self.root, self.slug)["oneshot"]["default_clip_id"], second["id"])

    def test_clips_do_not_collide_when_the_source_names_match(self):
        # Two files both called sample.wav from different folders must not
        # overwrite each other inside the character.
        a = cs.add_clip(self.root, self.slug, self.a_clip("sample.wav", b"AAAA"))
        b = cs.add_clip(self.root, self.slug, self.a_clip("sample.wav", b"BBBB"))
        self.assertNotEqual(a["file"], b["file"])
        self.assertEqual(len(list(cs.clips_dir(self.root, self.slug).iterdir())), 2)

    def test_adding_to_a_character_that_is_not_there_is_refused(self):
        with self.assertRaises(cs.CharacterStoreError):
            cs.add_clip(self.root, "nobody", self.a_clip())

    def test_adding_an_audio_file_that_is_not_there_is_refused(self):
        with self.assertRaises(cs.CharacterStoreError):
            cs.add_clip(self.root, self.slug, os.path.join(self.root, "ghost.wav"))

    def test_removing_a_clip_deletes_its_file_too(self):
        # Otherwise the clips folder grows forever with audio nothing points at.
        clip = cs.add_clip(self.root, self.slug, self.a_clip())
        stored = cs.character_dir(self.root, self.slug) / clip["file"]
        self.assertTrue(cs.remove_clip(self.root, self.slug, clip["id"]))
        self.assertFalse(stored.exists())

    def test_removing_the_default_promotes_another_clip(self):
        # Leaving default_clip_id pointing at a deleted clip would make
        # "use this voice" fail with nothing to explain it.
        first = cs.add_clip(self.root, self.slug, self.a_clip("one.wav"))
        second = cs.add_clip(self.root, self.slug, self.a_clip("two.wav"))
        cs.remove_clip(self.root, self.slug, first["id"])
        self.assertEqual(cs.load_character(self.root, self.slug)["oneshot"]["default_clip_id"], second["id"])

    def test_removing_the_last_clip_clears_the_default(self):
        clip = cs.add_clip(self.root, self.slug, self.a_clip())
        cs.remove_clip(self.root, self.slug, clip["id"])
        self.assertIsNone(cs.load_character(self.root, self.slug)["oneshot"]["default_clip_id"])

    def test_removing_a_clip_that_is_not_there_reports_false(self):
        self.assertFalse(cs.remove_clip(self.root, self.slug, "nosuchid"))

    def test_setting_a_default_that_does_not_exist_is_refused(self):
        self.assertFalse(cs.set_default_clip(self.root, self.slug, "nosuchid"))


class ResolveTests(_TempLibrary):
    def setUp(self):
        super().setUp()
        self.slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)

    def test_resolving_returns_the_default_when_no_clip_is_named(self):
        cs.add_clip(self.root, self.slug, self.a_clip())
        self.assertTrue(os.path.isfile(cs.resolve_clip_path(self.root, self.slug)))

    def test_resolving_a_character_with_no_clips_is_none(self):
        self.assertIsNone(cs.resolve_clip_path(self.root, self.slug))

    def test_resolving_an_unknown_character_is_none(self):
        self.assertIsNone(cs.resolve_clip_path(self.root, "nobody"))

    def test_a_document_pointing_at_a_deleted_file_resolves_to_none(self):
        # The UI has to be able to say "that clip is gone" rather than hand a
        # missing path to the engine.
        clip = cs.add_clip(self.root, self.slug, self.a_clip())
        (cs.character_dir(self.root, self.slug) / clip["file"]).unlink()
        self.assertIsNone(cs.resolve_clip_path(self.root, self.slug))


class DurabilityTests(_TempLibrary):
    def test_saving_leaves_no_temp_file_behind(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        leftovers = [p.name for p in cs.character_dir(self.root, slug).iterdir()
                     if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

    def test_the_saved_document_is_valid_json_on_disk(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        raw = (cs.character_dir(self.root, slug) / cs.CHARACTER_FILE).read_text(encoding="utf-8")
        self.assertEqual(json.loads(raw)["name"], "Narrator")

    def test_a_non_ascii_name_survives_the_round_trip(self):
        slug = cs.create_character(self.root, "Sørensen 日本", cs.MODE_ONESHOT)
        self.assertEqual(cs.load_character(self.root, slug)["name"], "Sørensen 日本")

    def test_deleting_removes_the_whole_folder(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        cs.add_clip(self.root, slug, self.a_clip())
        self.assertTrue(cs.delete_character(self.root, slug))
        self.assertFalse(cs.character_dir(self.root, slug).exists())

    def test_deleting_something_absent_reports_false(self):
        self.assertFalse(cs.delete_character(self.root, "nobody"))


class UnreadableLibraryTests(_TempLibrary):
    """Failures the filesystem can hand back, all found by the review pass.

    Each one used to surface as a raw traceback out of a UI handler, which is
    the worst place for it -- the user sees a spinner stop and nothing else.

    These force the error rather than trying to arrange it on disk. An earlier
    version of this class created a directory where a file was expected and a
    file where a directory was expected; both returned early at the is_file()
    and is_dir() guards, so the handlers under test never ran and all three
    tests passed against a build with the handling removed.
    """

    def test_a_character_file_that_cannot_be_opened_is_skipped_not_fatal(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        with mock.patch("builtins.open", side_effect=PermissionError(13, "denied")):
            self.assertIsNone(cs.load_character(self.root, slug))

    def test_one_unopenable_character_does_not_empty_the_dropdown(self):
        cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        with mock.patch("builtins.open", side_effect=PermissionError(13, "denied")):
            summaries = cs.list_characters(self.root)
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0]["unreadable"])

    def test_a_library_root_that_cannot_be_listed_returns_nothing(self):
        cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        with mock.patch.object(Path, "iterdir", side_effect=PermissionError(13, "denied")):
            self.assertEqual(cs.list_characters(self.root), [])

    def test_an_entry_that_cannot_be_stat_ed_is_skipped(self):
        cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        with mock.patch.object(Path, "is_dir", side_effect=OSError(5, "io error")):
            self.assertEqual(cs.list_characters(self.root), [])

    def test_a_clip_file_that_cannot_be_deleted_still_leaves_the_document_correct(self):
        # The document is what matters. A file we could not unlink is untidy;
        # a clip the user removed reappearing in the library is wrong.
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        clip = cs.add_clip(self.root, slug, self.a_clip())
        with mock.patch.object(Path, "unlink", side_effect=PermissionError(13, "locked")):
            self.assertTrue(cs.remove_clip(self.root, slug, clip["id"]))
        self.assertEqual(cs.load_character(self.root, slug)["oneshot"]["clips"], [])

    def test_a_character_that_cannot_be_deleted_reports_false_not_a_traceback(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        with mock.patch("shutil.rmtree", side_effect=PermissionError(13, "locked")):
            self.assertFalse(cs.delete_character(self.root, slug))

    def test_a_failed_save_does_not_leave_a_half_made_character_behind(self):
        # Otherwise the folder shows in the dropdown forever as unreadable.
        with mock.patch.object(cs, "_write_json_atomic", side_effect=OSError(28, "no space")):
            with self.assertRaises(OSError):
                cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        self.assertEqual(cs.list_characters(self.root), [])

    def test_a_failed_save_after_copying_a_clip_removes_the_orphan(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        with mock.patch.object(cs, "_write_json_atomic", side_effect=OSError(28, "no space")):
            with self.assertRaises(OSError):
                cs.add_clip(self.root, slug, self.a_clip())
        # No audio left behind that no document refers to.
        self.assertEqual(list(cs.clips_dir(self.root, slug).iterdir()), [])


class TrainingReadinessTests(_TempLibrary):
    """total_clip_seconds is what tells you a character is worth training from."""

    def setUp(self):
        super().setUp()
        self.slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)

    def test_it_sums_the_measured_durations(self):
        cs.add_clip(self.root, self.slug, self.a_clip("a.wav"), {"duration_s": 10.0})
        cs.add_clip(self.root, self.slug, self.a_clip("b.wav"), {"duration_s": 5.5})
        self.assertAlmostEqual(cs.total_clip_seconds(cs.load_character(self.root, self.slug)), 15.5)

    def test_a_clip_with_no_measured_duration_counts_as_zero(self):
        # Better than guessing, and better than crashing on a clip that was
        # added before the analysis ran.
        cs.add_clip(self.root, self.slug, self.a_clip("a.wav"), {"duration_s": 10.0})
        cs.add_clip(self.root, self.slug, self.a_clip("b.wav"))
        self.assertAlmostEqual(cs.total_clip_seconds(cs.load_character(self.root, self.slug)), 10.0)

    def test_an_empty_character_is_zero_seconds(self):
        self.assertEqual(cs.total_clip_seconds(cs.load_character(self.root, self.slug)), 0.0)

    def test_an_unusable_duration_counts_as_zero_rather_than_crashing(self):
        # Found by a review pass on the local box. A hand-edited document, or a
        # clip stored before the analysis ran, would otherwise take the whole
        # library readout down with a ValueError.
        document = {"oneshot": {"clips": [
            {"id": "a", "duration_s": 10.0},
            {"id": "b", "duration_s": "not-a-number"},
            {"id": "c", "duration_s": None},
            {"id": "d"},
        ]}}
        self.assertAlmostEqual(cs.total_clip_seconds(document), 10.0)

    def test_a_numeric_string_duration_still_counts(self):
        # JSON round-trips can produce "12.5"; treating that as zero would
        # under-report readiness just as badly as crashing.
        document = {"oneshot": {"clips": [{"id": "a", "duration_s": "12.5"}]}}
        self.assertAlmostEqual(cs.total_clip_seconds(document), 12.5)


if __name__ == "__main__":
    unittest.main()
