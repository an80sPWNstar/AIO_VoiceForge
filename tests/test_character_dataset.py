"""Exporting a character's clips as a training dataset.

Everything runs against a temp library. Real wavs are generated with the
wave module so duration comes from actual audio, not from trusting the
store's metadata -- the export reads the files it ships.
"""

import json
import os
import sys
import tempfile
import unittest
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import character_dataset
import character_store


def write_wav(path, seconds=1.0, rate=16000):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


class ExportDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "library")
        self.dest = os.path.join(self.tmp.name, "dataset")
        self.slug = character_store.create_character(
            self.root, "Narrator", character_store.MODE_ONESHOT)

    def _add_clip(self, name, seconds=1.0):
        source = write_wav(os.path.join(self.tmp.name, name), seconds)
        return character_store.add_clip(self.root, self.slug, source)

    def test_exports_every_clip_with_slug_prefixed_names(self):
        self._add_clip("a.wav")
        self._add_clip("b.wav")
        manifest = character_dataset.export_dataset(self.root, self.slug, self.dest)
        self.assertEqual(len(manifest["clips"]), 2)
        for clip in manifest["clips"]:
            self.assertTrue(clip["file"].startswith(f"{self.slug}_"))
            self.assertTrue(os.path.isfile(os.path.join(self.dest, clip["file"])))

    def test_durations_come_from_the_audio_not_the_metadata(self):
        self._add_clip("a.wav", seconds=2.5)
        manifest = character_dataset.export_dataset(self.root, self.slug, self.dest)
        self.assertAlmostEqual(manifest["total_seconds"], 2.5, places=2)

    def test_a_manifest_lands_beside_the_audio(self):
        self._add_clip("a.wav")
        character_dataset.export_dataset(self.root, self.slug, self.dest)
        path = os.path.join(self.dest, character_dataset.DATASET_MANIFEST_NAME)
        manifest = json.loads(open(path, encoding="utf-8").read())
        self.assertEqual(manifest["slug"], self.slug)

    def test_thin_audio_warns_but_still_exports(self):
        self._add_clip("a.wav", seconds=1.0)
        manifest = character_dataset.export_dataset(self.root, self.slug, self.dest)
        self.assertTrue(any("RVC" in w for w in manifest["warnings"]))

    def test_enough_audio_does_not_warn_about_thinness(self):
        # Ten one-second clips against a lowered floor: the warning keys off
        # the constant, so the test moves the constant rather than writing
        # ten minutes of wav to disk.
        for i in range(10):
            self._add_clip(f"clip{i}.wav", seconds=1.0)
        from unittest import mock
        with mock.patch.object(character_dataset, "RVC_RECOMMENDED_SECONDS", 9.0):
            manifest = character_dataset.export_dataset(self.root, self.slug, self.dest)
        self.assertEqual(manifest["warnings"], [])

    def test_an_unreadable_clip_is_skipped_and_named(self):
        self._add_clip("good.wav")
        bad = self._add_clip("bad.wav")
        clip_path = os.path.join(
            character_store.character_dir(self.root, self.slug), bad["file"])
        with open(clip_path, "wb") as handle:
            handle.write(b"not audio at all")
        manifest = character_dataset.export_dataset(self.root, self.slug, self.dest)
        self.assertEqual(len(manifest["clips"]), 1)
        self.assertTrue(any(bad["id"] in w for w in manifest["warnings"]))

    def test_no_character_is_an_error(self):
        with self.assertRaises(character_dataset.DatasetExportError):
            character_dataset.export_dataset(self.root, "nobody", self.dest)

    def test_no_clips_is_an_error(self):
        with self.assertRaises(character_dataset.DatasetExportError):
            character_dataset.export_dataset(self.root, self.slug, self.dest)

    def test_every_clip_unreadable_is_an_error_not_an_empty_dataset(self):
        # An empty dataset directory that looks ready to train from would
        # waste a GPU run; refusing is the honest failure.
        clip = self._add_clip("only.wav")
        clip_path = os.path.join(
            character_store.character_dir(self.root, self.slug), clip["file"])
        with open(clip_path, "wb") as handle:
            handle.write(b"garbage")
        with self.assertRaises(character_dataset.DatasetExportError):
            character_dataset.export_dataset(self.root, self.slug, self.dest)


if __name__ == "__main__":
    unittest.main()
