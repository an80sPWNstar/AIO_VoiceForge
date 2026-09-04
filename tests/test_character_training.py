"""Training a character's RVC model, with the trainer faked.

The dataset export runs for real against a temp library (it is cheap and
already covered elsewhere); rvc_engine.train is mocked, because what is
under test is the connector's contract -- what gets trained, and what gets
recorded where, and only after training proved its artifacts exist.
"""

import os
import sys
import tempfile
import unittest
import wave
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import character_dataset
import character_store
import character_training
import rvc_engine


def write_wav(path, seconds=1.0, rate=16000):
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))
    return path


FAKE_ARTIFACTS = {"model_path": r"D:\Applio\logs\voiceforge_narrator\x_40e_240s.pth",
                  "index_path": r"D:\Applio\logs\voiceforge_narrator\x.index"}


class TrainCharacterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = os.path.join(self.tmp.name, "library")
        self.slug = character_store.create_character(
            self.root, "Narrator", character_store.MODE_ONESHOT)
        source = write_wav(os.path.join(self.tmp.name, "clip.wav"), 2.0)
        character_store.add_clip(self.root, self.slug, source)

    def test_trains_on_the_exported_dataset_and_records_the_artifacts(self):
        def check_dataset(model_name, dataset_dir, **kwargs):
            self.assertEqual(model_name, f"voiceforge_{self.slug}")
            wavs = [f for f in os.listdir(dataset_dir) if f.endswith(".wav")]
            self.assertEqual(len(wavs), 1)
            return dict(FAKE_ARTIFACTS)
        with mock.patch.object(rvc_engine, "train", side_effect=check_dataset):
            result = character_training.train_character(self.root, self.slug)
        document = character_store.load_character(self.root, self.slug)
        self.assertEqual(document["rvc"]["model_path"], FAKE_ARTIFACTS["model_path"])
        self.assertEqual(document["rvc"]["index_path"], FAKE_ARTIFACTS["index_path"])
        self.assertAlmostEqual(document["rvc"]["trained_from_seconds"], 2.0, places=1)
        self.assertEqual(result["artifacts"], FAKE_ARTIFACTS)

    def test_the_stored_transpose_survives_retraining(self):
        document = character_store.load_character(self.root, self.slug)
        document["rvc"]["transpose"] = -3
        character_store.save_character(self.root, self.slug, document)
        with mock.patch.object(rvc_engine, "train", return_value=dict(FAKE_ARTIFACTS)):
            character_training.train_character(self.root, self.slug)
        document = character_store.load_character(self.root, self.slug)
        self.assertEqual(document["rvc"]["transpose"], -3)

    def test_a_failed_training_leaves_the_character_untouched(self):
        # A character must never point at weights that do not exist.
        with mock.patch.object(rvc_engine, "train",
                               side_effect=rvc_engine.RVCError("no artifacts")):
            with self.assertRaises(rvc_engine.RVCError):
                character_training.train_character(self.root, self.slug)
        document = character_store.load_character(self.root, self.slug)
        self.assertIsNone(document["rvc"]["model_path"])
        self.assertIsNone(document["rvc"]["index_path"])

    def test_a_character_with_no_clips_fails_before_training(self):
        empty = character_store.create_character(
            self.root, "Empty", character_store.MODE_ONESHOT)
        with mock.patch.object(rvc_engine, "train") as train:
            with self.assertRaises(character_dataset.DatasetExportError):
                character_training.train_character(self.root, empty)
        train.assert_not_called()

    def test_a_missing_character_fails_before_anything(self):
        with mock.patch.object(rvc_engine, "train") as train:
            with self.assertRaises(character_dataset.DatasetExportError):
                character_training.train_character(self.root, "nobody")
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
