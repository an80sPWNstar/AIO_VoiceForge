"""Tests for LoRA training integration with the character store."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import character_store as cs
import character_training
import character_dataset
import lora_engine


class _TempLibrary(unittest.TestCase):
    """Base: a throwaway library root."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = self._dir.name
        self.addCleanup(self._dir.cleanup)


class LoraBlockTests(_TempLibrary):
    """LoRA block presence and normalization."""

    def test_new_characters_carry_the_lora_block_with_defaults(self):
        slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)
        document = cs.load_character(self.root, slug)
        self.assertIn("lora", document)
        self.assertIsNone(document["lora"]["adapter_path"])
        self.assertEqual(document["lora"]["strength"], 1.0)

    def test_an_old_document_without_lora_loads_with_the_default_block(self):
        # Create a character folder and manually write a JSON without the lora key
        slug = "narrator"
        char_dir = cs.character_dir(self.root, slug)
        char_dir.mkdir(parents=True, exist_ok=True)
        (char_dir / cs.CLIPS_DIRNAME).mkdir(parents=True, exist_ok=True)

        old_document = {
            "_meta": {
                "format": cs.CHARACTER_FORMAT,
                "version": cs.CHARACTER_VERSION,
                "created_at": "2020-01-01T00:00:00+0000",
                "updated_at": "2020-01-01T00:00:00+0000",
            },
            "name": "Narrator",
            "mode": cs.MODE_ONESHOT,
            "notes": "",
            "oneshot": {"clips": [], "default_clip_id": None},
            "rvc": {"model_path": None, "index_path": None, "transpose": 0},
        }
        char_file = cs._character_file(self.root, slug)
        with open(char_file, "w", encoding="utf-8") as f:
            json.dump(old_document, f)

        document = cs.load_character(self.root, slug)
        self.assertIsNotNone(document)
        # The loader does NOT normalize (house convention: consumers read the
        # block defensively, same as `rvc`). What this test pins is the real
        # guarantee: training an old document must not crash on the missing
        # key, and must save the default strength.
        self.assertNotIn("lora", document)
        with mock.patch.object(lora_engine, "engine_supports_lora",
                               return_value=True):
            with mock.patch.object(character_dataset, "export_dataset",
                                   return_value={"total_seconds": 5.0}):
                adapter = Path(self.root) / "old.safetensors"
                adapter.write_bytes(b"fake")
                with mock.patch.object(lora_engine, "train_lora_full",
                                       return_value={"adapter_path": str(adapter),
                                                     "status_path": "s",
                                                     "samples_dir": "d"}):
                    character_training.train_character_lora(self.root, slug)
        saved = cs.load_character(self.root, slug)
        self.assertEqual(saved["lora"]["strength"], 1.0)
        self.assertEqual(saved["lora"]["adapter_path"], str(adapter))


class TrainCharacterLoraTests(_TempLibrary):
    """train_character_lora orchestration."""

    def setUp(self):
        super().setUp()
        self.slug = cs.create_character(self.root, "Narrator", cs.MODE_ONESHOT)

    def a_clip(self, name="sample.wav", payload=b"RIFFfake"):
        path = Path(self.root) / "_incoming" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        return str(path)

    def test_raises_when_character_not_found(self):
        with self.assertRaises(character_dataset.DatasetExportError) as cm:
            character_training.train_character_lora(self.root, "nobody")
        self.assertIn("No character named", str(cm.exception))

    def test_raises_when_engine_does_not_support_lora(self):
        cs.add_clip(self.root, self.slug, self.a_clip(), {"duration_s": 10.0})
        with mock.patch.object(lora_engine, "engine_supports_lora", return_value=False):
            with self.assertRaises(character_dataset.DatasetExportError) as cm:
                character_training.train_character_lora(self.root, self.slug)
        self.assertIn("INDEXTTS25_ROOT", str(cm.exception))
        # Verify document is untouched
        document = cs.load_character(self.root, self.slug)
        self.assertIsNone(document["lora"]["adapter_path"])

    def test_happy_path_saves_adapter_and_metadata(self):
        cs.add_clip(self.root, self.slug, self.a_clip(), {"duration_s": 10.0})

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as adapter_file:
            adapter_path = adapter_file.name
            adapter_file.write(b"fake adapter weights")

        self.addCleanup(lambda: os.unlink(adapter_path) if os.path.exists(adapter_path) else None)

        with tempfile.TemporaryDirectory() as samples_dir:
            manifest = {"total_seconds": 10.0}
            artifacts = {"adapter_path": adapter_path, "samples_dir": samples_dir}

            with mock.patch.object(character_dataset, "export_dataset", return_value=manifest), \
                 mock.patch.object(lora_engine, "engine_supports_lora", return_value=True), \
                 mock.patch.object(lora_engine, "train_lora_full", return_value=artifacts) as mock_train:
                result = character_training.train_character_lora(self.root, self.slug)

            self.assertEqual(result["manifest"], manifest)
            self.assertEqual(result["artifacts"], artifacts)

            # Verify train_lora_full was called with correct parameters
            mock_train.assert_called_once()
            call_kwargs = mock_train.call_args[1]
            self.assertEqual(call_kwargs["name"], f"voiceforge_{self.slug}")
            self.assertIn(self.slug, call_kwargs["work_root"])
            self.assertIn("lora_training", call_kwargs["work_root"])

            # Verify document was updated
            document = cs.load_character(self.root, self.slug)
            self.assertEqual(document["lora"]["adapter_path"], adapter_path)
            self.assertEqual(document["lora"]["trained_from_seconds"], 10.0)
            self.assertEqual(document["lora"]["samples_dir"], samples_dir)
            self.assertEqual(document["lora"]["strength"], 1.0)

    def test_preserves_strength_setting_across_training(self):
        cs.add_clip(self.root, self.slug, self.a_clip(), {"duration_s": 10.0})

        # Pre-set a custom strength
        document = cs.load_character(self.root, self.slug)
        document["lora"]["strength"] = 0.5
        cs.save_character(self.root, self.slug, document)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as adapter_file:
            adapter_path = adapter_file.name
            adapter_file.write(b"fake adapter weights")

        self.addCleanup(lambda: os.unlink(adapter_path) if os.path.exists(adapter_path) else None)

        with tempfile.TemporaryDirectory() as samples_dir:
            manifest = {"total_seconds": 10.0}
            artifacts = {"adapter_path": adapter_path, "samples_dir": samples_dir}

            with mock.patch.object(character_dataset, "export_dataset", return_value=manifest), \
                 mock.patch.object(lora_engine, "engine_supports_lora", return_value=True), \
                 mock.patch.object(lora_engine, "train_lora_full", return_value=artifacts):
                character_training.train_character_lora(self.root, self.slug)

            document = cs.load_character(self.root, self.slug)
            self.assertEqual(document["lora"]["strength"], 0.5)

    def test_work_root_defaults_to_character_folder(self):
        cs.add_clip(self.root, self.slug, self.a_clip(), {"duration_s": 10.0})

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as adapter_file:
            adapter_path = adapter_file.name
            adapter_file.write(b"fake adapter weights")

        self.addCleanup(lambda: os.unlink(adapter_path) if os.path.exists(adapter_path) else None)

        with tempfile.TemporaryDirectory() as samples_dir:
            manifest = {"total_seconds": 10.0}
            artifacts = {"adapter_path": adapter_path, "samples_dir": samples_dir}

            with mock.patch.object(character_dataset, "export_dataset", return_value=manifest), \
                 mock.patch.object(lora_engine, "engine_supports_lora", return_value=True), \
                 mock.patch.object(lora_engine, "train_lora_full", return_value=artifacts) as mock_train:
                character_training.train_character_lora(self.root, self.slug)

            call_kwargs = mock_train.call_args[1]
            expected_work_root = os.path.join(self.root, self.slug, "lora_training")
            self.assertEqual(call_kwargs["work_root"], expected_work_root)

    def test_failure_leaves_document_unchanged(self):
        cs.add_clip(self.root, self.slug, self.a_clip(), {"duration_s": 10.0})

        # Pre-set adapter and strength
        document = cs.load_character(self.root, self.slug)
        document["lora"]["adapter_path"] = "/existing/adapter.pth"
        document["lora"]["strength"] = 0.7
        cs.save_character(self.root, self.slug, document)

        error = lora_engine.LoraError("Training failed")

        with mock.patch.object(character_dataset, "export_dataset", return_value={"total_seconds": 10.0}), \
             mock.patch.object(lora_engine, "engine_supports_lora", return_value=True), \
             mock.patch.object(lora_engine, "train_lora_full", side_effect=error):
            with self.assertRaises(lora_engine.LoraError):
                character_training.train_character_lora(self.root, self.slug)

        # Document should be unchanged
        document = cs.load_character(self.root, self.slug)
        self.assertEqual(document["lora"]["adapter_path"], "/existing/adapter.pth")
        self.assertEqual(document["lora"]["strength"], 0.7)
        self.assertNotIn("trained_from_seconds", document["lora"])


if __name__ == "__main__":
    unittest.main()
