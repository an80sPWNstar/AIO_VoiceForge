"""Train an RVC model for a character and record it where the app looks.

The one call that connects the three pieces built underneath it: the clip
collection (character_store), the dataset layout trainers preprocess from
(character_dataset), and the Applio subprocess chain (rvc_engine). When it
finishes, the character's rvc block points at real weights, which is what
the mode field has promised since the store was written.

Blocks for the whole run -- minutes to hours depending on the dataset and
epochs -- so UI callers run it on a worker thread, same as generation.
"""

import os
import tempfile
from typing import Any, Dict, Optional

import character_dataset
import character_store
import lora_engine
import rvc_engine


def train_character(root: str, slug: str, *,
                    total_epochs: int = 200, batch_size: int = 8,
                    gpu: str = "0", sample_rate: int = 40000,
                    dataset_dir: Optional[str] = None,
                    applio_root: Optional[str] = None,
                    now: Optional[float] = None) -> Dict[str, Any]:
    """Export the character's clips, train, and store the model paths.

    Returns {"manifest": ..., "artifacts": ...}. The character document is
    updated ONLY after training verified its artifacts on disk: a character
    must never point at weights that do not exist, because generation would
    then fail at synthesis time instead of at training time, in front of the
    wrong error message.
    """
    document = character_store.load_character(root, slug)
    if document is None:
        raise character_dataset.DatasetExportError(
            f"No character named {slug!r} in the library.")

    with tempfile.TemporaryDirectory(prefix=f"rvc_dataset_{slug}_") as scratch:
        target = dataset_dir or scratch
        manifest = character_dataset.export_dataset(root, slug, target, now=now)
        artifacts = rvc_engine.train(
            f"voiceforge_{slug}", target,
            sample_rate=sample_rate, total_epochs=total_epochs,
            batch_size=batch_size, gpu=gpu, applio_root=applio_root,
        )

    document = character_store.load_character(root, slug)
    document["rvc"] = {
        "model_path": artifacts["model_path"],
        "index_path": artifacts["index_path"],
        "transpose": (document.get("rvc") or {}).get("transpose", 0),
    }
    document["rvc"]["trained_from_seconds"] = manifest["total_seconds"]
    character_store.save_character(root, slug, document, now=now)
    return {"manifest": manifest, "artifacts": artifacts}


def train_character_lora(root: str, slug: str, *,
                         work_root: Optional[str] = None,
                         progress_callback=None,
                         now: Optional[float] = None) -> Dict[str, Any]:
    """Export the character's clips, run the V5 LoRA pipeline, store the
    adapter path. Returns {"manifest": ..., "artifacts": ...}."""
    document = character_store.load_character(root, slug)
    if document is None:
        raise character_dataset.DatasetExportError(
            f"No character named {slug!r} in the library.")

    if not lora_engine.engine_supports_lora():
        raise character_dataset.DatasetExportError(
            "The installed engine has no LoRA training pipeline. "
            "Point INDEXTTS25_ROOT at the V5 install.")

    if work_root is None:
        work_root = os.path.join(root, slug, "lora_training")

    clips_export_dir = os.path.join(work_root, "clips")
    manifest = character_dataset.export_dataset(root, slug, clips_export_dir, now=now)
    artifacts = lora_engine.train_lora_full(
        clips_dir=clips_export_dir,
        name=f"voiceforge_{slug}",
        work_root=work_root,
        progress_callback=progress_callback,
    )

    document = character_store.load_character(root, slug)
    document["lora"] = {
        "adapter_path": artifacts["adapter_path"],
        "strength": (document.get("lora") or {}).get("strength", 1.0),
        "trained_from_seconds": manifest["total_seconds"],
        "samples_dir": artifacts["samples_dir"],
    }
    character_store.save_character(root, slug, document, now=now)
    return {"manifest": manifest, "artifacts": artifacts}
