"""The Train Voice tab: one place to take a character from clips to a trained voice.

The character panel's two train buttons started this; the tab is where training
actually becomes observable. It answers, in order: is this voice ready to train,
which lane should the run take (V5 LoRA or Applio RVC), how far along is the run
that is going, and — once checkpoints exist — which epoch should the voice speak
through. Everything after "start" is read off files the trainers already write:
status.json for live progress, samples/epoch_*.wav for per-epoch listening, and
analysis/checkpoint_eval.json for the measured recommendation. No new engine
code; this module only drives `character_training` and reads state directories.

Unlike its siblings, this module also BUILDS its tab: webui.py calls
`build_training_tab()` inside a `with gr.Tab(...)` block. New-tab layout does
not carry the re-indent-every-tab cost that kept older layout inline in
webui.py, and the house review already flags webui.py for splitting — a new
tab should not add to that file.

Handlers stay top-level plain functions so the tests can call them without a
Blocks context, exactly like webui_character_handlers.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

import character_store as store
import webui_character_handlers as characters
from webui_progress import render_progress_bar

# Lane identifiers double as the Radio's stored values.
LANE_LORA = "lora"
LANE_RVC = "rvc"

# Roughly what each trainer wants before a run is worth starting. Advisory
# only, same policy as the character panel's readiness note: telling the user
# where they are, never blocking. The LoRA figure matches RVC's ten minutes —
# the validated shakeout ran on ~18 minutes and V5 prep re-segments whatever
# it is given, so more consented audio is simply better.
LORA_TARGET_SECONDS = 600.0

TRAIN_PROGRESS_IDLE = render_progress_bar(0.0, "Idle")

# How long the drain loop waits on the progress queue before re-checking the
# worker thread. Long enough not to spin, short enough that a finished run is
# reported promptly.
PROGRESS_POLL_SECONDS = 1.0

# A finished worker gets this long to actually exit before we stop waiting and
# report anyway (mirrors SCAN_THREAD_JOIN_SECONDS in the segmentation module).
TRAIN_THREAD_JOIN_SECONDS = 30

# Dropdown placeholder: gradio rejects "" when it is not among the choices;
# None is the accepted value for "nothing selected" (see NO_SELECTION in
# webui_character_handlers).
NO_CHECKPOINT = None


def lane_choices() -> List[Tuple[str, str]]:
    """Choices for the lane Radio: (visible label, lane constant)."""
    return [("TTS LoRA — the voice itself (V5)", LANE_LORA),
            ("RVC — post-conversion (Applio)", LANE_RVC)]


def work_root(root: str, slug: str) -> str:
    """<root>/<slug>/lora_training — MUST match character_training's default."""
    return os.path.join(root, slug, "lora_training")


def adapter_dir(root: str, slug: str) -> str:
    """Where a character's LoRA run leaves checkpoints and state.

    <work_root>/adapters/voiceforge_<slug> — the layout train_lora_full and
    run_training produce (state dir IS the adapter dir).
    """
    return os.path.join(root, slug, "lora_training", "adapters", f"voiceforge_{slug}")


def dataset_state_dir(root: str, slug: str) -> str:
    """Where the prep stage writes its status.json / stop.flag.

    <work_root>/dataset/voiceforge_<slug>.
    """
    return os.path.join(root, slug, "lora_training", "dataset", f"voiceforge_{slug}")


def readiness_report(slug: str, root: Optional[str] = None) -> str:
    """Markdown block: is this voice worth pointing a GPU at, per lane.

    Always shows both LoRA and RVC status regardless of which lane is selected.
    This is a oneshot voice's readiness summary, not a mode-specific one.
    """
    if not slug:
        return "Select a voice first."

    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    doc = store.load_character(root, slug)
    if doc is None:
        return "Select a voice first."

    clip_count = len(doc.get("oneshot", {}).get("clips", []))
    total_seconds = store.total_clip_seconds(doc)

    lines = []
    lines.append(f"{clip_count} clips, {total_seconds:.0f}s total.")

    # Per-lane status — always show both lanes
    import lora_engine
    if lora_engine.engine_supports_lora():
        lines.append("LoRA: available.")
    else:
        lines.append("LoRA: engine has no LoRA training pipeline.")

    import rvc_paths
    missing = rvc_paths.missing_applio_parts()
    if not missing:
        lines.append("RVC: available.")
    else:
        lines.append(f"RVC: missing {', '.join(missing)}.")

    # Per-lane target check
    if total_seconds < LORA_TARGET_SECONDS:
        lines.append(f"Dataset is thin for LoRA ({total_seconds:.0f}s < {LORA_TARGET_SECONDS:.0f}s).")
    if doc.get("lora", {}).get("adapter_path") and os.path.exists(doc["lora"]["adapter_path"]):
        lines.append("already trained (LoRA).")

    rvc_target = characters.RVC_TARGET_SECONDS
    if total_seconds < rvc_target:
        lines.append(f"Dataset is thin for RVC ({total_seconds:.0f}s < {rvc_target:.0f}s).")
    if doc.get("rvc", {}).get("model_path") and os.path.exists(doc["rvc"]["model_path"]):
        lines.append("already trained (RVC).")

    # Unmeasured clips
    clips = doc.get("oneshot", {}).get("clips", [])
    unmeasured = sum(1 for c in clips if c.get("duration_s") is None)
    if unmeasured:
        lines.append(f"{unmeasured} clips have no measured duration and count as 0s.")

    return "\n".join(lines)


def on_lane_change(lane: str):
    """The lane Radio moved. Returns gr.update pair (stop_btn, rvc_epochs)."""
    if lane == LANE_LORA:
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)


def list_checkpoints(slug: str, root: Optional[str] = None) -> List[Dict[str, Any]]:
    """Every checkpoint a LoRA run left behind, measured where possible."""
    if not slug:
        return []

    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    ad = adapter_dir(root, slug)

    if not os.path.isdir(ad):
        return []

    # Load eval report if available
    eval_rows = []
    recommended_path = None
    eval_path = os.path.join(ad, "analysis", "checkpoint_eval.json")
    try:
        with open(eval_path, "r") as f:
            report = json.load(f)
        eval_rows = report.get("rows", [])
        recommended_path = report.get("recommended_checkpoint")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        eval_rows = []
        recommended_path = None

    # Build a lookup from eval rows by path
    eval_by_path: Dict[str, Dict[str, Any]] = {}
    for row in eval_rows:
        try:
            p = row["path"]
            eval_by_path[p] = row
        except (KeyError, TypeError):
            pass

    def _find_eval(abs_path: str, basename: str, kind: str) -> Optional[Dict[str, Any]]:
        # Try absolute path match
        if abs_path in eval_by_path:
            return eval_by_path[abs_path]
        # Try basename + kind match
        for row in eval_rows:
            try:
                if row.get("kind") == kind and os.path.basename(row.get("path", "")) == basename:
                    return row
            except (KeyError, TypeError):
                pass
        return None

    entries: List[Dict[str, Any]] = []

    # Scan top-level .safetensors
    for fname in glob.glob(os.path.join(ad, "*.safetensors")):
        stem = os.path.splitext(os.path.basename(fname))[0]
        abs_path = os.path.abspath(fname)

        kind = "final"
        epoch = None
        m = re.search(r"_epoch_(\d+)$", stem)
        if m:
            kind = "epoch"
            epoch = int(m.group(1))
        else:
            m = re.search(r"_step_(\d+)$", stem)
            if m:
                kind = "step"
            elif stem.endswith("_interrupted"):
                kind = "interrupted"

        ev = _find_eval(abs_path, os.path.basename(fname), kind)
        val_loss = None
        phase = None
        if ev:
            val_loss = ev.get("val_loss")
            phase = ev.get("phase")
            if epoch is None and ev.get("epoch") is not None:
                epoch = ev["epoch"]

        recommended = (abs_path == recommended_path) if recommended_path else False

        entries.append({
            "label": "",
            "path": abs_path,
            "kind": kind,
            "epoch": epoch,
            "val_loss": val_loss,
            "phase": phase,
            "recommended": recommended,
        })

    # Scan best/ subdirectory
    best_dir = os.path.join(ad, "best")
    if os.path.isdir(best_dir):
        for fname in glob.glob(os.path.join(best_dir, "*.safetensors")):
            stem = os.path.splitext(os.path.basename(fname))[0]
            abs_path = os.path.abspath(fname)

            kind = "best"
            epoch = None
            ev = _find_eval(abs_path, os.path.basename(fname), kind)
            val_loss = None
            phase = None
            if ev:
                val_loss = ev.get("val_loss")
                phase = ev.get("phase")
                epoch = ev.get("epoch")

            recommended = (abs_path == recommended_path) if recommended_path else False

            entries.append({
                "label": "",
                "path": abs_path,
                "kind": kind,
                "epoch": epoch,
                "val_loss": val_loss,
                "phase": phase,
                "recommended": recommended,
            })

    # Sort: best first, then epochs ascending, then steps, interrupted, final
    def _sort_key(e):
        kind_order = {"best": 0, "epoch": 1, "step": 2, "interrupted": 3, "final": 4}
        ko = kind_order.get(e["kind"], 5)
        ep = e["epoch"] if e["epoch"] is not None else 999999
        return (ko, ep)

    entries.sort(key=_sort_key)
    return entries


def checkpoint_choices(slug: str, root: Optional[str] = None) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """(dropdown choices, initially selected path) for a voice's checkpoints."""
    cps = list_checkpoints(slug, root)
    if not cps:
        return [], NO_CHECKPOINT

    choices: List[Tuple[str, str]] = []
    selected = None

    for cp in cps:
        parts = []
        kind = cp["kind"]

        if kind == "best":
            parts.append("best")
        elif kind == "epoch":
            parts.append(f"epoch {cp['epoch']}")
        elif kind == "step":
            parts.append(f"step {cp['epoch']}")
        elif kind == "interrupted":
            parts.append("interrupted")
        else:
            parts.append("final")

        if cp["val_loss"] is not None:
            parts.append(f"val {cp['val_loss']:.2f}")

        if cp["recommended"]:
            parts.append("recommended")

        label = " — ".join(parts)
        choices.append((label, cp["path"]))

        if cp["recommended"]:
            selected = cp["path"]

    if selected is None:
        # Fall back to best kind, then first
        for cp in cps:
            if cp["kind"] == "best":
                selected = cp["path"]
                break
        if selected is None and cps:
            selected = cps[0]["path"]

    return choices, selected


def sample_for_checkpoint(slug: str, checkpoint_path: Optional[str],
                          root: Optional[str] = None) -> Optional[str]:
    """The training-time voice sample closest to a checkpoint, if one exists."""
    if not slug or checkpoint_path is None:
        return None

    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    ad = adapter_dir(root, slug)
    samples_dir = os.path.join(ad, "samples")
    if not os.path.isdir(samples_dir):
        return None

    # Determine the target epoch
    target_epoch = None

    # Try to get epoch from the checkpoint filename
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    m = re.search(r"_epoch_(\d+)$", stem)
    if m:
        target_epoch = int(m.group(1))

    # If not found in filename, try eval report
    if target_epoch is None:
        eval_path = os.path.join(ad, "analysis", "checkpoint_eval.json")
        try:
            with open(eval_path, "r") as f:
                report = json.load(f)
            for row in report.get("rows", []):
                try:
                    if os.path.normcase(os.path.abspath(row["path"])) == os.path.normcase(os.path.abspath(checkpoint_path)):
                        target_epoch = row.get("epoch")
                        break
                except (KeyError, TypeError):
                    pass
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    # If still no epoch, it's "final" — find highest-numbered sample
    sample_files = glob.glob(os.path.join(samples_dir, "epoch_*.wav"))
    if not sample_files:
        return None

    sample_epochs: List[int] = []
    for sf in sample_files:
        sf_stem = os.path.splitext(os.path.basename(sf))[0]
        m2 = re.search(r"epoch_(\d+)", sf_stem)
        if m2:
            sample_epochs.append(int(m2.group(1)))

    if not sample_epochs:
        return None

    if target_epoch is None:
        # Final: pick highest
        target_epoch = max(sample_epochs)

    # Find matching sample
    for sf in sample_files:
        sf_stem = os.path.splitext(os.path.basename(sf))[0]
        m2 = re.search(r"epoch_(\d+)", sf_stem)
        if m2 and int(m2.group(1)) == target_epoch:
            return os.path.abspath(sf)

    return None


def checkpoint_detail(slug: str, checkpoint_path: Optional[str],
                      root: Optional[str] = None) -> str:
    """One short markdown block about the selected checkpoint."""
    if not slug or not checkpoint_path:
        return "Select a checkpoint."

    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    ad = adapter_dir(root, slug)
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]

    kind = "final"
    epoch = None
    m = re.search(r"_epoch_(\d+)$", stem)
    if m:
        kind = "epoch"
        epoch = int(m.group(1))
    else:
        m = re.search(r"_step_(\d+)$", stem)
        if m:
            kind = "step"
        elif stem.endswith("_interrupted"):
            kind = "interrupted"

    # Check if in best dir
    if os.path.dirname(os.path.abspath(checkpoint_path)).endswith("best"):
        kind = "best"

    # Try eval report for val_loss and phase
    val_loss = None
    phase = None
    eval_path = os.path.join(ad, "analysis", "checkpoint_eval.json")
    try:
        with open(eval_path, "r") as f:
            report = json.load(f)
        for row in report.get("rows", []):
            try:
                if os.path.normcase(os.path.abspath(row["path"])) == os.path.normcase(os.path.abspath(checkpoint_path)):
                    val_loss = row.get("val_loss")
                    phase = row.get("phase")
                    break
            except (KeyError, TypeError):
                pass
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass

    # Check if this is the current adapter
    doc = store.load_character(root, slug)
    is_current = False
    if doc and doc.get("lora", {}).get("adapter_path"):
        if os.path.normcase(os.path.abspath(doc["lora"]["adapter_path"])) == os.path.normcase(os.path.abspath(checkpoint_path)):
            is_current = True

    parts = [f"**{kind}**"]
    if epoch is not None:
        parts.append(f"epoch {epoch}")
    if val_loss is not None:
        parts.append(f"val_loss {val_loss:.2f}")
    if phase:
        parts.append(f"phase: {phase}")
    if is_current:
        parts.append("current adapter")
    parts.append(f"file: {os.path.basename(checkpoint_path)}")

    return " · ".join(parts)


def on_checkpoint_change(slug: str, checkpoint_path: Optional[str],
                         root: Optional[str] = None):
    """Checkpoint selection changed: return (gr.update for sample Audio, detail markdown)."""
    sample_path = sample_for_checkpoint(slug, checkpoint_path, root)
    detail = checkpoint_detail(slug, checkpoint_path, root)

    if sample_path is None:
        if checkpoint_path:
            detail = detail + "\n\nThere is no kept sample for this epoch."
        return gr.update(value=None), detail
    else:
        return gr.update(value=sample_path), detail


def use_checkpoint_ui(slug: str, checkpoint_path: Optional[str],
                      root: Optional[str] = None) -> Tuple[str, str]:
    """\"Use this checkpoint\" pressed. Returns (status line, readiness markdown)."""
    if not slug:
        return "Select a voice first.", ""

    if not checkpoint_path:
        return "Select a checkpoint.", ""

    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    if not os.path.exists(checkpoint_path):
        return f"Checkpoint does not exist: {os.path.basename(checkpoint_path)}.", ""

    doc = store.load_character(root, slug)
    if doc is None:
        return "Select a voice first.", ""

    # Preserve strength and other lora fields — mutate existing block
    lora_block = doc.get("lora", {})
    lora_block["adapter_path"] = checkpoint_path
    doc["lora"] = lora_block
    store.save_character(root, slug, doc)

    return (f"Using {os.path.basename(checkpoint_path)}. "
            "Speak with trained voice (LoRA) must be re-toggled."), readiness_report(slug, root)


def stop_training_ui(slug: str, root: Optional[str] = None) -> str:
    """Ask the running LoRA stage to stop, gracefully."""
    if not slug:
        return "Select a voice first."

    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    import lora_engine
    ds = dataset_state_dir(root, slug)
    ad = adapter_dir(root, slug)
    lora_engine.request_stop(ds)
    lora_engine.request_stop(ad)
    return "Stop requested."


def _format_live_status(state_dir: str, fallback: str) -> str:
    """Enrich a progress message from the stage's status.json, tolerantly."""
    status_path = os.path.join(state_dir, "status.json")
    try:
        with open(status_path, "r") as f:
            data = json.load(f)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return fallback

    parts = []

    epoch = data.get("epoch")
    total_epochs = data.get("total_epochs")
    if epoch is not None and total_epochs and total_epochs > 0:
        parts.append(f"epoch {epoch}/{total_epochs}")

    step = data.get("step")
    total_steps = data.get("total_steps")
    if step is not None and total_steps and total_steps > 0:
        parts.append(f"step {step}/{total_steps}")

    loss = data.get("loss")
    if isinstance(loss, (int, float)):
        parts.append(f"loss {loss:.2f}")

    eta_s = data.get("eta_s")
    if isinstance(eta_s, (int, float)) and eta_s > 0:
        eta_m = round(eta_s / 60)
        parts.append(f"eta {eta_m}m")

    if parts:
        return " · ".join(parts)
    return fallback


def start_training_ui(slug: str, lane: str, confirmed: bool,
                      rvc_epochs: Any, root: Optional[str] = None):
    """Train pressed. A generator; yields (progress_html, status_text,
    checkpoint dropdown gr.update).

    The lane determines which training path to take (LoRA or RVC). Oneshot
    voices can train either lane independently; this is not a voice mode choice.
    """
    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    if not slug:
        yield TRAIN_PROGRESS_IDLE, "Select a voice first.", gr.update()
        return

    if not confirmed:
        yield TRAIN_PROGRESS_IDLE, "Confirm training to proceed.", gr.update()
        return

    if lane == LANE_LORA:
        import lora_engine
        if not lora_engine.engine_supports_lora():
            yield TRAIN_PROGRESS_IDLE, "The installed engine has no LoRA training pipeline. Point INDEXTTS25_ROOT at the V5 install.", gr.update()
            return

    if lane == LANE_RVC:
        import rvc_paths
        missing = rvc_paths.missing_applio_parts()
        if missing:
            yield TRAIN_PROGRESS_IDLE, f"RVC is missing: {', '.join(missing)}.", gr.update()
            return

    if lane == LANE_LORA:
        import character_training
        import lora_engine
        import character_dataset

        q: queue.Queue = queue.Queue()
        final_result = [None]
        final_error = [None]

        def worker():
            try:
                result = character_training.train_character_lora(
                    root, slug,
                    work_root=work_root(root, slug),
                    progress_callback=q.put,
                )
                final_result[0] = result
                q.put(("done", result))
            except Exception as e:
                final_error[0] = e
                q.put(("error", e))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        last_fraction = None
        while thread.is_alive() or not q.empty():
            try:
                item = q.get(timeout=PROGRESS_POLL_SECONDS)
            except queue.Empty:
                continue

            if isinstance(item, tuple) and len(item) == 2 and item[0] in ("done", "error"):
                break

            # It's a LoraProgress
            if hasattr(item, 'stage'):
                stage = item.stage
                fraction = item.fraction
                if fraction is not None:
                    last_fraction = fraction
                # Build live status text from status.json
                if stage == "train":
                    state_dir = adapter_dir(root, slug)
                else:
                    state_dir = dataset_state_dir(root, slug)
                status_text = _format_live_status(state_dir, item.message)
                yield render_progress_bar(fraction if fraction is not None else (last_fraction or 0.0), stage), status_text, gr.update()

        thread.join(TRAIN_THREAD_JOIN_SECONDS)

        # Check for error from the queue item first
        if final_error[0] is not None:
            exc = final_error[0]
            if isinstance(exc, (character_dataset.DatasetExportError, lora_engine.LoraError)):
                yield TRAIN_PROGRESS_IDLE, f"LoRA training failed: {exc}", gr.update()
                return
            else:
                raise exc

        result = final_result[0]
        if result:
            artifacts = result.get("artifacts", {})
            adapter_path = artifacts.get("adapter_path", "")
            manifest = result.get("manifest", {})
            total_seconds = manifest.get("total_seconds", 0)
            ckpt_choices_result = checkpoint_choices(slug, root)
            yield (render_progress_bar(1.0, "Complete"),
                   f"Training complete. Adapter: {os.path.basename(adapter_path)}. "
                   f"Trained from {total_seconds:.0f}s of audio.",
                   gr.update(choices=ckpt_choices_result[0], value=ckpt_choices_result[1]))
        else:
            yield TRAIN_PROGRESS_IDLE, "Training finished with no result.", gr.update()

    elif lane == LANE_RVC:
        import character_training
        import rvc_engine
        import character_dataset

        yield TRAIN_PROGRESS_IDLE, ("RVC training started... preprocess, extract, "
                                    "train and index can take hours. No live "
                                    "progress exists for this lane."), gr.update()
        try:
            result = character_training.train_character(
                root, slug,
                total_epochs=int(rvc_epochs),
            )
            yield TRAIN_PROGRESS_IDLE, "RVC training complete.", gr.update()
        except (character_dataset.DatasetExportError, rvc_engine.RVCError) as e:
            yield TRAIN_PROGRESS_IDLE, str(e), gr.update()
        except Exception:
            raise


def refresh_training_panel(slug: str, root: Optional[str] = None):
    """Voice changed or Refresh pressed: update readiness and checkpoint panel."""
    readiness = readiness_report(slug, root)
    ckpt_choices_result = checkpoint_choices(slug, root)
    ckpt_update = gr.update(choices=ckpt_choices_result[0], value=ckpt_choices_result[1])
    sample_path = sample_for_checkpoint(slug, ckpt_choices_result[1], root)
    sample_update = gr.update(value=sample_path)
    detail = checkpoint_detail(slug, ckpt_choices_result[1], root)
    return readiness, ckpt_update, sample_update, detail


def initial_state(root: Optional[str] = None):
    """Layout-time defaults.

    Returns a four-tuple from characters.initial_state() along with checkpoint
    and sample details: (choices, first_slug, name, description, readiness,
    ckpt_choices, ckpt_value, sample_path, detail).
    """
    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    # Get character state from the character handler using its new signature
    initial_result = characters.initial_state(root)
    voice_choices, first_slug, name, description = initial_result

    readiness = readiness_report(first_slug, root)
    ckpt_result = checkpoint_choices(first_slug, root)
    ckpt_value = ckpt_result[1]
    sample_path = sample_for_checkpoint(first_slug, ckpt_value, root)
    detail = checkpoint_detail(first_slug, ckpt_value, root)

    return (voice_choices, first_slug, name, description, readiness,
            ckpt_result[0], ckpt_value, sample_path, detail)


def build_training_tab(root: Optional[str] = None) -> Dict[str, Any]:
    """Build the tab's components and wire its events.

    All voices in the library are oneshot voices; the UI no longer exposes
    a voice mode choice. The training lane (LoRA vs. RVC) is separate and
    controlled by the lane radio.
    """
    if root is None:
        root = characters.CHARACTER_LIBRARY_ROOT

    # Get initial state from character handler's new signature
    voice_choices, first_slug, name, description = characters.initial_state(root)
    voice_dd = gr.Dropdown(label="Voice", choices=voice_choices, value=first_slug)
    refresh_btn = gr.Button("Refresh", variant="secondary")

    readiness_md = gr.Markdown(readiness_report(first_slug, root))

    lane_radio = gr.Radio(lane_choices(), value=LANE_LORA, label="Training lane")
    rvc_epochs_num = gr.Number(label="RVC epochs", value=200, precision=0, visible=False)

    confirm_cb = gr.Checkbox(label="Confirm training", value=False,
                             info="Training requires a GPU and time. Check to proceed.")
    train_btn = gr.Button("Start training", variant="primary")
    stop_btn = gr.Button("Request stop", variant="stop", visible=True)

    progress_html = gr.HTML(value=TRAIN_PROGRESS_IDLE)
    status_tb = gr.Textbox(label="Status", interactive=False)

    ckpt_md = gr.Markdown("#### Checkpoints")
    ckpt_dd = gr.Dropdown(label="Checkpoint", choices=[], value=None)
    detail_md = gr.Markdown("Select a checkpoint.")
    sample_audio = gr.Audio(label="Epoch sample", interactive=False, value=None)
    use_btn = gr.Button("Use this checkpoint")

    with gr.Row():
        speak_cb = gr.Checkbox(
            label="Speak with trained voice (LoRA)",
            value=False,
            info="The TTS itself speaks the trained voice "
                 "— pace and style included.",
        )
        strength_sl = gr.Slider(
            label="Trained voice strength",
            minimum=0.0, maximum=2.0, step=0.05, value=1.0,
        )

    # Wire events
    voice_dd.change(refresh_training_panel, [voice_dd], [readiness_md, ckpt_dd, sample_audio, detail_md], queue=False)
    refresh_btn.click(refresh_training_panel, [voice_dd], [readiness_md, ckpt_dd, sample_audio, detail_md], queue=False)
    lane_radio.change(on_lane_change, [lane_radio], [stop_btn, rvc_epochs_num], queue=False)
    train_btn.click(start_training_ui, [voice_dd, lane_radio, confirm_cb, rvc_epochs_num], [progress_html, status_tb, ckpt_dd], queue=True)
    stop_btn.click(stop_training_ui, [voice_dd], [status_tb], queue=False)
    ckpt_dd.change(on_checkpoint_change, [voice_dd, ckpt_dd], [sample_audio, detail_md], queue=False)
    use_btn.click(use_checkpoint_ui, [voice_dd, ckpt_dd], [status_tb, readiness_md], queue=False)

    for _lora_event in (speak_cb.change, strength_sl.release, voice_dd.change):
        _lora_event(
            characters.on_lora_speak_change,
            [speak_cb, voice_dd, strength_sl],
            [status_tb],
            queue=False,
            show_progress="hidden",
        )

    return {
        "voice": voice_dd,
        "refresh": refresh_btn,
        "readiness": readiness_md,
        "lane": lane_radio,
        "rvc_epochs": rvc_epochs_num,
        "confirm": confirm_cb,
        "train": train_btn,
        "stop": stop_btn,
        "progress": progress_html,
        "status": status_tb,
        "ckpt": ckpt_dd,
        "detail": detail_md,
        "sample": sample_audio,
        "use": use_btn,
        "speak": speak_cb,
        "strength": strength_sl,
    }
