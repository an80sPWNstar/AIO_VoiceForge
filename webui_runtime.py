"""Command-line configuration and the device the engine should load on.

Both webui.py and the generation pipeline need these, so they live below
both to keep the imports acyclic. Importing this module parses argv --
that is the same moment it happened when this code sat at the top of
webui.py, since webui.py imports this first.

Split out of webui.py.
"""

import os
import threading

import engine_paths

import argparse
parser = argparse.ArgumentParser(
    description="IndexTTS WebUI",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose mode")
parser.add_argument("--port", type=int, default=7860, help="Port to run the web UI on")
parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the web UI on")
parser.add_argument("--model_dir", type=str, default=engine_paths.ENGINE_CHECKPOINTS,
                    help="IndexTTS-2.5 checkpoints directory")
parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 for inference if available")
parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed to accelerate if available")
parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel for inference if available")
parser.add_argument("--gui_seg_tokens", type=int, default=80, help="GUI: Max tokens per generation segment")
parser.add_argument("--share", action="store_true", default=False, help="Enable Gradio live sharing to create a public link")
cmd_args = parser.parse_args()

DEVICE_CPU = "cpu"
DEVICE_CPU_LABEL = "CPU (slow, always available)"
DEVICE_AUTO = "auto"
DEVICE_AUTO_LABEL = "Auto (first CUDA device)"


class DeviceSelection:
    """Holds the device the next model load should use.

    The model loads in the generation subprocess, so the choice cannot be
    passed down the call stack; it is read back out here when the request is
    built. Guarded by a lock because the UI thread writes it while a generation
    thread may be reading it.
    """

    def __init__(self, initial=DEVICE_AUTO):
        self._value = initial
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._value

    def set(self, value):
        """Store `value`; returns True when it actually changed."""
        with self._lock:
            if value == self._value:
                return False
            self._value = value
            return True


selected_device = DeviceSelection()


class LoraSelection:
    """Holds the LoRA adapter the next generation should speak through.

    Same shape as DeviceSelection and for the same reason: the adapter is
    applied in the generation subprocess, so the choice cannot travel down
    gen_single's call stack; the request builder reads it back out here.
    An empty path means "no adapter" and actively REMOVES one a previous
    request applied — the engine worker is persistent.
    """

    def __init__(self):
        self._path = ""
        self._strength = 1.0
        self._lock = threading.Lock()

    def get(self):
        with self._lock:
            return self._path, self._strength

    def set(self, path, strength):
        """Store the pair; returns True when either actually changed."""
        path = path or ""
        strength = float(strength)
        with self._lock:
            if (path, strength) == (self._path, self._strength):
                return False
            self._path, self._strength = path, strength
            return True


selected_lora = LoraSelection()


def _build_tts_runtime_options():
    # "auto" means leave device unset so IndexTTS2 picks it the way it always
    # has; anything else is an explicit user choice from the device dropdown.
    device = selected_device.get()
    return {
        "model_dir": cmd_args.model_dir,
        "cfg_path": os.path.join(cmd_args.model_dir, "config.yaml"),
        "use_fp16": bool(cmd_args.fp16),
        "use_deepspeed": bool(cmd_args.deepspeed),
        "use_cuda_kernel": bool(cmd_args.cuda_kernel),
        "device": None if device == DEVICE_AUTO else device,
    }


# degrades pronunciation rather than failing.
ENGINE_LANGUAGES = [
    ("English", "EN"),
    ("Chinese", "ZH"),
    ("Japanese", "JA"),
    ("Spanish", "ES"),
    ("Arabic", "AR"),
]
DEFAULT_ENGINE_LANGUAGE = "EN"


EMO_CHOICES_ALL = ["Same as speaker voice",
                "Use emotion reference audio",
                "Use emotion vector control",
                "Use emotion text description"]
