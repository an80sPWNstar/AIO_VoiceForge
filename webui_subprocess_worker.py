from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
INDEXTTS_DIR = os.path.join(CURRENT_DIR, "indextts")
if INDEXTTS_DIR not in sys.path:
    sys.path.append(INDEXTTS_DIR)

from webui_generation_runner import create_tts, run_generation_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IndexTTS WebUI subprocess worker")
    parser.add_argument("--request-file", required=True, help="Path to the generation request JSON file")
    parser.add_argument("--result-file", required=True, help="Path to the result JSON file")
    parser.add_argument(
        "--progress-file",
        default=None,
        help="Optional path to append progress events to, one JSON object per line. "
             "A gradio Progress object cannot cross a process boundary, so the parent "
             "tails this file instead.",
    )
    return parser.parse_args()


def make_progress_writer(path: str):
    """Return a callable matching IndexTTS2's gr_progress contract.

    IndexTTS2 calls it as gr_progress(value, desc=...). Each call appends one
    JSON line. A failure to write is reported but never propagated: progress is
    cosmetic and must not be able to kill a generation that is otherwise fine.
    """

    def write_progress(value, desc: str = "") -> None:
        try:
            # newline="" stops Windows translating \n into \r\n, so the byte
            # offsets the parent uses to tail this file stay exact.
            with open(path, "a", encoding="utf-8", newline="") as handle:
                handle.write(json.dumps({
                    "value": float(value or 0.0),
                    "desc": str(desc or ""),
                }) + "\n")
                handle.flush()
        except (OSError, TypeError, ValueError) as exc:
            print(f"progress write failed ({type(exc).__name__}: {exc})", file=sys.stderr)

    return write_progress


def main() -> int:
    args = parse_args()

    with open(args.request_file, "r", encoding="utf-8") as handle:
        request = json.load(handle)

    progress_callback = make_progress_writer(args.progress_file) if args.progress_file else None

    try:
        tts = create_tts(request["runtime"])
        result = run_generation_request(request, tts, progress_callback=progress_callback)
        payload = {"status": "ok", **result}
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        payload = {"status": "error", "error": str(exc)}
        exit_code = 1

    os.makedirs(os.path.dirname(args.result_file), exist_ok=True)
    with open(args.result_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
