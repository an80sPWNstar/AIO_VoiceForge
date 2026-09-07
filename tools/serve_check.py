"""Serve the real page and confirm the controls reach the browser.

`tools/gate.py` runs the code. This runs the *page*, which is a different
claim. `import webui` only proves the Blocks object builds; a component
gradio cannot render, and a Dropdown holding a value absent from its choices,
both survive construction and surface when the page is served. The character
panel also shipped once with 63 green handler tests and its main action
undiscoverable, which no test of any kind was ever going to catch -- this at
least proves every control is present and reachable.

So: launch headlessly on a spare port, fetch the config gradio hands the
browser, and look for each control by its label.

  E:\\vs_code_projects\\venv_voiceforge_host\\Scripts\\python.exe tools\\serve_check.py

inbrowser=False deliberately -- this runs unattended and must not open a tab.
Run it from a directory whose layout the app expects (see the note about the
audio-cleanup sidecar in the project notes); from the checkout itself the
page still serves, which is all this checks.
"""

import json
import os
import sys
import urllib.request

# Running from tools/ puts tools/ on the path, not the app. webui and every
# module it imports live one level up.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PORT = 7899

# One label per control that has to exist. Add to this when adding a panel:
# it is cheap, and it is the only automated check that a control was actually
# wired into the page rather than merely defined.
WANTED = [
    # Reference clip panel (Download & Extract Audio tab)
    "Find a reference clip",
    "Scan for Reference Clips",
    "Use Segment as Reference Voice",
    "Save Segment to Selected Voice",
    "Segment preview",
    "Best candidates first",
    "Advanced splitting settings",
    "Ignore anything shorter than (seconds)",
    "Label for the saved clip (optional)",
    # Character library (Audio Generation tab)
    "Save Loaded Voice As",
    "Add Clip to Selected Voice",
    # Train Voice tab. "Train RVC Model" used to be listed here as a character
    # panel button; 5c3d203 moved training off that panel and this list was not
    # updated with it, so the check sat red from then until the ingestion split
    # noticed. A label removed from the app has to be removed from here too --
    # a permanently-red check is one nobody reads.
    "Start training",
    "Request stop",
    "Use this checkpoint",
    "Confirm training",
    # The two earlier stages of the ingestion pipeline
    "Clean Up Audio",
    "Send Cleaned to Reference Voice",
]


def main():
    sys.argv = ["webui.py", "--port", str(PORT)]
    import webui

    webui.demo.queue(20)
    webui.demo.launch(
        server_name="127.0.0.1",
        server_port=PORT,
        share=False,
        inbrowser=False,
        prevent_thread_lock=True,
        theme=webui.theme,
    )
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/", timeout=30
        ) as page:
            html = page.read().decode("utf-8", "replace")
        print(f"  served page: {len(html)} bytes, HTTP OK")

        # Over HTTP rather than demo.get_config_file(): the in-process object
        # holds live callables and is not JSON-serialisable, and what the
        # browser is handed is what actually matters here.
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}/config", timeout=30
        ) as reply:
            config = json.loads(reply.read().decode("utf-8"))
        raw = json.dumps(config)
        print(f"  config: {len(raw)} bytes, "
              f"{len(config.get('components', []))} components, "
              f"{len(config.get('dependencies', []))} event bindings")

        missing = [label for label in WANTED if label not in raw]
        for label in WANTED:
            print(f"    {'OK  ' if label in raw else 'MISS'}  {label}")

        for component in config.get("components", []):
            headers = component.get("props", {}).get("headers")
            if headers:
                print(f"  table {component['props'].get('key')}: {headers}")

        print("\n  SERVE CHECK:", "PASS" if not missing else f"*** MISSING {missing} ***")
        return 0 if not missing else 1
    finally:
        webui.demo.close()


if __name__ == "__main__":
    sys.exit(main())
