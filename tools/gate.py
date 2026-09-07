"""The gate: undefined names, then every test.

Two checks, because either alone has already let a bug through.

`ruff --select F821` first. A module can import cleanly and still raise
NameError the first time a function body runs, which is how four bugs shipped
out of the webui.py split -- `os` in _build_tts_runtime_options, `re` and
`html` in the preview, `datetime` in the preset store. `ast.parse` passed,
`import webui` passed, and 179 tests passed, because none of them executed
those bodies. It keeps happening: a fade helper added in September used
`np.linspace` in a module that never imported numpy.

Then the suite. Test modules are DISCOVERED rather than listed, because the
other half of that failure was extracted modules with no test file at all --
a hand-maintained list cannot notice a file nobody wrote.

Run it with the install's interpreter. The checkout has no venv of its own
and gradio, librosa and torch live in the install:

  E:\\vs_code_projects\\venv_voiceforge_host\\Scripts\\python.exe tools\\gate.py

Exit status is 0 only when nothing failed, nothing errored, no module failed
to import, and the test count has not dropped below TEST_COUNT_FLOOR.
"""

import glob
import importlib
import io
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")

# The suite has only ever grown. A drop means tests stopped being collected --
# usually a renamed file or an import that quietly died -- which is invisible
# in a run that otherwise reports all green. Raise it when the suite grows.
TEST_COUNT_FLOOR = 559

# indextts/ is the vendored upstream engine and archive/ is kept for
# reference; neither is ours to fix, and F821 fires in both.
LINT_EXCLUDE = "indextts,archive,tools/i18n"

# Modules that are harnesses rather than unittest suites: they talk to the
# real engine and are run by hand.
NOT_A_SUITE = {"regression_test", "padding_test"}


def discovered_modules():
    """Every tests/test_*.py, by module name, in a stable order."""
    found = []
    for path in sorted(glob.glob(os.path.join(TESTS, "test_*.py"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in NOT_A_SUITE:
            found.append(name)
    return found


def undefined_names():
    """Names used but never defined or imported. See the module docstring."""
    lint = subprocess.run(
        ["ruff", "check", "--select", "F821", "--output-format", "concise",
         "--exclude", LINT_EXCLUDE, "."],
        cwd=REPO, capture_output=True, text=True,
    )
    if lint.returncode not in (0, 1):
        # ruff missing or broken is a failed gate, not a clean one: a check
        # that cannot run must never read as a check that passed.
        print(f"  ruff could not run (exit {lint.returncode}): "
              f"{lint.stderr.strip()[:200]}")
        return None
    return [line for line in lint.stdout.splitlines() if "F821" in line]


def main():
    sys.path.insert(0, REPO)
    sys.path.insert(0, TESTS)
    os.chdir(REPO)
    # webui_runtime parses argv at import time, so anything left on it from
    # this script's own invocation would reach argparse.
    sys.argv = ["webui.py"]

    undefined = undefined_names()
    if undefined is None:
        print("\n  GATE: *** FAIL *** (ruff did not run)")
        return 1
    print(f"  undefined names (F821): {len(undefined)}")
    for line in undefined:
        print("     ", line)

    modules = discovered_modules()
    print(f"  discovered {len(modules)} test modules")

    totals = {"ran": 0, "fail": 0, "err": 0}
    broken = []
    for name in modules:
        try:
            module = importlib.import_module(name)
        except BaseException as exc:            # noqa: BLE001 - report, never hide
            print(f"  {name:30s} IMPORT FAILED: {type(exc).__name__}: {exc}")
            broken.append(name)
            continue
        result = unittest.TextTestRunner(
            verbosity=0, stream=io.StringIO()
        ).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        totals["ran"] += result.testsRun
        totals["fail"] += len(result.failures)
        totals["err"] += len(result.errors)
        print(f"  {name:30s} ran={result.testsRun:3d} "
              f"fail={len(result.failures)} err={len(result.errors)}")
        for case, _ in result.failures + result.errors:
            broken.append(f"{name}:{case}")

    print(f"\n  TOTAL ran={totals['ran']} fail={totals['fail']} err={totals['err']}")
    print(f"  FLOOR {TEST_COUNT_FLOOR}")

    shortfall = totals["ran"] < TEST_COUNT_FLOOR
    if shortfall:
        print(f"  *** {TEST_COUNT_FLOOR - totals['ran']} fewer tests ran than "
              f"the floor -- tests stopped being collected")
    if broken:
        print("  non-passing:")
        for item in broken:
            print("     ", item)

    ok = not undefined and not broken and not shortfall and not (
        totals["fail"] or totals["err"])
    print("\n  GATE:", "PASS" if ok else "*** FAIL ***")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
