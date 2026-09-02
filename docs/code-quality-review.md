# Code quality review — AIO_VoiceForge

- **Date:** 2026-09-01
- **Commit:** `d5db0db`, branch `refactor/split-webui`
- **Standard:** `~/.claude/standards/code-quality.md` (cited by section, not restated)
- **Scope:** all 33 first-party source files. Excluded as vendored or historical:
  `indextts/` (upstream engine), `archive/`, `tools/i18n/`.
- **Note on timing:** run immediately after `webui.py` was split from 4265 to 1797
  lines across nine commits. Findings below are against the post-split layout;
  where something predates the split it says so.

---

## 1. File inventory

Reasons-to-change counted per §1, which treats that count as the diagnostic and
length as the symptom.

| File | Lines | Reasons to change | Verdict |
|---|---|---|---|
| `webui.py` | 1797 | 4 — tab layout · event wiring · preset field table · launch | **split** (§1, ≥800 automatic) |
| `webui_generation.py` | 809 | 4 — request shape · subprocess protocol · cancel · subtitle finalisation | **split** (§1, ≥800 automatic) |
| `webui_handlers.py` | 805 | 4 — device/engine · media fetch · cleanup · generation tab | **split** (§1, ≥800 automatic) |
| `webui_media_fetch.py` | 793 | 3 — yt-dlp interface · ffmpeg command building · quality presets | review |
| `audio_cleanup_worker.py` | 745 | 3 — stage implementations · model loading · CLI entry | review |
| `webui_generation_runner.py` | 657 | 3 — inference call · subtitle assembly · metadata writing | review |
| `subtitle_utils.py` | 584 | 2 — parsing formats · audio retiming | acceptable |
| `engine_worker.py` | 414 | 2 — process lifecycle · idle policy | acceptable |
| `webui_audio_cleanup.py` | 391 | 2 | fine |
| `webui_assets.py` | 333 | 1 — page chrome only | fine (§1 "data is not code", done) |
| `audio_cleanup_shared.py` | 311 | 1 — the stage table | fine |
| `webui_media_utils.py` | 293 | 2 | fine |
| `tools/run_pipeline_regression.py` | 291 | 2 | fine |
| `webui_tone_presets.py` | 250 | 1 | fine |
| `webui_subprocess_worker.py` | 217 | 2 | fine |
| `webui_voice_shaping.py` | 174 | 1 | fine |
| `webui_preview.py` | 155 | 1 | fine |
| `webui_progress.py` | 130 | 2 — console reporting · browser reporting | fine |
| `webui_preset_store.py` | 126 | 1 | fine |
| `webui_preset_normalize.py` | 115 | 1 | fine |
| `task_output_utils.py` | 106 | 1 | fine |
| `webui_runtime.py` | 95 | 2 — argv · device selection | fine |
| `engine_protocol.py` | 64 | 1 | fine |

Three files remain over the automatic-split threshold. Every other module is
under 400 and single-purpose, which is a large improvement on the single
4265-line file this started as.

---

## 2. SOLID

### S — single responsibility

Three findings, all the ≥800 files above. In each the responsibilities are
cleanly *separable* rather than tangled, which is why the staged plan (§10) can
split them without behaviour risk:

- `webui.py:264-1782` is one `with gr.Blocks(...)` block containing three tabs.
  Each tab is independently extractable; the only coupling is two cross-tab
  bridges (`webui.py:1496`, `webui.py:1505`, `webui.py:1520`) that hand a produced clip to the
  Audio Generation tab's reference slot.
- `webui_generation.py` mixes request construction (`:301-581`) with subprocess
  orchestration (`:582-698`) and two subtitle helpers (`:62-125`, `:160-207`)
  that do not belong to either — see §7, they are duplicates.
- `webui_handlers.py` holds four unrelated handler families in one file. It was
  created that way deliberately during the split as "everything the widgets
  call"; that was the right first cut but it is now over threshold.

### O — open/closed

**No finding — this is a strength.** The two extensions the project is likeliest
to need next are both declarative-table driven:

- A new clean-up stage: add to `audio_cleanup_shared.py:31` (`STAGE_ORDER`) and
  `:40` (`STAGE_LABELS`), implement in `audio_cleanup_worker.py`. The UI picks it
  up with no edit — `webui_handlers.py:418` derives the checkbox list from the
  table.
- A new tone preset: one file, `webui_tone_presets.py`. `webui.py:540` reads
  `TONE_PRESET_NAMES`.

That is exactly what §2's O question asks for.

### L — near-miss interfaces

One, minor. `webui_handlers.py:282` (`media_fetch_run_ui`) and `:460`
(`cleanup_run_ui`) are the same shape — start a worker thread, drain a queue,
yield five gradio updates — implemented twice, 97 and 98 lines. They are not
literal duplicates (different payloads) but a shared
`run_with_progress(work, log_lines, join_seconds)` would remove ~120 lines and
give one place to fix a threading bug rather than two.

### I — interface segregation

No finding. `engine_protocol.py` (64 lines) is a genuinely minimal boundary
between the two processes, and both sides use all of it.

### D — dependency inversion

One real finding. `webui_generation.py:301` (`_prepare_generation_request`, 281
lines) raises `gr.Error` at `:360`, so the entire request-building computation
requires gradio to be importable. Per §2's D question, its logic cannot be
exercised without the framework. The fix is to raise a plain
`GenerationRequestError` and let the one caller translate it at the UI edge.

This is the single change that would most improve testability: that function is
the largest pure-ish computation in the codebase and currently has no direct
test.

---

## 3. Purity audit

Classified per §3's three-way split.

### Already pure — keep

`webui_preset_normalize.py` (all 8 functions), `webui_progress.py:24`
(`format_elapsed_duration`), `:55` (`audio_duration_ms`),
`webui_generation.py:210` (`normalize_emo_vector`), `webui_preview.py:77`
(`estimate_text_processing_sections`), `subtitle_utils.py` parsers,
`task_output_utils.py:13,26`. This is a healthy pure core and it is now
importable without gradio, which it was not before the split.

### Pure computation with hidden inputs — fix

| Site | Hidden input | Corrected signature |
|---|---|---|
| `engine_worker.py:382,384` (`status`) | `time.time()` read inline | `status(self, now=None)` defaulting to `time.time()`; §3.2 |
| `webui_generation.py:294` (`_mark_metadata_canceled`) | `datetime.now()` inline | take `now` as a defaulted parameter |
| `webui_preset_store.py:85` (`_save_ui_preset`) | `datetime.now()` inline | same |

`engine_worker.status()` is the one that matters. `uptime_seconds` and
`idle_seconds` are computed from a clock read inside the function, so the idle
policy — the thing that decides whether to release your graphics card — cannot
be tested at its boundary. `tests/test_engine_worker.py` covers everything about
the idle limit *except* elapsed-time behaviour, precisely because of this.

### Legitimately impure — keep, already isolated

`webui_progress.py:20` (`current_timestamp`), `webui_media_utils.py` (all —
filesystem and ffmpeg by definition), `engine_worker.py` process management,
`webui_preset_store.py` file I/O. All are small, named, and single-effect per
§3.4.

---

## 4. Magic values

Priority-ordered per §4. The codebase is **strong** here — `engine_worker.py:39-46`,
`webui_progress.py:83` and `webui_handlers.py:63-71` are all correctly named and
grouped, with the explanatory comment on the constant rather than the use site.
Only one real finding.

| Priority | Site | Value | Fix |
|---|---|---|---|
| §4.2 product decision | `webui_generation.py:212` | `[0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]` | Name it `DEFAULT_EMOTION_BIASES`, and record that its order is positionally bound to the eight `emo_bias_*` controls (`webui.py` Advanced Parameters tab). Eight tuned weights with no name and no comment; reordering the sliders would silently mis-apply every one. |
| §4.3 grouping | `webui_handlers.py:106` | `1024 ** 3` | Minor — `BYTES_PER_GB`. Listed for completeness, not worth a commit on its own. |

---

## 5. Swallowed errors

Ten silent handlers. Triaged per §5.

### Hiding a defect — fix

| Site | Catch | Why it matters |
|---|---|---|
| `webui_media_utils.py:267` | **bare `except:`** | §5.1/§5.2. A bare except catches `KeyboardInterrupt` and `SystemExit`, so Ctrl-C during a long time-range extraction is swallowed into a temp-directory cleanup. Narrow to `OSError`. |
| `webui_preset_store.py:56` | `except Exception: pass` | §5.5 exactly: this is the write of the last-used preset pointer. If it fails, the user's preset silently stops being remembered across restarts with no indication. Log at warn. |

### Should log — lower priority

| Site | Catch | Note |
|---|---|---|
| `webui_subprocess_worker.py:105` | `except Exception` around `torch.cuda.empty_cache()` | Broad. A CUDA failure here is worth a line, since it means VRAM is not being released. |
| `webui_generation.py:260` | `except Exception` in `_cleanup_temp_file` | Narrow to `OSError`; a non-OSError here is a programming fault per §5.4. |
| `audio_cleanup_worker.py:687` | `except OSError` | Already narrow. Acceptable, a log line would still help. |

### Silent is fine — keep

`engine_worker.py:65,69,198,345,366`. All five are narrowly typed
(`ProcessLookupError`/`PermissionError`/`OSError`,
`subprocess.TimeoutExpired`, `ValueError`/`OSError`,
`BrokenPipeError`/`OSError`/`ValueError`) and sit on process-teardown paths where
the failure mode genuinely is expected — a process that already exited cannot be
signalled again. `:366` carries the comment §5.3 asks for: *"Never let closing the
app orphan a process holding a GPU."* This is the right pattern; the four above
should look like it.

---

## 6. Repo hygiene

- `.gitignore` replaced from `~/.claude/templates/gitignore/python.gitignore`
  with a merged `# Project-specific` section. `git ls-files | git check-ignore
  --stdin -v` prints nothing — clean swap. All 30 entries from the previous file
  are covered, either literally or by an equivalent template pattern.
- A test runner exists and the suite is meaningful: **56 tests, all passing**,
  up from 6 actually executing at the start of this session (two test modules
  had been importing from `indextts.utils.*`, a path that has not existed since
  those modules moved to the repo root).
- **No linter is configured.** §7 asks for one from the first commit. `ruff`
  would be the obvious choice and would have caught the bare `except:` above.

---

## 7. Duplication

**One finding, three functions, ~130 lines — and no comment anywhere marks it.**

| Function | Copy A (app process) | Copy B (engine process) | Bodies |
|---|---|---|---|
| `current_timestamp` | `webui_progress.py:20` | `webui_generation_runner.py:170` | identical |
| `finalize_subtitle_segment_audio` | `webui_generation.py:62` | `webui_generation_runner.py:219` | identical |
| `build_subtitle_status_message` | `webui_generation.py:160` | `webui_generation_runner.py:295` | identical |

Verified by AST-comparing the unparsed bodies: they differ **only** in type
annotations. Nothing has drifted yet, so this is a maintenance hazard rather
than a live bug — but §6 is explicit that a value existing in two files is a
finding regardless.

It matters more than usual here because the two copies run in **different
interpreters** — `webui_generation_runner.py` is executed by the engine's Python
(spawned at `engine_worker.py:112`) — and both write timestamps into the *same*
`metadata.json`. `webui_generation.py:294` parses `started_at` written by one
process while `updated_at` is written by the other. If the format in one copy is
ever changed, elapsed-time reporting breaks in a way that looks like a data bug.

§6 notes that two runtimes are not an excuse, and here it is not even hard: both
processes already import `subtitle_utils` and `task_output_utils` from this same
directory. A shared module is precedented and needs no build step.

---

## 8. Live bugs found while reading

### Fixed during this session (listed for the record)

1. `tests/test_subtitle_utils.py:8`, `tests/test_task_output_utils.py:5` —
   imported from `indextts.utils.*`, which has not existed since those modules
   moved to the root. Both files died at collection; 19 of 25 tests never ran.
2. `tests/test_webui_presets.py` — asserted on `webui.tts`, deleted in `bae9ea4`.
   Failing, not guarding.
3. `tools/run_pipeline_regression.py` — three faults: called `gen_single` with 53
   of 55 required arguments; unpacked a generator into two values; and ran a
   known-dead case first, so the run aborted before reaching any working case.

### Open

4. **The regression harness cannot run from this checkout.**
   `tools/run_pipeline_regression.py:63` uses `prompt="demo_voice_for_test.mp3"`
   and `:273-274` use `test2.srt`. Both are in `.gitignore`, untracked, and absent
   from the repo — they exist only in the deployed folder. Reproduction: from a
   clean clone, run the harness; it fails on a missing reference clip regardless
   of GPU. Either track the fixtures or have the harness fail with a message
   naming what is missing and where to get it.

---

## 9. Recommended module layout

```
webui.py                      launch + wiring only, target <150   (§1 entry point wires)
ui/
  tab_generation.py           Audio Generation layout
  tab_advanced.py             Advanced Parameters layout
  tab_media.py                Download & Extract layout
  preset_fields.py            field metadata, no components
handlers/
  device.py                   device + engine controls
  media_fetch.py              download tab handlers
  cleanup.py                  clean-up tab handlers
  reference.py                reference-voice + subtitle handlers
  progress_stream.py          the shared worker-thread/queue helper (§2 L)
generation/
  request.py                  _prepare_generation_request, framework-free
  subprocess_run.py           orchestration + cancel
shared/                       imported by BOTH interpreters
  timestamps.py               current_timestamp
  subtitle_render.py          finalize_subtitle_segment_audio,
                              build_subtitle_status_message
webui_assets.py               unchanged, already right
webui_preview.py              unchanged
webui_preset_store.py         unchanged
webui_preset_normalize.py     unchanged
webui_runtime.py              unchanged
engine_worker.py              unchanged
```

---

## 10. Staged plan

Each stage is independently shippable and independently revertible. The 56-test
suite plus an `import webui` smoke check gates every one — importing `webui`
executes the whole `gr.Blocks` body, so it proves every `fn=` reference resolves.

| # | Stage | Risk | Payoff |
|---|---|---|---|
| 1 | `shared/` module for the three duplicated functions (§7). Both interpreters already import from this directory. | **low** | Removes ~130 duplicated lines; closes the only cross-process format hazard |
| 2 | Fix the two error findings (§5): narrow the bare `except:`, log the preset-pointer write failure | **low** | One silent user-visible data loss becomes visible |
| 3 | Name `DEFAULT_EMOTION_BIASES` (§4) and record its positional binding | **low** | Removes a silent reordering hazard |
| 4 | Add `ruff` with a minimal config (§7) | **low** | Would have caught stage 2's bare except |
| 5 | Inject `now` into `engine_worker.status()` (§3.2) | **low** | Makes the idle/unload policy testable at its boundary — the last untested part of the engine swap |
| 6 | `_prepare_generation_request` raises a plain error, not `gr.Error` (§2 D) | medium | Makes the largest computation in the codebase testable without gradio |
| 7 | Extract `progress_stream.py` from the two worker-thread handlers (§2 L) | medium | −120 lines; one threading implementation instead of two |
| 8 | Split `webui_handlers.py` into four by family | medium | Three files under threshold |
| 9 | Split the `gr.Blocks` body into per-tab modules | **high** | `webui.py` under 150 lines. Needs the two cross-tab bridges passed explicitly, and there is no automated UI coverage — do this one with a manual click-through |
| 10 | Separate `_CONFIG_FIELDS` metadata from component binding | **high** | Frees the last 10 preset functions; enables stage 9 cleanly |

Stages 1–5 are mechanical and could all land in one sitting. Stage 9 is the only
one that genuinely needs a human at the keyboard afterwards.

---

## 11. What not to change

- **`--host 0.0.0.0` (`webui_runtime.py:22`) is deliberate.** §6b flags
  all-interface binds, but the README documents phone and LAN access as a
  feature and explains how to find the machine's address. Worth knowing rather
  than fixing: the Gradio app has no authentication, so anyone on the network can
  drive it and read `outputs/`. Correct for a personal LAN build; it would be a
  finding the moment this ran anywhere else.
- **The five `engine_worker.py` silent catches** (§5) are correct and one of them
  carries the model comment for the whole repo.
- **`webui_generation.py:283`** — `_mark_metadata_canceled` declining to rewrite a
  task already marked `completed`. This is race protection: the worker can finish
  between the cancel click and the handler. Now covered by
  `tests/test_generation_cancel.py`.
- **`_SUBPROCESS_STATE` being mutated in place rather than rebound**
  (`webui_generation.py:236-243`). Two threads hold it by reference; rebinding
  would leave the cancel handler watching a dict nobody writes to. Covered by a
  test that asserts identity.
- **The comment at `webui_preview.py:35-39`** explaining why the tokenizer is
  usually absent (2.5's config still names `bpe.model` but the release tokenizes
  from a tiktoken vocabulary). That is a measured gotcha and exactly what §4 means
  by keeping the comment attached to the thing it explains.
- **The whole of `indextts/`** — vendored upstream engine, not first-party.

---

## Delegation

**No local model was used for this audit.** The skill's judgment passes are
designed to run on `rtx`/`tesla`; both were reserved for another session for its
entire duration, so every finding here was read and verified directly. The
mechanical passes (§1–§7 greps, AST comparisons, the duplication diff) were run
locally in the shell and are reproducible from the commands in the skill.

Where a finding came from a claim rather than a reading, it was checked against
source before being written down — the duplication finding in §7 in particular was
verified by AST-unparsing both copies and diffing, which is what established that
the bodies are identical rather than drifted.
