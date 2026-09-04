# Overnight report — hearing the model, 2026-09-03

## What was built

`tools/hear.py` (committed, `04874cb` + `8186f4e`): whisper transcribes each
generated wav back to text for a word-error rate against the input text, and
librosa measures pace, loudness, internal silence gaps, clipping and pitch.
Validated both directions before first use: known-good wav → WER 0.0,
deliberately wrong wav → 0.92. One grader bug found and fixed mid-run:
whisper writes "March 3rd"/"74" for correctly spoken words, which scored
0.23+ on flawless audio until digits were normalized to words.

**Honesty box:** WER ≈ 0 proves the words are intelligible, not that the
delivery sounds good. Everything below is machine-measured; the wavs are in
`outputs/` (folders 0002–0037) for your ears.

## Sweeps run (65 generations, all logged)

| Sweep | What | Result |
|---|---|---|
| 1 (30) | 3 sentences × default + 8 emotion channels @0.8 + mixed, voice_01 | No channel wrecks intelligibility. All WER 0 after grader fix except 4 suspect rows |
| 2 (29) | 4× repeats of each suspect; voice_05/09 generalization; speed walk 0.5–2.0 | **All suspects closed as sampling noise** (0/8 reproductions). Voices generalize: WER 0.0 everywhere. Speech stays intelligible across the whole speed range |
| 3 (6) | ~90-word paragraph through the segmentation path, 3 voices × default/sad | WER ≤ 0.026, no stitching stalls (max internal gap 0.88s, and that's sad's expressive pausing) |

## Findings

**No defects.** Zero errors in 65 runs, nothing garbled, no truncation, no
segment-join stalls, no clipping anywhere.

**Observations for your ears (taste, not defects):**
- Depression and sad are the most differentiated channels (pitch ~285→150 Hz,
  slower, long pauses). Anger and disgust also move as expected.
- **Joy and surprise barely move the acoustics vs default** — if any biases
  deserve a listen and a retune, it's these two. (tesla's read of the sweep-1
  table, spot-checked against the numbers.)
- **Calm comes out slightly *faster* than default** (3.35 vs 3.18 words/s) with
  narrower pitch spread. The spread is calm-like; the speed isn't.
- The "Speaking speed" slider is a duration factor (right = slower). The info
  text does say so, and the tone-preset table is self-consistent — but a
  slider named "speed" where right means slower may be worth a rename someday.

## Pivot to full-model training (in progress)

Per your instruction, testing stopped after sweep 3. Trainer choice (your
call, from the remote question): **RVC via Applio** — stable/mature, MIT,
CLI-driven, matches the `rvc` schema already in character_store
(`model_path`/`index_path`/`transpose`).

Plan: dataset export from character clips → Applio install on D: → validation
training run on neutral audio → wire the trained model into the character
store and a post-generation conversion stage. Note: current characters hold
one 12–15s clip each; RVC wants ~10 minutes, so the media-fetch →
segmentation → clips pipeline is how a real character gets to trainable.

---

## Addendum (written at close of session, 2026-09-04)

The pivot section above happened: RVC training landed end to end
(`0dff66c`..`511d2f3`), was validated on a public-domain LibriVox narrator
(100 epochs, WER 0.0 through conversion), and the trained model is attached
to the `librivox-test-narrator` character. The UI gained the forge look
(`abef6c6`).

Upstream V5 (SECourses) was then assessed: it adds native LoRA/DoRA
fine-tuning of IndexTTS-2.5 itself (dataset prep with whisper transcripts,
batch-1 LR 4e-5 defaults, checkpoint grid). Installed and verified at
`G:\Index_TTS_v5\Premium_IndexTTS2_SECourses` (torch 2.13+cu130, 7.8 GB
models, app serves clean). Agreed next steps: point engine_paths at the V5
install and reconcile the runner contract, then add a Train LoRA lane
behind the character store beside the RVC lane.
