# This is a personal fork

This branch (`feature/media-fetch-tab`) adds a reference-voice pipeline and an
engine swap on top of SECourses' app. **It is not the original**, it is not
supported by the original author, and the upstream README below still describes
the app as it ships from Patreon.

## What this fork adds

- **Download & Extract Audio tab** -- pull a clip from YouTube or anything
  yt-dlp supports, transcode it with ffmpeg, and send it straight to the
  reference-voice box. Quality presets seed the manual controls.
- **Audio clean-up** -- isolate vocals, de-reverb, denoise, keep a single
  speaker, trim silence, loudness-normalise. Order is fixed in code because the
  chain is not commutative.
- **IndexTTS-2.5 as the engine**, with a **persistent worker**: the model loads
  once per session instead of once per click, so a second generation takes
  around 4 seconds rather than 35. It gives the GPU back on an idle timer
  (default 30 minutes, selectable), on an unload button, and at app exit.
- **Speaking speed** slider and **tone/delivery presets** that each carry their
  own pace, plus a language selector.
- **Voice shaping** with formant-preserving pitch shift, so a few semitones does
  not sound like a chipmunk.
- **A GPU/CPU picker** that labels each device by its real name -- CUDA's device
  order is not nvidia-smi's, so picking by number gets you the wrong card.
- **LAN access and mobile fixes** -- `--host`/`--port` were parsed and never
  passed to `launch()`, and the reference-voice box was microphone-only, which
  no phone can use over plain http.

## Three things that will bite you on a fresh machine

1. **`requirements.txt` is not in this repo.** It lives in the installer bundle
   one directory up, so the **gradio 6.17.3 pin does not travel with a clone**.
   That pin sits between two real failures: 6.11 throws Svelte into an infinite
   effect loop and hard-locks the browser tab, and 6.26 pulls in
   huggingface-hub 1.x which breaks transformers at model load *while the UI
   still looks perfectly healthy*. Get that file separately.
2. **The engine is a separate checkout.** `engine_paths.py` defaults to
   `D:\Index_TTS_v4\index-tts-2.5`; point `INDEXTTS25_ROOT` at yours. It needs
   its own Python 3.11 venv with numpy 2.x, because this app runs on Python 3.10
   with numpy 1.26 and the two cannot share a process. Install it with
   `uv sync` **without** `--all-extras` -- the deepspeed extra will not build on
   Windows.
3. **Audio clean-up needs its own sidecar venv**, built by
   `install_audio_cleanup.bat`, for the same reason. `checkpoints/` is
   gitignored, so models are downloaded rather than cloned.

---

# IndexTTS2 SECourses Premium Voice Cloning and Generation App - 1-Click to Install on Windows, RunPod and Massed Compute - Generate Entire Audiobooks With Consistent High Quality Voice

## This app is made only for SECourses Patreon users : https://www.patreon.com/posts/139297407

### Download app from here > https://www.patreon.com/posts/139297407

## 1-Click Installers 

<img height="600" alt="image" src="https://github.com/user-attachments/assets/9d7b363c-aecf-4c7c-bd96-b336a599d4d8" />

## 11 May 2026 Update V4.1

- New feature provide image and generate video automatically implemented

- Just run installer bat file, zip file is still same, to update

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/1ccb7828-5fce-44e6-ae8e-b8e472c000e6" />

<img  height="600" alt="image" src="https://github.com/user-attachments/assets/477f5510-8da1-418b-a1ac-a82c57be9bc7" />



## 5 April 2026 Update V4

- This is a massive update of the app so please make a fresh install

- Installers upgraded to uv thus now up to 100x faster

- - Tested on Windows, RunPod and Massed Compute

- - Linux users please use Massed Compute installers

- Now all models auto downloaded with our app

- How app has title on browser tabs and a custom favicon

- <img width="371" height="98" alt="image" src="https://github.com/user-attachments/assets/b3f77604-c5cf-402c-8287-c123a997ecce" />

- Interface completely upgraded

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/7a3c9fa1-fcbd-4d14-9a8a-fdadaf9ae5ab" />

- Now supports caption / subtitle srt files upload and processing it automatically

- You can also enable Use Caption Cue Timing to generate exactly same timing as caption

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/8f04f372-60e6-4595-a422-4e55371226f1" />

- This app works with reference audio file and now you can provide a video or an audio file or record audio from your microphone (recommended 15 seconds)

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/c701ba3c-aa0f-4cd8-92fc-da260a20da76" />

- Now supports sub-process processing thus absolutely 0 VRAM or RAM usage after processing is done

- Now supports real batch-size processing thus the speed gain is immerse if fits into your VRAM (this is fully my custom implementation no one else has this feature)

- <img height="600" alt="image" src="https://github.com/user-attachments/assets/c43757da-c6f2-495f-b48d-ff98e6ba1be3" />

- Now supports full preset save / load and remember system

- Now supports cancel generation

- <img width="898" height="1480" alt="image" src="https://github.com/user-attachments/assets/5c5b7e61-d567-4fa3-b366-271b0d575b51" />

- Now when generating more information on the CMD will be shown like status, speed, ETA, how much left, etc.

- <img width="2098" height="435" alt="image" src="https://github.com/user-attachments/assets/c43c8683-804e-424b-8eff-1381e5f255a0" />

- Other features

- <img width="3531" height="1542" alt="image" src="https://github.com/user-attachments/assets/98341bc6-0c45-48b5-ad17-9ef43cb14880" />

- Advanced parameters

- <img width="3555" height="1633" alt="image" src="https://github.com/user-attachments/assets/4b909f90-6dd1-4ef4-8a00-7225f1a28390" />

- A tutorial video coming soon hopefully

- <img width="3568" height="1553" alt="image" src="https://github.com/user-attachments/assets/84ec9dbc-1aff-4b16-9d1e-fe0ce7bd1ca0" />







