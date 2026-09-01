# AIO_VoiceForge

Clone a voice from almost any clip, then make it say whatever you type.

Point it at a YouTube link (or hand it an audio file), let it strip out the
music, echo and background chatter until one clean voice is left, and use that
voice to read your text aloud. Everything happens in one window -- no juggling
separate tools for downloading, cleaning and generating.

## What you can do with it

1. **Grab a voice.** Paste a link in the *Download & Extract Audio* tab. It
   downloads the audio and converts it for you.
2. **Clean it up.** Run the clean-up chain on that clip: separate the singing or
   music away from the speech, remove room echo, remove hiss, and keep only the
   main speaker when other people talk over them. Tick the stages you want or
   use a preset.
3. **Send it over.** One button hands the cleaned clip to the generator as your
   reference voice.
4. **Type and generate.** Enter your text, pick how it should be delivered, and
   press *Generate Speech*.

You can also skip straight to step 4 with any audio file you already have.

## Setting it up

**This repository is only part of what you need.** It holds the app's code, not
the things that make it run:

- the **installer bundle** that sits one folder above it, which carries the
  exact library versions the app needs -- getting these wrong is not obvious,
  since the app will start and look perfectly healthy while being broken
- the **speech engine and its voice models**, which live in their own folder
  and are downloaded rather than stored here
- the **clean-up tools**, installed once by running `install_audio_cleanup.bat`

Get the bundle from whoever sent you this, and keep the folder layout they used.
If the engine ends up somewhere different, set an environment variable called
`INDEXTTS25_ROOT` pointing at it.

It needs an NVIDIA graphics card. It will run without one, but slowly enough
that it is not worth doing.

## Running it

Double-click **`Windows_Start_App.bat`**, wait for the black window to finish
loading, then open **http://localhost:7860** in your browser.

It also accepts connections from other devices on your network. From a phone
or laptop, use **http://[the PC's address]:7860** -- to find that address, open
Command Prompt on the PC, type `ipconfig`, and look for the IPv4 line, which
usually starts with 192.168. The black window will not print this for you; it
only ever shows `0.0.0.0`, which is not an address you can type anywhere.

Close the black window to shut the app down.

## Getting the best results

**Give it a good reference clip.** Thirty seconds of clean, single-speaker
speech beats five minutes of a noisy interview. The clean-up tab exists to get
you there, so use it rather than feeding in a raw download.

**Never type stage directions into your text.** If you write *"in a low, deep
voice"* in the text box, it will read those words out loud. Delivery direction
goes in the **Emotion Description Text** box instead -- and that box is only
read when the emotion mode is set to the text-description option, so switch the
selector too.

**Describe delivery in several dimensions at once.** *"Speak slowly in a low,
deep voice with heavy chest resonance"* steers it far harder than *"sad"* does.
Single-word moods barely move it.

**To stress a word, TYPE IT IN CAPITALS.** No setting fixes emphasis or
question-mark intonation; capitals do.

**Use the tone presets as starting points.** Each one sets a delivery style and
a speaking pace together. You can adjust the *Speaking speed* slider afterwards
-- drag left to speak faster, right to slow down. Slower usually sounds more
deliberate and natural.

## Things that look broken but are not

**The first generation takes about half a minute; the rest take a few seconds.**
The voice model has to load once. After that it stays ready, so everything
following is fast.

**It goes slow again if you leave it alone for a while.** The model unloads
after 30 minutes of sitting idle so it stops occupying your graphics card. The
next generation reloads it. If you would rather it stayed ready, change
*Unload after* in the Compute Device panel -- there is a *Never unload* option.
There is also an *Unload engine* button if you want your graphics card back
immediately for a game or another program.

**The page spins forever and nothing happens.** The app was restarted while
that tab was open. Reload the page with Ctrl+F5.

**It is using the wrong graphics card.** The *Run the model on* dropdown lists
each card by name -- pick it by the name, not the number. The numbers do not
match the order you would expect, so "Auto" may not choose the card you assume.
Switching cards makes the next generation slow once while it moves over.

**Nothing appears when you press record on a phone.** Phones cannot use a
microphone over a plain network address. Upload a file instead.

## Credit

This is a personal, private build. The underlying app is the IndexTTS2 SECourses
Premium app by Furkan Gozukara, and the speech engine is IndexTTS by the
index-tts project. Their original README follows below, unchanged.

What this build adds on top: the download and extract tab, the audio clean-up
chain, the newer IndexTTS-2.5 speech engine that keeps itself loaded between
generations, tone presets with per-preset pacing, a speaking-speed control,
formant-preserving pitch shifting, a graphics-card picker, and network and
phone access.

---

# Original app README (SECourses)

> Kept below for credit. Note that its download and installation instructions
> refer to the original app, not to this build.

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







