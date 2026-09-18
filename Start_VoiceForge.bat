@echo off
REM AIO VoiceForge launcher - runs the app from this repo (the one location).
REM Pins the TTS engine to the RTX 5060 Ti so the other two cards stay free.
setlocal
cd /d "%~dp0"

set VENV=%~dp0..\venv_voiceforge_host

REM The venv was relocated from the old D: install, so its activate.bat still
REM names the old path - call the interpreter directly instead of activating.
set PATH=%VENV%\Scripts;%PATH%
set PYTHONWARNINGS=ignore

REM engine_worker points the engine child at the V5 hf_cache itself; this is
REM only a fallback so nothing in the host process reaches for a stale cache.
set HF_HOME=G:\Index_TTS_v5\Premium_IndexTTS2_SECourses\models\hf_cache

REM torch's device order is not nvidia-smi's - nvidia-smi calls the 5060 Ti
REM card 1, torch calls it cuda:2 - so the index is resolved by card name at
REM launch instead of hardcoded, and survives a card being added or moved.
for /f "delims=" %%A in ('""%VENV%\Scripts\python.exe" tools\pick_tts_device.py 5060"') do set "TTS_DEVICE=%%A"
if not defined TTS_DEVICE (
    set "TTS_DEVICE=cuda:2"
    echo Could not resolve the RTX 5060 Ti by name - falling back to cuda:2.
)

REM Seeds the device picker rather than hiding cards with CUDA_VISIBLE_DEVICES:
REM hiding them would make the in-app dropdown disagree with the engine about
REM what any index means. See _seed_selected_device_from_env in webui_runtime.
set "VOICEFORGE_TTS_DEVICE=%TTS_DEVICE%"
echo TTS engine will load on %VOICEFORGE_TTS_DEVICE% (RTX 5060 Ti).

REM Stop_VoiceForge drops this flag so the exit below can tell a deliberate
REM stop from a crash. Clearing it first means a stale flag from an earlier
REM run cannot swallow this run's traceback.
set "STOPFLAG=%TEMP%\voiceforge_stop.flag"
if exist "%STOPFLAG%" del "%STOPFLAG%" >nul 2>&1

"%VENV%\Scripts\python.exe" webui.py
set EXITCODE=%ERRORLEVEL%

REM taskkill /F makes python exit nonzero, so without this a normal Stop
REM would look like a crash and hold the window open for a keypress.
if exist "%STOPFLAG%" (
    del "%STOPFLAG%" >nul 2>&1
    set EXITCODE=0
)

REM A clean shutdown leaves nothing to read, so the window closes itself.
REM A crash keeps it open instead of flashing the traceback past.
if not "%EXITCODE%"=="0" pause
endlocal
