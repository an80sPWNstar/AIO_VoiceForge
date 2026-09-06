@echo off
REM AIO VoiceForge launcher — runs the app from this repo (the one location).
REM Interpreters live beside the repo in E:\vs_code_projects: the webui host
REM venv (venv_voiceforge_host), the cleanup sidecar (venv_audioclean), and
REM the cleanup models (audio_cleanup_models). The TTS engine venv lives in
REM the V5 install on G: and is spawned per-request by engine_worker.

cd /d "%~dp0"

set VENV=%~dp0..\venv_voiceforge_host

REM The venv was relocated from the old D: install, so its activate.bat still
REM names the old path — call the interpreter directly instead of activating.
set PATH=%VENV%\Scripts;%PATH%
set PYTHONWARNINGS=ignore

REM engine_worker points the engine child at the V5 hf_cache itself; this is
REM only a fallback so nothing in the host process reaches for a stale cache.
set HF_HOME=G:\Index_TTS_v5\Premium_IndexTTS2_SECourses\models\hf_cache

"%VENV%\Scripts\python.exe" webui.py

pause
