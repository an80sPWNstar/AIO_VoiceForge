@echo off
REM AIO VoiceForge launcher — runs the app from this repo (the one location).
REM Interpreters stay where they are installed: the webui host venv on D:,
REM the TTS engine venv on G: (spawned per-request by engine_worker), and the
REM cleanup sidecar venv on D:. Nothing is synced anywhere anymore.

cd /d "%~dp0"

call D:\Index_TTS_v4\Premium_IndexTTS2_SECourses\venv\Scripts\activate.bat || exit /b

set PYTHONWARNINGS=ignore

REM engine_worker points the engine child at the V5 hf_cache itself; this is
REM only a fallback so nothing in the host process reaches for a stale cache.
set HF_HOME=G:\Index_TTS_v5\Premium_IndexTTS2_SECourses\models\hf_cache

python webui.py

pause
